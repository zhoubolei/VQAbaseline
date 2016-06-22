require 'nn'
require 'cunn'

paths.dofile('LinearNB.lua')

function getFreeGPU()
    -- select the most available GPU to train
    local nDevice = cutorch.getDeviceCount()
    local memSet = torch.Tensor(nDevice)
    for i=1, nDevice do
        local tmp, _ = cutorch.getMemoryUsage(i)
        memSet[i] = tmp
    end
    local _, curDeviceID = torch.max(memSet,1)
    return curDeviceID[1]
end

function build_model(opt, manager_vocab) 
    -- function to build up baseline model
    local model
    if opt.method == 'BOW' then
        model = nn.Sequential()
        local module_tdata = nn.LinearNB(manager_vocab.nvocab_question, opt.embed_word)
        model:add(module_tdata)
        model:add(nn.Linear(opt.embed_word, manager_vocab.nvocab_answer))

    elseif opt.method == 'IMG' then
        model = nn.Sequential()
        model:add(nn.Linear(opt.vdim, manager_vocab.nvocab_answer))
    
    elseif opt.method == 'BOWIMG' then
        model = nn.Sequential()
        local module_tdata = nn.Sequential():add(nn.SelectTable(1)):add(nn.LinearNB(manager_vocab.nvocab_question, opt.embed_word))
        local module_vdata = nn.Sequential():add(nn.SelectTable(2))
        local cat = nn.ConcatTable():add(module_tdata):add(module_vdata)
        model:add(cat):add(nn.JoinTable(2))
        model:add(nn.LinearNB(opt.embed_word + opt.vdim, manager_vocab.nvocab_answer))

    else
        print('no such methods')

    end

    model:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()
    criterion.sizeAverage = false
    model:cuda()
    criterion:cuda()

    return model, criterion
end

function initial_params()
    local gpuidx = getFreeGPU()
    print('use GPU IDX=' .. gpuidx)
    cutorch.setDevice(gpuidx)

    local cmd = torch.CmdLine()
    
    -- parameters for general setting
    cmd:option('--savepath', 'model')

    -- parameters for the visual feature
    cmd:option('--vfeat', 'googlenetFC')
    cmd:option('--vdim', 1024)

    -- parameters for data pre-process
    cmd:option('--thresh_questionword',6, 'threshold for the word freq on question')
    cmd:option('--thresh_answerword', 3, 'threshold for the word freq on the answer')
    cmd:option('--batchsize', 100)
    cmd:option('--seq_length', 50)

    -- parameters for learning
    cmd:option('--uniformLR', 0, 'whether to use uniform learning rate for all the parameters')
    cmd:option('--epochs', 100)
    cmd:option('--nepoch_lr', 100)
    cmd:option('--decay', 1.2)
    cmd:option('--embed_word', 1024,'the word embedding dimension in baseline')

    -- parameters for universal learning rate
    cmd:option('--maxgradnorm', 20)
    cmd:option('--maxweightnorm', 2000)

    -- parameters for different learning rates for different layers
    cmd:option('--lr_wordembed', 0.8)
    cmd:option('--lr_other', 0.01)
    cmd:option('--weightClip_wordembed', 1500)
    cmd:option('--weightClip_other', 20)

    return cmd:parse(arg or {})
end

function loadPretrained(opt)
    --load the pre-trained model then evaluate on the test set then generate the csv file that could be submitted to the evaluation server
    local method = 'BOWIMG'
    local model_path = 'model/BOWIMG.t7'
    opt.method = method

    -- load pre-trained model 
    local f_model = torch.load(model_path)
    local manager_vocab = f_model.manager_vocab 
    -- Some simple fix for old models.
    if manager_vocab.vocab_map_question['END'] == nil then 
        manager_vocab.vocab_map_question['END'] = -1 
        manager_vocab.ivocab_map_question[-1] = 'END'
    end

    local model, criterion = build_model(opt, manager_vocab)
    local paramx, paramdx = model:getParameters()
    paramx:copy(f_model.paramx)

    return {
        model = model,
        criterion = criterion,
        manager_vocab = manager_vocab
    }
end
