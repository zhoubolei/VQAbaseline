
require 'paths'
require 'cunn'
require 'nn'

paths.dofile('opensource_base.lua')
paths.dofile('LinearNB.lua')
local debugger = require 'fb.debugger'

function build_model(method)
    --function to build up baseline model
    local g_model = {}
    if method == 'BOW' then

        g_model = nn.Sequential()
        local module_tdata = nn.LinearNB(manager_vocab.nvocab_question, g_params.embed_word)
        g_model:add(module_tdata)
        g_model:add(nn.Linear(g_params.embed_word, manager_vocab.nvocab_answer))


    elseif method == 'IMG' then
        g_model = nn.Sequential()
        g_model:add(nn.Linear(g_params.vdim, manager_vocab.nvocab_answer))
    

    elseif method == 'BOWIMG' then
        g_model = nn.Sequential()
        local module_tdata = nn.Sequential():add(nn.SelectTable(1)):add(nn.LinearNB(manager_vocab.nvocab_question, g_params.embed_word))
        local module_vdata = nn.Sequential():add(nn.SelectTable(2))
        local cat = nn.ConcatTable():add(module_tdata):add(module_vdata)
        g_model:add(cat):add(nn.JoinTable(2))
        g_model:add(nn.LinearNB(g_params['embed_word'] + g_params['vdim'], manager_vocab.nvocab_answer))

    else
        print('no such methods')

    end

    g_model:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()
    criterion.sizeAverage = false
    g_model:cuda()
    criterion:cuda()

    return g_model, criterion
end

function initial_params()
    local gpuidx = getFreeGPU()
    print('use GPU IDX=' .. gpuidx)
    cutorch.setDevice(gpuidx)

    local cmd = torch.CmdLine()
    
    -- parameters for general setting
    cmd:option('--savepath', 'models/')

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
    cmd:option('--embed_word', 512,'the word embedding dimension in baseline')

    -- parameters for universal learning rate
    cmd:option('--maxgradnorm', 20)
    cmd:option('--maxweightnorm', 2000)

    -- parameters for different learning rates for different layers
    cmd:option('--lr_wordembed', 0.8)
    cmd:option('--lr_other', 0.01)
    cmd:option('--weightClip_wordembed', 1500)
    cmd:option('--weightClip_other', 20)


    local g_params_ = cmd:parse(arg or {})
    return g_params_
end

function runTrainVal()
    local method = 'BOWIMG'
    local testCombine = false
    g_params = initial_params()
    g_params['method'] = method
    g_params['save'] = g_params['savepath'] .. method ..'.t7'
    -- load data inside
    state_train, manager_vocab = load_visualqadataset('trainval2014_train', nil)
    state_val, _ = load_visualqadataset('trainval2014_val', manager_vocab)
    g_model, criterion = build_model(method)
    g_paramx, g_paramdx = g_model:getParameters()
    params_current, gparams_current = g_model:parameters()

    config_layers, grad_last = config_layer_params(params_current, 1)
    print(params_current)
    print('start training ...')
    local epoches_lr = g_params['nepoch_lr']
    stat = {}
    for i=1, g_params.epochs do
        print(method .. ' epoch '..i)
        train_epoch(state_train,'train')
        train_epoch(state_val, 'val')
        stat[i] = {acc, acc_match_mostfreq, acc_match_openend, acc_match_multiple}
        epoches_lr = epoches_lr -1
        if epoches_lr == 0 then
            for j =1, #config_layers.lr_rates do
                config_layers.lr_rates[j] = config_layers.lr_rates[j] / g_params['decay']
            end
            epoches_lr = g_params['nepoch_lr']
        end
    end

    --select the best train epoch number and combine train2014 and val2014
    
    if testCombine then
        nEpoch_best = 1
        acc_openend_best = 0
        for i=1, #stat do
            if stat[i][3]> acc_openend_best then
                nEpoch_best = i
                acc_openend_best = stat[i][3]
            end
        end
            
        print('best epoch number is ' .. nEpoch_best)
        print('best acc is ' .. acc_openend_best)

        --combine train2014 and val2014
        local nEpoch_trainAll = 100 or nEpoch_best
        state_train, manager_vocab = load_visualqadataset('trainval2014', nil)
        print('start training on all data ...')
        local epoches_lr = g_params['nepoch_lr']
        stat = {}
        for i=1, nEpoch_trainAll do
            print('epoch '..i)
            train_epoch(state_train,'train')
            stat[i] = {acc, acc_match_mostfreq, acc_match_openend, acc_match_multiple}
            

            epoches_lr = epoches_lr -1
            if epoches_lr == 0 then
                for j =1, #config_layers.lr_rates do
                    config_layers.lr_rates[j] = config_layers.lr_rates[j] / g_params['decay']
                end
                epoches_lr = g_params['nepoch_lr']
            end

            local modelname_curr = g_params['save'] .. '_bestepoch' .. nEpoch_best ..'_going.t7model'
            save_model(modelname_curr)
        end
        stat[nEpoch_best][1] = acc_openend_best
        local modelname_curr = g_params['save'] .. '_bestepoch' .. nEpoch_best ..'_final.t7model'
        save_model(modelname_curr)
    end
end


runTrainVal()
