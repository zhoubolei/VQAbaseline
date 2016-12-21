require 'paths'

local stringx = require 'pl.stringx'
local file = require 'pl.file'

paths.dofile('opensource_base.lua')
paths.dofile('opensource_utils.lua')

local debugger = require 'fb.debugger'

function adjust_learning_rate(epoch_num, opt, config_layers)
    -- Every opt.nepoch_lr iterations, the learning rate is reduced.
    if epoch_num % opt.nepoch_lr == 0 then
        for j = 1, #config_layers.lr_rates do
            config_layers.lr_rates[j] = config_layers.lr_rates[j] / opt.decay
        end
    end
end

function runTrainVal()
    local method = 'BOWIMG'
    local step_trainval = true --  step for train and valiaton
    local step_trainall = true --  step for combining train2014 and val2014
    local opt = initial_params()
    opt.method = method
    opt.save = paths.concat(opt.savepath, method ..'.t7')

    local stat = {}
    -- load data inside
    if step_trainval then 
        local state_train, manager_vocab = load_visualqadataset(opt, 'trainval2014_train', nil)
        local state_val, _ = load_visualqadataset(opt, 'trainval2014_val', manager_vocab)
        local model, criterion = build_model(opt, manager_vocab)
        local paramx, paramdx = model:getParameters()
        local params_current, gparams_current = model:parameters()

        local config_layers, grad_last = config_layer_params(opt, params_current, gparams_current, 1)
    -- Save variables into context so that train_epoch could use.
        local context = {
            model = model,
            criterion = criterion,
            paramx = paramx,
            paramdx = paramdx,
            params_current = params_current, 
            gparams_current = gparams_current,
            config_layers = config_layers,
            grad_last = grad_last
        }
        print(params_current)
        print('start training ...')
        for i = 1, opt.epochs do
            print(method .. ' epoch '..i)
            train_epoch(opt, state_train, manager_vocab, context, 'train')
            _, _, perfs = train_epoch(opt, state_val, manager_vocab, context, 'val')
            -- Accumulate statistics
            stat[i] = {acc, perfs.most_freq, perfs.openend_overall, perfs.multiple_overall}
            -- Adjust the learning rate 
            adjust_learning_rate(i, opt, config_layers)
        end
    end
    
    if step_trainall then
        local nEpoch_best = 1
        local acc_openend_best = 0
        if step_trainval then

            -- Select the best train epoch number and combine train2014 and val2014
            for i = 1, #stat do
                if stat[i][3]> acc_openend_best then
                    nEpoch_best = i
                    acc_openend_best = stat[i][3]
                end
            end
          
            print('best epoch number is ' .. nEpoch_best)
            print('best acc is ' .. acc_openend_best)
        else
            nEpoch_best = 100
        end
        -- Combine train2014 and val2014
        local nEpoch_trainAll = nEpoch_best
        local state_train, manager_vocab = load_visualqadataset(opt, 'trainval2014', nil)
        -- recreate the model  
        local model, criterion = build_model(opt, manager_vocab)
        local paramx, paramdx = model:getParameters()
        local params_current, gparams_current = model:parameters()

        local config_layers, grad_last = config_layer_params(opt, params_current, gparams_current, 1)
        
        local context = {
            model = model,
            criterion = criterion,
            paramx = paramx,
            paramdx = paramdx,
            params_current = params_current, 
            gparams_current = gparams_current,
            config_layers = config_layers,
            grad_last = grad_last
        }
        print(params_current)
        
        print('start training on all data ...')
        stat = {}
        for i=1, nEpoch_trainAll do
            print('epoch '..i .. '/' ..nEpoch_trainAll)
            _, _, perfs = train_epoch(opt, state_train, manager_vocab, context, 'train')
            stat[i] = {acc, perfs.most_freq, perfs.openend_overall, perfs.multiple_overall}
            adjust_learning_rate(i, opt, config_layers)

            local modelname_curr = opt.save 
            save_model(opt, manager_vocab, context, modelname_curr)
        end
    end
end

function runTest()
    --load the pre-trained model then evaluate on the test set then generate the csv file that could be submitted to the evaluation server
    local opt = initial_params()
    local context = loadPretrained(opt)
    local manager_vocab = context.manager_vocab

    -- load test data
    local testSet = 'test-dev2015' --'test2015' and 'test-dev2015'
    local state_test, _ = load_visualqadataset(opt, testSet, manager_vocab)

    -- predict 
    local pred, prob, perfs = train_epoch(opt, state_test, manager_vocab, context, 'test')
    
    -- output to csv file to be submitted to the VQA evaluation server
    local file_json_openend = 'result/vqa_OpenEnded_mscoco_' .. testSet .. '_'.. opt.method .. '_results.json'
    local file_json_multiple = 'result/vqa_MultipleChoice_mscoco_' .. testSet .. '_'.. opt.method .. '_results.json'
    print('output the OpenEnd prediction to JSON file...' .. file_json_openend) 
    local choice = 0   
    outputJSONanswer(state_test, manager_vocab, prob, file_json_openend, choice)
    print('output the MultipleChoice prediction to JSON file...' .. file_json_multiple) 
    choice = 1
    outputJSONanswer(state_test, manager_vocab, prob, file_json_multiple, choice)

    collectgarbage()

end

runTrainVal()
runTest()
