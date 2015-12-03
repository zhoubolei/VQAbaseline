local debugger = require 'fb.debugger'

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


function config_layer_params(params_current, IDX_wordembed)
    local lr_wordembed = g_params['lr_wordembed']
    local lr_other = g_params['lr_other']
    local weightClip_wordembed = g_params['weightClip_wordembed']
    local weightClip_other = g_params['weightClip_other']

    local gradientClip_dummy = 0.1
    local weightRegConsts_dummy = 0.000005
    local initialRange_dummy = 0.1
    local moments_dummy = 0.9

    local config_layers = {}
    config_layers.lr_rates = {}
    config_layers.gradientClips = {}
    config_layers.weightClips = {}
    config_layers.moments = {}
    config_layers.weightRegConsts = {}
    config_layers.initialRange = {}

    local grad_last = {}
    if IDX_wordembed == 1 then
        -- assume wordembed matrix is the params_current[1]
        config_layers.lr_rates = {lr_wordembed}
        config_layers.gradientClips = {gradientClip_dummy}
        config_layers.weightClips = {weightClip_wordembed}
        config_layers.moments = {moments_dummy}
        config_layers.weightRegConsts = {weightRegConsts_dummy}
        config_layers.initialRange = {initialRange_dummy}
        for i = 2, #params_current do
            table.insert(config_layers.lr_rates, lr_other)
            table.insert(config_layers.moments, moments_dummy)
            table.insert(config_layers.gradientClips, gradientClip_dummy)
            table.insert(config_layers.weightClips, weightClip_other)
            table.insert(config_layers.weightRegConsts, weightRegConsts_dummy)
            table.insert(config_layers.initialRange, initialRange_dummy)    
        end

    else
        for i=1, #params_current do
            table.insert(config_layers.lr_rates, lr_other)
            table.insert(config_layers.moments, moments_dummy)
            table.insert(config_layers.gradientClips, gradientClip_dummy)
            table.insert(config_layers.weightClips, weightClip_other)
            table.insert(config_layers.weightRegConsts, weightRegConsts_dummy)
            table.insert(config_layers.initialRange, initialRange_dummy)
        end
    end

    for i=1, #gparams_current do
        grad_last[i] = gparams_current[i]:clone()
        grad_last[i]:fill(0)
    end
    return config_layers, grad_last
end

---------------------------------------
---- data IO relevant functions--------
---------------------------------------


function existfile(filename)
    local f=io.open(filename,"r")
    if f~=nil then io.close(f) return true else return false end
end

function load_filelist(fname)
    local data = file.read(fname)
    data = stringx.replace(data,'\n',' ')
    data = stringx.split(data)
    local imglist_ind = {}
    for i=1, #data do
        imglist_ind[i] = stringx.split(data[i],'.')[1]
    end
    return imglist_ind
end

function build_vocab(data, thresh, IDX_singleline, IDX_includeEnd)
    if IDX_singleline == 1 then
        data = stringx.split(data,'\n')
    else
        data = stringx.replace(data,'\n', ' ')
        data = stringx.split(data)
    end
    local countWord = {}
    for i=1, #data do
        if countWord[data[i]] == nil then
            countWord[data[i]] = 1
        else
            countWord[data[i]] = countWord[data[i]] + 1
        end
    end
    local vocab_map_ = {}
    local ivocab_map_ = {}
    local vocab_idx = 0
    if IDX_includeEnd==1 then
        vocab_idx = 1
        vocab_map_['NA'] = 1
        ivocab_map_[1] = 'NA'
    end

    for i=1, #data do
        if vocab_map_[data[i]]==nil then
            if countWord[data[i]]>=thresh then
                vocab_idx = vocab_idx+1
                vocab_map_[data[i]] = vocab_idx
                ivocab_map_[vocab_idx] = data[i]
                --print(vocab_idx..'-'.. data[i] ..'--'.. countWord[data[i]])
            else
                vocab_map_[data[i]] = vocab_map_['NA']
            end
        end 
    end
    vocab_map_['END'] = -1
    g_params.vocab_size = vocab_idx

    return vocab_map_, ivocab_map_, vocab_idx
end



function load_visualqadataset(dataType, manager_vocab)
--TODO: only need to load allanswer.txt, question.txt, choice.txt, question_type.txt, answer_type.txt
--TODO: two sets: train2014, val2014
    local path_imglist = 'datasets/coco_dataset/allimage2014'
    local path_dataset = '/data/vision/oliva/scenedataset/vqa_cache/'
    local featName = 'googlenetFCdense'
    local filename_question = path_dataset .. 'coco_' .. dataType .. '_question.txt'
    local filename_answer = path_dataset .. 'coco_' .. dataType .. '_answer.txt'
    local filename_imglist = path_dataset .. 'coco_' ..dataType ..'_imglist.txt'
    local filename_allanswer = path_dataset ..'coco_' ..dataType ..'_allanswer.txt'
    local filename_choice = path_dataset ..'coco_' .. dataType ..'_choice.txt'
    local filename_question_type = path_dataset ..'coco_' .. dataType ..'_question_type.txt'
    local filename_answer_type = path_dataset ..'coco_' .. dataType ..'_answer_type.txt'

    if existfile(filename_allanswer) then
        data_allanswer = file.read(filename_allanswer)
        data_allanswer = stringx.split(data_allanswer,'\n')
    end
    if existfile(filename_choice) then
        data_choice = file.read(filename_choice)
        data_choice = stringx.split(data_choice, '\n')
    end
    if existfile(filename_question_type) then
        data_question_type = file.read(filename_question_type)
        data_question_type = stringx.split(data_question_type,'\n')
    end
    if existfile(filename_answer_type) then
        data_answer_type = file.read(filename_answer_type)
        data_answer_type = stringx.split(data_answer_type, '\n')
    end
    local data_answer = file.read(filename_answer)
    local data_answer_split = stringx.split(data_answer,'\n')
    local data_question = file.read(filename_question)
    local data_question_split = stringx.split(data_question,'\n')
    local manager_vocab_ = {}

    if manager_vocab == nil then
        local vocab_map_answer, ivocab_map_answer, nvocab_answer = build_vocab(data_answer, g_params.thresh_answerword, 1, 0)
        local vocab_map_question, ivocab_map_question, nvocab_question = build_vocab(data_question,g_params.thresh_questionword, 0, 1)
        print(' no.vocab_question=' .. nvocab_question.. ', no.vocab_answer=' .. nvocab_answer)
        manager_vocab_ = {vocab_map_answer=vocab_map_answer, ivocab_map_answer=ivocab_map_answer, vocab_map_question=vocab_map_question, ivocab_map_question=ivocab_map_question, nvocab_answer=nvocab_answer, nvocab_question=nvocab_question}
    else
        manager_vocab_ = manager_vocab
    end
    
    print(string.format("loading textfile %s, size of data=%d", filename_answer, #data_answer_split))
    local curIDX = 0
    local imglist = load_filelist(filename_imglist)
    local nSample = #imglist
    if nSample >#data_question_split then
        nSample = #data_question_split
    end
    local x_question = torch.zeros(nSample, g_params.seq_length)
    local x_answer = torch.zeros(nSample):fill(-1)
    if g_params['multipleanswer'] == 1 then
        x_answer = torch.zeros(nSample, 10)
    end
    local nCount = 0
    local x_answer_num = torch.zeros(nSample)
    for i=1, nSample do
        local words = stringx.split(data_question_split[i])
        local answer = data_answer_split[i]
        if manager_vocab_.vocab_map_answer[answer]==nil then
            x_answer[i] = -1
        else
            x_answer[i] = manager_vocab_.vocab_map_answer[answer]
        end

        for j = 1, g_params.seq_length do
            curIDX = curIDX+1
            if j<=#words then
                if manager_vocab_.vocab_map_question[words[j]]==nil then
                    x_question[{i,j}] = 1 
                    nCount = nCount + 1
                else
                    x_question[{i,j}] = manager_vocab_.vocab_map_question[words[j]]
                end
            else
                x_question[{i,j}] = manager_vocab_.vocab_map_question['END']
            end
        end
    end

    
---------------------------
-- start loading features -
---------------------------
    local feature_prefix = {}
    local featureMap = {}

    if dataType == 'trainval2014' or dataType == 'trainval2014_train' then
        feature_prefixSet = {path_dataset .. 'coco_train2014_' .. featName, path_dataset .. 'coco_val2014_' .. featName}
        
        for setID=1, 2 do
            feature_prefix = feature_prefixSet[setID]
            local feature_imglist = torch.load(feature_prefix ..'_imglist.dat')
            local featureSet = torch.load(feature_prefix ..'_feat.dat')
            for i=1, #feature_imglist do
                local feat_in = torch.squeeze(featureSet[i])
                featureMap[feature_imglist[i]] = feat_in
            end
        end                 
        
    else if dataType == 'trainval2014_val' then
            feature_prefix = path_dataset .. 'coco_val2014_' ..featName
        end
        local feature_imglist = torch.load(feature_prefix ..'_imglist.dat')
        local featureSet = torch.load(feature_prefix ..'_feat.dat')
      
        for i=1, #feature_imglist do
            local feat_in = torch.squeeze(featureSet[i])
            featureMap[feature_imglist[i]] = feat_in
        end        
    end

    
    collectgarbage()
    local _state = {x_question = x_question, x_answer = x_answer, x_answer_num = x_answer_num, featureMap = featureMap, data_question = data_question_split,data_answer = data_answer_split, imglist = imglist, path_imglist = path_imglist, data_allanswer = data_allanswer, data_choice = data_choice, data_question_type = data_question_type, data_answer_type = data_answer_type, featureMap_question = featureMap_question}
    
    return _state, manager_vocab_
end


--------------------------------------------
-- training relevant code
--------------------------------------------
function save_model(path)
    print('saving model ' .. path)
    local d = {}
    d.params = g_params
    d.paramx = g_paramx:float()
    d.manager_vocab = manager_vocab
    d.stat = stat
    d.config_layers = config_layers
    d.g_params = g_params

    torch.save(path,d)
end

function bagofword(x_seq, nwords)
-- turn the list of word index into bag of word vector
    local outputVector = torch.zeros(nwords)
    for i=1, x_seq:size(1) do
        if x_seq[i]~=manager_vocab.vocab_map_question['END'] then    
            outputVector[x_seq[i]] = 1
        else
            break
        end
    end
    return outputVector
end

function evaluate_answer(state, pred_answer, prob_answer, selectIDX)
-- testing case for the VQA devi dataset

    if selectIDX == nil then
        selectIDX = torch.range(1, state.x_answer:size(1))
    end
    local pred_answer_word = {}
    local gt_answer_word = state.data_answer
    local gt_allanswer = state.data_allanswer
    local count_correct_mostfreq = 0
    local count_correct_openend = 0
    local count_correct_multiple = 0
    --local result_openend = torch.zeros(pred_answer:size(1))
    --local result_multiple = torch.zeros(pred_answer:size(1))
    local result_question_type_openend = {}
    local result_question_type_multiple = {}
    local result_answer_type_openend = {}
    local result_answer_type_multiple = {}

    local count_question_type = {}
    local count_answer_type = {}

    for sampleID=1, selectIDX:size(1) do
        local i = selectIDX[sampleID]
        if manager_vocab.ivocab_map_answer[pred_answer[i]]== gt_answer_word[i] then
            count_correct_mostfreq = count_correct_mostfreq + 1
        end
        -- estimate the standard criteria (min(#correct match/3, 1))
       -- estimate the mutiply choice case
        local question_type = state.data_question_type[i]
        local answer_type = state.data_answer_type[i]
        if count_question_type[question_type] == nil then
            count_question_type[question_type] = 0
            result_question_type_openend[question_type] = 0
            result_question_type_multiple[question_type] = 0
        end
        if count_answer_type[answer_type] == nil then
            count_answer_type[answer_type] = 0
            result_answer_type_openend[answer_type] = 0
            result_answer_type_multiple[answer_type] = 0
        end
        count_question_type[question_type] = count_question_type[question_type] + 1
        count_answer_type[answer_type] = count_answer_type[answer_type] + 1

        local choices = stringx.split(state.data_choice[i], ',')
        local score_choices = torch.zeros(#choices):fill(-1000000)
        for j=1, #choices do
            local IDX_pred = manager_vocab.vocab_map_answer[choices[j]]
            if IDX_pred ~= nil then
                local score = prob_answer[{i, IDX_pred}]
                if score ~= nil then
                    score_choices[j] = score
                end
            end
        end
        local val_max,IDX_max = torch.max(score_choices,1)
        local word_pred_answer_multiple = choices[IDX_max[1]]
        local word_pred_answer_openend = manager_vocab.ivocab_map_answer[pred_answer[i]]
        local count_curr_openend = 0
        local count_curr_multiple = 0
        local answers = stringx.split(gt_allanswer[i], ',')
        for j=1, #answers do
            if  word_pred_answer_openend == answers[j] then
                count_curr_openend = count_curr_openend + 1
            end
            if word_pred_answer_multiple == answers[j] then
                count_curr_multiple = count_curr_multiple + 1
            end
        end
        local increment = count_curr_openend * 1.0/3
        if increment > 1 then
            increment = 1
        end
        count_correct_openend = count_correct_openend + increment

        result_question_type_openend[question_type] = result_question_type_openend[question_type] + increment
        result_answer_type_openend[answer_type] = result_answer_type_openend[answer_type] + increment

        increment = count_curr_multiple * 1.0/3
        if increment > 1 then
            increment = 1
        end
        count_correct_multiple = count_correct_multiple + increment

        result_question_type_multiple[question_type] = result_question_type_multiple[question_type] + increment
        result_answer_type_multiple[answer_type] = result_answer_type_multiple[answer_type] + increment

    end
    for key, value in pairs(count_question_type) do
        result_question_type_multiple[key] = result_question_type_multiple[key]*1.0/value
        result_question_type_openend[key] = result_question_type_openend[key]*1.0/value
    end
    for key, value in pairs(count_answer_type) do
        result_answer_type_multiple[key] = result_answer_type_multiple[key]*1.0/value
        result_answer_type_openend[key] = result_answer_type_openend[key]*1.0/value
    end


    acc_match_mostfreq = count_correct_mostfreq*1.0/selectIDX:size(1)
    acc_match_multiple = count_correct_multiple*1.0/selectIDX:size(1)
    acc_match_openend = count_correct_openend*1.0/selectIDX:size(1)
    return acc_match_mostfreq, acc_match_openend, acc_match_multiple, result_question_type_multiple, result_answer_type_multiple
end


function train_epoch(state, updateIDX)

    local loss = 0.0
    local N = math.ceil(state.x_question:size(1) / g_params.batchsize)
    local prob_answer = torch.zeros(state.x_question:size(1), manager_vocab.nvocab_answer)
    local pred_answer = torch.zeros(state.x_question:size(1))
    local target = torch.zeros(g_params.batchsize)

    local featBatch_visual = torch.zeros(g_params.batchsize, g_params.vdim)
    local featBatch_word = torch.zeros(g_params.batchsize, manager_vocab.nvocab_question)
    local word_idx = torch.zeros(g_params.batchsize, g_params.seq_length)

    local IDXset_batch = torch.zeros(g_params.batchsize)
    local nSample_batch = 0
    local count_batch = 0
    local nBatch = 0

    local randIDX = torch.randperm(state.x_question:size(1))
    for iii=1, state.x_question:size(1) do
        local i = randIDX[iii]
        local first_answer = -1
        if updateIDX~='test' then
            first_answer = state.x_answer[i]
        end
        if first_answer == -1 and updateIDX == 'train' then
            --skip the sample with NA answer
        else
            nSample_batch = nSample_batch + 1
            IDXset_batch[nSample_batch] = i
            if updateIDX ~= 'test' then
                target[nSample_batch] = state.x_answer[i]
            end
            local filename = state.imglist[i]--'COCO_train2014_000000000092'
            --debugger:enter()
            local feat_visual = state.featureMap[filename]:clone()
            local feat_word = nil

            feat_word = bagofword(state.x_question[i], manager_vocab.nvocab_question)

            word_idx[nSample_batch] = state.x_question[i]
            featBatch_word[nSample_batch] = feat_word:clone()

            featBatch_visual[nSample_batch] = feat_visual:clone()
  
                
            while i == state.x_question:size(1) and nSample_batch< g_params.batchsize do
                -- padding the extra sample to complete a batch for training
                nSample_batch = nSample_batch+1
                IDXset_batch[nSample_batch] = i
                target[nSample_batch] = first_answer
                featBatch_visual[nSample_batch] = feat_visual:clone()
                featBatch_word[nSample_batch] = feat_word:clone()
                word_idx[nSample_batch] = state.x_question[i]
            end 
            if nSample_batch == g_params.batchsize then                
                nBatch = nBatch+1
                word_idx = word_idx:cuda()
                nSample_batch = 0
                target = target:cuda()
                featBatch_word = featBatch_word:cuda()
                featBatch_visual = featBatch_visual:cuda()
                ----------forward pass----------------------
                --switch between the baselines and the memn2n
                local prob_batch = {}
                local output = {}

                if g_params['method'] == 'BOW' then
                    input = featBatch_word
                elseif g_params['method'] == 'BOWIMG' then
                    input = {featBatch_word, featBatch_visual}
                elseif g_params['method'] == 'IMG' then
                    input = featBatch_visual
                else 
                    print('error baseline method \n')
                end
                output = g_model:forward(input)
                err = criterion:forward(output, target)
                prob_batch = output:float()

                loss = loss + err
                for j = 1, g_params.batchsize do
                    prob_answer[IDXset_batch[j]] = prob_batch[j]
                end
                --------------------backforward pass
                if updateIDX == 'train' then
                    g_model:zeroGradParameters()
                    local df = criterion:backward(output, target)
                    local df_model = g_model:backward(input, df)

                    -------------Update the params of baseline softmax---
                    if g_params['uniformLR'] ~= 1 then
                        for i=1, #params_current do

                            local gnorm = gparams_current[i]:norm()
                            if config_layers.gradientClips[i]>0 and gnorm > config_layers.gradientClips[i] then
                                gparams_current[i]:mul(config_layers.gradientClips[i]/gnorm)
                            end
                            grad_last[i]:mul(config_layers.moments[i])
                            local tmp = torch.mul(gparams_current[i],-config_layers.lr_rates[i])
                            grad_last[i]:add(tmp)
                            params_current[i]:add(grad_last[i])
                            if config_layers.weightRegConsts[i]>0 then
                                local a = config_layers.lr_rates[i] * config_layers.weightRegConsts[i]
                                params_current[i]:mul(1-a)
                            end
                            local pnorm = params_current[i]:norm()
                            if config_layers.weightClips[i]>0 and pnorm > config_layers.weightClips[i] then
                                params_current[i]:mul(config_layers.weightClips[i]/pnorm)
                            end

                        end
                    else
                        local norm_dw = g_paramdx:norm()
                        if norm_dw > g_params['max_gradientnorm'] then
                            local shrink_factor = g_params['max_gradientnorm'] / norm_dw
                            g_paramdx:mul(shrink_factor)
                        end
                        g_paramx:add(g_paramdx:mul(-g_params['lr']))
                    end
 
                end

                --batch finished
                count_batch = count_batch+1
                if count_batch == 120 then
                    collectgarbage()
                    count_batch = 0
                end


                featBatch_word = torch.zeros(g_params.batchsize, manager_vocab.nvocab_question)

            end

            --
        end-- end of the pass sample with -1 answer IDX
    end
    -- 1 epoch finished
    local y_max, i_max = torch.max(prob_answer,2)
    i_max = torch.squeeze(i_max)
    pred_answer = i_max:clone()    
    if updateIDX~='test' then

        local gtAnswer = state.x_answer:clone()
        gtAnswer = gtAnswer:long()
        local correctNum = torch.sum(torch.eq(pred_answer, gtAnswer))
        acc = correctNum*1.0/pred_answer:size(1)
    else
        acc = -1
    end
    print(updateIDX ..': acc (mostFreq) =' .. acc)
    acc_match_mostfreq = 0
    acc_match_openend = 0
    acc_match_multiple = 0
    local result_question_multiple = {}
    local result_answer_multiple = {}

    if updateIDX~= 'test' and state.data_allanswer ~= nil then
        -- using the standard evalution criteria of QA virginiaTech
        acc_match_mostfreq, acc_match_openend, acc_match_multiple, result_question_multiple, result_answer_multiple = evaluate_answer(state, pred_answer, prob_answer)
     --   print(updateIDX..': acc.match mostfreq=' .. acc_match_mostfreq)
        print(updateIDX..': acc.dataset (OpenEnd) =' .. acc_match_openend)
        print(updateIDX..': acc.dataset (MultipleChoice) =' .. acc_match_multiple)
--        print('---------acc (MultipleChoice) on different question types----')
--        for key, value in pairs(result_question_multiple) do
--            print(key..'\t'.. value)
--        end
--        print('--------acc (MultipleChoice) on different answer types----')
--        for key, value in pairs(result_answer_multiple) do
--            print(key..'\t\t'..value)
--        end


    end
    print(updateIDX .. ' loss=' .. loss/nBatch)
    return pred_answer, prob_answer
end

