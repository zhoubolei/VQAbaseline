require 'paths'
require 'cunn'
require 'nn'

local stringx = require 'pl.stringx'
local file = require 'pl.file'
local debugger = require 'fb.debugger'

paths.dofile('opensource_base.lua')
paths.dofile('opensource_utils.lua')

local rank_context = { }

function rankPrepare() 
    local opt = initial_params()
    rank_context.context = loadPretrained(opt)

    local manager_vocab = rank_context.context.manager_vocab
    local model = rank_context.context.model

    -- Question-Answer matrix: 
    --
    -- The entire model is like r = M_i * v_i + M_w * M_emb * v_onehot, there is no non-linearity.
    -- So we could just dump M_w * M_emb from here.
    --
    -- nvocab_answer * embed_word 
    local M_w = model.modules[3].weight[{ { }, { 1, opt.embed_word } }]
    -- embed_word * nvocab_question
    local M_emb = model.modules[1].modules[1].modules[2].weight

    -- nvocab_answer * nvocab_question
    rank_context.rankWord_M = torch.mm(M_w, M_emb)

    -- Image-Answer matrix: vdim * nvocab_answer
    rank_context.rankImage_M = model.modules[3].weight[{ { }, { opt.embed_word + 1, opt.embed_word + opt.vdim }}]:transpose(1, 2)

    -- load test data
    local testSet = 'test-dev2015' --'test2015' and 'test-dev2015'
    local state, _ = load_visualqadataset(opt, testSet, manager_vocab)

    rank_context.imglist = { }
    rank_context.inv_imglist = { }

    for k, v in pairs(state.featureMap) do
        table.insert(rank_context.imglist, k)
        rank_context.inv_imglist[k] = #rank_context.imglist
    end
    rank_context.featureMap = state.featureMap

    -- Dump rankImg matrix.
    local batch = 128
    local mat_feat = M_w.new():resize(batch, opt.vdim):zero()
    local rankImage_M2 = M_w.new():resize(#rank_context.imglist, manager_vocab.nvocab_answer):zero()

    for i = 1, #rank_context.imglist, batch do
        local this_batch = math.min(batch, #rank_context.imglist - i + 1)
        for j = 1, this_batch do
            local img_name = rank_context.imglist[j + i - 1]
            mat_feat[j]:copy(rank_context.featureMap[img_name])
        end
        -- batch * nvocab_answer
        local this_output = torch.mm(mat_feat:sub(1, this_batch), rank_context.rankImage_M)
        -- Collect the score.
        rankImage_M2[{ {i, i + this_batch - 1}, { } }]:copy(this_output)
    end
    -- nvocab_answer * #rank_context.imglist
    rank_context.rankImage_M2 = rankImage_M2:transpose(1, 2) 
end

function rankWord(answer, topn)
    local manager_vocab = rank_context.context.manager_vocab
    local model = rank_context.context.model

    -- Convert the answer to idx and return topn words
    -- require 'fb.debugger'.enter()
    if answer:sub(1, 1) == '"' then answer = answer:sub(2, -2) end
    local answerid = manager_vocab.vocab_map_answer[answer]
    if answerid == nil then return end

    local score = rank_context.rankWord_M[answerid]:clone():squeeze()
    local sortedScore, sortedIndices = score:sort(1, true)

    local res = { }
    for i = 1, topn do
        local w = manager_vocab.ivocab_map_question[sortedIndices[i]]
        table.insert(res, { word = w, score = sortedScore[i] })
    end

    return res 
end

function rankImage(answer, topn) 
    local manager_vocab = rank_context.context.manager_vocab
 
    -- Given the answer, rank the image most relevant to the answer.
    -- Convert the answer to idx and return topn words
    if answer:sub(1, 1) == '"' then answer = answer:sub(2, -2) end
    local answerid = manager_vocab.vocab_map_answer[answer]
    if answerid == nil then return end

    local score = rank_context.rankImage_M2[answerid]:clone():squeeze()
    local sortedScore, sortedIndices = score:sort(1, true)

    local res = { }
    for i = 1, topn do
        local idx = sortedIndices[i]
        -- Get the image link.
        local imgname = rank_context.imglist[idx]
        local id = tonumber(imgname:match("_(%d+)"))
        local url = "http://mscoco.org/images/" .. tostring(id)
        table.insert(res, { idx = idx, imgname = imgname, url = url, score = sortedScore[i] })
    end

    return res 
end

local function smart_split(s, quotes)
    quotes = quotes or { ['"'] = true }
    local res = { }
    local start = 1
    local quote_stack = { }
    for i = 1, #s do
        local c = s:sub(i, i)
        if c == ' ' then
            if #quote_stack == 0 then
                table.insert(res, s:sub(start, i - 1))
                start = i + 1
            end
        elseif quotes[c] then
            if #quote_stack == 0 or c ~= quote_stack[#quote_stack] then
                table.insert(quote_stack, c)
            else
                table.remove(quote_stack)
            end
        end
    end
    table.insert(res, s:sub(start, -1))
    return res
end

local commands = {
    rankw = {
        exec = function(tokens)
            local topn = tonumber(tokens[3]) or 5
            local res = rankWord(tokens[2], topn)
            local success = false
            local s = ""
            if res then
                for i, v in pairs(res) do
                    s = s .. string.format("[%d]: %s (%.2f)", i, v.word, v.score) .. "\n"
                end
                success = true
            end
            return success, s
        end,
        help = "\"rankw answer 5\" will rank the word and show top5 question words that give the answer. If 5 is not given, default to top5. If the answer contains white space, use quote to separate."
    },
    ranki = {
        exec = function(tokens) 
            local topn = tonumber(tokens[3]) or 5
            local res = rankImage(tokens[2], topn)
            local success = false
            local s = ""
            if res then 
                for i, v in pairs(res) do
                    s = s .. string.format("[%d]: %s (%.2f)", i, v.imgname, v.score) .. "\n"
                end
                for i, v in pairs(res) do
                    s = s .. string.format("<img src=\"%s\"/>", v.url)
                end
                s = s .. "\n"
                success = true
            end
            return success, s 
        end,
        help = "\"ranki answer 5\" will rank the image and show top5 images that give the answer. If 5 is not given, default to top5. If the answer contains white space, use quote to separate."
    },
    quit = {
        exec = function(tokens) 
            return true, "Bye bye!", true
        end,
        help = "Quit the interactive environment"
    }
}

local history = ""

local function init_webpage()
    history = "<html>"
end

local function write2webpage(s)
    local lines = stringx.split(s, "\n")
    for i = 1, #lines do
        history = history .. lines[i] .. "<br>\n"
    end
end

local function save_webpage(filename)
    io.open(filename, "w"):write(history .. "</html>"):close()
end

function run_interactive()
    -- Run interactive environment.
    print("Preload the model...")
    rankPrepare()

    -- Generate help string.
    local help_str = "Usage: \n"
    for k, v in pairs(commands) do
        help_str = help_str .. k .. ":\n  " .. commands[k].help .. "\n"
    end
    print("Ready...")

    init_webpage()

    while true do
        local command = io.read("*l")
        local tokens = smart_split(command)
        if #tokens > 0 then
            if tokens[1] == "help" then 
                print(help_str)
            else
                local success, s, is_quit = commands[tokens[1]].exec(tokens)
                print(s)
                if success then
                    write2webpage(command)
                    write2webpage(s)
                    save_webpage("/home/yuandong/public_html/webpage.html")
                end
                if is_quit then break end
            end
        end
    end
end

run_interactive()
