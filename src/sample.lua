require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

-- bad solution for storing all the file in cuda format 
-- later i will change it 

require 'cutorch'
require 'cunn'

require  'data_layer'

cmd = torch.CmdLine() 
cmd:text()
cmd:text('sample a sentence to description based on the specific image features')
cmd:text() 
cmd:text('Options')
cmd:option('-path_vgg_features', '../data/flickr8k/vgg_feats.mat', 'path to vgg feature files')
cmd:option('-path_image_info', '../data/flickr8k/dataset.json', 'path to image info file')
cmd:option('idx2word_file', '../data/flickr8k/Idx2Word.t7', 'specify the path of the dictionary, currently only support the flickr8k')
cmd:option('model', '../model/google_vic.t7', 'the directory of the model')
cmd:option('-seed', 123, 'random number generator\'s seed')

-- gpu/cpu
cmd:option('-gpuid', 0, '0-indexed of gpu to use. -1 = cpu')  -- must be tested using gpu version 


cmd:text() 

-- parse input params 
opt = cmd:parse(arg) 

-- load related dataset 
torch.manualSeed(opt.seed)


if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- torch is 1-indexed
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back to CPU mode')
        opt.gpuid = -1
    end
end


-- load the map 
-- used to map the predicted indx to word 
idx2word = torch.load(opt.idx2word_file)

--[[
-- word2idx, no need to compute the inverse map 
vocab = {} 
for idx, word in pairs(idx2word) do vocab[word] = idx end 
--]]

-- load dataset 
local loader = Dataset(opt)

loader:loadDataset(opt)

image_feats, _ = loader:getData() 

-- load the model 
local checkpoint = torch.load(opt.model)
protos = checkpoint.protos

local h_init = torch.zeros(3, 256)

-- shift it to cuda 
h_init = h_init:cuda() 

-- init_state 
init_state = {} 
table.insert(init_state, h_init:clone())
table.insert(init_state, h_init:clone())

rnn_state = {[0] = init_state} 

-- test_image tensor
test_image_feat = torch.Tensor(3, 4096):cuda() 
test_image_feat[{{1, 3}, {}}] = image_feats:narrow(2, 1, 3)

-- start token 
start_token = torch.ones(3):cuda() 

prev_idx = start_token 

seq_index = torch.LongTensor(3, 37):fill(1):cuda()-- only one images 


log_probs = nil 

for t  = 1, 37 do 

    if t == 1 then 
        -- forward the image features 
       test_image_embedding = protos.image_embedding_layer:forward(test_image_feat)

        --print(test_image_embedding:size())
        lst = protos.lstm:forward({test_image_embedding, unpack(rnn_state[0])})

        rnn_state[1] = {} 

        for i = 1, #init_state do table.insert(rnn_state[1], lst[i]:clone()) end 
    else
        -- forward the prev token
        cur_token_embedding = protos.word_embedding_layer:forward(prev_idx)
        --print('line 84')
        lst = protos.lstm:forward({cur_token_embedding, unpack(rnn_state[t-1])})

        rnn_state[t] = {} 

        for i = 1, #init_state do table.insert(rnn_state[t], lst[i]:clone()) end
        
        -- forward the lstm output to decoder 
        log_probs = protos.decoder:forward(rnn_state[t][#rnn_state[t]]) 
        local _, prev_idx = log_probs:max(2)
        print(prev_idx)

        seq_index[{{}, {t-1}}] = prev_idx
     end 
end 


