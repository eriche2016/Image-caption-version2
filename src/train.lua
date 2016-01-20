------------2015/1/5-----------------------------------------------
-- increasing the batch size can make the loss decrease more steady
-- otherwise, too small batch size will make the model fluctuates very heavily 
-- otherwise, too small batch size will make the model fluctuates more heavily 
-- otherwise, too small batch size will make the model fluctuates more heavily 
-- otherwise, too small batch size will make the model fluctuates more heavily 
-------------------------------------------------------------------

require 'torch'
require 'nngraph'
require 'nn'
require 'optim'

require 'data_layer'

local utils = require 'utils.misc'
local LSTM = require 'lstm'

------------------------------------------------------------------------------------------------
-- related external paramters 
------------------------------------------------------------------------------------------------
print "=> processing options"
local cmd = torch.CmdLine()
cmd:text()
cmd:text('image caption model parameters')
cmd:text()
cmd:text('Options: ')
cmd:option('-path_vgg_features', '../data/flickr8k/vgg_feats.mat', 'path to vgg feature files')
cmd:option('-path_image_info', '../data/flickr8k/dataset.json', 'path to image info file')
-- model params
cmd:option('-rnn_size', 256, 'the size of the rnn')
cmd:option('-embedding_size', 256, 'size of word/image embeddings')
-- optimization
cmd:option('-learning_rate', 4e-4, 'learning rate')
cmd:option('-learning_rate_decay', 0.95, 'learning rate decay')
cmd:option('-learning_rate_decay_after', 10, 'In number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate', 0.95, 'decay rate for rmsprop')
cmd:option('-batch_size',16, 'the size of a batch')
cmd:option('-max_epochs', 300, 'number of full passes through the training data')
cmd:option('-dropout', 0.5, 'Dropout')
cmd:option('-init_from', '', 'Initialize network parameters from checkpoint at this path')

-- bookkeeping
cmd:option('-seed', 12345, 'Torch manual random number generator seed')
cmd:option('-save_every', 300, 'number of iterations after which store the checkpoint')
cmd:option('-save_every', 300, 'number of iterations after which stroe the checkpoint')
cmd:option('-save_every', 300, 'number of iterations after which stroe the checkpoint')
cmd:option('-save_every', 300, 'number of iterations after which stroe the checkpoint')
cmd:option('-checkpoint_dir', '../model/', 'checkpoint directory')
cmd:option('-save_file', 'google_vic', 'filename of checkpoint to be saved')
cmd:option('-print_every', 10, 'how often to print loss')
-- gpu/cpu
cmd:option('-gpuid', 0, '0-indexed of gpu to use. -1 = cpu')


local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- currently no gpu
torch.setdefaulttensortype('torch.FloatTensor')

if opt.gpuid >= 0 then 
	local ok, cunn = pcall(require, 'cunn')
	local ok2, cutorch = pcall(require, 'cutorch')

	if not ok then print('cunn not found') end 
	if not ok2 then print('cutorch not found') end 

	if ok and ok2 then 
		print('using cuda on gpu ' .. opt.gpuid .. '...')
		cutorch.setDevice(opt.gpuid + 1) -- torch is 1-indexed
		cutorch.manualSeed(opt.seed)
	else 
		print('cannnot run in the gpu mode because of cuda is not installed correctly or cunn and cutorch are not installed')
		print('falling back to cpu mode')
		opt.gpuid = -1 
	end 
end


--------------------------------------------------------------------------------------------------
-- prepare the data 
--------------------------------------------------------------------------------------------------
local loader = Dataset(opt)
loader:loadDataset(opt)
-- vocab size(including the end token)
local vocab_size = loader:getVocabSize()

if not path.exists(opt.checkpoint_dir) then path.mkdir(opt.checkpoint_dir) end 

local do_random_init = true
if string.len(opt.init_from) > 0 then 
	print('loading model from checkpoint' .. opt.init_from)
	local checkpoint = torch.load(opt.init_from)
	protos = checkpoint.protos
	do_random_init = false 
else 
	protos = {} 

    -- construct model from scratch 
	-- word embedding 
	protos.word_embedding_layer = nn.Sequential(): add(nn.LookupTable(vocab_size, opt.embedding_size))
	--protos.word_embedding_layer:add.Dropout(opt.dropout) -- may add dropout
    
	-- image embedding 
	protos.image_embedding_layer = nn.Sequential() 
	local  image_feature_size = loader:getFeatSize()  
	protos.image_embedding_layer:add(nn.Linear(image_feature_size, opt.embedding_size)) -- image feature is of size 4096 by default
	protos.image_embedding_layer:add(nn.ReLU()) -- can also use nn.Tanh() 
	-- can also add dropout layer 
    --protos.image_embedding_layer:add(nn.Dropout(opt.dropout))

	protos.lstm = LSTM.create(opt.embedding_size, opt.rnn_size)

	-- decoder 
	protos.decoder = nn.Sequential() 
	protos.decoder:add(nn.Linear(opt.rnn_size, vocab_size))
	protos.decoder:add(nn.LogSoftMax())
    -- negative log-likelihood loss
	protos.criterion = nn.ClassNLLCriterion()

	if opt.gpuid >= 0 then   -- shift model and criterion to gpu, currently not consider it 
		protos.image_embedding_layer = protos.image_embedding_layer:cuda()
		protos.word_embedding_layer = protos.word_embedding_layer:cuda()
	    protos.lstm = protos.lstm:cuda()
		protos.decoder = protos.decoder:cuda()
		
		protos.criterion = protos.criterion:cuda()
	end 
end 

-- put all the language part  things into one flattened parameters tensor
params, grad_params = utils.combine_all_parameters(protos.word_embedding_layer,  protos.lstm, protos.decoder)

print('Paramters: ' .. params:size(1))
print('Batches: '..loader:getNbatches('train')) -- for flickr8k, its 6000/8 = 750 batches per epoch 

-- initialization 
if do_random_init then 
	params:uniform(-0.08, 0.08)
end

-- create a bunch of clones 
-- just clones the modules 
clones = {} 
-- should keep image embeding layer separate for training 

-- the below is model parameters should be trained to together separatly from image_embedding layer 
clones['word_embedding_layer']  = utils.clone_many_times(protos.word_embedding_layer, loader.max_seq_length + 1) 
clones['lstm'] = utils.clone_many_times(protos.lstm, loader.max_seq_length + 2) -- one for the image, one for 'START' token, the rest are for the sequence 
clones['decoder'] = utils.clone_many_times(protos.decoder, loader.max_seq_length + 2)  -- also clone the decoder because it is part of the top model    
-- need to clone the criterion max_seq_length + 1 times, becuause no need to forward the image to the criterion
clones['criterion'] = utils.clone_many_times(protos.criterion, loader.max_seq_length + 1) 


collectgarbage() -- good practice garbage once in a shile 

-- opt.batch_size = opt.batch_size * 5  -- for sample mode 2

init_state = {} 
local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
if opt.gpuid >= 0 then h_init = h_init:cuda() end 
table.insert(init_state, h_init:clone()) -- init_state[1] = prev_c
table.insert(init_state, h_init:clone()) -- init_state[2] = prec_h

local init_state_global = utils.clone_list(init_state)

--[[used for validation, need to modify  
feval_val = function(max_batches)
	count = 0
	n = loader:getNbatches('val')
	if max_batches ~= nil then n = math.min(n, max_batches) end 

	protos.word_embedding_layer:evaluate()
	protos.image_embedding_layer:evaluate()

	for i = 1, n do 
		word_batch, image_batch = loader:getNextBatch('val')

		word_embeding_batch = protos.word_embedding_layer:forward(word_batch)
		image_embedding_batch = protos.image_embedding_layer:forward(image_batch) -- image_batch: batch_size * 4096(feature size is 4096)

		if opt.gpuid >= 0 then 
			image_embedding_batch:cuda()
		end 

		rnn_state = {[0] = init_state_global}
		-- input the image
		lst = protos.clones.lstm[loader.max_seq_length + 1]:forward({image_embedding_batch, unpack(rnn_state[0])})
		for t = 2, loader.max_seq_length+1 do 
			lst = protos.clones.lstm[t]:forward({word_embeding_batch:select(2, t-1), unpack(rnn_state[t-1])}) -- lst contains next_c, next_h 
			rnn_state[t] = {} 
			for i = 1, #init_state do table.insert(rnn_state[t], lst[i]) end 
			prediction = protos.clones.decoder:forward(lst)
		end 

	end 

	protos.word_embedding_layer:training()
	protos.image_embedding_layer:training()

	-- return count / (n * opt.batch_size) -- ??
end 
--]]

-- used for training
-- do backward and forward propagation and return loss and grad_params

feval = function(x) 
    if x ~= params then 
    	params:copy(x)
    end

    grad_params:zero()
    ----------------------------------------------------------------------------------------------
    --get minibatch
    ---------------------------------------------------------------------------------------------
	word_batch, image_batch = loader:getNextBatch()
    local start_token_batch = torch.LongTensor(opt.batch_size):fill(1)
    
    
    if opt.gpuid >= 0 then 
        -- word_batch = word_batch:cuda()  -- no need to shift to gpu, because already there 
	    -- image_batch = image_batch:cuda() -- no need to shift to gpu, because already there
        start_token_batch = start_token_batch:cuda()
    end 


    ----------------------------------------------------------------------------------------------
    -- forward pass 
    ----------------------------------------------------------------------------------------------
    loss = 0.0
    rnn_state = {[0] = init_state_global}
    embeddings = {} 
    predictions = {} -- record the output at each time step 
    loss = 0.0
    number_predictions = 0 

    for t = 1, loader.max_seq_length+2 do 
        if t ==1 then 
        	-- t = 1, just forward  a batch of images 
            image_embedding_batch = protos.image_embedding_layer:forward(image_batch) 
            embeddings[t] = image_embedding_batch  -- record embeddings[t]

	        lst = clones.lstm[1]:forward({embeddings[t], unpack(rnn_state[0])}) -- prev_c, prev_h 
            rnn_state[1] = {}
            for i = 1, #init_state do table.insert(rnn_state[1], lst[i]) end 
            -- output of the decoder, actually no need to record it, hereby just record it 
            predictions[1] = clones.decoder[1]:forward(lst[#lst]) -- actually no need to output the predictions because no loss is calculated during training 
           
        elseif t == 2 then 
            -- forward start token
            start_batch_embedding = clones.word_embedding_layer[t-1]:forward(start_token_batch)
            embeddings[t] = start_batch_embedding
            lst = clones.lstm[t]:forward{embeddings[t], unpack(rnn_state[t-1])}         -- lst[t] = {prev_c, prev_h}

            rnn_state[t] = {} 
            for i = 1, #init_state  do table.insert(rnn_state[t], lst[i]) end  -- rnn_state[t] = {prev_c, prev_h}
            
            number_predictions = number_predictions + opt.batch_size

            predictions[t] = clones.decoder[t]:forward(lst[#lst])  
            loss = loss + opt.batch_size * clones.criterion[t-1]:forward(predictions[t], word_batch:select(2, t-1)) -- its target is at t-1th 
        else 
            --forward the sequence, t == 3,  
            --input is the first batch token in the sequence batch,  here t ==3, so 3 - 2 == 1 
            local input_batch = word_batch:select(2, t-2) 
            local targets = word_batch:select(2, t-1)

            word_embedding_batch = clones.word_embedding_layer[t-1]:forward(input_batch)
            embeddings[t] = word_embedding_batch
            lst = clones.lstm[t]:forward({embeddings[t], unpack(rnn_state[t-1])})

            rnn_state[t] = {}
            for i = 1, #init_state do table.insert(rnn_state[t], lst[i]) end
            --print(lst[#lst]:size())
            predictions[t] = clones.decoder[t]:forward(lst[#lst])
            
            -- zero out the related predictions which corresponding to 1 in input 
            mask = torch.gt(input_batch, 1):view(-1, 1) -- 0 indicates end, 1 indicates no the end, so not zero it out             
            e_mask = torch.expand(mask, predictions[t]:size(1), predictions[t]:size(2))
            
            --print('line 274')
            if opt.gpuid >= 0 then 
                e_mask = e_mask:float():clone():cuda()  
            else 
                e_mask = e_mask:float():clone() 
            end 
            
            number_predictions = number_predictions + mask:sum()
            predictions[t] = predictions[t]:cmul(e_mask)

            -- loss
	        loss = loss + opt.batch_size * clones.criterion[t-1]:forward(predictions[t], targets)
        end 

    end

    -- average the loss w.r.t batch size 
    -- in the classNLL, we have already done that for each input, really need to divide it by the batch_size  other than sequence length????????????????????
    -- answer 2015 12/31: no 
    loss = loss / number_predictions 
	

    -- average the loss w.r.t batch size
    -- loss = loss / opt.batch_size 
    -- no need to average the loss by batch size , because when doing batch input, ClassNLLCriterion has already average it 

    -----------------------------------------------------------------------------------------------
	-- backward pass  
	-----------------------------------------------------------------------------------------------
    -- create initial  hidden state  at loader.max_seq_length + 2 
    drnn_state = {[loader.max_seq_length+2] = utils.clone_list(init_state, true)}
    drnn_state[loader.max_seq_length+2][2] = nil    

    for t = loader.max_seq_length+2, 1, -1 do 
        if t == 1 then
            -- train the image model part 
            -- the image   
            dlst = clones.lstm[t]:backward({embeddings[t], unpack(rnn_state[t-1])}, drnn_state[t])
           
            -- training the image embedding part 
            -- backward through the embeddings
            protos.image_embedding_layer:zeroGradParameters() 
            protos.image_embedding_layer:backward(image_batch, dlst[1]) 
            -- update parameters
            protos.image_embedding_layer:updateParameters(opt.learning_rate)
            -- print('im part start')
            -- _, gp = protos.image_embedding_layer:getParameters() 
            -- print(gp:sum())
            -- print('impart end ')

        elseif t == 2 then 
            
            -- the start token batch 
            dloss = clones.criterion[1]:backward(predictions[t], start_token_batch)
           

            dloss:mul(opt.batch_size) -- because the mask will be all one, so no need to multilpy with mask 
            
            doutput_t = clones.decoder[2]:backward(rnn_state[2][#rnn_state[2]], dloss) 
            drnn_state[t][2]:add(doutput_t)
            -- backward through LSTM 
            dlst = clones.lstm[t]:backward({embeddings[t], unpack(rnn_state[t-1])}, drnn_state[t])
            
            -- record to be used in previous step, ie, in 1 step 
            drnn_state[t-1] = {} 
            table.insert(drnn_state[t-1], dlst[2]) -- drnn_state[1] = dlst[2]
            table.insert(drnn_state[t-1], dlst[3]) -- drnn_state[2] = dlst[3]

            -- backward through the embeddings 
            clones.word_embedding_layer[t-1]:backward(start_token_batch, dlst[1])   

        else -- the sequence  
             -- backward to the criterion
            input_cur = word_batch:select(2, t-2)  -- coresponding targets is word_batch:select(2, t-1)
         
            dloss = clones.criterion[t-1]:backward(predictions[t], word_batch:select(2, t-1))
        
            -- we need to zero out corresponding dloss   
            -- zero out the related predictions which corresponding to 1 in input 
            mask = torch.gt(input_cur, 1):view(-1, 1) -- 0 indicates end, 1 indicates no the end, so not zero it out             
            

            e_mask = torch.expand(mask, dloss:size(1), dloss:size(2))
           
            if opt.gpuid >= 0 then 
                e_mask = e_mask:float():clone():cuda()  
            else 
                e_mask = e_mask:float():clone() 
            end 

            ----------------------------------------------------------------------------------------------------------------------   
            -- !!!problem: dloss cannot be corectly computed, all are scaled
            -- ex: if no element is zero out, then dloss:sum() will be -1, otherwise, it will be a scaled version, for example, because all the dloss for batchsize of 8
            -- each row will be assigned a weight of 1/8, which is terrible 
            -------------------------------------------------------------------------------------------------------------------
            dloss = dloss:cmul(e_mask) 
            
            if t == loader.max_seq_length+2 then 
                assert(drnn_state[loader.max_seq_length+2][2] == nil)
                doutput_t = clones.decoder[t]:backward(rnn_state[t][#rnn_state[t]], dloss) -- rnn_state[t][#rnn_state[t]] record the input of the decoder at time step t 
                drnn_state[t][2] = doutput_t
            else
                doutput_t = clones.decoder[t]:backward(rnn_state[t][#rnn_state[t]], dloss) -- rnn_state[t][#rnn_state[t]] record the input of the decoder at time step t 
                drnn_state[t][2]:add(doutput_t)
            end 
            
            -- backward through LSTM 
            -- Note that dlst = dlst, rnn_state[t-1], this will be used at previous step, note this is a backward step 
            dlst = clones.lstm[t]:backward({embeddings[t], unpack(rnn_state[t-1])}, drnn_state[t])
            
            drnn_state[t-1] = {}
            table.insert(drnn_state[t-1], dlst[2]) -- drnn_state[1] = dlst[2]
            table.insert(drnn_state[t-1], dlst[3]) -- drnn_state[2] = dlst[3]
           
            -- backward through the embeddings 
            clones.word_embedding_layer[t-1]:backward(input_cur, dlst[1]) 
            -- print('lm part start')
            -- print(grad_params:sum())  -- maybe 1-norm is more suitable to give us some intuition  
            -- print('lm part end')
        end 
    end 

    --grad_params:clamp(-5, 5)
    grad_params:div(number_predictions):clamp(-5, 5)

    return loss, grad_params
end 

-----------------------------
------------------------------------------------------------------------
-- set optimization  parameters configuration
--------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------
-- train the model 
-----------------------------------------------------------------------------------------------------

local losses = {}

local optim_state = {learningRate = 4e-4, alpha = 0.8, epsilon = 1e-8}
--local optim_state = {learningRate = 1e-1}

local iterations = opt.max_epochs * loader:getNbatches() 

for i = 1, iterations do 
    --local _, loss = optim.adagrad(feval, params, optim_state)
    local _, loss = optim.rmsprop(feval, params, optim_state)
    losses[#losses + 1] = loss[1]
   
    --not save model, just test it
    if i % opt.save_every == 0 then 
	    print(string.format('saving checkpoint to ' .. opt.checkpoint_dir))
        local checkpoint = {} 
	    checkpoint.protos = protos
	    torch.save(opt.checkpoint_dir..opt.save_file..'.t7', checkpoint)
    end


    if i % opt.print_every == 0 then 
        print(string.format("iterations %4d, loss = %6.8f", i, loss[1]))
    end 
    
    if i % 10 == 0 then 
        collectgarbage() 
    end 
end 

