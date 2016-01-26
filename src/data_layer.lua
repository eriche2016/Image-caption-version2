-- input data 
--  support cuda, ie shift all the data to gpu
-- modified v1: currently partially support cuda 
-- later can move all data to cuda to support cuda compiuting fullyfledged
-- modified v2: currently support cuda from end to end 


-- modified v1: currently partially support cuda 
-- later can move all data to cuda to support cuda compiuting fullyfledged
-- modified v2: currently support cuda from end to end 
local json = require 'json'
local matio = require 'matio'
local hdf5 = require 'hdf5'

local Dataset = torch.class('Dataset')

function Dataset:__init(opt)
	self.batch_size = opt.batch_size or 32 -- by default batch_size is of size 32 
	self.seq_len = 0

	self.begin_index = 1
	self.cycle = 0
    self.gpuid = opt.gpuid or -1 -- by default use cpu, ie load data to cpu 
    self.split = opt.split or 'train'
    self.seq_per_image = self.seq_per_image or 5 
end

function Dataset:loadDataset(opt) 
    print('loading features files extracted using vgg')
    local img_feats = matio.load(opt.path_vgg_features)
    self.image_feats = img_feats.feats
    print('loading json files which contains dict')
    local info = json.load(opt.coco_dict_split) 
    self.idx2word = info['idx_to_word']
    self:getVocabSize() 

    self.split_info = info['imgs']  --size = 123287
    
    -- open hdf5 file 
    print('loading hdf5 file')
    self.h5_file = hdf5.open(opt.h5_file, 'r') 
    -- extract label sequences(of number) from the file
    local seq_size = self.h5_file:read('/labels'):dataspaceSize() 
    -- seq_size = {616767, 49}
    -- set seq_len 
    self.seq_len = seq_size[2] -- 49
    
    print('max squence length is ' .. self.seq_len)

    -- load the pointers to RAM(should be small because torch's limitations on the overhead of table)
    self.label_start_idx = self.h5_file:read('/label_start_idx'):all()
    self.label_end_idx = self.h5_file:read('/label_end_idx'):all() 

    -- separate out indexes for each of the provided splits
    self.split_idx = {} 
    self.iterators = {} 
    
    for i, img_split in pairs(self.split_info) do 
        local split = img_split['split'] 

        if not self.split_idx[split] then 
            -- initialize new split, which will store split for each split image
            self.split_idx[split] = {} 
            self.iterators[split] = 1 
        end 
        table.insert(self.split_idx[split], i)
    end 

    -- check it 
    for k, v in pairs(self.split_idx) do 
        print(string.format('assigned %d images to split %s', #v, k)) 
    end 

    collectgarbage() 

end 

function Dataset:resetIterator(split)
    self.iterators[split] = 1 
end 

function Dataset:getVocabSize() 
    if not self.vocab_size then 
        self.vocab_size = 0 
        for k, v in pairs(self.idx2word) do 
            self.vocab_size = self.vocab_size + 1 
        end 
    end 
    self.vocab_size = self.vocab_size  return self.vocab_size -- include UNK token, note that this is not END token  
end


function Dataset:getNextBatch() 
    local split = self.split or 'train'
    local seq_per_image = self.seq_per_image or 5 -- by default set it to 5 

    local split_idx = self.split_idx[split]

    assert(split_idx, 'split' .. split .. 'not found') 

    -- make batch of image features and sentence indices 
    local sent_tensor = torch.Tensor(self.batch_size*self.seq_per_image, self.seq_len+1):fill(0) -- the last element indicates the end token 
    local image_batch = torch.Tensor(self.batch_size, self.image_feats:size(1))
    
    local max_index = #split_idx
    local wrapped = false 

    for i = 1, self.batch_size do 
        local ri = self.iterators[split] -- get next index from iterator 
        local ri_next = ri + 1 

        if ri_next > max_index then 
            ri_next = 1
            wrapped = true
            self.cycle  = self.cycle + 1
        end 
        self.iterators[split] = ri_next

        idx = split_idx[ri]

        
        assert(idx ~= nil, 'bug: split' .. split .. 'was accessed out of bound' .. ri)

        -- fetch the image features 
        image_batch[{{i}, {}}] = self.image_feats:select(2, idx)
        
        -- fetch the sequence 
        local idx1 = self.label_start_idx[idx]
        local idx2 = self.label_end_idx[idx]
        
        local ncap = idx2 - idx1 + 1  -- number of captions available for the image 
        assert(ncap > 0, 'an image has no labels?? cannot handle')
        local seq
        if ncap < seq_per_image then 
            -- we need subsample(with replacement)
            seq = torch.LongTensor(seq_per_image, self.seq_len)
            for q = 1, seq_per_image do
                local idx1 = torch.random(idx1, idx2)
                seq[{{q, q}}] = self.h5_file:read('labels'):partial({idx1, idx1}, {1, self.seq_len})
            end 

        else
            -- there are enough captions for this image 
            local idx1 = torch.random(idx1, idx2-seq_per_image+1)
            seq = self.h5_file:read('labels'):partial({idx1, idx1 + seq_per_image - 1}, {1, self.seq_len})
        end 

        local i1 = (i-1) * seq_per_image + 1 
        sent_tensor[{{i1, i1 + seq_per_image-1}, {1, self.seq_len}}]  = seq 
    end

    -- repeate the batch of features, so that each features will expand to self.seq_per_image features, note that repeatTensor function reallocates memory  
    image_batch  = torch.repeatTensor(image_batch, 1, self.seq_per_image)
    image_batch = image_batch:view(-1, self.image_feats:size(1))

    
    -- we need to shift data to cuda 
    if self.gpuid >= 0 then 
        image_batch = image_batch:cuda()
        sent_tensor = sent_tensor:cuda()
    end
    
    return image_batch, sent_tensor
end 


function Dataset:getIdx2Word() 
    return self.idx2word
end


function Dataset:getFeatSize()  -- will be 4096 here
	return self.image_feats:size(1)
end


--[[
-------------------deprecated-----------------------------
-- sample data mode 1, choose image, then sample a sequence 
function Dataset:getNextBatch(split) 
	local split = split or 'train'
	
	-- note that 1 indicates the end, we just
	local sent_tensor = torch.Tensor(self.batch_size, self.max_seq_length+1):fill(1)  
	local image_batch = torch.Tensor(self.batch_size, self.image_feats:size(1))  -- will be self.batch_size * 4096



	local t = 1
	for i = self.begin_index, math.min(self.begin_index + self.batch_size - 1, #self.images_info[split]) do 
		    local sentid_r = torch.random(1, 5) -- 1, 2, 3, 4, 5
            local image_idx = self.images_info[split][i]['imgid']
			local sent_idx = image_idx*5 + sentid_r  -- note that image_idx is 0-indexed 

			image_batch[{{t}, {}}] = self.image_feats:select(2, image_idx+1)

			sent_tensor[{{t}, {}}] = self.train_sents_tensor:select(1, sent_idx)
			t = t + 1
    end 
    -- here, t must be euqal to self.batch_size+1 or less than  self.batch_size + 1
	self.begin_index = self.begin_index + self.batch_size
	--reset self.begin_index to 1 if it is larger than  #self.images_info['train']
	--if self.begin_index > #self.images_info[split] then 
	
    if t <= self.batch_size  then -- if this condition happens, this means we donnot extract self.batch_size number of samples 
        -- if t is less than self.batch_size, then we set self.begin_index to 1 and sample from the very start 
        self.begin_index = 1
		self.cycle = self.cycle + 1

		-- make sure every batch is of the same size regardless of the batch_size being set previously
		-- when t = self.batch_size + 1, then: 'for remains = 1, 0 do' will not execute , so it works right
		for remains = 1, self.batch_size -t + 1 do 
            local sentid_r = torch.random(1, 5) -- 1, 2, 3, 4, 5
			local image_idx = self.images_info[split][remains]['imgid']
			local sent_idx = image_idx * 5 + sentid_r

			image_batch[{{t}, {}}] = self.image_feats:select(2, image_idx+1)

			sent_tensor[{{t}, {}}] = self.train_sents_tensor:select(1, sent_idx)
			t = t + 1
			self.begin_index = self.begin_index + 1
		end  

	end

    
    -- we need to shift data to cuda 
    if self.gpuid >= 0 then 
        image_batch = image_batch:cuda()
        sent_tensor = sent_tensor:cuda()
    end

	--return inputs, image_batch
	return sent_tensor, image_batch

end
--]] 
