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
	self.max_seq_length = 0
	self.begin_index = 1
	self.cycle = 0
    self.gpuid = opt.gpuid or -1 -- by default use cpu, ie load data to cpu 
end

-----------added on Jan 21, 2016--------------------------
function Dataset:loadDataset(opt) 
    print('loading features files extracted using vgg')
    local img_feats = matio.load(opt.path_vgg_features)
    self.image_feats = image_feats.feats
    print('loading json files which contains dict')
    local info = json.load(opt.coco_dict_split) 
    self.idx2word = info['idx_to_word']
    self.split_info = info['imgs']  --size = 123287
    
    -- open hdf5 file 
    print('loading hdf5 file')
    self.h5_file = hdf5.open('opt.h5_file', 'r') 
    -- extract label sequences(of number) from the file
    local seq_size = self.h5_file:read('/labels'):dataspaceSize() 
    -- seq_size = {616767, 49}
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

    return self.vocab_size 
end 

-------------------dreprecated version---------------------------------------

function Dataset:loadDataset(opt)
	local image_feats = matio.load(opt.path_vgg_features)
	self.image_feats = image_feats.feats
   
    --[[
    --support cuda processing 
    --shift image feature data to cuda 
    if self.gpuid >= 0 then    
        self.image_feats = self.image_feats:cuda() 
    end 
    --]] 
    

	local img_info_raw = json.load(opt.path_image_info)
	local images_info = img_info_raw["images"]

	local img_train, img_test, img_val = {}, {}, {}

	-- split the data into 'train', 'val', 'test' part
	for i = 1, #images_info do 
		if images_info[i]["split"] == "train" then
			table.insert(img_train, images_info[i])
		elseif images_info[i]["split"] == "test" then
			table.insert(img_test, images_info[i])
		else --img[i]["split"] == "val" 
			table.insert(img_val, images_info[i])
		end
	end

	-- empty images_info table 
	self.images_info = {}
	self.images_info["train"] = img_train
	self.images_info["test"] = img_test
	self.images_info["val"] = img_val
     

	collectgarbage()

	-- call the __buildVocab to get the dictionary and inverse map 
	self: __buildVocab()

    -- compress all the training sentences into a big Long tensor of size, later can compress the val and test to two big tensor repectively 
    self.train_sents_tensor = torch.LongTensor(5 * #self.images_info["train"], self.max_seq_length + 1):fill(1) -- make sure all the element has an 'end' token 
    
    for i = 1, #self.images_info["train"] do
         
        local sents = self.images_info["train"][i]["sentences"] -- descriptions to describe i-th image(5 description per each image)
        for j = 1, #sents do -- 5 sents per image 
            -- just validate that imgid is equal to .... 
            image_id = self.images_info["train"][i]["sentences"][j]["imgid"]
       
            --[[
            if image_id+1 ~= i then 
                error('omg, img_id+1 is not equal to corresponding to i ')
            end --]]


             local index_list = {}
             local sent_words = self.images_info["train"][i]['sentences'][j]['tokens']  --jth caption for ith image 
 
             for k = 1, #sent_words do 
			 	 table.insert(index_list, self.word2idx[sent_words[k]])
			 end  
             -- in the end, for the flickr8k train data, will be 30,000 * 38
             self.train_sents_tensor[{{(i-1)*5+j}, {1, #index_list}}] = torch.LongTensor(index_list)
        end 
    end
    -- bad idea to shift all the data to gpu, donot do it, we only need to load a batch to gpu 

    -- shift the training sents data to cuda 
    --[[
    if self.gpuid >= 0 then
        self.train_sents_tensor = self.train_sents_tensor:cuda()
    end 
    --]]

end 

-- helper method  
function Dataset:__buildVocab(word_count_threshod)
	local word_count_threshod = word_count_threshod or 1 
	print(string.format('=> word threshod %d', word_count_threshod))
	
    -- do stastics on train dataset
	local word_counts = {}
	local vocab_map = {}
	local vocab_idx = 1
	local nsents = 0  -- record the number of sents in the training set 

	for i = 1, #self.images_info['train'] do 
		local sents = self.images_info['train'][i]['sentences']
		for j = 1, #sents do  -- 5 sents per image
			nsents = nsents + 1
			sent_tokens = sents[j]['tokens']

			for k = 1, #sent_tokens  do 
				if word_counts[sent_tokens[k]] == nil then
					word_counts[sent_tokens[k]] =  1
				else -- just increment the count of this word
					word_counts[sent_tokens[k]] = word_counts[sent_tokens[k]] + 1
				end
			end

			-- in flickr8k, max occurs at images_info['train'][723],  at training set: filename : "2354456107_bf5c766a05.jpg"
			if self.max_seq_length < #sent_tokens then 
                -- update the value of max_seq_length
				self.max_seq_length = #sent_tokens
			end 
		end
	end

	for word, v in pairs(word_counts) do 
		if v >= word_count_threshod then
			if vocab_map[word] == nil then  -- always be true, because we have already doing statistics on the sents dataset  
				vocab_idx = vocab_idx + 1
				vocab_map[word] = vocab_idx
			end 
			-- if word alread exits in the vocab_map, we do nothing
		end 
	end

	-- return vocab_map
	-- now vocab_map contains K distinct words
	-- but there are K + 1 outputs/inputs(the start/end token(same token) )	
	-- we use idx2word to take the predicted indices and map them to words for output visualizations
	-- we use word2idx to take raw words and get their index 
	self.idx2word = {}
	self.word2idx = {}

	self.idx2word[1] = "." -- period at the end of sentence. only needed at the output 
	self.word2idx["#START#"] = 1 -- make the first word be the start token 
	-- construct our word2idx and idx2word using vocab_map 
	local idx = 2
	for w, _ in pairs(vocab_map) do 
		self.word2idx[w] = idx
		self.idx2word[idx] = w
		idx = idx + 1
	end 
end

function Dataset:getData()
	return self.image_feats, self.train_sents_tensor
end

function Dataset:getVocabSize()
	return #self.idx2word -- including end token(1)
end

function Dataset:getIdx2Word() 
    return self.idx2word
end 

function Dataset:getWord2Idx()
    return self.word2idx
end 

function Dataset:getFeatSize()  -- will be 4096 here
	return self.image_feats:size(1)
end

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


--[[
-- sample mode 2, work
function Dataset:getNextBatch(split) 
	local split = split or 'train'
	
	-- note that 1 indicates the end, we just
	local sent_tensor = torch.Tensor(self.batch_size*5, self.max_seq_length+1):fill(1)  
	local image_batch = torch.Tensor(self.batch_size*5, self.image_feats:size(1))  -- will be self.batch_size * 4096

    
    -- we need to shift data to cuda 
    if self.gpuid >= 0 then 
        image_batch = image_batch:cuda()
        sent_tensor = sent_tensor:cuda()
    end 
    -- 5 * images + 5 sentence 
    --
    
	local t = 1
	for i = self.begin_index, math.min(self.begin_index + self.batch_size - 1, #self.images_info[split]) do 
         
            local image_idx = self.images_info[split][i]['imgid']
            
            -- the first sent_idx
			local sent_idx = image_idx*5 + 1 -- note that image_idx is 0-indexed 
            
            for k = 1, 5 do 
                -- copy image 5 times 
			    image_batch[{{(t-1)*5+k}, {}}] = self.image_feats:select(2, image_idx+1)
			    sent_tensor[{{(t-1)*5+k}, {}}] = self.train_sents_tensor:select(1, sent_idx+k-1)
            end 
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
            local image_idx = self.images_info[split][remains]['imgid']
            
            -- the first sent_idx
			local sent_idx = image_idx*5 + 1 -- note that image_idx is 0-indexed 
            
            for k = 1, 5 do 
                -- copy image 5 times 
			    image_batch[{{(t-1)*5+k}, {}}] = self.image_feats:select(2, image_idx+1)
			    sent_tensor[{{(t-1)*5+k}, {}}] = self.train_sents_tensor:select(1, sent_idx+k-1)
            end 

			t = t + 1
			self.begin_index = self.begin_index + 1
		end  

	end

	--return inputs, image_batch
	return sent_tensor, image_batch

end
--]]




function Dataset:getNbatches(split)
	split = split or 'train'
	-- make sure we pad the batch with start example to make sure all the data are visited and the batch size 
	-- for every training iterations is of equal size
	return math.ceil(#self.images_info[split]/self.batch_size)
end 

