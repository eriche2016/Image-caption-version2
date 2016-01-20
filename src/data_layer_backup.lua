-- input data 
-- currently partially support cuda 
-- later can move all data to cuda to support cuda compiuting fullyfledged

local json = require 'json'
local matio = require 'matio'

local Dataset = torch.class('Dataset')

function Dataset:__init(opt)
	self.batch_size = opt.batch_size
	self.max_seq_length = 0
	self.begin_index = 1
	self.cycle = 0
    self.gpuid = opt.gpuid 
end

function Dataset:loadDataset(opt)
	local image_feats = matio.load(opt.path_vgg_features)
	self.image_feats = image_feats.feats
    ------- support cuda processing 
    if opt.gpuid >= 0 then
        self.image_feats = self.image_feats:cuda() 
    end 
    

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
end 

-- helper function 
function Dataset:__buildVocab(word_count_threshod)
	local word_count_threshod = word_count_threshod or 1 
	print(string.format('=> word threshod %d', word_count_threshod))
	-- do stastics on train dataset
	local word_counts = {}
	local vocab_map = {}
	local vocab_idx = 1
	local nsents = 0

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
				self.max_seq_length = #sent_tokens
			end 
		end
	end

	for word, v in pairs(word_counts) do 
		if v >= word_count_threshod then
			if vocab_map[word] == nil then
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
	return self.image_feats, self.images_info
end

function Dataset:getVocabSize()
	return #self.idx2word -- including end token(1)
end

function Dataset:getFeatSize()  -- will be 4096 here
	return self.image_feats:size(1)
end

function Dataset:getNextBatch(split) 
	local split = split or 'train'
	
	-- note that 1 indicates the end, we just
	local sent_tensor = torch.Tensor(self.batch_size, self.max_seq_length+1):fill(1) -- make sure that the last cloumn are all 1, indicate 'End' token  
	inputs = {}
	-- inputs[2] is the index list that will not contains start token, that will processed in the encoder layer  
	-- sample a sentence as label sequence for the image 

	local image_batch = torch.Tensor(self.batch_size, self.image_feats:size(1))  -- will be self.batch_size * 4096
    -- note that image_batch is located on cpu size memory, so need to move it to cuda 
    if self.gpuid >= 0 then 
        image_batch = image_batch:cuda()
    end 

	local sentid_r = torch.random(1, 5) -- 1, 2, 3, 4, 5
	local t = 1
	for i = self.begin_index, math.min(self.begin_index + self.batch_size - 1, #self.images_info[split]) do 
			local image_idx = self.images_info[split][i]['imgid']
			local sent_words = self.images_info[split][i]['sentences'][sentid_r]['tokens']

			-- change  sent_words to a list of idices 
			local index_list = {}
			for k = 1, #sent_words do 
				table.insert(index_list, self.word2idx[sent_words[k]])
			end  

			image_batch[{{t}, {}}] = self.image_feats:select(2, image_idx+1)

			sent_tensor[{{t}, {1, #index_list}}] = torch.LongTensor(index_list)
			t = t + 1
	end 

	self.begin_index = self.begin_index + self.batch_size
	--reset self.begin_index to 1 if it is larger than  #self.images_info['train']
	if self.begin_index > #self.images_info[split] then 
		self.begin_index = 1
		self.cycle = self.cycle + 1

		-- make sure every batch is of the same size regardless of the batch_size being set previously
		-- when t = self.batch_size + 1, then: 'for remains = 1, 0 do' will not execute , so it works right
		for remains = 1, self.batch_size -t + 1 do  
			local image_idx = self.images_info[split][remains]['imgid']
			local sent_words = self.images_info[split][remains]['sentences'][sentid_r]['tokens']

			-- change  sent_words to a list of idices 
			local index_list = {}
			for k = 1, #sent_words do 
				table.insert(index_list, self.word2idx[sent_words[k]])
			end  

			image_batch[{{t}, {}}] = self.image_feats:select(2, image_idx+1)

			sent_tensor[{{t}, {1, #index_list}}] = torch.LongTensor(index_list)
			t = t + 1
			self.begin_index = self.begin_index + 1
		end  

	end

	--return inputs, image_batch
	return sent_tensor, image_batch

end


function Dataset:getNbatches(split)
	split = split or 'train'
	-- make sure we pad the batch with start example to make sure all the data are visited and the batch size 
	-- for every training iterations is of equal size
	return math.ceil(#self.images_info[split]/self.batch_size)
end 


