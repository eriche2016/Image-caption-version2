require 'torch'
require 'data_layer'
-- use this file to save the vocabulary 

-------------------------------------
-- input argument
-------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('test a data-layer')
cmd:text()
cmd:text('Options')

-- define input arguments
cmd:option('-path_vgg_features', '../data/flickr8k/vgg_feats.mat', 'path to vgg feature files')
cmd:option('-path_image_info', '../data/flickr8k/dataset.json', 'path to image info file')
cmd:option('-batch_size', 8, 'the size of a batch')
-- we do statistics on the vocabulary used in  flickr8k training caption dataset and store it for later use 
cmd:option('-idx2word_save', '../data/flickr8k/Idx2Word.t7', 'specify the file to store the vocab')
cmd:option('-word2idx_save', '../data/flickr8k/Word2Idx.t7', 'file to sore idx2word')

cmd:text()
local opt = cmd:parse(arg)
torch.manualSeed(123)

loader = Dataset(opt)

loader:loadDataset(opt)
print('testing')
print(string.format('vocabulary size is %d', loader:getVocabSize()))

idx2word = loader:getIdx2Word() 
word2idx = loader:getWord2Idx()

torch.save(opt.idx2word_save, idx2word)
torch.save(opt.word2idx_save, word2idx)

print('get a batch:')
-- split = 'train'
inputs, image_batch = loader:getNextBatch('train')

print(inputs)



