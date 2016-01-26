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
cmd:option('-path_vgg_features', '../data/coco/vgg_feats.mat', 'path to vgg feature files')
cmd:option('-coco_dict_split', '../data/coco/coco_dict_path_split.json', 'path to image info file')
cmd:option('-h5_file','../data/coco/coco_labels.h5', 'coco labels info')

cmd:option('-batch_size', 8, 'the size of a batch')
cmd:option('-split', 'train', 'train or test or val') 

cmd:text()
local opt = cmd:parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')

torch.manualSeed(123)

loader = Dataset(opt)

loader:loadDataset(opt)

print('testing')
print(string.format('vocabulary size is %d', loader:getVocabSize()))

idx2word = loader:getIdx2Word() 

print('get a batch:')
-- split = 'train'
image_batch, sent_tensor = loader:getNextBatch()

print(inputs)


