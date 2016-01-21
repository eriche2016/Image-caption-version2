"""
  if preprocess the caption file such as coco data set in lua,  it will cause
  lujit out of memory error,  because lujit doesnot allow tables consumming too
  much memory(but loading big tensor is okay),  so we need to process it in
  python language and transform it to an array type.
  input json file has the form:
   {
      {'sentids':[770337, 771687, 772707, 776154,781998 ], 'imgid': 0,
      'split':'test', 'sentences': [{'tokens': ['a', 'red', ...], 'raws':
      'a red ...', 'imgid':0,  'sentid':770337}, {'tokens', ...}
      ], 'cocoid', 391895
      },
      {

      }
      ...
   }
"""
import os
import json
import argparse

import h5py
import numpy as np

# img_info will be original json file content
def __buildVocab(imgs_info, params):
    count_thr = params['word_count_threshold']

    #count up the number of words
    counts = {} # dict
    for img in imgs_info:
        txt_list = img['sentences']  # img['sentences'] is a list of 5 elements
                                     #merge list of list in one list(words list to describe the whole image)
        words_list = [word for tokens in txt_list for word in tokens['tokens']]
        for w in words_list:
            counts[w] = counts.get(w, 0) + 1

    cw = sorted([(count, w) for w, count in counts.iteritems()], reverse=True)

    print 'top words and their counts: '
    print '\n'.join(map(str, cw[:20]))

    # print some statistics
    total_words = sum(counts.itervalues())
    print 'total words: ', total_words

    bad_words = [w for w, n in counts.iteritems() if n <= count_thr]
    vocab = [w for w, n in counts.iteritems() if n > count_thr]
    #the occurances of UNK(treat all bad words as UNK)
    bad_count = sum(counts[w] for w in bad_words)

    print 'number of bad words: %d/%d = %.2f%%' %(len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
    print "number of words in the vocab is %d" %(len(vocab))
    print "number of UNKs: %d/%d = %.2f%%" %(bad_count, total_words, bad_count*100.0/total_words)

    #let us look at the distribution of the sentence lengths as well
    sent_lengths = {} # dict which stores element like (nw, freq)
    for img in imgs_info:
        txt_list = img['sentences']
        sents_len = [len(tokens['tokens']) for tokens in txt_list] #ex, [14, 12, 9, 29, 18]
        for nw in sents_len:
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1

    max_len = max(sent_lengths.keys())
    print 'max length in raw data: ', max_len

    # record the max (word) length of the captions

    # we can overide or specify max_len, now just overide
    params['max_len'] = max_len

    print 'sentence length distribution (count, number of words)'
    # ie, total number of sentences
    num_sents = sum(sent_lengths.values())

    for i in xrange(max_len+1):
        print '%2d: %10d  %f%%' %(i, sent_lengths.get(i, 0), sent_lengths.get(i, 0)*100.0/num_sents)

    # now insert UNK token to the vocab
    if bad_count > 0:
        print 'inserting UNK token to the vocab'
        vocab.append('UNK')

    # generate the final captions(contains UNK) for each image
    for img in imgs_info:
        # create a new field to store the final caption
        img['final_captions'] = []
        txt_list = img['sentences']
        for tokens in txt_list:
            final_caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in tokens['tokens']]
            img['final_captions'].append(final_caption)

    return vocab


def encode_captions(imgs_info, params, wtoi):
    """
    encode all captions into a large array, note that we also need to record the range of indices
    of the large array coresponding to each image
    """
    max_len = params['max_len']
    nimgs = len(imgs_info)
    ncaptions = sum(len(img['final_captions']) for img in imgs_info)
    print 'nimgs: %d  ncaptions: %d  max_len: %d' %(nimgs, ncaptions, max_len)

    label_arrays = []
    label_start_idx = np.zeros(nimgs, dtype='uint32')
    label_end_idx = np.zeros(nimgs, dtype='uint32')
    label_length = np.zeros(ncaptions, dtype='uint32')

    caption_counter = 0
    counter = 1
    for i, img in enumerate(imgs_info):
        n = len(img['final_captions'])

        assert n > 0, 'error, some images has no captions'

        Li = np.zeros((n, max_len), dtype='uint32')

        for j, s in enumerate(img['final_captions']):
            # actually, here the length will always less than max_len
            label_length[caption_counter] = min(max_len, len(s)) # record the length of the sequence
            caption_counter += 1
            for k, w in enumerate(s): # iterate all the words in the caption s
                if k < max_len:
                    Li[j, k] = wtoi[w]

        # note word indices are 1-indexed, captions are padded with zeros
        label_arrays.append(Li)
        label_start_idx[i] = counter
        label_end_idx[i] = counter + n - 1

        counter += n
    L = np.concatenate(label_arrays, axis=0) # put all labels together, along dimention 0

    assert L.shape[0] == ncaptions, 'length donot match, weird'

    assert np.all(label_length > 0), 'error, some captions donot have words?'

    print 'encoded captions to array of size: ', L.shape

    return L, label_start_idx, label_end_idx, label_length



def main(params):
    imgs = json.load(open(params['input_json'], 'r'))

    imgs_info = imgs['images']

    # create the vocab
    vocab = __buildVocab(imgs_info, params)
    # if loaded from lua, the key will be string
    itow = {i+1:w for i, w in enumerate(vocab)}
    wtoi = {w:i+1 for i, w in enumerate(vocab)}

    L, label_start_idx, label_end_idx, label_length =  encode_captions(imgs_info, params, wtoi)

    # create h5 file, just stores the image captions info
    # no need to store images, cause features already extracted, and stored in vgg_feats
    f = h5py.File(params['output_h5'], "w")

    f.create_dataset("labels", dtype='uint32', data=L)
    f.create_dataset('label_start_idx', dtype='uint32', data=label_start_idx)
    f.create_dataset('label_end_idx', dtype='uint32', data=label_end_idx)
    f.create_dataset('label_length', dtype='uint32', data=label_length)

    f.close()
    print 'wrote ', params['output_h5']

    # create output json file
    out = {}
    out['idx_to_word'] = itow

    out['imgs'] = []

    for i, img in enumerate(imgs_info):
        img_aux_info = {}
        # we also change restval to train dataset
        if img['split'] == 'restval':
            img['split'] = 'train'
        img_aux_info['split'] = img['split']
        if 'file_path' in img: img_aux_info['file_path'] = img['file_path']
        if 'id' in img: img_aux_info['id'] = img['id']
        out['imgs'].append(img_aux_info)

    # now the 123287 images are splited as: validation set: 5000, test set: 5000 train set: 113287
    train_s, val_s, test_s = 0, 0, 0
    for i, img in enumerate(out['imgs']):
        if img['split'] == 'train':
            train_s += 1
        elif img['split'] == 'val':
            val_s += 1
        else:
            test_s += 1
    print 'split the dataset as: '
    print 'train: %d, val: %d, test: %d'%(train_s, val_s, test_s)

    # dump dict out to file
    json.dump(out, open(params['output_json'], 'w'))
    print 'wrote ', params['output_json']
    vocab = __buildVocab(imgs_info, params)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input json file
    parser.add_argument('-input_json', required = True, help = 'input json file')
    parser.add_argument('-output_h5', required = True, help = 'the output hdf5 file')
    parser.add_argument('-output_json', required = True, help = 'the output json file which contains auxilary information of the file, ex, file path and idx_to_word')

   # summary stuff
    parser.add_argument('-word_count_threshold', default=5, type=int, help = 'only words occurs greater than a threshod can it be added to the vocabulary')
    parser.add_argument('-max_len', default = 20, type=int, help='the max length of the sentences apprears in the caption')

    # summary stuff
    parser.add_argument('-word_count_threshold', default=5, type=int, help = 'only words occurs greater than a threshod can it be added to the vocabulary')

    args = parser.parse_args()
    params = vars(args)

    print json.dumps(params, indent = 2)
    main(params)
