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


def main(params):
    imgs = json.load(open(params['input_json'], 'r'))

    imgs_info = imgs['images']

    # create the vocab
    vocab = __buildVocab(imgs_info, params)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input json file
    parser.add_argument('-input_json', required = True, help = 'input json file')

    # summary stuff
    parser.add_argument('-word_count_threshold', default=5, type=int, help = 'only words occurs greater than a threshod can it be added to the vocabulary')

    args = parser.parse_args()
    params = vars(args)

    print json.dumps(params, indent = 2)
    main(params)
