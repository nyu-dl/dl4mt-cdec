import cPickle as pkl
import fileinput
import numpy
import sys
import codecs

from collections import OrderedDict


short_list = 300

def main():
    for filename in sys.argv[1:]:
        print 'Processing', filename
        word_freqs = OrderedDict()

        with open(filename, 'r') as f:
            for line in f:
                words_in = line.strip()
                words_in = list(words_in.decode('utf8'))
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1

        words = word_freqs.keys()
        freqs = word_freqs.values()

        sorted_idx = numpy.argsort(freqs)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]

        worddict = OrderedDict()
        worddict['eos'] = 0
        worddict['UNK'] = 1

        if short_list is not None:
            for ii in xrange(min(short_list, len(sorted_words))):
                worddict[sorted_words[ii]] = ii + 2
        else:
            for ii, ww in enumerate(sorted_words):
                worddict[ww] = ii + 2

        with open('%s.%d.pkl' % (filename, short_list), 'wb') as f:
            pkl.dump(worddict, f)

        f.close()
        print 'Done'
        print len(worddict)

if __name__ == '__main__':
    main()
