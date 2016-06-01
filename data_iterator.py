import nltk
import numpy
import os
import random

import cPickle
import gzip
import codecs

from tempfile import mkstemp


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self,
                 source, source_dict,
                 target=None, target_dict=None,
                 source_word_level=0,
                 target_word_level=0,
                 batch_size=128,
                 job_id=0,
                 sort_size=20,
                 n_words_source=-1,
                 n_words_target=-1,
                 shuffle_per_epoch=False):
        self.source_file = source
        self.target_file = target
        self.source = fopen(source, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = cPickle.load(f)
        if target is not None:
            self.target = fopen(target, 'r')
            if target_dict is not None:
                with open(target_dict, 'rb') as f:
                    self.target_dict = cPickle.load(f)
        else:
            self.target = None

        self.source_word_level = source_word_level
        self.target_word_level = target_word_level
        self.batch_size = batch_size

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target
        self.shuffle_per_epoch = shuffle_per_epoch

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * sort_size

        self.end_of_data = False
        self.job_id = job_id

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle_per_epoch:
            # close current files
            self.source.close()
            if self.target is None:
                self.shuffle([self.source_file])
                self.source = fopen(self.source_file + '.reshuf_%d' % self.job_id, 'r')
            else:
                self.target.close()
                # shuffle *original* source files,
                self.shuffle([self.source_file, self.target_file])
                # open newly 're-shuffled' file as input
                self.source = fopen(self.source_file + '.reshuf_%d' % self.job_id, 'r')
                self.target = fopen(self.target_file + '.reshuf_%d' % self.job_id, 'r')
        else:
            self.source.seek(0)
            if self.target is not None:
                self.target.seek(0)

    @staticmethod
    def shuffle(files):
        tf_os, tpath = mkstemp()
        tf = open(tpath, 'w')
        fds = [open(ff) for ff in files]
        for l in fds[0]:
            lines = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
            print >>tf, "|||".join(lines)
        [ff.close() for ff in fds]
        tf.close()
        tf = open(tpath, 'r')
        lines = tf.readlines()
        random.shuffle(lines)
        fds = [open(ff+'.reshuf','w') for ff in files]
        for l in lines:
            s = l.strip().split('|||')
            for ii, fd in enumerate(fds):
                print >>fd, s[ii]
        [ff.close() for ff in fds]
        os.remove(tpath)
        return

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        if self.target is not None:
            assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()

                if ss == "":
                    break

                if self.source_word_level:
                    ss = ss.strip().split()
                else:
                    ss = ss.strip()
                    ss = list(ss.decode('utf8'))

                self.source_buffer.append(ss)

                if self.target is not None:
                    tt = self.target.readline()

                    if tt == "":
                        break

                    if self.target_word_level:
                        tt = tt.strip().split()
                    else:
                        tt = tt.strip()
                        tt = list(tt.decode('utf8'))

                    self.target_buffer.append(tt)

            if self.target is not None:
                # sort by target buffer
                tlen = numpy.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()
                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]
                self.target_buffer = _tbuf
            else:
                slen = numpy.array([len(s) for s in self.source_buffer])
                sidx = slen.argsort()
                _sbuf = [self.source_buffer[i] for i in sidx]

            self.source_buffer = _sbuf

        if self.target is not None:
            if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration
        elif len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            # actual work here
            while True:
                # read from source file and map to word index
                try:
                    ss_ = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.source_dict[w] if w in self.source_dict else 1 for w in ss_]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]
                source.append(ss)
                if self.target is not None:
                    # read from target file and map to word index
                    tt_ = self.target_buffer.pop()
                    tt = [self.target_dict[w] if w in self.target_dict else 1 for w in tt_]
                    if self.n_words_target > 0:
                        tt = [w if w < self.n_words_target else 1 for w in tt]
                    target.append(tt)

                if len(source) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if self.target is not None:
            if len(source) <= 0 or len(target) <= 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration
            return source, target
        else:
            if len(source) <= 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration
            return source
