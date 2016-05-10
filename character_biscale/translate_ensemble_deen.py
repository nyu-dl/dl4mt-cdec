'''
Translates a source file using a translation model.
'''
import argparse

import numpy
import cPickle as pkl

from nmt import (build_sampler, init_params)
from mixer import *

from multiprocessing import Process, Queue


def gen_sample(tparams, f_inits, f_nexts, x, options, trng=None,
               k=1, maxlen=500, stochastic=True, argmax=False):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    rets = []
    next_state_chars = []
    next_state_words = []
    next_bound_chars = []
    next_bound_words = []
    ctx0s = []

    for f_init in f_inits:
        ret = f_init(x)
        next_state_chars.append(ret[0])
        next_state_words.append(ret[1])
        ctx0s.append(ret[2])
        next_bound_chars.append(numpy.zeros((1, options['dec_dim'])).astype('float32'))
        next_bound_words.append(numpy.zeros((1, options['dec_dim'])).astype('float32'))
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    num_models = len(f_inits)

    for ii in xrange(maxlen):

        temp_next_state_char = []
        temp_next_state_word = []
        temp_next_bound_char = []
        temp_next_bound_word = []
        temp_next_p = []

        for i in xrange(num_models):

            ctx = numpy.tile(ctx0s[i], [live_k, 1])
            inps = [next_w, ctx, next_state_chars[i], next_state_words[i], next_bound_chars[i], next_bound_words[i]]
            ret = f_nexts[i](*inps)
            next_p, _, next_state_char, next_state_word, next_bound_char, next_bound_word = ret[0], ret[1], ret[2], ret[3], ret[4], ret[5]
            temp_next_p.append(next_p)
            temp_next_state_char.append(next_state_char)
            temp_next_state_word.append(next_state_word)
            temp_next_bound_char.append(next_bound_char)
            temp_next_bound_word.append(next_bound_word)
        #next_p = numpy.log(numpy.array(temp_next_p)).sum(axis=0) / num_models
        next_p = numpy.log(numpy.array(temp_next_p).mean(axis=0))

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score += next_p[0, nw]
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - next_p
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states_chars = []
            new_hyp_states_words = []
            new_hyp_bounds_chars = []
            new_hyp_bounds_words = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])

            for i in xrange(num_models):
                new_hyp_states_char = []
                new_hyp_states_word = []
                new_hyp_bounds_char = []
                new_hyp_bounds_word = []

                for ti in trans_indices:
                    new_hyp_states_char.append(copy.copy(temp_next_state_char[i][ti]))
                    new_hyp_states_word.append(copy.copy(temp_next_state_word[i][ti]))
                    new_hyp_bounds_char.append(copy.copy(temp_next_bound_char[i][ti]))
                    new_hyp_bounds_word.append(copy.copy(temp_next_bound_word[i][ti]))

                new_hyp_states_chars.append(new_hyp_states_char)
                new_hyp_states_words.append(new_hyp_states_word)
                new_hyp_bounds_chars.append(new_hyp_bounds_char)
                new_hyp_bounds_words.append(new_hyp_bounds_word)

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])

            for i in xrange(num_models):
                hyp_states_char = []
                hyp_states_word = []
                hyp_bounds_char = []
                hyp_bounds_word = []

                for idx in xrange(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] != 0:
                        hyp_states_char.append(new_hyp_states_chars[i][idx])
                        hyp_states_word.append(new_hyp_states_words[i][idx])
                        hyp_bounds_char.append(new_hyp_bounds_chars[i][idx])
                        hyp_bounds_word.append(new_hyp_bounds_words[i][idx])

                next_state_chars[i] = numpy.array(hyp_states_char)
                next_state_words[i] = numpy.array(hyp_states_word)
                next_bound_chars[i] = numpy.array(hyp_bounds_char)
                next_bound_words[i] = numpy.array(hyp_bounds_word)

            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


def translate_model(queue, rqueue, pid, models, options, k, normalize):

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)

    # allocate model parameters
    params = []
    for i in xrange(len(models)):
        params.append(init_params(options))

    # load model parameters and set theano shared variables
    tparams = []
    for i in xrange(len(params)):
        params[i] = load_params(models[i], params[i])
        tparams.append(init_tparams(params[i]))

    # word index
    use_noise = theano.shared(numpy.float32(0.))
    f_inits = []
    f_nexts = []
    for i in xrange(len(tparams)):
        f_init, f_next = build_sampler(tparams[i], options, trng, use_noise)
        f_inits.append(f_init)
        f_nexts.append(f_next)

    def _translate(seq):
        use_noise.set_value(0.)
        # sample given an input sequence and obtain scores
        sample, score = gen_sample(tparams, f_inits, f_nexts,
                                   numpy.array(seq).reshape([len(seq), 1]),
                                   options, trng=trng, k=k, maxlen=500,
                                   stochastic=False, argmax=False)

        # normalize scores according to sequence lengths
        if normalize:
            lengths = numpy.array([len(s) for s in sample])
            score = score / lengths
        sidx = numpy.argmin(score)
        return sample[sidx]

    while True:
        req = queue.get()
        if req is None:
            break

        idx, x = req[0], req[1]
        print pid, '-', idx
        seq = _translate(x)

        rqueue.put((idx, seq))

    return


def main(models, dictionary, dictionary_target, source_file, saveto, k=5,
         normalize=False, n_process=5, encoder_chr_level=False,
         decoder_chr_level=False, utf8=False):

    # load model model_options
    pkl_file = models[0].split('.')[0] + '.pkl'
    with open(pkl_file, 'rb') as f:
        options = pkl.load(f)

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    # create input and output queues for processes
    queue = Queue()
    rqueue = Queue()
    processes = [None] * n_process
    for midx in xrange(n_process):
        processes[midx] = Process(
            target=translate_model,
            args=(queue, rqueue, midx, models, options, k, normalize))
        processes[midx].start()

    # utility function
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                if utf8:
                    ww.append(word_idict_trg[w].encode('utf-8'))
                else:
                    ww.append(word_idict_trg[w])
            if decoder_chr_level:
                capsw.append(''.join(ww))
            else:
                capsw.append(' '.join(ww))
        return capsw

    def _send_jobs(fname):
        with open(fname, 'r') as f:
            for idx, line in enumerate(f):
                if encoder_chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()
                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words_src'] else 1, x)
                x += [0]
                #print '=============================='
                #print line
                #print '------------------------------'
                #print ' '.join([word_idict[wx] for wx in x])
                #print '=============================='
                queue.put((idx, x))
        return idx+1

    def _finish_processes():
        for midx in xrange(n_process):
            queue.put(None)

    def _retrieve_jobs(n_samples):
        trans = [None] * n_samples
        for idx in xrange(n_samples):
            resp = rqueue.get()
            trans[resp[0]] = resp[1]
            if numpy.mod(idx, 10) == 0:
                print 'Sample ', (idx+1), '/', n_samples, ' Done'
        return trans

    print 'Translating ', source_file, '...'
    n_samples = _send_jobs(source_file)
    trans = _seqs2words(_retrieve_jobs(n_samples))
    _finish_processes()
    with open(saveto, 'w') as f:
        print >>f, '\n'.join(trans)
    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('-p', type=int, default=5)
    parser.add_argument('-n', action="store_true", default=False)
    parser.add_argument('-enc_c', action="store_true", default=False)
    parser.add_argument('-dec_c', action="store_true", default=False)
    parser.add_argument('-utf8', action="store_true", default=False)
    parser.add_argument('saveto', type=str)

    model_path = '/misc/kcgscratch1/ChoGroup/junyoung_exp/acl2016/wmt15/deen/bpe2char_seg_gru_decoder/0209/'
    model1 = model_path + 'bpe2char_seg_gru_decoder_adam_en1.430000.npz'
    model2 = model_path + 'bpe2char_seg_gru_decoder_adam_en2.420000.npz'
    model3 = model_path + 'bpe2char_seg_gru_decoder_adam_en3.445000.npz'
    model4 = model_path + 'bpe2char_seg_gru_decoder_adam.375000.npz'
    models = [model1, model2, model3, model4]
    dictionary = '/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/deen/train/all_de-en.en.tok.bpe.word.pkl'
    dictionary_target = '/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/deen/train/all_de-en.de.tok.300.pkl'
    source = '/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/deen/dev/newstest2013.en.tok.bpe'
    #source = '/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/deen/test/newstest2014-deen-src.en.tok.bpe'
    #source = '/misc/kcgscratch1/ChoGroup/junyoung_exp/wmt15/deen/test/newstest2015-deen-src.en.tok.bpe'

    args = parser.parse_args()

    main(models, dictionary, dictionary_target, source,
         args.saveto, k=args.k, normalize=args.n, n_process=args.p,
         encoder_chr_level=args.enc_c,
         decoder_chr_level=args.dec_c,
         utf8=args.utf8)
