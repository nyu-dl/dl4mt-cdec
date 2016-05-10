'''
Build a simple neural language model using GRU units
'''
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle
import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict
from mixer import *


def init_params(options):
    params = OrderedDict()

    print "source dictionary size: %d" % options['n_words_src']
    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word_src'])
    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

    # encoder
    params = get_layer('gru')[0](options, params,
                                 prefix='encoder',
                                 nin=options['dim_word_src'],
                                 dim=options['enc_dim'])
    params = get_layer('gru')[0](options, params,
                                 prefix='encoderr',
                                 nin=options['dim_word_src'],
                                 dim=options['enc_dim'])
    ctxdim = 2 * options['enc_dim']

    # init_state of decoder
    params = get_layer('ff')[0](options, params,
                                prefix='ff_init_state_char',
                                nin=ctxdim,
                                nout=options['dec_dim'])
    params = get_layer('ff')[0](options, params,
                                prefix='ff_init_state_word',
                                nin=ctxdim,
                                nout=options['dec_dim'])

    print "target dictionary size: %d" % options['n_words']
    # decoder
    params = get_layer('two_layer_gru_decoder')[0](options, params,
                                                   prefix='decoder',
                                                   nin=options['dim_word'],
                                                   dim_char=options['dec_dim'],
                                                   dim_word=options['dec_dim'],
                                                   dimctx=ctxdim)

    # readout
    params = get_layer('fff')[0](options, params, prefix='ff_logit_rnn',
                                 nin1=options['dec_dim'], nin2=options['dec_dim'],
                                 nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx',
                                nin=ctxdim,
                                nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'])

    return params


def build_model(tparams, options):
    opt_ret = OrderedDict()

    trng = RandomStreams(numpy.random.RandomState(numpy.random.randint(1024)).randint(numpy.iinfo(numpy.int32).max))
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')
    x.tag.test_value = numpy.zeros((5, 63), dtype='int64')
    x_mask.tag.test_value = numpy.ones((5, 63), dtype='float32')
    y.tag.test_value = numpy.zeros((7, 63), dtype='int64')
    y_mask.tag.test_value = numpy.ones((7, 63), dtype='float32')

    xr = x[::-1]
    xr_mask = x_mask[::-1]

    n_samples = x.shape[1]
    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]

    # word embedding for forward RNN (source)
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word_src']])

    # word embedding for backward RNN (source)
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word_src']])

    # pass through gru layer, recurrence here
    proj = get_layer('gru')[1](tparams, emb, options,
                               prefix='encoder', mask=x_mask)
    projr = get_layer('gru')[1](tparams, embr, options,
                                prefix='encoderr', mask=xr_mask)

    # context
    ctx = concatenate([proj, projr[::-1]], axis=proj.ndim-1)

    # context mean
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # initial decoder state
    init_state_char = get_layer('ff')[1](tparams, ctx_mean, options,
                                         prefix='ff_init_state_char', activ='tanh')
    init_state_word = get_layer('ff')[1](tparams, ctx_mean, options,
                                         prefix='ff_init_state_word', activ='tanh')

    # word embedding and shifting for targets
    yemb = tparams['Wemb_dec'][y.flatten()]
    yemb = yemb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    yemb_shited = tensor.zeros_like(yemb)
    yemb_shited = tensor.set_subtensor(yemb_shited[1:], yemb[:-1])
    yemb = yemb_shited

    char_h, word_h, ctxs, alphas = \
            get_layer('two_layer_gru_decoder')[1](tparams, yemb, options,
                                                  prefix='decoder',
                                                  mask=y_mask,
                                                  context=ctx,
                                                  context_mask=x_mask,
                                                  one_step=False,
                                                  init_state_char=init_state_char,
                                                  init_state_word=init_state_word)

    opt_ret['dec_alphas'] = alphas

    # compute word probabilities
    logit_rnn = get_layer('fff')[1](tparams, char_h, word_h, options,
                                    prefix='ff_logit_rnn', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, yemb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_rnn + logit_prev + logit_ctx)

    if options['use_dropout']:
        print 'Using dropout'
        logit = dropout_layer(logit, use_noise, trng)

    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost


def build_sampler(tparams, options, trng, use_noise):
    x = tensor.matrix('x', dtype='int64')
    xr = x[::-1]

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word_src']])
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word_src']])

    proj = get_layer('gru')[1](tparams, emb, options, prefix='encoder')
    projr = get_layer('gru')[1](tparams, embr, options, prefix='encoderr')

    ctx = concatenate([proj, projr[::-1]], axis=proj.ndim-1)
    ctx_mean = ctx.mean(0)

    init_state_char = get_layer('ff')[1](tparams, ctx_mean, options,
                                         prefix='ff_init_state_char', activ='tanh')
    init_state_word = get_layer('ff')[1](tparams, ctx_mean, options,
                                         prefix='ff_init_state_word', activ='tanh')

    print 'Building f_init...',
    outs = [init_state_char, init_state_word, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print 'Done'

    y = tensor.vector('y_sampler', dtype='int64')
    init_state_char = tensor.matrix('init_state_char', dtype='float32')
    init_state_word = tensor.matrix('init_state_word', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    yemb = tensor.switch(y[:, None] < 0,
                         tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                         tparams['Wemb_dec'][y])

    next_state_char, next_state_word, next_ctx, next_alpha = \
            get_layer('two_layer_gru_decoder')[1](tparams, yemb, options,
                                                  prefix='decoder',
                                                  context=ctx,
                                                  mask=None,
                                                  one_step=True,
                                                  init_state_char=init_state_char,
                                                  init_state_word=init_state_word)

    logit_rnn = get_layer('fff')[1](tparams,
                                    next_state_char,
                                    next_state_word,
                                    options,
                                    prefix='ff_logit_rnn',
                                    activ='linear')
    logit_prev = get_layer('ff')[1](tparams,
                                    yemb,
                                    options,
                                    prefix='ff_logit_prev',
                                    activ='linear')
    logit_ctx = get_layer('ff')[1](tparams,
                                   next_ctx,
                                   options,
                                   prefix='ff_logit_ctx',
                                   activ='linear')
    logit = tensor.tanh(logit_rnn + logit_prev + logit_ctx)

    if options['use_dropout']:
        print 'Sampling for dropoutted model'
        logit = dropout_layer(logit, use_noise, trng)

    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit',
                               activ='linear')
    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    print 'Building f_next...',
    inps = [y, ctx, init_state_char, init_state_word]
    outs = [next_probs, next_sample, next_state_char, next_state_word]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_init, f_next


def gen_sample(tparams, f_init, f_next, x, options, trng=None,
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
    ret = f_init(x)
    next_state_char, next_state_word, ctx0 = ret[0], ret[1], ret[2]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        inps = [next_w, ctx, next_state_char, next_state_word]
        ret = f_next(*inps)
        next_p, next_w, next_state_char, next_state_word = ret[0], ret[1], ret[2], ret[3]
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
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states_char = []
            new_hyp_states_word = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states_char.append(copy.copy(next_state_char[ti]))
                new_hyp_states_word.append(copy.copy(next_state_word[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states_char = []
            hyp_states_word = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states_char.append(new_hyp_states_char[idx])
                    hyp_states_word.append(new_hyp_states_word[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state_char = numpy.array(hyp_states_char)
            next_state_word = numpy.array(hyp_states_word)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score
