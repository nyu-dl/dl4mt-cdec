'''
Build a simple neural machine translation model using GRU units
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

from data_iterator import TextIterator
from collections import OrderedDict
from mixer import *


# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None, maxlen_trg=None,
                 n_words_src=30000, n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen_trg:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask, n_samples


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True, verboseFreq=None):
    probs = []

    n_done = 0
    cnt = 0

    for x, y in iterator:
        n_done += len(x)
        cnt += 1

        x, x_mask, y, y_mask, n_x = prepare_data(x, y,
                                                 n_words_src=options['n_words_src'],
                                                 n_words=options['n_words'])

        pprobs = f_log_probs(x, x_mask, y, y_mask)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            import ipdb
            ipdb.set_trace()

        if verbose:
            if numpy.mod(cnt, verboseFreq) == 0:
                print >>sys.stderr, '%d samples computed' % (cnt * n_done)

    return numpy.array(probs)


def train(
      dim_word=100,
      dim_word_src=200,
      enc_dim=1000,
      dec_dim=1000,  # the number of LSTM units
      patience=-1,  # early stopping patience
      max_epochs=5000,
      finish_after=-1,  # finish after this many updates
      decay_c=0.,  # L2 regularization penalty
      alpha_c=0.,  # alignment regularization
      clip_c=-1.,  # gradient clipping threshold
      lrate=0.01,  # learning rate
      n_words_src=100000,  # source vocabulary size
      n_words=100000,  # target vocabulary size
      maxlen=100,  # maximum length of the description
      maxlen_trg=None,  # maximum length of the description
      maxlen_sample=1000,
      optimizer='rmsprop',
      batch_size=16,
      valid_batch_size=16,
      sort_size=20,
      save_path=None,
      save_file_name='model',
      save_best_models=0,
      dispFreq=100,
      validFreq=100,
      saveFreq=1000,   # save the parameters after every saveFreq updates
      sampleFreq=-1,
      verboseFreq=10000,
      datasets=[
          'data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
          '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'],
      valid_datasets=['../data/dev/newstest2011.en.tok',
                      '../data/dev/newstest2011.fr.tok'],
      dictionaries=[
          '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
          '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],
      source_word_level=0,
      target_word_level=0,
      use_dropout=False,
      re_load=False,
      re_load_old_setting=False,
      uidx=None,
      eidx=None,
      cidx=None,
      layers=None,
      save_every_saveFreq=0,
      save_burn_in=20000,
      use_bpe=0,
      init_params=None,
      build_model=None,
      build_sampler=None,
      gen_sample=None,
      **kwargs
    ):

    if maxlen_trg is None:
        maxlen_trg = maxlen * 10
    # Model options
    model_options = locals().copy()
    del model_options['init_params']
    del model_options['build_model']
    del model_options['build_sampler']
    del model_options['gen_sample']

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        with open(dd, 'rb') as f:
            worddicts[ii] = cPickle.load(f)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    print 'Building model'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = '%s%s.npz' % (save_path, save_file_name)
    best_file_name = '%s%s.best.npz' % (save_path, save_file_name)
    opt_file_name = '%s%s%s.npz' % (save_path, save_file_name, '.grads')
    best_opt_file_name = '%s%s%s.best.npz' % (save_path, save_file_name, '.grads')
    model_name = '%s%s.pkl' % (save_path, save_file_name)
    params = init_params(model_options)
    cPickle.dump(model_options, open(model_name, 'wb'))
    history_errs = []

    # reload options
    if re_load and os.path.exists(file_name):
        print 'You are reloading your experiment.. do not panic dude..'
        if re_load_old_setting:
            with open(model_name, 'rb') as f:
                models_options = cPickle.load(f)
        params = load_params(file_name, params)
        # reload history
        model = numpy.load(file_name)
        history_errs = list(model['history_errs'])
        if uidx is None:
            uidx = model['uidx']
        if eidx is None:
            eidx = model['eidx']
        if cidx is None:
            cidx = model['cidx']
    else:
        if uidx is None:
            uidx = 0
        if eidx is None:
            eidx = 0
        if cidx is None:
            cidx = 0

    print 'Loading data'
    train = TextIterator(source=datasets[0],
                         target=datasets[1],
                         source_dict=dictionaries[0],
                         target_dict=dictionaries[1],
                         n_words_source=n_words_src,
                         n_words_target=n_words,
                         source_word_level=source_word_level,
                         target_word_level=target_word_level,
                         batch_size=batch_size,
                         sort_size=sort_size)
    valid = TextIterator(source=valid_datasets[0],
                         target=valid_datasets[1],
                         source_dict=dictionaries[0],
                         target_dict=dictionaries[1],
                         n_words_source=n_words_src,
                         n_words_target=n_words,
                         source_word_level=source_word_level,
                         target_word_level=target_word_level,
                         batch_size=valid_batch_size,
                         sort_size=sort_size)

    # create shared variables for parameters
    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask]

    print 'Building sampler...\n',
    f_init, f_next = build_sampler(tparams, model_options, trng, use_noise)
    #print 'Done'

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'
    if re_load:
        use_noise.set_value(0.)
        valid_errs = pred_probs(f_log_probs, prepare_data,
                                model_options, valid, verboseFreq=verboseFreq)
        valid_err = valid_errs.mean()

        if numpy.isnan(valid_err):
            import ipdb
            ipdb.set_trace()

        print 'Reload sanity check: Valid ', valid_err

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0) // x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    if clip_c > 0:
        grads, not_finite, clipped = gradient_clipping(grads, tparams, clip_c)
    else:
        not_finite = 0
        clipped = 0

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    if re_load and os.path.exists(file_name):
        if clip_c > 0:
            f_grad_shared, f_update, toptparams = eval(optimizer)(lr, tparams, grads, inps, cost=cost,
                                                                  not_finite=not_finite, clipped=clipped,
                                                                  file_name=opt_file_name)
        else:
            f_grad_shared, f_update, toptparams = eval(optimizer)(lr, tparams, grads, inps, cost=cost,
                                                                  file_name=opt_file_name)
    else:
        if clip_c > 0:
            f_grad_shared, f_update, toptparams = eval(optimizer)(lr, tparams, grads, inps, cost=cost,
                                                                  not_finite=not_finite, clipped=clipped)
        else:
            f_grad_shared, f_update, toptparams = eval(optimizer)(lr, tparams, grads, inps, cost=cost)
    print 'Done'

    print 'Optimization'
    best_p = None
    bad_counter = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    # Training loop
    ud_start = time.time()
    estop = False

    if re_load:
        print "Checkpointed minibatch number: %d" % cidx
        for cc in xrange(cidx):
            if numpy.mod(cc, 1000)==0:
                print "Jumping [%d / %d] examples" % (cc, cidx)
            train.next()

    for epoch in xrange(max_epochs):
        cidx = 0
        n_samples = 0
        NaN_grad_cnt = 0
        NaN_cost_cnt = 0
        clipped_cnt = 0

        for x, y in train:
            cidx += 1
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask, y, y_mask, n_x = prepare_data(x, y, maxlen=maxlen,
                                                     maxlen_trg=maxlen_trg,
                                                     n_words_src=n_words_src,
                                                     n_words=n_words)
            n_samples += n_x

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                uidx = max(uidx, 0)
                continue

            # compute cost, grads and copy grads to shared variables
            if clip_c > 0:
                cost, not_finite, clipped = f_grad_shared(x, x_mask, y, y_mask)
            else:
                cost = f_grad_shared(x, x_mask, y, y_mask)

            if clipped:
                clipped_cnt += 1

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                NaN_cost_cnt += 1

            if not_finite:
                NaN_grad_cnt += 1
                continue

            # do the update on parameters
            f_update(lrate)

            if numpy.isnan(cost) or numpy.isinf(cost):
                continue

            if float(NaN_grad_cnt) > max_epochs * 0.5 or float(NaN_cost_cnt) > max_epochs * 0.5:
                print 'Too many NaNs, abort training'
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                ud = time.time() - ud_start
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'NaN_in_grad', NaN_grad_cnt,\
                      'NaN_in_cost', NaN_cost_cnt, 'Gradient_clipped', clipped_cnt, 'UD ', ud
                ud_start = time.time()

            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0 and sampleFreq != -1:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5, x.shape[1])):
                    stochastic = True
                    use_noise.set_value(0.)
                    sample, score = gen_sample(tparams, f_init, f_next,
                                               x[:, jj][:, None],
                                               model_options, trng=trng, k=1,
                                               maxlen=maxlen_sample,
                                               stochastic=stochastic,
                                               argmax=False)
                    print
                    print 'Source ', jj, ': ',
                    if source_word_level:
                        for vv in x[:, jj]:
                            if vv == 0:
                                break
                            if vv in worddicts_r[0]:
                                if use_bpe:
                                    print (worddicts_r[0][vv]).replace('@@', ''),
                                else:
                                    print worddicts_r[0][vv],
                            else:
                                print 'UNK',
                        print
                    else:
                        source_ = []
                        for vv in x[:, jj]:
                            if vv == 0:
                                break
                            if vv in worddicts_r[0]:
                                source_.append(worddicts_r[0][vv])
                            else:
                                source_.append('UNK')
                        print "".join(source_)
                    print 'Truth ', jj, ' : ',
                    if target_word_level:
                        for vv in y[:, jj]:
                            if vv == 0:
                                break
                            if vv in worddicts_r[1]:
                                if use_bpe:
                                    print (worddicts_r[1][vv]).replace('@@', ''),
                                else:
                                    print worddicts_r[1][vv],
                            else:
                                print 'UNK',
                        print
                    else:
                        truth_ = []
                        for vv in y[:, jj]:
                            if vv == 0:
                                break
                            if vv in worddicts_r[1]:
                                truth_.append(worddicts_r[1][vv])
                            else:
                                truth_.append('UNK')
                        print "".join(truth_)
                    print 'Sample ', jj, ': ',
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    if target_word_level:
                        for vv in ss:
                            if vv == 0:
                                break
                            if vv in worddicts_r[1]:
                                if use_bpe:
                                    print (worddicts_r[1][vv]).replace('@@', ''),
                                else:
                                    print worddicts_r[1][vv],
                            else:
                                print 'UNK',
                        print
                    else:
                        sample_ = []
                        for vv in ss:
                            if vv == 0:
                                break
                            if vv in worddicts_r[1]:
                                sample_.append(worddicts_r[1][vv])
                            else:
                                sample_.append('UNK')
                        print "".join(sample_)
                    print

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid, verboseFreq=verboseFreq)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    best_optp = unzip(toptparams)
                    bad_counter = 0

                if saveFreq != validFreq and save_best_models:
                    numpy.savez(best_file_name, history_errs=history_errs, uidx=uidx, eidx=eidx,
                                cidx=cidx, **best_p)
                    numpy.savez(best_opt_file_name, **best_optp)

                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min() and patience != -1:
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    import ipdb
                    ipdb.set_trace()

                print 'Valid ', valid_err

            # save the best model so far
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                params = unzip(tparams)
                optparams = unzip(toptparams)
                numpy.savez(file_name, history_errs=history_errs, uidx=uidx, eidx=eidx,
                            cidx=cidx, **params)
                numpy.savez(opt_file_name, **optparams)

                if save_every_saveFreq and (uidx >= save_burn_in):
                    this_file_name = '%s%s.%d.npz' % (save_path, save_file_name, uidx)
                    this_opt_file_name = '%s%s%s.%d.npz' % (save_path, save_file_name, '.grads', uidx)
                    numpy.savez(this_file_name, history_errs=history_errs, uidx=uidx, eidx=eidx,
                                cidx=cidx, **params)
                    numpy.savez(this_opt_file_name, history_errs=history_errs, uidx=uidx, eidx=eidx,
                                cidx=cidx, **params)
                    if best_p is not None and saveFreq != validFreq:
                        this_best_file_name = '%s%s.%d.best.npz' % (save_path, save_file_name, uidx)
                        numpy.savez(this_best_file_name, history_errs=history_errs, uidx=uidx, eidx=eidx,
                                    cidx=cidx, **best_p)
                print 'Done...',
                print 'Saved to %s' % file_name

            # finish after this many updates
            if uidx >= finish_after and finish_after != -1:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples
        eidx += 1

        if estop:
            break

    use_noise.set_value(0.)
    valid_err = pred_probs(f_log_probs, prepare_data,
                           model_options, valid).mean()

    print 'Valid ', valid_err

    params = unzip(tparams)
    optparams = unzip(toptparams)
    file_name = '%s%s.%d.npz' % (save_path, save_file_name, uidx)
    opt_file_name = '%s%s%s.%d.npz' % (save_path, save_file_name, '.grads', uidx)
    numpy.savez(file_name, history_errs=history_errs, uidx=uidx, eidx=eidx, cidx=cidx, **params)
    numpy.savez(opt_file_name, **optparams)
    if best_p is not None and saveFreq != validFreq:
        best_file_name = '%s%s.%d.best.npz' % (save_path, save_file_name, uidx)
        best_opt_file_name = '%s%s%s.%d.best.npz' % (save_path, save_file_name, '.grads',uidx)
        numpy.savez(best_file_name, history_errs=history_errs, uidx=uidx, eidx=eidx, cidx=cidx, **best_p)
        numpy.savez(best_opt_file_name, **best_optp)

    return valid_err


if __name__ == '__main__':
    pass
