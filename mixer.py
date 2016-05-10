'''
Mixer containing essential functions or building blocks
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

profile = False


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'fff': ('param_init_ffflayer', 'ffflayer'),
          'gru_decoder': ('param_init_gru_decoder', 'gru_decoder'),
          'gru_cond_decoder': ('param_init_gru_cond_decoder',
                               'gru_cond_decoder'),
          'two_layer_gru_decoder': ('param_init_two_layer_gru_decoder',
                                    'two_layer_gru_decoder'),
          'two_layer_gru_decoder_both': ('param_init_two_layer_gru_decoder_both',
                                         'two_layer_gru_decoder_both'),
          'biscale_decoder': ('param_init_biscale_decoder',
                              'biscale_decoder'),
          'biscale_decoder_both': ('param_init_biscale_decoder_both',
                                   'biscale_decoder_both'),
          'biscale_decoder_attc': ('param_init_biscale_decoder_attc',
                                   'biscale_decoder_attc'),
          'gru': ('param_init_gru', 'gru_layer')
          }


# utility function to slice a tensor
def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n*dim:(n+1)*dim]
    return _x[:, n*dim:(n+1)*dim]


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# some utilities
def ortho_weight(ndim, scale=0.01):
    W = scale * numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_vector(nin, scale=0.01):
    V = scale * numpy.random.randn(nin)
    return V.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape, dtype=tensor_list[0].dtype)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True, scale=0.01):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=scale, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


# feedforward layer short-cut: affine transformation + point-wise nonlinearity
def param_init_ffflayer(options, params, prefix='fff', nin1=None, nin2=None, nout=None,
                        ortho=True, scale1=0.01, scale2=0.01):
    if nin1 is None:
        nin1 = options['dim_proj']
    if nin2 is None:
        nin2 = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin1, nout, scale=scale1, ortho=ortho)
    params[_p(prefix, 'U')] = norm_weight(nin2, nout, scale=scale2, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def ffflayer(tparams, state_below1, state_below2, options, prefix='rconv',
             activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below1, tparams[_p(prefix, 'W')]) +
        tensor.dot(state_below2, tparams[_p(prefix, 'U')]) +
        tparams[_p(prefix, 'b')])


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):

    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['rnn_dim']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, options, prefix='gru',
              mask=None, one_step=False, init_state=None, **kwargs):

    if one_step:
        assert init_state, 'previous state must be provided'

    n_steps = state_below.shape[0]
    if state_below.ndim in [2, 3]:
        n_samples = state_below.shape[1]
    elif state_below.ndim == 1:
        if not  one_step:
            raise ValueError('if state_below.ndim is 1, one_step shoud also be 1')
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if state_below.dtype == 'int64':
        state_below_ = tparams[_p(prefix, 'W')][state_below.flatten()]
        state_belowx = tparams[_p(prefix, 'Wx')][state_below.flatten()]
        if state_below.ndim == 2:
            state_below_ = state_below_.reshape((n_steps, n_samples, -1))
            state_belowx = state_belowx.reshape((n_steps, n_samples, -1))
        state_below_ += tparams[_p(prefix, 'b')]
        state_belowx += tparams[_p(prefix, 'bx')]
    else:
        # projected x to hidden state proposal
        state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
            tparams[_p(prefix, 'b')]
        # projected x to gates
        state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
            tparams[_p(prefix, 'bx')]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # step function to be used by scan
    def _step(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_
        preact = tensor.nnet.sigmoid(preact)

        # reset and update gates
        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx *= r
        preactx += xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    if one_step:
        rval = _step(*(seqs+[init_state]+shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state],
                                    non_sequences=shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=n_steps,
                                    profile=profile,
                                    strict=True)

    return rval


# Conditional GRU layer without Attention
def param_init_gru_decoder(options, params, prefix='gru_decoder', nin=None,
                           dim=None, dimctx=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']

    params = param_init_gru(options, params, prefix, nin=nin, dim=dim)

    # context to GRU gates
    Wc = norm_weight(dimctx, dim*2)
    params[_p(prefix, 'Wc')] = Wc

    # context to hidden proposal
    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    return params


def gru_decoder(tparams, state_below, options, prefix='gru_decoder',
                mask=None, context=None, one_step=False,
                init_state=None, **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    n_steps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    assert context.ndim == 2, 'Context must be 2-d: #sample x dim'
    # projected context to GRU gates
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc')])
    # projected context to hidden state proposal
    pctxx_ = tensor.dot(context, tparams[_p(prefix, 'Wcx')])

    # projected x to hidden state proposal
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # projected x to gates
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences | outputs-info| non-seqs
    def _step(m_, x_, xx_, h_,          pctx_, pctxx_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_
        preact += pctx_
        preact = tensor.nnet.sigmoid(preact)

        # reset and update gates
        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx *= r
        preactx += xx_
        preactx += pctxx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h
    # prepare scan arguments

    seqs = [mask, state_below_, state_belowx]

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    if one_step:
        rval = _step(*(seqs+[init_state, pctx_, pctxx_]+shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state],
                                    non_sequences=[pctx_, pctxx_]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=n_steps,
                                    profile=profile,
                                    strict=True)
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond_decoder(options, params, prefix='gru_cond_decoder',
                                nin=None, dim=None, dimctx=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']

    params = param_init_gru(options, params, prefix, nin=nin, dim=dim)

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: prev -> hidden
    Wi_att = norm_weight(nin, dimctx)
    params[_p(prefix, 'Wi_att')] = Wi_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim, dimctx)
    params[_p(prefix, 'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    return params


def gru_cond_decoder(tparams, state_below, options, prefix='gru_cond_decoder',
                     mask=None, context=None, one_step=False, init_state=None,
                     context_mask=None, **kwargs):

    assert context, 'Context must be provided'
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:  # sampling or beamsearch
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    pctx_ = tensor.dot(context, tparams[_p(prefix, 'Wc_att')]) + \
        tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x into hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]
    # projected x into gru gates
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # projected x into attention module
    state_belowc = tensor.dot(state_below, tparams[_p(prefix, 'Wi_att')])

    # step function to be used by scan
    # arguments    | sequences      |  outputs-info   | non-seqs ...
    def _step_slice(m_, x_, xx_, xc_, h_, ctx_, alpha_, pctx_, cc_,
                    U, Wc, Wd_att, U_att, c_tt, Ux, Wcx):

        # attention
        # project previous hidden state
        pstate_ = tensor.dot(h_, Wd_att)

        # add projected context
        pctx__ = pctx_ + pstate_[None, :, :]

        # add projected previous output
        pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)

        # compute alignment weights
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0))
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)

        # conpute the weighted averages - current context to gru
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)

        # conditional gru layer computations
        preact = tensor.dot(h_, U)
        preact += x_
        preact += tensor.dot(ctx_, Wc)
        preact = tensor.nnet.sigmoid(preact)

        # reset and update gates
        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = tensor.dot(h_, Ux)
        preactx *= r
        preactx += xx_
        preactx += tensor.dot(ctx_, Wcx)

        # hidden state proposal, leaky integrate and obtain next hidden state
        h = tensor.tanh(preactx)
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, ctx_, alpha.T

    seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'Wd_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')]]

    if one_step:
        rval = _step(*(
            seqs+[init_state, None, None, pctx_, context]+shared_vars))
    else:
        rval, updates = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=[init_state,
                          tensor.alloc(0., n_samples, context.shape[2]),
                          tensor.alloc(0., n_samples, context.shape[0])],
            non_sequences=[pctx_,
                           context]+shared_vars,
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            profile=profile,
            strict=True)
    return rval


def param_init_two_layer_gru_decoder(options, params,
                                     prefix='two_layer_gru_decoder',
                                     nin=None,
                                     dim_char=None,
                                     dim_word=None,
                                     dimctx=None):
    if nin is None:
        nin = options['n_words']
    if dim_char is None:
        dim_char = options['dec_dim']
    if dim_word is None:
        dim_word = options['dec_dim']
    if dimctx is None:
        dimctx = options['enc_dim'] * 2

    # embedding to gates transformation weights, biases
    W_xc = numpy.concatenate([norm_weight(nin, dim_char),
                           norm_weight(nin, dim_char)], axis=1)
    params[_p(prefix, 'W_xc')] = W_xc
    params[_p(prefix, 'b_c')] = numpy.zeros((2 * dim_char,)).astype('float32')

    # recurrent transformation weights for gates
    U_cc = numpy.concatenate([ortho_weight(dim_char),
                           ortho_weight(dim_char)], axis=1)
    params[_p(prefix, 'U_cc')] = U_cc

    # embedding to hidden state proposal weights, biases
    Wx_xc = norm_weight(nin, dim_char)
    params[_p(prefix, 'Wx_xc')] = Wx_xc
    params[_p(prefix, 'bx_c')] = numpy.zeros((dim_char,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux_cc = ortho_weight(dim_char)
    params[_p(prefix, 'Ux_cc')] = Ux_cc

    # embedding to gates transformation weights, biases
    W_cw = numpy.concatenate([norm_weight(dim_char, dim_word),
                              norm_weight(dim_char, dim_word)], axis=1)
    params[_p(prefix, 'W_cw')] = W_cw
    params[_p(prefix, 'b_w')] = numpy.zeros((2 * dim_word,)).astype('float32')

    # recurrent transformation weights for gates
    U_ww = numpy.concatenate([ortho_weight(dim_word),
                              ortho_weight(dim_word)], axis=1)
    params[_p(prefix, 'U_ww')] = U_ww

    # embedding to hidden state proposal weights, biases
    Wx_cw = norm_weight(dim_char, dim_word)
    params[_p(prefix, 'Wx_cw')] = Wx_cw
    params[_p(prefix, 'bx_w')] = numpy.zeros((dim_word,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux_ww = ortho_weight(dim_word)
    params[_p(prefix, 'Ux_ww')] = Ux_ww

    # context to GRU gates: char-level
    W_ctxc = numpy.concatenate([norm_weight(dimctx, dim_char),
                                norm_weight(dimctx, dim_char)], axis=1)
    params[_p(prefix, 'W_ctxc')] = W_ctxc

    # context to hidden proposal: char-level
    Wx_ctxc = norm_weight(dimctx, dim_char)
    params[_p(prefix, 'Wx_ctxc')] = Wx_ctxc

    # context to GRU gates: word-level
    W_ctxw = numpy.concatenate([norm_weight(dimctx, dim_word),
                                norm_weight(dimctx, dim_word)], axis=1)
    params[_p(prefix, 'W_ctxw')] = W_ctxw

    # context to hidden proposal: word-level
    Wx_ctxw = norm_weight(dimctx, dim_word)
    params[_p(prefix, 'Wx_ctxw')] = Wx_ctxw

    # attention: prev -> hidden
    Winp_att = norm_weight(nin, dimctx)
    params[_p(prefix, 'Winp_att')] = Winp_att

    # attention: context -> hidden
    Wctx_att = norm_weight(dimctx)
    params[_p(prefix, 'Wctx_att')] = Wctx_att

    # attention: decoder -> hidden
    Wdec_att = norm_weight(dim_word, dimctx)
    params[_p(prefix, 'Wdec_att')] = Wdec_att

    # attention: hidden bias
    params[_p(prefix, 'b_att')] = numpy.zeros((dimctx,)).astype('float32')

    # attention
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_att')] = c_att

    return params


def two_layer_gru_decoder(tparams, state_below, options,
                          prefix='two_layer_gru_decoder',
                          mask=None, one_step=False,
                          context=None, context_mask=None,
                          init_state_char=None, init_state_word=None,
                          **kwargs):

    assert context, 'Context must be provided'
    assert context.ndim == 3, \
        'Context must be 3-D: #annotation x #sample x #dim'

    if one_step:
        assert init_state_char, 'previous state must be provided'
        assert init_state_word, 'previous state must be provided'

    n_steps = state_below.shape[0]
    if state_below.ndim in [2, 3]:
        n_samples = state_below.shape[1]
    elif state_below.ndim == 1:
        if not  one_step:
            raise ValueError('if state_below.ndim is 1, one_step shoud also be 1')
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim_char = tparams[_p(prefix, 'Ux_cc')].shape[1]
    dim_word = tparams[_p(prefix, 'Ux_ww')].shape[1]

    if state_below.dtype == 'int64':
        state_below_emb = tparams[_p(prefix, 'W_xc')][state_below.flatten()] + tparams[_p(prefix, 'b_c')]
        state_belowx_emb = tparams[_p(prefix, 'Wx_xc')][state_below.flatten()] + tparams[_p(prefix, 'bx_c')]
        state_belowctx_emb = tparams[_p(prefix, 'Winp_att')][state_below.flatten()]
        if state_below.ndim == 2:
            state_below_emb = state_below_emb.reshape((n_steps, n_samples, -1))
            state_belowx_emb = state_belowx_emb.reshape((n_steps, n_samples, -1))
            state_belowctx_emb = state_belowctx_emb.reshape((n_steps, n_samples, -1))
    else:
        state_below_emb = tensor.dot(state_below, tparams[_p(prefix, 'W_xc')]) + tparams[_p(prefix, 'b_c')]
        state_belowx_emb = tensor.dot(state_below, tparams[_p(prefix, 'Wx_xc')]) + tparams[_p(prefix, 'bx_c')]
        state_belowctx_emb = tensor.dot(state_below, tparams[_p(prefix, 'Winp_att')])

    # initial/previous state
    if init_state_char is None:
        init_state_char = tensor.alloc(0., n_samples, dim_char)
    if init_state_word is None:
        init_state_word = tensor.alloc(0., n_samples, dim_word)

    # projected context
    proj_ctx = tensor.dot(context, tparams[_p(prefix, 'Wctx_att')]) + tparams[_p(prefix, 'b_att')]

    # step function to be used by scan
    def _step(m_t,
              state_below_emb_t,
              state_belowx_emb_t,
              state_belowctx_emb_t,
              h_c_tm1, h_w_tm1,
              ctx_t,
              alpha_t,
              proj_ctx_all,
              context,
              U_cc, Ux_cc,
              W_cw, Wx_cw, U_ww, Ux_ww, b_w, bx_w,
              W_ctxc, Wx_ctxc, W_ctxw, Wx_ctxw,
              Wdec_att,
              U_att, c_att):
        # ~~ attention ~~ #
        # project previous hidden states
        proj_state = tensor.dot(h_w_tm1, Wdec_att)

        # add projected context
        proj_ctx = proj_ctx_all + proj_state[None, :, :] + state_belowctx_emb_t
        proj_h = tensor.tanh(proj_ctx)

        # compute alignment weights
        alpha = tensor.dot(proj_h, U_att) + c_att
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0))
        #alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)

        # compute the weighted averages - current context to GRU
        ctx_t = (context * alpha[:, :, None]).sum(0)

        # compute char-level
        preact_c = tensor.dot(h_c_tm1, U_cc) + state_below_emb_t + tensor.dot(ctx_t, W_ctxc )
        preact_c = tensor.nnet.sigmoid(preact_c)

        # update gates
        r_c = _slice(preact_c, 0, dim_char)
        u_c = _slice(preact_c, 1, dim_char)

        # compute the hidden state proposal: char-level
        preactx_c = tensor.dot(h_c_tm1, Ux_cc) * r_c + state_belowx_emb_t + tensor.dot(ctx_t, Wx_ctxc)

        # hidden state proposal
        h_c = tensor.tanh(preactx_c)

        # leaky integrate and obtain next hidden state
        h_c_t = u_c * h_c_tm1 + (1. - u_c) * h_c
        h_c_t = m_t[:, None] * h_c_t + (1. - m_t)[:, None] * h_c_tm1

        # compute char-level
        preact_w = tensor.dot(h_w_tm1, U_ww) + tensor.dot(h_c_t, W_cw) + tensor.dot(ctx_t, W_ctxw) + b_w
        preact_w = tensor.nnet.sigmoid(preact_w)

        # update gates
        r_w = _slice(preact_w, 0, dim_char)
        u_w = _slice(preact_w, 1, dim_char)

        # compute the hidden state proposal: char-level
        preactx_w = tensor.dot(h_w_tm1, Ux_ww) * r_w + tensor.dot(h_c_t, Wx_cw) + tensor.dot(ctx_t, Wx_ctxw) + bx_w

        # hidden state proposal
        h_w = tensor.tanh(preactx_w)

        # leaky integrate and obtain next hidden state
        h_w_t = u_w * h_w_tm1 + (1. - u_w) * h_w
        h_w_t = m_t[:, None] * h_w_t + (1. - m_t)[:, None] * h_w_tm1

        return h_c_t, h_w_t, ctx_t, alpha.T

    # prepare scan arguments
    seqs = [mask, state_below_emb, state_belowx_emb, state_belowctx_emb]

    shared_vars = [
            tparams[_p(prefix, 'U_cc')],
            tparams[_p(prefix, 'Ux_cc')],
            tparams[_p(prefix, 'W_cw')],
            tparams[_p(prefix, 'Wx_cw')],
            tparams[_p(prefix, 'U_ww')],
            tparams[_p(prefix, 'Ux_ww')],
            tparams[_p(prefix, 'b_w')],
            tparams[_p(prefix, 'bx_w')],
            tparams[_p(prefix, 'W_ctxc')],
            tparams[_p(prefix, 'Wx_ctxc')],
            tparams[_p(prefix, 'W_ctxw')],
            tparams[_p(prefix, 'Wx_ctxw')],
            tparams[_p(prefix, 'Wdec_att')],
            tparams[_p(prefix, 'U_att')],
            tparams[_p(prefix, 'c_att')],
        ]

    if one_step:
        rval = _step(*(seqs+[init_state_char, init_state_word,
                             None, None,
                             proj_ctx, context]+shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[
                                        init_state_char,
                                        init_state_word,
                                        tensor.alloc(0., n_samples, context.shape[2]),
                                        tensor.alloc(0., n_samples, context.shape[0])
                                    ],
                                    non_sequences=[proj_ctx, context]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=n_steps,
                                    profile=profile,
                                    strict=True)
    return rval



def param_init_two_layer_gru_decoder_both(options, params,
                                          prefix='two_layer_gru_decoder_both',
                                          nin=None,
                                          dim_char=None,
                                          dim_word=None,
                                          dimctx=None):
    if nin is None:
        nin = options['n_words']
    if dim_char is None:
        dim_char = options['dec_dim']
    if dim_word is None:
        dim_word = options['dec_dim']
    if dimctx is None:
        dimctx = options['enc_dim'] * 2

    # embedding to gates transformation weights, biases
    W_xc = numpy.concatenate([norm_weight(nin, dim_char),
                           norm_weight(nin, dim_char)], axis=1)
    params[_p(prefix, 'W_xc')] = W_xc
    params[_p(prefix, 'b_c')] = numpy.zeros((2 * dim_char,)).astype('float32')

    # recurrent transformation weights for gates
    U_cc = numpy.concatenate([ortho_weight(dim_char),
                           ortho_weight(dim_char)], axis=1)
    params[_p(prefix, 'U_cc')] = U_cc

    # embedding to hidden state proposal weights, biases
    Wx_xc = norm_weight(nin, dim_char)
    params[_p(prefix, 'Wx_xc')] = Wx_xc
    params[_p(prefix, 'bx_c')] = numpy.zeros((dim_char,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux_cc = ortho_weight(dim_char)
    params[_p(prefix, 'Ux_cc')] = Ux_cc

    # embedding to gates transformation weights, biases
    W_cw = numpy.concatenate([norm_weight(dim_char, dim_word),
                              norm_weight(dim_char, dim_word)], axis=1)
    params[_p(prefix, 'W_cw')] = W_cw
    params[_p(prefix, 'b_w')] = numpy.zeros((2 * dim_word,)).astype('float32')

    # recurrent transformation weights for gates
    U_ww = numpy.concatenate([ortho_weight(dim_word),
                              ortho_weight(dim_word)], axis=1)
    params[_p(prefix, 'U_ww')] = U_ww

    # embedding to hidden state proposal weights, biases
    Wx_cw = norm_weight(dim_char, dim_word)
    params[_p(prefix, 'Wx_cw')] = Wx_cw
    params[_p(prefix, 'bx_w')] = numpy.zeros((dim_word,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux_ww = ortho_weight(dim_word)
    params[_p(prefix, 'Ux_ww')] = Ux_ww

    # context to GRU gates: char-level
    W_ctxc = numpy.concatenate([norm_weight(dimctx, dim_char),
                                norm_weight(dimctx, dim_char)], axis=1)
    params[_p(prefix, 'W_ctxc')] = W_ctxc

    # context to hidden proposal: char-level
    Wx_ctxc = norm_weight(dimctx, dim_char)
    params[_p(prefix, 'Wx_ctxc')] = Wx_ctxc

    # context to GRU gates: word-level
    W_ctxw = numpy.concatenate([norm_weight(dimctx, dim_word),
                                norm_weight(dimctx, dim_word)], axis=1)
    params[_p(prefix, 'W_ctxw')] = W_ctxw

    # context to hidden proposal: word-level
    Wx_ctxw = norm_weight(dimctx, dim_word)
    params[_p(prefix, 'Wx_ctxw')] = Wx_ctxw

    # attention: prev -> hidden
    Winp_att = norm_weight(nin, dimctx)
    params[_p(prefix, 'Winp_att')] = Winp_att

    # attention: context -> hidden
    Wctx_att = norm_weight(dimctx)
    params[_p(prefix, 'Wctx_att')] = Wctx_att

    # attention: decoder -> hidden
    Wdecc_att = norm_weight(dim_char, dimctx)
    params[_p(prefix, 'Wdecc_att')] = Wdecc_att
    Wdecw_att = norm_weight(dim_word, dimctx)
    params[_p(prefix, 'Wdecw_att')] = Wdecw_att

    # attention: hidden bias
    params[_p(prefix, 'b_att')] = numpy.zeros((dimctx,)).astype('float32')

    # attention
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_att')] = c_att

    return params


def two_layer_gru_decoder_both(tparams, state_below, options,
                               prefix='two_layer_gru_decoder_both',
                               mask=None, one_step=False,
                               context=None, context_mask=None,
                               init_state_char=None, init_state_word=None,
                               **kwargs):

    assert context, 'Context must be provided'
    assert context.ndim == 3, \
        'Context must be 3-D: #annotation x #sample x #dim'

    if one_step:
        assert init_state_char, 'previous state must be provided'
        assert init_state_word, 'previous state must be provided'

    n_steps = state_below.shape[0]
    if state_below.ndim in [2, 3]:
        n_samples = state_below.shape[1]
    elif state_below.ndim == 1:
        if not  one_step:
            raise ValueError('if state_below.ndim is 1, one_step shoud also be 1')
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim_char = tparams[_p(prefix, 'Ux_cc')].shape[1]
    dim_word = tparams[_p(prefix, 'Ux_ww')].shape[1]

    if state_below.dtype == 'int64':
        state_below_emb = tparams[_p(prefix, 'W_xc')][state_below.flatten()] + tparams[_p(prefix, 'b_c')]
        state_belowx_emb = tparams[_p(prefix, 'Wx_xc')][state_below.flatten()] + tparams[_p(prefix, 'bx_c')]
        state_belowctx_emb = tparams[_p(prefix, 'Winp_att')][state_below.flatten()]
        if state_below.ndim == 2:
            state_below_emb = state_below_emb.reshape((n_steps, n_samples, -1))
            state_belowx_emb = state_belowx_emb.reshape((n_steps, n_samples, -1))
            state_belowctx_emb = state_belowctx_emb.reshape((n_steps, n_samples, -1))
    else:
        state_below_emb = tensor.dot(state_below, tparams[_p(prefix, 'W_xc')]) + tparams[_p(prefix, 'b_c')]
        state_belowx_emb = tensor.dot(state_below, tparams[_p(prefix, 'Wx_xc')]) + tparams[_p(prefix, 'bx_c')]
        state_belowctx_emb = tensor.dot(state_below, tparams[_p(prefix, 'Winp_att')])

    # initial/previous state
    if init_state_char is None:
        init_state_char = tensor.alloc(0., n_samples, dim_char)
    if init_state_word is None:
        init_state_word = tensor.alloc(0., n_samples, dim_word)

    # projected context
    proj_ctx = tensor.dot(context, tparams[_p(prefix, 'Wctx_att')]) + tparams[_p(prefix, 'b_att')]

    # step function to be used by scan
    def _step(m_t,
              state_below_emb_t,
              state_belowx_emb_t,
              state_belowctx_emb_t,
              h_c_tm1, h_w_tm1,
              ctx_t,
              alpha_t,
              proj_ctx_all,
              context,
              U_cc, Ux_cc,
              W_cw, Wx_cw, U_ww, Ux_ww, b_w, bx_w,
              W_ctxc, Wx_ctxc, W_ctxw, Wx_ctxw,
              Wdecc_att, Wdecw_att,
              U_att, c_att):
        # ~~ attention ~~ #
        # project previous hidden states
        proj_state = tensor.dot(h_w_tm1, Wdecw_att) + tensor.dot(h_c_tm1, Wdecc_att)

        # add projected context
        proj_ctx = proj_ctx_all + proj_state[None, :, :] + state_belowctx_emb_t
        proj_h = tensor.tanh(proj_ctx)

        # compute alignment weights
        alpha = tensor.dot(proj_h, U_att) + c_att
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0))
        #alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)

        # compute the weighted averages - current context to GRU
        ctx_t = (context * alpha[:, :, None]).sum(0)

        # compute char-level
        preact_c = tensor.dot(h_c_tm1, U_cc) + state_below_emb_t + tensor.dot(ctx_t, W_ctxc)
        preact_c = tensor.nnet.sigmoid(preact_c)

        # update gates
        r_c = _slice(preact_c, 0, dim_char)
        u_c = _slice(preact_c, 1, dim_char)

        # compute the hidden state proposal: char-level
        preactx_c = tensor.dot(h_c_tm1, Ux_cc) * r_c + state_belowx_emb_t + tensor.dot(ctx_t, Wx_ctxc)

        # hidden state proposal
        h_c = tensor.tanh(preactx_c)

        # leaky integrate and obtain next hidden state
        h_c_t = u_c * h_c_tm1 + (1. - u_c) * h_c
        h_c_t = m_t[:, None] * h_c_t + (1. - m_t)[:, None] * h_c_tm1

        # compute char-level
        preact_w = tensor.dot(h_w_tm1, U_ww) + tensor.dot(h_c_t, W_cw) + tensor.dot(ctx_t, W_ctxw) + b_w
        preact_w = tensor.nnet.sigmoid(preact_w)

        # update gates
        r_w = _slice(preact_w, 0, dim_char)
        u_w = _slice(preact_w, 1, dim_char)

        # compute the hidden state proposal: char-level
        preactx_w = tensor.dot(h_w_tm1, Ux_ww) * r_w + tensor.dot(h_c_t, Wx_cw) + tensor.dot(ctx_t, Wx_ctxw) + bx_w

        # hidden state proposal
        h_w = tensor.tanh(preactx_w)

        # leaky integrate and obtain next hidden state
        h_w_t = u_w * h_w_tm1 + (1. - u_w) * h_w
        h_w_t = m_t[:, None] * h_w_t + (1. - m_t)[:, None] * h_w_tm1

        return h_c_t, h_w_t, ctx_t, alpha.T

    # prepare scan arguments
    seqs = [mask, state_below_emb, state_belowx_emb, state_belowctx_emb]

    shared_vars = [
            tparams[_p(prefix, 'U_cc')],
            tparams[_p(prefix, 'Ux_cc')],
            tparams[_p(prefix, 'W_cw')],
            tparams[_p(prefix, 'Wx_cw')],
            tparams[_p(prefix, 'U_ww')],
            tparams[_p(prefix, 'Ux_ww')],
            tparams[_p(prefix, 'b_w')],
            tparams[_p(prefix, 'bx_w')],
            tparams[_p(prefix, 'W_ctxc')],
            tparams[_p(prefix, 'Wx_ctxc')],
            tparams[_p(prefix, 'W_ctxw')],
            tparams[_p(prefix, 'Wx_ctxw')],
            tparams[_p(prefix, 'Wdecc_att')],
            tparams[_p(prefix, 'Wdecw_att')],
            tparams[_p(prefix, 'U_att')],
            tparams[_p(prefix, 'c_att')],
        ]

    if one_step:
        rval = _step(*(seqs+[init_state_char, init_state_word,
                             None, None,
                             proj_ctx, context]+shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[
                                        init_state_char,
                                        init_state_word,
                                        tensor.alloc(0., n_samples, context.shape[2]),
                                        tensor.alloc(0., n_samples, context.shape[0])
                                    ],
                                    non_sequences=[proj_ctx, context]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=n_steps,
                                    profile=profile,
                                    strict=True)
    return rval


def param_init_biscale_decoder(options, params,
                               prefix='biscale_decoder',
                               nin=None,
                               dim_char=None,
                               dim_word=None,
                               dimctx=None,
                               scalar_bound=False):
    if nin is None:
        nin = options['n_words']
    if dim_char is None:
        dim_char = options['dec_dim']
    if dim_word is None:
        dim_word = options['dec_dim']
    if dimctx is None:
        dimctx = options['enc_dim'] * 2

    # embedding to gates transformation weights, biases
    if scalar_bound:
        W_xc = norm_vector(nin)
        params[_p(prefix, 'b_c')] = numpy.zeros((1,)).astype('float32')
    else:
        W_xc = norm_weight(nin, dim_char)
        params[_p(prefix, 'b_c')] = numpy.zeros((dim_char,)).astype('float32')
    params[_p(prefix, 'W_xc')] = W_xc

    # recurrent transformation weights for gates
    if scalar_bound:
        U_cc = norm_vector(dim_char)
        U_wc = norm_vector(dim_char)
    else:
        U_cc = ortho_weight(dim_char)
        U_wc = ortho_weight(dim_char)
    params[_p(prefix, 'U_cc')] = U_cc
    params[_p(prefix, 'U_wc')] = U_wc

    # embedding to hidden state proposal weights, biases
    Wx_xc = norm_weight(nin, dim_char)
    params[_p(prefix, 'Wx_xc')] = Wx_xc
    params[_p(prefix, 'bx_c')] = numpy.zeros((dim_char,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux_cc = ortho_weight(dim_char)
    params[_p(prefix, 'Ux_cc')] = Ux_cc
    Ux_wc = ortho_weight(dim_char)
    params[_p(prefix, 'Ux_wc')] = Ux_wc

    # embedding to gates transformation weights, biases
    if scalar_bound:
        W_cw = norm_vector(dim_char)
        params[_p(prefix, 'b_w')] = numpy.zeros((1,)).astype('float32')
    else:
        W_cw = norm_weight(dim_char, dim_word)
        params[_p(prefix, 'b_w')] = numpy.zeros((dim_word,)).astype('float32')
    params[_p(prefix, 'W_cw')] = W_cw

    # recurrent transformation weights for gates
    if scalar_bound:
        U_ww = norm_vector(dim_word)
    else:
        U_ww = ortho_weight(dim_word)
    params[_p(prefix, 'U_ww')] = U_ww

    # embedding to hidden state proposal weights, biases
    Wx_cw = norm_weight(dim_char, dim_word)
    params[_p(prefix, 'Wx_cw')] = Wx_cw
    params[_p(prefix, 'bx_w')] = numpy.zeros((dim_word,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux_ww = ortho_weight(dim_word)
    params[_p(prefix, 'Ux_ww')] = Ux_ww

    # context to GRU gates: char-level
    if scalar_bound:
        W_ctxc = norm_vector(dimctx)
    else:
        W_ctxc = norm_weight(dimctx, dim_char)
    params[_p(prefix, 'W_ctxc')] = W_ctxc

    # context to hidden proposal: char-level
    Wx_ctxc = norm_weight(dimctx, dim_char)
    params[_p(prefix, 'Wx_ctxc')] = Wx_ctxc

    # context to GRU gates: word-level
    if scalar_bound:
        W_ctxw = norm_vector(dimctx)
    else:
        W_ctxw = norm_weight(dimctx, dim_word)
    params[_p(prefix, 'W_ctxw')] = W_ctxw

    # context to hidden proposal: word-level
    Wx_ctxw = norm_weight(dimctx, dim_word)
    params[_p(prefix, 'Wx_ctxw')] = Wx_ctxw

    # attention: prev -> hidden
    Winp_att = norm_weight(nin, dimctx)
    params[_p(prefix, 'Winp_att')] = Winp_att

    # attention: context -> hidden
    Wctx_att = norm_weight(dimctx)
    params[_p(prefix, 'Wctx_att')] = Wctx_att

    # attention: decoder -> hidden
    Wdec_att = norm_weight(dim_word, dimctx)
    params[_p(prefix, 'Wdec_att')] = Wdec_att

    # attention: hidden bias
    params[_p(prefix, 'b_att')] = numpy.zeros((dimctx,)).astype('float32')

    # attention
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_att')] = c_att

    return params


def biscale_decoder(tparams, state_below, options,
                    prefix='biscale_decoder',
                    mask=None, one_step=False,
                    context=None, context_mask=None,
                    init_state_char=None, init_state_word=None,
                    init_bound_char=None, init_bound_word=None,
                    scalar_bound=False,
                    **kwargs):

    assert context, 'Context must be provided'
    assert context.ndim == 3, \
        'Context must be 3-D: #annotation x #sample x #dim'

    if one_step:
        assert init_state_char, 'previous state must be provided'
        assert init_state_word, 'previous state must be provided'
        assert init_bound_char, 'previous bound must be provided'
        assert init_bound_word, 'previous bound must be provided'

    n_steps = state_below.shape[0]
    if state_below.ndim in [2, 3]:
        n_samples = state_below.shape[1]
    elif state_below.ndim == 1:
        if not  one_step:
            raise ValueError('if state_below.ndim is 1, one_step shoud also be 1')
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim_char = tparams[_p(prefix, 'Ux_cc')].shape[1]
    dim_word = tparams[_p(prefix, 'Ux_ww')].shape[1]

    if state_below.dtype == 'int64':
        state_below_emb = tparams[_p(prefix, 'W_xc')][state_below.flatten()]
        if scalar_bound:
            state_below_emb += tensor.addbroadcast(tparams[_p(prefix, 'b_c')], 0)
        else:
            state_below_emb += tparams[_p(prefix, 'b_c')]
        state_belowx_emb = tparams[_p(prefix, 'Wx_xc')][state_below.flatten()] + tparams[_p(prefix, 'bx_c')]
        state_belowctx_emb = tparams[_p(prefix, 'Winp_att')][state_below.flatten()]
        if state_below.ndim == 2:
            state_below_emb = state_below_emb.reshape((n_steps, n_samples, -1))
            state_belowx_emb = state_belowx_emb.reshape((n_steps, n_samples, -1))
            state_belowctx_emb = state_belowctx_emb.reshape((n_steps, n_samples, -1))
    else:
        state_below_emb = tensor.dot(state_below, tparams[_p(prefix, 'W_xc')])
        if scalar_bound:
            state_below_emb += tensor.addbroadcast(tparams[_p(prefix, 'b_c')], 0)
        else:
            state_below_emb += tparams[_p(prefix, 'b_c')]
        state_belowx_emb = tensor.dot(state_below, tparams[_p(prefix, 'Wx_xc')]) + tparams[_p(prefix, 'bx_c')]
        state_belowctx_emb = tensor.dot(state_below, tparams[_p(prefix, 'Winp_att')])

    # initial/previous state
    if init_state_char is None:
        init_state_char = tensor.alloc(0., n_samples, dim_char).astype('float32')
    if init_state_word is None:
        init_state_word = tensor.alloc(0., n_samples, dim_word).astype('float32')
    if scalar_bound:
        if init_bound_char is None:
            init_bound_char = tensor.alloc(0, n_samples).astype('float32')
        if init_bound_word is None:
            init_bound_char = tensor.alloc(0, n_samples).astype('float32')
    else:
        if init_bound_char is None:
            init_bound_char = tensor.zeros_like(init_state_char)
        if init_bound_word is None:
            init_bound_word = tensor.zeros_like(init_state_word)

    # projected context
    proj_ctx = tensor.dot(context, tparams[_p(prefix, 'Wctx_att')]) + tparams[_p(prefix, 'b_att')]

    # step function to be used by scan
    def _step(m_t,
              state_below_emb_t,
              state_belowx_emb_t,
              state_belowctx_emb_t,
              h_c_tm1, h_w_tm1,
              bd_c_tm1, bd_w_tm1,
              ctx_t,
              alpha_t,
              proj_ctx_all,
              context,
              U_cc, Ux_cc, U_wc, Ux_wc,
              W_cw, Wx_cw, U_ww, Ux_ww, b_w, bx_w,
              W_ctxc, Wx_ctxc, W_ctxw, Wx_ctxw,
              Wdec_att,
              U_att, c_att):
        # ~~ attention ~~ #
        # project previous hidden states
        proj_state = tensor.dot(h_w_tm1, Wdec_att)

        # add projected context
        proj_ctx = proj_ctx_all + proj_state[None, :, :] + state_belowctx_emb_t
        proj_h = tensor.tanh(proj_ctx)

        # compute alignment weights
        alpha = tensor.dot(proj_h, U_att) + c_att
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0))
        #alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)

        # compute the weighted averages - current context to GRU
        ctx_t = (context * alpha[:, :, None]).sum(0)

        if scalar_bound:
            bd_c_tm1 = bd_c_tm1[:, None]
            bd_w_tm1 = bd_w_tm1[:, None]

        # compute char-level
        preact_c = tensor.dot((1 - bd_c_tm1) * h_c_tm1, U_cc) + tensor.dot(bd_c_tm1 * h_w_tm1, U_wc) + tensor.dot(ctx_t, W_ctxc )

        if scalar_bound:
            preact_c += state_below_emb_t
            preact_c = preact_c[:, None]
        else:
            preact_c += state_below_emb_t

        # update gates
        bd_c_t = tensor.nnet.sigmoid(preact_c)

        # compute the hidden state proposal: char-level
        preactx_c = tensor.dot((1 - bd_c_tm1) * h_c_tm1, Ux_cc) + tensor.dot(bd_c_tm1 * h_w_tm1, Ux_wc) + tensor.dot(ctx_t, Wx_ctxc) + state_belowx_emb_t
        h_c_t = tensor.tanh(preactx_c)
        h_c_t = m_t[:, None] * h_c_t + (1. - m_t)[:, None] * h_c_tm1

        # compute word-level
        preact_w = tensor.dot((1 - bd_w_tm1) * h_w_tm1, U_ww) + tensor.dot(bd_c_t * h_c_t, W_cw) + tensor.dot(ctx_t, W_ctxw)

        if scalar_bound:
            preact_w += b_w[:, None]
            preact_w = preact_w.T
        else:
            preact_w += b_w

        # update gates for word-level
        bd_w_t = tensor.nnet.sigmoid(preact_w)

        # compute the hidden state proposal: word-level
        preactx_w = tensor.dot((1 - bd_w_tm1) * h_w_tm1, Ux_ww) + tensor.dot(bd_c_t * h_c_t, Wx_cw) + tensor.dot(ctx_t, Wx_ctxw) + bx_w
        h_w_t = tensor.tanh(preactx_w)
        h_w_t = bd_c_t * h_w_t + (1. - bd_c_t) * h_w_tm1
        h_w_t = m_t[:, None] * h_w_t + (1. - m_t)[:, None] * h_w_tm1

        if scalar_bound:
            bd_c_t = bd_c_t.flatten()
            bd_w_t = bd_w_t.flatten()

        return h_c_t, h_w_t, bd_c_t, bd_w_t, ctx_t, alpha.T

    # prepare scan arguments
    seqs = [mask, state_below_emb, state_belowx_emb, state_belowctx_emb]

    shared_vars = [
            tparams[_p(prefix, 'U_cc')],
            tparams[_p(prefix, 'Ux_cc')],
            tparams[_p(prefix, 'U_wc')],
            tparams[_p(prefix, 'Ux_wc')],
            tparams[_p(prefix, 'W_cw')],
            tparams[_p(prefix, 'Wx_cw')],
            tparams[_p(prefix, 'U_ww')],
            tparams[_p(prefix, 'Ux_ww')],
            tparams[_p(prefix, 'b_w')],
            tparams[_p(prefix, 'bx_w')],
            tparams[_p(prefix, 'W_ctxc')],
            tparams[_p(prefix, 'Wx_ctxc')],
            tparams[_p(prefix, 'W_ctxw')],
            tparams[_p(prefix, 'Wx_ctxw')],
            tparams[_p(prefix, 'Wdec_att')],
            tparams[_p(prefix, 'U_att')],
            tparams[_p(prefix, 'c_att')],
        ]

    if one_step:
        rval = _step(*(seqs+[init_state_char, init_state_word,
                             init_bound_char, init_bound_word,
                             None, None,
                             proj_ctx, context]+shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[
                                        init_state_char,
                                        init_state_word,
                                        init_bound_char,
                                        init_bound_word,
                                        tensor.alloc(0., n_samples, context.shape[2]),
                                        tensor.alloc(0., n_samples, context.shape[0])
                                    ],
                                    non_sequences=[proj_ctx, context]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=n_steps,
                                    profile=profile,
                                    strict=True)
    return rval


def param_init_biscale_decoder_attc(options, params,
                                    prefix='biscale_decoder_attc',
                                    nin=None,
                                    dim_char=None,
                                    dim_word=None,
                                    dimctx=None,
                                    scalar_bound=False):
    if nin is None:
        nin = options['n_words']
    if dim_char is None:
        dim_char = options['dec_dim']
    if dim_word is None:
        dim_word = options['dec_dim']
    if dimctx is None:
        dimctx = options['enc_dim'] * 2

    # embedding to gates transformation weights, biases
    if scalar_bound:
        W_xc = norm_vector(nin)
        params[_p(prefix, 'b_c')] = numpy.zeros((1,)).astype('float32')
    else:
        W_xc = norm_weight(nin, dim_char)
        params[_p(prefix, 'b_c')] = numpy.zeros((dim_char,)).astype('float32')
    params[_p(prefix, 'W_xc')] = W_xc

    # recurrent transformation weights for gates
    if scalar_bound:
        U_cc = norm_vector(dim_char)
        U_wc = norm_vector(dim_char)
    else:
        U_cc = ortho_weight(dim_char)
        U_wc = ortho_weight(dim_char)
    params[_p(prefix, 'U_cc')] = U_cc
    params[_p(prefix, 'U_wc')] = U_wc

    # embedding to hidden state proposal weights, biases
    Wx_xc = norm_weight(nin, dim_char)
    params[_p(prefix, 'Wx_xc')] = Wx_xc
    params[_p(prefix, 'bx_c')] = numpy.zeros((dim_char,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux_cc = ortho_weight(dim_char)
    params[_p(prefix, 'Ux_cc')] = Ux_cc
    Ux_wc = ortho_weight(dim_char)
    params[_p(prefix, 'Ux_wc')] = Ux_wc

    # embedding to gates transformation weights, biases
    if scalar_bound:
        W_cw = norm_vector(dim_char)
        params[_p(prefix, 'b_w')] = numpy.zeros((1,)).astype('float32')
    else:
        W_cw = norm_weight(dim_char, dim_word)
        params[_p(prefix, 'b_w')] = numpy.zeros((dim_word,)).astype('float32')
    params[_p(prefix, 'W_cw')] = W_cw

    # recurrent transformation weights for gates
    if scalar_bound:
        U_ww = norm_vector(dim_word)
    else:
        U_ww = ortho_weight(dim_word)
    params[_p(prefix, 'U_ww')] = U_ww

    # embedding to hidden state proposal weights, biases
    Wx_cw = norm_weight(dim_char, dim_word)
    params[_p(prefix, 'Wx_cw')] = Wx_cw
    params[_p(prefix, 'bx_w')] = numpy.zeros((dim_word,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux_ww = ortho_weight(dim_word)
    params[_p(prefix, 'Ux_ww')] = Ux_ww

    # context to GRU gates: char-level
    if scalar_bound:
        W_ctxc = norm_vector(dimctx)
    else:
        W_ctxc = norm_weight(dimctx, dim_char)
    params[_p(prefix, 'W_ctxc')] = W_ctxc

    # context to hidden proposal: char-level
    Wx_ctxc = norm_weight(dimctx, dim_char)
    params[_p(prefix, 'Wx_ctxc')] = Wx_ctxc

    # context to GRU gates: word-level
    if scalar_bound:
        W_ctxw = norm_vector(dimctx)
    else:
        W_ctxw = norm_weight(dimctx, dim_word)
    params[_p(prefix, 'W_ctxw')] = W_ctxw

    # context to hidden proposal: word-level
    Wx_ctxw = norm_weight(dimctx, dim_word)
    params[_p(prefix, 'Wx_ctxw')] = Wx_ctxw

    # attention: prev -> hidden
    Winp_att = norm_weight(nin, dimctx)
    params[_p(prefix, 'Winp_att')] = Winp_att

    # attention: context -> hidden
    Wctx_att = norm_weight(dimctx)
    params[_p(prefix, 'Wctx_att')] = Wctx_att

    # attention: decoder -> hidden
    Wdec_att = norm_weight(dim_char, dimctx)
    params[_p(prefix, 'Wdec_att')] = Wdec_att

    # attention: hidden bias
    params[_p(prefix, 'b_att')] = numpy.zeros((dimctx,)).astype('float32')

    # attention
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_att')] = c_att

    return params


def biscale_decoder_attc(tparams, state_below, options,
                         prefix='biscale_decoder_attc',
                         mask=None, one_step=False,
                         context=None, context_mask=None,
                         init_state_char=None, init_state_word=None,
                         init_bound_char=None, init_bound_word=None,
                         scalar_bound=False,
                         **kwargs):

    assert context, 'Context must be provided'
    assert context.ndim == 3, \
        'Context must be 3-D: #annotation x #sample x #dim'

    if one_step:
        assert init_state_char, 'previous state must be provided'
        assert init_state_word, 'previous state must be provided'
        assert init_bound_char, 'previous bound must be provided'
        assert init_bound_word, 'previous bound must be provided'

    n_steps = state_below.shape[0]
    if state_below.ndim in [2, 3]:
        n_samples = state_below.shape[1]
    elif state_below.ndim == 1:
        if not  one_step:
            raise ValueError('if state_below.ndim is 1, one_step shoud also be 1')
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim_char = tparams[_p(prefix, 'Ux_cc')].shape[1]
    dim_word = tparams[_p(prefix, 'Ux_ww')].shape[1]

    if state_below.dtype == 'int64':
        state_below_emb = tparams[_p(prefix, 'W_xc')][state_below.flatten()]
        if scalar_bound:
            state_below_emb += tensor.addbroadcast(tparams[_p(prefix, 'b_c')], 0)
        else:
            state_below_emb += tparams[_p(prefix, 'b_c')]
        state_belowx_emb = tparams[_p(prefix, 'Wx_xc')][state_below.flatten()] + tparams[_p(prefix, 'bx_c')]
        state_belowctx_emb = tparams[_p(prefix, 'Winp_att')][state_below.flatten()]
        if state_below.ndim == 2:
            state_below_emb = state_below_emb.reshape((n_steps, n_samples, -1))
            state_belowx_emb = state_belowx_emb.reshape((n_steps, n_samples, -1))
            state_belowctx_emb = state_belowctx_emb.reshape((n_steps, n_samples, -1))
    else:
        state_below_emb = tensor.dot(state_below, tparams[_p(prefix, 'W_xc')])
        if scalar_bound:
            state_below_emb += tensor.addbroadcast(tparams[_p(prefix, 'b_c')], 0)
        else:
            state_below_emb += tparams[_p(prefix, 'b_c')]
        state_belowx_emb = tensor.dot(state_below, tparams[_p(prefix, 'Wx_xc')]) + tparams[_p(prefix, 'bx_c')]
        state_belowctx_emb = tensor.dot(state_below, tparams[_p(prefix, 'Winp_att')])

    # initial/previous state
    if init_state_char is None:
        init_state_char = tensor.alloc(0., n_samples, dim_char).astype('float32')
    if init_state_word is None:
        init_state_word = tensor.alloc(0., n_samples, dim_word).astype('float32')
    if scalar_bound:
        if init_bound_char is None:
            init_bound_char = tensor.alloc(0, n_samples).astype('float32')
        if init_bound_word is None:
            init_bound_char = tensor.alloc(0, n_samples).astype('float32')
    else:
        if init_bound_char is None:
            init_bound_char = tensor.zeros_like(init_state_char)
        if init_bound_word is None:
            init_bound_word = tensor.zeros_like(init_state_word)

    # projected context
    proj_ctx = tensor.dot(context, tparams[_p(prefix, 'Wctx_att')]) + tparams[_p(prefix, 'b_att')]

    # step function to be used by scan
    def _step(m_t,
              state_below_emb_t,
              state_belowx_emb_t,
              state_belowctx_emb_t,
              h_c_tm1, h_w_tm1,
              bd_c_tm1, bd_w_tm1,
              ctx_t,
              alpha_t,
              proj_ctx_all,
              context,
              U_cc, Ux_cc, U_wc, Ux_wc,
              W_cw, Wx_cw, U_ww, Ux_ww, b_w, bx_w,
              W_ctxc, Wx_ctxc, W_ctxw, Wx_ctxw,
              Wdec_att,
              U_att, c_att):
        # ~~ attention ~~ #
        # project previous hidden states
        proj_state = tensor.dot(h_c_tm1, Wdec_att)

        # add projected context
        proj_ctx = proj_ctx_all + proj_state[None, :, :] + state_belowctx_emb_t
        proj_h = tensor.tanh(proj_ctx)

        # compute alignment weights
        alpha = tensor.dot(proj_h, U_att) + c_att
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0))
        #alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)

        # compute the weighted averages - current context to GRU
        ctx_t = (context * alpha[:, :, None]).sum(0)

        if scalar_bound:
            bd_c_tm1 = bd_c_tm1[:, None]
            bd_w_tm1 = bd_w_tm1[:, None]

        # compute char-level
        preact_c = tensor.dot((1 - bd_c_tm1) * h_c_tm1, U_cc) + tensor.dot(bd_c_tm1 * h_w_tm1, U_wc) + tensor.dot(ctx_t, W_ctxc )

        if scalar_bound:
            preact_c += state_below_emb_t
            preact_c = preact_c[:, None]
        else:
            preact_c += state_below_emb_t

        # update gates
        bd_c_t = tensor.nnet.sigmoid(preact_c)

        # compute the hidden state proposal: char-level
        preactx_c = tensor.dot((1 - bd_c_tm1) * h_c_tm1, Ux_cc) + tensor.dot(bd_c_tm1 * h_w_tm1, Ux_wc) + tensor.dot(ctx_t, Wx_ctxc) + state_belowx_emb_t
        h_c_t = tensor.tanh(preactx_c)
        h_c_t = m_t[:, None] * h_c_t + (1. - m_t)[:, None] * h_c_tm1

        # compute word-level
        preact_w = tensor.dot((1 - bd_w_tm1) * h_w_tm1, U_ww) + tensor.dot(bd_c_t * h_c_t, W_cw) + tensor.dot(ctx_t, W_ctxw)

        if scalar_bound:
            preact_w += b_w[:, None]
            preact_w = preact_w.T
        else:
            preact_w += b_w

        # update gates for word-level
        bd_w_t = tensor.nnet.sigmoid(preact_w)

        # compute the hidden state proposal: word-level
        preactx_w = tensor.dot((1 - bd_w_tm1) * h_w_tm1, Ux_ww) + tensor.dot(bd_c_t * h_c_t, Wx_cw) + tensor.dot(ctx_t, Wx_ctxw) + bx_w
        h_w_t = tensor.tanh(preactx_w)
        h_w_t = bd_c_t * h_w_t + (1. - bd_c_t) * h_w_tm1
        h_w_t = m_t[:, None] * h_w_t + (1. - m_t)[:, None] * h_w_tm1

        if scalar_bound:
            bd_c_t = bd_c_t.flatten()
            bd_w_t = bd_w_t.flatten()

        return h_c_t, h_w_t, bd_c_t, bd_w_t, ctx_t, alpha.T

    # prepare scan arguments
    seqs = [mask, state_below_emb, state_belowx_emb, state_belowctx_emb]

    shared_vars = [
            tparams[_p(prefix, 'U_cc')],
            tparams[_p(prefix, 'Ux_cc')],
            tparams[_p(prefix, 'U_wc')],
            tparams[_p(prefix, 'Ux_wc')],
            tparams[_p(prefix, 'W_cw')],
            tparams[_p(prefix, 'Wx_cw')],
            tparams[_p(prefix, 'U_ww')],
            tparams[_p(prefix, 'Ux_ww')],
            tparams[_p(prefix, 'b_w')],
            tparams[_p(prefix, 'bx_w')],
            tparams[_p(prefix, 'W_ctxc')],
            tparams[_p(prefix, 'Wx_ctxc')],
            tparams[_p(prefix, 'W_ctxw')],
            tparams[_p(prefix, 'Wx_ctxw')],
            tparams[_p(prefix, 'Wdec_att')],
            tparams[_p(prefix, 'U_att')],
            tparams[_p(prefix, 'c_att')],
        ]

    if one_step:
        rval = _step(*(seqs+[init_state_char, init_state_word,
                             init_bound_char, init_bound_word,
                             None, None,
                             proj_ctx, context]+shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[
                                        init_state_char,
                                        init_state_word,
                                        init_bound_char,
                                        init_bound_word,
                                        tensor.alloc(0., n_samples, context.shape[2]),
                                        tensor.alloc(0., n_samples, context.shape[0])
                                    ],
                                    non_sequences=[proj_ctx, context]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=n_steps,
                                    profile=profile,
                                    strict=True)
    return rval


def param_init_biscale_decoder_both(options, params,
                                    prefix='biscale_decoder_both',
                                    nin=None,
                                    dim_char=None,
                                    dim_word=None,
                                    dimctx=None,
                                    scalar_bound=False):
    if nin is None:
        nin = options['n_words']
    if dim_char is None:
        dim_char = options['dec_dim']
    if dim_word is None:
        dim_word = options['dec_dim']
    if dimctx is None:
        dimctx = options['enc_dim'] * 2

    # embedding to gates transformation weights, biases
    if scalar_bound:
        W_xc = norm_vector(nin)
        params[_p(prefix, 'b_c')] = numpy.zeros((1,)).astype('float32')
    else:
        W_xc = norm_weight(nin, dim_char)
        params[_p(prefix, 'b_c')] = numpy.zeros((dim_char,)).astype('float32')
    params[_p(prefix, 'W_xc')] = W_xc

    # recurrent transformation weights for gates
    if scalar_bound:
        U_cc = norm_vector(dim_char)
        U_wc = norm_vector(dim_char)
    else:
        U_cc = ortho_weight(dim_char)
        U_wc = ortho_weight(dim_char)
    params[_p(prefix, 'U_cc')] = U_cc
    params[_p(prefix, 'U_wc')] = U_wc

    # embedding to hidden state proposal weights, biases
    Wx_xc = norm_weight(nin, dim_char)
    params[_p(prefix, 'Wx_xc')] = Wx_xc
    params[_p(prefix, 'bx_c')] = numpy.zeros((dim_char,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux_cc = ortho_weight(dim_char)
    params[_p(prefix, 'Ux_cc')] = Ux_cc
    Ux_wc = ortho_weight(dim_char)
    params[_p(prefix, 'Ux_wc')] = Ux_wc

    # embedding to gates transformation weights, biases
    if scalar_bound:
        W_cw = norm_vector(dim_char)
        params[_p(prefix, 'b_w')] = numpy.zeros((1,)).astype('float32')
    else:
        W_cw = norm_weight(dim_char, dim_word)
        params[_p(prefix, 'b_w')] = numpy.zeros((dim_word,)).astype('float32')
    params[_p(prefix, 'W_cw')] = W_cw

    # recurrent transformation weights for gates
    if scalar_bound:
        U_ww = norm_vector(dim_word)
    else:
        U_ww = ortho_weight(dim_word)
    params[_p(prefix, 'U_ww')] = U_ww

    # embedding to hidden state proposal weights, biases
    Wx_cw = norm_weight(dim_char, dim_word)
    params[_p(prefix, 'Wx_cw')] = Wx_cw
    params[_p(prefix, 'bx_w')] = numpy.zeros((dim_word,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux_ww = ortho_weight(dim_word)
    params[_p(prefix, 'Ux_ww')] = Ux_ww

    # context to GRU gates: char-level
    if scalar_bound:
        W_ctxc = norm_vector(dimctx)
    else:
        W_ctxc = norm_weight(dimctx, dim_char)
    params[_p(prefix, 'W_ctxc')] = W_ctxc

    # context to hidden proposal: char-level
    Wx_ctxc = norm_weight(dimctx, dim_char)
    params[_p(prefix, 'Wx_ctxc')] = Wx_ctxc

    # context to GRU gates: word-level
    if scalar_bound:
        W_ctxw = norm_vector(dimctx)
    else:
        W_ctxw = norm_weight(dimctx, dim_word)
    params[_p(prefix, 'W_ctxw')] = W_ctxw

    # context to hidden proposal: word-level
    Wx_ctxw = norm_weight(dimctx, dim_word)
    params[_p(prefix, 'Wx_ctxw')] = Wx_ctxw

    # attention: prev -> hidden
    Winp_att = norm_weight(nin, dimctx)
    params[_p(prefix, 'Winp_att')] = Winp_att

    # attention: context -> hidden
    Wctx_att = norm_weight(dimctx)
    params[_p(prefix, 'Wctx_att')] = Wctx_att

    # attention: decoder -> hidden
    Wdecc_att = norm_weight(dim_char, dimctx)
    params[_p(prefix, 'Wdecc_att')] = Wdecc_att
    Wdecw_att = norm_weight(dim_word, dimctx)
    params[_p(prefix, 'Wdecw_att')] = Wdecw_att

    # attention: hidden bias
    params[_p(prefix, 'b_att')] = numpy.zeros((dimctx,)).astype('float32')

    # attention
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_att')] = c_att

    return params


def biscale_decoder_both(tparams, state_below, options,
                         prefix='biscale_decoder_both',
                         mask=None, one_step=False,
                         context=None, context_mask=None,
                         init_state_char=None, init_state_word=None,
                         init_bound_char=None, init_bound_word=None,
                         scalar_bound=False,
                         **kwargs):

    assert context, 'Context must be provided'
    assert context.ndim == 3, \
        'Context must be 3-D: #annotation x #sample x #dim'

    if one_step:
        assert init_state_char, 'previous state must be provided'
        assert init_state_word, 'previous state must be provided'
        assert init_bound_char, 'previous bound must be provided'
        assert init_bound_word, 'previous bound must be provided'

    n_steps = state_below.shape[0]
    if state_below.ndim in [2, 3]:
        n_samples = state_below.shape[1]
    elif state_below.ndim == 1:
        if not  one_step:
            raise ValueError('if state_below.ndim is 1, one_step shoud also be 1')
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim_char = tparams[_p(prefix, 'Ux_cc')].shape[1]
    dim_word = tparams[_p(prefix, 'Ux_ww')].shape[1]

    if state_below.dtype == 'int64':
        state_below_emb = tparams[_p(prefix, 'W_xc')][state_below.flatten()]
        if scalar_bound:
            state_below_emb += tensor.addbroadcast(tparams[_p(prefix, 'b_c')], 0)
        else:
            state_below_emb += tparams[_p(prefix, 'b_c')]
        state_belowx_emb = tparams[_p(prefix, 'Wx_xc')][state_below.flatten()] + tparams[_p(prefix, 'bx_c')]
        state_belowctx_emb = tparams[_p(prefix, 'Winp_att')][state_below.flatten()]
        if state_below.ndim == 2:
            state_below_emb = state_below_emb.reshape((n_steps, n_samples, -1))
            state_belowx_emb = state_belowx_emb.reshape((n_steps, n_samples, -1))
            state_belowctx_emb = state_belowctx_emb.reshape((n_steps, n_samples, -1))
    else:
        state_below_emb = tensor.dot(state_below, tparams[_p(prefix, 'W_xc')])
        if scalar_bound:
            state_below_emb += tensor.addbroadcast(tparams[_p(prefix, 'b_c')], 0)
        else:
            state_below_emb += tparams[_p(prefix, 'b_c')]
        state_belowx_emb = tensor.dot(state_below, tparams[_p(prefix, 'Wx_xc')]) + tparams[_p(prefix, 'bx_c')]
        state_belowctx_emb = tensor.dot(state_below, tparams[_p(prefix, 'Winp_att')])

    # initial/previous state
    if init_state_char is None:
        init_state_char = tensor.alloc(0., n_samples, dim_char).astype('float32')
    if init_state_word is None:
        init_state_word = tensor.alloc(0., n_samples, dim_word).astype('float32')
    if scalar_bound:
        if init_bound_char is None:
            init_bound_char = tensor.alloc(0, n_samples).astype('float32')
        if init_bound_word is None:
            init_bound_char = tensor.alloc(0, n_samples).astype('float32')
    else:
        if init_bound_char is None:
            init_bound_char = tensor.zeros_like(init_state_char)
        if init_bound_word is None:
            init_bound_word = tensor.zeros_like(init_state_word)

    # projected context
    proj_ctx = tensor.dot(context, tparams[_p(prefix, 'Wctx_att')]) + tparams[_p(prefix, 'b_att')]

    # step function to be used by scan
    def _step(m_t,
              state_below_emb_t,
              state_belowx_emb_t,
              state_belowctx_emb_t,
              h_c_tm1, h_w_tm1,
              bd_c_tm1, bd_w_tm1,
              ctx_t,
              alpha_t,
              proj_ctx_all,
              context,
              U_cc, Ux_cc, U_wc, Ux_wc,
              W_cw, Wx_cw, U_ww, Ux_ww, b_w, bx_w,
              W_ctxc, Wx_ctxc, W_ctxw, Wx_ctxw,
              Wdecc_att, Wdecw_att,
              U_att, c_att):
        # ~~ attention ~~ #
        # project previous hidden states
        proj_state = tensor.dot(h_w_tm1, Wdecw_att) + tensor.dot(h_c_tm1, Wdecc_att)

        # add projected context
        proj_ctx = proj_ctx_all + proj_state[None, :, :] + state_belowctx_emb_t
        proj_h = tensor.tanh(proj_ctx)

        # compute alignment weights
        alpha = tensor.dot(proj_h, U_att) + c_att
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0))
        #alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)

        # compute the weighted averages - current context to GRU
        ctx_t = (context * alpha[:, :, None]).sum(0)

        if scalar_bound:
            bd_c_tm1 = bd_c_tm1[:, None]
            bd_w_tm1 = bd_w_tm1[:, None]

        # compute char-level
        preact_c = tensor.dot((1 - bd_c_tm1) * h_c_tm1, U_cc) + tensor.dot(bd_c_tm1 * h_w_tm1, U_wc) + tensor.dot(ctx_t, W_ctxc )

        if scalar_bound:
            preact_c += state_below_emb_t
            preact_c = preact_c[:, None]
        else:
            preact_c += state_below_emb_t

        # update gates
        bd_c_t = tensor.nnet.sigmoid(preact_c)

        # compute the hidden state proposal: char-level
        preactx_c = tensor.dot((1 - bd_c_tm1) * h_c_tm1, Ux_cc) + tensor.dot(bd_c_tm1 * h_w_tm1, Ux_wc) + tensor.dot(ctx_t, Wx_ctxc) + state_belowx_emb_t
        h_c_t = tensor.tanh(preactx_c)
        h_c_t = m_t[:, None] * h_c_t + (1. - m_t)[:, None] * h_c_tm1

        # compute word-level
        preact_w = tensor.dot((1 - bd_w_tm1) * h_w_tm1, U_ww) + tensor.dot(bd_c_t * h_c_t, W_cw) + tensor.dot(ctx_t, W_ctxw)

        if scalar_bound:
            preact_w += b_w[:, None]
            preact_w = preact_w.T
        else:
            preact_w += b_w

        # update gates for word-level
        bd_w_t = tensor.nnet.sigmoid(preact_w)

        # compute the hidden state proposal: word-level
        preactx_w = tensor.dot((1 - bd_w_tm1) * h_w_tm1, Ux_ww) + tensor.dot(bd_c_t * h_c_t, Wx_cw) + tensor.dot(ctx_t, Wx_ctxw) + bx_w
        h_w_t = tensor.tanh(preactx_w)
        h_w_t = bd_c_t * h_w_t + (1. - bd_c_t) * h_w_tm1
        h_w_t = m_t[:, None] * h_w_t + (1. - m_t)[:, None] * h_w_tm1

        if scalar_bound:
            bd_c_t = bd_c_t.flatten()
            bd_w_t = bd_w_t.flatten()

        return h_c_t, h_w_t, bd_c_t, bd_w_t, ctx_t, alpha.T

    # prepare scan arguments
    seqs = [mask, state_below_emb, state_belowx_emb, state_belowctx_emb]

    shared_vars = [
            tparams[_p(prefix, 'U_cc')],
            tparams[_p(prefix, 'Ux_cc')],
            tparams[_p(prefix, 'U_wc')],
            tparams[_p(prefix, 'Ux_wc')],
            tparams[_p(prefix, 'W_cw')],
            tparams[_p(prefix, 'Wx_cw')],
            tparams[_p(prefix, 'U_ww')],
            tparams[_p(prefix, 'Ux_ww')],
            tparams[_p(prefix, 'b_w')],
            tparams[_p(prefix, 'bx_w')],
            tparams[_p(prefix, 'W_ctxc')],
            tparams[_p(prefix, 'Wx_ctxc')],
            tparams[_p(prefix, 'W_ctxw')],
            tparams[_p(prefix, 'Wx_ctxw')],
            tparams[_p(prefix, 'Wdecc_att')],
            tparams[_p(prefix, 'Wdecw_att')],
            tparams[_p(prefix, 'U_att')],
            tparams[_p(prefix, 'c_att')],
        ]

    if one_step:
        rval = _step(*(seqs+[init_state_char, init_state_word,
                             init_bound_char, init_bound_word,
                             None, None,
                             proj_ctx, context]+shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[
                                        init_state_char,
                                        init_state_word,
                                        init_bound_char,
                                        init_bound_word,
                                        tensor.alloc(0., n_samples, context.shape[2]),
                                        tensor.alloc(0., n_samples, context.shape[0])
                                    ],
                                    non_sequences=[proj_ctx, context]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=n_steps,
                                    profile=profile,
                                    strict=True)
    return rval


# optimizers
def gradient_clipping(grads, tparams, clip_c=10):
    g2 = 0.
    for g in grads:
        g2 += (g**2).sum()

    g2 = tensor.sqrt(g2)
    not_finite = tensor.or_(tensor.isnan(g2), tensor.isinf(g2))
    new_grads = []

    for p, g in zip(tparams.values(), grads):
        new_grads.append(tensor.switch(g2 > clip_c,
                                       g * (clip_c / g2),
                                       g))

    return new_grads, not_finite, tensor.lt(clip_c, g2)


def adam(lr, tparams, grads, inp, cost, not_finite=None, clipped=None,
         b1=0.9, b2=0.999, eps=1e-8, file_name=None):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]

    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    if not_finite is not None and clipped is not None:
        f_grad_shared = theano.function(inp, [cost, not_finite, clipped], updates=gsup, profile=profile)
    else:
        f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

    updates = OrderedDict()
    optparams = OrderedDict()
    optparams['i'] = numpy.float32(0.)

    for k, p in tparams.items():
        optparams[_p(k, 'm')] = p.get_value() * 0.
        optparams[_p(k, 'v')] = p.get_value() * 0.

    if file_name is not None:
        optparams = load_params(file_name, optparams)

    toptparams = init_tparams(optparams)

    i_t = toptparams['i'] + 1.
    fix1 = b1**i_t
    fix2 = b2**i_t
    lr_t = lr * tensor.sqrt(1. - fix2) / (1. - fix1)

    for (k, p), g in zip(tparams.items(), gshared):
        m_t = b1 * toptparams[_p(k, 'm')] + (1. - b1) * g
        v_t = b2 * toptparams[_p(k, 'v')] + (1. - b2) * g**2
        g_t = lr_t * m_t / (tensor.sqrt(v_t) + eps)
        p_t = p - g_t
        updates[toptparams[_p(k, 'm')]] = m_t
        updates[toptparams[_p(k, 'v')]] = v_t
        updates[p] = p_t
    updates[toptparams['i']] = i_t
    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update, toptparams


def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in
             zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, inp, cost, not_finite=None, clipped=None, mom=0.9, sec_mom=0.95, eps=1e-4):

    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, sec_mom * rg + (1. - sec_mom) * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, sec_mom * rg2 + (1. - sec_mom) * g**2)
             for rg2, g in zip(running_grads2, grads)]

    if not_finite is not None or clipped is not None:
        f_grad_shared = theano.function(inp, [cost, not_finite, clipped], updates=zgup+rgup+rg2up, profile=profile)
    else:
        f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up, profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, mom * ud - lr * zg / tensor.sqrt(rg2 - rg**2 + eps))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, x, mask, y, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update
