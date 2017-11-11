from __future__ import absolute_import
#from keras import backend as K
import numpy as np
from conf import *
import sys

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams

from theano.compile.nanguardmode import NanGuardMode

import lasagne
import cPickle

#theano.config.exception_verbosity="high"
#theano.config.optimizer="fast_compile"

from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

np.random.seed(args.random_seed)

def init_weight(n_in,n_out,activation_fn=sigmoid,pre="",special=False,uni=True,ones=False):
    rng = np.random.RandomState(1234)
    if special:
        W_values = glorot_uniform((n_in, n_out)) 
    else:
        if uni:
            W_values = np.asarray(rng.normal(size=(n_in, n_out), scale= .01, loc = .0), dtype = theano.config.floatX)
        else:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / np.sqrt(n_in + n_out)),
                    high=np.sqrt(6. / np.sqrt(n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation_fn == theano.tensor.nnet.sigmoid:
                W_values *= 4
                W_values /= 6

    b_values = np.zeros((n_out,), dtype=theano.config.floatX)

    if ones:
        b_values = np.ones((n_out,), dtype=theano.config.floatX)

    w = theano.shared(
        value=W_values,
        name='%sw'%pre, borrow=True
    )
    b = theano.shared(
        value=b_values,
        name='%sb'%pre, borrow=True
    )
    return w,b

def init_weight_file(fn,dimention=100,pre="embedding"):
    f = file(fn, 'rb')
    numnum = 1.
    eMatrix = []
    eMatrix.append([0.0]*dimention)
    embedding_list = cPickle.load(f)
    for word,em in embedding_list:
        eMatrix.append(em)
    print >> sys.stderr, "Total Read Embedding ", len(eMatrix)

    W_values = np.asarray(eMatrix,dtype = theano.config.floatX)

    w = theano.shared(
        value=W_values,
        name='%sw'%pre, borrow=True
    )
    return w

def get_fans(shape, dim_ordering='th'):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # assuming convolution kernels (2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if dim_ordering == 'th':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif dim_ordering == 'tf':
            receptive_field_size = np.prod(shape[:2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


def uniform(shape, scale=0.05, name=None):
    return random_uniform_variable(shape, -scale, scale, name=name)


def normal(shape, scale=0.05, name=None):
    return random_normal_variable(shape, 0.0, scale, name=name)


def lecun_uniform(shape, name=None, dim_ordering='th'):
    ''' Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    '''
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    scale = np.sqrt(3. / fan_in)
    return uniform(shape, scale, name=name)


def glorot_normal(shape, name=None, dim_ordering='th'):
    ''' Reference: Glorot & Bengio, AISTATS 2010
    '''
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, s, name=name)


def glorot_uniform(shape, name=None, dim_ordering='th'):
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s, name=name)


def he_normal(shape, name=None, dim_ordering='th'):
    ''' Reference:  He et al., http://arxiv.org/abs/1502.01852
    '''
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = np.sqrt(2. / fan_in)
    return normal(shape, s, name=name)


def he_uniform(shape, name=None, dim_ordering='th'):
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = np.sqrt(6. / fan_in)
    return uniform(shape, s, name=name)

'''
def orthogonal(shape, scale=1.1, name=None):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return K.variable(scale * q[:shape[0], :shape[1]], name=name).get_value()
'''

def random_uniform_variable(shape, low, high, dtype=theano.config.floatX, name=None):
    return np.asarray(np.random.uniform(low=low, high=high, size=shape),
                    dtype=dtype)
def random_normal_variable(shape, mean, scale, dtype=theano.config.floatX, name=None):
    return np.asarray(np.random.normal(loc=0.0, scale=scale, size=shape),
                    dtype=dtype)


#print lecun_uniform((2,3))
#print orthogonal((2,3))
#print he_normal((2,3))
