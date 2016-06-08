from __future__ import absolute_import
import theano.tensor as T


def softmax(x):
    #return T.nnet.softmax(x)
    return T.nnet.softmax(x.reshape((-1, x.shape[-1]))).reshape(x.shape)


def time_distributed_softmax(x):
    import warnings
    warnings.warn("time_distributed_softmax is deprecated. Just use softmax!", DeprecationWarning)
    return softmax(x)


def softplus(x):
    return T.nnet.softplus(x)


def relu(x):
    return T.nnet.relu(x, 0)

def lutanh(x):
    return T.nnet.relu(x+1, 0) - T.nnet.relu(x-1, 0) -1

def lusigmoid(x):
    k = 0.25
    return T.nnet.relu(k*(x+1/(2*k)), 0) - T.nnet.relu(k*(x-1/(2*k)), 0)


def tanh(x):
    return T.tanh(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)


def linear(x):
    '''
    The function returns the variable that is passed in, so all types work
    '''
    return x


from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'activation function')
