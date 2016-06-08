# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations
from ..utils.theano_utils import shared_scalar, shared_zeros, alloc_zeros_matrix, sharedX, shared_ones
from ..layers.core import Layer, MaskedLayer
from six.moves import range
import pdb

class Recurrent(MaskedLayer):
    input_ndim = 3

    def get_output_mask(self, train=None):
        if self.return_sequences:
            return super(Recurrent, self).get_output_mask(train)
        else:
            return None

    def get_padded_shuffled_mask(self, train, X, pad=0):
        mask = self.get_input_mask(train)
        if mask is None:
            mask = T.ones_like(X.sum(axis=-1))  # is there a better way to do this without a sum?

        # mask is (nb_samples, time)
        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # the new dimension (the '1') is made broadcastable
        # see http://deeplearning.net/software/theano/library/tensor/basic.html#broadcasting-in-theano-vs-numpy
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)

        if pad > 0:
            # left-pad in time with 0
            padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
            mask = T.concatenate([padding, mask], axis=0)
        return mask.astype('int8')

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.output_dim)
        else:
            return (input_shape[0], self.output_dim)


class SimpleRNN(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        self.dummy = theano.shared(np.zeros((50, 50, self.output_dim)).astype('float32'))
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(SimpleRNN, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W = self.init((input_dim, self.output_dim), 0.5)
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))
        self.params = [self.W, self.U, self.b]
        self.params_ = [self.W, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, dummy, mask_tm1, h_tm1, u):
        '''
            Variable names follow the conventions from:
            http://deeplearning.net/software/theano/library/scan.html

        '''
        return dummy + self.activation(x_t + mask_tm1 * T.dot(h_tm1, u))

    def get_output(self, dummy, train=False):
        X = self.get_input(train)  # shape: (nb_samples, time (padded with zeros), input_dim)
        # new shape: (time, nb_samples, input_dim) -> because theano.scan iterates over main dimension
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))
        x = T.dot(X, self.W) + self.b

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        outputs, updates = theano.scan(
            self._step,  # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=[x, dummy, dict(input=padded_mask, taps=[-1])],  # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=self.U,  # static inputs to _step
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))








class SimpleRNNlinear(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(SimpleRNNlinear, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W = self.init((input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))
        self.params = [self.W, self.U, self.b]
        self.params_ = [self.W, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, mask_tm1, h_tm1, u):
        '''
            Variable names follow the conventions from:
            http://deeplearning.net/software/theano/library/scan.html

        '''
        return x_t + mask_tm1 * T.dot(h_tm1, u)

    def get_output(self, train=False):
        X = self.get_input(train)  # shape: (nb_samples, time (padded with zeros), input_dim)
        # new shape: (time, nb_samples, input_dim) -> because theano.scan iterates over main dimension
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))
        x = T.dot(X, self.W) + self.b

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        outputs, updates = theano.scan(
            self._step,  # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=[x, dict(input=padded_mask, taps=[-1])],  # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=self.U,  # static inputs to _step
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))








class IdentityRNN(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(IdentityRNN, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W = self.init((input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        #self.D = self.init((self.output_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.W_gate = self.init((input_dim, self.output_dim)) 
        self.U_gate = self.init((self.output_dim, self.output_dim))
        self.b_gate = shared_zeros((self.output_dim))
        #self.th_gate = sharedX(0.05) 
        #self.U_ = T.dot(self.U, np.identity(self.output_dim).astype('float32')+self.D)
        self.params = [self.W, self.U, self.b] #+ [self.U_gate, self.b_gate]
        self.params_ = [self.W, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, mask_tm1, h_tm1, u):
        '''
            Variable names follow the conventions from:
            http://deeplearning.net/software/theano/library/scan.html

        '''
        # changed by saizheng
        #mask = ((np.random.uniform(size = (self.U.get_value().shape[0],))>0.)*2-1).astype('float32')     
        #mask = T.nnet.sigmoid(T.dot(h_tm1, self.U_gate) + self.b_gate)
        return self.activation(x_t + mask_tm1 * (T.dot(h_tm1, u) + h_tm1)) #+(1-mask)*h_tm1#+ h_tm1*(self.th_gate*T.tanh(T.dot(h_tm1, self.U_gate) + self.b_gate))))

    def _step2(self, x_t, mask_tm1, h_tm1, u):
        '''
            Variable names follow the conventions from:
            http://deeplearning.net/software/theano/library/scan.html

        '''
        # changed by saizheng
        v = T.tanh(T.dot(h_tm1, self.U))
        return self.activation(x_t + mask_tm1 * (T.sum(h_tm1*v, axis=1)*v + h_tm1)) #+ h_tm1*(self.th_gate*T.tanh(T.dot(h_tm1, self.U_gate) + self.b_gate))))


    def get_output(self, train=False):
        X = self.get_input(train)  # shape: (nb_samples, time (padded with zeros), input_dim)
        # new shape: (time, nb_samples, input_dim) -> because theano.scan iterates over main dimension
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))
        x = T.dot(X, self.W) + self.b

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        outputs, updates = theano.scan(
            self._step,  # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=[x, dict(input=padded_mask, taps=[-1])],  # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=self.U,  # static inputs to _step
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ForgetgateRNN(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.gate_dim = 20
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(ForgetgateRNN, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W = self.init((input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        #self.D = self.init((self.output_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        gate_dim = self.gate_dim
        self.W_gate = self.init((self.output_dim, gate_dim)) 
        self.U_gate = self.init((gate_dim, gate_dim))
        self.b_gate = shared_zeros((gate_dim))
        self.G_gate = self.init((gate_dim, self.output_dim))
        self.Gb_gate = shared_zeros((self.output_dim))

        self.params = [self.W, self.U, self.b] + [self.W_gate, self.U_gate, self.b_gate, self.G_gate, self.Gb_gate] 
        self.params_gate = [self.U_gate, self.W_gate, self.b_gate]
        self.params_ = [self.W, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def f(self, x):
        return T.switch(T.ge(x,0.),1.,0.)

    def _step(self, x_t, mask_tm1, h_tm1, h_tm1_gate):
        '''
            Variable names follow the conventions from:
            http://deeplearning.net/software/theano/library/scan.html

        '''
        mask = T.nnet.sigmoid(T.dot(h_tm1_gate, self.G_gate) + self.Gb_gate)

        return mask*self.activation(x_t + T.dot(h_tm1, self.U)+h_tm1) + (1.-mask)*h_tm1, self.activation(T.dot(h_tm1, self.W_gate) + T.dot(h_tm1_gate, self.U_gate)+h_tm1_gate + self.b_gate)
    
    def get_output(self, train=False):
        X = self.get_input(train)  # shape: (nb_samples, time (padded with zeros), input_dim)
        # new shape: (time, nb_samples, input_dim) -> because theano.scan iterates over main dimension
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))
        x = T.dot(X, self.W) + self.b

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        outputs, updates = theano.scan(
            self._step,  # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=[x, dict(input=padded_mask, taps=[-1])],  # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=[T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                          T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.gate_dim), 1)],
            truncate_gradient=self.truncate_gradient)

        # TODO: saizheng fix this! we have 2 output sequences now!
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[0][-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SelectiveRNN(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(SelectiveRNN, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W = self.init((input_dim, self.output_dim))
        self.U = self.inner_init((self.output_dim, self.output_dim))
        #self.D = self.init((self.output_dim, self.output_dim))
        self.b = shared_zeros((self.output_dim))

        self.W_gate = self.init((self.output_dim, self.output_dim)) 
        self.U_gate = self.init((self.output_dim, self.output_dim))
        self.b_gate = shared_zeros((self.output_dim))
        #self.b_gate = sharedX(np.ones(self.output_dim)*50.)
        self.W_gate_aux = shared_zeros((self.output_dim, self.output_dim)) 
        self.U_gate_aux = shared_zeros((self.output_dim, self.output_dim))
        self.b_gate_aux = shared_zeros((self.output_dim))
        #self.th_gate = sharedX(0.05) 
        #self.U_ = T.dot(self.U, np.identity(self.output_dim).astype('float32')+self.D)
        self.params = [self.W, self.U, self.b] #+ 
        self.params_gate = [self.U_gate, self.W_gate, self.b_gate]
        self.params_gate_aux = [self.U_gate_aux, self.W_gate_aux, self.b_gate_aux]
        self.params_ = [self.W, self.b]
        self.m = sharedX(np.random.normal(size = (self.output_dim,)))

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def f(self, x):
        return T.switch(T.ge(x,0.),1.,0.)

    def _step(self, x_t, mask_tm1, h_tm1):
        '''
            Variable names follow the conventions from:
            http://deeplearning.net/software/theano/library/scan.html

        '''
        gate_ = T.dot(x_t, self.W_gate) + T.dot(h_tm1, self.U_gate) + self.b_gate
        #gate = T.nnet.sigmoid(gate_)
        gate = gate_#T.tanh(gate_)
        grad_gate = T.grad(T.sum(gate), gate_)
        gate_aux = grad_gate*(T.dot(x_t, self.W_gate_aux) + T.dot(h_tm1, self.U_gate_aux) + self.b_gate_aux)

        mask = gate_aux + self.f(gate)

        return mask*self.activation(x_t + T.dot(h_tm1, self.U)+0.95*h_tm1) + (1.-mask)*h_tm1 
    
    def get_output(self, train=False):
        X = self.get_input(train)  # shape: (nb_samples, time (padded with zeros), input_dim)
        # new shape: (time, nb_samples, input_dim) -> because theano.scan iterates over main dimension
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))
        x = T.dot(X, self.W) + self.b

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        outputs, updates = theano.scan(
            self._step,  # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=[x, dict(input=padded_mask, taps=[-1])],  # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=[T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(SimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class SimpleDeepRNN(Recurrent):
    '''
        Fully connected RNN where the output of multiple timesteps
        (up to "depth" steps in the past) is fed back to the input:

        output = activation( W.x_t + b + inner_activation(U_1.h_tm1) + inner_activation(U_2.h_tm2) + ... )

        This demonstrates how to build RNNs with arbitrary lookback.
        Also (probably) not a super useful model.
    '''
    def __init__(self, output_dim, depth=3,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.depth = depth
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(SimpleDeepRNN, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()
        self.W = self.init((input_dim, self.output_dim))
        self.Us = [self.inner_init((self.output_dim, self.output_dim)) for _ in range(self.depth)]
        self.b = shared_zeros((self.output_dim))
        self.params = [self.W] + self.Us + [self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, *args):
        o = x_t
        for i in range(self.depth):
            mask_tmi = args[i]
            h_tmi = args[i + self.depth]
            U_tmi = args[i + 2*self.depth]
            o += mask_tmi*self.inner_activation(T.dot(h_tmi, U_tmi))
        return self.activation(o)

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=self.depth)
        X = X.dimshuffle((1, 0, 2))

        x = T.dot(X, self.W) + self.b

        if self.depth == 1:
            initial = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
        else:
            initial = T.unbroadcast(T.unbroadcast(alloc_zeros_matrix(self.depth, X.shape[1], self.output_dim), 0), 2)

        outputs, updates = theano.scan(
            self._step,
            sequences=[x, dict(
                input=padded_mask,
                taps=[(-i) for i in range(self.depth)]
            )],
            outputs_info=[dict(
                initial=initial,
                taps=[(-i-1) for i in range(self.depth)]
            )],
            non_sequences=self.Us,
            truncate_gradient=self.truncate_gradient
        )

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "depth": self.depth,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(SimpleDeepRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GRU(Recurrent):
    '''
        Gated Recurrent Unit - Cho et al. 2014

        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            On the Properties of Neural Machine Translation: Encoder–Decoder Approaches
                http://www.aclweb.org/anthology/W14-4012
            Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
                http://arxiv.org/pdf/1412.3555v1.pdf
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(GRU, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_z = self.init((input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t + T.dot(h_mask_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = z * h_mask_tm1 + (1 - z) * hh_t
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(GRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RNN_bu(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
                 U_init = "identity",
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_bu, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            if way == "identity":
                return theano.shared(np.identity(self.output_dim).astype("float32"))
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))

        self.W1 = self.init((input_dim, self.output_dim))
        self.U1 = init_U() 
        self.V1 = theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))
        #self.V1 = self.inner_init((self.output_dim, self.output_dim))

        self.W2 = self.init((self.output_dim, self.output_dim))
        self.U2 = init_U() 
        self.V2 = theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))
        #self.V2 = self.inner_init((self.output_dim, self.output_dim))

        self.b1 = shared_zeros((self.output_dim))
        self.b2 = shared_zeros((self.output_dim))

        self.params = [self.W1, self.U1] +\
                      [self.W2, self.U2, self.V2] +\
                      [self.b1, self.b2]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h1_tm1, h2_tm1):
        h1 = self.activation(T.dot(x_t, self.W1) + T.dot(h1_tm1, self.U1) + self.b1)
        h2 = self.activation(T.dot(h1, self.W2) + T.dot(h2_tm1, self.U2) + T.dot(h1_tm1, self.V2) + self.b2)
        return h1, h2 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        [H1, H2], updates = theano.scan(
            self._step,  
            sequences=[X],
            outputs_info=[T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                          T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return H2.dimshuffle((1, 0, 2))
        return H2[-1]


class RNN_td(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
                 U_init = "identity",
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_td, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            if way == "identity":
                return theano.shared(np.identity(self.output_dim).astype("float32"))
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))

        self.W1 = self.init((input_dim, self.output_dim))
        self.U1 = init_U() 
        self.V1 = theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))
        #self.V1 = self.inner_init((self.output_dim, self.output_dim))

        self.W2 = self.init((self.output_dim, self.output_dim))
        self.U2 = init_U() 
        self.V2 = theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))
        #self.V2 = self.inner_init((self.output_dim, self.output_dim))

        self.b1 = shared_zeros((self.output_dim))
        self.b2 = shared_zeros((self.output_dim))

        self.params = [self.W1, self.U1, self.V1] +\
                      [self.W2, self.U2] +\
                      [self.b1, self.b2]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h1_tm1, h2_tm1):
        h1 = self.activation(T.dot(x_t, self.W1) + T.dot(h1_tm1, self.U1) + T.dot(h2_tm1, self.V1) + self.b1)
        h2 = self.activation(T.dot(h1, self.W2) + T.dot(h2_tm1, self.U2) + self.b2) 
        return h1, h2 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        [H1, H2], updates = theano.scan(
            self._step,  
            sequences=[X],
            outputs_info=[T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                          T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return H2.dimshuffle((1, 0, 2))
        return H2[-1]


class LSTM_bu(Recurrent):
    '''
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    '''
    def __init__(self, output_dim, 
                 init='uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid',
                 U_init = 'identity',
                 v_init = 0.1,
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.v_init = v_init
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim, self.input_dim_c)
        super(LSTM_bu, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            U_ = np.zeros((self.output_dim, self.output_dim * 6)).astype("float32")
            for k in xrange(6):
                if way == "identity":
                    U_[:, k*self.output_dim: (k+1)*self.output_dim] = np.identity(self.output_dim).astype("float32")*0.95
                if way == "orthogonal":
                    U = self.inner_init((self.output_dim, self.output_dim)).get_value()
                    U_[:, k*self.output_dim: (k+1)*self.output_dim] = U
                if way == "uniform":
                    U = self.init((self.output_dim, self.output_dim), self.v_init).get_value()
                    U_[:, k*self.output_dim: (k+1)*self.output_dim] = U
            return U_


        # the general LSTM connection looks like this:
        # | |
        # 0-0-
        # |X|
        # 0-0-
        # | |
        # so we need two hidden states.

        # U_1 is a big matrix for the low states:
        # U for hid-hid, W for in-hid, V for skew-top-down.
        # [ W_f_w, W_f_u, W_f_v, W_i, W_o, W_c;
        #   U_f_w, U_f_u, U_f_v, U_i, U_o, U_c; 
        #   V_f_w, V_f_u, V_f_v, V_i, V_o, V_c ]
        self.W1 = self.init((input_dim, self.output_dim * 6), self.v_init)
        self.U1 = self.init((self.output_dim, self.output_dim * 6), self.v_init)
        self.U1.set_value(init_U())

        self.V1 = self.init((self.output_dim, self.output_dim * 6), self.v_init)

        # U_2 is a big matrix for the high states:
        # U for hid-hid, W for in-hid, V for skew-bottom-up.
        # [ W_f_w, W_f_u, W_f_v, W_i, W_o, W_c;
        #   U_f_w, U_f_u, U_f_v, U_i, U_o, U_c; 
        #   V_f_w, V_f_u, V_f_v, V_i, V_o, V_c ]
        self.W2 = self.init((self.output_dim, self.output_dim * 6), self.v_init)
        self.U2 = self.init((self.output_dim, self.output_dim * 6), self.v_init)
        self.U2.set_value(init_U())
        self.V2 = self.init((self.output_dim, self.output_dim * 6), self.v_init)

        self.b1 = shared_zeros((self.output_dim*6))
        self.b2 = shared_zeros((self.output_dim*6))

        # initialize b so that b for U_f_w and V_f_w be -k,  U_f_u be 1.
        b = np.zeros((self.output_dim*6), dtype = "float32")
        b[0:3*self.output_dim] = -7*np.ones((3*self.output_dim), dtype = "float32")
        b[self.output_dim: 2*self.output_dim] = np.ones((self.output_dim), dtype = "float32")
        self.b1.set_value(b)
        self.b2.set_value(b)

        self.params = [self.W1, self.U1] + \
                      [self.W2, self.U2, self.V2] + \
                      [self.b1, self.b2]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h1_tm1, h2_tm1, c1_tm1, c2_tm1):

        #h1_tm1, c1_tm1 is lower layer states 
        #h2_tm1, c2_tm1 is higher layer states 

        g1 = self.b1 + T.dot(x_t, self.W1) + T.dot(h1_tm1, self.U1) #+ T.dot(h2_tm1, self.V1) 
        f1 = self.inner_activation(g1[:, 0 : 3*self.output_dim])
        i1 = self.inner_activation(g1[:, 3*self.output_dim : 4*self.output_dim])
        o1 = self.inner_activation(g1[:, 4*self.output_dim : 5*self.output_dim])

        # f_1 =[f_2_w, f_2_u, f_2_v]
        c1 = f1[:, 1*self.output_dim : 2*self.output_dim] * c1_tm1 + \
             i1 * self.activation(g1[:, 5*self.output_dim : 6*self.output_dim])
        h1 = o1 * self.activation(c1)

        g2 = self.b2 + T.dot(h1, self.W2) + T.dot(h2_tm1, self.U2) + T.dot(h1_tm1, self.V2)
        f2 = self.inner_activation(g2[:, 0 : 3*self.output_dim])
        i2 = self.inner_activation(g2[:, 3*self.output_dim : 4*self.output_dim])
        o2 = self.inner_activation(g2[:, 4*self.output_dim : 5*self.output_dim])

        # f_2 =[f_2_w, f_2_u, f_2_v]
        c2 = f2[:, 0 : self.output_dim] * c1 + \
             f2[:, 1*self.output_dim : 2*self.output_dim] * c2_tm1 + \
             f2[:, 2*self.output_dim : 3*self.output_dim] * c1_tm1 + \
             i2 * self.activation(g2[:, 5*self.output_dim : 6*self.output_dim])
        h2 = o2 * self.activation(c2)

        return h1, h2, c1, c2 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        #X_ = T.dot(X, self.W1) + self.b1
        [H1, H2, C1, C2], updates = theano.scan(
            self._step,
            sequences=[X],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
            ],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return H2.dimshuffle((1, 0, 2))
        return H2[-1]



class LSTM_td(Recurrent):
    '''
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    '''
    def __init__(self, output_dim, 
                 init='uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid',
                 U_init = 'identity',
                 v_init = 0.1,
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.v_init = v_init
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim, self.input_dim_c)
        super(LSTM_td, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            U_ = np.zeros((self.output_dim, self.output_dim * 6)).astype("float32")
            for k in xrange(6):
                if way == "identity":
                    U_[:, k*self.output_dim: (k+1)*self.output_dim] = np.identity(self.output_dim).astype("float32")*0.95
                if way == "orthogonal":
                    U = self.inner_init((self.output_dim, self.output_dim)).get_value()
                    U_[:, k*self.output_dim: (k+1)*self.output_dim] = U 
                if way == "uniform":
                    U = self.init((self.output_dim, self.output_dim), self.v_init).get_value()
                    U_[:, k*self.output_dim: (k+1)*self.output_dim] = U
            return U_

        # the general LSTM connection looks like this:
        # | |
        # 0-0-
        # |X|
        # 0-0-
        # | |
        # so we need two hidden states.

        # U_1 is a big matrix for the low states:
        # U for hid-hid, W for in-hid, V for skew-top-down.
        # [ W_f_w, W_f_u, W_f_v, W_i, W_o, W_c;
        #   U_f_w, U_f_u, U_f_v, U_i, U_o, U_c; 
        #   V_f_w, V_f_u, V_f_v, V_i, V_o, V_c ]
        self.W1 = self.init((input_dim, self.output_dim * 6), self.v_init)
        self.U1 = self.init((self.output_dim, self.output_dim * 6), self.v_init)
        self.U1.set_value(init_U())
        self.V1 = self.init((self.output_dim, self.output_dim * 6), self.v_init)

        # U_2 is a big matrix for the high states:
        # U for hid-hid, W for in-hid, V for skew-bottom-up.
        # [ W_f_w, W_f_u, W_f_v, W_i, W_o, W_c;
        #   U_f_w, U_f_u, U_f_v, U_i, U_o, U_c; 
        #   V_f_w, V_f_u, V_f_v, V_i, V_o, V_c ]
        self.W2 = self.init((self.output_dim, self.output_dim * 6), self.v_init)
        self.U2 = self.init((self.output_dim, self.output_dim * 6), self.v_init)
        self.U2.set_value(init_U())
        self.V2 = self.init((self.output_dim, self.output_dim * 6), self.v_init)

        self.b1 = shared_zeros((self.output_dim*6))
        self.b2 = shared_zeros((self.output_dim*6))

        # initialize b so that b for U_f_w and V_f_w be -k,  U_f_u be 1.
        b = np.zeros((self.output_dim*6), dtype = "float32")
        b[0:3*self.output_dim] = -7*np.ones((3*self.output_dim), dtype = "float32")
        b[self.output_dim: 2*self.output_dim] = np.ones((self.output_dim), dtype = "float32")
        self.b1.set_value(b)
        self.b2.set_value(b)

        self.params = [self.W1, self.U1, self.V1] + \
                      [self.W2, self.U2] + \
                      [self.b1, self.b2]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h1_tm1, h2_tm1, c1_tm1, c2_tm1):

        #h1_tm1, c1_tm1 is lower layer states 
        #h2_tm1, c2_tm1 is higher layer states 

        g1 = self.b1 + T.dot(x_t, self.W1) + T.dot(h1_tm1, self.U1) + T.dot(h2_tm1, self.V1) 
        f1 = self.inner_activation(g1[:, 0 : 3*self.output_dim])
        i1 = self.inner_activation(g1[:, 3*self.output_dim : 4*self.output_dim])
        o1 = self.inner_activation(g1[:, 4*self.output_dim : 5*self.output_dim])

        # f_1 =[f_2_w, f_2_u, f_2_v]
        c1 = f1[:, 1*self.output_dim : 2*self.output_dim] * c1_tm1 + \
             f1[:, 2*self.output_dim : 3*self.output_dim] * c2_tm1 + \
             i1 * self.activation(g1[:, 5*self.output_dim : 6*self.output_dim])
        h1 = o1 * self.activation(c1)

        g2 = self.b2 + T.dot(h1, self.W2) + T.dot(h2_tm1, self.U2) #+ T.dot(h1_tm1, self.V2)
        f2 = self.inner_activation(g2[:, 0 : 3*self.output_dim])
        i2 = self.inner_activation(g2[:, 3*self.output_dim : 4*self.output_dim])
        o2 = self.inner_activation(g2[:, 4*self.output_dim : 5*self.output_dim])

        # f_2 =[f_2_w, f_2_u, f_2_v]
        # f_2[:, 2*self.output_dim : 3*self.output_dim] * c1_tm1 + \
        c2 = f2[:, 0 : self.output_dim] * c1 + \
             f2[:, 1*self.output_dim : 2*self.output_dim] * c2_tm1 + \
             i2 * self.activation(g2[:, 5*self.output_dim : 6*self.output_dim])
        h2 = o2 * self.activation(c2)

        return h1, h2, c1, c2 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        #X_ = T.dot(X, self.W1) + self.b1
        [H1, H2, C1, C2], updates = theano.scan(
            self._step,
            sequences=[X],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
            ],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return H2.dimshuffle((1, 0, 2))
        return H2[-1]

class LSTMgrave(Recurrent):
    '''
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    '''
    def __init__(self, output_dim, batchsize, init_value = 0.01,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.batchsize = batchsize
        self.init_value = init_value
        self.H0 = theano.shared(np.zeros((self.batchsize, self.output_dim)).astype('float32'))
        self.C0 = theano.shared(np.zeros((self.batchsize, self.output_dim)).astype('float32'))

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(LSTMgrave, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_i = self.init((input_dim, self.output_dim), self.init_value)
        self.U_i = self.inner_init((self.output_dim, self.output_dim), self.init_value)
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((input_dim, self.output_dim), self.init_value)
        self.U_f = self.inner_init((self.output_dim, self.output_dim), self.init_value)
        self.b_f = theano.shared(1.*np.ones((self.output_dim,)).astype('float32'))#self.forget_bias_init((self.output_dim))

        self.W_c = self.init((input_dim, self.output_dim),self.init_value)
        self.U_c = self.inner_init((self.output_dim, self.output_dim),self.init_value)
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((input_dim, self.output_dim),self.init_value)
        self.U_o = self.inner_init((self.output_dim, self.output_dim),self.init_value)
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c):
        h_mask_tm1 = mask_tm1 * h_tm1
        c_mask_tm1 = mask_tm1 * c_tm1

        i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1, u_f))
        c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc, padded_mask],
            outputs_info=[self.H0, self.C0],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2)), memories.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class LSTM(Recurrent):
    '''
        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        For a step-by-step description of the algorithm, see:
        http://deeplearning.net/tutorial/lstm.html

        References:
            Long short-term memory (original 97 paper)
                http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
            Learning to forget: Continual prediction with LSTM
                http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015
            Supervised sequence labelling with recurrent neural networks
                http://www.cs.toronto.edu/~graves/preprint.pdf
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(LSTM, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = theano.shared(1.*np.ones((self.output_dim,)).astype('float32'))#self.forget_bias_init((self.output_dim))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c):
        h_mask_tm1 = mask_tm1 * h_tm1
        c_mask_tm1 = mask_tm1 * c_tm1

        i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1, u_f))
        c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc, padded_mask],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class JZS1(Recurrent):
    '''
        Evolved recurrent neural network architectures from the evaluation of thousands
        of models, serving as alternatives to LSTMs and GRUs. See Jozefowicz et al. 2015.

        This corresponds to the `MUT1` architecture described in the paper.

        Takes inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            An Empirical Exploration of Recurrent Network Architectures
                http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(JZS1, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_z = self.init((input_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        # P_h used to project X onto different dimension, using sparse random projections
        if input_dim == self.output_dim:
            self.Pmat = theano.shared(np.identity(self.output_dim, dtype=theano.config.floatX), name=None)
        else:
            P = np.random.binomial(1, 0.5, size=(input_dim, self.output_dim)).astype(theano.config.floatX) * 2 - 1
            P = 1 / np.sqrt(input_dim) * P
            self.Pmat = theano.shared(P, name=None)

        self.params = [
            self.W_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.U_h, self.b_h,
            self.Pmat
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t)
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = hh_t * z + h_mask_tm1 * (1 - z)
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.tanh(T.dot(X, self.Pmat)) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient)
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(JZS1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class JZS2(Recurrent):
    '''
        Evolved recurrent neural network architectures from the evaluation of thousands
        of models, serving as alternatives to LSTMs and GRUs. See Jozefowicz et al. 2015.

        This corresponds to the `MUT2` architecture described in the paper.

        Takes inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            An Empirical Exploration of Recurrent Network Architectures
                http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(JZS2, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_z = self.init((input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        # P_h used to project X onto different dimension, using sparse random projections
        if input_dim == self.output_dim:
            self.Pmat = theano.shared(np.identity(self.output_dim, dtype=theano.config.floatX), name=None)
        else:
            P = np.random.binomial(1, 0.5, size=(input_dim, self.output_dim)).astype(theano.config.floatX) * 2 - 1
            P = 1 / np.sqrt(input_dim) * P
            self.Pmat = theano.shared(P, name=None)

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
            self.Pmat
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t + T.dot(h_mask_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = hh_t * z + h_mask_tm1 * (1 - z)
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.Pmat) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient)
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(JZS2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class JZS3(Recurrent):
    '''
        Evolved recurrent neural network architectures from the evaluation of thousands
        of models, serving as alternatives to LSTMs and GRUs. See Jozefowicz et al. 2015.

        This corresponds to the `MUT3` architecture described in the paper.

        Takes inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            An Empirical Exploration of Recurrent Network Architectures
                http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(JZS3, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        self.W_z = self.init((input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t + T.dot(T.tanh(h_mask_tm1), u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = hh_t * z + h_mask_tm1 * (1 - z)
        return h_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h
        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h],
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "truncate_gradient": self.truncate_gradient,
                  "return_sequences": self.return_sequences,
                  "input_dim": self.input_dim,
                  "input_length": self.input_length}
        base_config = super(JZS3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
