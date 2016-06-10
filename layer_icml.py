# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
import keras

from keras import activations, initializations
from keras.utils.theano_utils import shared_scalar, shared_zeros, alloc_zeros_matrix, sharedX, shared_ones
from keras.layers.core import Layer, MaskedLayer
from keras.layers.recurrent import Recurrent
from six.moves import range
import pdb
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams

################ depth related  ###################

class RNN_bu(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='relu', weights=None,
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
                return theano.shared(np.identity(self.output_dim).astype("float32")*0.6)
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))
            if way == "uniform":
                return self.init((self.output_dim, self.output_dim))

        self.W1 = self.init((input_dim, self.output_dim))
        self.U1 = init_U() 
        #self.V1 = theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))
        self.V1 = self.init((self.output_dim, self.output_dim))

        self.W2 = self.init((self.output_dim, self.output_dim))
        self.U2 = init_U() 
        #self.V2 = theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))
        self.V2 = self.init((self.output_dim, self.output_dim))

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
                 init='glorot_uniform', inner_init='orthogonal', activation='relu', weights=None,
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
                return theano.shared(np.identity(self.output_dim).astype("float32")*0.6)
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))
            if way == "uniform":
                return self.init((self.output_dim, self.output_dim))

        self.W1 = self.init((input_dim, self.output_dim))
        self.U1 = init_U() 
        #self.V1 = theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))
        self.V1 = self.init((self.output_dim, self.output_dim))
        self.V1 = theano.shared(np.identity(self.output_dim).astype("float32")*0.1)

        self.W2 = self.init((self.output_dim, self.output_dim))
        self.U2 = init_U()
        #self.V2 = theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))
        #self.V2 = self.init((self.output_dim, self.output_dim))
        self.V2 = theano.shared(np.identity(self.output_dim).astype("float32")*0.7)

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
                 b_init = 0,
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
        self.b_init = b_init
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
        b[0:3*self.output_dim] = self.b_init*np.ones((3*self.output_dim), dtype = "float32")
        #b[self.output_dim: 2*self.output_dim] = np.ones((self.output_dim), dtype = "float32")
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
                 b_init = 0,
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
        self.b_init = b_init
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
        b[0:3*self.output_dim] = self.b_init*np.ones((3*self.output_dim), dtype = "float32")
        b[self.output_dim: 2*self.output_dim] = -1*np.ones((self.output_dim), dtype = "float32")
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






################ shallowness related  ###################


class LSTM_sh(Recurrent):
    def __init__(self, output_dim, shallowness, 
                 init='uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid',
                 U_init = 'uniform',
                 v_init = 0.01,
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.sh = shallowness
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
        super(LSTM_sh, self).__init__(**kwargs)

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

        # U_1 is a big matrix for the low states:
        # U for hid-hid, W for in-hid, V for skew-top-down.
        # [ W_f_w, W_f_u, W_f_v, W_i, W_o, W_c;
        #   U_f_w, U_f_u, U_f_v, U_i, U_o, U_c; 
        #   V_f_w, V_f_u, V_f_v, V_i, V_o, V_c ]
        self.W1 = self.init((input_dim, self.output_dim * 6), self.v_init)
        self.U1 = self.init((self.output_dim, self.output_dim * 6), self.v_init)
        self.U1.set_value(init_U())

        self.V1 = self.init((self.output_dim, self.output_dim * 6), self.v_init)
        self.b1 = shared_zeros((self.output_dim*6))

        # initialize b so that b for U_f_w and V_f_w be -k,  U_f_u be 1.
        b = np.zeros((self.output_dim*6), dtype = "float32")
        b[0:3*self.output_dim] = -1*np.ones((3*self.output_dim), dtype = "float32")
        b[self.output_dim: 2*self.output_dim] = 1*np.ones((self.output_dim), dtype = "float32")
        self.b1.set_value(b)

        self.params = [self.W1, self.U1, self.V1, self.b1]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h1_tm1, h1_tms, c1_tm1, c1_tms):

        #h1_tm1, c1_tm1 is lower layer states 
        #h2_tm1, c2_tm1 is higher layer states 

        g1 = self.b1 + T.dot(x_t, self.W1) + T.dot(h1_tm1, self.U1) + T.dot(h1_tms, self.V1) 
        f1 = self.inner_activation(g1[:, 0 : 3*self.output_dim])
        i1 = self.inner_activation(g1[:, 3*self.output_dim : 4*self.output_dim])
        o1 = self.inner_activation(g1[:, 4*self.output_dim : 5*self.output_dim])

        # f_1 =[f_2_w, f_2_u, f_2_v]
        c1 = f1[:, 1*self.output_dim : 2*self.output_dim] * c1_tm1 + \
             f1[:, 2*self.output_dim : 3*self.output_dim] * c1_tms +\
             i1 * self.activation(g1[:, 5*self.output_dim : 6*self.output_dim])
        h1 = o1 * self.activation(c1)

        return h1, c1 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        #X_ = T.dot(X, self.W1) + self.b1
        [H1, C1], updates = theano.scan(
            self._step,
            sequences=[X],
            outputs_info=[
                dict(initial = alloc_zeros_matrix(self.sh, X.shape[1], self.output_dim), taps = [-1, -self.sh]), 
                dict(initial = alloc_zeros_matrix(self.sh, X.shape[1], self.output_dim), taps = [-1, -self.sh]),       
                ],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return H1.dimshuffle((1, 0, 2))
        return H1[-1]


class LSTM_dp(Recurrent):
    def __init__(self, output_dim, depth, 
                 init='uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid',
                 U_init = 'uniform',
                 v_init = 0.01,
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.dp = depth
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
        super(LSTM_dp, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init, n = 6):
            U_ = np.zeros((self.output_dim, self.output_dim * n)).astype("float32")
            for k in xrange(n):
                if way == "identity":
                    U_[:, k*self.output_dim: (k+1)*self.output_dim] = np.identity(self.output_dim).astype("float32")*0.95
                if way == "orthogonal":
                    U = self.inner_init((self.output_dim, self.output_dim)).get_value()
                    U_[:, k*self.output_dim: (k+1)*self.output_dim] = U
                if way == "uniform":
                    U = self.init((self.output_dim, self.output_dim), self.v_init).get_value()
                    U_[:, k*self.output_dim: (k+1)*self.output_dim] = U
            return U_

        # U is a big matrix for all the hidden layers
        # for each hidden layer, U = [U_f, U_i, U_o, U_c]
        U = np.zeros((self.output_dim*(self.dp-1), self.output_dim*4)).astype('float32')
        for i in xrange(self.dp-1):
            U[i*self.output_dim:(i+1)*self.output_dim, :] = init_U(n=4)
        self.U = theano.shared(U)
        self.b = shared_zeros((self.dp-1, self.output_dim*4))
        b = np.zeros((self.dp-1, self.output_dim*4), dtype = "float32")
        #############important###########set b#########
        b[:, 0:3*self.output_dim] = 5*np.ones((self.dp-1, 3*self.output_dim), dtype = "float32")
        b[:, 0:1*self.output_dim] = -5*np.ones((self.dp-1, self.output_dim), dtype = "float32")
        self.b.set_value(b)

        # U_1 is a big matrix for the low states:
        # U for hid-hid, W for in-hid, V for skew-top-down.
        # [  W_f_u,  W_i, W_o, W_c;
        #    U_f_u,  U_i, U_o, U_c; 
        #    V_f_u,  V_i, V_o, V_c ]
        self.W1 = self.init((input_dim, self.output_dim *4), self.v_init)
        self.U1 = self.init((self.output_dim, self.output_dim * 4), self.v_init)
        self.U1.set_value(init_U(n=4))

        self.b1 = shared_zeros((self.output_dim*4))

        # initialize b so that b for U_f_w and V_f_w be -k,  U_f_u be 1.
        b = np.zeros((self.output_dim*4), dtype = "float32")
        b[0:self.output_dim] = np.ones((self.output_dim), dtype = "float32")
        self.b1.set_value(b)

        self.params = [self.U, self.b, self.W1, self.U1, self.b1]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h1_tm1, c1_tm1):
        h, c = h1_tm1, c1_tm1
        for i in xrange(self.dp-1):
            g = self.b[i] + T.dot(h, self.U[i*self.output_dim:(i+1)*self.output_dim, :])
            f = self.inner_activation(g[:, 0:self.output_dim])
            ii = self.inner_activation(g[:, self.output_dim : 2*self.output_dim])
            o = self.inner_activation(g[:, 2*self.output_dim : 3*self.output_dim])
            c = f*c + i*self.activation(g[:, 3*self.output_dim : 4*self.output_dim]) 
            h = o*self.activation(c)

        g1 = self.b1 + T.dot(x_t, self.W1) + T.dot(h, self.U1) 
        f1 = self.inner_activation(g1[:, 0 : self.output_dim])
        i1 = self.inner_activation(g1[:, self.output_dim : 2*self.output_dim])
        o1 = self.inner_activation(g1[:, 2*self.output_dim : 3*self.output_dim])

        # f_1 =[f_2_w, f_2_u, f_2_v]
        c1 = f1 * c + \
             i1 * self.activation(g1[:, 3*self.output_dim : 4*self.output_dim])
        h1 = o1 * self.activation(c1)

        return h1, c1 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        #X_ = T.dot(X, self.W1) + self.b1
        [H1, C1], updates = theano.scan(
            self._step,
            sequences=[X],
            outputs_info=[
                alloc_zeros_matrix(X.shape[1], self.output_dim), 
                alloc_zeros_matrix(X.shape[1], self.output_dim)],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return H1.dimshuffle((1, 0, 2))
        return H1[-1]


class RNN_shallow_grave(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim, batchsize, 
                 init='glorot_uniform', inner_init='orthogonal', activation='tanh', weights=None,
                 U_init = "orthogonal",
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
        self.batchsize = batchsize
        self.H0 = theano.shared(np.zeros((self.batchsize, self.output_dim)).astype('float32'))
        self.C0 = theano.shared(np.zeros((self.batchsize, self.output_dim)).astype('float32'))


        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_shallow_grave, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            if way == "identity":
                return theano.shared(np.identity(self.output_dim).astype("float32"))
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))
            if way == "uniform":
                return self.init((self.output_dim, self.output_dim))

        self.W = self.init((input_dim, self.output_dim), 0.5)
        self.U = init_U()#theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))

        self.b = shared_zeros((self.output_dim))

        self.params = [self.W, self.U, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h_tm1):
        h = self.activation(T.dot(x_t, self.W) + T.dot(h_tm1, self.U) + self.b)
        return h 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        H, updates = theano.scan(
            self._step,  
            sequences=[X],
            outputs_info=[self.H0])

        if self.return_sequences:
            return H.dimshuffle((1, 0, 2)), H.dimshuffle((1, 0, 2))
        return H[-1]




class RNN_shallow(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='tanh', weights=None,
                 U_init = "orthogonal",
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
        super(RNN_shallow, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            if way == "identity":
                return theano.shared(np.identity(self.output_dim).astype("float32"))
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))
            if way == "uniform":
                return self.init((self.output_dim, self.output_dim))

        self.W = self.init((input_dim, self.output_dim))
        self.U = init_U()#theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))

        self.b = shared_zeros((self.output_dim))

        self.params = [self.W, self.U, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h_tm1):
        h = self.activation(T.dot(x_t, self.W) + T.dot(h_tm1, self.U) + self.b)
        return h 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        H, updates = theano.scan(
            self._step,  
            sequences=[X],
            outputs_info=[alloc_zeros_matrix(X.shape[1], self.output_dim)])

        if self.return_sequences:
            return H.dimshuffle((1, 0, 2))
        return H[-1]



class RNN_sh(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='uniform', activation='tanh', weights=None,
                 U_init = "uniform",
                 shallowness = 2,
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.sh = shallowness
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_sh, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            if way == "identity":
                return theano.shared(np.identity(self.output_dim).astype("float32"))
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))
            if way == "uniform":
                return self.init((self.output_dim, self.output_dim))

        self.W = self.init((input_dim, self.output_dim))
        self.U = init_U()#theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))
        self.Us = init_U()#theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))

        self.b = shared_zeros((self.output_dim))

        self.params = [self.W, self.U, self.Us, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h_tm1, h_tms):
        h = self.activation(T.dot(x_t, self.W) + T.dot(h_tm1, self.U) + T.dot(h_tms, self.Us) + self.b)
        return h 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        H, updates = theano.scan(
            self._step,  
            sequences=[X],
            outputs_info=[dict(initial = alloc_zeros_matrix(self.sh, X.shape[1], self.output_dim), taps = [-1, -self.sh])])

        if self.return_sequences:
            return H.dimshuffle((1, 0, 2))
        return H[-1]


class RNN_dp_old(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='relu', weights=None,
                 U_init = "identity",
                 depth = 2,
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.dp = depth 
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_dp_old, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            U = np.zeros((self.output_dim, self.dp*self.output_dim)).astype("float32")
            for i in xrange(self.dp):
                if way == "identity":
                    U[:, i*self.output_dim: (i+1)*self.output_dim] = np.identity(self.output_dim).astype("float32")
                if way == "orthogonal":
                    U[:, i*self.output_dim: (i+1)*self.output_dim] = self.inner_init((self.output_dim, self.output_dim)).get_value()
                if way == "uniform":
                    U[:, i*self.output_dim: (i+1)*self.output_dim] =self.init((self.output_dim, self.output_dim)).get_value()
            return theano.shared(U)

        self.W = self.init((input_dim, self.output_dim))
        self.U = init_U()

        self.b = shared_zeros((self.dp * self.output_dim))

        self.params = [self.W, self.U, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h_tm1):
        h = self.activation(T.dot(x_t, self.W) + \
                            T.dot(h_tm1, self.U[:, 0:self.output_dim]) + \
                            self.b[0:self.output_dim])
        for i in xrange(1, self.dp):
            h = self.activation(T.dot(h, self.U[:, i*self.output_dim:(i+1)*self.output_dim]) + self.b[i*self.output_dim:(i+1)*self.output_dim])
        return h 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        H, updates = theano.scan(
            self._step,  
            sequences=[X],
            outputs_info=[T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return H.dimshuffle((1, 0, 2))
        return H[-1]


class RNN_dp(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='relu', weights=None,
                 U_init = "identity",
                 depth = 2,
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.dp = depth 
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_dp, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            U = np.zeros((self.output_dim, self.dp*self.output_dim)).astype("float32")
            for i in xrange(self.dp):
                if way == "identity":
                    U[:, i*self.output_dim: (i+1)*self.output_dim] = np.identity(self.output_dim).astype("float32")
                if way == "orthogonal":
                    U[:, i*self.output_dim: (i+1)*self.output_dim] = self.inner_init((self.output_dim, self.output_dim)).get_value()
                if way == "uniform":
                    U[:, i*self.output_dim: (i+1)*self.output_dim] =self.init((self.output_dim, self.output_dim)).get_value()
            return theano.shared(U)

        self.W = self.init((input_dim, self.output_dim))
        self.U = init_U()

        self.b = shared_zeros((self.dp * self.output_dim))

        self.params = [self.W, self.U, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h_tm1):
        h = h_tm1
        for i in xrange(0, self.dp-1):
            h = self.activation(T.dot(h, self.U[:, i*self.output_dim:(i+1)*self.output_dim]) + self.b[i*self.output_dim:(i+1)*self.output_dim])
        h = self.activation(T.dot(x_t, self.W) + \
                T.dot(h_tm1, self.U[:, (self.dp-1)*self.output_dim:]) + \
                self.b[(self.dp-1)*self.output_dim:])

        return h 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        H, updates = theano.scan(
            self._step,  
            sequences=[X],
            outputs_info=[T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return H.dimshuffle((1, 0, 2))
        return H[-1]






###################shallowness verification####################

class RNN_shxieshang(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='uniform', inner_init='uniform', activation='tanh', weights=None,
                 sh = 3,
                 U_init = "uniform",
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.sh = sh
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_shxieshang, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            if way == "identity":
                return theano.shared(np.identity(self.output_dim).astype("float32")*0.6)
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))
            if way == "uniform":
                return self.init((self.output_dim, self.output_dim))

        self.W1 = self.init((input_dim, self.output_dim))
        self.U1 = init_U() 

        self.W2 = self.init((self.output_dim, self.output_dim))
        self.U2 = init_U() 
        #self.V2 = theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))
        self.V2 = self.init((self.output_dim, self.output_dim))

        self.b1 = shared_zeros((self.output_dim))
        self.b2 = shared_zeros((self.output_dim))

        self.params = [self.W1, self.U1] +\
                      [self.W2, self.U2, self.V2] +\
                      [self.b1, self.b2]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h1_tm1, h1_tms, h2_tm1):
        h1 = self.activation(T.dot(x_t, self.W1) + T.dot(h1_tm1, self.U1) + self.b1)
        h2 = self.activation(T.dot(h1, self.W2) + T.dot(h2_tm1, self.U2) + T.dot(h1_tms, self.V2) + self.b2)
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
            outputs_info=[dict(initial = alloc_zeros_matrix(self.sh, X.shape[1], self.output_dim), taps = [-1, -self.sh]),
                          alloc_zeros_matrix(X.shape[1], self.output_dim)])

        if self.return_sequences:
            return H2.dimshuffle((1, 0, 2))
        return H2[-1]

class RNN_shping(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='uniform', inner_init='uniform', activation='tanh', weights=None,
                 sh = 3,
                 U_init = "uniform",
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.sh = sh
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_shping, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            if way == "identity":
                return theano.shared(np.identity(self.output_dim).astype("float32")*0.6)
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))
            if way == "uniform":
                return self.init((self.output_dim, self.output_dim))

        self.W1 = self.init((input_dim, self.output_dim))
        self.U1 = init_U() 

        self.W2 = self.init((self.output_dim, self.output_dim))
        self.U2 = init_U() 
        self.V2 = self.init((self.output_dim, self.output_dim))
        #self.V2 = theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))

        self.b1 = shared_zeros((self.output_dim))
        self.b2 = shared_zeros((self.output_dim))

        self.params = [self.W1, self.U1] +\
                      [self.W2, self.U2, self.V2] +\
                      [self.b1, self.b2]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h1_tm1, h2_tm1, h2_tms):
        h1 = self.activation(T.dot(x_t, self.W1) + T.dot(h1_tm1, self.U1) + self.b1)
        h2 = self.activation(T.dot(h1, self.W2) + T.dot(h2_tm1, self.U2) + T.dot(h2_tms, self.V2) + self.b2)
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
            outputs_info=[alloc_zeros_matrix(X.shape[1], self.output_dim),
                          dict(initial = alloc_zeros_matrix(self.sh, X.shape[1], self.output_dim), taps = [-1, -self.sh])])

        if self.return_sequences:
            return H2.dimshuffle((1, 0, 2))
        return H2[-1]

class RNN_shxiexia(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='uniform', inner_init='uniform', activation='tanh', weights=None,
                 sh = 3,
                 U_init = "uniform",
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.sh = sh
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_shxiexia, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            if way == "identity":
                return theano.shared(np.identity(self.output_dim).astype("float32")*0.6)
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))
            if way == "uniform":
                return self.init((self.output_dim, self.output_dim))

        self.W1 = self.init((input_dim, self.output_dim))
        self.U1 = init_U() 
        self.V1 = self.init((self.output_dim, self.output_dim))

        self.W2 = self.init((self.output_dim, self.output_dim))
        self.U2 = init_U() 
        #self.V2 = theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))

        self.b1 = shared_zeros((self.output_dim))
        self.b2 = shared_zeros((self.output_dim))

        self.params = [self.W1, self.U1, self.V1] +\
                      [self.W2, self.U2] +\
                      [self.b1, self.b2]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h1_tm1, h2_tm1, h2_tms):
        h1 = self.activation(T.dot(x_t, self.W1) + T.dot(h1_tm1, self.U1) + T.dot(h2_tms, self.V1) + self.b1)
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
            outputs_info=[alloc_zeros_matrix(X.shape[1], self.output_dim),
                          dict(initial = alloc_zeros_matrix(self.sh, X.shape[1], self.output_dim), taps = [-1, -self.sh])])

        if self.return_sequences:
            return H2.dimshuffle((1, 0, 2))
        return H2[-1]

class RNN_relugate(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='relu', weights=None,
                 U_init = "identity",
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.init_gate = initializations.get('uniform')
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.activation_gate = activations.get('tanh')
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_relugate, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            if way == "identity":
                return theano.shared(0.1*np.identity(self.output_dim).astype("float32"))
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))
            if way == "uniform":
                return self.init((self.output_dim, self.output_dim))

        self.W = self.init((input_dim, self.output_dim), 0.001)
        self.W_gate = self.init((input_dim, self.output_dim), 0.001)
        self.U = init_U()#theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))
        self.U_gate = self.init_gate((self.output_dim, self.output_dim), 0.001)

        self.b = shared_zeros((self.output_dim))
        self.b_gate = shared_zeros((self.output_dim))

        self.params = [self.W, self.W_gate, self.U, self.U_gate, self.b, self.b_gate]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h_tm1):
        h_gate = self.activation_gate(T.dot(x_t, self.W_gate) + T.dot(h_tm1, self.U_gate) + self.b_gate)
        h = self.activation(T.dot(x_t, self.W) + T.dot(h_tm1, self.U) + self.b)
        return h*h_gate 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        H, updates = theano.scan(
            self._step,  
            sequences=[X],
            outputs_info=[alloc_zeros_matrix(X.shape[1], self.output_dim)])

        if self.return_sequences:
            return H.dimshuffle((1, 0, 2))
        return H[-1]

class RNN_ens(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='tanh', weights=None,
                 U_init = "uniform",
                 shallowness = 2,
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.sh = shallowness
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_ens, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            if way == "identity":
                return theano.shared(np.identity(self.output_dim).astype("float32"))
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))
            if way == "uniform":
                return self.init((self.output_dim, self.output_dim))

        self.W = self.init((input_dim, self.output_dim))
        self.U = theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))
        self.U1 = init_U()#theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))
        self.U2 = init_U()#theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))
        self.Us = theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))

        self.b = shared_zeros((self.output_dim))
        self.b2 = shared_zeros((self.output_dim))

        self.params = [self.W, self.U1, self.U2, self.Us, self.b, self.b2]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h_tm1, h_tms):
        hid = T.dot(self.activation(T.dot(h_tm1, self.U1) + self.b2), self.U2)
        #h = self.activation(T.dot(x_t, self.W) + hid + T.dot(h_tm1, self.U) + T.dot(h_tms, self.Us) + self.b)
        h = self.activation(T.dot(x_t, self.W) + hid + T.dot(h_tms, self.Us) + self.b)
        return h 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        H, updates = theano.scan(
            self._step,  
            sequences=[X],
            outputs_info=[dict(initial = alloc_zeros_matrix(self.sh, X.shape[1], self.output_dim), taps = [-1, -self.sh])])

        if self.return_sequences:
            return H.dimshuffle((1, 0, 2))
        return H[-1]

class RNN_2tanh(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='tanh', activation2 = 'tanh', weights=None,
                 U_init = "uniform", gate_num = 1, noise = False, 
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.activation2 = activations.get(activation2)
        self.gate_num = gate_num
        self.noise = noise
        self.RNG = RandomStreams(max(np.random.RandomState(1364).randint(2 ** 15), 1))
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_2tanh, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            if way == "identity":
                return theano.shared(np.identity(self.output_dim).astype("float32"))
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))
            if way == "uniform":
                return self.init((self.output_dim, self.output_dim))

        self.W1 = self.init((input_dim, self.output_dim))
        self.U1 = init_U()
        self.b1 = theano.shared(0.*np.ones((self.output_dim,)).astype('float32'))
        self.b2 = theano.shared(0.*np.ones((self.output_dim,)).astype('float32'))

        self.W = []
        self.U = []
        self.b = []
        for i in xrange(self.gate_num):
            self.W.append(self.init((input_dim, self.output_dim), 0.01))
            self.U.append(init_U())
            self.b.append(theano.shared(0.*np.ones((self.output_dim,)).astype('float32')))
        self.params = [self.U1, self.b1, self.b2] + self.W + self.b

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h_tm1):
        h1 = T.dot(h_tm1, self.U1) + self.b1
        for i in xrange(self.gate_num):
            h1 = h1*(T.dot(x_t, self.W[i]) + self.b[i])
        return self.activation(h1 - self.b2) 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))
        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        H, updates = theano.scan(
            self._step,  
            sequences=[X],
            outputs_info=[alloc_zeros_matrix(X.shape[1], self.output_dim)])
        if self.return_sequences:
            return H.dimshuffle((1, 0, 2))
        return H[-1]


class RNN_ntanh(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', activation='tanh', activation2 = 'tanh', weights=None,
                 U_init = "uniform", tanh_n = 2, 
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.activation2 = activations.get(activation2)
        self.tanh_n = tanh_n 
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_ntanh, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            if way == "identity":
                return theano.shared(np.identity(self.output_dim).astype("float32"))
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))
            if way == "uniform":
                return self.init((self.output_dim, self.output_dim))

        self.W = []
        self.U = []
        self.b = []
        self.b2 = theano.shared(0.*np.ones((self.output_dim,)).astype('float32')) 
        for i in xrange(self.tanh_n):
            self.W.append(self.init((input_dim, self.output_dim)))
            self.U.append(init_U())
            self.b.append(theano.shared(0.*np.ones((self.output_dim,)).astype('float32')))

        self.params = self.W + self.U + self.b 

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h_tm1):
        h = T.dot(x_t, self.W[0]) + T.dot(h_tm1, self.U[0]) + self.b[0]
        for i in xrange(1, self.tanh_n):
            h = h*(T.dot(x_t, self.W[i]) + T.dot(h_tm1, self.U[i]) + self.b[i])
        return self.activation(h + self.b2)

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        H, updates = theano.scan(
            self._step,  
            sequences=[X],
            outputs_info=[alloc_zeros_matrix(X.shape[1], self.output_dim)])

        if self.return_sequences:
            return H.dimshuffle((1, 0, 2))
        return H[-1]


class RNN_multidp(Recurrent):
    def __init__(self, output_dim,
                 init='uniform', inner_init='uniform', activation='tanh', weights=None,
                 U_init = "uniform",
                 depth = 2,
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.dp = depth 
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_multidp, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            U = np.zeros((self.output_dim, self.dp*self.output_dim)).astype("float32")
            for i in xrange(self.dp):
                if way == "identity":
                    U[:, i*self.output_dim: (i+1)*self.output_dim] = np.identity(self.output_dim).astype("float32")
                if way == "orthogonal":
                    U[:, i*self.output_dim: (i+1)*self.output_dim] = self.inner_init((self.output_dim, self.output_dim)).get_value()
                if way == "uniform":
                    U[:, i*self.output_dim: (i+1)*self.output_dim] =self.init((self.output_dim, self.output_dim)).get_value()
            return theano.shared(U)

        self.W = self.init((input_dim, self.output_dim))
        self.bw = theano.shared(0.*np.ones((self.output_dim,)).astype('float32')) 
        self.bb = theano.shared(-0.*np.ones((self.output_dim,)).astype('float32'))
        self.U = init_U()

        self.b = theano.shared(0.*np.ones((self.dp*self.output_dim,)).astype('float32'))

        self.params = [self.W, self.U, self.b, self.bw, self.bb]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h_tm1):
        for i in xrange(0, self.dp-1):
            h = self.activation(T.dot(h_tm1, self.U[:, i*self.output_dim:(i+1)*self.output_dim]) + self.b[i*self.output_dim:(i+1)*self.output_dim])
        h = self.activation((T.dot(x_t, self.W)+self.bw)*(T.dot(h, self.U[:, (self.dp-1)*self.output_dim:]) + self.b[(self.dp-1)*self.output_dim:])+self.bb)

        return h 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        H, updates = theano.scan(
            self._step,  
            sequences=[X],
            outputs_info=[T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return H.dimshuffle((1, 0, 2))
        return H[-1]


class LSTM_multi(Recurrent):
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
        super(LSTM_multi, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()
        r = 0.5
        bf = 0.25#0.#0.25#1.0
        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = theano.shared(r*np.ones((self.output_dim)).astype('float32'))
        self.b_i1 = theano.shared(r*np.ones((self.output_dim)).astype('float32'))
        self.b_i2 = theano.shared(-r**2*np.ones((self.output_dim)).astype('float32'))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = theano.shared(r*np.ones((self.output_dim)).astype('float32'))
        self.b_f1 =  theano.shared(r*np.ones((self.output_dim)).astype('float32'))
        self.b_f2 =  theano.shared((bf-r**2)*np.ones((self.output_dim)).astype('float32'))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = theano.shared(r*np.ones((self.output_dim)).astype('float32'))
        self.b_c1 =  theano.shared(r*np.ones((self.output_dim)).astype('float32'))
        self.b_c2 =  theano.shared(-r**2*np.ones((self.output_dim)).astype('float32'))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = theano.shared(r*np.ones((self.output_dim)).astype('float32'))
        self.b_o1 =  theano.shared(r*np.ones((self.output_dim)).astype('float32'))
        self.b_o2 =  theano.shared(-r**2*np.ones((self.output_dim)).astype('float32'))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ] + [ self.b_i1, self.b_i2,
              self.b_f1, self.b_f2,
              self.b_c1, self.b_c2,
              self.b_o1, self.b_o2]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c):
        h_mask_tm1 = mask_tm1 * h_tm1
        c_mask_tm1 = mask_tm1 * c_tm1

        i_t = self.inner_activation(xi_t *(T.dot(h_mask_tm1, u_i) + self.b_i1) + self.b_i2)
        f_t = self.inner_activation(xf_t * (T.dot(h_mask_tm1, u_f) + self.b_f1) +self.b_f2)
        c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t * (T.dot(h_mask_tm1, u_c) + self.b_c1) + self.b_c2)
        o_t = self.inner_activation(xo_t * (T.dot(h_mask_tm1, u_o) +self.b_o1) + self.b_o2)
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

class LSTM_u(Recurrent):
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
        super(LSTM_u, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()
        r_i = 1.0
        r_f = 1. 
        r = 0.5
        r_b = 0.
        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_ii = theano.shared(r_i*np.ones((self.output_dim)).astype('float32'))
        self.b_i = theano.shared(r_b*np.ones((self.output_dim)).astype('float32'))
        self.b_i1 = theano.shared(r*np.ones((self.output_dim)).astype('float32'))
        self.b_i2 = theano.shared(r*np.ones((self.output_dim)).astype('float32'))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_ff = theano.shared(r_i*np.ones((self.output_dim)).astype('float32'))
        self.b_f = theano.shared(r_f*np.ones((self.output_dim)).astype('float32'))
        self.b_f1 = theano.shared(r*np.ones((self.output_dim)).astype('float32'))
        self.b_f2 = theano.shared(r*np.ones((self.output_dim)).astype('float32'))


        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_cc = theano.shared(r_i*np.ones((self.output_dim)).astype('float32'))
        self.b_c = theano.shared(r_b*np.ones((self.output_dim)).astype('float32'))
        self.b_c1 = theano.shared(r*np.ones((self.output_dim)).astype('float32'))
        self.b_c2 = theano.shared(r*np.ones((self.output_dim)).astype('float32'))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_oo = theano.shared(r_i*np.ones((self.output_dim)).astype('float32'))
        self.b_o = theano.shared(r_b*np.ones((self.output_dim)).astype('float32'))
        self.b_o1 = theano.shared(r*np.ones((self.output_dim)).astype('float32'))
        self.b_o2 = theano.shared(r*np.ones((self.output_dim)).astype('float32'))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ] + [ self.b_ii, self.b_i1, self.b_i2,
              self.b_ff, self.b_f1, self.b_f2,
              self.b_cc, self.b_c1, self.b_c2,
              self.b_oo, self.b_o1, self.b_o2]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c):
        h_mask_tm1 = mask_tm1 * h_tm1
        c_mask_tm1 = mask_tm1 * c_tm1

        hi_t = T.dot(h_mask_tm1, u_i)
        i_t = self.inner_activation(self.b_ii*xi_t*hi_t + xi_t*self.b_i1  + hi_t*self.b_i2 + self.b_i)
        hf_t = T.dot(h_mask_tm1, u_f)
        f_t = self.inner_activation(self.b_ff*xf_t * hf_t + xf_t*self.b_f1 + hf_t*self.b_f2 + self.b_f)
        hc_t = T.dot(h_mask_tm1, u_c)
        c_t = f_t * c_mask_tm1 + i_t * self.activation(self.b_cc*xc_t*hc_t + xc_t*self.b_c1 + hc_t*self.b_c2 + self.b_c)
        ho_t = T.dot(h_mask_tm1, u_o)
        o_t = self.inner_activation(self.b_oo*xo_t * ho_t +xo_t*self.b_o1 + ho_t*self.b_o2 + self.b_o)
        h_t = o_t * self.activation(c_t)
        return h_t, c_t, f_t

    def get_output(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        xi = T.dot(X, self.W_i)
        xf = T.dot(X, self.W_f)
        xc = T.dot(X, self.W_c)
        xo = T.dot(X, self.W_o)

        [outputs, memories, ff], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc, padded_mask],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1), None
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2)), memories.dimshuffle((1, 0, 2)), ff.dimshuffle((1, 0, 2)) 
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

class LSTM_uu(Recurrent):
    def __init__(self, output_dim,
                 rr = 1,
                 rf = 1,
                 r = 1,
                 init='uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid',
                 U_init = 'uniform',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.tanh = activations.get('tanh')
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.rr = rr
        self.rf = rf
        self.r = r
        self.input_dim = input_dim
        self.input_length = input_length

        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim, self.input_dim_c)
        super(LSTM_uu, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            U_ = np.zeros((self.output_dim, self.output_dim * 4)).astype("float32")
            for k in xrange(4):
                if way == "identity":
                    U_[:, k*self.output_dim: (k+1)*self.output_dim] = np.identity(self.output_dim).astype("float32")*0.95
                if way == "orthogonal":
                    U = self.inner_init((self.output_dim, self.output_dim)).get_value()
                    U_[:, k*self.output_dim: (k+1)*self.output_dim] = U
                if way == "uniform":
                    U = self.init((self.output_dim, self.output_dim)).get_value()
                    U_[:, k*self.output_dim: (k+1)*self.output_dim] = U
            return U_

        # U_1 is a big matrix for the low states:
        # U for hid-hid, W for in-hid, V for skew-top-down.
        # [ W_f, W_i, W_o, W_c;
        #   U_f, U_i, U_o, U_c;] 
        self.W1 = self.init((input_dim, self.output_dim * 4))
        self.U1 = self.init((self.output_dim, self.output_dim * 4))
        self.U1.set_value(init_U())

        self.bb = theano.shared(self.rr*np.ones((self.output_dim*4)).astype('float32'))
        self.bf = theano.shared(self.rf*np.ones((self.output_dim)).astype('float32'))
        self.b1  = theano.shared(self.r*np.ones((self.output_dim*4)).astype('float32'))
        self.b2  = theano.shared(self.r*np.ones((self.output_dim*4)).astype('float32'))
        self.b0 = theano.shared(0.*np.ones((self.output_dim*4)).astype('float32'))

        # initialize b so that b for U_f_w and V_f_w be -k,  U_f_u be 1.
        self.params = [self.W1, self.U1, self.bb, self.bf, self.b1, self.b2, self.b0]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h1_tm1, c1_tm1):
        x1 = T.dot(x_t, self.W1)
        h1 = T.dot(h1_tm1, self.U1)
        g1 = self.b0 + self.b1*x1 + self.b2*h1 + self.bb*x1*h1 
        f1 = self.inner_activation(g1[:, 0 : self.output_dim] + self.bf)
        i1 = self.inner_activation(g1[:, 1*self.output_dim : 2*self.output_dim])
        #i1 = self.tanh(g1[:, 1*self.output_dim : 2*self.output_dim])
        o1 = self.inner_activation(g1[:, 2*self.output_dim : 3*self.output_dim])

        # f_1 =[f_2_w, f_2_u, f_2_v]
        c1 = f1 * c1_tm1 + \
             i1 * self.activation(g1[:, 3*self.output_dim : 4*self.output_dim])
        h1 = o1 * self.activation(c1)

        return h1, c1 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        #X_ = T.dot(X, self.W1) + self.b1
        [H1, C1], updates = theano.scan(
            self._step,
            sequences=[X],
            outputs_info=[
                alloc_zeros_matrix(X.shape[1], self.output_dim),
                alloc_zeros_matrix(X.shape[1], self.output_dim)
                ],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return H1.dimshuffle((1, 0, 2))
        return H1[-1]


class LSTM_uugrave(Recurrent):
    def __init__(self, output_dim, batchsize,
                 init='uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='hard_sigmoid',
                 U_init = 'uniform',
                 weights=None, truncate_gradient=-1, return_sequences=False,
                 input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.tanh = activations.get('tanh')
        self.inner_activation = activations.get(inner_activation)
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.input_dim = input_dim
        self.input_length = input_length
        self.batchsize = batchsize
        self.H0 = theano.shared(np.zeros((self.batchsize, self.output_dim)).astype('float32'))
        self.C0 = theano.shared(np.zeros((self.batchsize, self.output_dim)).astype('float32'))

        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim, self.input_dim_c)
        super(LSTM_uugrave, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            U_ = np.zeros((self.output_dim, self.output_dim * 4)).astype("float32")
            for k in xrange(4):
                if way == "identity":
                    U_[:, k*self.output_dim: (k+1)*self.output_dim] = np.identity(self.output_dim).astype("float32")*0.95
                if way == "orthogonal":
                    U = self.inner_init((self.output_dim, self.output_dim)).get_value()
                    U_[:, k*self.output_dim: (k+1)*self.output_dim] = U
                if way == "uniform":
                    U = self.init((self.output_dim, self.output_dim)).get_value()
                    U_[:, k*self.output_dim: (k+1)*self.output_dim] = U
            return U_

        # U_1 is a big matrix for the low states:
        # U for hid-hid, W for in-hid, V for skew-top-down.
        # [ W_f, W_i, W_o, W_c;
        #   U_f, U_i, U_o, U_c;] 
        self.W1 = self.init((input_dim, self.output_dim * 4))
        self.U1 = self.init((self.output_dim, self.output_dim * 4))
        self.U1.set_value(init_U())

        self.bb = theano.shared(1.*np.ones((self.output_dim*4)).astype('float32'))
        self.bf = theano.shared(1.*np.ones((self.output_dim)).astype('float32'))
        self.b1  = theano.shared(0.5*np.ones((self.output_dim*4)).astype('float32'))
        self.b2  = theano.shared(0.5*np.ones((self.output_dim*4)).astype('float32'))
        self.b0 = theano.shared(0.*np.ones((self.output_dim*4)).astype('float32'))

        # initialize b so that b for U_f_w and V_f_w be -k,  U_f_u be 1.
        self.params = [self.W1, self.U1, self.bb, self.bf, self.b1, self.b2, self.b0]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h1_tm1, c1_tm1):
        x1 = T.dot(x_t, self.W1)
        h1 = T.dot(h1_tm1, self.U1)
        g1 = self.b0 + self.b1*x1 + self.b2*h1 + self.bb*x1*h1 
        f1 = self.inner_activation(g1[:, 0 : self.output_dim] + self.bf)
        i1 = self.inner_activation(g1[:, 1*self.output_dim : 2*self.output_dim])
        #i1 = self.tanh(g1[:, 1*self.output_dim : 2*self.output_dim])
        o1 = self.inner_activation(g1[:, 2*self.output_dim : 3*self.output_dim])

        # f_1 =[f_2_w, f_2_u, f_2_v]
        c1 = f1 * c1_tm1 + \
             i1 * self.activation(g1[:, 3*self.output_dim : 4*self.output_dim])
        h1 = o1 * self.activation(c1)

        return h1, c1 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        #X_ = T.dot(X, self.W1) + self.b1
        [H1, C1], updates = theano.scan(
            self._step,
            sequences=[X],
            outputs_info=[self.H0, self.C0],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return H1.dimshuffle((1, 0, 2)), C1.dimshuffle((1,0,2))
        return H1[-1]

class RNN_utanh_grave(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim, batchsize,
                 init='glorot_uniform', inner_init='orthogonal', activation='tanh', weights=None,
                 U_init = "uniform",
                 init_value = 0.01,
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.RNG = RandomStreams(max(np.random.RandomState(1364).randint(2 ** 15), 1))
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.init_value = init_value
        self.input_dim = input_dim
        self.input_length = input_length
        self.batchsize = batchsize
        self.H0 = theano.shared(np.zeros((self.batchsize, self.output_dim)).astype('float32'))
        self.C0 = theano.shared(np.zeros((self.batchsize, self.output_dim)).astype('float32'))

        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_utanh_grave, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            if way == "identity":
                return theano.shared(np.identity(self.output_dim).astype("float32"))
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))
            if way == "uniform":
                return self.init((self.output_dim, self.output_dim), self.init_value)

        self.W = self.init((input_dim, self.output_dim), self.init_value)
        self.U = init_U()
        r =0. 
        r_bb = 1.
        r_b = 0.
        self.b1 = theano.shared(r*np.ones((self.output_dim,)).astype('float32'))
        self.b2 = theano.shared(r*np.ones((self.output_dim,)).astype('float32'))
        self.bb = theano.shared(r_bb*np.ones((self.output_dim,)).astype('float32'))
        self.b = theano.shared(r_b*np.ones((self.output_dim,)).astype('float32'))


        self.params = [self.W, self.U, self.bb, self.b1, self.b2, self.b] 

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h_tm1):
        h_tmp = T.dot(h_tm1, self.U)
        x_tmp = T.dot(x_t, self.W)
        return self.activation(self.bb*h_tmp*x_tmp + self.b1*x_tmp + self.b2*h_tmp + self.b)

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))
        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        H, updates = theano.scan(
            self._step,  
            sequences=[X],
            outputs_info=[self.H0])
        if self.return_sequences:
            return H.dimshuffle((1, 0, 2)), H.dimshuffle((1, 0, 2)) 
        return H[-1]


class RNN_utanh(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 r_bb = 1,
                 r = 0,
                 r_b = 0,
                 init_u = 0.02,
                 init_w = 0.02,
                 a_fixed = False,
                 b_fixed = False,
                 init='glorot_uniform', inner_init='orthogonal', activation='tanh', weights=None,
                 U_init = "uniform",  
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.RNG = RandomStreams(max(np.random.RandomState(1364).randint(2 ** 15), 1))
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.input_dim = input_dim
        self.input_length = input_length
        self.r_bb = r_bb
        self.r = r
        self.r_b = r_b
        self.init_u = init_u
        self.init_w = init_w
        self.a_fixed = a_fixed
        self.b_fixed = b_fixed
        self.dummy = theano.shared(np.zeros((50, 50, self.output_dim)).astype('float32'))
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_utanh, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            if way == "identity":
                return theano.shared(np.identity(self.output_dim).astype("float32"))
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))
            if way == "uniform":
                return self.init((self.output_dim, self.output_dim), self.init_u)

        self.W = self.init((input_dim, self.output_dim), self.init_w)
        self.U = init_U()
        r =self.r 
        r_bb = self.r_bb
        r_b = self.r_b
        self.b1 = theano.shared(r*np.ones((self.output_dim,)).astype('float32'))
        self.b2 = theano.shared(r*np.ones((self.output_dim,)).astype('float32'))
        self.bb = theano.shared(r_bb*np.ones((self.output_dim,)).astype('float32'))
        self.b = theano.shared(r_b*np.ones((self.output_dim,)).astype('float32'))
        self.params = [self.W, self.U, self.b] 
        if not self.a_fixed:
            self.params += [self.bb] 
        if not self.b_fixed:
            self.params += [self.b1, self.b2]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, dummy, h_tm1):
        h_tmp = T.dot(h_tm1, self.U)
        x_tmp = T.dot(x_t, self.W)
        h = self.activation(self.bb*h_tmp*x_tmp + self.b1*x_tmp + self.b2*h_tmp + self.b)
        return dummy + h

    def get_output(self, dummy, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))
        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        H, updates = theano.scan(
            self._step,  
            sequences=[X,dummy],
            outputs_info=[alloc_zeros_matrix(X.shape[1], self.output_dim)])#, None, None, None,  None, None])
        if self.return_sequences:
            return H.dimshuffle((1, 0, 2))#, H[1].dimshuffle((1, 0, 2)), H[2].dimshuffle((1, 0, 2)), H[3].dimshuffle((1, 0, 2)), H[4].dimshuffle((1, 0, 2)), H[5].dimshuffle((1, 0, 2))
        return H[-1]




class RNN_utanhlinear(Recurrent):
    '''
        Fully connected RNN where output is to fed back to input.

        Not a particularly useful model,
        included for demonstration purposes
        (demonstrates how to use theano.scan to build a basic RNN).
    '''
    def __init__(self, output_dim,
                 r_bb = 1,
                 r = 0,
                 r_b = 0,
                 init='glorot_uniform', inner_init='orthogonal', activation='tanh', weights=None,
                 U_init = "uniform",  
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.RNG = RandomStreams(max(np.random.RandomState(1364).randint(2 ** 15), 1))
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.input_dim = input_dim
        self.input_length = input_length
        self.r_bb = r_bb
        self.r = r
        self.r_b = r_b
        #self.dummy_tensor = theano.shared(np.zeros((50, 50, self.output_dim)).astype('float32'))
        #self.dummy_index = 0
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_utanhlinear, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            if way == "identity":
                return theano.shared(np.identity(self.output_dim).astype("float32"))
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))
            if way == "uniform":
                return self.init((self.output_dim, self.output_dim))

        self.W = self.init((input_dim, self.output_dim))
        self.U = init_U()
        r =self.r 
        r_bb = self.r_bb
        r_b = self.r_b
        self.b1 = theano.shared(r*np.ones((self.output_dim,)).astype('float32'))
        self.b2 = theano.shared(r*np.ones((self.output_dim,)).astype('float32'))
        self.bb = theano.shared(r_bb*np.ones((self.output_dim,)).astype('float32'))
        self.b = theano.shared(r_b*np.ones((self.output_dim,)).astype('float32'))

        self.params = [self.W, self.U, self.bb, self.b1, self.b2, self.b] 

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h_tm1):
        h_tmp = T.dot(h_tm1, self.U)
        x_tmp = T.dot(x_t, self.W)
        turb = 1.
        added = 0.0
        h = self.bb*h_tmp*x_tmp*turb + self.b1*x_tmp + self.b2*h_tmp + self.b + added #+ self.dummy_tensor[self.dummy_index]
        #self.dummy_index += 1
        return h

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))
        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        H, updates = theano.scan(
            self._step,  
            sequences=[X],
            outputs_info=[alloc_zeros_matrix(X.shape[1], self.output_dim)])#, None, None, None,  None, None])
        #self.dummy_index = 0
        if self.return_sequences:
            return H.dimshuffle((1, 0, 2))#, H[1].dimshuffle((1, 0, 2)), H[2].dimshuffle((1, 0, 2)), H[3].dimshuffle((1, 0, 2)), H[4].dimshuffle((1, 0, 2)), H[5].dimshuffle((1, 0, 2))
        return H[-1]






class RNN_utanh_skip(Recurrent):
    def __init__(self, output_dim, r_us=1, r_uw=1, r_ws=1, r=1,
                 init='glorot_uniform', inner_init='uniform', activation='tanh', weights=None,
                 U_init = "uniform",
                 shallowness = 2,
                 truncate_gradient=-1, return_sequences=False, input_dim=None, input_length=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.U_init = U_init
        self.sh = shallowness
        self.input_dim = input_dim
        self.input_length = input_length
        self.r_us = r_us
        self.r_uw = r_uw
        self.r_ws = r_ws
        self.r = r

        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(RNN_utanh_skip, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]

        def init_U(way = self.U_init):
            if way == "identity":
                return theano.shared(np.identity(self.output_dim).astype("float32"))
            if way == "orthogonal":
                return self.inner_init((self.output_dim, self.output_dim))
            if way == "uniform":
                return self.init((self.output_dim, self.output_dim))

        self.W = self.init((input_dim, self.output_dim))
        self.U = init_U()#theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))
        self.Us = init_U()#theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))

        r_us = self.r_us
        r_uw = self.r_uw
        r_ws = self.r_ws
        r = self.r
        self.b_us = theano.shared(r_us*np.ones((self.output_dim,)).astype('float32')) 
        self.b_wu = theano.shared(r_uw*np.ones((self.output_dim,)).astype('float32')) 
        self.b_ws = theano.shared(r_ws*np.ones((self.output_dim,)).astype('float32')) 
        self.b_u  = theano.shared(r*np.ones((self.output_dim,)).astype('float32')) 
        self.b_s  = theano.shared(r*np.ones((self.output_dim,)).astype('float32')) 
        self.b_w  = theano.shared(r*np.ones((self.output_dim,)).astype('float32')) 
        self.b = shared_zeros((self.output_dim))
        
        self.params = [self.W, self.U, self.Us] + \
                      [self.b_us, self.b_wu, self.b_ws, self.b_u, self.b_s, self.b_w, self.b]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self, x_t, h_tm1, h_tms):
        h1 = T.dot(h_tm1, self.U)
        hs = T.dot(h_tms, self.Us)
        x1 = T.dot(x_t, self.W)
        h = self.activation(self.b_us*h1*hs + self.b_wu*x1*h1 + self.b_ws*x1*hs +
                            self.b_u*h1 + self.b_s*hs + self.b_w*x1 + self.b)
        return h 

    def get_output(self, train=False):
        X = self.get_input(train)
        X = X.dimshuffle((1, 0, 2))

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        H, updates = theano.scan(
            self._step,  
            sequences=[X],
            #outputs_info=[dict(initial = alloc_zeros_matrix(self.sh, X.shape[1], self.output_dim), taps = [-1, -self.sh])])
            outputs_info=[dict(initial = alloc_zeros_matrix(self.sh, X.shape[1], self.output_dim), taps = [-1, -self.sh])])

        if self.return_sequences:
            return H.dimshuffle((1, 0, 2)),  H.dimshuffle((1, 0, 2))
        return H[-1]


