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

        self.W2 = self.init((self.output_dim, self.output_dim))
        self.U2 = init_U()
        #self.V2 = theano.shared(np.zeros((self.output_dim, self.output_dim)).astype('float32'))
        self.V2 = self.init((self.output_dim, self.output_dim))

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
        #b[self.output_dim: 2*self.output_dim] = np.ones((self.output_dim), dtype = "float32")
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
    def __init__(self, output_dim,
                 init='uniform', inner_init='uniform', activation='tanh', weights=None,
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


