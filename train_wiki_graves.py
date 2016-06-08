# coding=utf-8
__author__ = 'Saizheng Zhang, saizheng.zhang@gmail.com'
import theano
import numpy as np
import theano.tensor as T
import pdb
import keras
import time
import pickle
"""
Definition of hyperparameters
"""
# Min/max sequence length
MIN_LENGTH = 50
MAX_LENGTH = 55
LENGTH = MAX_LENGTH
# Number of training samples
N_TRAIN = 1000
# Number of testing samples
N_TEST = 1000
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
# Number of training sequences in each batch
N_BATCH = 100
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 10

def onehot(x,numclasses=205):
    x = np.array(x)
    if x.shape==():
        x = x[np.newaxis]
    if numclasses is None:
        numclasses = x.max() + 1
    result = np.zeros(list(x.shape) + [numclasses],dtype=np.float32)
    z = np.zeros(x.shape)
    for c in range(numclasses):
        z *= 0
        z[np.where(x==c)] = 1
        result[...,c] += z
    return result


def gen_data_wiki():
    from theano.tensor.shared_randomstreams import RandomStreams
    numpy_rng  = np.random.RandomState(1)
    theano_rng = RandomStreams(1)

    print 'loading data...'
    framelen = 1
    maxnumframes = 100 #50
    testframes = 10000
    alphabetsize = 205
       
    data = np.load('/data/lisatmp3/zablocki/jun_data.npz', 'rb')
    # roughly 8.64G for training set, 1.08G for valid set, and 1.08G for test set
    # numtrain shoud be an integer of times than numvalid
    numtrain, numvalid, numtest = 90000000, 5000000,5000000
    #pdb.set_trace()
    #train_features_numpy = onehot(data['train']).reshape(numtrain/maxnumframes, maxnumframes, 205)
    #valid_features_numpy = onehot(data['valid']).reshape(numvalid/testframes,testframes, 205)
    #test_features_numpy = onehot(data['test']).reshape(numtest/testframes,testframes, 205)
    print '... done'
    
    #numpy_rng.shuffle(train_features_numpy)
    #numpy_rng.shuffle(valid_features_numpy)
    #valid_features = theano.shared(valid_features_numpy, name='valid_set', borrow=True)
    #test_features = theano.shared(test_features_numpy, name = 'test_set', borrow=True)  
    return data['train'], data['valid'], data['test'] 


def test_lstm():
    
    # load wiki data
    X_train_np, X_valid_np, X_test_np = gen_data_wiki()
    batchsize = 100
    blocklength = 25000 #450000
    bsize_test = batchsize 
    numframe = 100
    numframe_test = 1250#2500#5000 
    X_valid = onehot(X_valid_np).reshape(bsize_test, X_valid_np.shape[0]/bsize_test, 205)
    X_test = onehot(X_test_np).reshape(bsize_test, X_test_np.shape[0]/bsize_test, 205)
    nb_classes= 205 

    X_train_shared = theano.shared(np.zeros((batchsize,blocklength, nb_classes)).astype('float32'), name = 'train_set', borrow=True)
    X_valid_shared = theano.shared(np.zeros((bsize_test, numframe_test, nb_classes)).astype('float32'), name = 'valid_set', borrow=True)
    X_test_shared = theano.shared(np.zeros((bsize_test, numframe_test, nb_classes)).astype('float32'), name = 'test_set', borrow=True)

    # build the model
    from keras.layers.recurrent import LSTM, SimpleRNN, LSTMgrave 
    from layer_icml import LSTM_bu, LSTM_td, RNN_td, RNN_bu, RNN_sh, RNN_dp, LSTM_dp, RNN_shallow 
    from layer_icml import RNN_relugate, RNN_ens, RNN_2tanh, RNN_ntanh, RNN_multidp, LSTM_multi, LSTM_u, RNN_utanh, LSTM_uu, LSTM_uugrave 
    from keras.layers.core import Dense, Activation, TimeDistributedDense
    from keras.initializations import normal, identity

    x = T.tensor3()
    y = T.matrix()

    name_init = 'uniform'
    n_h = 2450; L1 = LSTMgrave(output_dim = n_h, init = 'uniform', batchsize = batchsize, inner_init = 'uniform',input_shape = (None, nb_classes), return_sequences=True); name_model= 'lstm_shallowgrave_' + str(n_h) + name_init + '0.01'+ '_batchsize' + str(batchsize) + '_numframe' + str(numframe)

    # RNN
    name_act = 'tanh'; name_init = 'uniform' 
    #n_h=2048;L1 = RNN_shallow(output_dim = n_h, init = 'uniform', U_init = name_init, activation = name_act, input_shape = (None, nb_classes), return_sequences=True);name_model = "rnn_tanh" + str(n_h) + "_"+name_act+ name_init + '0.1'
    #n_h = 2048;L1 = SimpleRNN(output_dim = n_h, init = 'uniform', inner_init = 'uniform', activation = name_act, input_shape = (None, nb_classes), return_sequences=True);name_model = "rnn_shallow"+str(n_h)+name_act+ name_init + '0.05'
    #n_h = 4096;L1 = RNN_utanh(output_dim = n_h, init = 'uniform', U_init = name_init, activation = name_act, input_shape = (None, nb_classes), return_sequences=True);name_model = "rnn_utanh_2_0_0" + str(n_h) + "_"+name_act+ name_init +'0.01' 
    n_h = 2048; in_act = 'tanh';L1 = LSTM_uugrave(output_dim = n_h, batchsize = batchsize, init = 'uniform', inner_init = 'uniform', input_shape = (None, nb_classes), return_sequences=True); name_model= 'lstm_u_grave'+in_act+'_1.0_1.0_1.0_0' + str(n_h) + name_init + '0.01' + '_batchsize' + str(batchsize) + '_numframe' + str(numframe)
    #n_h = 1200; in_act = 'tanh';L2 = LSTM_uu(output_dim = n_h, init = 'uniform', inner_init = 'uniform', input_shape = (None, n_h), return_sequences=True); name_model= 'lstm_u_stack2'+in_act+'_1.0_1.0_1.0_0' + str(n_h) + name_init + '0.01'
    #n_h = 700; L2 = LSTM_uu(output_dim = n_h, init = 'uniform', inner_init = 'uniform',input_shape = (None, n_h), return_sequences=True); name_model= 'lstm_u_1.0_1.0_0.5_0' + str(n_h) + name_init + '0.03'
    #n_h = 700; L3 = LSTM_uu(output_dim = n_h, init = 'uniform', inner_init = 'uniform',input_shape = (None, n_h), return_sequences=True); name_model= 'lstm_u_1.0_1.0_0.5_0' + str(n_h) + name_init + '0.03'
    #n_h = 700; L4 = LSTM_uu(output_dim = n_h, init = 'uniform', inner_init = 'uniform',input_shape = (None, n_h), return_sequences=True); name_model= 'lstm_u_1.0_1.0_0.5_0' + str(n_h) + name_init + '0.03'
    #n_h = 700; L5 = LSTM_uu(output_dim = n_h, init = 'uniform', inner_init = 'uniform',input_shape = (None, n_h), return_sequences=True); name_model= '7005layerlstm_uu_1.0_1.0_0.5_0' + str(n_h) + name_init + '0.03'

    D1 = TimeDistributedDense(nb_classes);D1._input_shape = [None, None, n_h]
    O = Activation('softmax')

    #layers = [L1, L2, L3, L4, L5, D1, O]
    layers = [L1, D1, O]
    #layers = [L1, L2, D1, O]

    load_model = True 
    if load_model:
        #f_model = open('/data/lisatmp3/zhangsa/lstm/models/180rnn_td_reluidentityotherinit_identity_sgd0.1_clip10.pkl', 'rb')
        #f_model = open('/data/lisatmp4/zhangsa/rnn_trans/models/wiki100lstm_u_gravetanh_1.0_1.0_1.0_02048uniform0.01_batchsize100_numframe100_adam0.001inorder_withtestfinetune5e-4inorder_withtest.pkl', 'rb')
        #f_model = open('/data/lisatmp4/zhangsa/rnn_trans/models/wiki100lstm_u_gravetanh_1.0_1.0_1.0_02048uniform0.01_batchsize100_numframe100_adam0.001inorder_withtest.pkl', 'rb')
        #f_model = open('/data/lisatmp4/zhangsa/rnn_trans/models/wiki100lstm_u_gravetanh_1.0_1.0_1.0_02048uniform0.01_batchsize100_numframe100_adam0.001inorder_withtest.pkl', 'rb')
        f_model = open('/data/lisatmp4/zhangsa/rnn_trans/models/wiki100lstm_u_gravetanh_1.0_0.5_1.0_02048uniform0.01_batchsize100_numframe100_adam0.001inorder_withtestfinetune1e-5inorder_withtest.pkl', 'rb')
        layers = pickle.load(f_model)
        f_model.close()
        name_model_load = 'wiki100lstm_u_gravetanh_1.0_0.5_1.0_02048uniform0.01_batchsize100_numframe100_adam0.001inorder_withtest' + 'finetune2e-6'
        #name_perpmat_load = 'wiki100lstm_u_gravetanh_1.0_1.0_1.0_02048uniform0.01_batchsize100_numframe100_adam0.001inorder_withtest.npy'
        L1 = layers[0]

    out  = x
    params = []
    for l in layers: 
        if not load_model:
            l.build()
        l.input = out
        params += l.params
        if l == L1:
            out = l.get_output()[0]
            h0 = l.get_output()[0]
            c0 = l.get_output()[1]
        else:
            out = l.get_output()

    # compute the loss
    loss = -T.mean(T.log(out)[:,:numframe-1,:] *x[:,1:,:])
    logperp_valid = T.mean(-T.log2(T.sum(out[:,:numframe_test-1,:]*x[:,1:,:],axis=2)))
    logperp_train = T.mean(-T.log2(T.sum(out[:,:numframe-1,:]*x[:,1:,:],axis=2)))

    # set optimizer
    from keras.constraints import identity as ident 
    from keras.optimizers import RMSprop, SGD, Adam

    lr_ = 2*1e-6
    clipnorm_ = 10000
    rmsprop = RMSprop(lr=lr_, clipnrom = clipnorm_)
    sgd = SGD(lr=lr_, momentum=0.9, clipnorm=clipnorm_)
    adam = Adam(lr=lr_)

    #opt = sgd; name_opt = 'sgd'+str(lr_); clip_flag = False 
    #opt = rmsprop; name_opt = 'rmsprop'+str(lr_)
    opt = adam; name_opt = 'adam' + str(lr_); clip_flag = False

    if clip_flag: 
        name_opt = name_opt + '_clip'+str(clipnorm_)

    #param update for regular parameters
    constraints = [ident() for p in params]    
    updates = opt.get_updates(params, constraints, loss)

    index = T.iscalar()
    f_train = theano.function([index], [loss, h0, c0], updates = updates,
            givens={x:X_train_shared[:,index*numframe : (index+1)*numframe, :]})

    # perplexity function
    f_perp_valid = theano.function([], [logperp_valid, h0, c0], givens={x:X_valid_shared})
    f_perp_test = theano.function([], [logperp_valid, h0, c0], givens={x:X_test_shared})

    #f_perp_valid = theano.function([index], [logperp_valid], givens={x:X_valid_shared[index*bsize_test : (index+1)*bsize_test]})
    #f_perp_test = theano.function([index], [logperp_valid], givens={x:X_test_shared[index*bsize_test : (index+1)*bsize_test]})


    def perp_valid():
        logperp_acc = 0
        n = 0
        L1.H0.set_value(np.zeros((batchsize, n_h)).astype('float32'))
        L1.C0.set_value(np.zeros((batchsize, n_h)).astype('float32'))
        for k in xrange(X_valid.shape[1]/numframe_test):
            X_valid_shared.set_value(X_valid[:, k*numframe_test:(k+1)*numframe_test, :])
            perp, h0, c0 = f_perp_valid()
            logperp_acc += perp
            L1.H0.set_value(h0[:,-1,:])
            L1.C0.set_value(c0[:,-1,:])
            n += 1
        return (logperp_acc/n)

    def perp_test():
        logperp_acc = 0
        n = 0
        L1.H0.set_value(np.zeros((batchsize, n_h)).astype('float32'))
        L1.C0.set_value(np.zeros((batchsize, n_h)).astype('float32'))
        for k in xrange(X_test.shape[1]/numframe_test):
            X_test_shared.set_value(X_test[:, k*numframe_test:(k+1)*numframe_test, :])
            perp, h0, c0 = f_perp_test()
            logperp_acc += perp
            L1.H0.set_value(h0[:,-1,:])
            L1.C0.set_value(c0[:,-1,:])
            n += 1
        return (logperp_acc/n)


    #def perp_valid():
    #    logperp_acc = 0
    #    n = 0
    #    for k in xrange(X_valid_np.shape[0]/(bsize_test*numframe_test)):
    #        X_valid_shared.set_value(onehot(X_valid_np[k*bsize_test*numframe_test:(k+1)*bsize_test*numframe_test]).reshape((bsize_test, numframe_test, 205)))
    #        for i in xrange(X_valid_shared.get_value().shape[0]/bsize_test):
    #            logperp_acc += f_perp_valid(i)
    #            n += 1
    #    return (logperp_acc/n)

    #def perp_test():
    #    logperp_acc = 0
    #    n = 0
    #    for k in xrange(X_test_np.shape[0]/(bsize_test*numframe_test)):
    #        X_test_shared.set_value(onehot(X_test_np[k*bsize_test*numframe_test:(k+1)*bsize_test*numframe_test]).reshape((bsize_test, numframe_test, 205)))
    #        for i in xrange(X_test_shared.get_value().shape[0]/bsize_test):
    #            logperp_acc += f_perp_test(i)
    #            n += 1
    #    return (logperp_acc/n)


    ######## testmodel ########
    #test_score = perp_valid()
    #pdb.set_trace()


    epoch_ = 9000 
    perpmat = np.zeros((epoch_, 3))
    t_start = time.time()
    name = 'wiki100'+ name_model + '_' +  name_opt 

    if load_model:
        name = name_model_load 
        #perpmat = np.load(name_perpmat_load)

    #only_block = False
    #if only_block:
    #    name = name + 'random_only_block'
    #else:
    #    name = name + 'random_per_row_in_block'
    name = name+'inorder'
    blocksize = batchsize*blocklength
    bestscore = 100000000
    for epoch in xrange(epoch_):
        for k in xrange(X_train_np.shape[0]/blocksize):
            t_s = time.time()
            print "reloading " + str(k) + " th train patch..."

            #if only_block:
            #    pos = np.random.randint(0, X_train_np.shape[0]-blocksize)
            #    X_train_shared.set_value(onehot(X_train_np[pos: pos + blocksize]).reshape(batchsize, blocklength, 205))
            #else:    
            #    pos = np.random.randint(0, X_train_np.shape[0]-blocklength, batchsize)
            #    tmp = np.zeros((batchsize, blocklength, 205)).astype('float32')
            #    for j in xrange(batchsize):
            #        tmp[j] = onehot(X_train_np[pos[j]: pos[j] + blocklength])
            #    X_train_shared.set_value(tmp)
            X_train_shared.set_value(onehot(X_train_np[k*blocksize: (k+1)*blocksize]).reshape(batchsize, blocklength, 205)) 
            print "reloading finished, time cost: " + str(time.time()-t_s)
            L1.H0.set_value(np.zeros((batchsize, n_h)).astype('float32'))
            L1.C0.set_value(np.zeros((batchsize, n_h)).astype('float32'))
            for i in xrange(blocklength/numframe):
                loss, h0, c0 = f_train(i)
                L1.H0.set_value(h0[:,-1,:])
                L1.C0.set_value(c0[:,-1,:])
                if i%10 == 0:
                    t_end = time.time()
                    print "Time consumed: " + str(t_end - t_start) + " secs."
                    t_start = time.time()
                    print "Epoch "+ str(epoch)+" " + name + ": The training loss in batch " + str(k*(blocklength/numframe)+i) +" is: " + str(loss) + "."
            if k%6 == 0:
                #save results
                m = epoch*X_train_np.shape[0]/(blocksize*6) +k/6
                perpmat[m][0], perpmat[m][1] = 0, perp_valid()
                perpmat[m][2] = perp_test()
                np.save('/data/lisatmp4/zhangsa/rnn_trans/results/' + name +'_withtest.npy', perpmat)

                #save model
                if perpmat[m][1] < bestscore:
                    bestscore = perpmat[m][1]
                    f_model = open('/data/lisatmp4/zhangsa/rnn_trans/models/' + name + '_withtest.pkl', 'wb+')
                    pickle.dump(layers, f_model)
                    f_model.close()
       
        print "Epoch "+ str(epoch)+ " " + name + ": The training perp is: " + str(perpmat[epoch][0]) \
                      + ", test perp is: " + str(perpmat[epoch][1]) + "." 



if __name__ == "__main__":
    test_lstm()
