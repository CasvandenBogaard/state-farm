#!/usr/bin/env python

"""
Lasagne implementation of CIFAR-10 examples from "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385)

Check the accompanying files for pretrained models. The 32-layer network (n=5), achieves a validation error of 7.42%,
while the 56-layer network (n=9) achieves error of 6.75%, which is roughly equivalent to the examples in the paper.
"""

from __future__ import print_function

import sys
import os
import time
import string
import random
import pickle
import datetime

import numpy as np
import theano.tensor as T
import theano
import lasagne
import inceptionv3_preprocess as dp
import pandas as pd

# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)

# ##################### Load data from CIFAR-10 dataset #######################
# this code assumes the cifar dataset from 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# has been extracted in current working directory

def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

def load_data():
    print("no loading")
    X_train, Y_train, train_index = dp.get_train_data()

    print(X_train.shape)


    return dict(
        X_train = lasagne.utils.floatX(X_train),
        Y_train = Y_train.astype('int32')
    )

    # return dict(
    #     X_train=lasagne.utils.floatX(X_train),
    #     Y_train=Y_train.astype('int32'),
    #     X_test = lasagne.utils.floatX(X_test),
    #     Y_test = Y_test.astype('int32'),)

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)

# ##################### Build the neural network model #######################


# Inception-v3, model from the paper:
# "Rethinking the Inception Architecture for Computer Vision"
# http://arxiv.org/abs/1512.00567
# Original source:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/imagenet/classify_image.py
# License: http://www.apache.org/licenses/LICENSE-2.0

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/inception_v3.pkl


from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import ConcatLayer
from lasagne.layers.normalization import batch_norm
from lasagne.nonlinearities import softmax


def bn_conv(input_layer, **kwargs):
    l = Conv2DLayer(input_layer, **kwargs)
    l = batch_norm(l, epsilon=0.001)
    return l


def inceptionA(input_layer, nfilt):
    # Corresponds to a modified version of figure 5 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=5, pad=2)

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][2], filter_size=3, pad=1)

    l4 = Pool2DLayer(
        input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2, l3, l4])


def inceptionB(input_layer, nfilt):
    # Corresponds to a modified version of figure 10 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=3, stride=2)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=3, pad=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=3, stride=2)

    l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

    return ConcatLayer([l1, l2, l3])


def inceptionC(input_layer, nfilt):
    # Corresponds to figure 6 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=(7, 1), pad=(3, 0))
    l3 = bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 7), pad=(0, 3))
    l3 = bn_conv(l3, num_filters=nfilt[2][3], filter_size=(7, 1), pad=(3, 0))
    l3 = bn_conv(l3, num_filters=nfilt[2][4], filter_size=(1, 7), pad=(0, 3))

    l4 = Pool2DLayer(
        input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2, l3, l4])


def inceptionD(input_layer, nfilt):
    # Corresponds to a modified version of figure 10 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)
    l1 = bn_conv(l1, num_filters=nfilt[0][1], filter_size=3, stride=2)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))
    l2 = bn_conv(l2, num_filters=nfilt[1][3], filter_size=3, stride=2)

    l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

    return ConcatLayer([l1, l2, l3])


def inceptionE(input_layer, nfilt, pool_mode):
    # Corresponds to figure 7 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2a = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 3), pad=(0, 1))
    l2b = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(3, 1), pad=(1, 0))

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
    l3a = bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 3), pad=(0, 1))
    l3b = bn_conv(l3, num_filters=nfilt[2][3], filter_size=(3, 1), pad=(1, 0))

    l4 = Pool2DLayer(
        input_layer, pool_size=3, stride=1, pad=1, mode=pool_mode)

    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2a, l2b, l3a, l3b, l4])


def build_network():
    net = {}

    net['input'] = InputLayer((None, 3, 299, 299))
    net['conv'] = bn_conv(net['input'],
                          num_filters=32, filter_size=3, stride=2)
    net['conv_1'] = bn_conv(net['conv'], num_filters=32, filter_size=3)
    net['conv_2'] = bn_conv(net['conv_1'],
                            num_filters=64, filter_size=3, pad=1)
    net['pool'] = Pool2DLayer(net['conv_2'], pool_size=3, stride=2, mode='max')

    net['conv_3'] = bn_conv(net['pool'], num_filters=80, filter_size=1)

    net['conv_4'] = bn_conv(net['conv_3'], num_filters=192, filter_size=3)

    net['pool_1'] = Pool2DLayer(net['conv_4'],
                                pool_size=3, stride=2, mode='max')
    net['mixed/join'] = inceptionA(
        net['pool_1'], nfilt=((64,), (48, 64), (64, 96, 96), (32,)))
    net['mixed_1/join'] = inceptionA(
        net['mixed/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

    net['mixed_2/join'] = inceptionA(
        net['mixed_1/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

    net['mixed_3/join'] = inceptionB(
        net['mixed_2/join'], nfilt=((384,), (64, 96, 96)))

    net['mixed_4/join'] = inceptionC(
        net['mixed_3/join'],
        nfilt=((192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)))

    net['mixed_5/join'] = inceptionC(
        net['mixed_4/join'],
        nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

    net['mixed_6/join'] = inceptionC(
        net['mixed_5/join'],
        nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

    net['mixed_7/join'] = inceptionC(
        net['mixed_6/join'],
        nfilt=((192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,)))

    net['mixed_8/join'] = inceptionD(
        net['mixed_7/join'],
        nfilt=((192, 320), (192, 192, 192, 192)))

    net['mixed_9/join'] = inceptionE(
        net['mixed_8/join'],
        nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
        pool_mode='average_exc_pad')

    net['mixed_10/join'] = inceptionE(
        net['mixed_9/join'],
        nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
        pool_mode='max')

    net['pool3'] = GlobalPoolLayer(net['mixed_10/join'])

    net['softmax'] = DenseLayer(
        net['pool3'], num_units=1008, nonlinearity=softmax)

    return net







# ############################# Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper :
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0,high=8,size=(batchsize,2))
            for r in range(batchsize):
                random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+299),crops[r,1]:(crops[r,1]+299)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]

# ############################## Main program ################################


from gradientzoo import LasagneGradientzoo



def main(num_epochs=82, model=None):
    # Load the dataset
    print("Loading data...")
    data = load_data()

    X_train = data['X_train']
    Y_train = data['Y_train']
    # X_test = data['X_test']
    # Y_test = data['Y_test']

    # Prepare Theano variables for inputs and targets

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")

    # Import Lasagne model zoo's build_network() func

    network_n = build_network()
    network = network_n['softmax']

    zoo = LasagneGradientzoo('commons/lasagne-inception-v3')
    zoo.load(network)

    # Now your network is ready to use

    print("number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))

    if True:
        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        # add weight decay
        all_layers = lasagne.layers.get_all_layers(network)
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
        loss = loss + l2_penalty

        print("create update expression")
        # Create update expressions for training
        # Stochastic Gradient Descent (SGD) with momentum
        print("params")
        params = lasagne.layers.get_all_params(network, trainable=True)
        lr = 0.001
        print("theanoshared")
        sh_lr = theano.shared(lasagne.utils.floatX(lr))
        print("updates")
        updates = lasagne.updates.momentum(
                loss, params, learning_rate=sh_lr, momentum=0.9)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        print("function")
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

    print("create loss expression")
    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)

    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    print("compile test function")
    test_fn = theano.function([input_var], test_prediction)

    if True:
        # # launch the training loop
        print("Starting training...")
        #We iterate over epochs:
        for epoch in range(num_epochs):
            # shuffle training data
            print("bla")
            print(X_train.shape[0])
            print(epoch)
            train_indices = np.arange(X_train.shape[0])
            np.random.shuffle(train_indices)
            X_train = X_train[train_indices,:,:,:]
            Y_train = Y_train[train_indices]

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            print("START BATCHING!")
            for batch in iterate_minibatches(X_train, Y_train, 8, shuffle=True, augment=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # # And a full pass over the validation data:
            # val_err = 0
            # val_acc = 0
            # val_batches = 0
            # for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
            #     inputs, targets = batch
            #     err, acc = val_fn(inputs, targets)
            #     val_err += err
            #     val_acc += acc
            #     val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            # print("  validation accuracy:\t\t{:.2f} %".format(
            #     val_acc / val_batches * 100))

            # adjust learning rate as in paper
            # 32k and 48k iterations should be roughly equivalent to 41 and 61 epochs
            if (epoch+1) == 41 or (epoch+1) == 61:
                new_lr = sh_lr.get_value() * 0.1
                print("New LR:"+str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))

        # dump the network weights to a file :
        np.savez('inceptionv3_run1.npz', *lasagne.layers.get_all_param_values(network))
    else:
        # load network weights from model file\
        print("GOIJAOIJGIOJIOEJIOEJAIOGOIEPHIOGHIOEHIOAHGIOHEAOIHGIOHIO")
        print("load params..")
        with np.load(model) as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)



    print("testing..")
    batch_size = 64
    batches, total = dp.generate_test_batches(batch_size)
    test_ids = []
    yfull_test = np.zeros((total, 10))

    for i, batch in enumerate(batches):
        test_data, test_id = dp.read_and_normalize_test_data(batch, i)
        scores = test_fn(lasagne.utils.floatX(test_data))
        print(scores[0])
        print(scores[1])
        yfull_test[i*batch_size:i*batch_size+len(scores),:] = scores
        test_ids += test_id

    info_string = 'inceptionv3' + str(299) + '_c_' + str(299)

    create_submission(yfull_test, test_ids, info_string)



if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains InceptionV3.")

        # Inception-v3, model from the paper:
        #
        # http://arxiv.org/abs/1512.00567
        # Original source:
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/imagenet/classify_image.py
        # License: http://www.apache.org/licenses/LICENSE-2.0

        print("Network architecture and training parameters are as in 'Rethinking the Inception Architecture for Computer Vision.'")
        print("Usage: %s [N [MODEL]]" % sys.argv[0])
        print()
        print("MODEL: saved model file to load (for validation) (default: None)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['n'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['model'] = sys.argv[2]
        #main(**kwargs)
        #main(5,2,"cifar_model_n5.npz")
        main(5, "inceptionv3_run1.npz")