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


# BLVC Googlenet, model from the paper:
# "Going Deeper with Convolutions"
# Original source:
# https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
# License: unrestricted use

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/blvc_googlenet.pkl

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear


def build_inception_module(name, input_layer, nfilters):
    # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    net = {}
    net['pool'] = PoolLayerDNN(input_layer, pool_size=3, stride=1, pad=1)
    net['pool_proj'] = ConvLayer(
        net['pool'], nfilters[0], 1, flip_filters=False)

    net['1x1'] = ConvLayer(input_layer, nfilters[1], 1, flip_filters=False)

    net['3x3_reduce'] = ConvLayer(
        input_layer, nfilters[2], 1, flip_filters=False)
    net['3x3'] = ConvLayer(
        net['3x3_reduce'], nfilters[3], 3, pad=1, flip_filters=False)

    net['5x5_reduce'] = ConvLayer(
        input_layer, nfilters[4], 1, flip_filters=False)
    net['5x5'] = ConvLayer(
        net['5x5_reduce'], nfilters[5], 5, pad=2, flip_filters=False)

    net['output'] = ConcatLayer([
        net['1x1'],
        net['3x3'],
        net['5x5'],
        net['pool_proj'],
        ])

    return {'{}/{}'.format(name, k): v for k, v in net.items()}


def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, None, None))
    net['conv1/7x7_s2'] = ConvLayer(
        net['input'], 64, 7, stride=2, pad=3, flip_filters=False)
    net['pool1/3x3_s2'] = PoolLayer(
        net['conv1/7x7_s2'], pool_size=3, stride=2, ignore_border=False)
    net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
    net['conv2/3x3_reduce'] = ConvLayer(
        net['pool1/norm1'], 64, 1, flip_filters=False)
    net['conv2/3x3'] = ConvLayer(
        net['conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False)
    net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
    net['pool2/3x3_s2'] = PoolLayer(
      net['conv2/norm2'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_3a',
                                      net['pool2/3x3_s2'],
                                      [32, 64, 96, 128, 16, 32]))
    net.update(build_inception_module('inception_3b',
                                      net['inception_3a/output'],
                                      [64, 128, 128, 192, 32, 96]))
    net['pool3/3x3_s2'] = PoolLayer(
      net['inception_3b/output'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_4a',
                                      net['pool3/3x3_s2'],
                                      [64, 192, 96, 208, 16, 48]))
    net.update(build_inception_module('inception_4b',
                                      net['inception_4a/output'],
                                      [64, 160, 112, 224, 24, 64]))
    net.update(build_inception_module('inception_4c',
                                      net['inception_4b/output'],
                                      [64, 128, 128, 256, 24, 64]))
    net.update(build_inception_module('inception_4d',
                                      net['inception_4c/output'],
                                      [64, 112, 144, 288, 32, 64]))
    net.update(build_inception_module('inception_4e',
                                      net['inception_4d/output'],
                                      [128, 256, 160, 320, 32, 128]))
    net['pool4/3x3_s2'] = PoolLayer(
      net['inception_4e/output'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_5a',
                                      net['pool4/3x3_s2'],
                                      [128, 256, 160, 320, 32, 128]))
    net.update(build_inception_module('inception_5b',
                                      net['inception_5a/output'],
                                      [128, 384, 192, 384, 48, 128]))

    net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])
    net['loss3/classifier'] = DenseLayer(net['pool5/7x7_s1'],
                                         num_units=1000,
                                         nonlinearity=linear)
    net['prob'] = NonlinearityLayer(net['loss3/classifier'],
                                    nonlinearity=softmax)
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
                random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+224),crops[r,1]:(crops[r,1]+244)]
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

    # Import Lasagne model zoo's build_model() func

    network = build_model()
    output_layer = network['prob']

    zoo = LasagneGradientzoo('commons/lasagne-googlenet')
    zoo.load(output_layer)

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
        np.savez('Googlenet_run1.npz', *lasagne.layers.get_all_param_values(network))
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

    info_string = 'googlenet' + str(224) + '_c_' + str(224)

    create_submission(yfull_test, test_ids, info_string)



if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains Googlenet.")

        # BLVC Googlenet, model from the paper:
        # "Going Deeper with Convolutions"
        # Original source:
        # https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
        # License: unrestricted use




        print("Network architecture and training parameters are as in 'Going Deeper with Convolutions'")
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
        main(5, "Googlenet_run1.npz")