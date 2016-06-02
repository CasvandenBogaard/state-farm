import numpy as np

import os
import glob
import math
import cPickle as pickle
import datetime
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from scipy.misc import imread, imresize
from models.vgg import vgg16_adaptation
from keras import backend as K

USE_CACHE = False
# color type: 1 - grey, 3 - rgb
COLOR_TYPE = 3
IMG_SHAPE = (224, 224)


# color_type = 1 - gray
# color_type = 3 - RGB
def get_im_skipy(path):
    img = imread(path, COLOR_TYPE == 1)
    resized = imresize(img, IMG_SHAPE)
    return resized

def get_driver_data():
    dr = dict()
    path = os.path.join('data', 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr

def load_test(files):
    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_skipy(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id


def cache_data(data, path):
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data

def read_and_normalize_train_data(batch, batch_num):
    cache_path = os.path.join('cache', 'train_{}x{}_t{}_b{}.dat'.format(IMG_SHAPE[0], IMG_SHAPE[1], COLOR_TYPE, batch_num))
    
    if not os.path.isfile(cache_path) or not USE_CACHE:
        train_data, train_id = load_test(batch)
        cache_data((train_data, train_id), cache_path)
    else:
        print('Restore test from cache!')
        (train_data, train_id) = restore_data(cache_path)

    train_data = np.array(train_data, dtype=np.uint8)
    train_data = train_data.transpose((0, 3, 1, 2))
    train_data = train_data.astype('float32')

    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        train_data[:, c, :, :] = (train_data[:, c, :, :] - mean_pixel[c])

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_id


def read_and_normalize_test_data(batch, batch_num):
    print("Reading test batch {}".format(batch_num))

    cache_path = os.path.join('cache', 'test_{}x{}_t{}_b{}.dat'.format(IMG_SHAPE[0], IMG_SHAPE[1], COLOR_TYPE, batch_num))

    if not os.path.isfile(cache_path) or not USE_CACHE:
        test_data, test_id = load_test(batch)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))
    test_data = test_data.astype('float32')

    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        test_data[:, c, :, :] = (test_data[:, c, :, :] - mean_pixel[c])

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id


def generate_test_batches(size):
    path = os.path.join('data', 'imgs', 'test', '*.jpg')
    files = glob.glob(path)

    batches = [files[i:i+size] for i in range(0, len(files), size)]
    return batches, len(files)

def generate_train_batches(j, size):
    path = os.path.join('data', 'imgs', 'train', 'c{}'.format(j), '*.jpg')
    files = glob.glob(path)

    batches = [files[i:i+size] for i in range(0, len(files), size)]
    return batches, len(files)    


def run_single_train():
    batch_size = 2

    print "training with batch size " + str(batch_size)

    model = vgg16_adaptation(IMG_SHAPE[0], IMG_SHAPE[1], COLOR_TYPE)

    get_fc1_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[32].output])
    get_fc2_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[34].output])
    
    for j in range(10):
        print "Starting subset {}".format(j)
        
        batches, total = generate_train_batches(j, batch_size)
        
        train_ids = []
        yfull_train = np.zeros((total, 8192))
        
        for i, batch in enumerate(batches):
            print("Doing batch {} of {}".format(i+1, len(batches)))
            train_data, train_id = read_and_normalize_train_data(batch, i)
            print train_id
            activations1 = get_fc1_output([train_data, 0])[0]
            activations2 = get_fc2_output([train_data, 0])[0]

            activations = np.concatenate((activations1, activations2), axis=1)

            yfull_train[i*batch_size:i*batch_size+len(activations),:] = activations
            train_ids += train_id


        df = pd.DataFrame(yfull_train)
        df.loc[:, 'img'] = pd.Series(train_ids, index=df.index)

        df.to_csv("activations/activations_train_c{}.csv".format(j), index=False)
        print("Train activations for c{} saved".format(j))

def run_single_test():
    batch_size = 64

    batches, total = generate_test_batches(batch_size)
    model = vgg16_adaptation(IMG_SHAPE[0], IMG_SHAPE[1], COLOR_TYPE)
    
    test_ids = []
    yfull_test = np.zeros((total, 8192))

    get_fc1_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[32].output])
    get_fc2_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[34].output])
    
    for i, batch in enumerate(batches):
        print("Doing batch {} of {}".format(i+1, len(batches)))
        test_data, test_id = read_and_normalize_test_data(batch, i)
        print test_id
        activations1 = get_fc1_output([test_data, 0])[0]
        activations2 = get_fc2_output([test_data, 0])[0]

        activations = np.concatenate((activations1, activations2), axis=1)

        yfull_test[i*batch_size:i*batch_size+len(activations),:] = activations
        test_ids += test_id


    df = pd.DataFrame(yfull_test)
    df.loc[:, 'img'] = pd.Series(test_ids, index=df.index)
    
    df.to_csv("activations/activations_test", index=False)
    print("Test activations saved")
    
run_single_train()
run_single_test()