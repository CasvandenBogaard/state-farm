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
from scipy.misc import imread, imresize
from models import vgg16
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

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)

def read_and_normalize_test_data(batch, batch_num):
    print("Reading test batch {}".format(batch_num))

    cache_path = os.path.join('cache', 'test_{}x{}_t{}_b{}.dat'.format(IMG_SHAPE[0], IMG_SHAPE[1], COLOR_TYPE, batch_num))

    if not os.path.isfile(cache_path) or USE_CACHE:
        test_data, test_id = load_test(batch)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(test_data.shape[0], COLOR_TYPE, IMG_SHAPE[0], IMG_SHAPE[1])
    test_data = test_data.astype('float32')
    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def merge_several_folds_geom(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a *= np.array(data[i])
    a = np.power(a, 1/nfolds)
    return a.tolist()


def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)
    return data, target, index


def create_model_v1(img_rows, img_cols, color_type=1):
    nb_classes = 10
    nb_filters = 5 # number of convolutional filters to use
    nb_pool = 2 # size of pooling area for max pooling
    nb_conv = 4 # convolution kernel size

    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(color_type, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.6))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=0, momentum=0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def generate_test_batches(size):
    path = os.path.join('data', 'imgs', 'test', '*.jpg')
    files = glob.glob(path)

    batches = [files[i:i+size] for i in range(0, len(files), size)]
    return batches

def run_single():
    batch_size = 64

    batches = generate_test_batches(batch_size)
    model = vgg16(IMG_SHAPE[0], IMG_SHAPE[1], COLOR_TYPE)
    for i, batch in enumerate(batches):
        test_data, test_id = read_and_normalize_test_data(batch, i)
        result = model.predict(test_data, verbose=1)
        print(result.shape)


run_single()


# assume test-data are loaded

# load the model:
model = vgg16
# index of layer for which you want to extract the features, i.e for the last fully connected layr
layerIDX = len(model.layers)-1

getFeatures = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layersIDX].output])

for idx in len(testSet):
    img = testSet[idx]
    z,y,x = img.shape
    # need to reshape to make it work
    img = img.reshape(1,zy,x)
    feat = getFeatures([img, 0])[0][0] # the zero next to img differentiates between train and test, did not notice a difference so far

# plotting example
# plots the first feature map for the lastly input image
import matplotlib.pyplot as plt
plt.imshow(feat[0])
plt.show()
