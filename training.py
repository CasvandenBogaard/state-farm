import numpy as np

import os
import glob
import pickle
import sys

from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from sklearn.metrics import log_loss

from models.vgg import vgg16_adaptation
from tools import get_im_skipy, cache_data, restore_data
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imresize

from numpy.random import permutation
from random import randrange
import numpy as np

USE_CACHE = False
# color type: 1 - grey, 3 - rgb
COLOR_TYPE = 3
IMG_SHAPE = (246, 328)
VALID_SHAPE = (224, 298)
NETWORK_IMG_SHAPE = (224, 224)

TRAIN_NUM = sys.argv[1]

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


def load_train():
    X_train = []
    y_train = []
    driver_id = []

    driver_data = get_driver_data()

    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('data', 'imgs', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_skipy(fl, COLOR_TYPE, IMG_SHAPE)
            X_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[flbase])

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers


def read_and_normalize_train_data():
    cache_path = os.path.join('cache', 'train_{}x{}_t{}.dat'.format(IMG_SHAPE[0], IMG_SHAPE[1], COLOR_TYPE))
    if not os.path.isfile(cache_path) or (not USE_CACHE):
        train_data, train_target, driver_id, unique_drivers = load_train()
        cache_data((train_data, train_target, driver_id, unique_drivers), cache_path)
    else:
        print('Restore train from cache!')
        (train_data, train_target, driver_id, unique_drivers) = restore_data(cache_path)

    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = train_data.transpose((0, 3, 1, 2))
    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')

    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        train_data[:, c, :, :] = (train_data[:, c, :, :] - mean_pixel[c])

    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, driver_id, unique_drivers


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

def split_drivers(driver_list):
    splits = [
        {'test': ['p056', 'p081', 'p035'],
         'train': ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024', 'p039', 'p041', 'p042', 'p049',
                   'p051', 'p052', 'p061', 'p064', 'p066', 'p072', 'p075', 'p050', 'p026', 'p047', 'p045']},
        {'test': ['p039', 'p061', 'p075'],
         'train': ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024', 'p041', 'p042', 'p049', 'p051',
                   'p052', 'p056', 'p064', 'p066', 'p072', 'p081', 'p050', 'p026', 'p047', 'p045', 'p035']},
        {'test': ['p012', 'p041', 'p064'],
         'train': ['p002', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024', 'p039', 'p042', 'p049', 'p051', 'p052',
                   'p056', 'p061', 'p066', 'p072', 'p075', 'p081', 'p050', 'p026', 'p047', 'p045', 'p035']},
        {'test': ['p014', 'p042', 'p066'],
         'train': ['p002', 'p012', 'p015', 'p016', 'p021', 'p022', 'p024', 'p039', 'p041', 'p049', 'p051', 'p052',
                   'p056', 'p061', 'p064', 'p072', 'p075', 'p081', 'p050', 'p026', 'p047', 'p045', 'p035']},
        {'test': ['p015', 'p072', 'p045'],
         'train': ['p002', 'p012', 'p014', 'p016', 'p021', 'p022', 'p024', 'p039', 'p041', 'p042', 'p049', 'p051',
                   'p052', 'p056', 'p061', 'p064', 'p066', 'p075', 'p081', 'p050', 'p026', 'p047', 'p035']},
        {'test': ['p002', 'p016', 'p047'],
         'train': ['p012', 'p014', 'p015', 'p021', 'p022', 'p024', 'p039', 'p041', 'p042', 'p049', 'p051', 'p052',
                   'p056', 'p061', 'p064', 'p066', 'p072', 'p075', 'p081', 'p050', 'p026', 'p045', 'p035']},
        {'test': ['p021', 'p049'],
         'train': ['p002', 'p012', 'p014', 'p015', 'p016', 'p022', 'p024', 'p039', 'p041', 'p042', 'p051', 'p052',
                   'p056', 'p061', 'p064', 'p066', 'p072', 'p075', 'p081', 'p050', 'p026', 'p047', 'p045', 'p035']},
        {'test': ['p022', 'p050'],
         'train': ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p024', 'p039', 'p041', 'p042', 'p049', 'p051',
                   'p052', 'p056', 'p061', 'p064', 'p066', 'p072', 'p075', 'p081', 'p026', 'p047', 'p045', 'p035']},
        {'test': ['p024', 'p051'],
         'train': ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p039', 'p041', 'p042', 'p049', 'p052',
                   'p056', 'p061', 'p064', 'p066', 'p072', 'p075', 'p081', 'p050', 'p026', 'p047', 'p045', 'p035']},
        {'test': ['p052', 'p026'],
         'train': ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024', 'p039', 'p041', 'p042', 'p049',
                   'p051', 'p056', 'p061', 'p064', 'p066', 'p072', 'p075', 'p081', 'p050', 'p047', 'p045', 'p035']}
    ]

    split = splits[int(TRAIN_NUM)]

    train = split['train']
    test = split['test']

    return train, test

def cropping_generator(flow):

    while 1:
        X, Y = flow.next()

        result = np.zeros((len(X), COLOR_TYPE, NETWORK_IMG_SHAPE[0], NETWORK_IMG_SHAPE[1]))
        for i in range(len(X)):
            x = X[i]
            x = x[:,11:IMG_SHAPE[0]-11,:]

            cropping = randrange(0,3)
            if cropping == 0: # Left
                x = x[:, :, 0:NETWORK_IMG_SHAPE[1]]
            if cropping == 1: # Middle
                x = x[:, :, (IMG_SHAPE[1]-NETWORK_IMG_SHAPE[1])//2:(IMG_SHAPE[1]+NETWORK_IMG_SHAPE[1])//2]
            if cropping == 2: # Right
                x = x[:, :, (IMG_SHAPE[1]-NETWORK_IMG_SHAPE[1]):IMG_SHAPE[1]]

            result[i] = x

        yield (result, Y)

def validation_generator(X_valid, Y_valid, ignore_y=False):
    batch_size = 64
    index = 0

    max_index = len(X_valid)

    while 1:
        X_batch, Y_batch = X_valid[index:index+batch_size], Y_valid[index:index+batch_size]

        result = np.zeros((len(X_batch), COLOR_TYPE, NETWORK_IMG_SHAPE[0], NETWORK_IMG_SHAPE[1]))
        for i in range(len(X_batch)):
            x = X_batch[i]
            x = imresize(x, VALID_SHAPE).transpose((2,0,1))

            result[i] = x[:,:,(VALID_SHAPE[1]-NETWORK_IMG_SHAPE[1])//2:(VALID_SHAPE[1]+NETWORK_IMG_SHAPE[1])//2]

        index += batch_size
        if index > max_index:
            index = 0

        if (ignore_y):
            yield result
        else:
            yield (result, Y_batch)


def run_single():
    # input image dimensions
    batch_size = 64
    nb_epoch = 2

    train_data, train_target, driver_id, unique_drivers = read_and_normalize_train_data()
    
    train_drivers, test_drivers = split_drivers(unique_drivers)

    X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, train_drivers)
    X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, test_drivers)

    print('Start Single Run')
    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_valid), len(Y_valid))
    print('Train drivers: ', train_drivers)
    print('Test drivers: ', test_drivers)

    augmentationgenerator = ImageDataGenerator(
        rotation_range=5,
        height_shift_range=0.1,
        zoom_range=0.05,
    )
    augflow = augmentationgenerator.flow(X_train, Y_train, batch_size=batch_size)


    model = vgg16_adaptation(NETWORK_IMG_SHAPE[0], NETWORK_IMG_SHAPE[1], COLOR_TYPE)
    model.fit_generator(cropping_generator(augflow), nb_epoch=nb_epoch, verbose=1, samples_per_epoch=len(X_train),
                        nb_val_samples=len(Y_valid), validation_data=validation_generator(X_valid, Y_valid))

    predictions_valid = model.predict_generator(validation_generator(X_valid, Y_valid, ignore_y=True), len(Y_valid))
    score = log_loss(Y_valid, predictions_valid)
    print('Score log_loss: ', score)

    model.save_weights(os.path.join('cache', 'vgg_adaptation_weights_{}.h5'.format(TRAIN_NUM)), True)

run_single()