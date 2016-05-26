import numpy as np

import os
import glob
import pickle

from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from sklearn.metrics import log_loss

from models.vgg import vgg16_adaptation
from tools import get_im_skipy, cache_data, restore_data

USE_CACHE = False
# color type: 1 - grey, 3 - rgb
COLOR_TYPE = 3
IMG_SHAPE = (224, 224)


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
            img = get_im_skipy(fl)
            X_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[flbase])

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers


def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


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
    train_data = train_data.reshape(train_data.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], COLOR_TYPE)
    train_data = train_data.transpose(0, 3, 1, 2)
    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')

    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        train_data[:, c, :, :] = (train_data[:, c, :, :] - mean_pixel[c])/255

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


def run_single():
    # input image dimensions
    batch_size = 64
    nb_epoch = 2

    train_data, train_target, driver_id, unique_drivers = read_and_normalize_train_data()

    unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                     'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                     'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                     'p075']
    X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
    unique_list_valid = ['p081']
    X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

    print('Start Single Run')
    print('Split train: ', len(X_train), len(Y_train))
    print('Split valid: ', len(X_valid), len(Y_valid))
    print('Train drivers: ', unique_list_train)
    print('Test drivers: ', unique_list_valid)

    model = vgg16_adaptation(IMG_SHAPE[0], IMG_SHAPE[1], COLOR_TYPE)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_data=(X_valid, Y_valid))

    predictions_valid = model.predict(X_valid, batch_size=64, verbose=1)
    score = log_loss(Y_valid, predictions_valid)
    print('Score log_loss: ', score)

    model.save_weights(os.path.join('cache', 'vgg16_weights.h5'), True)

run_single()