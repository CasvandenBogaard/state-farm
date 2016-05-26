import numpy as np

import os
import glob
import datetime
import pandas as pd

from models.vgg import vgg16_adaptation
from tools import get_im_skipy, cache_data, restore_data

USE_CACHE = True
# color type: 1 - grey, 3 - rgb
COLOR_TYPE = 3
IMG_SHAPE = (224, 224)


def load_test(files):
    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_skipy(fl, COLOR_TYPE, IMG_SHAPE)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id


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

    if not os.path.isfile(cache_path) or not USE_CACHE:
        test_data, test_id = load_test(batch)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(test_data.shape[0], COLOR_TYPE, IMG_SHAPE[0], IMG_SHAPE[1])
    test_data = test_data.astype('float32')

    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        test_data[:, c, :, :] = (test_data[:, c, :, :] - mean_pixel[c])/255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id


def generate_test_batches(size):
    path = os.path.join('data', 'imgs', 'test', '*.jpg')
    files = glob.glob(path)

    batches = [files[i:i+size] for i in range(0, len(files), size)]
    return batches, len(files)


def run_single():
    batch_size = 5000

    batches, total = generate_test_batches(batch_size)
    model = vgg16_adaptation(IMG_SHAPE[0], IMG_SHAPE[1], COLOR_TYPE)
    model.load_weights(os.path.join('cache', 'vgg16_weights.h5'))

    test_ids = []
    yfull_test = np.zeros((total, 10))


    for i, batch in enumerate(batches):
        test_data, test_id = read_and_normalize_test_data(batch, i)
        result = model.predict(test_data, verbose=1, batch_size=128)

        yfull_test[i*batch_size:i*batch_size+len(result),:] = result
        test_ids += test_id


    info_string = 'loss_r_' + str(IMG_SHAPE[0]) + '_c_' + str(IMG_SHAPE[1])

    create_submission(yfull_test, test_ids, info_string)


run_single()
