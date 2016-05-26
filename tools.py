from scipy.misc import imread, imresize
import cPickle as pickle
import os

# color_type = 1 - gray
# color_type = 3 - RGB
def get_im_skipy(path, color_type, img_shape):
    img = imread(path, color_type == 1)
    resized = imresize(img, img_shape)
    return resized

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
