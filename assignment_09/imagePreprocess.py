# -*- coding: utf-8 -*-
# @Time    : 2019-09-01 16:12
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : imagePreprocess.py
# @Description:
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

data_root = '/Users/henry/Documents/application/nlp_assignments/data/' # Change me to store data elsewhere

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
train_size = 200000
valid_size = 10000
test_size = 10000
num_labels = 10

def load_letter(folder, min_num_images):
    """Load the data for a single letter label.把图片序列化"""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            #不明白什么意思
            img = imageio.imread(image_file)
            img  = img.astype(float)
            img = img - pixel_depth / 2
            image_data2 = img / pixel_depth
            image_data = (imageio.imread(image_file).astype(float) -
                         pixel_depth / 2) / pixel_depth

            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except (IOError, ValueError) as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    #算术平均值。
    print('Mean:', np.mean(dataset))
    #标计算准差
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            #plt.imshow(dataset[0])
            #plt.show()
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)#整型，最高协议版本，参数protocol是序列化模式，默认是0（ASCII协议，表示以文本的形式进行序列化），protocol的值还可以是1和2（1和2表示以二进制的形式进行序列化。其中，1是老式的二进制协议；2是新二进制协议）
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    #“/ "表示 浮点数除法，返回浮点结果;" // "表示整数除法
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels

def randomize(dataset, labels):
    #生成随机序列
  permutation = np.random.permutation(labels.shape[0])
    #没看懂什么操作
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels



def createSet(test_datasets,train_datasets):


    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
    return (train_dataset,train_labels),(valid_dataset,valid_labels),(test_dataset,test_labels)

def save(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):
    pickle_file = os.path.join(data_root, 'notMNIST.pickle')

    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)

def main():
    test_folders = ['/Users/henry/Documents/application/nlp_assignments/data/notMNIST_small/A',
                    '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_small/B',
                    '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_small/C',
                    '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_small/D',
                    '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_small/E',
                    '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_small/F',
                    '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_small/G',
                    '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_small/H',
                    '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_small/I',
                    '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_small/J']
    train_folders = ['/Users/henry/Documents/application/nlp_assignments/data/notMNIST_large/A', '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_large/B', '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_large/C', '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_large/D', '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_large/E', '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_large/F', '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_large/G', '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_large/H', '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_large/I', '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_large/J']

    train_datasets = maybe_pickle(train_folders, 45000)
    test_datasets = maybe_pickle(test_folders, 1800)
    (train_dataset, train_labels), (valid_dataset, valid_labels), (test_dataset, test_labels) = createSet(test_datasets, train_datasets)
    '''
    train_dataset = dataset[0][0]
    train_labels = dataset[0][1]
    valid_dataset  = dataset[1][0]
    valid_labels = dataset[1][1]
    test_dataset  = dataset[2][0]
    test_labels = dataset[2][1]
    '''
    #[(train_dataset, train_labels), (valid_dataset, valid_labels), (test_dataset, test_labels)]
    save(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)

def getDataSet(pickle_file='/Users/henry/Documents/application/nlp_assignments/data/notMNIST.pickle'):
        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            valid_dataset = save['valid_dataset']
            valid_labels = save['valid_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
            del save  # hint to help gc free up memory
            print('Training set', train_dataset.shape, train_labels.shape)
            print('Validation set', valid_dataset.shape, valid_labels.shape)
            print('Test set', test_dataset.shape, test_labels.shape)
        return (train_dataset,train_labels),(valid_dataset,valid_labels),(test_dataset,test_labels)

def reformat(dataset, labels):
     dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
     labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
     return dataset, labels

if __name__=='__main__':
     main()




