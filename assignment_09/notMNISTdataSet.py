# -*- coding: utf-8 -*-
# @Time    : 2019-09-01 15:44
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : notMNISTdataSet.py
# @Description:
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '/Users/henry/Documents/application/nlp_assignments/data/' # Change me to store data elsewhere


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent



def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  dest_filename = os.path.join(data_root, filename)
  if force or not os.path.exists(dest_filename):
    print('Attempting to download:', filename)
    #urlretrieve直接将远程数据下载到本地。
    filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', dest_filename)
  else:
    raise Exception(
      'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
  return dest_filename


num_classes = 10
np.random.seed(133)


def maybe_extract(filename, force=False):
    #分离文件名与扩展名
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        #tar 打包相关操作
        tar = tarfile.open(filename)
        #刷新stdout，这样就能每隔一秒输出一个数字了
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


if __name__=='__main__':
    train_filename ='/Users/henry/Documents/application/nlp_assignments/data/notMNIST_large.tar.gz'
    #maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = '/Users/henry/Documents/application/nlp_assignments/data/notMNIST_small.tar.gz'
    #maybe_download('notMNIST_small.tar.gz', 8458043)
    #
    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)