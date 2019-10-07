# -*- coding: utf-8 -*-
# @Time    : 2019-09-14 10:31
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : preprocessing.py
# @Description:语料预处理

import logging, jieba, os, re
from functools import lru_cache
import tensorflow as tf
import csv
import jieba
import  re
import pickle
import numpy as np
from gensim.models import Word2Vec
TRAIN_NUM = 0.8#训练集
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep+'word2vect'+ os.sep
fearture_num = 100
labelnum = 5
class W2V(object):
    """
    利用word2Vector产生句向量
    """
    def __init__(self,modelFile):
        if modelFile  is not None:
             self.model = Word2Vec.load(modelFile)

    def w2vfSentence2Vector(self,sentence):
        """
                   利用word2Vector求句子向量
                   :param words:
                   :return:
                   """

        # words = [k for k in jieba.cut(sentence) if k not in stopword_list]
        words = cut2list(sentence)
        sen_vec = np.sum([self.model.wv[k] for k in words if k in self.model], axis=0) / len(words)

        return sen_vec

    def getW2V(self,word):
        return self.model.wv[word]

#word2vector的模型地址
s2v_w2v = W2V(DIR + 'w2v.model')

def getDataSet(pickle_file='/Users/henry/Documents/application/nlp_assignments/data/word2vect/s2v_w2v.pickle'):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        labelsSet = save['labelsSet']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
        print('labelsSet', labelsSet)
    return (train_dataset, train_labels), (valid_dataset, valid_labels), (test_dataset, test_labels),labelsSet

def getRawDataSet(pickle_file='/Users/henry/Documents/application/nlp_assignments/data/word2vect/s2v_w2v.pickle'):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(MacOSFile(f))
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        labelsSet = save['labelsSet']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
        print('labelsSet', labelsSet)
    return (train_dataset, train_labels), (valid_dataset, valid_labels), (test_dataset, test_labels),labelsSet

def saveRawDataset(contentColumns, labelColumn,pickle_dir, input_file):
    """

    :param contentColumns:  csv的内容字段
    :param labelColumn: csv的label字段
    :param pickle_dir: 数据集存储的文件目录
    :param input_file:
    :return:
    """

    (datasets, datalabels, labelsSet) = formatRawDataset(contentColumns, labelColumn, input_file=input_file)
    dataset_num = len(datasets)
    indexArray = np.arange(dataset_num)
    np.random.shuffle(indexArray)
    trainMaxIndex = int(dataset_num * 0.8)
    validMaxIndex = trainMaxIndex + (dataset_num - trainMaxIndex) // 2
    train_dataset = datasets[:trainMaxIndex + 1]
    train_labels = datalabels[:trainMaxIndex + 1]

    valid_dataset = datasets[trainMaxIndex + 1:validMaxIndex]
    valid_labels = datalabels[trainMaxIndex + 1:validMaxIndex]

    test_dataset = datasets[validMaxIndex:]
    test_labels = datalabels[validMaxIndex:]

    print('dataset_num = {},trainMaxIndex = {} ,validMaxIndex = {} '.format(dataset_num, trainMaxIndex, validMaxIndex))
    pickle_file = os.path.join(pickle_dir, 's2v_w2v_raw.pickle')

    try:
        f = open(pickle_file, 'wb')
        print('train_dataset.shape = {} ,train_labels.shape  = {}  '.format(train_dataset.shape, train_labels.shape))
        print('valid_dataset.shape = {} ,valid_labels.shape  = {}  '.format(valid_dataset.shape, valid_labels.shape))
        print('test_dataset.shape = {} ,test_labels.shape  = {}  '.format(test_dataset.shape, test_labels.shape))
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
            'labelsSet': labelsSet
        }
        pickle_dump(save, f)
        #pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)


def saveDataset(contentColumns,labelColumn,pickle_dir,input_file):
    """

    :param contentColumns: csv的内容字段
    :param labelColumn: csv的label字段
    :param pickle_dir: 数据集存储的文件目录
    :param input_file:
    :return:
    """
    (datasets, datalabels, labelsSet) =formatDataset(contentColumns, labelColumn, input_file=input_file)
    dataset_num = len(datasets)
    indexArray = np.arange(dataset_num)
    np.random.shuffle(indexArray)
    trainMaxIndex = int(dataset_num*0.8)
    validMaxIndex = trainMaxIndex + (dataset_num-trainMaxIndex )//2
    #train_dataset = []
    #train_labels = []
    train_dataset,train_labels = make_arrays(trainMaxIndex+1, fearture_num)
    #valid_dataset = []
    #valid_labels = []
    valid_dataset, valid_labels = make_arrays(validMaxIndex-trainMaxIndex, fearture_num)
    #test_dataset = []
    #test_labels = []
    test_dataset, test_labels = make_arrays(dataset_num-validMaxIndex, fearture_num)

    print('dataset_num = {},trainMaxIndex = {} ,validMaxIndex = {} '.format(dataset_num,trainMaxIndex,validMaxIndex))
    trainIndex = 0
    valiIndex = 0
    testIndex = 0
    for i,v in enumerate(indexArray):
        if i <= trainMaxIndex:
            #train_dataset.append (datasets[v])
            #train_labels.append(datalabels[v])
            train_dataset[trainIndex] = datasets[v]
            train_labels[trainIndex] = datalabels[v]
            trainIndex += 1
        elif i > trainMaxIndex and i <= validMaxIndex:
            #valid_dataset.append(datasets[v])
            #valid_labels.append(datalabels[v])
            valid_dataset[valiIndex] = datasets[v]
            valid_labels[valiIndex] = datalabels[v]
            valiIndex += 1
        else:
            #test_dataset.append(datasets[v])
            #test_labels.append(datalabels[v])
            test_dataset[testIndex] = datasets[v]
            test_labels[testIndex] = datalabels[v]
            testIndex += 1


    pickle_file = os.path.join(pickle_dir, 's2v_w2v.pickle')

    try:
        f = open(pickle_file, 'wb')
        print('train_dataset.shape = {} ,train_labels.shape  = {}  '.format(train_dataset.shape,train_labels.shape))
        print('valid_dataset.shape = {} ,valid_labels.shape  = {}  '.format(valid_dataset.shape, valid_labels.shape))
        print('test_dataset.shape = {} ,test_labels.shape  = {}  '.format(test_dataset.shape, test_labels.shape))
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
            'labelsSet': labelsSet
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)

        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)

def make_arrays(nb_rows, cls):
    if nb_rows:
        dataset = np.ndarray((nb_rows, cls), dtype=np.float32)
        labels = np.zeros((nb_rows, labelnum))
    else:
        dataset, labels = None, None
    return dataset, labels

def createLabel(labelnum,index):
    label = np.zeros((1, labelnum))
    label[0,index-1] = 1
    return label

def buildLabel(labelnum,index):
    label = np.zeros(labelnum)
    label[index-1] = 1
    return label


def formatRawDataset(contentColumns,labelColumn,input_file):
    """
    将csv数据读出每个文档向量化（句向量，段向量，文档向量），然后将分词后([datasets],[datalabels],{label})存储起来
    :param file: csv数据集文件
    :return:([datasets],[datalabels],{label})
    """
    #textcount = 0
    datasets = []
    datalabels = []
    labelsSet = set()
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            #textcount += 1
            #if textcount == 100:
                #break
            content_line = ''
            for column in contentColumns:
                content_line += row[column]
                la = row[labelColumn]
            if  la!=labelColumn  and len(content_line.strip()) > 0:
               v =  cut2list(content_line)
               datasets.append(' '.join(v))
               datalabels.append(buildLabel(labelnum,int(la)))
               if la not in labelsSet:
                   labelsSet.add(la)



    print("dataset total number is  {} ,datalabels total number is  {}  ".format(len(datasets), len(datalabels)))
    return (np.array(datasets),np.array(datalabels),labelsSet)


def formatDataset(contentColumns,labelColumn,input_file):
    """
    将csv数据读出每个文档向量化（句向量，段向量，文档向量），然后将其序列化为([datasets],[datalabels],{label})存储起来
    :param file: csv数据集文件
    :return:([datasets],[datalabels],{label})
    """
    #textcount = 0
    datasets = []
    datalabels = []
    labelsSet = set()
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            #textcount += 1
            #if textcount == 100:
                #break
            content_line = ''
            for column in contentColumns:
                content_line += row[column]
                la = row[labelColumn]
            if  la!=labelColumn  and len(content_line.strip()) > 0:
               v = s2v_w2v.w2vfSentence2Vector(content_line)
               datasets.append(v)
               datalabels.append( createLabel(labelnum,int(la)))
               if la not in labelsSet:
                   labelsSet.add(la)



    print("dataset total number is  {} ,datalabels total number is  {}  ".format(len(datasets), len(datalabels)))
    return (np.array(datasets),np.array(datalabels),labelsSet)



@lru_cache(maxsize=2 ** 10)
def get_stopwords(stopwordsFile = '/Users/henry/Documents/application/nlp_assignments/data/stopwords.txt'):
    print("start load stopwords")
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    # 加载停用词表
    stopword_set = set()
    if(os.path.exists(stopwordsFile)):
        stopword_list = [k.strip() for k in open(stopwordsFile, encoding='utf8').readlines() if k.strip() != '']
        stopword_set = set(stopword_list)
    print("end load stopwords")
    return stopword_set


def cut(string):
    # 获取停用词表
    stopwords = get_stopwords()
    article_contents = ''
    sens = strQ2B(string)
    for sen in sens:
        # 使用jieba进行分词
        words = jieba.cut(sen, cut_all=False)
        for word in words:
            if word not in stopwords:
                article_contents += word + " "
    return article_contents

def cut2list(string):
    """
    返回list
    :param string:
    :return:
    """
    # 获取停用词表
    stopwords = get_stopwords()
    tokens = []
    sens = strQ2B(string)
    for sen in sens:
        # 使用jieba进行分词
        words = jieba.cut(sen, cut_all=False)
        for word in words:
            if word not in stopwords:
                tokens.append(word)
    return tokens

def tokenizeFormCsv( input_file, columns,save_file_path):
    """Reads a tab separated value file."""
    # 写文件
    output = open(save_file_path, "w+", encoding="utf-8")


    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.DictReader(f)
        i = 0
        for row in reader:
            content_line = ''
            i += 1
            for column in columns:
                content_line += cut(row[column])

            output.write(content_line + "\n")
            print("line {} is finished : {}".format(i, content_line))


    output.close()
    print("csv finished")


p = r'\”|\《|\。|\{|\！|？|｡|\＂|＃|＄|％|\＆|\＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|＠|\［|\＼|\］|\＾|＿|｀|\～|｟|｠|\、|〃|》|「|」|『|』|【|】|〔|〕|〖|〗|〘|〙|〚|〛|〜|\〝|\〞|〟|〰|〾|〿|–—|\‘|\“|\„|\‟|\…|\‧|﹏|\.'

def strQ2B(ustring):
    return [re.sub(p, ' ', ustring)] #re.split(p,ustring)#

class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    return pickle.dump(obj, MacOSFile(file_path), protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(file_path):
    return pickle.load(MacOSFile(file_path))

if __name__ == '__main__':
    dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep
    print('dir = ', dir)
    saveRawDataset(contentColumns=['comment', 'name'], labelColumn='star', pickle_dir=dir,
                input_file=dir + 'movie_comments.csv')
    (train_dataset, train_labels), (valid_dataset, valid_labels), (
        test_dataset, test_labels), labelsSet = getRawDataSet(dir+'s2v_w2v_raw.pickle')
    end = 0
    #saveDataset(contentColumns= ['comment','name'], labelColumn= 'star', pickle_dir=dir,input_file=dir+'movie_comments.csv')
    #(train_dataset, train_labels), (valid_dataset, valid_labels), (
    #    test_dataset, test_labels), labelsSet = getDataSet(dir+'s2v_w2v.pickle')
    #tokenizeFormCsv(input_file='/Users/henry/Documents/application/nlp_assignments/data/movie_comments.csv',columns= ['comment','name'],save_file_path='/Users/henry/Documents/application/nlp_assignments/data/movie_comments.txt')
