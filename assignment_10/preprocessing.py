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
from sentence2Vector import W2V
import numpy as np

TRAIN_NUM = 0.8#训练集
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep

#word2vector的模型地址
s2v_w2v = W2V(DIR + 'w2v.model')

def getDataSet(pickle_file='/Users/henry/Documents/application/nlp_assignments/data/s2v_w2v.pickle'):
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

def saveDataset(contentColumns,labelColumn,pickle_dir,input_file=s2v_w2v):
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
    train_dataset =[]
    train_labels =[]
    valid_dataset =[]
    valid_labels =[]
    test_dataset =[]
    test_labels =[]

    print('dataset_num = {},trainMaxIndex = {} ,validMaxIndex = {} '.format(dataset_num,trainMaxIndex,validMaxIndex))
    for i,v in enumerate(indexArray):
        if i <= trainMaxIndex:
            train_dataset.append(datasets[v])
            train_labels.append(labelsSet[v])
        elif i > trainMaxIndex and i <= validMaxIndex:
            valid_dataset.append(datasets[v])
            valid_labels.append(labelsSet[v])
        else:
            test_dataset.append(datasets[v])
            test_labels.append(labelsSet[v])



    pickle_file = os.path.join(pickle_dir, 's2v_w2v.pickle')

    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
            'labelsSet': labelsSet,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)

def formatDataset(contentColumns,labelColumn,input_file):
    """
    将csv数据读出每个文档向量化（句向量，段向量，文档向量），然后将其序列化为([datasets],[datalabels],{label})存储起来
    :param file: csv数据集文件
    :return:([datasets],[datalabels],{label})
    """
    datasets = []
    datalabels = []
    labelsSet = set()
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content_line = ''
            for column in contentColumns:
                content_line += row[column]
                la = row[labelColumn]
            if  la!=labelColumn  and len(content_line.strip()) > 0:
               v = s2v_w2v.w2vfSentence2Vector(content_line)
               print(v.shape)
               datasets.append(v)
               datalabels.append(la)
               if la not in labelsSet:
                   labelsSet.add(la)


    print("dataset total number is  {} ,datalabels total number is  {}  : {}".format(len(datasets), len(datalabels)))
    return (datasets,datalabels,labelsSet)



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


if __name__ == '__main__':
    saveDataset(contentColumns= ['comment','name'], labelColumn= 'star', pickle_dir='/Users/henry/Documents/application/nlp_assignments/data/',input_file='/Users/henry/Documents/application/nlp_assignments/data/movie_comments.csv')
    (train_dataset, train_labels), (valid_dataset, valid_labels), (
        test_dataset, test_labels), labelsSet = getDataSet()
    #tokenizeFormCsv(input_file='/Users/henry/Documents/application/nlp_assignments/data/movie_comments.csv',columns= ['comment','name'],save_file_path='/Users/henry/Documents/application/nlp_assignments/data/movie_comments.txt')
