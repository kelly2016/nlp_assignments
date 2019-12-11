# -*- coding: utf-8 -*-
# @Time    : 2019-11-19 17:30
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : util.py
# @Description:常用工具类
import os
import setproctitle

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import FastText

from plot_utils import *


def translate(sentence,max_length_inp,vocab,evaluate):
    sentence = pad_proc(sentence, max_length_inp, vocab)

    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))


def pad_proc(sentence, max_len, vocab):
    '''
    < start > < end > < pad > < unk >
    '''
    # 0.按空格统计切分出词
    words = sentence.strip().split(' ')
    # 1. 截取规定长度的词数
    words = words[:max_len]
    # 2. 填充< unk > ,判断是否在vocab中, 不在填充 < unk >
    sentence = [word if word in vocab else '<UNK>' for word in words]
    # 3. 填充< start > < end >
    sentence = ['<START>'] + sentence + ['<STOP>']
    # 4. 判断长度，填充　< pad >
    sentence = sentence + ['<PAD>'] * (max_len  - len(words))
    return ' '.join(sentence)



# 遇到未知词就填充unk的索引
def transform_data(sentence,vocab,unkownchar = '<UNK>'):
    unk_index = vocab[unkownchar]
    # 字符串切分成词
    words=sentence.split(' ')
    # 按照vocab的index进行转换
    ids=[vocab[word] if word in vocab else unk_index for word in words]
    return ids


def load_dataset(file,vocab):
    """
    将输入语料文字转换成索引index
    :param file:
    :param vocab:
    :return:
    """
    df = pd.read_csv(file, encoding='utf-8', header=None,  sep= '\t')
    # 将词转换成索引  [<START> 方向机 重 ...] -> [32800, 403, 986, 246, 231
    ids = df[df.columns[0]].apply(lambda x: transform_data(x, vocab))

    # 将索引列表转换成矩阵 [32800, 403, 986, 246, 231] --> array([[32800,   403,   986 ]]
    return np.array(ids.tolist())

def config_gpu():
    ## 获取所有的物理GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:

                tf.config.experimental.set_memory_growth(gpu, True)
                # 获取所有的逻辑GPU
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

def getEmbedding_matrix(dictFile,vectorFile):
    """
    通过读取字典文件和向量文件
    :param dictFile:
    :param vectorFile:
    :return:返回词表和对应的向量举证
    """
    wv = {}
    f = open(vectorFile, 'r')
    lines = f.readlines()
    for (n, i) in enumerate(lines):
        if n == 0:
            continue
        i = i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        wv[i[0]] = v


    f = open(dictFile, 'r')
    lines = f.readlines()
    we = []
    embedding_matrix = []
    for (i, w) in enumerate(lines):
        ws = w.split()
        if len(ws) != 3:
            print('the {} line {} is invalid'.format(i,w))
            continue
        we.append(ws[0])
        if ws[0] not in wv:
            print('{} is not in vectorFile'.format(ws[0]))

        v = wv[ws[0]]
        if v is None:
            print('{} is null'.format(ws[0]))
        else:
            embedding_matrix.append(wv[ws[0]])
    return (we, embedding_matrix)


def getEmbedding_matrixFromModel( modelFile):
    """
    从模型中构建词向量矩阵
    :param model:fastText
    :return:
    """
    model = FastText.load(modelFile)  # instantiate
    vocab = {word:index for index,word in enumerate(model.wv.index2word)}
    reverse_vocab = {index: word for index, word in enumerate(model.wv.index2word)}
    embedding_matrix = model.wv.vectors
    return vocab,reverse_vocab,embedding_matrix

if __name__=='__main__':
    setproctitle.setproctitle('kelly')
    dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep
    print('dir = ', dir)
    modelFile = 'AutoMaster/fastmodel/'
    getEmbedding_matrixFromModel(modelFile)

    dictFile =dir+'AutoMaster/AutoMaster_Counter.txt'
    vectorFile =dir+'AutoMaster/fasttext_jieba.v'
    we, embedding_matrix = getEmbedding_matrix(dictFile, vectorFile)
    print(we)
    print('---------')
    print(embedding_matrix)

