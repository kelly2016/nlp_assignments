# -*- coding: utf-8 -*-
# @Time    : 2019-11-19 17:30
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : util.py
# @Description:常用工具类
import os
import setproctitle

from gensim.models import FastText


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

