# -*- coding: utf-8 -*-
# @Time    : 2019-11-19 17:30
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : util.py
# @Description:常用工具类

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
        we.append(ws[0])
        v = wv[ws[0]]
        if v is None:
            print('{} is null'.format(ws[0]))
        else:
            embedding_matrix.append(wv[ws[0]])
    return (we, embedding_matrix)

