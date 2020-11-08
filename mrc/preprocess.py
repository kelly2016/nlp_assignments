# -*- coding: utf-8 -*-
# @Time    : 2020-01-26 16:23
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : preprocess.py
# @Description:


import json
import multiprocessing
import os
import setproctitle
from collections import Counter

cores = multiprocessing.cpu_count()
partitions = cores


from cutWords import Analyzer




def loadJsonData(file):

    with open(file, 'r') as f:
        data = json.load(f)
        return data



def preprocess(file,counter,analyzer):
    """
    和业务相关的字符串处理
    :param str:
    :param counter:计数器
    :return:
    """

    #tokens
    for f in file:
            datas= loadJsonData(f)
            data = datas['data']  # [0]('paragraphs')
            for d in data:
                pg = d['paragraphs']
                for p in pg:
                    token = analyzer.cut2list(p['context'])
                    #tokens += token

                    counter.update(token)

    return counter


def preprocessing(file,dictFile,analyzer):
    """
    构造词典词频
    :param tokens:词语
    :param file:词典文件
    :return:
    """

    counter = preprocess(file, Counter(),analyzer)

    for k, v in counter.items():
        print(k, v)

    sorted(counter.items(), key=lambda x:x[1],reverse=True)
    for k, v in counter.items():
        print(k, v)

    output = open(dictFile, "w+", encoding="utf-8")
    for key, value in counter.items():
        output.write(key + ' ' + str(value) + "\n")
        print('write:', (key + ' ' + str(value) + "\n"))

    output.close()









if __name__=='__main__':
    setproctitle.setproctitle('kelly')
    dir = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'rc' + os.sep + 'dureader' + os.sep

    file = [dir+"demo_dev.json"]
    dictFile = dir+"/cnDict.txt"
    analyzer = Analyzer(Analyzer.ANALYZERS.Jieba, replaceP=True, useStopwords=True)
    preprocessing(file,dictFile,analyzer)

    preprocessing(dir)