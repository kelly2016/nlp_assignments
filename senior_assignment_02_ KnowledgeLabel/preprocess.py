# -*- coding: utf-8 -*-
# @Time    : 2020-01-26 16:23
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : preprocess.py
# @Description:


import multiprocessing
import os
import setproctitle

import numpy as np
import pandas as pd

cores = multiprocessing.cpu_count()
partitions = cores

from cutWords import Analyzer

analyzer = Analyzer(Analyzer.ANALYZERS.Jieba,replaceP=True,useStopwords=True)




def preprocess2(string):
    return preprocess(string,flag = 2)

def preprocess(string,flag = 1):
    """
    和业务相关的字符串处理
    :param str:
    :return:
    """
    if string is  None or  type(string) != str or len(string) == 0:
        return ' '
    content = string.split('[知识点：]')

    if flag == 1:
        item = analyzer.cut2list(content[0].replace('知识点','').replace('【考点精析】','').replace('[题目]','').replace('\t','').replace('\n','').replace('|','。').replace('【解答】','').replace(',','，').replace(',',""))#。这么做是因为ltp的切词好怪异"。德尔福"在一起
        return ' '.join(item)
    elif flag == 2:
        if len(content) < 2:
            return ' '
        knowledge =  analyzer.cut2list(content[1])
        return' '.join(knowledge)



def parallelize(df,func):
    """

    :param df:
    :param func:
    :return:
    """
    #将df横切
    data_split = np.array_split(df,partitions)
    pool  = multiprocessing.Pool(cores)
    data = pd.concat(pool.map(func,data_split))
    #关闭pool，使其不在接受新的任务。
    pool.close()
    #主进程阻塞，等待子进程的退出， join方法要在close或terminate之后使用。
    pool.join()
    return data

def data_fram_proc(df):
    df['items']= df['item'].apply(preprocess)
    df['knowledge'] = df['item'].apply(preprocess2)

    return df

def preprocessing(path,ratio=0.8):
    """
    进行数据预处理，获取训练集和测试集
    :param file:语料文件
    :param ratio:测试训练的比列
    :return:
    """

    #for root ,dirs,files in os.walk(path):
    dirs = os.listdir(path)
    for file in dirs:
        print(os.path.join(path,file))
        path2 = os.path.join(path, file)
        if os.path.isdir(path2) :
           dirs2 = os.listdir(path2)
           for file2 in dirs2:
                file3 = os.path.join(path2, file2)
                print(file3)
                if os.path.isfile(file3) and  file2.endswith('.csv') and not file2.endswith('_cleaned.csv'):
                    src_df = pd.read_csv(file3)

                    src_df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
                    src_df.dropna(subset=['item'], how='any', inplace=True)
                    src_df = parallelize(src_df, data_fram_proc)
                    src_df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
                    src_df.dropna(subset=['items'], how='any', inplace=True)
                    #merged_df = pd.concat([src_df['items'],src_df['knowledge']], axis=0)
                    merged_df  = src_df.loc[:,["items","knowledge"]]
                    name = file2.split('.')[0]
                    merged_df.to_csv(os.path.join(path2, name+'_cleaned.csv'), index=None, header=True)



if __name__=='__main__':
    setproctitle.setproctitle('kelly')
    dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'KnowledgeLabel' + os.sep+'corpus2' + os.sep
    preprocessing(dir)