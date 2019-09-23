# -*- coding: utf-8 -*-
# @Time    : 2019-09-23 12:04
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : extractiveText.py
# @Description:抽取式摘要生成
import  numpy as np
WORDFILE  =  ''# word vector file, can be downloaded from GloVe website; it's quite large but you can truncate it and use only say the top 50000 word vectors to save time
WEIGHTFILE = ''# each line is a word and its frequency
N = 5

def extract(text):
    """

    :param text:
    :return: 返回抽取后的结果
    """

    oriSentences = getSentences(text)
    sentences = preproccessSentences(oriSentences)
    embeddings = getSentences(sentences)
    totalEmbedding = getSentences(getTotalSentences(text))
    topSentences = getTopSentences(embeddings,totalEmbedding)

    return organize(topSentences,oriSentences)

def organize(topSentences,oriSentences):
    """
    组织句顺
    :param topSentences:
     :param oriSentences: 原始句子
    :return: 
    """
    text = ''
    return text


def getTopSentences(embeddings,totalEmbedding,N=N):
    """
    获取和全文句子向量最相似的n句话
    :param embeddings:
    :param totalEmbedding: 全文句子向量
    :return:
    """


def getSentences(sentences):
    """
    获取句子向量
    :param sentences:
    :return:
    """
    embeddings = None
    return embeddings

def getTotalSentences(text):
    """
    把全文当成一句话，进行预处理
    :param text:
    :return:
    """
def preproccessSentences(text):
    pass
def getSentences(text):
    """
    对文档分句
    :param text:
    :return:
    """
    pass

def cosine_dis(vector1, vector2):
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))


if __name__=='__main__':
    pass