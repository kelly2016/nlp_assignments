# -*- coding: utf-8 -*-
# Filename: sentence2Vector.py
# @Time    : 2019-09-14 10:13
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : Sentence2Vector.py
# @Description:用不同的方案生成句子向量，有时间在做
import jieba
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import  preprocessing
import numpy as np
import os

#stopword_list =preprocessing.get_stopwords()

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
        words = preprocessing.cut2list(sentence)
        sen_vec = np.sum([self.model.wv[k] for k in words if k in self.model], axis=0) / len(words)

        return sen_vec

def d2vfSentence2Vector():
    pass


def tfidfSentence2Vector():
    pass




if __name__=='__main__':
    pass