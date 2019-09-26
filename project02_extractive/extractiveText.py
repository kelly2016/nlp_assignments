# -*- coding: utf-8 -*-
# @Time    : 2019-09-23 12:04
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : extractiveText.py
# @Description:抽取式摘要生成
import  numpy as np
import pyltpAnalyzer
import re
from sentence_embedding import data_io,params, SIF_embedding
from functools import cmp_to_key
import jieba
import fastText

WORDFILE  =  ''# word vector file, can be downloaded from GloVe website; it's quite large but you can truncate it and use only say the top 50000 word vectors to save time
WEIGHTFILE = ''# each line is a word and its frequency
N = 5
PUNCTUATION_PATTERN = r'\”|\《|\。|\{|\！|？|｡|\＂|＃|＄|％|\＆|\＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|＠|\［|\＼|\］|\＾|＿|｀|\～|｟|｠|\、|〃|》|「|」|『|』|【|】|〔|〕|〖|〗|〘|〙|〚|〛|〜|\〝|\〞|〟|〰|〾|〿|–—|\‘|\“|\„|\‟|\…|\‧|﹏|\.'
MODELFILE = ''
analyzer = pyltpAnalyzer.PyltpAnalyzer()
model = fastText.FastText.load(MODELFILE)
# input
wordfile = '/Users/henry/Documents/application/newsExtract/news/data/ltp_data/glove.840B.300d.txt' # word vector file, can be downloaded from GloVe website; it's quite large but you can truncate it and use only say the top 50000 word vectors to save time
weightfile = '/Users/henry/Documents/application/newsExtract/news/data/ltp_data/zhwiki_vocab_fre.txt' # each line is a word and its frequency
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme


def extract(text):
    """

    :param text:
    :return: 返回抽取后的结果
    """
    #分句
    oriSentences = pyltpAnalyzer.text2sentences(text)
    #对句子进行分词预处理
    sentences = preproccessSentences(oriSentences)
    #获得每句话的句向量
    embeddings = getSentencesEmbeddings(sentences)
    #将全文当成一句话，获得其句子向量
    totalEmbedding = getSentencesEmbeddings(preproccessSentences([text]))
    #将每一句和全文句子比较获得排名前N的句子
    topSentences = getTopSentences(embeddings,totalEmbedding)

    return organize(topSentences,oriSentences)

def organize(scores,oriSentences):
    """
    组织句顺
    :param topSentences:
     :param oriSentences: 原始句子
    :return: 
    """
    #按句子顺序排
    def mycmp(score_a, score_b):
        if score_a[1] > score_b[1]:
            return 1
        else:
            return -1

    scores.sort(key=cmp_to_key(mycmp))
    print('scores=',scores)
    text = ''
    for score in scores:
        text += (oriSentences[score[1]]+'。')
    return text


def getSentencesEmbeddings(sentences):
    """
    获取句子向量
    :param sentences:
    :return:
    """
    embeddings = None

    # load word weights
    word2weight = data_io.getWordWeight(weightfile, weightpara)  # word2weight['str'] is the weight for the word 'str'
    # load sentences
    x, m = data_io.sentences2matrixFromModel(sentences)  # x is the array of word , m is the binary mask indicating whether there is a word in that location
    w = data_io.seq2weightFromFreq(x, m, word2weight)  # get word weights

    # set parameters
    parameters = params.params()
    parameters.rmpc = rmpc
    # get SIF embedding
    embedding = SIF_embedding.SIF_embeddingFromModel(model, x, w, parameters)  # embedding[i,:] is the embedding for sentence i

    return embeddings


def getSentencesEmbeddings2(sentences):
    """
    获取句子向量
    :param sentences:
    :return:
    """
    embeddings = None
    # load word vectors
    (words, We) = data_io.getWordmap(wordfile)
    # load word weights
    word2weight = data_io.getWordWeight(weightfile, weightpara)  # word2weight['str'] is the weight for the word 'str'
    weight4ind = data_io.getWeight(words, word2weight)  # weight4ind[i] is the weight for the i-th word
    # load sentences
    x, m = data_io.sentences2idx(sentences,
                                 words)  # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    w = data_io.seq2weight(x, m, weight4ind)  # get word weights

    # set parameters
    parameters = params.params()
    parameters.rmpc = rmpc
    # get SIF embedding
    embedding = SIF_embedding.SIF_embedding(We, x, w, parameters)  # embedding[i,:] is the embedding for sentence i

    return embeddings

def getTopSentences(embeddings,totalEmbedding,n=N):
    """
    把全文当成一句话，进行预处理
    :param embeddings:所有分句
    :param totalEmbedding:全文整句
    :param n:
    :return:
    """

    def mycmp(score_a, score_b):
        if score_a[0] < score_b[0]:
            return 1
        else:
            return -1
    scores = []
    for index,embedding in enumerate(embeddings):
        scores.append((cosine_dis(embedding, totalEmbedding),index))
    scores.sort(key=cmp_to_key(mycmp))
    return scores[:n]

def preproccessSentences(sentences):
    """
    对句子进行分词预处理
    :param text:
    :return:
    """

    def strQ2B(ustring):
        return
    newSentences = []
    if sentences is not None:
        for sentence in sentences:
            words = analyzer.segmentSentence(re.sub(PUNCTUATION_PATTERN, ' ', sentence))
            #words = jieba.cut(re.sub(PUNCTUATION_PATTERN, ' ', sentence), cut_all=False)
            newSentences.append(' '.join(words))
    return newSentences


def cosine_dis(vector1, vector2):
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))


if __name__=='__main__':
    def mycmp(score_a, score_b):
        if score_a[1] > score_b[1]:
            return 1
        else:
            return -1
    scores = [(23,3),(4,5),(6,7),(82,112),(0,22)]
    scores.sort(key=cmp_to_key(mycmp))
    print( scores[:4])