# -*- coding: utf-8 -*-
# @Time    : 2019-09-23 12:59
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : wordsTask.py
# @Description:用faskText 生成词向量c词频文件

import multiprocessing
import os
import setproctitle

import matplotlib.pylab as plt
# from gensim.models.utils_any2vec import _save_word2vec_format, _load_word2vec_format, _compute_ngrams, _ft_hash
import pandas as pd
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from sklearn.manifold import TSNE

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def train(corpusFile,modelFile,vectorFile):
    """

    :param corpusFile: 训练模型语料
    :param modelFile: 保存的模型文件地址
    :param vectorFile: 保存的词量文件地址
    :return:
    """

    sentences = []
    src_df = pd.read_csv(corpusFile, encoding='utf-8', sep='\t')
    words = []
    for j, values in enumerate(src_df.values):
        sentences += [values[0].split()]


    print('initial training   ')
    model = FastText( min_count=2,size=300)  # instantiate
    #云服务器的老版本
    print('start  build_vocab  ')

    model.build_vocab(sentences=sentences)  # scan over corpus to build the vocabulary
    print('start training   ')
    model.train(sentences, total_examples=len(sentences), epochs=10,workers=multiprocessing.cpu_count())
    '''
    model.build_vocab(corpus_file=corpusFile)  # scan over corpus to build the vocabulary
    total_words = model.corpus_total_words  # number of words in the corpus
    print('start training   ')
    model.train(corpus_file=corpusFile, total_words=total_words, epochs=10,workers=multiprocessing.cpu_count())
    '''
    print('training finished  ' )
    fname = get_tmpfile(modelFile)
    model.save(fname)
    model.wv.save_word2vec_format(vectorFile, binary=False)
    print('training finished the modelFile = {} and the vectorFile = {} '.format(modelFile, vectorFile))
    return modelFile,vectorFile

def retrain(corpusFile,modelFile,vectorFile):
    """


    增量训练需要的文件
    :param corpusFile: 训练模型语料
    :param modelFile: 保存的模型文件地址
    :param vectorFile: 保存的词量文件地址
    :return:
    """
    print('modelFile=', modelFile)
    print('initial training   ')
    model = FastText.load(modelFile)  # instantiate
    sentences = []
    src_df = pd.read_csv(corpusFile, encoding='utf-8', sep='\t')
    for j, values in enumerate(src_df.values):
        sentences += [values[0].split()]
    print('start  build_vocab  ')
    model.build_vocab(sentences=sentences, update=True)
    model.train(sentences=sentences, total_examples=model.corpus_count, epochs=50, workers=multiprocessing.cpu_count())
    print('end train')
    fname = get_tmpfile(modelFile)
    model.save(fname)
    model.wv.save_word2vec_format(vectorFile, binary=False)
    print('retrain finished the modelFile = {} and the vectorFile = {} '.format(modelFile,vectorFile))
    return modelFile,vectorFile

def retrains(corpusFiles,modelFile,vectorFile):
    """


    增量训练需要的文件
    :param corpusFile: 训练模型语料
    :param modelFile: 保存的模型文件地址
    :param vectorFile: 保存的词量文件地址
    :return:
    """
    print('modelFile=', modelFile)
    model = FastText.load(modelFile)  # instantiate
    count = 0
    for corpusFile in corpusFiles:
        sentences = []
        src_df = pd.read_csv(corpusFile, encoding='utf-8', sep='\t')
        for j, values in enumerate(src_df.values):
            sentences += [values[0].split()]
        print('statrt train the {} file -- {} '.format(count,corpusFile))
        model.build_vocab(sentences=sentences, update=True)
        model.train(sentences=sentences, total_examples=model.corpus_count, epochs=2, workers=multiprocessing.cpu_count())
        count += 1
    print('end train')
    fname = get_tmpfile(modelFile)
    model.save(fname)
    model.wv.save_word2vec_format(vectorFile, binary=False)
    print('retrains finished the modelFile = {} and the vectorFile = {} '.format(modelFile,vectorFile))
    return modelFile,vectorFile


def fastTextTest(modelFile):
    """
    测试模型
    :param modelFile:
    :return:
    """
    model = FastText.load(modelFile)
    print('wv.vector_size' , model.wv.vector_size)

    print('美方' in model.wv.vocab)
    print(model.most_similar("说"))
    #print(fastTextNgramsVector(model))
   #查看字典
    #for word in model.wv.vocab:
        #print("{}'s vector is {}".format(word,model.wv[word]))
   #词向量存储在model.wv的KeyedVectors实例中，可以直接在KeyedVectors中查询词向量。
    print("1 .{} 's vector is {}".format('研究',model.wv['研究']))
    #第二个应用是看两个词向量的相近程度，这里给出了书中两组人的相似程度：
    print("2 .{} and {} 's similarity is {}".format(u'定理',u'公理',model.wv.similarity(u'定理',u'公理')))
    #计算一个词的最近似的词，倒排序
    print("3 .{} 's  most similar are {}".format('说', model.wv.most_similar(['说'])))
    #查找异类词
    print("4 . Which is different In {}? is {} ".format('中国,美国,叙利亚,水果',model.wv.doesnt_match(['中国','美国','叙利亚','水果'])))
    #word2vec一个很大的亮点：支持词语的加减运算（实际中可能只有少数例子比较符合）
    print('5 .',model.wv.most_similar(positive = ['概念','学科'],negative = ['结构'],topn = 4))
    #计算两个集合之间的余弦似度,两句话的相似度
    list1 = ['贸易','利用','数学']#
    list2 = ['研究','某种','计算']#
    list_sim1 = model.wv.n_similarity(list1,list2)
    print('6 .',list_sim1)


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

def view(modelFile):
    """
    可视化查看
    :return:
    """
    model = FastText.load(modelFile)
    tsne_plot(model)

'''
def fastTextNgramsVector(fasttext_model):
    fasttext_word_list = fasttext_model.wv.vocab.keys()
    ngramsVector = {}
    ngram_weights = fasttext_model.wv.vectors_ngrams # (10, 4)
    for word in fasttext_word_list:
        ngrams = _compute_ngrams(word,min_n = fasttext_model.wv.min_n,max_n = fasttext_model.wv.max_n)
        for ngram in ngrams:
            ngram_hash = _ft_hash(ngram) % fasttext_model.wv.bucket
            if ngram_hash in fasttext_model.wv.hash2index:
                ngramsVector[ngram] = ngram_weights[fasttext_model.wv.hash2index[ngram_hash]]
    return ngramsVector
'''

if __name__=='__main__':
    setproctitle.setproctitle('kelly')
    dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +  os.sep+'data'+ os.sep+ 'AutoMaster' + os.sep
    print('dir = ',dir)
    modelFile = dir +'fasttext3/fasttext_jieba.model'#
    train_x_pad_path = dir + 'AutoMaster_Train_X.csv'
    train_y_pad_path = dir + 'AutoMaster_Train_Y.csv'
    test_x_pad_path = dir + 'AutoMaster_Test_X.csv'
    train(corpusFile=dir+'trainv2wcotpus_jieba_2.csv',modelFile=modelFile , vectorFile=dir+'fasttext_jieba_2.v')#
    #retrain(dir + 'trainv2wcotpus_ltp.csv', modelFile, dir + 'fasttext_ltp.v')
    # retrains(corpusFiles=[train_x_pad_path,train_y_pad_path,test_x_pad_path], modelFile=modelFile, vectorFile=dir+'fasttext_jieba.v')

    fastTextTest(modelFile)
    view(modelFile)

