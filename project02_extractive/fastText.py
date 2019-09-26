# -*- coding: utf-8 -*-
# @Time    : 2019-09-23 12:59
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : wordsTask.py
# @Description:用faskText 生成词向量c词频文件

import setproctitle
import multiprocessing
import os
from gensim.models import FastText
#import matplotlib.pylab as plt
from sklearn.manifold import TSNE
from gensim.test.utils import get_tmpfile
#from gensim.models.utils_any2vec import _save_word2vec_format, _load_word2vec_format, _compute_ngrams, _ft_hash

#plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus'] = False

def train(corpusFile,modelFile,vectorFile):
    """

    :param corpusFile: 训练模型语料
    :param modelFile: 保存的模型文件地址
    :param vectorFile: 保存的词量文件地址
    :return:
    """
    sentences = []
    with open(corpusFile) as f:
        lines = f.readlines()
        for line in lines:
            sentences += [line.split()]
    model = FastText( min_count=1)  # instantiate
    #云服务器的老版本

    model.build_vocab(sentences=sentences)  # scan over corpus to build the vocabulary
    model.train(sentences, total_examples=len(sentences), epochs=10,workers=multiprocessing.cpu_count())
    '''
    model.build_vocab(corpus_file=corpusFile)  # scan over corpus to build the vocabulary
    total_words = model.corpus_total_words  # number of words in the corpus
    model.train(corpus_file=corpusFile, total_words=total_words, epochs=5,workers=multiprocessing.cpu_count())
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
    model = FastText.load(modelFile)  # instantiate
    model.build_vocab(corpus_file=corpusFile, update=True)
    total_words = model.corpus_total_words  # number of words in the corpus
    model.train(corpus_file=corpusFile, total_words=total_words, epochs=5,workers=multiprocessing.cpu_count())
    fname = get_tmpfile(modelFile)
    model.save(fname)
    model.wv.save_word2vec_format(vectorFile, binary=False)
    print('retrain finished the modelFile = {} and the vectorFile = {} '.format(modelFile,vectorFile))
    return modelFile,vectorFile


def fastTextTest(modelFile):
    """
    测试模型
    :param modelFile:
    :return:
    """
    model = FastText.load(modelFile)
    print('wv.vector_size' , model.wv.vector_size)

    print("文学's vector is {}".format( model.wv['文学']))
    print('美方' in model.wv.vocab)
    print(model.most_similar("文学"))
    #print(fastTextNgramsVector(model))
    print("希腊语's vector is {}".format(model.wv['希腊语']))
    print("商品's vector is {}".format(model.wv['商品']))
    #查看字典
    for word in model.wv.vocab:
        print("{}'s vector is {}".format(word,model.wv[word]))




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

'''
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
    setproctitle.setproctitle('newrun')
    dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +  os.sep+'data'+os.sep
    print('dir = ',dir)
    modelFile = dir +'fasttext.model'#
    train(corpusFile=dir+'wiki_corpus_ltp',modelFile=modelFile , vectorFile=dir+'fasttext.v')#
    #modelFile = 'fasttext.model'  #
    #train(corpusFile= 'test', modelFile=modelFile, vectorFile= 'fl.v')  #
    #retrain(dir + 'zh_wiki_corpus01', modelFile, dir + 'w2v.v')
    #retrain(dir + 'zh_wiki_corpus02', modelFile, dir + 'w2v.v')
    fastTextTest(modelFile)
    #view(modelFile)