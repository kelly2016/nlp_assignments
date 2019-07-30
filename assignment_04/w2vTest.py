# -*- coding: utf-8 -*-
# @Time    : 2019-07-29 20:27
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : w2vTest.py
# @Description:Word2Vector

import multiprocessing
import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import matplotlib.pylab as plt
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
    model = Word2Vec(LineSentence(corpusFile), size=100, window=5, min_count=2, workers=multiprocessing.cpu_count())
    model.save(modelFile)
    model.wv.save_word2vec_format(vectorFile, binary=False)
    return modelFile,vectorFile

def w2vTest(modelFile):
    """
    测试模型
    :param modelFile:
    :return:
    """
    model = Word2Vec.load(modelFile)
    #查看字典
    print(model.wv.vocab)
    #词向量存储在model.wv的KeyedVectors实例中，可以直接在KeyedVectors中查询词向量。
    print("1 .{} 's vector is {}".format('研究',model.wv['研究']))
    #第二个应用是看两个词向量的相近程度，这里给出了书中两组人的相似程度：
    print("2 .{} and {} 's similarity is {}".format(u'定理',u'公理',model.wv.similarity(u'定理',u'公理')))
    #计算一个词的最近似的词，倒排序
    print("3 .{} 's  most similar are {}".format('定理', model.most_similar(['定理'])))
    #查找异类词
    print("4 . Which is different In {}? is {} ".format('中国,美国,叙利亚,水果',model.wv.doesnt_match(['中国','美国','叙利亚','水果'])))
    #word2vec一个很大的亮点：支持词语的加减运算（实际中可能只有少数例子比较符合）
    print('5 .',model.wv.most_similar(positive = ['概念','学科'],negative = ['结构'],topn = 4))
    #计算两个集合之间的余弦似度,两句话的相似度
    list1 = ['政治学','是', '一门', '社会科学','学科']#['我','跑','我','学校']
    list2 = ['研究','政治','行为','政治体制']#['我','去','家']
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
    model = Word2Vec.load(modelFile)
    tsne_plot(model)


if __name__=='__main__':
    dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +  os.sep+'data'+os.sep
    modelFile = dir + 'w2v.model'
    #train(dir+'wiki_corpus',modelFile , dir+'w2v.v')
    #w2vTest(modelFile)
    view(modelFile)