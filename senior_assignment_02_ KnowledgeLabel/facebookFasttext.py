# -*- coding: utf-8 -*-
# @Time    : 2020-02-02 12:17
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : faceBookFasttext.py
# @Description:利用facebook的fasttext进行分类
import os
import setproctitle

import fasttext


def retrains(modelfile):
    model = fasttext.load_model(modelfile)
    print(model.words)
    print(len(model.words))
    print(model.labels)
    print(model.predict("社会主义是中国人民的历史性选择,发展中国特色社会主义文化"))
    result = model.test(test_data_path,k = 1)
    print("P@1:", result[1])  # 准确率
    print("R@2:", result[2])  # 召回率
    return model

def train_unsupervised(train_data_path,test_data_path,modelfile,isQuantize = True):
    model = fasttext.train_unsupervised(model="skipgram",input=train_data_path, dim=300, epoch=50, lr=1.0, wordNgrams=3, verbose=2, minCount=1)
    predict(model)

def train_supervised(train_data_path,test_data_path,modelfile,isQuantize = True):
    """

    :param train_data_path: 训练集文件，
    :param test_data_path: 测试集文件
    :param modelfile:
    :param isQuantize:
    :return:
    """




    model = fasttext.train_supervised(input=train_data_path, dim=300,epoch=50, lr=1.0, wordNgrams=2, verbose=2, minCount=1)
    print(model.words)
    print(len(model.words))
    print(model.labels)
    print(model.predict("社会主义是中国人民的历史性选择,发展中国特色社会主义文化"))
    result = model.test(test_data_path)
    print("P@1:", result.p)  # 准确率
    print("R@2:", result.recall)  # 召回率
    print("Number of examples:", result.nexamples)  # 预测错的例子

    print_results(*model.test(test_data_path))
    if isQuantize is True:
         model.quantize(input=train_data_path, qnorm=True, retrain=True, cutoff=100000)
         print_results(*model.test(test_data_path))
         model.save_model(modelfile)
         print_results(*model.test(test_data_path))
    return model

def predict(model):
    print("细胞 :".format(model.get_word_vector("细胞")))
    print("社会主义是中国人民的历史性选择,发展中国特色社会主义文化 :".format(model.get_sentence_vector('社会主义是中国人民的历史性选择,发展中国特色社会主义文化')))


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


if __name__=='__main__':
    setproctitle.setproctitle('kelly')
    dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'KnowledgeLabel' + os.sep+'corpus2' + os.sep
    train_data_path = dir + 'train_data.csv'
    test_data_path = dir + 'test_data.csv'
    modelfile = dir + 'his.ftz'
    model = train_supervised(train_data_path, test_data_path, modelfile)
    #model = retrains(modelfile)
    #model = train_unsupervised(train_data_path, test_data_path, modelfile)
