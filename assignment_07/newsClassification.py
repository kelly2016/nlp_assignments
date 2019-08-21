# -*- coding: utf-8 -*-
# @Time    : 2019-08-21 17:17
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : newsClassification.py
# @Description:用各种分类算法预测判断是否是新华社的文章
import pandas as pd
from sklearn.metrics import classification_report
from  sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import  re
import math
from sklearn.metrics import accuracy_score
from sklearn import metrics
import random

PUNCTUATION_PATTERN = r'\”|\《|\。|\{|\！|？|｡|\＂|＃|＄|％|\＆|\＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|＠|\［|\＼|\］|\＾|＿|｀|\～|｟|｠|\、|〃|》|「|」|『|』|【|】|〔|〕|〖|〗|〘|〙|〚|〛|〜|\〝|\〞|〟|〰|〾|〿|–—|\‘|\“|\„|\‟|\…|\‧|﹏|\.'
stopword_list = [k.strip() for k in open('/Users/henry/Documents/application/nlp_assignments/data/stopwords.txt', encoding='utf8').readlines() if k.strip() != '']
stopword_list = set(stopword_list)
vectorized = TfidfVectorizer(max_features = 300)

def predict(model,textList,labelList):
    """
    用该模型 判断一篇文章是否是新华社的文章，如果判断出来是新华社的，但是，它的source并不是新华社的，那么，我们就说，这个文章是抄袭的新华社的文章
    :param text:
    :param label:
    :return:False ,盗版新华社；True ：原版
    """
    documents_words = [list(k for k in jieba.cut(re.sub(PUNCTUATION_PATTERN, ' ', document)) if k not in stopword_list)
                       for document in textList]

    documents = [" ".join(document_words) for document_words in documents_words]
    X_test_tfidf_model = vectorized.transform(documents)
    y_pred = model.predict(X_test_tfidf_model)
    print(y_pred.shape)
    ret = []
    for yi,label in zip(y_pred,labelList):
        if yi == 1 and label != '新华社':
            ret.append(False)
        else:
            ret.append(True)
    return ret




def preprocessing(file,ratio=0.8):
    """
    进行数据预处理，获取训练集和测试集
    :param file:语料文件
    :param ratio:测试训练的比列
    :return:
    """

    #cutWords = [k for k in jieba.cut(article) if k not in stopword_list]

    content = pd.read_csv(file, encoding='gb18030')
    documents = content.apply(lambda row: (str(row['author'])+' '+str(row['title'])+' '+str(row['content'])), axis=1).values.tolist()
    documents_words =  [list(k for k in jieba.cut(re.sub(PUNCTUATION_PATTERN, ' ', document)) if k not in stopword_list) for document in documents]
    #[list(jieba.cut(re.sub(PUNCTUATION_PATTERN, ' ', document))) for document in documents]
    documents = [" ".join(document_words) for document_words in documents_words]
    print(len(documents))
    y = [ 1 if yi == '新华社' else  0 for yi in content['source'] ]  #content['source'].values.tolist()
    #y = [ 1 if yi == '新华社' else  0 for yi in y ]
    print('-- {}   {} {}{}'.format(y[0],y[1],y[2],y[3]))
    Z = [(Xi, yi) for Xi, yi in zip(documents, y)]
    random.shuffle(Z)
    documents = []
    y = []
    for z in Z :
        documents.append(z[0])
        y.append(z[1])

    print('len(documents ) ={} and len(y ) = {}'.format(len(documents),len(y)))
    lastIndex = math.ceil(len(documents)*ratio)
    X_train = documents[:lastIndex]
    y_train = y[:lastIndex]
    print('len(y_train 0) ={},len(y_train) = {} '.format(len([x for x in y_train if x== 0]),len(y_train)))


    X_train_tfidf_model = vectorized.fit_transform(X_train)

    X_test = documents[lastIndex:]
    y_test = y[lastIndex:]
    print('len(y_test 0) ={} ,len(y_test) = {}'.format(len([x for x in y_test if x == 0]),len(y_test)))
    X_test_tfidf_model = vectorized.transform(X_test)

    #ss = StandardScaler()
    #x_train_s = ss.fit_transform(x_train)
    #x_test_s = ss.transform(x_test)

    return X_train_tfidf_model,y_train,X_test_tfidf_model,y_test


def classificationreport(y_true , y_pred ,target_names ):
    """

    :param y_true:真实数值
    :param y_pred: 预测数值
    :param target_names:类别label
    :return:
    """
    print(classification_report(y_true, y_pred, target_names=target_names))


class kNN(object):

    def __init__(self, k):
        """

        :param k: 超参k
        """
        self.knn = KNeighborsClassifier(n_neighbors=k)

    def train(self,X_train,y_train):
        self.knn.fit(X_train,y_train)

    def predict(self,X_test):
        """

        :param X: 测试列表
        :return:
        """
        return self.knn.predict(X_test)

class Logistic_Regression(object):
    def __init__(self):
        """

        :param k: 超参k
        """
        self.lr = LogisticRegression()


    def train(self,X_train,y_train):
        self.lr.fit(X_train,y_train)

    def predict(self,X_test):
        """

        :param X: 测试列表
        :return:
        """
        return self.lr.predict(X_test)






if __name__=='__main__':
    fname = '/Users/henry/Documents/application/nlp_assignments/data/sqlResult_1558435.csv'
    X_train_tfidf_model, y_train, X_test_tfidf_model, y_test = preprocessing(fname)
    target_names = ['0','1']
    #knn
    knn = kNN(5)
    knn.train(X_train_tfidf_model,y_train)
    y_pred = knn.predict(X_test_tfidf_model)
    #print(accuracy_score(y_test, y_pred))
    #print(metrics.recall_score(y_test, y_pred, average='micro'))
    #print( metrics.f1_score(y_test, y_pred, average='weighted'))

    classificationreport(y_test, y_pred,target_names)

    #LR
    lr = Logistic_Regression()
    lr.train(X_train_tfidf_model,y_train)
    y_pred = lr.predict(X_test_tfidf_model)
    classificationreport(y_test, y_pred,target_names)
    #预测盗版
    text = '“受够了！香港，不能再乱下去了！”日前，逾47万香港市民冒雨参加“反暴力、救香港”集会，发出反对暴力、呼唤稳定的香港社会主流声音。连日来，爱国爱港的正义力量不断汇聚，正义呼声响彻香江。“反暴力是香港现在最大及唯一的‘一大诉求’”，这是香港工商界知名人士吴光正的由衷感慨；“希望香港尽快恢复安宁”“还我们一个安稳日子”，这是香港市民的热切期盼……字字句句，无不表达着对暴力行径的强烈谴责，彰显了止暴制乱、恢复秩序的民心所向。'
    label = '网易新闻'
    print(predict(lr, [text], [label]))