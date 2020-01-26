# -*- coding: utf-8 -*-
# @Time    : 2019-08-21 17:17
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : newsClassification.py
# @Description:用各种分类算法预测判断是否是新华社的文章
import datetime
import os
import setproctitle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

vectorized = TfidfVectorizer(max_features = 500)#





def preprocessing(path,ratio=0.8):
    """
    进行数据预处理，获取训练集和测试集
    class biological分子与细胞_cleaned.csv : 11
class biological现代生物技术专题_cleaned.csv : 12
class biological生物技术实践_cleaned.csv : 13
class biological生物科学与社会_cleaned.csv : 14
class biological稳态与环境_cleaned.csv : 15
class biological遗传与进化_cleaned.csv : 16
class geography分子与细胞_cleaned.csv : 21
class geography现代生物技术专题_cleaned.csv : 22
class geography生物技术实践_cleaned.csv : 23
class geography生物科学与社会_cleaned.csv : 24
class geography稳态与环境_cleaned.csv : 25
class geography遗传与进化_cleaned.csv : 26
class history古代史_cleaned.csv : 31
class history现代史_cleaned.csv : 32
class history近代史_cleaned.csv : 33
class political公民道德与伦理常识_cleaned.csv : 41
class political时事政治_cleaned.csv : 42
class political生活中的法律常识_cleaned.csv : 43
class political科学思维常识_cleaned.csv : 44
class political科学社会主义常识_cleaned.csv : 45
class political经济学常识_cleaned.csv : 46
    :param file:语料文件
    :param ratio:测试训练的比列
    :return:
    """
    dirs = os.listdir(path)
    x_list = []
    y_list = []
    label11 = 0

    for file in dirs:
        #print(os.path.join(path, file))
        path2 = os.path.join(path, file)
        if os.path.isdir(path2):
            category = file
            dirs2 = os.listdir(path2)
            label12 = 0
            for file2 in dirs2:
                file3 = os.path.join(path2, file2)
                if os.path.isfile(file3) and file2.endswith('.csv'):
                    print('class {}{} : {}{}'.format(file, file2, label11, label12))
                    src_df = pd.read_csv(file3)
                    x = np.array(src_df['item']).tolist()
                    x_list += x# list
                    y_list += [str(label11)+''+str(label12) for i in range(len(x))]
                    bug = 0
                label12 += 1
        label11 += 1



    X = vectorized.fit_transform(x_list)
    print(X.shape)
    #x, y = zip(*datas)
    x_train, x_test, y_train, y_test = train_test_split(X.toarray(), y_list, test_size = 0.2,random_state=111)


    print(len(x_train))
    print(len(y_train))
    return x_train, y_train, x_test, y_test

    #X_train_tfidf_model = vectorized.fit_transform(x_train)

    #X_test_tfidf_model = vectorized.transform(x_test)

    #ss = StandardScaler()
    #x_train_s = ss.fit_transform(x_train)
    #x_test_s = ss.transform(x_test)


    #return X_train_tfidf_model,y_train,X_test_tfidf_model,y_test


def classificationreport(classmethodBame ,y_true , y_pred ,target_names ):
    """

    :param y_true:真实数值
    :param y_pred: 预测数值
    :param target_names:类别label
    :return:
    """
    print('{} accuracy:{}'.format(classmethodBame,accuracy_score(y_true, y_pred)))
    #print(classification_report(y_true, y_pred, target_names=target_names))
    print(classification_report(y_true, y_pred))

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


class Svm(object):
    def __init__(self,dict = None,class_weight = {1:1,0:1}):
        """

        :param dict:  各类的权重 {1:10 ,0:1}
        """
        self.clf  = LinearSVC(C=0.7,class_weight=class_weight)
            #svm.SVC(C=0.7,class_weight={1:1,0:1},kernel='sigmoid')#ovr
        #
        #NuSVC
            #
        #C=1.0,class_weight='balanced' if dict==None else dict,kernel='rbf',gamma='auto',decision_function_shape='ovo'
        #C、kernel、degree、gamma

    def train(self,X_train,y_train):
        self.clf.fit(X_train,y_train)

    def predict(self,X_test):
        """

        :param X: 测试列表
        :return:
        """
        return self.clf.predict(X_test)


class NaiveBayes(object):

    def __init__(self,classes = None):
        """

        :param dict:  各类的权重 {1:10 ,0:1}
        """
        self.gnb = MultinomialNB() #,GaussianNB GaussianNB()#
        self.classes = classes


    def train(self,X_train,y_train):
        self.gnb.fit(X_train,y_train)

        #self.gnb.partial_fit(X_train,y_train,classes=self.classes)

    def predict(self,X_test):
        """

        :param X: 测试列表
        :return:
        """
        return self.gnb.predict(X_test)

class RandomForest(object):
    def __init__(self,n_estimators=100):
        """

        :param n_estimators: 随机生成树的个数
        """
        self.rf = RandomForestClassifier(n_estimators=n_estimators,oob_score=True)



    def train(self,X_train,y_train):
        self.rf.fit(X_train,y_train)

    def predict(self,X_test):
        """

        :param X: 测试列表
        :return:
        """
        return self.rf.predict(X_test)




if __name__=='__main__':
    setproctitle.setproctitle('kelly')
    dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'KnowledgeLabel' + os.sep+'corpus' + os.sep

    X_train_tfidf_model, y_train, X_test_tfidf_model, y_test = preprocessing(dir)
    target_names = ['11','12','13','14','15','16','21','22','23','24','25','26','31','32','33','41','42','43','44','45','46']
    print('data has been prepared')

    # NB
    nb = NaiveBayes(target_names)
    nb.train(X_train_tfidf_model, y_train)
    y_pred = nb.predict(X_test_tfidf_model)
    classificationreport('NaiveBayes', y_test, y_pred, target_names)


    #knn
    knn = kNN(5)
    knn.train(X_train_tfidf_model,y_train)
    y_pred = knn.predict(X_test_tfidf_model)
    #print(accuracy_score(y_test, y_pred))
    #print(metrics.recall_score(y_test, y_pred, average='micro'))
    #print( metrics.f1_score(y_test, y_pred, average='weighted'))

    classificationreport('kNN',y_test, y_pred,target_names)

    #LR
    lr = Logistic_Regression()
    lr.train(X_train_tfidf_model,y_train)
    y_pred = lr.predict(X_test_tfidf_model)
    classificationreport('LogisticRegression',y_test, y_pred,target_names)
     


    #SVM
    start = datetime.datetime.now()
    sm = Svm(class_weight = None)
    #{'11':0.04,'12':0.04,'13':0.04,'14':0.04,'15':0.04,'16':0.04,'21':0.04,'22':0.04,'23':0.04,'24':0.04,'25':0.04,'26':0.04,'31':0.04,'32':0.01,'33':0.01,'41':0.01,'42':0.01,'43':0.01,'44':0.01,'45':0.01,'46':0.01})

    sm.train(X_train_tfidf_model, y_train)
    y_pred = sm.predict(X_test_tfidf_model)
    classificationreport('SVM',y_test, y_pred, target_names)
    end = datetime.datetime.now()
    print('run time = ',end - start)

    #RandomForest
    rf = RandomForest()
    rf.train(X_train_tfidf_model, y_train)
    y_pred = rf.predict(X_test_tfidf_model)
    classificationreport('RandomForest', y_test, y_pred, target_names)

   

