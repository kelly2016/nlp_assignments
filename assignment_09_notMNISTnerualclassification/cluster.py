# -*- coding: utf-8 -*-
# @Time    : 2019-09-09 12:07
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : cluster.py
# @Description:

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import  classification



class Kmeans(object):
    def __init__(self,n_clusters=3,miniBatch=False,batch_size=100,random_state=9):
        """
        :param miniBatch:是否用批聚类
        :param n_clusters: 聚类数
        :param  random_state 随机生成簇中心的状态条件,譬如设置random_state = 9
        :param batch_size：即用来跑Mini Batch KMeans算法的采样集的大小，默认是100.如果发现数据集的类别较多或者噪音点较多，需要增加这个值以达到较好的聚类效果。

        """
        self.miniBatch =miniBatch
        if not miniBatch:
            self.km = kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        else:#random_state 随机生成簇中心的状态条件,譬如设置random_state = 9
             #batch_size：即用来跑Mini Batch KMeans算法的采样集的大小，默认是100.如果发现数据集的类别较多或者噪音点较多，需要增加这个值以达到较好的聚类效果。
            self.km = MiniBatchKMeans(n_clusters=n_clusters,random_state=random_state,batch_size=batch_size)


    def train(self,X_train):
        self.km.fit(X_train)


    def cluster_centers(self):
        return self.km.cluster_centers_


    def labels(self):
        return self.km.labels_

    def predict(self,X_test):
        """

        :param X: 测试列表
        :return:
        """
        return self.km.predict(X_test)

if __name__=='__main__':
    fname = '/Users/henry/Documents/application/nlp_assignments/data/sqlResult_1558435.csv'
    X_train_tfidf_model, y_train, X_test_tfidf_model, y_test = classification.preprocessing(fname)
    print('data has been prepared')
    km = Kmeans()
    km.train(X_train_tfidf_model)
    print(km.cluster_centers())
    print(km.labels())
