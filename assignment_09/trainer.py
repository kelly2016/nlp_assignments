# -*- coding: utf-8 -*-
# @Time    : 2019-09-02 14:56
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : traditonalTrainer.py
# @Description:

import  classification
import imagePreprocess

def traditonalTrain():
    """
    RandomForest accuracy:0.9524
              precision    recall  f1-score   support

           0       0.97      0.96      0.96      1000
           1       0.93      0.95      0.94      1000
           2       0.97      0.95      0.96      1000
           3       0.96      0.96      0.96      1000
           4       0.93      0.94      0.94      1000
           5       0.96      0.96      0.96      1000
           6       0.93      0.95      0.94      1000
           7       0.97      0.95      0.96      1000
           8       0.94      0.94      0.94      1000
           9       0.96      0.96      0.96      1000

   micro avg       0.95      0.95      0.95     10000
   macro avg       0.95      0.95      0.95     10000
    :return:
    """
    (train_dataset, train_labels), (_, _), (test_dataset, test_labels) = imagePreprocess.getDataSet()

    nsamples, nx, ny = train_dataset.shape
    d2_train_dataset = train_dataset.reshape((nsamples, nx * ny))
    nsamples_t, nx_t, ny_t = test_dataset.shape
    d2_test_dataset = test_dataset.reshape((nsamples_t, nx_t * ny_t))

    # RandomForest
    rf = classification.RandomForest()
    rf.train(d2_train_dataset, train_labels)
    pred_labels = rf.predict(d2_test_dataset)
    target_names = ['0','1','2','3','4','5','6','7','8','9']
    classification.classificationreport('RandomForest', test_labels, pred_labels, target_names)



if __name__=='__main__':
    traditonalTrain()


