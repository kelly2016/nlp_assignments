# -*- coding: utf-8 -*-
# @Time    : 2019-10-22 15:22
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : fineGainedCommentTrainer.py
# @Description:

from keras.callbacks import Callback

def getCorpus():
    """
    获取语料
    :return:
    """
    pass

def train():
    """
    训练
    :return:
    """
    pass

class Model(object):
    """
    训练的模型
    """
    pass

class RocAucEvaluation(Callback):
    """
    评价函数
    """
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))


if __name__=='__main__':
    pass