# -*- coding: utf-8 -*-
# @Time    : 2019-12-02 10:36
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : seq2seq.py
# @Description:seq2seq模型
import tensorflow as tf

class Seq2seq(object):
    def __init__(self, train_X,train_Y,embedding_dim,vocab,units,BATCH_SIZE = 32):
        """

        :param train_X: 训练集输入
        :param train_Y: 训练集输出
        :param embedding_dim: 词向量维度
        :param vocab: 词表
        :param units: 隐藏层单元数
        :param BATCH_SIZE:
        """

        assert train_X is not  None,'train_X can not be None '
        assert train_Y is not None, 'train_Y can not be None '
        #训练集的长度
        self.BUFFER_SIZE = len(train_X)
        #输入的长度
        self.max_length_inp =  train_X.shape[1]
        #输出的长度
        self.max_length_max_targ = train_Y.shape[1]

        self.BATCH_SIZE  = BATCH_SIZE
        #每一轮的步数，取整除 - 向下取接近除数的整数>>> 9//2  =  4   >>> -9//2  = -5
        self.steps_per_epoch = self.BUFFER_SIZE//BATCH_SIZE
        #词向量维度
        self.embedding_dim = embedding_dim
        #词表大小
        self.vocab_size = len(vocab)
        #隐藏层，单元数
        self.units = units
        #构建训练集
        #dataset = tf.data.Dataset.from_generator()
        #数据集不是很大的时候
        dataset = tf.data.Dataset.from_tensor_slices((train_X,train_Y)).shuffle(self.BUFFER_SIZE)
        #用于标示是否对于最后一个batch如果数据量达不到batch_size时保留还是抛弃
        dataset = dataset.batch(self.BATCH_SIZE,drop_remainder=True)





if __name__ == '__main__':
    pass
