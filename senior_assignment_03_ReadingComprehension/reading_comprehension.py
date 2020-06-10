# -*- coding: utf-8 -*-
# @Time    : 2020-06-05 17:00
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : reading_comprehension.py
# @Description:用于单论阅读理解的模型



from keras.backend.tensorflow_backend import _get_available_gpus
from keras.layers import Layer, Dense, Permute
from keras.models import Model
from keras.utils import multi_gpu_model
from model.bert4keras.backend import K
from model.bert4keras.models import build_transformer_model
from model.bert4keras.optimizers import Adam


class Reading_Comprehension(object):
    def __init__(self,config_path,checkpoint_path,best_model_file ,learing_rate= 2e-5,fine_tune=False,is_gpu = False,nbr_gpus=0):
        self.model = build_transformer_model(
            config_path,
            checkpoint_path,
        )

        output = Dense(2)(self.model.output)
        output = MaskedSoftmax()(output)
        # 因为在给出的predlabel 结构为labels = [[start_index], [start_index + len(a_token_ids) - 1]],permute函数仅切换轴的位置，dims参数告诉Keras您希望最终位置如何,将输入的维度按照给定模式进行重排
        #将结构[batchsize,maxsize,2] -> [batchsize,2,maxsize]
        output = Permute((2, 1))(output)
        self.nbr_gpus = nbr_gpus
        self.model = Model(self.model.input, output)
        if is_gpu is True:
            if self.nbr_gpus <=0:
                self.nbr_gpus = len(_get_available_gpus()) - 1
            if self.nbr_gpus <= 0:
                self.model = multi_gpu_model(self.model, gpus=self.nbr_gpus)
            print('{} 个GPU被使用'.format(self.nbr_gpus))


        self.best_model_file  = best_model_file
        if fine_tune:
            self.model.load_weights(best_model_file)
        self.model.summary()
        self.model.compile(
            loss=self.sparse_categorical_crossentropy,
            optimizer=Adam(learing_rate),
            metrics=[self.sparse_accuracy]
        )

    def getModel(self):
        """
        获得模型
        :return:
        """
        return self.model

    def sparse_categorical_crossentropy(self,y_true, y_pred):
        # y_true需要重新明确一下shape和dtype
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        y_true = K.one_hot(y_true, K.shape(y_pred)[2])
        # 计算交叉熵
        return K.mean(K.categorical_crossentropy(y_true, y_pred))

    def sparse_accuracy(self,y_true, y_pred):
        # y_true需要重新明确一下shape和dtype
        y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
        y_true = K.cast(y_true, 'int32')
        # 计算准确率
        y_pred = K.cast(K.argmax(y_pred, axis=2), 'int32')
        return K.mean(K.cast(K.equal(y_true, y_pred), K.floatx()))



class MaskedSoftmax(Layer):
    """
    在序列长度那一维进行softmax，并mask掉padding部分
    """
    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, 2)
            inputs = inputs - (1.0 - mask) * 1e12#？如果mask是1，inputs不变是需要计算的，如果mask是0，inputs将变成一个很大的负数，在之后的softmax中配合指数函数这一项就会为0
        return K.softmax(inputs, 1)

