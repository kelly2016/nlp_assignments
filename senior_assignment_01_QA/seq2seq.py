# -*- coding: utf-8 -*-
# @Time    : 2019-12-02 10:36
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : seq2seq.py
# @Description:seq2seq模型
import os
import setproctitle
import time

import numpy as np
import tensorflow as tf

import preprocessing
import util
from layer import Encoder, Decoder


class Seq2seq(object):
    def __init__(self, train_X,train_Y,vocab, reverse_vocab,embedding_matrix,units=1024,modelFile='data/checkpoints/training_checkpoints',BATCH_SIZE = 32,paddingChr = '<PAD>'):
        """
        :param train_X: 训练集输入
        :param train_X: 训练集输入
        :param modelFile: 词向量模型文件
        :param units: 隐藏层单元数
        :param BATCH_SIZE:
        :param paddingChr:语料中padding的字符
        """

        assert train_X is not  None,'train_X can not be None '
        assert train_Y is not None, 'train_Y can not be None '


        self.vocab = vocab
        self.reverse_vocab = reverse_vocab
        self.embedding_matrix =  embedding_matrix
        # 词向量维度
        self.embedding_dim = self.embedding_matrix.shape[1]
        # 词表大小
        self.vocab_size = len(self.vocab)
        self.pad_index = self.vocab[paddingChr]

        #训练集的长度
        self.BUFFER_SIZE = len(train_X)
        #输入的长度
        self.max_length_inp =  train_X.shape[1]
        #输出的长度
        self.max_length_max_targ = train_Y.shape[1]

        self.BATCH_SIZE  = BATCH_SIZE
        #每一轮的步数，取整除 - 向下取接近除数的整数>>> 9//2  =  4   >>> -9//2  = -5
        self.steps_per_epoch = self.BUFFER_SIZE//BATCH_SIZE
        #隐藏层，单元数
        self.units = units
        #from_logits=True 表示内部就会自动做softmax
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')#稀疏交叉墒

        self.encoder = Encoder(self.vocab_size, self.embedding_dim, self.embedding_matrix, self.units, self.BATCH_SIZE)
        self.decoder = Decoder(self.vocab_size, self.embedding_dim, self.embedding_matrix, self.units,self.BATCH_SIZE)

        self.optimizer = tf.keras.optimizers.Adam()
        #构建训练集
        #dataset = tf.data.Dataset.from_generator()
        #数据集不是很大的时候
        dataset = tf.data.Dataset.from_tensor_slices((train_X,train_Y)).shuffle(self.BUFFER_SIZE)
        #用于标示是否对于最后一个batch如果数据量达不到batch_size时保留还是抛弃
        dataset = dataset.batch(self.BATCH_SIZE,drop_remainder=True)

        checkpoint_dir = modelFile+'/training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                         encoder=self.encoder,
                                         decoder=self.decoder)

    def train(self):
        """
        训练函数
        :return:
        """
        EPOCHS = 10
        for epoch in range(EPOCHS):
            start = time.time()

            # 初始化隐藏层
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(self.dataset.take(self.steps_per_epoch)):
                #
                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 1 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / self.steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    @tf.function
    def train_step(self,inp, targ, enc_hidden):
        """

        :param inp: input
        :param targ: 目标
        :param enc_hidden: 隐藏层
        :return:
        """

        loss = 0

        with tf.GradientTape() as tape:
            # 1. 构建encoder inp?
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            # 2. 复制
            dec_hidden = enc_hidden
            # 3. <START> * BATCH_SIZE  BATCH_SIZE*1* self.embedding_dim ?
            dec_input = tf.expand_dims([self.vocab['<START>']] * self.BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # decoder(x, hidden, enc_output)
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

                loss += self.loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))
            #取出encoder和decoder中的变量参数
            variables =  self.encoder.trainable_variables +  self.decoder.trainable_variables
            #计算梯度，更新权重
            gradients = tape.gradient(loss, variables)
            #用优化器更新
            self.optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss

        def evaluate(sentence):
            attention_plot = np.zeros((self.max_length_targ, self.max_length_inp + 2))

            inputs = preprocessing.pad_proc(sentence, self.max_length_inp, vocab)

            inputs = tf.convert_to_tensor(inputs)

            result = ''

            hidden = [tf.zeros((1, self.units))]
            enc_out, enc_hidden = self.encoder(inputs, hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([vocab['<START>']], 0)

            for t in range(self.max_length_targ):
                predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                     dec_hidden,
                                                                     enc_out)

                # storing the attention weights to plot later on
                attention_weights = tf.reshape(attention_weights, (-1,))

                attention_plot[t] = attention_weights.numpy()
                predicted_id = tf.argmax(predictions[0]).numpy()

                result += reverse_vocab[predicted_id] + ' '
                if reverse_vocab[predicted_id] == '<STOP>':
                    return result, sentence, attention_plot

                # the predicted ID is fed back into the model
                dec_input = tf.expand_dims([predicted_id], 0)

            return result, sentence, attention_plot

    def loss_function(self,real, pred,):
        """
        损失函数
        :param real:真实值
        :param pred:预测值
        :return:
        """
        # 判断logit为1和0的数量,计算出<PAD>的数量有多少，并在mask中标记为0
        mask = tf.math.logical_not(tf.math.equal(real,  self.pad_index))
        # 计算decoder的长度，除去<PAD>字符数
        dec_lens = tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=-1)
        # 计算loss值
        loss_ = self.loss_object(real, pred)
        # 转换mask的格式
        mask = tf.cast(mask, dtype=loss_.dtype)
        # 调整loss，将<PAD>的loss归0
        loss_ *= mask
        # 确认下是否有空的摘要别加入计算，每一行累加
        loss_ = tf.reduce_sum(loss_, axis=-1) / dec_lens
        return tf.reduce_mean(loss_)



if __name__ == '__main__':
    setproctitle.setproctitle('kelly')
    dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'AutoMaster' + os.sep
    print(dir)
    embeddingModelFile = dir + 'fasttext/fasttext_jieba.model'
    vocab,reverse_vocab, embedding_matrix = util.getEmbedding_matrixFromModel(embeddingModelFile)
    train_x_pad_path = dir + 'AutoMaster_Train_X_jieba.csv'
    train_y_pad_path = dir + 'AutoMaster_Train_Y_jieba.csv'
    test_x_pad_path = dir + 'AutoMaster_Test_X_jieba.csv'
    train_X = preprocessing.load_dataset(train_x_pad_path,vocab)
    train_Y = preprocessing.load_dataset(train_y_pad_path,vocab)
    #test_X = preprocessing.load_dataset(test_x_pad_path,vocab)

    modelFile = 'data/checkpoints/training_checkpoints'
    seq2seq = Seq2seq( train_X = train_X,train_Y = train_Y,vocab = vocab,reverse_vocab =reverse_vocab, embedding_matrix= embedding_matrix,modelFile=modelFile)
    seq2seq.train()


