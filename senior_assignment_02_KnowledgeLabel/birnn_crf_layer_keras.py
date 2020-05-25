# encoding=utf-8

"""
bert-blstm-crf layer
@Author:Macan
"""

from enum import Enum

import tensorflow as tf
from keras.layers import Dense, TimeDistributed
from keras.layers import Dropout
from keras_contrib.layers import CRF
from tensorflow.contrib.layers.python.layers import initializers


class CELL_TYPE(Enum):
    GRU  = 'gru'
    LSTM = 'lstm'

class BIRNN_CRF(object):

    def __init__(self, embedded_chars, cell_type,
                  num_labels, seq_length, labels, lengths, is_training, dropout_rate=0.5, initializers=initializers,num_layers=1,hidden_unit=512):
        """
        BLSTM-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: LSTM的隐含单元个数
        :param cell_type: RNN类型（LSTM OR GRU ）
        :param num_layers: RNN的层数
        :param droupout_rate: droupout rate
        :param initializers: variable init class
        :param num_labels: 标签数量
        :param seq_length: 序列最大长度
        :param labels: 真实标签
        :param lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """
        self.hidden_unit = hidden_unit
        self.dropout_rate = dropout_rate
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths
        self.embedding_dims = embedded_chars.shape[-1].value
        self.is_training = is_training

    def add_birnn_crf_layer(self, crf_only=False):
        """
        blstm-crf网络
        :return:
        """
        if self.is_training:
            # lstm input dropout rate i set 0.9 will get best score
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)

        if crf_only:
            logits = self.project_crf_layer(self.embedded_chars)
        else:
            # birnn
            output = self.birnn_layer(self.embedded_chars)
            # project
            logits = self.project_birnn_layer(output)
        # crf
        loss, out = self.crf_layer(logits)
        return (loss, out)

    def _witch_cell(self):
        """
        RNN 类型
        :return:
        """
        cell_tmp = None
        if self.cell_type == CELL_TYPE.LSTM:
            cell_tmp =  tf.keras.layers.LSTM(self.hidden_unit,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        elif self.cell_type == CELL_TYPE.GRU:
            cell_tmp =  tf.keras.layers.GRU(self.hidden_unit,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        return cell_tmp



    def birnn_layer(self, embedding_chars):
        """

        :return:
        """
        with tf.variable_scope('rnn_layer'):
            cell= self._witch_cell()
            hiddenlist = None
            outputs = embedding_chars
            for num_layer in self.num_layers:
                birnn = tf.keras.layers.Bidirectional(cell, merge_mode='concat',name = ('bi'+self.cell_type))
                outputs, forward_state, backward_state = birnn(outputs, initial_state=hiddenlist)
                hiddenlist = tf.concat([forward_state, backward_state], axis=1)
                outputs = Dropout(self.dropout_rate)(outputs)
        return outputs

    def project_birnn_layer(self, outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                dense_layer = TimeDistributed(Dense(self.num_labels, activation="softmax"), name="time_distributed")(outputs)

                """
                 W = tf.get_variable("W", shape=[self.hidden_unit * 2, self.hidden_unit],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.hidden_unit], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(outputs, shape=[-1, self.hidden_unit * 2])
                hidden = tf.matmul(output, W) + b #tf.nn.xw_plus_b(output, W, b)
                


            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.hidden_unit, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.matmul(hidden, W)+b
                """
            return dense_layer


    def project_crf_layer(self, embedding_chars, name=None):
        """
        hidden layer between input layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.embedding_dims, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(self.embedded_chars,
                                    shape=[-1, self.embedding_dims])  # [batch_size, embedding_dims]
                pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):

            crf = CRF(self.num_labels)
            out = crf(logits)
            return  crf.loss_function,out


