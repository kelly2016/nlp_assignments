# -*- coding: utf-8 -*-
# @Time    : 2020-02-10 16:59
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : textcnn.py
# @Description:TextCNN multi label multi class
import logging
import multiprocessing
import os
import setproctitle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import utils.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, concatenate, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from utils.plot_utils import plot_confusion_matrix

cores = multiprocessing.cpu_count()
partitions = cores
learning_rate = 0.001
vocab_size=30000
padding_size=390
batch_size = 8
epoch = 50
embed_size=300
dropout_rate = 0.5
filter_sizes = 2

def  preprocess(file,label,isMulti=False):
    """

    :param file: 语料文件地址
    :param labelindex: label 索引
    :param label:label field名字
    :param isMulti:是否是multi label
    :return:
    """

    df = pd.read_csv(file)
    text_preprocesser = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
    text_preprocesser.fit_on_texts(df['item'])
    x = text_preprocesser.texts_to_sequences(df['item'])
    x = pad_sequences(x, maxlen=padding_size, padding='post', truncating='post')
    word_dict = text_preprocesser.word_index
    lb = None
    y = None
    if isMulti is False:
        lb = LabelBinarizer()
        lb.fit(df[label])
        y = lb.transform(df[label])

    else:
         lb = MultiLabelBinarizer()
         y = lb.fit_transform(df[label])


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return lb,y.shape[1],X_train, X_test, y_train, y_test

class CustomModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, model, path, monitor_index):
        self.model = model
        self.path = path
        self.monitor_index = monitor_index
        self.best_acc = 0.70

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs[self.monitor_index]
        if val_acc > self.best_acc:
            print("\nValidation accuracy increased from {} to {}, saving model".format(self.best_acc, val_acc))
            self.model.save(self.path)#.save_weights(self.path, overwrite=True)#
            self.best_acc = val_acc

class TextCNN(object):
        """
        TextCNN:
        1.embedding layers,
        2.convolution layer,
        3.max-pooling,
        4.softmax layer.
        """
        def __init__(self,classes_dim,embedding_dim=embed_size,max_token_num=vocab_size,embedding_matrix = None,model_img_path = False,best_model_file = None,isMulti = False,filter_sizes=filter_sizes,kernel_sizes = [2, 3, 4],max_sequence_length=padding_size):
            self.isMulti = isMulti
            if best_model_file is not None:
                model_path = os.path.expanduser(best_model_file)
                assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
                self.model = load_model(model_path, compile=False)
            else:

                    x_input = Input(shape=(max_sequence_length,))
                    logging.info("x_input.shape: %s" % str(x_input.shape))  # (?, 60)

                    if embedding_matrix is None:
                        x_emb = Embedding(input_dim=max_token_num, output_dim=embedding_dim, input_length=max_sequence_length)(
                            x_input)
                    else:
                        x_emb = Embedding(input_dim=max_token_num, output_dim=embedding_dim, input_length=max_sequence_length,
                                          weights=[embedding_matrix], trainable=True)(x_input)

                    logging.info("x_emb.shape: %s" % str(x_emb.shape))  # (?, 60, 300)

                    pool_output = []
                    for kernel_size in kernel_sizes:
                        c = Conv1D(filters=filter_sizes, kernel_size=kernel_size, strides=1)(x_emb)
                        p = MaxPool1D(pool_size=int(c.shape[1]))(c)
                        pool_output.append(p)
                        logging.info(
                            "kernel_size: %s \t c.shape: %s \t p.shape: %s" % (kernel_size, str(c.shape), str(p.shape)))

                    pool_output = concatenate([p for p in pool_output])
                    logging.info("pool_output.shape: %s" % str(pool_output.shape))  # (?, 1, 6)

                    x_flatten = Flatten()(pool_output)  # (?, 6)
                    output = Dropout(dropout_rate)(x_flatten)#kelly add
                    activation = 'softmax' if self.isMulti == False else 'sigmoid'
                    print('activation = ',activation)
                    y = Dense(classes_dim, activation=activation)(output)  # (?, 2)
                    logging.info("y.shape: %s \n" % str(y.shape))
                    self.model = Model([x_input], outputs=[y])

                    #if model_img_path:
                        #plot_model(self.model, to_file=model_img_path, show_shapes=True, show_layer_names=False)
            print(self.model.summary())


        def train(self,X_train,y_train,X_test,y_test,best_model_file):
            loss = 'categorical_crossentropy' if self.isMulti == False else 'binary_crossentropy'
            print('loss = ', loss)
            self.model.compile(tf.optimizers.Adam(learning_rate=learning_rate),
                              loss=loss,
                              metrics=[  metrics.macro_f1,metrics.micro_f1])
            print('Train...')
            early_stopping = EarlyStopping(monitor='val_micro_f1', patience=10, mode='max')
            reduce_lr = ReduceLROnPlateau(monitor='val_micro_f1', factor=0.5, patience=5 , verbose=1,min_lr=0.001 )
            best_model = CustomModelCheckpoint(self.model, best_model_file, monitor_index='val_micro_f1')
            print(type(X_train))
            print(type(y_train))
            print(type(X_test))
            print(type(y_test))
            history = self.model.fit(X_train, y_train,
                                    batch_size=batch_size,
                                    epochs= epoch,
                                    workers=cores,
                                    use_multiprocessing=True,
                                    callbacks=[best_model,early_stopping,reduce_lr],#[early_stopping],#
                                    validation_data=(X_test, y_test))

            history_dict = history.history
            history_dict.keys()
            micro_f1 = history_dict['micro_f1']
            val_micro_f1 = history_dict['val_micro_f1']
            loss = history_dict['loss']
            val_loss = history_dict['val_loss']
            epochs = range(1, len(micro_f1) + 1)

            # “bo”代表 "蓝点"
            plt.plot(epochs, loss, 'bo', label='Training loss')
            # b代表“蓝色实线”
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            plt.clf()  # 清除数字
            plt.plot(epochs, micro_f1, 'bo', label='Training micro_f1')
            plt.plot(epochs, val_micro_f1, 'b', label='Validation val_micro_f1')
            plt.title('Training and validation micro_f1')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()


        def predict(self, lb, X_test, y_test=None, isMulti=False):
            y_pred = self.model.predict(X_test)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test, axis=1)
            print(classification_report(y_true, y_pred))
            if isMulti == False:
                # 计算混淆矩阵
                conf_mat = confusion_matrix(y_true, y_pred)
                # 画混淆矩阵
                plot_confusion_matrix(conf_mat, classes=[0, 1])
                f1_score(y_test, y_pred, average='samples')
                lb.inverse_transform(y_pred)
                lb.inverse_transform(y_test)


def predict(model_path,lb,X_test,y_test=None,isMulti = False):
    model_path = os.path.expanduser(model_path)
    assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
    model = load_model(model_path, compile=False)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred))
    if isMulti == False:
        # 计算混淆矩阵
        conf_mat = confusion_matrix(y_true, y_pred)
        # 画混淆矩阵
        plot_confusion_matrix(conf_mat, classes=[0, 1])
        f1_score(y_test, y_pred, average='samples')
        lb.inverse_transform(y_pred)
        lb.inverse_transform(y_test)




if __name__=='__main__':
    setproctitle.setproctitle('kelly')
    dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'KnowledgeLabel' + os.sep+'corpus2' + os.sep
    file  =    dir +'corpus.csv'
    best_model_file = dir+'textcnn1.h5'
    model_img_path =  'model.png'
    label = 'label'




    lb, num_classes,X_train, X_test, y_train, y_test = preprocess(file, label)

    textcnn = TextCNN(best_model_file=best_model_file,classes_dim=num_classes,model_img_path =model_img_path,kernel_sizes = [2, 3, 4])
    textcnn.train(X_train,y_train,X_test,y_test,best_model_file)
    textcnn.predict(model_path = best_model_file,lb=lb,X_test=X_test,y_test=y_test)
    #predict(model_path = best_model_file,lb=lb,X_test=X_test,y_test=y_test)

    best_model_file = dir + 'textcnn2.h5'
    label = 'multiLabels'
    lb,num_classes, X_train, X_test, y_train, y_test = preprocess(file, label, isMulti=True)
    textcnn = TextCNN(best_model_file=best_model_file,classes_dim=num_classes,  kernel_sizes=[2, 3, 4],isMulti=True)
    textcnn.train(X_train, y_train, X_test, y_test, best_model_file)
    #predict(model_path = best_model_file,lb=lb,X_test=X_test,y_test=y_test,isMulti = True)



