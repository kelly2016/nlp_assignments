# -*- coding: utf-8 -*-
# @Time    : 2020-02-01 20:23
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : FastText.py
# @Description:自己写的fastText
import os
import setproctitle

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Input, Model
from tensorflow.keras import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

batch_size = 128
embedding_dims = 300
epochs = 20
max_features = 10000

def draw(history):
    history_dict = history.history
    history_dict.keys()

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # “bo”代表 "蓝点"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b代表“蓝色实线”
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def evaluate(model,X_test,y_test):
    print('Test...')
    y_pred = model.predict(X_test)
    y_pred = y_pred.argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    print('sequences 20 epochs max_features{} classification_report: \n'.format(max_features))
    print(classification_report(y_true, y_pred))
    print('sequences 20 epochs confusion_matrix: \n')
    confusion_matrix(y_true, y_pred)

def train(maxlen, max_features,class_num,X_train, y_train,X_test, y_test):
    print('Build model...')
    model = FastText(maxlen, max_features, embedding_dims, class_num).get_model()
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    print('Train...')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        workers=32,
                        use_multiprocessing=True,
                        callbacks=[early_stopping],
                        validation_data=(X_test, y_test))
    return model,history



class FastText(object):
    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=1,
                 last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        input = Input((self.maxlen,))

        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)(input)
        x = GlobalAveragePooling1D()(embedding)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model



if __name__=='__main__':
    setproctitle.setproctitle('kelly')
    dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'KnowledgeLabel' + os.sep+'corpus' + os.sep
    preprocessing(dir)