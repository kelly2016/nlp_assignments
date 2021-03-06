# -*- coding: utf-8 -*-
# @Time    : 2019-10-07 12:05
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : ToxicCommentClassifier.py
# @Description:

import numpy as np
#import  preprocessing
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from gensim.models import FastText
import warnings
import  pickle

warnings.filterwarnings('ignore')

import os

#设置系统环境变量程序执行的线程
os.environ['OMP_NUM_THREADS'] = '2'
dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep
MODELFILE = dir+'/fasttext_ltp.model'
print('MODELFILE = ',MODELFILE)
modelf = FastText.load(MODELFILE)
class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size

#EMBEDDING_FILE = '/Users/henry/Documents/application/nlp_assignments／data/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
def getRawDataSet(pickle_file='/Users/henry/Documents/application/nlp_assignments/data/word2vect/s2v_w2v.pickle'):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(MacOSFile(f))
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        labelsSet = save['labelsSet']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
        print('labelsSet', labelsSet)
    return (train_dataset, train_labels), (valid_dataset, valid_labels), (test_dataset, test_labels),labelsSet

(train_dataset, train_labels), (valid_dataset, valid_labels), (
        test_dataset, test_labels), labelsSet = getRawDataSet(dir+'s2v_w2v_raw_ltp.pickle')

#train = pd.read_csv('/Users/henry/Documents/application/nlp_assignments/data/jigsawtoxiccommentclassificationchallenge/train.csv')
    #(dir+'movie_comments.csv')
#test = pd.read_csv('/Users/henry/Documents/application/nlp_assignments/data/jigsawtoxiccommentclassificationchallenge/test.csv')
submission = pd.read_csv(dir+'sample_submission.csv')

X_train = train_dataset #train["comment_text"].fillna("fillna").values
y_train = train_labels #train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values#t
X_valid = valid_dataset
y_valid = valid_labels
l = len(test_dataset)
X_test = test_dataset[:l-2] #test["comment_text"].fillna("fillna").values
y_test = test_labels[:l-2]
max_features = 30000
maxlen = 500#1071
embed_size = 100#300


tokenizer = text.Tokenizer(num_words=max_features)
#通过列表text来构建Tokenizer类生成词典
tokenizer.fit_on_texts(list(X_train) + list(X_test))
#句子转换成单词索引序列
X_train = tokenizer.texts_to_sequences(X_train)
X_valid = tokenizer.texts_to_sequences(X_valid)
X_test = tokenizer.texts_to_sequences(X_test)
#序列填充 将多个序列截断或补齐为相同长度 maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0.

x_train = sequence.pad_sequences(X_train, maxlen=maxlen)#
x_valid = sequence.pad_sequences(X_valid, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)#

#array和asarray都可将结构数据转换为ndarray类型。但是主要区别就是当数据源是ndarray时，array仍会copy出一个副本，占用新的内存，但asarray不会。
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


#embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

#字典，将单词（字符串）映射为它们的排名或者索引。仅在调用fit_on_texts之后设置
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = modelf.wv[word]#preprocessing.s2v_w2v.getW2V(word)#embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))


def get_model():
    inp = Input(shape=(maxlen,))
    #该层只能用作模型中的第一层。
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    #正整数，输出空间的维度
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(5, activation="sigmoid")(conc)#softmax

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model




model = get_model()

batch_size = 32
epochs = 4

#X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(x_valid, y_valid), interval=1)

hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid),
                 callbacks=[RocAuc], verbose=2)

y_pred = model.predict(x_test, batch_size=batch_size)
score = roc_auc_score(y_test, y_pred)
print("\n ROC-AUC - test - score: %.6f \n" % (score))
submission[["1", "2", "3", "4", "5"]] = y_pred
#保存预测结果
submission.to_csv('submission.csv', index=False)

if __name__=='__main__':
    pass