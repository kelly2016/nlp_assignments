# -*- coding:utf-8 -*-
# Created by LuoJie at 11/22/19
import numpy as np
# 引入日志配置




# SENTENCE_START = '<s>'
# SENTENCE_END = '</s>'

# PAD_TOKEN = '<PAD>'
# UNKNOWN_TOKEN = '<UNK>'
# START_DECODING = '<START>'
# STOP_DECODING = '<STOP>'


class Vocab:
    PAD_TOKEN = '<PAD>'
    UNKNOWN_TOKEN = '<UNK>'
    START_DECODING = '<START>'
    STOP_DECODING = '<STOP>'

    def __init__(self, word2id, id2word, vocab_max_size=None):
        """
        Vocab 对象,vocab基本操作封装
        :param vocab_file: Vocab 存储路径
        :param vocab_max_size: 最大字典数量
        """
        self.word2id = word2id
        self.id2word = id2word
            #self.load_vocab(vocab_file, vocab_max_size)
        self.count = len(self.word2id)

    @staticmethod
    def load_vocab(self,file_path, vocab_max_size=None):
        """
        读取字典
        :param file_path: 文件路径
        :return: 返回读取后的字典
        """
        """
        vocab = {}
        reverse_vocab = {}
        for line in open(file_path, "r", encoding='utf-8').readlines():
            word, index = line.strip().split("\t")
            index = int(index)
            # 如果vocab 超过了指定大小
            # 跳出循环 截断
            if vocab_max_size and index > vocab_max_size:
                print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                    vocab_max_size, index))
                break
            vocab[word] = index
            reverse_vocab[index] = word
        """
        return self.word2id, self.id2word

    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id[self.UNKNOWN_TOKEN]
        return self.word2id[word]

    def id_to_word(self, word_id):
        if word_id not in self.id2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id2word[word_id]

    def size(self):
        return self.count


def load_embedding_matrix(filepath=None):
    """
    加载 embedding_matrix_path
    """
    return np.load(filepath + '.npy')


if __name__ == '__main__':
   pass

