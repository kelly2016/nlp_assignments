# -*- coding:utf-8 -*-
# Created by LuoJie at 11/29/19

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('BASE_DIR=',BASE_DIR)
sys.path.append(BASE_DIR)

import setproctitle

import tensorflow as tf

from pgn_pack.pgn import PGN
from pgn_pack.train_helper import train_model
from pgn_pack.utils import util
from pgn_pack.utils.gpu_utils import config_gpu
from pgn_pack.utils.params_utils import get_default_params
from pgn_pack.utils.wv_loader import Vocab

def train(params, vocab, reverse_vocab, embedding_matrix,checkpoint_dir):
    # GPU资源配置
    config_gpu(True)
    # 读取vocab训练
    print("Building the model ...")
    vocab = Vocab(vocab, reverse_vocab)

    # 构建模型
    print("Building the model ...")
    # model = Seq2Seq(params)

    model = PGN(params,embedding_matrix)

    print("Creating the batcher ...")
    # dataset = batcher(params["train_seg_x_dir"], params["train_seg_y_dir"], vocab, params)
    # print('dataset is ', dataset)

    # 获取保存管理者
    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    # 训练模型
    print("Starting the training ...")
    train_model(model, vocab, params, checkpoint_manager)



if __name__ == '__main__':
    setproctitle.setproctitle('kelly_gpu')

    dir = '/root/private/kelly/project01/data/AutoMaster/'
         # '/Users/henry/Documents/application/nlp_assignments/data/AutoMaster/' \
          #
    #'' #os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'AutoMaster' + os.sep
    print(dir)
    modelFile = dir + 'fasttext/fasttext_jieba.model'
    checkpoint_dir = dir + 'checkpoints'
    print('checkpoint_dir=',checkpoint_dir)

    '''
    def decode_line(line):
        # Decode the csv line to tensor

        return '', line
    #test
    dataset = tf.data.TextLineDataset('/Users/henry/Documents/application/nlp_assignments/data/AutoMaster/AutoMaster_Train_X.csv')
    dataset = dataset.repeat(10)
    dataset = dataset.map(decode_line)
    '''
    vocab, reverse_vocab, embedding_matrix = util.getEmbedding_matrixFromModel(modelFile)
    # 获得参数
    params = get_default_params(vocab, embedding_matrix)

    # 训练模型
    train(params,vocab, reverse_vocab, embedding_matrix,checkpoint_dir)
