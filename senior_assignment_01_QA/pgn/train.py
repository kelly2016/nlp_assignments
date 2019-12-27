# -*- coding:utf-8 -*-
# Created by LuoJie at 11/29/19

import os
import setproctitle

import tensorflow as tf

from pgn.pgn import PGN
from pgn.train_helper import train_model
from pgn.utils import util
from pgn.utils.gpu_utils import config_gpu
from pgn.utils.params_utils import get_default_params
from pgn.utils.wv_loader import Vocab


def train(params, vocab, reverse_vocab, embedding_matrix,checkpoint_dir):
    # GPU资源配置
    config_gpu()
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
    setproctitle.setproctitle('kelly')
    dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'AutoMaster' + os.sep
    print(dir)
    modelFile = dir + 'fasttext/fasttext_jieba.model'
    checkpoint_dir = dir + 'checkpoints'
    vocab, reverse_vocab, embedding_matrix = util.getEmbedding_matrixFromModel(modelFile)

    # 获得参数
    params = get_default_params( vocab, embedding_matrix)
    # 训练模型
    train(params,vocab, reverse_vocab, embedding_matrix,checkpoint_dir)
