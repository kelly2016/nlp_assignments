#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-02-11 14:52
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : util.py
# @Description:
import os
import setproctitle

import numpy as np


def format(matrix):
    """
    matrix矩阵，NumPy数组中，沿轴1将最大值设置为1，其余值设置为零:
    [[0.36659517 0.4424879  0.96776616 0.69099763]
 [0.11664181 0.12562079 0.18756865 0.37555337]
 [0.36661335 0.37456415 0.73158797 0.31750937]
 [0.67948714 0.03370266 0.5097918  0.65935103]]
 转为
[[0. 0. 1. 0.]
 [0. 0. 0. 1.]
 [0. 0. 1. 0.]
 [1. 0. 0. 0.]]
    :param matrix:
    :return
    """
    out = np.zeros(matrix.shape)
    idx = matrix.argmax(axis=1)
    out[np.arange(matrix.shape[0]) ,idx] = 1
    return out


def get_max_len(data):
    """
    获得合适的最大长度值
    :param data: 待统计的数据
    :return: 最大长度值
    """
    max_lens = data.apply(lambda x: x.count(' ')+1)
    return int(np.mean(max_lens) + 2 * np.std(max_lens))



if __name__=='__main__':
    setproctitle.setproctitle('kelly')
    dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'KnowledgeLabel' + os.sep+'corpus2' + os.sep
    file  =    dir +'corpus.csv'
    best_model_file = dir+'textcnn.h5'
    model_img_path =  'model.png'
    label = 'label'


    def get_max_len(data):
        """
        获得合适的最大长度值
        :param data: 待统计的数据
        :return: 最大长度值
        """
        max_lens = data.apply(lambda x: x.count(' ') + 1)
        return int(np.mean(max_lens) + 2 * np.std(max_lens))
    df = pd.read_csv(file)
    print(get_max_len(df['item']))
