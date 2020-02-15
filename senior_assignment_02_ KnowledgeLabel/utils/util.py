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
