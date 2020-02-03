# -*- coding: utf-8 -*-
# @Time    : 2020-02-02 14:52
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : train.py
# @Description:
import os
import setproctitle

if __name__=='__main__':
    setproctitle.setproctitle('kelly')
    dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'KnowledgeLabel' + os.sep+'corpus' + os.sep
    preprocessing(dir)







