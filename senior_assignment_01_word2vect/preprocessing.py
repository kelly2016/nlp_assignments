# -*- coding: utf-8 -*-
# @Time    : 2019-11-11 17:34
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : preprocessing.py
# @Description:1.作业1：
# 1.熟悉项目数据
# 2.分词以及清晰数据
# 3.通过训练数据和测试数据建立vocab词汇表
# 数据集下载地址：https://aistudio.baidu.com/aistudio/competition/detail/3
# ------------------------
# 作业1提交截止日期：2019年11月15日（周五）晚8点
# ------------------------
# 提交作业格式：建议使用zip压缩包上传个人作业
# 文件命名格式：“作业1+姓名+班级名+提交日期”，例如：【作业1+伊帆+图灵班+11.15】

import pandas as pd
import tensorflow as tf
from collections import Counter
from  cutWords import Analyzer

analyzer = Analyzer(Analyzer.ANALYZERS.Jieba,False)
counter = Counter()
def deal(src_file,output_file):
    """

    :param src_file: 待处理的源数据
    :param output_file: 处理后的数据
    :param dict: 处理完的字典，格式为词语 词频 ，eg， word 12
    :return:
    """

    num_written_lines = 0
    with tf.gfile.GFile(output_file, "w") as writer:
        src_df = pd.read_csv(src_file, encoding='utf-8',header=None,sep =None)
        orLabels = ['QID','Brand','Model','Question','Dialogue','Report']

        title_line =  ",".join(label for label in orLabels) + "\n"
        writer.write(title_line)
        for j, values in enumerate(src_df.values):
            if j == 0:
                continue
            if  values[len(values)-1] == None:#如果Report无数值
                continue
            output_line =''

            for (i, v) in enumerate(values):
                if j == 52:
                    bug = 0
                if i  == 5 :
                    bug = 0
                line = ''
                #print('j = {} ,i = {} , v= {}'.format(j,i,v))
                if v is not None and type(v) == str and len(v) > 0:
                   words = analyzer.cut2list(v)
                   counter.update(words)
                   line = ' '.join(words)
                else:
                   line = ' '
                if i < len(values) and i > 0:
                   output_line =  output_line + ","+line
                elif i == 0:
                    output_line = line
            if len(values) == 5:
                output_line + output_line+ ","+' '


            writer.write(output_line+ '\n' )
            num_written_lines += 1
    print('total num_written_lines =',num_written_lines)

def wirteDict(dictFile):
    f = open(dictFile, 'w')  # 若是'wb'就表示写二进制文件
    for k, v in counter.items():
        f.write('{} {} \n'.format(k,v))
        print('{} {} \n'.format(k,v))
    f.close()



if __name__ == '__main__':

    src_file = '/Users/henry/Documents/application/nlp_assignments/data/AutoMaster/AutoMaster_TrainSet.csv'
    output_file = '/Users/henry/Documents/application/nlp_assignments/data/AutoMaster/AutoMaster_TrainSet_cleared.csv'
    deal(src_file, output_file)

    src_file = '/Users/henry/Documents/application/nlp_assignments/data/AutoMaster/AutoMaster_TestSet.csv'
    output_file = '/Users/henry/Documents/application/nlp_assignments/data/AutoMaster/AutoMaster_TestSet_cleared.csv'
    deal(src_file, output_file)

    dictFile = '/Users/henry/Documents/application/nlp_assignments/data/AutoMaster/AutoMaster_Counter.txt'
    wirteDict(dictFile)

