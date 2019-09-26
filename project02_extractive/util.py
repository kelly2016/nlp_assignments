# -*- coding: utf-8 -*-
# @Time    : 2019-09-25 14:25
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : util.py
# @Description:
import os

def saveWeightfile(file,srcFile):
    """
    读取分词文件srcFile，统计每个词的词频，并写入文件file
    :param file:
    :param srcFile:
    :return:
    """
    wordMap = {}
    with open(srcFile) as f:
        line = f.readline()  # str类型
        while line:
            print('line= ', line)
            words = line.strip().split()
            for w in words:
                if w in wordMap:
                    wordMap[w] = (wordMap[w]+1)
                else:
                    wordMap[w] = 1
            line = f.readline()  # str类型

    print('statistic over ',len(wordMap))
    output = open(file, "w+", encoding="utf-8")
    for key, value in wordMap.items():
        output.write(key+' '+str(value) + "\n")
        print('write:',(key+' '+str(value) + "\n"))

    output.close()

if __name__ == '__main__':
    dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep
    print('dir = ', dir)

    saveWeightfile( dir+'zhwiki_vocab_fre_ltp2.txt',dir+'wiki_corpus_ltp')#