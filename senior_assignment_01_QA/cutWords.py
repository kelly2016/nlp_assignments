# -*- coding: utf-8 -*-
# @Time    : 2019-11-12 11:01
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : cutWord.py
# @Description:封装各类分词器，每次复制粘贴代码挺乱的

import jieba
import logging
import os
import re
from enum import Enum
from functools import lru_cache

import pyltpAnalyzer

#标点符号punctuation
P = r'\||\[|\]|\r|\n|\”|\《|\。|\{|\！|？|｡|\＂|＃|＄|％|\＆|\＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|＠|\［|\＼|\］|\＾|＿|｀|\～|｟|｠|\、|〃|》|「|」|『|』|【|】|〔|〕|〖|〗|〘|〙|〚|〛|〜|\〝|\〞|〟|〰|〾|〿|–—|\‘|\“|\„|\‟|\…|\‧|﹏|\.'
#分词起



@lru_cache(maxsize=2 ** 10)
def get_stopwords(stopwordsFile = '/Users/henry/Documents/application/nlp_assignments/data/stopword_ltp.txt.txt'):
    print("start load stopwords")
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    # 加载停用词表
    stopword_set = set()
    if(os.path.exists(stopwordsFile)):
        stopword_list = [k.strip() for k in open(stopwordsFile, encoding='utf8').readlines() if k.strip() != '']
        stopword_set = set(stopword_list)
    print("end load stopwords")
    return stopword_set

# 获取停用词表
stopwords = get_stopwords()


class Analyzer(object):
    ANALYZERS = Enum('Analyzer', ('Jieba', 'LTP'))
    """
    根据指定类型来创建分词器：目前是jieba和哈工大的分词器
    """

    def __init__(self,type,replaceP=True,useStopwords = False,userdict = None):
        self.type = type
        if type ==Analyzer.ANALYZERS.Jieba:
            self.analyzer = jieba
            if userdict is not None:
                self.analyzer.load_userdict(userdict)
        elif type ==Analyzer.ANALYZERS.LTP:
            self.analyzer =pyltpAnalyzer.PyltpAnalyzer()
            if userdict is not None:
                self.analyzer.loadSegmentorUserdict(userdict)


        #是否过停用词
        self.useStopwords = useStopwords
        self.replaceP=replaceP#会否替换标点符号

    def cut(self,string):
        """
        用分词器切词并用空格隔开
        :param string:
        :return: 返回格式是字符串
        """

        article_contents = ''
        sens = ''
        if self.replaceP == True:
            sens = split(string)
        else:
            sens = [string]#strB2Q(string)
        for sen in sens:
            if self.type == Analyzer.ANALYZERS.Jieba:
                # 使用jieba进行分词
                words = self.analyzer.cut(sen, cut_all=False)
            elif self.type == Analyzer.ANALYZERS.LTP:
                #使用ltp进行分词
                words = self.analyzer.segmentSentence(sen)
            if self.useStopwords == True:
                    for word in words:
                        if word not in stopwords:
                            article_contents += word + " "
            else:
                article_contents= ' '.join(words)

        return article_contents

    def cut2list(self,string):
        """
        返回list
        :param string:
        :return:
        """

        tokens = []
        if self.replaceP == True:
            sens = split(string,'' if self.type == Analyzer.ANALYZERS.Jieba else ' ')
        else:
            sens =  [string]#[strB2Q(string)]
        for sen in sens:
            if self.type == Analyzer.ANALYZERS.Jieba:
                # 使用jieba进行分词
                words = self.analyzer.cut(sen, cut_all=False)
            elif self.type == Analyzer.ANALYZERS.LTP:
                #使用ltp进行分词
                words = self.analyzer.segmentSentence(sen)
            if self.useStopwords == True:
                for word in words:
                    if word not in stopwords:
                        tokens.append(word)
            else:
                tokens += words
        return tokens

def split(ustring,ch = ' '):
    """
    用标点符号分句
    :param ustring:
    :return:
    """
    return [re.sub(P, ch, ustring)] #re.split(p,ustring)#

def strQ2B(ustring):
    """把字符串全角转半角"""
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)

def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 32:                                 #半角空格直接转化
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:        #半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += chr(inside_code)
    return rstring



if __name__ == '__main__':
    analyzer = Analyzer(Analyzer.ANALYZERS.Jieba, replaceP=False, useStopwords=False,
                        userdict='/Users/henry/Documents/application/nlp_assignments/data/AutoMaster/userDict.txt')

    print(analyzer.cut('微型货车，金杯西部牛仔，4f18发动机。德尔福电脑。 问题'))
    #print(analyzer.cut2list('启辰D50'))