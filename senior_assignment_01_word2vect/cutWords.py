# -*- coding: utf-8 -*-
# @Time    : 2019-11-12 11:01
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : cutWord.py
# @Description:封装各类分词器，每次复制粘贴代码挺乱的

import  logging,jieba, os, re
from functools import lru_cache
from enum import Enum
import pyltpAnalyzer


#标点符号
P = r'\||\[|\]|\r|\n|\”|\《|\。|\{|\！|？|｡|\＂|＃|＄|％|\＆|\＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|＠|\［|\＼|\］|\＾|＿|｀|\～|｟|｠|\、|〃|》|「|」|『|』|【|】|〔|〕|〖|〗|〘|〙|〚|〛|〜|\〝|\〞|〟|〰|〾|〿|–—|\‘|\“|\„|\‟|\…|\‧|﹏|\.'
#分词起




class Analyzer(object):
    ANALYZERS = Enum('Analyzer', ('Jieba', 'LTP'))
    """
    根据指定类型来创建分词器：目前是jieba和哈工大的分词器
    """

    def __init__(self,type,useStopwords = False):
        self.type = type
        if type ==Analyzer.ANALYZERS.Jieba:
            self.analyzer = jieba
        elif type ==Analyzer.ANALYZERS.LTP:
            self.analyzer =pyltpAnalyzer.PyltpAnalyzer()
        #是否过停用词
        self.useStopwords = useStopwords

    def cut(self,string):
        """
        用分词器切词并用空格隔开
        :param string:
        :return: 返回格式是字符串
        """
        # 获取停用词表
        stopwords = get_stopwords()
        article_contents = ''
        sens = _split(string)
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
        # 获取停用词表
        stopwords = get_stopwords()
        tokens = []
        sens = _split(string)
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

def _split(ustring):
    """
    用标点符号分句
    :param ustring:
    :return:
    """
    return [re.sub(P, ' ', ustring)] #re.split(p,ustring)#

@lru_cache(maxsize=2 ** 10)
def get_stopwords(stopwordsFile = '/Users/henry/Documents/application/nlp_assignments/data/stopwords.txt'):
    print("start load stopwords")
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    # 加载停用词表
    stopword_set = set()
    if(os.path.exists(stopwordsFile)):
        stopword_list = [k.strip() for k in open(stopwordsFile, encoding='utf8').readlines() if k.strip() != '']
        stopword_set = set(stopword_list)
    print("end load stopwords")
    return stopword_set





if __name__ == '__main__':
    analyzer = Analyzer(Analyzer.ANALYZERS.Jieba,True)
    print(analyzer.cut('当我们需要定义常量时\r，一个方法是用大写变量通过整数来定义，例如月份'))
    print(analyzer.cut2list('当我们需要定义常量时，一个方法是用大写变量通过整数来定义，例如月份'))