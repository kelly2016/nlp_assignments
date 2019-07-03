# -*- coding: utf-8 -*-
# @Time    : 2019-07-02 14:44
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : biGram.py
# @Description:句子生成作业，选中了白毛女剧本，人物分别有3个：喜儿，杨白劳，黄世仁


'''
杨白劳：喜儿，我回来啦
喜儿：爹！
杨白劳：喜儿，看我给你买了什么
喜儿：爹，您看红头绳
喜儿：人家的闺女有花戴
喜儿：我爹钱少
喜儿：不能买
喜儿：扯上二尺红头绳
喜儿：给我扎起来
杨白劳：好，爹爹这就给喜儿戴上
'''

import random
import tensorflow as tf
import csv
import re
from collections import Counter

from assignment_01 import preprocessing

news = """
news => 主语 状语 谓语 宾语
主语 => 人名 | 人名 与 主语
人名 => 赵本山 | 秋瓷炫 | 向佐 | 柳岩 | 范冰冰 | 李晨
状语 => 跪地 | 悲痛地 |状语
谓语 => 被拍 | 被爆 |透露 | 完婚 |吸毒 |求婚 | 谓语 谓语 
宾语 => null | 浪漫 |假分手
"""
xier = """
xier => 称谓 ！ 
xier2 =>称谓 ， 主语 谓语 宾语 | 主语 谓语 宾语 | 状语 谓语 |谓语 宾语 | 状语 谓语 补语
称谓 => 爹 | 闺女
代词 => 您 | 我 
谓语 => 看 | 买 | 扯上 | 戴 | 有 | 钱少
宾语 => 定语 宾语 | null | 红头绳 | 花 补语 
主语 => 定语 称谓 | 代词 |  称谓 
定语 => null | 人家的 | 我 |二尺
补语 => 戴 | 起来
状语 => 不能 | 给我
"""

yangbailao = """
yangbailao => 称谓 ， 代词 谓语 补语 
yangbailao2 =>称谓 ， 叹词 代词 谓语 宾语  
yangbailao3 =>叹词 ， 代词 状语 谓语 
称谓 => 喜儿
代词 => 我 ，爹爹
谓语 => 回 | 买了 | 给你 | 给喜儿 | 谓语 
补语 => 来了 | 戴上
叹词 => 看 | 好
宾语 => 什么
状语 => 这就
 
"""


choice = random.choice


def generate_best(m,grammar_str,target,filename):
    """
    生成多个句子，
    :return:返回生成的句子
    """
    bigram = BiGram(filename)
    grammar = create_grammar(grammar_str)
    sens = []
    maxPro = 0.0
    maxSen = ''
    for sen in [generate(gram=grammar, target=target) for i in range(m)]:
        prob = bigram.get_probablity(sen)
        print('sentence:{} with Prb: {}'.format(sen, prob))
        if maxPro <= prob:
            maxPro = prob
            maxSen = sen
    print('max sentence :{} with Prb: {}'.format(maxSen, maxPro))
    return maxSen,maxPro

def create_grammar(grammar_str, split='=>', line_split='\n'):
    """
    生成语法树
    :param grammar_str: 原字符串
    :param split: 左右分隔符
    :param line_split: 换行符
    :return:语法树
    """
    grammar = {}
    for line in grammar_str.split(line_split):
        if not line.strip(): continue
        # print(line)
        exp, stmt = line.split(split)

        grammar[exp.strip()] = [s.split() for s in stmt.split('|')]
    return grammar





def generate(gram, target):
    """
    通过给定语法树gram，以target为起点生成一句话
    :param gram:
    :param target:
    :return:
    """
    if target not in gram: return target
    expaned = [generate(gram, t) for t in random.choice(gram[target])]
    # print(expaned)
    return ''.join([e if e != '/n' else '\n' for e in expaned if e != 'null'])





def generate_n(gramstargets):
    """
    生成多个句子，这里生成杨白劳和喜儿的对话
    :return:返回生成的句子
    """
    sens = []
    for gram, target in gramstargets:
        sen = generate(gram=gram, target=target)
        print(sen)
        sens.append(sen)
    return sens






class BiGram(object):

    def __init__(self, filename):
        """
        2-gram 构建器
        :filename :  语料地址
        """
        #所有分词结果
        TOKEN = preprocessing.tokenizeFormCsv(filename, ['comment','name'] )
        #词频表
        self.words_count = Counter(TOKEN)
        self.totalCount = len(TOKEN)

        TOKEN_2_GRAM = [
            ''.join(TOKEN[i:i + 2]) for i in range(len(TOKEN[:-2]))
        ]
        self.words_count_2 = Counter(TOKEN_2_GRAM)
        self.totalCount_2 = len(TOKEN_2_GRAM)

    def prob_1(self,word):
        return self.words_count[word] / self.totalCount

    def prob_2(self,word1, word2):
        if word1 + word2 in self.words_count_2:
            return self.words_count_2[word1 + word2] / self.totalCount_2
        else:
            return 1 / self.totalCount_2

    def get_probablity(self,sentence):
        words =  preprocessing.cut(sentence)
        sentence_pro = 1
        for i ,word in enumerate(words[:-1]):
            next_ = words[i+1]
            probability = self.prob_2(word,next_)
            sentence_pro *= probability
        return sentence_pro



if __name__ == '__main__':
    filename = '/Users/henry/Documents/application/nlp_assignments/data/movie_comments.csv'
    generate_best(10, news, 'news', filename)
    '''
    sentence:赵本山跪地透露浪漫 with Prb: 4.123325387743087e-58
    sentence:柳岩悲痛地完婚 with Prb: 2.601321461591346e-49
    sentence:秋瓷炫与赵本山与赵本山与赵本山与李晨悲痛地求婚完婚被拍被爆完婚 with Prb: 1.1195568680032818e-203
    sentence:柳岩与范冰冰跪地被爆假分手 with Prb: 5.179695926072617e-82
    sentence:范冰冰与赵本山跪地透露 with Prb: 5.401768821831015e-70
    sentence:赵本山跪地被爆浪漫 with Prb: 7.097527306770885e-58
    sentence:李晨与柳岩悲痛地被拍 with Prb: 4.301237288335587e-65
    sentence:秋瓷炫与赵本山跪地求婚浪漫 with Prb: 1.1277460536732827e-84
    sentence:范冰冰与秋瓷炫与李晨跪地求婚完婚透露假分手 with Prb: 7.193585500444851e-134
    sentence:李晨跪地完婚浪漫 with Prb: 2.0134890956135e-53
    max sentence :柳岩悲痛地完婚 with Prb: 2.601321461591346e-49
    '''
    #bigram = BiGram(filename)

    '''
    xier_gram = create_grammar(xier)
    yangbailao_gram = create_grammar(yangbailao)
    dialogue = [[yangbailao_gram,'yangbailao'],[xier_gram,'xier'],
     [yangbailao_gram,'yangbailao2'],[xier_gram,'xier2'],
     [xier_gram, 'xier2'],[xier_gram,'xier2'],
     [xier_gram,'xier2'],[xier_gram,'xier2'],
     [xier_gram, 'xier2'],[yangbailao_gram,'yangbailao3']]
    generate_n(dialogue)
    '''


