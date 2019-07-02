# -*- coding: utf-8 -*-
# @Time    : 2019-07-02 18:30
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : preprocessing.py
# @Description:预处理相关内容

import tensorflow as tf
import csv
import nltk.stem
import jieba

PUNCTUATION_PATTERN = ['。', '!', '?', '｡', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '/', ':',
                       ';', '<', '=', '>', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '｟', '｠', '｢',
                       '｣', '､', '、', '〃', '》', '「', '」', '『', '』', '【', '】', '〔', '〕', '〖', '〗', '〘', '〙', '〚',
                       '〛', '〜', '〝', '〞', '〟', '〰', '〾', '〿', '–—', '‘', '’', '‛', '“', '”', '„', '‟', '…', '‧',
                       '﹏', '.', '\',','《']
BEGIN = '<begin>'
END = '<end>'

def __cut(string):
    return list(jieba.cut(string))

def read_csv( input_file, columns):
    """Reads a tab separated value file."""
    lines = []
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for column in columns:
                sens = strQ2B(row[column])
                for sen in sens:
                    lines.append(BEGIN +sen +END)

    return lines

def cut(string):
    TOKEN = []
    sens = strQ2B(string)
    TOKEN.append(BEGIN)
    for sen in sens:
        TOKEN += __cut(sen)
    TOKEN.append(END)
    return TOKEN

def tokenizeFormCsv( input_file, columns):
    """Reads a tab separated value file."""

    TOKEN = []
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for column in columns:
                TOKEN += cut(row[column])
                '''
                sens = strQ2B(row[column])
                TOKEN.append(BEGIN)
                for sen in sens:
                    TOKEN += __cut(sen)
                TOKEN.append(END)
                '''

    return TOKEN


def strQ2B(ustring):

    """全角转半角"""
    rsents = []
    rstring = ''
    #当前连续字符
    curIsLetter = False
    letter = ''
    stemmer = nltk.stem.SnowballStemmer('english')
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
            curIsLetter = False
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
            curIsLetter = False
        elif (inside_code >= 65 and inside_code <= 122):#如果是字符
            curIsLetter = True
            if(inside_code >= 65 and inside_code <= 90):
                inside_code += 32#如果是大写字符，专成小写
        else:
            curIsLetter = False
        curChar = chr(inside_code)
        if(curIsLetter):
            letter += curChar
        else:
            if(len(letter)>0):
                #stemming英文
                rstring += stemmer.stem(letter)
                letter = ''
            if (curChar in PUNCTUATION_PATTERN):
                if (len(rstring) > 0):
                    rsents.append(rstring)
                    rstring = ''
            else:
                rstring += curChar

    if (len(letter) > 0):
        # stemming英文
        rstring += stemmer.stem(letter)
        letter = ''
    if (len(rstring) > 0):
        rsents.append(rstring)
        rstring = ''
    return rsents

if __name__ == '__main__':
    tokenizeFormCsv('/Users/henry/Documents/application/nlp_assignments/data/movie_comments.csv', ['comment'] )