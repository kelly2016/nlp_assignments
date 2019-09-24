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
        i = 0
        for row in reader:
            #i += 1
            #if(i > 10000): break;
            #print(i,' : ',row[columns[0]])
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

p = r'\”|\《|\。|\{|\！|？|｡|\＂|＃|＄|％|\＆|\＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|＠|\［|\＼|\］|\＾|＿|｀|\～|｟|｠|\、|〃|》|「|」|『|』|【|】|〔|〕|〖|〗|〘|〙|〚|〛|〜|\〝|\〞|〟|〰|〾|〿|–—|\‘|\“|\„|\‟|\…|\‧|﹏|\.'
import  re
def strQ2B(ustring):
    return re.sub(p, '', ustring)


if __name__ == '__main__':
    tokenizeFormCsv('/Users/henry/Documents/application/nlp_assignments/data/movie_comments.csv', ['comment'] )