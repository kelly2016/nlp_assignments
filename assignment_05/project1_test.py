# -*- coding: utf-8 -*-
# @Time    : 2019-08-07 17:41
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : project1_test.py
# @Description:

from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

text = """我是一条天狗呀！
    我把月来吞了，
    我把日来吞了，
    我把一切的星球来吞了，
    我把全宇宙来吞了。
    我便是我了！"""
sentences = text.split()
sent_words = [list(jieba.cut(sent0)) for sent0 in sentences]
document = [" ".join(sent0) for sent0 in sent_words]

print(document)
#tfidf_model = TfidfVectorizer().fit(document)
vectorized = TfidfVectorizer()
#print(vectorized.vocabulary_)
tfidf_model = vectorized.fit_transform(document)
##返回的是非系数矩阵 (0, 1)	0.7071067811865476，表示： 对应vectorized.vocabulary_ 中 第0个文档的第1个term的tf-idf的值为0.7071067811865476

print(vectorized.vocabulary_)
    # {'一条': 1, '天狗': 4, '日来': 5, '一切': 0, '星球': 6, '全宇宙': 3, '便是': 2}

print(tfidf_model)

print(tfidf_model.shape)
vector_of_d_2 = tfidf_model[0].toarray()[0]
print(vector_of_d_2)