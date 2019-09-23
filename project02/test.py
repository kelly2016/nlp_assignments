# -*- coding: utf-8 -*-
# @Time    : 2019-09-23 12:03
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : test.py
# @Description:
import  numpy

def cosine_dis(vector1, vector2):
    return numpy.dot(vector1,vector2)/(numpy.linalg.norm(vector1)*(numpy.linalg.norm(vector2)))

if __name__=='__main__':
        emb1 = numpy.array([1,1,1])#embedding[0,:]
        emb2 = numpy.array([2,2,2])#embedding[1,:]
        inn = (emb1 * emb2).sum()
        emb1norm = numpy.sqrt((emb1 * emb1).sum())
        s1 = numpy.sqrt(3)
        emb2norm = numpy.sqrt((emb2 * emb2).sum())
        s2 = numpy.sqrt(12)
        score = inn / emb1norm / emb2norm
        r1 = inn /(s1 *s2)

        r2 = (inn*s2) / s1
        print('score=',score)