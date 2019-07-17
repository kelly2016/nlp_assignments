# -*- coding: utf-8 -*-
# @Time    : 2019-07-17 10:45
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : editDistanceTest.py
# @Description:

from functools import lru_cache
import re

solution = {}

ADD = 'ADD'
DEL =  'DEL'
SUB = 'SUB'

@lru_cache(maxsize=2 ** 10)
def edit_distance(string1, string2):
    if len(string1) == 0: return len(string2)
    if len(string2) == 0: return len(string1)

    tail_s1 = string1[-1]
    tail_s2 = string2[-1]

    candidates = [
        (edit_distance(string1[:-1], string2) + 1, 'DEL {}'.format(tail_s1)),  # string 1 delete tail
        (edit_distance(string1, string2[:-1]) + 1, 'ADD {}'.format(tail_s2)),  # string 1 add tail of string2
    ]

    if tail_s1 == tail_s2:
        both_forward = (edit_distance(string1[:-1], string2[:-1]) + 0, '')
    else:
        both_forward = (edit_distance(string1[:-1], string2[:-1]) + 1, 'SUB {} => {}'.format(tail_s1, tail_s2))

    candidates.append(both_forward)
    #if i ≥ 1  且 j ≥ 1 ，edit(i, j) == min{ edit(i-1, j) + 1, edit(i, j-1) + 1, edit(i-1, j-1) + f(i, j) }，当第一个字符串的第i个字符不等于第二个字符串的第j个字符时，f(i, j) = 1；否则，f(i, j) = 0。
    min_distance, operation = min(candidates, key=lambda x: x[0])

    solution[(string1, string2)] = operation

    return min_distance



def printSteps(string1,string2,steps ):#
    """
    打印string1变化到string2的步骤
    :param string1:
    :param string2:
    :return:
    """

    if len(string1) == 0:
        return steps
    if len(string2) == 0:
        return steps
    operationOperation = solution[(string1, string2)]

    if operationOperation != '' :
        steps.append(operationOperation)

        if ADD in operationOperation :#如果是添加矩阵横坐标不变，纵坐标-1
            printSteps(string1, string2[: -1],steps)
        elif DEL in operationOperation :#如果是减少矩阵纵坐标不变，横坐标-1
            printSteps(string1[:-1], string2,steps)
        elif SUB in operationOperation:  # 如果是替换，
            printSteps(string1[:-1], string2[: -1],steps)
    else:
        if edit_distance(string1[:-1], string2[: -1]) > 0:#如果是替换操作
            printSteps(string1[:-1], string2[: -1],steps)


    return steps


def excSteps(string1,operator,operatorStr1,operatorStr2=None):
    """
    演示执行
    :param string1:
    :param operator:
    :param operatorStr1:
    :param operatorStr2:
    :return:
    """
    if string1 == '' or string1 == None:
        return
    if operatorStr1 == '' or operatorStr1 == None:
        return

    if operator == ADD  :  #
        string1 += operatorStr1
    elif operator ==  DEL :  #
        string1 = string1[:-1]
    elif operator ==  SUB :  #
        string1 =  string1.replace(operatorStr1, operatorStr2, 1)

    return string1



stepss = []
str1 ='ABCDE'
str2 = 'ABCCEF'
print('{} 和 {}的编辑距离是：{}'.format(str1,str2,edit_distance(str1, str2)))#算出编辑距离
print('匹配过程是：')
stepss = printSteps(str1, str2,stepss)
print(str1,end=' ')
len = len(stepss)

what_we_want = r'(SUB|ADD|DEL)\s+([a-zA-Z0-9]+)?\s?(=>)?\s?([a-zA-Z0-9]+)?' #'(SUB)\s+([a-zA-Z0-9]+)?\s*=>\s([a-zA-Z0-9]+)?'
for i  in range(0,len):
    pattern = re.compile(what_we_want, re.DOTALL)
    index = len-1-i
    print(stepss[index],end=' ')
    step = pattern.findall(stepss[index])
    tmp = excSteps(str1,step[0][0],step[0][1],operatorStr2=step[0][3])
    print(tmp,end=' ')
    str1 = tmp


