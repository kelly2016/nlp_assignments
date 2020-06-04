# -*- coding: utf-8 -*-
# @Time    : 2020-06-04 11:08
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : test.py
# @Description:

def foo():
    print("starting...")
    i = 0
    while True:
        res = yield i
        print("res:",res)
        i = i+1
g = foo()
print(next(g))
print("*"*20)
print(next(g))
