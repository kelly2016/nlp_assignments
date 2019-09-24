# -*- coding: utf-8 -*-
# @Time    : 2019-07-17 16:53
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : k_person_salesman.py
# @Description:

import random
import matplotlib.pylab as plt
import math
import copy
from functools import lru_cache
from functools import cmp_to_key

solution = []


latitudes = [random.randint(-100, 100) for _ in range(5)]
longitude = [random.randint(-100, 100) for _ in range(5)]


chosen_p = (-50, 10)
chosen_p2 = (1, 30)
chosen_p3 = (99, 15)

plt.scatter(latitudes, longitude)
plt.scatter([chosen_p[0]], [chosen_p[1]], color='r')
plt.scatter([chosen_p2[0]], [chosen_p2[1]], color='r')
plt.scatter([chosen_p3[0]], [chosen_p3[1]], color='r')
plt.show()


chosen_pss = [chosen_p,chosen_p2,chosen_p3]#]#
address = chosen_pss


#初始化数据
for i,j in zip(latitudes,longitude):
    address += [(i,j)]

def distance(node1,node2):
    """
    两点之间的距离
    :param node1: （x1,y1）
    :param mode2: （x2,y2）
    :return:
    """
    if node2 == None or node1 == None:
        return 0
    return math.sqrt(((node1[0]-node2[0])**2)+((node1[1]-node2[1])**2))

def isOver(curNodes):
    """
    当前是否已经经历了所有节点
    :param curNodes:
    :return:
    """
    return (curNodes == set(address))


'''
@lru_cache(maxsize=2 ** 10)
def min_path2(curentNode, preNode = None,chosen_ps=None):
    """

    :param curentNodeIndex: 当前节点
    :param path: 经过的路径
    :param nodes: 经过的节点，节点重复复也只记录一次
    :param preNodeIndex: 上一级节点
    :return:
    """
    #min_distance = float('inf')
    #min_node = None
    if isOver(set(solution)):
        return 0

    candidates = []
    solution.append(curentNode)
    for i in address:
         if (i == curentNode or i ==preNode ):
             continue
         curd = distance(i, curentNode)
         solution.append(i)
         candidates += [curd +min_path2(i,preNode =curentNode)]



    return min(candidates)

'''
#@lru_cache(maxsize=2 ** 10)
def min_path( curentNode,path,nodes,preNode = None,chosen_ps=None):
    """

    :param curentNodeIndex: 当前节点
    :param path: 经过的路径
    :param nodes: 经过的节点，节点重复复也只记录一次
    :param preNodeIndex: 上一级节点
    :return:
    """
    nodes.add(curentNode)
    path += [curentNode]
    if isOver(nodes):
        return  distance(curentNode,preNode),path,nodes

    candidates = [ ]
    for i  in address:
        '''
        if(i == curentNode or i ==preNode  ):#(i  in nodes): #

            if chosen_ps==None:
                continue
            else:#如果存在着几个点，且只能经过一次,
                if (i in chosen_ps) and (i in nodes):
                    continue
        '''
        if (i == curentNode or i  in nodes):
            continue
        p = copy.deepcopy(path)
        n = copy.deepcopy(nodes)
        min_distance2, path2, nodes2 = min_path(i, p, n, preNode =curentNode)
        curd = distance(curentNode,preNode)
        candidates += [(curd+min_distance2, path2, nodes2 )]
    min_distance3, path3,nodes3 = min(candidates, key=lambda x: x[0])

    return min_distance3, path3,nodes3




#随机生成1个起始点
i = random.randint(0,len(address)-1)
path = []
nodes = set()

#print( min_path2( address[i]))

min_distance3, path3,nodes3 = min_path( address[i],path,nodes)

print(' 最短长度为（任何点都只经过一次:( 计算量就已经很大了很大了很大了很大了。。。。 ）{}  '.format(min_distance3))
print(' 最短路径为（任何点都只经过一次:( 计算量就已经很大了很大了很大了很大了。。。。 ）{}  '.format(path3))
'''
sumv = 0
for i in range(1,len(path3)):
    sumv += distance(path3[i-1], path3[i])

print(' sumv : min_distance3  = {} : {}'.format(sumv,min_distance3))
'''



if __name__ == '__main__':
    pass