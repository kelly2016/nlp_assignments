# -*- coding: utf-8 -*-
# @Time    : 2019-07-11 09:45
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : SubWay.py
# @Description:北京地铁图，以及相关查询功能
import  os
import requests
import re
from collections import defaultdict
from enum import Enum

import networkx as nx
import matplotlib.pyplot as plt
#源文件分割符
SPLIT = ' '
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def initSubwayData(url,filename=None):
    """
    将北京地铁线路写到文本文件中
    :param url:爬取的URL地址
    :param filename:文件存储地址，默认为上一级目录齐平的data目录下
    :return:
    """
    if filename == None:
        filename = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep+'data'+os.sep+'subway.txt'
    response = requests.get(url,verify=False)
    what_we_want =r'<div\s+class="content">.+?<div\s+class="other"\s+style="display:none;"' #r'<div\s+class="content">(<div/s+class="line_name">.+[<div/s+class="line_name">|<div\s+class="other"\s+style="display:none;">])' #<div\s+class="content">(.+)<div\s+class="other"\s+style="display:none;"
    data = response.text.encode("latin1").decode("gbk")
    pattern = re.compile(what_we_want, re.DOTALL)
    data = re.sub("<a\s+href=\".+?\">|</a>", " ", data)
    likes = pattern.findall(data)
    if likes :
        f = open(filename, 'w')
        what_we_want_2 = r'(<div\s+class="subway_num.+?>(.+?)</div>)|(station">(.+?)</div>)'
        for content in likes[0].split('class="line_name"'):
            #print(content)
            pattern2 = re.compile(what_we_want_2, re.DOTALL)
            l2 = pattern2.findall(content)
            index = 0
            tmpList = []
            for value in l2:
                if index == 0:
                    tmpList.append(value[1])#print(value[1])
                else:
                    tmpList.append(SPLIT)
                    tmpList.append(value[3])
                    #print(index,' : ',value[3])
                index += 1
            f.writelines(tmpList)
            f.writelines(['\n'])
            #print('------------------------------')

        f.close()

def pretty_print(cities):
        return  '->>>>>>'.join(cities)


def drawMap(connetion_info):
    nx.draw(  nx.Graph(connetion_info),with_labels = True,node_size=10)
    plt.show()

class BeijingSubway(object):
    #SPP :Shortest Path Priority（路程最短优先）, MTP,Minimum Transfer Priority(最少换乘优先)
    SORT = Enum('Sort', ('SSP', 'MTP'))

    def __init__(self, filename=None):
        """
        从北京地铁图的数据源文件读取站点信息，构建
        :param filename: 构建北京地铁图的数据源文件地址
        """
        #站点信息
        self.connection_graph =  defaultdict(list)
        #线路名
        self.subways = defaultdict(list)

        if filename == None:
            filename = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'subway.txt'
        with open(filename, 'r') as f1:
            line = f1.readline()
            while line:
                index = 0
                key = None
                for value in line.split(SPLIT):
                    value = value.replace('\n','')
                    if len(value.strip()) == 0:continue
                    if index == 0:
                        key = value
                        self.subways[key] = []

                    else:
                        self.subways[key].append(value)
                    index += 1

                line = f1.readline()
            f1.close()

        for key in self.subways:
            listTmp = self.subways[key]
            for i in range(1,len(listTmp)):
                preCity = listTmp[i - 1]
                value = listTmp[i]
                self.connection_graph[value].append(preCity)
                self.connection_graph[preCity].append(value)
        drawMap(self.connection_graph)
        '''
        for key in self.connection_graph:
            print(str(key) + ':' + str(self.connection_graph[key]))
            print('------------------------------')
         '''



    def search(self,start,desitionation,by_way=None,sort=None):
        """
        搜索从起始站start，到目的地站点destination的路径
        :param start:开始站点
        :param destination:终点
        :param by_way:途径站点
        :param sort:排序方式
        :return:
        """

        pathes = [[start]]

        visited = set()
        while pathes:

            path = pathes.pop(0)  # 转战最少
            # print('path=',path)
            froninter = path[-1]
            # print('froninter=',froninter)
            if froninter == '角门西' or froninter=='公益西桥' or froninter == '义和庄':
                g = 0
            if froninter in visited: continue

            # print('froninter=',froninter)
            if froninter in self.connection_graph:
                succesors = self.connection_graph[froninter]
                for city in succesors:
                    if city in path: continue
                    new_path = path + [city]
                    # print('new_path=',new_path)
                    pathes.append(new_path)
                    # print(pathes)
                    if city == desitionation:
                        return pretty_print(new_path)
            visited.add(froninter)
            if sort :
                if sort == BeijingSubway.SORT.SSP:#路程最短优先
                    pathes = self.transfer_station_first(pathes)
                elif sort == BeijingSubway.SORT.MTP:#最少换乘优先
                    pathes = self.shortest_path_first(pathes)


        return pretty_print(pathes)

    def transfer_station_first(self,paths):#最小换乘优先
        return sorted(paths,key=len)

    def shortest_path_first(self,pathes):  # 最短距离优先
        if len(pathes) < 1: return pathes
        pass





if __name__ == '__main__':


    #initSubwayData(url='https://www.bjsubway.com/station/xltcx/')
    subway = BeijingSubway()
    print(subway.search('苹果园','天宫院'))