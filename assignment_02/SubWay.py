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
from functools import cmp_to_key
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
    #SPP :Shortest Path Priority（路程最短优先约等于最少站点）, MTP,Minimum Transfer Priority(最少换乘优先)
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
        #站点其他相关属性：如属于哪些线路
        self.station = defaultdict(set)
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
                        self.station[value].add(key)
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



    def search(self, start, desitionation, by_way=None, sort=None):
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
            if froninter == '角门西' or froninter == '公益西桥' or froninter == '义和庄':
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
            if sort:
                if sort == BeijingSubway.SORT.SSP:  # 最少站点优先
                    pathes = self.transfer_station_first(pathes)
                elif sort == BeijingSubway.SORT.MTP:  # 最少换乘优先
                    pathes = self.minimum_transfer_first(pathes)
                else:
                    pathes = self.comprehensive_first(pathes)
            else:
                pathes = self.comprehensive_first(pathes)
        return pretty_print(pathes)



    def transfer_station_first(self,paths):#最少站点优先
        return sorted(paths,key=len)

    def get_path_linenum(self,path):
            prePreLine = {}  # 上上一站的所属线路
            preLine = {}#上一站的所属线路
            currentLine = {}  # 当前站的所属线路
            index = 0
            lineNum = 0 #线路数
            for station in path:
                if index == 0:
                    currentLine = self.station[station]
                    preLine =  currentLine
                    prePreLine = preLine
                else:
                    currentLine = self.station[station]
                    if len(currentLine&preLine&prePreLine) == 0:#当前站站点和前面站，前前面站点的无交集，代表来到了一个新站
                        lineNum += 1
                    prePreLine = preLine
                    preLine = currentLine
                index += 1
            #print(path,' : ',lineNum)
            return lineNum

    def minimum_transfer_first(self,pathes):  # 最少换乘
        if len(pathes) < 1: return pathes



        return sorted(pathes, key=self.get_path_linenum)



    def comprehensive_first(self,pathes):#Comprehensive Priority(综合优先) 先按照站点数小的，再按换乘少的
            def mycmp(path_a, path_b):
                if len(path_a) > len(path_b):
                    return 1
                elif len(path_a) > len(path_b):
                    return -1
                else:#

                    linenum_a = self.get_path_linenum(path_a)
                    linenum_b = self.get_path_linenum(path_b)
                    if linenum_a > linenum_b:
                        return 1
                    elif linenum_a < linenum_b:
                        return -1
                    else:
                        return 0

            pathes.sort(key=cmp_to_key(mycmp))
            return pathes






if __name__ == '__main__':

    #爬取站点信息   14号线 官网几个站点顺序不对，人工调整
    #14号线 张郭庄 园博园 大瓦窑 郭庄子 大井 七里庄 西局 善各庄 陶然桥 永定门外 景泰 蒲黄榆 方庄   十里河 北工大西门 平乐园 北京南站 九龙山 大望路 朝阳公园 枣营 东风北桥 高家园 阜通 望京 金台路 将台 望京南 东湖渠 来广营
    #initSubwayData(url='https://www.bjsubway.com/station/xltcx/')
    subway = BeijingSubway()
    #最少站点
    print(subway.search('磁器口','大望路',sort=BeijingSubway.SORT.SSP))
    #最少换乘
    print(subway.search('磁器口', '大望路',sort= BeijingSubway.SORT.MTP))
    #综合排序：最少站点-》最少换乘
    print(subway.search('磁器口', '大望路'))