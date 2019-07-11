# -*- coding: utf-8 -*-
# @Time    : 2019-07-11 09:45
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : SubWay.py
# @Description:北京地铁图，以及相关查询功能
import  os
import requests
import re

SPLIT = ' '

def initSubwayData(url,filename=None):
    """

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







    #likes = pattern.findall(response.text)


class BeijingSubway(object):
    def __init__(self, filename):
        """

        :param filename: 构建北京地铁图的数据源文件地址
        """
        #站点信息
        self.connection_graph = None
        #排序方式
        self.sort_condition =None
        pass

    def search(self,start,destination,by_way=None,sort=None):
        """
        搜索从起始站start，到目的地站点destination的路径
        :param start:开始站点
        :param destination:终点
        :param by_way:途径站点
        :param sort:排序方式
        :return:
        """
        pass

if __name__ == '__main__':

    initSubwayData(url='https://www.bjsubway.com/station/xltcx/')
