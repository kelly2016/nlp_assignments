# -*- coding: utf-8 -*-
# @Time    : 2020-01-16 16:37
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : test.py
# @Description:
import json
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def save_graph(graph,file_name):
    #initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=100)
    plt.axis('off')
    fig = plt.figure(1)

    pos = nx.spring_layout(graph,scale = 10)
    nx.draw_networkx_nodes(graph,pos)
    nx.draw_networkx_edges(graph,pos)
    nx.draw_networkx_labels(graph,pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    #plt.xlim(0, xmax)
    #plt.ylim(0, ymax)

    plt.savefig(file_name,bbox_inches="tight")
    plt.close()
    del fig




def drawMap(dg):
    nx.draw(dg, pos=nx.random_layout(dg),node_color = 'b', edge_color = 'r', with_labels = True,font_size = 18, node_size = 20)
    plt.show()

def count():
    f = open('test2.json')
    text = f.read()
    res = re.findall(r'\"name\":\"*.+\",', text)  # \"+[0-9]+_[^0-9]+\",
    # \"+[0-9]+_[0-9]+_[^0-9]+\"
    # \"+[0-9]+_[0-9]+_[0-9]+_[0-9]+_[^0-9]+\",
    # \"+[0-9]+_[0-9]+_[0-9]+_[^0-9]+\"
    # \"[^0-9]+\
    print(len(res))
    s = sorted(set(res))

    print(s)
    print(len(s))

def creatTree(data):
    #connection_graph = defaultdict(list)

    dg = nx.DiGraph()
    catch = {}
    sunCatch = {}
    for d in data['dialogNodes']:
        catch[d['id']] = (d['name'])
        catch[d['name']] = (d['id'])
        dg.add_node(d['name'])

    for d in data['dialogNodes']:

        olist = d['output']
        for o  in olist:
            if 'jump_to' in o.keys():
                print(o['jump_to']['dialog_node_id'])
                parentname = catch[o['jump_to']['dialog_node_id']]
                dg.add_edge(parentname,d['name'])
    return dg

def search(data,node,catch):
    if data is None or len(data) < 1:
        return False
    if node['parent']  in data.keys():
        value = data[node['parent']]
        output = []
        condition = ''
        condition_type = ''
        normal_condition = {}
        if 'condition_type' in node.keys():
            condition_type = node['condition_type']
        if 'condition' in node.keys():
            condition = node['condition']
        if 'normal_condition' in node.keys():
            normal_condition = node['normal_condition']
        if 'output' in node.keys():
            output = node['output']
            for op in output:
                if 'jump_to' in op.keys():
                    id = op['jump_to']['dialog_node_id']
                    op['jump_to']['dialog_node_id'] = catch[id]
        value['son'][node['name']] = {'son': {}, 'output': output,"condition":condition,"condition_type":condition_type,"normal_condition":normal_condition}
        return True
    else:
        for key,value in data.items():
            if search(value['son'], node, catch) is True:
                return True



def check(connection_graph,index = 0,totalcout = {}):

    # num : set(str)
    for key, value in connection_graph.items():
        if  index not in totalcout.keys():
            totalcout[index] = set()
        totalcout[index].add(key)
        check(value['son'], index=index+1, totalcout=totalcout)
    return totalcout

def cmp(ele):
        return ele['name']

def creatTree2(data):

    connection_graph = defaultdict(list)

    catch = {}
    dialogNodes = data['dialogNodes']
    dialogNodes.sort(key=cmp)
    for i,d in enumerate(dialogNodes):
        catch[d['id']] = (d['name'])
        catch[d['name']] = (d['id'])
        #print(d['name'])

    while len(dialogNodes) > 0:
          removelist = []
          for i,dialogNode in enumerate(dialogNodes):
              if dialogNode['parent'] is None:#第一层节点
                  output = []
                  condition=''
                  condition_type = ''
                  normal_condition = {}
                  if 'output' in dialogNode.keys():
                      output = dialogNode['output']
                      for op in output:
                          if 'jump_to' in op.keys():
                              id = op['jump_to']['dialog_node_id']
                              op['jump_to']['dialog_node_id'] = catch[id]
                  if 'condition_type' in dialogNode.keys():
                      condition_type = dialogNode['condition_type']
                  if 'condition' in dialogNode.keys():
                      condition = dialogNode['condition']
                  if 'normal_condition' in dialogNode.keys():
                      normal_condition = dialogNode['normal_condition']
                  connection_graph[dialogNode['name']] = {'son': {}, 'output': output,"condition":condition,"condition_type":condition_type,"normal_condition":normal_condition}
                  removelist.append(i)
              else:
                  if search(connection_graph, dialogNode, catch) is True:
                      removelist.append(i)
                  bug = 0
          totalcout = check(connection_graph)
          j = 0
          for y in removelist:
              p = dialogNodes.pop(y - j)
              inset = False
              for value in totalcout.values():
                  if p['name'] in value:
                      inset = True
                      break
              if inset is False:
                  print('{} is not in set  '.format(p['name']))
              j += 1
              count = 0

              for value in totalcout.values():
                  count += len(value)
          print("len(removelist) = {} ,totalcout= {}  ".format(len(removelist),count))
          if len(removelist) == 0:
              dialogNodes.sort(key=cmp,reverse=True)



    return connection_graph






def dealJson(file):

    with open(file, 'r') as f:
        data = json.load(f)
        return data

if __name__ == '__main__':

    data =  dealJson('test2.json')
    p = creatTree2(data)
    totalcout = check(p)
    index = 0
    for value in totalcout.values():
        print('the {} number is {}  ,content = {}  '.format(index,len(value),value))
    fp = open('dialogtree.json', 'w')
    json_obj1 = json.dump(p, fp, ensure_ascii=False)

    # Assuming that the graph g has nodes and edges entered
    #save_graph(creatTree(data), "my_graph.pdf")
    #drawMap()

