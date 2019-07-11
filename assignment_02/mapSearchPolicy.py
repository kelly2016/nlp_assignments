# -*- coding: utf-8 -*-
# @Time    : 2019-07-09 22:39
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : mapSearchPolicy.py
# @Description:


from collections import defaultdict

simple_connetion_info_src = {
    '北京': ['太原'],
    '太原': ['北京', '西安', '郑州'],
    '郑州': ['太原'],
    '兰州': ['西安'],
    '西安': ['兰州', '长沙'],
    '长沙': ['南宁', '福州'],
    '沈阳': ['北京']

}
simple_connetion_info = defaultdict(list)
simple_connetion_info.update(simple_connetion_info_src)#函数把字典dict2的键/值对更新到dict里


def bfs(graph, start):
    # breath first search
    visited = [start]
    seen = set()
    print('visited=', visited)
    while visited:
        froninter = visited.pop()
        if froninter in seen: continue
        print('froninter=', froninter)
        for successor in graph[froninter]:
            if successor in seen: continue
            print(successor)
            # visited.append(successor)
            # visited = [successor] +visited
            visited = visited + [successor]
        seen.add(froninter)

    # return visited
    return seen


number_graph = defaultdict(list)
number_graph.update({
    1: [2, 3],
    2: [1, 4],
    3: [1, 5],
    4: [2, 6],
    5: [3, 7]

})


seen = bfs(number_graph,1)
print(seen)


