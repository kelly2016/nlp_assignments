# -*- coding: utf-8 -*-
# @Time    : 2019-07-03 12:36
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : aiRobot.py
# @Description:基于规则的机器人对话系统，就是写一个正则解析器
import random
defined_patterns = {
    "I need ?X": ["Image you will get ?X soon", "Why do you need ?X ?"],
    "My ?X told me something": ["Talk about more about your ?X", "How do you think about your ?X ?"]
}


def get_response(saying, rules = defined_patterns):
    """
    依据规则rules，给saying一个反馈
    :param saying:
    :param rules:
    :return:
    """
    saying = saying.split()
    for key,value in rules.items():
        pattern = key.split()
        got_patterns = pat_match(pattern, saying)
        if(len(got_patterns)):#匹配
            retString =  random.choice(value)
            return ' '.join(subsitite(retString.split(), pat_to_dict(got_patterns)))


def is_pattern_segment(pattern):
    return pattern.startswith('?*') and all(a.isalpha() for a in pattern[2:])
from collections import defaultdict


def is_variable(pat):
    """
    是否是模版字符
    :param pat:
    :return:
    """
    return pat.startswith('?') and all(s.isalpha() for s in pat[1:])

def pat_match(pattern, saying):
    """
    saying是否与模版一只
    :param pattern:
    :param saying:
    :return:
    """
    if not pattern or not saying: return []

    if is_variable(pattern[0]):
        return [(pattern[0], saying[0])] + pat_match(pattern[1:], saying[1:])
    else:
        if pattern[0] != saying[0]:
            return []
        else:
            return pat_match(pattern[1:], saying[1:])




def pat_to_dict(patterns):
    """
    把解析出来的结果变成字典
    :param patterns:
    :return:
    """
    return {k: v for k, v in patterns}


def subsitite(rule, parsed_rules):
    """
    依据pat_to_dict出的 dictionary ，依照定义的方式进行替换。
    :param rule:
    :param parsed_rules:
    :return:
    """
    if not rule: return []
    tmp = parsed_rules.get(rule[0],rule[0])#dict.get(key, default=None) default -- 如果指定键的值不存在时，返回该默认值值。
    return [tmp] + subsitite(rule[1:], parsed_rules)

if __name__ == '__main__':
    print(get_response('I need iPhone'))
    print(get_response('My mother told me something'))

    pattern = 'I want ?X'.split()
    saying = "I want iPhone".split()
    got_patterns = pat_match(pattern, saying)
    print(' '.join(subsitite("What if you mean if you got a ?X".split(), pat_to_dict(got_patterns))))



