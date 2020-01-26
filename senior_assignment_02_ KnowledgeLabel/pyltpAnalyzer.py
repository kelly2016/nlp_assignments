# -*- coding: utf-8 -*-
# @Time    : 2019-08-20 14:06
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : pyltpAnalyzer.py
# @Description:哈工大pyltp分析器



#分词
import os
from pyltp import SentenceSplitter
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
from pyltp import Parser
import numpy as np
import re
# 分词
import os
import re
from pyltp import NamedEntityRecognizer
from pyltp import Parser
from pyltp import Postagger
from pyltp import Segmentor
from pyltp import SentenceSplitter

import numpy as np

print('------------------------###########')
LTP_DATA_DIR = '/Users/henry/Documents/application/newsExtract/news/data/ltp_data/'  # ltp模型目录的路径
THRESHOLD = 0.8
THRESHOLD_NAME = 0.85

stopword_list = [k.strip() for k in open('/Users/henry/Documents/application/nlp_assignments/data/stopwords.txt', encoding='utf8').readlines() if k.strip() != '']
stopword_list = set(stopword_list)


file = open('/Users/henry/Documents/application/newsExtract/news/data/verb.txt')
verbs =set(set(line.strip('\n') for line in file.readlines()))


def parse_sentence(arcs,netags,words,sentence, postags,ws=False):
    """
    老师的解析方法
    :param arcs:
    :param netags:
    :param words:
    :param sentence:
    :param ws:
    :return:
    """
    # sentence = ' '.join([x for x in sentence.split('，') if x])
    print("sen", sentence)

    # 判断是否有‘说’相关词：
    # print(cuts)
    mixed = [word for word in words if word in verbs]
    # print("mixed  ",mixed)
    if not mixed: return False


    wp_relation = [w.relation for w in arcs]
    name = ''
    stack = []
    for k, v in enumerate(arcs):
        # save the most recent Noun
        if postags[k] in ['nh', 'ni', 'ns']:  # person name geographical name organization name
            stack.append(words[k])
        if v.relation == 'SBV' and (words[v.head - 1] in mixed):  # 确定第一个主谓句
            name = get_name(words[k], words[v.head - 1], words, wp_relation, netags)
            saying = get_saying(words, wp_relation, [i.head for i in arcs], v.head)
            if not saying:
                quotations = re.findall(r'“(.+?)”', sentence)
                if quotations: says = quotations[-1]
            return name, saying
        # 若找到‘：’后面必定为言论。
        if words[k] == '：':
            name = stack.pop()
            saying = ''.join(words[k + 1:])
            return name, saying
    return False

def merge(sentencesParsed,sentences):
    """
    合并的意思相近的句子
    :param sentencesParsed: 句法分析后的句子
    :param sentences: 原句
    :return:
    """
    # merge
    deadline = len(sentencesParsed)
    i = 0
    cache = set()
    while (i < deadline):
        if i  in cache  or sentencesParsed[i] is  None or sentences[i]  is None or len(sentences[i]) == 0:
            i += 1
            continue
        for   j in range((i + 1),deadline):

            if j in cache   or sentences[j] is None or  len(sentences[j]) == 0:
                continue
            value = cosine_dis(analyzer.sen_vec_W2V(sentences[i]), analyzer.sen_vec_W2V(sentences[j]))
            if value >= THRESHOLD:
                     if sentencesParsed[j] is not None:
                         # 如果过主语是同一个人
                         th_name = cosine_dis(analyzer.sen_vec_W2V(sentencesParsed[i][0]), analyzer.sen_vec_W2V(sentencesParsed[j][0]))
                         if  th_name > THRESHOLD_NAME:
                         #if sentencesParsed[i][0] == sentencesParsed[j][0] :
                             sentencesParsed[i] = (sentencesParsed[i][0], sentencesParsed[i][1] + '\n' + sentencesParsed[j][1])
                             if j not in cache:
                                 cache.add(j)
                             sentencesParsed[j] = None
                         else:
                             print('sentencesParsed[i].name = {} and sentencesParsed[j].name ={}'.format(sentencesParsed[i][0],sentencesParsed[j][0]))


                     else:

                         sentencesParsed[i] = (sentencesParsed[i][0],sentencesParsed[i][1]+'\n' + sentences[j])
                         if j in cache :
                             cache.add(j)
                         sentencesParsed[j] = None
                     if i in cache  :
                         cache.add(i)

        i += 1
    return sentencesParsed





def extract(arcs,netags,words,verbSet=verbs):
    """
    这是一句话，提取出这句话的SBV 主语 ，谓语 以及主语表达的观点
    :param arcs: 句法依存结果列表
    :param netags:ner结果列表
    :param words:分词结果列表
    :param verbSet:待处理的谓词范围集合
    :return:
    """
    subjectIndex = 0
    verb = None
    object = ''
    retList = []
    lastIndex = len(arcs)
    wp_relation = [w.relation for w in arcs]
    j = 0
    while j < lastIndex:
        arc = arcs[j]
        objecLastIndex = j + 1
        if arc.relation == 'SBV' :# 符合'SBV'关系的处理,and netags[j] != 'O'且主语是明确的名词
            subjectIndex = j
            verbIndex = arcs[subjectIndex].head - 1  # 找到谓词索引号
            if words[verbIndex] in verbSet:#如果谓词是我们处理范围内的
                name = get_name(words[subjectIndex], words[verbIndex], words, wp_relation, netags)
                objecLastIndex = lastIndex
                if arcs[verbIndex].head > 0  and arcs[verbIndex].head-1 >verbIndex:#如果谓词父节点是不是root，即不是整句的中心词 而且 如果父节点在当前谓语的后面，那么最后节点就是父节点
                    objecLastIndex = arcs[verbIndex].head-1
                #获得宾语
                for i  in range(verbIndex+1,objecLastIndex):
                    if arcs[i].head == 0:
                        break;
                    object += words[i]

                retList.append((name,object))#(words[subjectIndex],words[verbIndex],object)
                object = ''
        j = max(j + 1, objecLastIndex)
    return retList[0] if len(retList) > 0 else None

def extractNoneMerge(arcs,netags,words,verbSet=verbs):
    """
    这是一句话，提取出这句话的SBV 主语 ，谓语 以及主语表达的观点
    :param arcs: 句法依存结果列表
    :param netags:ner结果列表
    :param words:分词结果列表
    :param verbSet:待处理的谓词范围集合
    :return:
    """
    subjectIndex = 0
    verb = None
    object = ''
    retList = []
    lastIndex = len(arcs)
    wp_relation = [w.relation for w in arcs]
    j = 0
    while j < lastIndex:
        arc = arcs[j]
        objecLastIndex = j + 1
        if arc.relation == 'SBV' :# 符合'SBV'关系的处理,and netags[j] != 'O'且主语是明确的名词
            subjectIndex = j
            verbIndex = arcs[subjectIndex].head - 1  # 找到谓词索引号
            if words[verbIndex] in verbSet:#如果谓词是我们处理范围内的
                name = get_name(words[subjectIndex], words[verbIndex], words, wp_relation, netags)
                objecLastIndex = lastIndex
                if arcs[verbIndex].head > 0  and arcs[verbIndex].head-1 >verbIndex:#如果谓词父节点是不是root，即不是整句的中心词 而且 如果父节点在当前谓语的后面，那么最后节点就是父节点
                    objecLastIndex = arcs[verbIndex].head-1
                #获得宾语
                for i  in range(verbIndex+1,objecLastIndex):
                    if arcs[i].head == 0:
                        break;
                    object += words[i]

                retList.append((name,object))#(words[subjectIndex],words[verbIndex],object)
                object = ''
        j = max(j + 1, objecLastIndex)
    return retList

def get_name( name, predic, words, property, ne):
    index = words.index(name)
    cut_property = property[index + 1:] #截取到name后第一个词语
    pre=words[:index]#前半部分
    pos=words[index+1:]#后半部分
    #向前拼接主语的定语
    while pre:
        w = pre.pop(-1)
        w_index = words.index(w)

        if property[w_index] == 'ADV': continue
        if property[w_index] in ['WP', 'ATT', 'SVB'] and (w not in ['，','。','、','）','（']):
            name = w + name
        else:
            pre = False

    while pos:
        w = pos.pop(0)
        p = cut_property.pop(0)
        if p in ['WP', 'LAD', 'COO', 'RAD'] and w != predic and (w not in ['，', '。', '、', '）', '（']):
            name = name + w # 向后拼接
        else: #中断拼接直接返回
            return name
    return name


# 获取谓语之后的言论
def get_saying(sentence, proper, heads, pos):
    # word = sentence.pop(0) #谓语
    if '：' in sentence:
        return ''.join(sentence[sentence.index('：')+1:])
    while pos < len(sentence):
        w = sentence[pos]
        p = proper[pos]
        h = heads[pos]
        # 谓语尚未结束
        if p in ['DBL', 'CMP', 'RAD']:
            pos += 1
            continue
        # 定语
        if p == 'ATT' and proper[h-1] != 'SBV':
            pos = h
            continue
        # 宾语
        if p == 'VOB':
            pos += 1
            continue
        # if p in ['ATT', 'VOB', 'DBL', 'CMP']:  # 遇到此性质代表谓语未结束，continue
        #    continue
        else:
            if w == '，':
                return ''.join(sentence[pos+1:])
            else:
                return ''.join(sentence[pos:])



def cosine_dis(vector1, vector2):
    """
    求vector1，vector2的余弦
    :param vector1:
    :param vector2:
    :return:
    """

    if vector1 is None or  vector2 is None or len(vector1) == 0 or len(vector2) ==0 :
        return float('-inf')

    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))





def text2sentences(text):
    """
    将文本text切成句子，以list返回
    :param text:
    :return: [],一句一个 元素
    """
    sentences = []
    if text and len(text.strip()) > 0:
        for sentence in SentenceSplitter.split(text):
            sentences.append(sentence[:-1])
    return sentences


class PyltpAnalyzer(object):
    def __init__(self, fileDir=LTP_DATA_DIR ):
        """

        :param filename:
        """
        print('77777&777777777777777')
        self.fileDir = fileDir
        # 初始化分词实例
        self.cws_model_path = os.path.join(self.fileDir, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
        self.segmentor = Segmentor()
        self.segmentor.load(self.cws_model_path)  # 加载模型
        # 初始化标注实例
        self.pos_model_path = os.path.join(self.fileDir, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
        self.postagger = Postagger()
        self.postagger.load(self.pos_model_path)  # 加载模型

        # 初始化命名实体识别实例
        self.ner_model_path = os.path.join(self.fileDir, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load( self.ner_model_path)  # 加载模型

        #依存句法分析
        self.par_model_path = os.path.join(self.fileDir, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
        self.parser = Parser() # 初始化实例
        self.parser.load(self.par_model_path)  # 加载模型

    def loadSegmentorUserdict(self,user_dict):
        """
        载入用户分词词典
        :param user_dict:
        :return:
        """
        self.segmentor.load_with_lexicon(self.cws_model_path, user_dict)

    def segmentSentence(self, sentence):
        return list(self.segmentor.segment(sentence))

    def segment(self,sentences):
        """

        :param sentences: 句子列表
        :return:句子分词结果
        """
        wordsList = []
        if sentences:
            for sentence in sentences:
                wordsList.append(list(self.segmentor.segment(sentence)))
        return wordsList

    def postag(self,wordsList):
        """

        :param wordsList: 句子分词列表
        :return: 句子分词词性标注结果
        """
        postagsList = []
        if wordsList:
            for words in wordsList:
                postagsList.append(list(self. postagger.postag(words)))
        return postagsList

    def recognize(self,wordsList,postagsList):
        """

        :param wordsList: 句子分词列表
        :param postagsList: 句子标注列表
        :return: 句子命名实体识别结果
        """
        netagsList = []
        if  wordsList and postagsList :
            if len(wordsList) == len(postagsList):
                for words, postags in zip(wordsList,postagsList):
                    netagsList.append(list(self.recognizer.recognize(words, postags)))
            else:
                print("wordsList = {} ,len(wordsList) = {}  and postagsList = {} ,len(postagsList)".format(wordsList,len(wordsList), postagsList,len(postagsList)))
        else:
            print("wordsList = {}  and postagsList = {}".format(wordsList,postagsList))

        return netagsList

    def dependencyParse(self,wordsList,postagsList):
        """

        :param wordsList: 句子分词列表
        :param postagsList: 句子标注列表
        :return: 句子句法分析结果
        """
        arcsList = []
        if  wordsList and postagsList :
            if len(wordsList) == len(postagsList):
                for words, postags in zip(wordsList, postagsList):
                    arcsList .append(list(self. parser.parse(words, postags)))#arc.head 父节点, arc.relation 依存关系
            else:
                print("wordsList = {} ,len(wordsList) = {}  and postagsList = {} ,len(postagsList)".format(wordsList,len(wordsList), postagsList,len(postagsList)))
        else:
            print("wordsList = {}  and postagsList = {}".format(wordsList,postagsList))

        return arcsList




    def finalize(self):
        """
        释放所有没用到的模型
        :return:
        """
        self.segmentor.release()  # 释放分词模型
        self.postagger.release()  # 释放词性模型
        self.recognizer.release()  # 释放命名实体模型
        self.parser.release()  # 释放依存句法模型



if __name__ == '__main__':
    text = "新华社深圳8月19日电(记者 白瑜)华为公司19日晚发布媒体声明，称反对美国商务部将另外46家华为实体列入\“实体名单\”，呼吁美国政府停止对华为的不公正对待，将华为移出\“实体名单\”。8月19日晚间，美国商务部宣布将对华为的临时采购许可证延长90天，并决定将会把46家华为附属公司加入\“实体名单\”。针对这一表态，华为公司发布媒体声明称，\“华为反对美国商务部将另外46家华为实体列入\‘实体名单\’。美国选择在这个时间点做出这个决定，再次证明该决定是政治驱动的结果，与美国国家安全毫无关系，这种做法违反市场经济的自由竞争原则，不会使任何一方从中受益，包括美国公司在内。美国也不会通过打压华为获得技术领先的地位。我们呼吁美国政府停止对华为的不公正对待，将华为移出\‘实体名单\’。克林顿表示，美国政府今天发布的延期临时许可，没有改变华为被不公正对待的事实。不管临时许可延期与否，华为经营受到的实质性影响有限，我们会继续聚焦做好自己的产品，服务于全球客户。克林顿表示，美国政府今天发布的延期临时许可，没有改变华为被不公正对待的事实。"
        #'他叫汤姆去拿外衣。'
        #"中新网8月18日电 综合香港媒体报道，香港教育专业人员协会(简称教协)17日发起所谓教育界黑衣游行，颠倒是非，公然纵暴，企图美化暴力行为，为违法者脱罪。 "
        #"新华社深圳8月19日电(记者 白瑜)华为公司19日晚发布媒体声明，称反对美国商务部将另外46家华为实体列入\“实体名单\”，呼吁美国政府停止对华为的不公正对待，将华为移出\“实体名单\”。8月19日晚间，美国商务部宣布将对华为的临时采购许可证延长90天，并决定将会把46家华为附属公司加入\“实体名单\”。针对这一表态，华为公司发布媒体声明称，\“华为反对美国商务部将另外46家华为实体列入\‘实体名单\’。美国选择在这个时间点做出这个决定，再次证明该决定是政治驱动的结果，与美国国家安全毫无关系，这种做法违反市场经济的自由竞争原则，不会使任何一方从中受益，包括美国公司在内。美国也不会通过打压华为获得技术领先的地位。我们呼吁美国政府停止对华为的不公正对待，将华为移出\‘实体名单\’。华为表示，美国政府今天发布的延期临时许可，没有改变华为被不公正对待的事实。不管临时许可延期与否，华为经营受到的实质性影响有限，我们会继续聚焦做好自己的产品，服务于全球客户。"
    analyzer = PyltpAnalyzer()
    sentences = text2sentences(text)
    wordsList = analyzer.segment(sentences)
    postagsList = analyzer.postag(wordsList)
    netagsList = analyzer.recognize(wordsList, postagsList)
    arcsList = analyzer.dependencyParse(wordsList,postagsList)
    sentencesParsed = []
    for arcs,netags,words,postags in zip(arcsList,netagsList,wordsList,postagsList):
       #for k, v in enumerate(arcs):
           #print(words[k], words[v.head-1])
           #w = words.index(k)
       #print("%d:%s" % (arc.head, arc.relation) for arc in arcs)
       sen = extract(arcs,netags,words)
       sentencesParsed.append(sen)
       print('extract = ',sen)
       #print('parse_sentence = ',parse_sentence(arcs,netags,words,text, postags))
    #merge
    sentencesParsed = merge(sentencesParsed, sentences)
    for s in sentencesParsed :
        if s is not None:
            print('{} --- {}'.format(s[0], s[1]))
    analyzer.finalize()

    # '




