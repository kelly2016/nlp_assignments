# -*- coding: utf-8 -*-
# @Time    : 2020-01-31 12:54
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : lda.py
# @Description:
import multiprocessing
import os
import setproctitle
from functools import partial

import numpy as np
import pandas as pd
from gensim import utils
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from sklearn.model_selection import train_test_split

cores = multiprocessing.cpu_count()
partitions = cores

def preprocess(string):
    """
    和业务相关的字符串处理
    :param str:
    :return:
    """
    #s
    return str(string)+' '+str(string)+' '+str(string)


def data_fram_proc1(df,dictionary,lda):
    df['item'] = df['item'].apply(preprocess1,dictionary =dictionary,lda = lda)
    return df


def preprocess1(string,dictionary,lda):
    """
    和业务相关的字符串处理
    :param str:
    :return:
    """

    tv = getDocVector(dictionary, lda, string)
    maxp = 0
    maxt = tv[0][0]
    for (t,p) in tv:
         if maxp <= p:
             maxp = p
             maxt = t

    terms = lda.get_topic_terms(maxt, topn=20)
    v = ''
    for word_id, probability in terms:
        probability *= 100
        for count in range(int(probability)):
            v = v + ' ' + dictionary.id2token[word_id]
    return v+ ' ' +string


def parallelize(df,func,dictionary = None,lda = None):
    """

    :param df:
    :param func:
    :return:
    """
    #将df横切
    data_split = np.array_split(df,partitions)
    pool  = multiprocessing.Pool(cores)
    data = None
    if dictionary != None and lda != None:
        data = pd.concat(pool.map(partial(func,dictionary=dictionary,lda=lda), data_split))
    else:
        data = pd.concat(pool.map(func,data_split))
    #关闭pool，使其不在接受新的任务。
    pool.close()
    #主进程阻塞，等待子进程的退出， join方法要在close或terminate之后使用。
    pool.join()
    return data

def data_fram_proc(df):
    df['knowledge']= df['knowledge'].apply(preprocess)
    return df



class LDAModel(object):
    """

    """

    def __init__(self,path,model_file,dictionary_file,corpus_file,num_topics=21):
            """
            进行数据预处理，获取训练集和测试集
            class biological分子与细胞_cleaned.csv : 12
            class biological现代生物技术专题_cleaned.csv : 14
            class biological生物技术实践_cleaned.csv : 16
            class biological生物科学与社会_cleaned.csv : 18
            class biological稳态与环境_cleaned.csv : 110
            class biological遗传与进化_cleaned.csv : 112
            class geography人口与城市_cleaned.csv : 42
            class geography区域可持续发展_cleaned.csv : 44
            class geography地球与地图_cleaned.csv : 46
            class geography宇宙中的地球_cleaned.csv : 48
            class geography生产活动与地域联系_cleaned.csv : 410
            class history古代史_cleaned.csv : 52
            class history现代史_cleaned.csv : 54
            AttributeError: 'PyDB' object has no attribute 'has_plugin_line_breaks'
            Exception ignored in: '_pydevd_frame_eval.pydevd_frame_evaluator_darwin_36_64.get_bytecode_while_frame_eval'
            AttributeError: 'PyDB' object has no attribute 'has_plugin_line_breaks'
            class history近代史_cleaned.csv : 56
            class political公民道德与伦理常识_cleaned.csv : 102
            class political时事政治_cleaned.csv : 104
            class political生活中的法律常识_cleaned.csv : 106
            class political科学思维常识_cleaned.csv : 108
            class political科学社会主义常识_cleaned.csv : 1010
            class political经济学常识_cleaned.csv : 1012
            :param file:语料文件
            :param ratio:测试训练的比列
            :return lda:返回lda模型
            """



            dirs = os.listdir(path)
            x_list = []
            item_x = []
            labels = []
            multiLabels = []
            label11 = 0

            for file in dirs:
                #print(os.path.join(path, file))
                path2 = os.path.join(path, file)
                if os.path.isdir(path2):
                    category = file
                    dirs2 = os.listdir(path2)
                    label12 = 0
                    for file2 in dirs2:
                        file3 = os.path.join(path2, file2)
                        if os.path.isfile(file3) and file2.endswith('_cleaned.csv'):
                            print('class {}{} : {}{}'.format(file, file2, label11, label12))
                            src_df = pd.read_csv(file3)
                            src_df = parallelize(src_df, data_fram_proc) #上采样

                            #merged_df = pd.concat([src_df['items'], src_df['knowledge']], axis=1)
                            src_df['item'] = src_df['items'] + src_df['knowledge']
                            x = np.array(src_df['item']).tolist()
                            item_x += x
                            x = [[word for word in doc.split(' ') if word != "" ] for  doc in x]
                            x_list+= x # list
                            #labels += ['__label__'+str(label11)+''+str(label12) for i in range(len(x))]
                            fn = str(file2).replace('_cleaned.csv','').replace('\t','').replace('\n','')
                            labels += ['__label__' + str(file) + '_' + fn  for i in range(len(x))]
                            bug = 0
                            mls = np.array(src_df['label']).tolist()
                            multiLabels += [ str(file).replace('_',' ') +' '+fn+' '+  str(ml).replace('\t','').replace('\n','') for ml in mls ]
                            bug = 1
                        label12 += 1
                label11 += 1

            c = {'label': labels,'item': item_x,'multiLabels':multiLabels}  # 合并成一个新的字典c
            df = pd.DataFrame(c)  # 将c传入DataFrame并创建
            df.to_csv(corpus_file, index=None,  header=True)


            # 把文章转成list,字典里面 "token2id "
            self.dictionary = Dictionary(x_list)
            # 把文本转成词袋形式  id : freq
            self.corpus = [self.dictionary.doc2bow(text) for text in x_list]

            # 调用lda模型，并指定10个主题
            self.lda = LdaModel(self.corpus, id2word=self.dictionary, num_topics=num_topics)
            # 检查结果
            results = self.lda.print_topics(num_topics, num_words=50)
            for result in results:
                print(result)

            # Save model to disk.
            self.lda.save(model_file)

            self.dictionary.save_as_text(dictionary_file)


    def __retrain(self, model_file,other_texts):
           """
           lda = LdaModel.load(model_file)
           other_corpus = [self.dictionary.doc2bow(text) for text in other_texts]
           lda.update(other_corpus)
           """

    def getDocSVector(self):
        self.docSVector = []
        for d in self.corpus:
            self.docSVector.append(self.lda.get_document_topics(d,minimum_probability = 0))
        return self.docSVector



def getDocVector(dictionary,lda,doc):
        """

        :param doc: list ['w1','w2'],原始分词后的文档语料，没有转成bow
        :return:返回该文档的主题分布
        """
        x = [word for word in doc.split(' ') if word != ""]
        docBow =  dictionary.doc2bow(x)
        return lda.get_document_topics(docBow,minimum_probability=0)

def saveLDACorpus(train_data_path,test_data_path,model_file,dictionary_file,corpus_file):
        ""
        lda = LdaModel.load(model_file)
        dictionary = Dictionary.load_from_text(dictionary_file)
        dictionary.id2token = utils.revdict(dictionary.token2id)
        src_df = pd.read_csv(corpus_file)
        src_df = parallelize(src_df, data_fram_proc1,dictionary ,lda) #计入ida特征
        train_data, test_data = train_test_split(src_df[['label','multiLabels','item']], test_size=0.2, random_state=42)
        train_data.to_csv(train_data_path,  index=None )#, header=None
        test_data.to_csv(test_data_path, index=None )#, header=None

if __name__=='__main__':
    setproctitle.setproctitle('kelly')
    dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'KnowledgeLabel' + os.sep+'corpus2' + os.sep
    model_file = dir +'knowledge_lda.bin'
    dictionary_file= dir +'dictionary.text'
    corpus_file =  dir +'corpus.csv'
    lda = LDAModel(path = dir,model_file = model_file,dictionary_file = dictionary_file,corpus_file =corpus_file)
    train_data_path = dir +'train_data.csv'
    test_data_path = dir +'test_data.csv'
    saveLDACorpus(train_data_path=train_data_path,test_data_path=test_data_path,model_file=model_file,dictionary_file=dictionary_file,corpus_file=corpus_file)
