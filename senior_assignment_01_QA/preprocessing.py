# -*- coding: utf-8 -*-
# @Time    : 2019-11-11 17:34
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : preprocessing.py
# @Description:1.作业1：
# 1.熟悉项目数据
# 2.分词以及清晰数据
# 3.通过训练数据和测试数据建立vocab词汇表
# 数据集下载地址：https://aistudio.baidu.com/aistudio/competition/detail/3
# ------------------------
# 作业1提交截止日期：2019年11月15日（周五）晚8点
# ------------------------
# 提交作业格式：建议使用zip压缩包上传个人作业
# 文件命名格式：“作业1+姓名+班级名+提交日期”，例如：【作业1+伊帆+图灵班+11.15】

import multiprocessing
import os
import setproctitle
from collections import Counter

import pandas as pd

import cutWords
from cutWords import Analyzer

analyzer = Analyzer(Analyzer.ANALYZERS.Jieba,replaceP=False,useStopwords=False,userdict ='/Users/henry/Documents/application/nlp_assignments/data/AutoMaster/userDict.txt')

cores = multiprocessing.cpu_count()
partitions = cores
import numpy as np

def preprocess(string):
    """
    和业务相关的字符串处理
    :param str:
    :return:
    """
    if string is  None or  type(string) != str or len(string) == 0:
        return ' '
    words = analyzer.cut2list(string.replace('[语音]','').replace('\t','').replace('\n','').replace('|','。').replace('[图片]',' ').replace(',','，'))#。这么做是因为ltp的切词好怪异"。德尔福"在一起
    return ' '.join(words)

def data_fram_proc(df):
    print( df.columns)
    if len(df.columns) == 6:
        #刷选过缺失值得新数据是存为副本还是直接在原数据上进行修改。
         #df[5] = df[5].apply(preprocess)
         df['Report'] = df['Report'].apply(preprocess)


    for col_name in  ['Brand','Model','Question','Dialogue']:
        df[col_name] = df[col_name].apply(preprocess)


    return df

def parallelize(df,func):
    """

    :param df:
    :param func:
    :return:
    """
    #将df横切
    data_split = np.array_split(df,partitions)
    pool  = multiprocessing.Pool(cores)
    data = pd.concat(pool.map(func,data_split))
    #关闭pool，使其不在接受新的任务。
    pool.close()
    #主进程阻塞，等待子进程的退出， join方法要在close或terminate之后使用。
    pool.join()
    return data



def deal2(src_file,output_file):
    """

        :param src_file: 待处理的源数据
        :param output_file: 处理后的数据
        :param dict: 处理完的字典，格式为词语 词频 ，eg， word 12
        :return:
        """
    src_df = pd.read_csv(src_file, encoding='utf-8',  sep=None)
    src_df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
    if len(src_df.columns) == 6:
         src_df.dropna(subset=['Question', 'Dialogue', 'Report'], how='any', inplace=True)
    else:
         src_df.dropna(subset=['Question', 'Dialogue'], how='any', inplace=True)
    src_df = parallelize(src_df,data_fram_proc)
    #分词过滤后可能又有空的
    src_df.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
    if len(src_df.columns) == 6:
        src_df.dropna(subset=['Question', 'Dialogue', 'Report'], how='any', inplace=True)
    else:
        src_df.dropna(subset=['Question', 'Dialogue'], how='any', inplace=True)

    src_df.to_csv(output_file,index=None,header =True)

'''
def deal(src_file,output_file):
    """

    :param src_file: 待处理的源数据
    :param output_file: 处理后的数据
    :param dict: 处理完的字典，格式为词语 词频 ，eg， word 12
    :return:
    """

    num_written_lines = 0
    with tf.gfile.GFile(output_file, "w") as writer:
        src_df = pd.read_csv(src_file, encoding='utf-8',header=None,sep =None)
        orLabels = ['QID','Brand','Model','Question','Dialogue','Report']

        title_line =  ",".join(label for label in orLabels) + "\n"
        writer.write(title_line)
        for j, values in enumerate(src_df.values):
            if j == 0:
                continue
            if  values[len(values)-1] == None:#如果Report无数值
                continue
            output_line =''

            for (i, v) in enumerate(values):
                if i == 0 :
                    output_line = v
                    continue
                if j == 52:
                    bug = 0
                if i  == 5 :
                    bug = 0
                line = ''
                #print('j = {} ,i = {} , v= {}'.format(j,i,v))
                if v is not None and type(v) == str and len(v) > 0:
                   words = analyzer.cut2list(preprocess(v))
                   counter.update(words)
                   line = ' '.join(words)
                else:
                   line = ' '
                output_line =  output_line + ","+line

            if len(values) == 5:
                output_line + output_line+ ","+' '


            writer.write(output_line+ '\n' )
            num_written_lines += 1
    print('total num_written_lines =',num_written_lines)
'''


def wirteDict(src_file,dictFile):
    src_df = pd.read_csv(src_file, encoding='utf-8',  sep = '\t')
    words = []
    for j, values in enumerate(src_df.values):

        if values[0]:
            if '\n' in values[0]:
                bug = 0
            words += values[0].split(' ')

    counter = Counter(words)
    f = open(dictFile, 'w')  # 若是'wb'就表示写二进制文件
    # 统计单词出现频次
    total_count = len(counter.items())

    # 计算单词频率
    words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    for k, v in words:
        f.write('{} {} \n'.format(k, v ))
        print('{} {} \n'.format(k, v ))
    f.close()



def createUserDict(src_file,output_file):
    """
    产生用户词典
    :param src_file:
    :param output_file:
    :return:
    """


    output = open(output_file, "w+", encoding="utf-8")
    src_df = pd.read_csv(src_file, encoding='utf-8', header=None, sep=None)
    src_df= src_df.iloc[1:,1:3]
    src_df = src_df.dropna()
    for j, values in enumerate(src_df.values):
        output.write( cutWords.strQ2B(values[0].strip()) + "\n")
        output.write(cutWords.strQ2B(values[1].strip()) + "\n")
    output.close()


def process(series):
    string = ''
    if series.size == 3:
       if type(series['Question']) == float:
            bug = 0
       if type(series['Dialogue']) == float:
            bug = 0
       if type(series['Report']) == float:
            bug = 0
       string =  series['Question'] + ' ' + series['Dialogue'] + ' ' + series['Report']
    else:
        string = series['Question'] + ' ' + series['Dialogue']
    return  string




def createEmbeddingCorpus(src_file,src_file2,output_file):
    """
    把csv专成txt
    :return:
    """
    src_df = pd.read_csv(src_file, encoding='utf-8', sep=None)
    src_df['merged'] = src_df[[ 'Question', 'Dialogue','Report']].apply( process,axis=1)#

    src_df2 = pd.read_csv(src_file2, encoding='utf-8',   sep=None)
    src_df2['merged'] = src_df[[ 'Question', 'Dialogue']].apply(process,axis=1)

    merged_df = pd.concat([src_df[['merged']],src_df2[['merged']]],axis=0)
    merged_df.to_csv(output_file,index=None,header = False)
    print('train data size {},test data size {},merged_df data size {}'.format(len(src_df), len(src_df2),
                                                                               len(merged_df)))

def pad_proc(sentence, max_len, vocab):
    '''
    < start > < end > < pad > < unk >
    '''
    # 0.按空格统计切分出词
    words = sentence.strip().split(' ')
    # 1. 截取规定长度的词数
    words = words[:max_len]
    # 2. 填充< unk > ,判断是否在vocab中, 不在填充 < unk >
    sentence = [word if word in vocab else '<UNK>' for word in words]
    # 3. 填充< start > < end >
    sentence = ['<START>'] + sentence + ['<STOP>']
    # 4. 判断长度，填充　< pad >
    sentence = sentence + ['<PAD>'] * (max_len  - len(words))
    return ' '.join(sentence)

def get_max_len(data):
    """
    获得合适的最大长度值
    :param data: 待统计的数据  train_df['Question']
    :return: 最大长度值
    """
    max_lens = data.apply(lambda x: x.count(' ')+1)
    return int(np.mean(max_lens) + 2 * np.std(max_lens))

def formatDataset(trainFile,testFile,vocab,train_x_pad_path,train_y_pad_path,test_x_pad_path):
    """
     格式化好训练的数据集
    :param trainFile: 训练集
    :param testFile: 测试集
    :param vocab: 词表
    :return:
    """
    #读入分好词的数据
    train_df = pd.read_csv(trainFile, encoding='utf-8', sep=None)
    test_df = pd.read_csv(testFile, encoding='utf-8', sep=None)
    #合并训练数据
    train_df['X'] = train_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    test_df['X'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

    # 获取输入数据 适当的最大长度
    x_max_len = max(get_max_len(test_df['X']), get_max_len(train_df['X']))
    # 获取标签数据 适当的最大长度
    train_y_max_len = get_max_len(train_df['Report'])
    print('x_max_len = {} ,train_y_max_len = {} '.format(x_max_len,train_y_max_len))
    #对数据进行格式化：截取，填充
    # 训练集X处理
    train_df['X'] = train_df['X'].apply(lambda x: pad_proc(x, x_max_len, vocab))
    # 训练集Y处理
    train_df['Y'] = train_df['Report'].apply(lambda x: pad_proc(x, train_y_max_len, vocab))
    # 测试集X处理
    test_df['X'] = test_df['X'].apply(lambda x: pad_proc(x, x_max_len, vocab))

    #保存中间结果
    train_df['X'].to_csv(train_x_pad_path, index=None, header=False)
    train_df['Y'].to_csv(train_y_pad_path, index=None, header=False)
    test_df['X'].to_csv(test_x_pad_path, index=None, header=False)

    return  train_df,test_df,x_max_len,train_y_max_len

# 遇到未知词就填充unk的索引
def transform_data(sentence,vocab,unkownchar = '<UNK>'):
    unk_index = vocab[unkownchar]
    # 字符串切分成词
    words=sentence.split(' ')
    # 按照vocab的index进行转换
    ids=[vocab[word] if word in vocab else unk_index for word in words]
    return ids

def load_dataset(file,vocab):
    """
    将输入语料文字转换成索引index
    :param file:
    :param vocab:
    :return:
    """
    df = pd.read_csv(file, encoding='utf-8', header=None,  sep= '\t')
    # 将词转换成索引  [<START> 方向机 重 ...] -> [32800, 403, 986, 246, 231
    ids = df[df.columns[0]].apply(lambda x: transform_data(x, vocab))

    # 将索引列表转换成矩阵 [32800, 403, 986, 246, 231] --> array([[32800,   403,   986 ]]
    return np.array(ids.tolist())




def mergeDataset(trainFile,testFile,train_x_pad_path,train_y_pad_path,test_x_pad_path):
    """
     格式化好训练的数据集
    :param trainFile: 训练集
    :param testFile: 测试集
    :param vocab: 词表
    :return:
    """
    #读入分好词的数据
    train_df = pd.read_csv(trainFile, encoding='utf-8', sep=None)
    test_df = pd.read_csv(testFile, encoding='utf-8', sep=None)
    #合并训练数据
    train_df['X'] = train_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    train_df['Y'] = train_df[['Report']].apply(lambda x: ' '.join(x), axis=1)
    test_df['X'] = test_df[['Question', 'Dialogue']] .apply(lambda x: ' '.join(x), axis=1)

    #保存中间结果
    train_df['X'].to_csv(train_x_pad_path, index=None, header=False)
    train_df['Y'].to_csv(train_y_pad_path, index=None, header=False)
    test_df['X'].to_csv(test_x_pad_path, index=None, header=False)

    return  train_df,test_df



if __name__ == '__main__':
    setproctitle.setproctitle('kelly')
    dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep+ 'AutoMaster' + os.sep
    print(dir )

    '''
    #创建专业词表
    src_file = '/Users/henry/Documents/application/nlp_assignments/data/AutoMaster/AutoMaster_TrainSet.csv'
    output_file = '/Users/henry/Documents/application/nlp_assignments/data/AutoMaster/userDict.txt'
    createUserDict(src_file, output_file)
    src_file = '/Users/henry/Documents/application/nlp_assignments/data/AutoMaster/AutoMaster_TestSet.csv'
    output_file = '/Users/henry/Documents/application/nlp_assignments/data/AutoMaster/userDict.txt'
    createUserDict(src_file, output_file)
    '''

    #生成训练s2s模型的训练集测试集

    src_file =dir + 'AutoMaster_TrainSet.csv'
    output_file = dir +'AutoMaster_TrainSet_cleared.csv'
    #deal2(src_file, output_file)

    src_file2 = dir +'AutoMaster_TestSet.csv'
    output_file2 = dir +'AutoMaster_TestSet_cleared.csv'
    #deal2(src_file2, output_file2)

    '''
    #生成训练v2w的语料
    output_file3 = dir + 'trainv2wcotpus_jieba.csv'
    createEmbeddingCorpus(output_file, output_file2, output_file3)

    #生成字典
    dictFile =dir + 'AutoMaster_Counter_jieba.txt'
    wirteDict(output_file3,dictFile)
   '''

    trainFile = output_file
    testFile =  output_file2
    modelFile = dir +  'fasttext/fasttext_jieba.model'

    train_x_pad_path = dir + 'AutoMaster_Train_X_cleared.csv'
    train_y_pad_path = dir + 'AutoMaster_Train_Y_cleared.csv'
    test_x_pad_path = dir + 'AutoMaster_Test_X_cleared.csv'

    mergeDataset(trainFile, testFile,  train_x_pad_path, train_y_pad_path, test_x_pad_path)
    '''
    vocab, reverse_vocab, embedding_matrix   = util.getEmbedding_matrixFromModel(modelFile)

    train_x_pad_path =dir + 'AutoMaster_Train_X_jieba.csv'
    train_y_pad_path = dir + 'AutoMaster_Train_Y_jieba.csv'
    test_x_pad_path = dir + 'AutoMaster_Test_X_jieba.csv'
    formatDataset(trainFile, testFile, vocab, train_x_pad_path, train_y_pad_path, test_x_pad_path)
    '''