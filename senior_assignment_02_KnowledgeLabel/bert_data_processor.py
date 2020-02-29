import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from model.bert import tokenization
import multiprocessing
import os
import setproctitle
from sklearn.utils import shuffle

cores = multiprocessing.cpu_count()
setproctitle.setproctitle('kelly')
dir = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'KnowledgeLabel' + os.sep+'corpus2' + os.sep
file = dir + 'corpus.csv'


DATA_OUTPUT_DIR = dir
vocab_file = '/Users/henry/Documents/application/multi-label-bert/data/chinese_roberta_wwm_ext_L-12_H-768_A-122/vocab.txt'

# Load data

df = pd.read_csv(file, dtype=str) #names=["label", "item","multiLabels"],
df = shuffle(df)
#输出全部label
label = df.label.apply(lambda x: x.split())
tmp = []
for row in label:
    tmp.append(row[0])
label  = tmp
label = set(label)
print(len(label))
print(label)
print('------------------')
label2 = df.multiLabels.apply(lambda x: x.split())
tmp = []
for row in label2:
    for r in row:
        tmp.append(r)
label2  = tmp
label2 = set(label2)
print(len(label2))
print(label2)

# Create files
if not os.path.exists(DATA_OUTPUT_DIR):
    os.makedirs(os.path.join(DATA_OUTPUT_DIR, "train"))
    os.makedirs(os.path.join(DATA_OUTPUT_DIR, "valid"))
    os.makedirs(os.path.join(DATA_OUTPUT_DIR, "test"))

# Split dataset
df_train = df[:int(len(df) * 0.8)]
df_valid = df[int(len(df) * 0.8):int(len(df) * 0.9)]
df_test = df[int(len(df) * 0.9):]

# Save dataset
file_set_type_list = ["train", "valid", "test"]
for file_set_type, df_data in zip(file_set_type_list, [df_train, df_valid, df_test]):
    predicate_out_f = open(os.path.join(os.path.join(DATA_OUTPUT_DIR, file_set_type),
                                        "predicate_out.txt"), "w",
                           encoding='utf-8')
    predicate_out_single_f = open(os.path.join(os.path.join(DATA_OUTPUT_DIR, file_set_type),
                                        "predicate_single_out.txt"), "w",
                           encoding='utf-8')
    text_f = open(os.path.join(os.path.join(DATA_OUTPUT_DIR, file_set_type),
                               "text.txt"), "w",
                  encoding='utf-8')
    token_in_f = open(os.path.join(os.path.join(DATA_OUTPUT_DIR, file_set_type),
                                   "token_in.txt"), "w",
                      encoding='utf-8')
    token_in_not_UNK_f = open(os.path.join(os.path.join(DATA_OUTPUT_DIR, file_set_type),
                                           "token_in_not_UNK.txt"), "w",
                              encoding='utf-8')

    # Processing
    bert_tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)  # 初始化 bert_token 工具
    # feature
    text = '\n'.join(df_data.item)
    text_tokened = df_data.item.apply(bert_tokenizer.tokenize)
    text_tokened = '\n'.join([' '.join(row) for row in text_tokened])
    text_tokened_not_UNK = df_data.item.apply(bert_tokenizer.tokenize_not_UNK)
    text_tokened_not_UNK = '\n'.join([' '.join(row) for row in text_tokened_not_UNK])
    # label only choose first 3 lables: 高中 学科 一级知识点
    # if you want all labels
    # just remove list slice
    predicate_list = df_data.multiLabels.apply(lambda x: x.split())
    predicate_list_str = '\n'.join([' '.join(row) for row in predicate_list])

    #single label
    predicate_list_single = df_data.label.apply(lambda x: x.split())
    predicate_list_single_str = '\n'.join([' '.join(row) for row in predicate_list_single])

    print(f'datasize: {len(df_data)}')
    text_f.write(text)
    token_in_f.write(text_tokened)#不在词表中的用unk代替
    token_in_not_UNK_f.write(text_tokened_not_UNK) #不在词表中的词是原型
    predicate_out_f.write(predicate_list_str)
    predicate_out_single_f.write(predicate_list_single_str)

    text_f.close()
    token_in_f.close()
    token_in_not_UNK_f.close()
    predicate_out_f.close()
    predicate_out_single_f.close()

