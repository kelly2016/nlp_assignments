# -*- coding: utf-8 -*-
# @Time    : 2020-06-03 16:43
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : train_helper.py
# @Description:


import io
import json
import multiprocessing
import os
from collections import OrderedDict

import numpy as np
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.callbacks import TensorBoard
from tqdm import tqdm

from evaluate import evaluate as src_evaluate
from model.bert4keras.backend import keras
from model.bert4keras.snippets import open
from model.bert4keras.snippets import sequence_padding, DataGenerator
from model.bert4keras.tokenizers import Tokenizer
from reading_comprehension import Reading_Comprehension

#from keras.utils import training_utils
# 基本信息
maxlen = 512#128
epochs = 20
batch_size = 1
learing_rate = 2e-5
data_dir='/Users/henry/Documents/application/nlp_assignments/data/rc'
output_dir='/Users/henry/Documents/application/nlp_assignments/data/rc/output2'
bert_dir = '/Users/henry/Documents/application/multi-label-bert/data/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16'
#'/Users/henry/Documents/application/multi-label-bert/data/chinese_L-12_H-768_A-12'
    #
config_path = f'{bert_dir}/bert_config.json'
checkpoint_path = f'{bert_dir}/bert_model.ckpt'
dict_path = f'{bert_dir}/vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)
best_model_file = os.path.join(output_dir,'roberta_best_model.h5')
#nbr_gpus = len(training_utils._get_available_devices()) - 1
cores = multiprocessing.cpu_count()-1

def train():
    fine_tune = os.path.isfile(best_model_file)
    model = Reading_Comprehension(config_path,checkpoint_path,best_model_file,fine_tune=fine_tune).getModel()
    train_data = load_data(
        # os.path.join(data_dir,'train.json')
        os.path.join(data_dir, 'demo_train.json')
    )
    #数据生成
    train_generator = Data_generator(train_data, batch_size)
    #评价函数
    evaluator = Evaluator(model,'demo_dev.json')
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator],
        verbose=1,
        workers=cores,
        use_multiprocessing=True
    )


def train2():
    model = Reading_Comprehension(config_path, checkpoint_path, best_model_file).getModel()

    train_data = load_data(
        # os.path.join(data_dir,'train.json')
        os.path.join(data_dir, 'demo_train.json')
    )
    #数据生成
    train_generator = Data_generator(train_data, batch_size)
    val_data = load_data(
        # os.path.join(data_dir,'train.json')
        os.path.join(data_dir, 'demo_dev.json')
    )
    # 验证集数据生成
    val_generator = Data_generator(val_data, batch_size)
    #评价函数
    evaluator = Evaluator()

    reduce_lr = ReduceLROnPlateau(monitor=model.sparse_accuracy, factor=0.5, patience=1, verbose=1,
                                  min_lr=0.0001 )
    early_stop = EarlyStopping(monitor=model.sparse_accuracy, patience=1 , verbose=1, min_delta=0.001)#多少轮不变patience
    #tensorboard --logdir /Users/henry/Documents/application/zhitu2/branches/test/logs --port=6001
    callbacks = [ evaluator,reduce_lr, early_stop, TensorBoard(log_dir=os.path.join(data_dir, 'tb_log') )]

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        #validation_data=val_generator.forfit(),
        #validation_steps=len(val_generator),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        max_queue_size=10,
        workers=cores,
        use_multiprocessing=True
    )

def test():
    predict_to_file(os.path.join(data_dir, 'test1.json'), os.path.join(output_dir, 'pred1.json'))


class Data_generator(DataGenerator):
    """
    数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            context, question, answers = item[1:]
            token_ids, segment_ids = tokenizer.encode(
                question, context, max_length=maxlen
            )
            a = np.random.choice(answers)
            a_token_ids = tokenizer.encode(a)[0][1:-1]
            start_index = search(a_token_ids, token_ids)
            if start_index != -1:
                labels = [[start_index], [start_index + len(a_token_ids) - 1]]
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_labels = sequence_padding(batch_labels)
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def load_data(filename):
    D = []
    for d in json.load(open(filename))['data'][0]['paragraphs']: #list<dict>
        for qa in d['qas']:
            D.append([
                qa['id'], d['context'], qa['question'],
                [a['text'] for a in qa.get('answers', [])]
            ])
    return D

def predict_to_file(infile, out_file,model):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    R = {}
    for d in tqdm(load_data(infile)):
        a = extract_answer(d[2], d[1],model=model)
        R[d[0]] = a
    R = json.dumps(R, ensure_ascii=False, indent=4)
    fw.write(R)
    fw.close()

def extract_answer(question, context,model, max_a_len=16):
    """
    抽取答案函数
    """
    max_q_len = 64
    q_token_ids = tokenizer.encode(question, max_length=max_q_len)[0]
    c_token_ids = tokenizer.encode(
        context, max_length=maxlen - len(q_token_ids) + 1
    )[0]
    token_ids = q_token_ids + c_token_ids[1:]
    segment_ids = [0] * len(q_token_ids) + [1] * (len(c_token_ids) - 1)
    c_tokens = tokenizer.tokenize(context)[1:-1]
    mapping = tokenizer.rematch(context, c_tokens)
    probas = model.predict([[token_ids], [segment_ids]])[0] #返回的是 每一个位置成为stat和end的可能性
    probas = probas[:, len(q_token_ids):-1]
    start_end, score = None, -1
    for start, p_start in enumerate(probas[0]):
        for end, p_end in enumerate(probas[1]):
            if end >= start and end < start + max_a_len:
                if p_start * p_end > score: #通过start和end的联合score来判断组合
                    start_end = (start, end)
                    score = p_start * p_end
    start, end = start_end
    return context[mapping[start][0]:mapping[end][-1] + 1]

class Evaluator(keras.callbacks.Callback):

    """
    评估和保存模型
    """
    def __init__(self,model,val_dilename):
        self.best_val_f1 = 0.
        self.model = model
        self.val_dilename = val_dilename

    def evaluate(self, filename):
        """
        评测函数（官方提供评测脚本evaluate.py）
        """
        predict_to_file(filename, filename + '.pred.json',self.model)
        ref_ans = json.load(io.open(filename))
        pred_ans = json.load(io.open(filename + '.pred.json'))
        F1, EM, TOTAL, SKIP = src_evaluate(ref_ans, pred_ans)
        output_result = OrderedDict()
        output_result['F1'] = '%.3f' % F1
        output_result['EM'] = '%.3f' % EM
        output_result['TOTAL'] = TOTAL
        output_result['SKIP'] = SKIP
        return output_result

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(
            os.path.join(data_dir,self.val_dilename)
            # os.path.join(data_dir,'demo_dev.json')
        )
        if float(metrics['F1']) >= self.best_val_f1:
            self.best_val_f1 = float(metrics['F1'])
            #self.model.save_weights(best_model_file)
            self.model.save(best_model_file)
        metrics['BEST_F1'] = self.best_val_f1
        print(metrics)

if __name__ == '__main__':
    train()