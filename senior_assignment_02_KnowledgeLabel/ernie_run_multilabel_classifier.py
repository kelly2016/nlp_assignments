# -*- coding: utf-8 -*-
# @Time    : 2020-03-10 16:46
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : ernie_multi_label_classifier.py.py
# @Description:
"""Finetuning on classification task """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import codecs
import csv
import os

import numpy as np
import paddlehub as hub
import pandas as pd
from paddlehub.common.logger import logger
from paddlehub.dataset.dataset import InputExample, BaseDataset
from sklearn.metrics import f1_score

import ernie_data_processor

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=1, help="Total examples' number in batch for training.")
parser.add_argument("--use_taskid", type=ast.literal_eval, default=False, help="Whether to user ernie v2 , if not to use bert.")
args = parser.parse_args()

# yapf: enable.
class KLBaseDataset(BaseDataset):
    """
    ChnSentiCorp (by Tan Songbo at ICT of Chinese Academy of Sciences, and for
    opinion mining)
    """

    def __init__(self,dataset_dir):
        self.dataset_dir = dataset_dir
        if not os.path.exists(self.dataset_dir):
            logger.info("Dataset not exists.".format(self.dataset_dir))
        else:
            logger.info("Dataset {} already cached.".format(self.dataset_dir))

        self._load_train_examples()
        self._load_test_examples()
        self._load_dev_examples()

    def _load_train_examples(self):
        self.train_file = os.path.join(self.dataset_dir, "train.tsv")
        self.train_examples = self._read_tsv(self.train_file)

    def _load_dev_examples(self):
        self.dev_file = os.path.join(self.dataset_dir, "dev.tsv")
        self.dev_examples = self._read_tsv(self.dev_file)

    def _load_test_examples(self):
        self.test_file = os.path.join(self.dataset_dir, "test.tsv")
        self.test_examples = self._read_tsv(self.test_file)

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_labels(self):
        return pd.read_csv(os.path.join(self.dataset_dir ,'classes.csv'),names=['labels'])['labels'].tolist()

    @property
    def num_labels(self):
        """
        Return the number of labels in the dataset.
        """
        return len(self.get_labels())

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with codecs.open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            examples = []
            seq_id = 0
            header = next(reader)  # skip header
            for line in reader:
                example = InputExample(
                    guid=seq_id, label=[int(i) for i in line[0].split(' ')], text_a=line[1])
                seq_id += 1
                examples.append(example)

            return examples

def inverse_predict_array(batch_result):
    return np.argmax(batch_result, axis=2).T

def evaluate(dataset,multi_label_cls_task,mlb):

    # 预测数据
    data = [[d.text_a, d.text_b] for d in dataset.get_test_examples()]
    # 预测标签
    test_label = np.array([d.label for d in dataset.get_test_examples()])
    # 预测
    run_states = multi_label_cls_task.predict(data)
    results = [run_state.run_results for run_state in run_states]
    predict_label = np.concatenate([inverse_predict_array(batch_result) for batch_result in results])
    print('f1 micro:{}'.format(f1_score(test_label, predict_label, average='micro')))
    print('f1 samples:{}'.format(f1_score(test_label, predict_label, average='samples')))
    print('f1 macro:{}'.format(f1_score(test_label, predict_label, average='macro')))
    mlb.inverse_transform(predict_label)[2:5]
    mlb.inverse_transform(test_label)[2:5]

if __name__ == '__main__':
    # Load Paddlehub BERT pretrained model
    import setproctitle

    # Finetune and evaluate by PaddleHub's API
    # will finish training, evaluation, testing, save model automatically
    setproctitle.setproctitle('kelly')

    max_seq_len = 256
    use_gpu = False
    weight_decay = 0.01
    learning_rate = 5e-5
    warmup_proportion = 0.0
    num_epoch = 10
    batch_size = 32
    checkpoint_dir = '/Users/henry/Documents/application/nlp_assignments/data/KnowledgeLabel/corpus2/chnsenticorp/output/checkpoints/'

    module = hub.Module(name="bernie_v2_eng_base")  # (name="bert_chinese_L-12_H-768_A-12") 'ernie'
    inputs, outputs, program = module.context(trainable=True, max_seq_len=max_seq_len)
    # Download dataset and use MultiLabelReader to read dataset
    dataset = KLBaseDataset()

    reader = hub.reader.MultiLabelClassifyReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=max_seq_len,
        use_task_id=False)

    metrics_choices = ['acc', 'f1']
    # Construct transfer learning network
    # Use "pooled_output" for classification tasks on an entire sentence.

    #优化器设置
    # Select finetune strategy, setup config and finetune
    strategy = hub.AdamWeightDecayStrategy(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        warmup_proportion=warmup_proportion,
        lr_scheduler="linear_decay")

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        use_cuda=use_gpu,
        num_epoch=num_epoch,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        strategy=strategy)

    #运行模型
    pooled_output = outputs["pooled_output"]
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name
    ]
    # Define a classfication finetune task by PaddleHub's API
    multi_label_cls_task = hub.MultiLabelClassifierTask(
        data_reader=reader,
        feature=pooled_output,
        feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config,
        metrics_choices = metrics_choices)

    dir = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'KnowledgeLabel' + os.sep + 'corpus2' + os.sep
    multi_label_cls_task.finetune_and_eval()

    y, mlb, df =ernie_data_processor.getMlb(dir)
    evaluate(dataset, multi_label_cls_task, mlb)