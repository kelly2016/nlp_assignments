# -*- coding: utf-8 -*-
# @Time    : 2020-03-22 13:47
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : bert_server.py
# @Description:

import setproctitle

from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser


def start():

    args = get_args_parser().parse_args(['-model_dir', '/Users/henry/Documents/application/multi-label-bert/data/chinese_L-12_H-768_A-12/',
                                         '-tuned_model_dir','/Users/henry/Documents/application/nlp_assignments/data/KnowledgeLabel/corpus2/output/',
                                         '-port', '12544',
                                         '-ckpt_name','model.ckpt-1000',
                                         '-port_out', '12546',
                                         '-http_port','12547',
                                         '-max_seq_len', '128',
                                         '-mask_cls_sep',
                                         '-show_tokens_to_client',
                                         '-pooling_strategy','NONE',
                                         '-cpu'])
    server = BertServer(args)
    server.start()

def stop(port=12544,port_out=12546):
    BertServer.shutdown(port=port,port_out=port_out)

if __name__ == '__main__':
    setproctitle.setproctitle('bertshow')
    start()
    #stop(port=11068,port_out=11070)

