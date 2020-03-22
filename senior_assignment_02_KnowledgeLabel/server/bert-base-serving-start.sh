#!/usr/bin/env bash
#-model_pb_dir $TRAINED_CLASSIFIER\
#-ckpt_name model.ckpt-2000 \

export BERT_BASE_DIR=/root/private/kelly/data/chinese_L-12_H-768_A-12
export TRAINED_CLASSIFIER=/root/private/kelly/data/KnowledgeLabel/corpus2/output
export EXP_NAME=mobile_0

nohup bert-base-serving-start \
    -model_dir $TRAINED_CLASSIFIER  \
    -model_pb_dir $TRAINED_CLASSIFIER \
    -bert_model_dir $BERT_BASE_DIR \
    -max_seq_len 128 \
    -http_port 7007 \
    -port 8888 \
    -cpu \
    -port_out 6006 \
    -mode CLASS >nohup_base_sever.out &