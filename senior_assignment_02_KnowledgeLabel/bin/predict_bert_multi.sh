export BERT_BASE_DIR=/root/private/kelly/data/chinese_L-12_H-768_A-12
export MY_DATASET=/root/private/kelly/data/KnowledgeLabel/corpus2/
export CODEDIR=/root/private/kelly/p2/
export BPYTHONUNBUFFERED=1


python3 $CODEDIR/run_multilabel_classifier.py \
--do_predict=true \
--data_dir=$MY_DATASET/ \
--task_name=knowledgelabel \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$CODEDIR/config/bert_config.json \
--output_dir=$MY_DATASET/output \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt  \
--max_seq_length=128 >nohup_predict_multi.out &


