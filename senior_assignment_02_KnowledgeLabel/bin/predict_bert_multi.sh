export BERT_BASE_DIR=/Users/henry/Documents/application/multi-label-bert/data/chinese_roberta_wwm_ext_L-12_H-768_A-122
export MY_DATASET=/Users/henry/Documents/application/nlp_assignments/data/KnowledgeLabel/corpus2/
export CODEDIR=/Users/henry/Documents/application/nlp_assignments/senior_assignment_02_KnowledgeLabel/
export BPYTHONUNBUFFERED=1


python3 $CODEDIR/run_multilabel_classifier.py \
--do_predict=true \
--data_dir=$MY_DATASET/ \
--task_name=knowledgelabel \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$CODEDIR/congfig/bert_config.json \
--output_dir=$MY_DATASET/output \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt  \
--max_seq_length=128 >nohup_predict.out &


