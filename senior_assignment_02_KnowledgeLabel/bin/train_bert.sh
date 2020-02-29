export BERT_BASE_DIR=/Users/henry/Documents/application/multi-label-bert/data/chinese_roberta_wwm_ext_L-12_H-768_A-122
export MY_DATASET=/Users/henry/Documents/application/nlp_assignments/data/KnowledgeLabel/corpus2/
export CODEDIR=/Users/henry/Documents/application/nlp_assignments/senior_assignment_02_KnowledgeLabel/
export BPYTHONUNBUFFERED=1

nohup python3 $CODEDIR/bert_run_multilabel_classifier.py \
--do_train=true \
--do_eval=true \
--do_predict=false \
--data_dir=$MY_DATASET/ \
--task_name=knowledgelabel \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$CODEDIR/config/bert_config.json \
--output_dir=$MY_DATASET/output \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=8 \
--learning_rate=5e-5 \
--num_train_epochs=4.0 \
--eval_batch_size=8 >nohup_train.out &

