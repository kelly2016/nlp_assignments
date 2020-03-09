set -eux

export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
#export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=/Users/henry/Documents/application/nlp_assignments/data/KnowledgeLabel/ernie/ERNIE_1.0_max-len-512/
export TASK_DATA_PATH=/root/private/kelly/data/KnowledgeLabel/corpus2/
export PYTHONPATH=/root/private/kelly/p2/
python -u ${PYTHONPATH}/run_classifier.py \
                   --use_cuda false \
                   --verbose true \
                   --do_train true \
                   --do_val true \
                   --do_test true \
                   --batch_size 4 \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --train_set ${TASK_DATA_PATH}/chnsenticorp/train.tsv \
                   --dev_set ${TASK_DATA_PATH}/chnsenticorp/dev.tsv,
                   --test_set ${TASK_DATA_PATH}/chnsenticorp/test.tsv \
                   --vocab_path ${MODEL_PATH}/vocab.txt \
                   --checkpoints ./checkpoints \
                   --save_steps 10 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --validation_steps 5 \
                   --epoch 10 \
                   --max_seq_len 256 \
                   --ernie_config_path ${MODEL_PATH}/ernie_config.json \
                   --learning_rate 5e-5 \
                   --skip_steps 1 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --random_seed 1