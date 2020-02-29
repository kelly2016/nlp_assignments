# -*- coding: utf-8 -*-
# @Time    : 2020-02-02 14:52
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : train.py
# @Description:
import os
import setproctitle
import time

import tensorflow as tf

from model.transformer_tf2.layers import CustomSchedule
from model.transformer_tf2.model import Transformer
from model.transformer_tf2.utils import create_padding_mask
from utils.metrics import micro_f1, macro_f1

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = 50000
output_dim = 95
dropout_rate = 0.1
maximum_position_encoding=10000
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
loss_function=tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, output_dim,
                          maximum_position_encoding,
                          rate=dropout_rate)

checkpoint_path = "data/checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 如果检查点存在，则恢复最新的检查点。
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

EPOCHS = 10
# 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地
# 执行。该函数专用于参数张量的精确形状。为了避免由于可变序列长度或可变
# 批次大小（最后一批次较小）导致的再追踪，使用 input_signature 指定
# 更多的通用形状。

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    enc_padding_mask = create_padding_mask(inp)

    with tf.GradientTape() as tape:
        predictions = transformer(inp, True, enc_padding_mask=enc_padding_mask)
        loss = loss_function(tar, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar, predictions)

    mi_f1 = micro_f1(tar, predictions)
    ma_f1 = macro_f1(tar, predictions)
    return mi_f1, ma_f1

def predict(inp,tar,enc_padding_mask):
    predictions = transformer(inp,False,enc_padding_mask=enc_padding_mask)
    mi_f1=micro_f1(tar, predictions)
    ma_f1=macro_f1(tar, predictions)
    return mi_f1,ma_f1

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# 将数据集缓存到内存中以加快读取速度。
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(params.BUFFER_SIZE,reshuffle_each_iteration=True).batch(params.BATCH_SIZE, drop_remainder=True)

test_dataset=test_dataset.batch(params.BATCH_SIZE)
# 流水线技术 重叠训练的预处理和模型训练步骤。当加速器正在执行训练步骤 N 时，CPU 开始准备步骤 N + 1 的数据。这样做可以将步骤时间减少到模型训练与抽取转换数据二者所需的最大时间（而不是二者时间总和）。
# 没有流水线技术，CPU 和 GPU/TPU 大部分时间将处于闲置状态:
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_dataset):
        mic_f1, mac_f1 = train_step(inp, tar)

        if batch % 50 == 0:
            test_input, test_target = next(iter(test_dataset))
            enc_padding_mask = create_padding_mask(test_input)
            val_mic_f1, val_mac_f1 = predict(test_input, test_target, enc_padding_mask)

            print(
                'Epoch {} Batch {} Loss {:.4f} micro_f1 {:.4f} macro_f1 {:.4f} val_micro_f1 {:.4f} val_macro_f1 {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), mic_f1, mac_f1, val_mic_f1, val_mac_f1))

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

if __name__=='__main__':
    setproctitle.setproctitle('kelly')
    dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep + 'KnowledgeLabel' + os.sep+'corpus' + os.sep
    preprocessing(dir)







