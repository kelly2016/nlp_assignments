# -*- coding: utf-8 -*-
# @Time    : 2019-09-14 10:23
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : classification.py
# @Description:各种分类器
import tensorflow as tf
import numpy as np
import  preprocessing
import datetime
import os
import matplotlib.pyplot as plt
#train_subset = 10000
batch_size = 32
#beta = 0.5
#句向量的纬度
fearture_num = 100



def  neuralTrain(num_steps = 4000,modelDir = '/Users/henry/Documents/application/nlp_assignments/data/'):
    (train_dataset, train_labels), (valid_dataset, valid_labels), (
        test_dataset, test_labels),labelsSet = preprocessing.getDataSet()
    print('train_datasetnum = {},valid_dataset num = {} ,test_dataset num = {} '.format(len(train_dataset), len(valid_dataset),  len(test_labels)))
    num_labels = len(labelsSet)
    nodes_num = 1024
    graph = tf.Graph()

    with graph.as_default():
        '''
        所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        '''
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, fearture_num))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        weights_1 = tf.Variable(tf.truncated_normal([fearture_num, nodes_num]))
        biases_1 = tf.Variable(tf.zeros([nodes_num]))

        weights_2 = tf.Variable(tf.truncated_normal([nodes_num, num_labels]))
        biases_2 = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
        relu_layer = tf.nn.relu(logits_1)
        # Dropout on hidden layer: RELU layer
        keep_prob = 0.9
        relu_layer_dropout = tf.nn.dropout(relu_layer, keep_prob)

        logits_2 = tf.matmul(relu_layer_dropout, weights_2) + biases_2

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_2))
        # Regularization是不是计算越多l2的时候要加完？
        #regularization = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
        #loss = tf.reduce_mean(loss + beta * regularization)

        start_learning_rate = 0.01
        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)#, global_step=global_step

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits_2)
        # for valid_prediction
        logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
        relu_layer = tf.nn.relu(logits_1)
        logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
        valid_prediction = tf.nn.softmax(logits_2)
        # for test
        logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
        relu_layer = tf.nn.relu(logits_1)
        logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
        test_prediction = tf.nn.softmax(logits_2)
        saver = tf.train.Saver()

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print("initialized")
            losses = []
            ValiAcc = []
            maxAccuracy = 0.0
            lastModel = ''
            for step in range(num_steps):
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                # Generate a minibatch.
                batch_data = train_dataset[offset:(offset + batch_size), :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                losses.append(l)
                if (step % 100 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    acc = accuracy(valid_prediction.eval(), valid_labels)
                    ValiAcc.append(acc)
                    print("Validation accuracy: %.1f%%" %acc)
                    if acc > maxAccuracy:
                        maxAccuracy=acc
                        nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        lastModel = modelDir + '/' + nowTime + '_model.ckpt'
                        saver.save(session, lastModel)
                        print("Save current model  : ", lastModel)

            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
            print("MaxAccuracy validation accuracy:: %.1f%%" % maxAccuracy)
            print("LastModel is : " ,lastModel)
            plt.plot(losses)
            plt.plot(ValiAcc)



def accuracy(predictions,labels):
    #numpy.argmax(a, axis=None, out=None) 返回沿轴axis最大值的索引。
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

if __name__=='__main__':
    dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + 'data' + os.sep
    print('dir = ', dir)
    neuralTrain( modelDir=dir)

