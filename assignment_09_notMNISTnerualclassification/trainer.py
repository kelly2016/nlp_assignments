# -*- coding: utf-8 -*-
# @Time    : 2019-09-02 14:56
# @Author  : Kelly
# @Email   : 289786098@qq.com
# @File    : traditonalTrainer.py
# @Description:

import  classification
import imagePreprocess
import tensorflow as tf
import numpy as np

image_size = 28
num_labels = 10

train_subset = 10000
batch_size = 32
beta = 0.5




def  nnTrain8(num_steps = 4000):
    """
     加1层1024隐藏层+learning_rate+Dropout+正则
      Minibatch loss at step 3900: 2.540312
     Minibatch accuracy: 34.4%
     Validation accuracy: 32.0%
     Test accuracy: 38.1%


    :return:
    """
    (train_dataset, train_labels), (valid_dataset, valid_labels), (
        test_dataset, test_labels) = imagePreprocess.getDataSet()
    train_dataset, train_labels = imagePreprocess.reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = imagePreprocess.reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = imagePreprocess.reformat(test_dataset, test_labels)
    nodes_num = 1024
    graph = tf.Graph()

    with graph.as_default():
        '''
        所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        '''
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, nodes_num]))
        biases_1 = tf.Variable(tf.zeros([nodes_num]))

        weights_2 = tf.Variable(tf.truncated_normal([nodes_num, num_labels]))
        biases_2 = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
        relu_layer = tf.nn.relu(logits_1)
        # Dropout on hidden layer: RELU layer
        keep_prob = 0.5
        relu_layer_dropout = tf.nn.dropout(relu_layer, keep_prob)

        logits_2 = tf.matmul(relu_layer_dropout, weights_2) + biases_2

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_2))
        # Regularization是不是计算越多l2的时候要加完？
        regularization = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
        loss = tf.reduce_mean(loss + beta * regularization)

        start_learning_rate = 0.01
        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

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

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print("initialized")
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
                if (step % 100 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

def  nnTrain7(num_steps = 4000):
    """
     加1层1024隐藏层+learning_rate+Dropout
    Minibatch loss at step 3900: 12.434942
    Minibatch accuracy: 81.2%
    Validation accuracy: 81.0%
    Test accuracy: 86.9%


    :return:
    """
    (train_dataset, train_labels), (valid_dataset, valid_labels), (
        test_dataset, test_labels) = imagePreprocess.getDataSet()
    train_dataset, train_labels = imagePreprocess.reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = imagePreprocess.reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = imagePreprocess.reformat(test_dataset, test_labels)
    nodes_num = 1024
    graph = tf.Graph()

    with graph.as_default():
        '''
        所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        '''
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, nodes_num]))
        biases_1 = tf.Variable(tf.zeros([nodes_num]))

        weights_2 = tf.Variable(tf.truncated_normal([nodes_num, num_labels]))
        biases_2 = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
        relu_layer = tf.nn.relu(logits_1)
        # Dropout on hidden layer: RELU layer
        keep_prob = 0.5
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

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print("initialized")
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
                if (step % 100 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

def  nnTrain6(num_steps = 4000):
    """
     加1层1024隐藏层+learning_rate
     Minibatch loss at step 3900: 11.188302
     Minibatch accuracy: 78.1%
     Validation accuracy: 79.7%
     Test accuracy: 86.2%

    :return:
    """
    (train_dataset, train_labels), (valid_dataset, valid_labels), (
        test_dataset, test_labels) = imagePreprocess.getDataSet()
    train_dataset, train_labels = imagePreprocess.reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = imagePreprocess.reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = imagePreprocess.reformat(test_dataset, test_labels)
    nodes_num = 1024
    graph = tf.Graph()

    with graph.as_default():
        '''
        所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        '''
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, nodes_num]))
        biases_1 = tf.Variable(tf.zeros([nodes_num]))

        weights_2 = tf.Variable(tf.truncated_normal([nodes_num, num_labels]))
        biases_2 = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
        relu_layer = tf.nn.relu(logits_1)
        logits_2 = tf.matmul(relu_layer, weights_2) + biases_2

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
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

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

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print("initialized")
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
                if (step % 100 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


def  nnTrain5(num_steps = 4000):
    """
     加2层1024隐藏层
     Minibatch loss at step 4900: 14.449371
     Minibatch accuracy: 59.4%
     Validation accuracy: 59.5%
    Test accuracy: 66.3%
    :return:
    """
    (train_dataset, train_labels), (valid_dataset, valid_labels), (
    test_dataset, test_labels) = imagePreprocess.getDataSet()
    train_dataset, train_labels = imagePreprocess.reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = imagePreprocess.reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = imagePreprocess.reformat(test_dataset, test_labels)
    nodes_num = 1024
    graph = tf.Graph()

    with graph.as_default():
        '''
        所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        '''
        tf_train_dataset = tf.placeholder(tf.float32,shape = (batch_size,image_size*image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        #其实定义的是中乘法
        weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, nodes_num]))
        biases_1 = tf.Variable(tf.zeros([nodes_num]))

        weights_2 = tf.Variable(tf.truncated_normal([nodes_num, nodes_num]))
        biases_2 = tf.Variable(tf.zeros([nodes_num]))

        weights_3 = tf.Variable(tf.truncated_normal([nodes_num, num_labels]))
        biases_3 = tf.Variable(tf.zeros([num_labels]))


        # Training computation.
        logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
        relu_layer_1 = tf.nn.relu(logits_1)

        logits_2 = tf.matmul(relu_layer_1, weights_2) + biases_2
        relu_layer_2 = tf.nn.relu(logits_2)

        logits_3 = tf.matmul(relu_layer_2, weights_3) + biases_3

        loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_3))
        #Regularization是不是计算越多l2的时候要加完？
        #regularization = tf.nn.l2_loss(weights_1)+tf.nn.l2_loss(weights_2)+tf.nn.l2_loss(weights_3)
        #loss = tf.reduce_mean(loss + beta * regularization)



        learning_rate = 0.01
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits_3)
        #for valid_prediction
        logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
        relu_layer_1 = tf.nn.relu(logits_1)

        logits_2 = tf.matmul(relu_layer_1, weights_2) + biases_2
        relu_layer_2 = tf.nn.relu(logits_2)

        logits_3 = tf.matmul(relu_layer_2, weights_3) + biases_3
        valid_prediction = tf.nn.softmax(logits_3)
        #for test
        logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
        relu_layer_1 = tf.nn.relu(logits_1)

        logits_2 = tf.matmul(relu_layer_1, weights_2) + biases_2
        relu_layer_2 = tf.nn.relu(logits_2)

        logits_3 = tf.matmul(relu_layer_2, weights_3) + biases_3
        test_prediction = tf.nn.softmax(logits_3)

        with tf.Session(graph = graph) as session:
            tf.global_variables_initializer().run()
            print("initialized")
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
                if (step % 100 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

def  nnTrain4(num_steps = 4000):
    """
     加一层1024隐藏层
     Minibatch loss at step 3900: 17.288506
     Minibatch accuracy: 78.1%
     Validation accuracy: 78.0%
     Test accuracy: 85.8%
    :return:
    """
    (train_dataset, train_labels), (valid_dataset, valid_labels), (
    test_dataset, test_labels) = imagePreprocess.getDataSet()
    train_dataset, train_labels = imagePreprocess.reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = imagePreprocess.reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = imagePreprocess.reformat(test_dataset, test_labels)
    nodes_num = 1024
    graph = tf.Graph()

    with graph.as_default():
        '''
        所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        '''
        tf_train_dataset = tf.placeholder(tf.float32,shape = (batch_size,image_size*image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, nodes_num]))
        biases_1 = tf.Variable(tf.zeros([nodes_num]))

        weights_2 = tf.Variable(tf.truncated_normal([nodes_num, num_labels]))
        biases_2 = tf.Variable(tf.zeros([num_labels]))


        # Training computation.
        logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
        relu_layer = tf.nn.relu(logits_1)
        logits_2 = tf.matmul(relu_layer, weights_2) + biases_2

        loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_2))
        #Regularization是不是计算越多l2的时候要加完？
        #regularization = tf.nn.l2_loss(weights_1)+tf.nn.l2_loss(weights_2)
        #loss = tf.reduce_mean(loss + beta * regularization)

        loss = tf.nn.l2_loss(tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_2)))

        learning_rate = 0.01
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits_2)
        #for valid_prediction
        logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
        relu_layer = tf.nn.relu(logits_1)
        logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
        valid_prediction = tf.nn.softmax(logits_2)
        #for test
        logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
        relu_layer = tf.nn.relu(logits_1)
        logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
        test_prediction = tf.nn.softmax(logits_2)

        with tf.Session(graph = graph) as session:
            tf.global_variables_initializer().run()
            print("initialized")
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
                if (step % 100 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


def  nnTrain3(num_steps = 4000):

    """
     stochastic gradient descent training，去掉卷基层，去掉赤化层
     Minibatch accuracy: 93.8%
     Validation accuracy: 87.4%
    Test accuracy: 93.4%
    :return:
    """
    (train_dataset, train_labels), (valid_dataset, valid_labels), (
    test_dataset, test_labels) = imagePreprocess.getDataSet()
    train_dataset, train_labels = imagePreprocess.reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = imagePreprocess.reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = imagePreprocess.reformat(test_dataset, test_labels)

    graph = tf.Graph()

    with graph.as_default():
        '''
        所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        '''
        #预备
        tf_train_dataset = tf.placeholder(tf.float32,shape = (None,image_size*image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        # 输入
        nums = 32
        initvalue = 0.0
        #隐藏层
        # 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
        #所以卷积神经网络中的卷积核是从训练数据中学习得来的，当然为使得算法正常运行，你需要给定一个初始值
        #x_image =tf.reshape(tf_train_dataset,[-1,28,28,1])
        #卷积核
        #W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, nums]))
        #b_conv1 =tf.Variable(tf.constant(initvalue, shape=[nums]))
        #h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)# 28x28x1 与1个5x5x1滤波器 --> 28x28xnums
        #pooling层 大尺寸的卷积核可以带来更大的感受野，但也意味着更多的参数 用 2 个连续的 3×3 卷积层( stride=1)组成的小网络来代替单个的 5×5卷积层可以保持感受野范围的同时又减少了参数量
        #h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")# 28x28x1 -->14x14x1
        # flat(平坦化)
        flat = tf.reshape(tf_train_dataset, [-1, 28*28 * 1])
        #插入1024个神经元 全链接
        W_fc1 = tf.Variable(tf.truncated_normal([28*28 * 1, 1024]))
        b_fc1 = tf.Variable(tf.constant(initvalue, shape=[1024]))
        h_fc1 = tf.nn.relu(tf.matmul(flat, W_fc1) + b_fc1)
        #第二个全链接
        W_fc2 = tf.Variable(tf.truncated_normal([1024, num_labels]))
        b_fc2 = tf.Variable(tf.constant(initvalue, shape=[10]))
        '''
        conv1 = tf.layers.conv2d(
            inputs=x_image,
            filters=32,
            kernel_size=[5, 5],
            strides=1,
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            bias_constraint=b_conv1
        )
        '''
        #tf.truncated_normal 从截断的正态分布中输出随机值 全链接
        #weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
        #biases = tf.Variable(tf.zeros([num_labels]))


        # Training computation.
        # #logits: 神经网络的输出值
        logits = tf.matmul(h_fc1, W_fc2) + b_fc2
        # Loss function using L2 Regularization
        regularizer = tf.nn.l2_loss(W_fc2)

        loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        learning_rate = 0.001
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(logits)
        test_prediction =  tf.nn.softmax(logits)

        with tf.Session(graph = graph) as session:
            tf.global_variables_initializer().run()
            print("initialized")
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
                #feed_dict参数
                #可选项，给数据流图提供运行时数据。feed_dict的数据结构为python中的字典，其元素为各种键值对。"key"为各种Tensor对象的句柄；"value"很广泛，但必须和“键”的类型相匹配，或能转换为同一类型。
                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                if (step % 100 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    valid_feed_dict = {tf_train_dataset: valid_dataset}
                    valid_predictions = session.run(valid_prediction, feed_dict=valid_feed_dict)
                    print("Validation accuracy: %.1f%%" % accuracy(valid_predictions, valid_labels))
            test_feed_dict = {tf_train_dataset: test_dataset}
            test_predictions = session.run(test_prediction, feed_dict=test_feed_dict)
            print("Test accuracy: %.1f%%" % accuracy(test_predictions, test_labels))


def  nnTrain2(num_steps = 4000):
    """
     stochastic gradient descent training
    :return:
    """
    (train_dataset, train_labels), (valid_dataset, valid_labels), (
    test_dataset, test_labels) = imagePreprocess.getDataSet()
    train_dataset, train_labels = imagePreprocess.reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = imagePreprocess.reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = imagePreprocess.reformat(test_dataset, test_labels)

    graph = tf.Graph()

    with graph.as_default():
        '''
        所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        '''
        #预备
        tf_train_dataset = tf.placeholder(tf.float32,shape = (None,image_size*image_size))/255.
        tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        # 输入
        nums = 32
        initvalue = 1.0
        #隐藏层
        # 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
        #所以卷积神经网络中的卷积核是从训练数据中学习得来的，当然为使得算法正常运行，你需要给定一个初始值
        x_image =tf.reshape(tf_train_dataset,[-1,28,28,1])
        #卷积核
        W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, nums]))
        b_conv1 =tf.Variable(tf.constant(initvalue, shape=[nums]))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)# 28x28x1 与1个5x5x1滤波器 --> 28x28x1
        #pooling层 大尺寸的卷积核可以带来更大的感受野，但也意味着更多的参数 用 2 个连续的 3×3 卷积层( stride=1)组成的小网络来代替单个的 5×5卷积层可以保持感受野范围的同时又减少了参数量
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")# 28x28x1 -->14x14x1
        # flat(平坦化)
        flat = tf.reshape(h_pool1, [-1, 14*14 * nums])
        #插入1024个神经元 全链接
        W_fc1 = tf.Variable(tf.truncated_normal([14*14 * nums, 1024]))
        b_fc1 = tf.Variable(tf.constant(initvalue, shape=[1024]))
        h_fc1 = tf.nn.relu(tf.matmul(flat, W_fc1) + b_fc1)
        #第二个全链接
        W_fc2 = tf.Variable(tf.truncated_normal([1024, num_labels]))
        b_fc2 = tf.Variable(tf.constant(initvalue, shape=[10]))
        '''
        conv1 = tf.layers.conv2d(
            inputs=x_image,
            filters=32,
            kernel_size=[5, 5],
            strides=1,
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            bias_constraint=b_conv1
        )
        '''
        #tf.truncated_normal 从截断的正态分布中输出随机值 全链接
        #weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
        #biases = tf.Variable(tf.zeros([num_labels]))


        # Training computation.
        # #logits: 神经网络的输出值
        logits = tf.matmul(h_fc1, W_fc2) + b_fc2
        loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        learning_rate = 0.001
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(logits)
        test_prediction =  tf.nn.softmax(logits)

        with tf.Session(graph = graph) as session:
            tf.global_variables_initializer().run()
            print("initialized")
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
                #feed_dict参数
                #可选项，给数据流图提供运行时数据。feed_dict的数据结构为python中的字典，其元素为各种键值对。"key"为各种Tensor对象的句柄；"value"很广泛，但必须和“键”的类型相匹配，或能转换为同一类型。
                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                if (step % 100 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    valid_feed_dict = {tf_train_dataset: valid_dataset}
                    valid_predictions = session.run(valid_prediction, feed_dict=valid_feed_dict)
                    print("Validation accuracy: %.1f%%" % accuracy(valid_predictions, valid_labels))
            test_feed_dict = {tf_train_dataset: test_dataset}
            test_predictions = session.run(test_prediction, feed_dict=test_feed_dict)
            print("Test accuracy: %.1f%%" % accuracy(test_predictions, test_labels))

def  nnTrain(num_steps = 8001):
    """
     stochastic gradient descent training
    :return:
    """
    (train_dataset, train_labels), (valid_dataset, valid_labels), (
    test_dataset, test_labels) = imagePreprocess.getDataSet()
    train_dataset, train_labels = imagePreprocess.reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = imagePreprocess.reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = imagePreprocess.reformat(test_dataset, test_labels)

    graph = tf.Graph()

    with graph.as_default():
        '''
        所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        '''
        #预备
        tf_train_dataset = tf.placeholder(tf.float32,shape = (None,image_size*image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        # 输入

        #隐藏层
        # 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
        #所以卷积神经网络中的卷积核是从训练数据中学习得来的，当然为使得算法正常运行，你需要给定一个初始值
        x_image =tf.reshape(tf_train_dataset,[-1,28,28,1])
        #卷积核
        W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 16]))
        b_conv1 =tf.Variable(tf.constant(0.1, shape=[16]))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)# 28x28x1 与1个5x5x1滤波器 --> 28x28x1
        #pooling层 大尺寸的卷积核可以带来更大的感受野，但也意味着更多的参数 用 2 个连续的 3×3 卷积层( stride=1)组成的小网络来代替单个的 5×5卷积层可以保持感受野范围的同时又减少了参数量
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")# 28x28x1 -->14x14x1
        # flat(平坦化)
        flat = tf.reshape(h_pool1, [-1, 14*14 * 16])
        #插入1024个神经元 全链接
        W_fc1 = tf.Variable(tf.truncated_normal([14*14 * 16, 1024]))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
        h_fc1 = tf.nn.relu(tf.matmul(flat, W_fc1) + b_fc1)
        #第二个全链接
        W_fc2 = tf.Variable(tf.truncated_normal([1024, num_labels]))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
        '''
        conv1 = tf.layers.conv2d(
            inputs=x_image,
            filters=32,
            kernel_size=[5, 5],
            strides=1,
            padding='same',
            use_bias=True,
            activation=tf.nn.relu,
            bias_constraint=b_conv1
        )
        '''
        #tf.truncated_normal 从截断的正态分布中输出随机值 全链接
        #weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
        #biases = tf.Variable(tf.zeros([num_labels]))


        # Training computation.
        # #logits: 神经网络的输出值
        logits = tf.matmul(h_fc1, W_fc2) + b_fc2
        loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        learning_rate = 0.001
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(logits)
        test_prediction =  tf.nn.softmax(logits)

        with tf.Session(graph = graph) as session:
            tf.global_variables_initializer().run()
            print("initialized")
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
                #feed_dict参数
                #可选项，给数据流图提供运行时数据。feed_dict的数据结构为python中的字典，其元素为各种键值对。"key"为各种Tensor对象的句柄；"value"很广泛，但必须和“键”的类型相匹配，或能转换为同一类型。
                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                if (step % 100 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    valid_feed_dict = {tf_train_dataset: valid_dataset}
                    valid_predictions = session.run(valid_prediction, feed_dict=valid_feed_dict)
                    print("Validation accuracy: %.1f%%" % accuracy(valid_predictions, valid_labels))
            test_feed_dict = {tf_train_dataset: test_dataset}
            test_predictions = session.run(test_prediction, feed_dict=test_feed_dict)
            print("Test accuracy: %.1f%%" % accuracy(test_predictions, test_labels))


def  batchTrain(num_steps = 60001):
    """
     stochastic gradient descent training
    :return:
    """
    (train_dataset, train_labels), (valid_dataset, valid_labels), (
    test_dataset, test_labels) = imagePreprocess.getDataSet()
    train_dataset, train_labels = imagePreprocess.reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = imagePreprocess.reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = imagePreprocess.reformat(test_dataset, test_labels)

    graph = tf.Graph()

    with graph.as_default():
        '''
        所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        '''
        tf_train_dataset = tf.placeholder(tf.float32,shape = (batch_size,image_size*image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))
        # Training computation.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        learning_rate = 0.01
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

        with tf.Session(graph = graph) as session:
            tf.global_variables_initializer().run()
            print("initialized")
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
                if (step % 100 == 0):
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

def train(num_steps = 801):
    (train_dataset, train_labels), (valid_dataset, valid_labels), (test_dataset, test_labels) = imagePreprocess.getDataSet()
    train_dataset, train_labels = imagePreprocess.reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = imagePreprocess.reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = imagePreprocess.reformat(test_dataset, test_labels)

    graph = tf.Graph()
    with graph.as_default():
        '''
        Input data.
        Load the training, validation and test data into constants that are
        attached to the graph.
        表示实例化了一个类，一个用于 tensorflow 计算和表示用的数据流图，通俗来讲就是：在代码中添加的操作（画中的结点）和数据（画中的线条）都是画在纸上的“画”，而图就是呈现这些画的纸，你可以利用很多线程生成很多张图，但是默认图就只有一张
        '''

        #Creates a constant tensor
        tf_train_dataset = tf.constant(train_dataset[:train_subset,:])
        tf_train_labels =  tf.constant(train_labels[:train_subset])
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
        #truncated_normal 从截断的正态分布中输出随机值
        # Variables.
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random values following a (truncated)
        # normal distribution. The biases get initialized to zero.
        weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        #https://www.jianshu.com/p/648d791b55b0
        logits =tf.matmul(tf_train_dataset,weights)+ biases
        #等价于y=tf.nn.softmax(logits) cross_entropy = -tf.reduce_sum(tf_train_labels*tf.log(y))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        train_prediction = tf.nn.softmax(logits)
        valid_prediction =  tf.nn.softmax(tf.matmul(tf_valid_dataset,weights)+ biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)



    with tf.Session(graph = graph) as session:
        #返回一个用来初始化 计算图中 所有global variable的 op。
        tf.global_variables_initializer().run()
        print('initialized')
        for step in range(num_steps):
             _,l,predictions = session.run([optimizer, loss, train_prediction])
        print('step=',step)
        if (step % 100 == 0):
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(predictions, train_labels[:train_subset, :]))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

def accuracy(predictions,labels):
    #numpy.argmax(a, axis=None, out=None) 返回沿轴axis最大值的索引。
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
# 定义weight的初始化函数便于重复利用，并且设置标准差为0.1
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 定义bias的初始化函数，并加一个小的正值，避免death节点
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 定义二维卷积函数，strides代表模板移动的步长，padding让卷积的输出与输入保持相同
#strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
def traditonalTrain():
    """
    RandomForest accuracy:0.9524
              precision    recall  f1-score   support

           0       0.97      0.96      0.96      1000
           1       0.93      0.95      0.94      1000
           2       0.97      0.95      0.96      1000
           3       0.96      0.96      0.96      1000
           4       0.93      0.94      0.94      1000
           5       0.96      0.96      0.96      1000
           6       0.93      0.95      0.94      1000
           7       0.97      0.95      0.96      1000
           8       0.94      0.94      0.94      1000
           9       0.96      0.96      0.96      1000

   micro avg       0.95      0.95      0.95     10000
   macro avg       0.95      0.95      0.95     10000
    :return:
    """
    (train_dataset, train_labels), (_, _), (test_dataset, test_labels) = imagePreprocess.getDataSet()

    nsamples, nx, ny = train_dataset.shape
    d2_train_dataset = train_dataset.reshape((nsamples, nx * ny))
    nsamples_t, nx_t, ny_t = test_dataset.shape
    d2_test_dataset = test_dataset.reshape((nsamples_t, nx_t * ny_t))
    ''' 
    d3_train_dataset ,train_labels2= imagePreprocess.reformat(train_dataset,train_labels)
    d3_test_dataset, test_labels2 = imagePreprocess.reformat(test_dataset, test_labels)
    '''
    # RandomForest
    rf = classification.RandomForest()
    rf.train(d2_train_dataset, train_labels)
    pred_labels = rf.predict(d2_test_dataset)
    target_names = ['0','1','2','3','4','5','6','7','8','9']
    classification.classificationreport('RandomForest', test_labels, pred_labels, target_names)




if __name__=='__main__':
    #traditonalTrain()
    #train()
    #batchTrain()
    #nnTrain4()
    #nnTrain6()
    #nnTrain7(40000)
    nnTrain8()
    #nnTrain5()
    #nnTrain2_1()