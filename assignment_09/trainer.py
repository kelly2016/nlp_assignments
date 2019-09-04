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

def  batchTrain(num_steps = 3001):
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
        learning_rate = 0.5
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
    train()
    #batchTrain()
