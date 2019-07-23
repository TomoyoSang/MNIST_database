import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("D:/memory_download/文档/大一(下)/PPCA/Machine_Learning",one_hot=True)

#常量定义
INPUT_NODE=784
OUTPUT_NODE=10
HIDEN_NODE=500
BATCH_SIZE=100
STEPS=50000

#建立计算图
x=tf.placeholder(dtype=tf.float32,shape=(None,784))
y_=tf.placeholder(dtype=tf.float32,shape=(None,10))

w1=tf.Variable(tf.random_normal(shape=[784,100],dtype=tf.float32))
b1=tf.Variable(tf.random_normal(shape=[1,100],dtype=tf.float32))
a=tf.nn.relu(tf.matmul(x,w1)+b1)

w2=tf.Variable(tf.random_normal(shape=[100,10],dtype=tf.float32))
b2=tf.Variable(tf.random_normal(shape=[1,10],dtype=tf.float32))
y=tf.matmul(a,w2)+b2


loss_func=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1)))
Optimizer=tf.train.AdamOptimizer().minimize(loss_func)

    
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#建立会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}

    test_feed={x:mnist.test.images,y_:mnist.test.labels}

    for i in range(STEPS):
        if i %1000==0:
            validate_acc=sess.run(accuracy,feed_dict=validate_feed)
            print("STEPS:",i," validate_accuracy:",validate_acc)
        x_value,y_value=mnist.train.next_batch(BATCH_SIZE)
        sess.run(Optimizer,feed_dict={x:x_value,y_:y_value})

    test_acc=sess.run(accuracy,feed_dict=test_feed)
    print("STEPS: ",i," test_accuracy:",test_acc)



