#卷积处理模式
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("D:/memory_download/文档/大一(下)/PPCA/Machine_Learning",one_hot=True)

#配置神经网络的参数
INPUT_NODE=784
OUTPUT_NODE=10

IMAGE_SIZE=28
NUM_CHANNELS=1
NUM_LABELS=10

BATCH_SIZE=32

#第一层卷积层的深度和尺寸
CONV1_DEEP=32
CONV1_SIZE=5
#第二层卷积层的深度和尺寸
CONV2_DEEP=64
CONV2_SIZE=14
#全连接节点的个数
FC_SIZE=512

STEPS=30000
#构建计算图模型

x=tf.placeholder(dtype=tf.float32 ,shape=[None,IMAGE_SIZE,\
                  IMAGE_SIZE,NUM_CHANNELS])
y_=tf.placeholder(dtype=tf.float32,shape=[None,NUM_LABELS])
#第一层卷积层
w1=tf.get_variable(name='w1',shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
b1=tf.get_variable(name='b1',shape=[CONV1_DEEP],initializer=tf.constant_initializer(0.0))

a1=tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='SAME')
conv1=tf.nn.relu(tf.nn.bias_add(a1,b1))
#第一层池化层
pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#第二层卷积层
w2=tf.get_variable(name='w2',shape=[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
b2=tf.get_variable(name='b2',shape=[CONV2_DEEP],initializer=tf.constant_initializer(0.0))

a2=tf.nn.conv2d(pool1,w2,strides=[1,1,1,1],padding='SAME')
conv2=tf.nn.relu(tf.nn.bias_add(a2,b2))

#第二层池化层
pool2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#第一层全连接层
pool_shape=pool2.get_shape().as_list()
nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
pool2=tf.reshape(pool2,(-1,7*7*64))

w3=tf.get_variable(name='w3',shape=[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
b3=tf.get_variable(name='b3',shape=[FC_SIZE],initializer=tf.constant_initializer(0.1))

a3=tf.nn.relu(tf.matmul(pool2,w3)+b3)

#第二层全连接层

w4=tf.get_variable(name='w4',shape=[FC_SIZE,NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))
b4=tf.get_variable(name='b4',shape=[NUM_LABELS],initializer=tf.constant_initializer(0.1))

y=tf.matmul(a3,w4)+b4


loss_func=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1)))
Optimizer=tf.train.AdamOptimizer(0.01).minimize(loss_func)

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#建立会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    validate_feed={x:mnist.validation.images.reshape([-1,28,28,1]),y_:mnist.validation.labels}
    test_feed={x:mnist.test.images.reshape([-1,28,28,1]),y_:mnist.test.labels}

    for i in range(STEPS):
        if i%50==0:
            validate_acc=sess.run(accuracy,feed_dict=validate_feed)
            print("STEPS %d ,Accuracy %g"%(i,validate_acc))

        x_s,y_value=mnist.train.next_batch(BATCH_SIZE)
        x_value=np.reshape(x_s,(BATCH_SIZE,IMAGE_SIZE,\
                  IMAGE_SIZE,NUM_CHANNELS))
        sess.run(Optimizer,feed_dict={x:x_value,y_:y_value})

    test_acc=sess.run(accuracy,feed_dict=test_feed)
    print("STEPS %d ,TEST_Accuracy %g"%(i,test_acc))


        
        





