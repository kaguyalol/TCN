import sys
import os
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../src")
import tensorflow as tf
import numpy as np
from tcn import *
minstdata=tf.keras.datasets.mnist.load_data('mnist.npz')

inputplhd=tf.placeholder(tf.float32,shape=(None,28,28))
labelplhd=tf.placeholder(tf.int32,shape=(None))
sess = tf.Session()
istcn='tcn'
if istcn=='tcn':
    input=tf.reshape(inputplhd,[-1,784,1])
    out = tcn_layer(input, 3,filter_size=10)
    out = tcn_block(out, input, use_conv=True)

    flatten = tf.layers.flatten(out)

elif istcn=='dense':
    input = tf.reshape(inputplhd, [-1, 784])
    flatten = tf.layers.dense(input,36)
else:

    flatten=tf.layers.conv2d(input,16,5,padding='valid')
    flatten = tf.layers.flatten(flatten)

out_digit = tf.layers.dense(flatten, 10, activation='sigmoid')
label = tf.reshape(labelplhd, [-1, 1])
loss = tf.losses.sparse_softmax_cross_entropy(label, out_digit)
opt = tf.train.AdamOptimizer()
train = opt.minimize(loss)
sparse_output=tf.argmax(out_digit,axis=1)
acc=tf.metrics.accuracy(label,sparse_output)
print('initalize the variables')
init_op = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
sess.run(init_op)
sess.run(init_l)
bs=32
lendata=len(minstdata[0][0])
for j in range(1,100000):
    i=(j% int(lendata/bs))

    data=minstdata[0][0][bs*i:bs*i+bs]
    tlabel=minstdata[0][1][bs*i:bs*i+bs]
    temp = sess.run([train, loss, acc, sparse_output, label], feed_dict={inputplhd: data, labelplhd: tlabel})
    if not i%100:
        print(temp[1:3])
#test:


pass
