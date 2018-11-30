import sys
import os
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
sys.path.append("../src")
from tcn import *
import numpy as np
import tensorflow as tf
test_data = np.random.normal(1, 2, (10, 32, 2, 1)) * np.random.rand(10, 32, 2, 1)
# input=tf.constant(test_data,tf.float32)
input = tf.placeholder(tf.float32, test_data.shape)
out = tcn_layer(input, 2)
out = tcn_block(out, input, use_conv=True)
sess = tf.Session()

flatten = tf.layers.flatten(out)
out_digit = tf.layers.dense(flatten, 1, activation='relu')
label = tf.reshape(tf.abs(tf.reduce_sum(input, axis=[1, 2, 3])), [-1, 1])
loss = tf.losses.mean_squared_error(label, out_digit)
opt = tf.train.AdamOptimizer()
train = opt.minimize(loss)
print('initalize the variables')
init_op = tf.global_variables_initializer()
#init_l = tf.local_variables_initializer()
sess.run(init_op)
#sess.run(init_l)
for i in range(100):
    test_data = np.random.normal(1, 2, (10, 32, 2, 1)) * np.random.rand(10, 32, 2, 1)
    print(sess.run([train, loss], feed_dict={input: test_data}))