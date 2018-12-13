import tensorflow as tf

filter=tf.constant([1,2],dtype=tf.float32)
filter=tf.reshape(filter,shape=[2,1,1])
data=tf.constant([[1,2,3,4],[5,6,7,8]],dtype=tf.float32)
data=tf.reshape(data,shape=[2,4,1])
sess=tf.Session()
b=sess.run(data)
print(b)
c=tf.nn.convolution(data,filter,padding='SAME')
print(sess.run(c))
pass