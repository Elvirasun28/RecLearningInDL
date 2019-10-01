import tensorflow as tf

## define the variables
with tf.name_scope('input'):
    input1 = tf.constant(3.0, name='A')
    input2 = tf.constant(4.0, name='B')
    input3 = tf.constant(5.0, name='C')
with tf.name_scope('op'):
    add = tf.add(input2, input3)
    mul = tf.multiply(input1, add)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('tensorboard_intro/logs/',sess.graph)
    result = sess.run([mul,add])
    print(result)