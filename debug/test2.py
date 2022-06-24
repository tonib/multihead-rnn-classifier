"""
Script to make tf operations tests
"""

import tensorflow as tf

x = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
x = tf.reshape(x, (2, 2, 2))
print(x)

y = tf.constant([10, 11, 12, 13])
y = tf.reshape(y, (1, 2, 2))
print(y)

#print(x+y)
print(tf.add(x,y))
