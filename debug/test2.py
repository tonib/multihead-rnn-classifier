import tensorflow as tf

x = tf.constant( [ 0 , 1 , 2, -1, -1] )
mask = tf.not_equal(x, -1)
x = tf.boolean_mask(x, mask)
print(x)
