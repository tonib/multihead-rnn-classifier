import tensorflow as tf


x = tf.constant( [ 1 , 2 , 3 , 4 ] )
trainable = tf.constant( [ 0 , 0 , 0 , 0 ] )

non_trainable_indices = trainable
non_trainable_indices = tf.equal( non_trainable_indices , 0 )
non_trainable_indices = tf.where( non_trainable_indices ) # Ex. [ [0] , [2] ]

number_non_trainable = tf.shape(non_trainable_indices)[0]
non_trainable_values = tf.repeat( -1 , number_non_trainable )

x = tf.tensor_scatter_nd_update(x, non_trainable_indices, non_trainable_values)

print( x )
