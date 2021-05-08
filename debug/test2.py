import tensorflow as tf

t = tf.constant(3)

mask = 1 - tf.linalg.band_part(tf.ones((t, t)), -1, 0)
print(mask)

a = tf.range(t)
mask = a[:,None]<a[None,:]# true in upper_triangular
mask = tf.cast(mask, tf.float32)
#mask = tf.logical_and(mask, mask1)
print(mask)