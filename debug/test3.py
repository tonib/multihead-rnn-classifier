import tensorflow as tf

m = tf.keras.metrics.SparseCategoricalAccuracy()
#m.update_state([[2], [1]], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]])
#m.update_state( [1, 1], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]])
m.update_state( 0, [0.1, 0.6, 0.3])

print( m.result().numpy() )

