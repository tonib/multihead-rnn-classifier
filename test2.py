import tensorflow as tf

input_a = tf.keras.Input(name='a', dtype=tf.int32, shape=())
input_b = tf.keras.Input(name='b', dtype=tf.int32, shape=())

outputs = { 'e': input_a , 'd': input_b}

model = tf.keras.Model(inputs=[input_b, input_a], outputs=outputs)

i = { 'a': tf.constant(1) , 'b': tf.constant(2) }
print(model(i))

i = { 'b': tf.constant(2) , 'a': tf.constant(1) }
print(model(i))
