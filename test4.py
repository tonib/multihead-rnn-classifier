import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices(
               [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensors( tf.reduce_max(x) ) )

print( list(dataset.as_numpy_iterator()) )
for x in dataset:
    print(x)

