from data_directory import DataDirectory
from model import Model
import tensorflow as tf

data_dir = DataDirectory('data')

print("Testing data set")
for row in data_dir.traverse_sequences( padding_element=[0,0] , sequence_length=3 ):
    print(row)

model = Model()


# def input_fn() -> tf.data.Dataset:

#     # The dataset
#     ds = tf.data.Dataset.from_generator( generator=data_dir.generator, 
#         output_types=( { 'character' : tf.string } , tf.string ),
#         output_shapes=( { 'character' : (Model.SEQUENCE_LENGHT,) } , () )
#     )

#     ds = ds.batch(64)
#     ds = ds.prefetch(1)

#     return ds
