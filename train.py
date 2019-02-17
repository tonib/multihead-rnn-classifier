from data_directory import DataDirectory
from model_data_definition import ModelDataDefinition
from model import Model
import tensorflow as tf

data_definition = ModelDataDefinition( 'data' )
data_dir = DataDirectory( data_definition )

# print("Testing data set")
# for row in data_dir.traverse_sequences( data_definition ):
#     print(row)

model = Model( data_definition )


def input_fn() -> tf.data.Dataset:

    # The dataset
    ds = tf.data.Dataset.from_generator( 
        generator=lambda: data_dir.traverse_sequences( data_definition ), 
        output_types = data_definition.model_input_output_types(),
        output_shapes = data_definition.model_input_output_shapes()
    )

    #ds = ds.repeat(1000)
    ds = ds.batch(64)
    ds = ds.prefetch(64)

    return ds

while True:
    print("training...")
    model.estimator.train(input_fn=input_fn)

    print("evaluating...")
    result = model.estimator.evaluate(input_fn=input_fn)
    print("Evaluation: ", result)
