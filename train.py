from data_directory import DataDirectory
from model_data_definition import ModelDataDefinition
from model import Model
import tensorflow as tf

data_definition = ModelDataDefinition( 'data' )
data_dir = DataDirectory( data_definition )

# print("Testing data set")
# for row in data_dir.traverse_sequences( padding_element=PADDING_ELEMENT , sequence_length=SEQUENCE_LENGHT ):
#     print(row)

model = Model()


def input_fn() -> tf.data.Dataset:

    # The dataset
    ds = tf.data.Dataset.from_generator( 
        generator=lambda: data_dir.traverse_sequences( data_definition ), 
        output_types=( 
            { 'column1' : tf.int32 , 'column2' : tf.int32 } , 
            { 'headcol1' : tf.int32 , 'headcol2' : tf.int32 } 
        )
        ,output_shapes=( 
            { 'column1' : (data_definition.sequence_length,) , 'column2' : (data_definition.sequence_length,) } , 
            { 'headcol1' : () , 'headcol2' : () } 
        )
    )

    #ds = ds.repeat(1000)
    ds = ds.batch(64)
    ds = ds.prefetch(64)

    return ds


# print("training...")
# model.estimator.train(input_fn=input_fn)

# print("evaluating...")
# result = model.estimator.evaluate(input_fn=input_fn)
# print("Evaluation: ", result)
