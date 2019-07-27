import tensorflow as tf
import tensorflow.compat.v1.feature_column as tf_feature_column
import tensorflow.keras.layers

##########################################################################################
# NOT WORKING !!!
# TEST TO TRY TO LEARN TO CREATE A CUSTOM ESTIMATOR FROM A KERAS MODEL
##########################################################################################

# Code taken from https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py

# Input data set. A XOR function. 
x0_inputs = [ 0 , 0 , 1 , 1 ]
x1_inputs = [ 0 , 1 , 0 , 1 ]
y_outputs = [ 0 , 1 , 1 , 0 ]


def input_fn(batch_size, repeat_times):
    """An input function for training"""
    # Convert the inputs to a Dataset.

    xor_dataset_inputs = {
        'input_1' : tf.constant( x0_inputs ),
        'input_2' : tf.constant( x1_inputs ),
    }
    # TODO: Originally key name was 'y', but they are ignored... Use default name 'output_1'
    xor_dataset_outputs = { 'output_1'  : tf.constant( y_outputs ) }

    dataset = tf.data.Dataset.from_tensors( ( xor_dataset_inputs , xor_dataset_outputs ) )

    # TODO: Why thist don't work if .batch() is used?
    # Shuffle, repeat, and DO NOT BATCH the examples. Each ( xor_dataset_inputs , xor_dataset_outputs ) is a batch of shape (4,)
    if repeat_times:
        dataset = dataset.repeat(repeat_times)

    # Return the dataset.
    return dataset


############## INPUTS

# Feature columns describe how to use the input.
# TODO: Originally key names where 'x1' and 'x2', but they are ignored... Use default names input_1, input_2
feature_columns = []
c = tf_feature_column.categorical_column_with_identity( key='input_1', num_buckets=2)
feature_columns.append( tf_feature_column.indicator_column( c ) )
c = tf_feature_column.categorical_column_with_identity( key='input_2', num_buckets=2)
feature_columns.append( tf_feature_column.indicator_column( c ) )

############## MODEL

keras_model = tf.keras.Sequential()
keras_model.add( tf.keras.layers.DenseFeatures( feature_columns ) )
keras_model.add(tf.keras.layers.Dense(4, activation='relu'))
keras_model.add(tf.keras.layers.Dense(4, activation='relu'))
# Add a softmax layer with 2 output units:
keras_model.add(tf.keras.layers.Dense(2, activation='softmax'))

keras_model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy', # binary_crossentropy
              metrics=['accuracy'])
#keras_model.summary()

############## MODEL TO ESTIMATOR
estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_model, model_dir="keras_estimator_model")

# print(keras_model.input_names)
# print(keras_model.output_names)

############## DO THE WORK

# Configuration:
batch_size = 2
n_repeats = 100

# Train the model
estimator.train( input_fn=lambda:input_fn(batch_size, n_repeats) )

# Evaluate the model.
eval_result = estimator.evaluate( input_fn=lambda:input_fn(batch_size, n_repeats) )
print('\nEval result:', eval_result)

# Do predictions
predictions = estimator.predict( input_fn=lambda:input_fn(1, 0) )
print("Predictions:")
for prediction in predictions:
    print(prediction)
