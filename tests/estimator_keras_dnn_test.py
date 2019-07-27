import tensorflow as tf
import tensorflow.compat.v2.feature_column as tf_feature_column
import tensorflow.keras.layers

##########################################################################################
# TEST TO TRY TO LEARN TO CREATE A CUSTOM ESTIMATOR FROM A KERAS MODEL 
# AND REMOVE A BUNCH OF DEPRECATED MESSAGES
##########################################################################################

# Uncomment this to show warnings
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Code taken from https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py

# Input data set. A XOR function. 
x0_inputs = [ 0 , 0 , 1 , 1 ]
x1_inputs = [ 0 , 1 , 0 , 1 ]
y_outputs = [ 0 , 1 , 1 , 0 ]


def input_fn(batch_size, repeat_times):
    """An input function for training"""
    # Convert the inputs to a Dataset.

    xor_dataset_inputs = {
        'x1' : tf.constant( x0_inputs ),
        'x2' : tf.constant( x1_inputs ),
    }
    xor_dataset_outputs = { 'y'  : tf.constant( y_outputs ) }

    dataset = tf.data.Dataset.from_tensors( ( xor_dataset_inputs , xor_dataset_outputs ) )

    # TODO: Why thist don't work if .batch() is used?
    # Shuffle, repeat, and DO NOT BATCH the examples. Each ( xor_dataset_inputs , xor_dataset_outputs ) is a batch of shape (4,)
    if repeat_times:
        dataset = dataset.repeat(repeat_times)

    # Return the dataset.
    return dataset

def model_fn(
   features, # Doc says: "This is batch_features from input_fn". THEY ARE THE NET INPUTS
   labels,   # Doc says: "This is batch_labels from input_fn". THEY ARE THE EXPECTED NET OUTPUTS. I guess they are not feeded in 
             # prediction mode. TODO: Check it
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration
    """ Function that defines the model """

    # Outputs must to be specified manually. We have only one, so, here is:
    # (labels will be None in prediction time...)
    if labels:
        labels = labels['y']

    # Let's see what really they are
    # The asterisks are to remark info over the million of deprecation messages
    print( "****************" , features )
    print( "****************" , labels )

    # Use `input_layer` to apply the feature columns.
    # The functional use of DenseFeatures seems unsupported right now... (tf1.14)
    #net = tf.keras.layers.DenseFeatures(params['feature_columns'])
    net = tf.compat.v1.feature_column.input_layer(features, params['feature_columns'])

    # Build and stack the hidden layers, sized according to the 'hidden_units' param.
    for units in params['hidden_units']:
        net = tf.keras.layers.Dense(units, activation='relu')(net)

    # Output layer. Compute logits (1 per class)
    logits = tf.keras.layers.Dense(params['n_classes'], activation=None)(net)

    # Compute predictions.
    # I guess dimension 0 is the batch and axis 1 are the real logits, I guess this computes a vector with the max for each batch
    # So logits = [ [0.1, 0.2] , [0.3 , 0.4] ] will become predicted_classes = [ 0.2 , 0.4 ]
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            # Example says "There is", hehehe
            # "predicted_classes[:, tf.newaxis]" I guess converts the logits for each batch to an array of arrays
            # So [ 1 , 2 ] will become [ [1] , [2] ]. "There is"
            'class_ids': predicted_classes[:, tf.newaxis],
            # Ok
            'probabilities': tf.nn.softmax(logits),
            # Ok
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # If we still here, we are training or evaluating. We will need the loss, calculated against the predicted logits
    # TODO: If there are only 2 classes, this stills valid?
    # TODO: WHY "sparse"? logits should be a dense vector... Maybe it refers to the single activated output for the class. Check it
    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # The operation that will compute the predictions accuracy
    accuracy = tf.compat.v1.metrics.accuracy(labels=labels,
                                predictions=predicted_classes,
                                name='accuracy')

    # The object that we will return as metrics
    metrics = { 'accuracy': accuracy }

    # This is for Tensorboard...
    # tf.metrics.accuracy returns (values, update_ops). So accuracy[1] I guess is update_ops. Documentation says
    # "An operation that increments the total and count variables appropriately and whose value matches accuracy"
    # Sooo... I guess this is an operation that will accumulate accuracy across all batches feeded to the net
    tf.compat.v1.summary.scalar('accuracy', accuracy[1])

    # If we are evaluating the model, we are done:
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # If we still here, we are training
    # Create the optimizer (AdagradOptimizer in this case)
    optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.1)
    # Create the "optimize weigths" operacion, based on the given optimizer
    # TODO: What is "global_step"?
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())

    # Done for training
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


# Feature columns describe how to use the input.
feature_columns = []
c = tf_feature_column.categorical_column_with_identity( key='x1', num_buckets=2)
feature_columns.append( tf_feature_column.indicator_column( c ) )

c = tf_feature_column.categorical_column_with_identity( key='x2', num_buckets=2)
feature_columns.append( tf_feature_column.indicator_column( c ) )

# Create the estimator
estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    params={
        'feature_columns': feature_columns,
        # One hidden layers of 4 nodes
        'hidden_units': [4],
        # The model must choose between 2 classes.
        'n_classes': 2,
    })

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
