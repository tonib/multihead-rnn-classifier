import tensorflow as tf
import tensorflow.feature_column as feature_column
from tensorflow.contrib.estimator import RNNClassifier
import tensorflow.contrib.feature_column as contrib_feature_column
import tensorflow.compat.v1.feature_column as tf_feature_column
from typing import List

# Test to make a char RNN predictor

# Simple sample text to learn: repetitions of "0123456789"
text=""
for _ in range(10):
    text += "0123456789"

# Sequence length that will be feeded to the network
SEQUENCE_LENGHT = 7

# The real vocabulary:
vocabulary = list( set(text) )

# As far I know, Tensorflow RNN estimators don't support variable length sequences, so I'll use the char "_" as padding
# Maybe it's supported, right now I dont' know how
vocabulary.append('_')

# Important! Otherwise, with different executions, the list can be in different orders (really)
vocabulary.sort()

def pad_sequence( text_sequence : str ) -> List[str]:
    """
    Pads the text_sequence lenght to a minimum length of SEQUENCE_LENGHT, and returns it as a List of characters

    As far I know, Tensorflow RNN estimators don't support variable length sequences, so I'll use the char "_" as padding.
    If text_sequence has a len(text_sequence) < SEQUENCE_LENGHT, the text will be padded as "_...text_sequence", up to SEQUENCE_LENGHT characters.
    
    Args:
        text_sequence: The text to pad 

    Retunrs:
        The "text_sequence", padded with "_" characters, as a characters List
    """

    l = len(text_sequence)
    if l < SEQUENCE_LENGHT:
        # Pad string: "__...__text_sequence"
        text_sequence = text_sequence.rjust( SEQUENCE_LENGHT , '_')
    
    # Return the text as a characters list
    return list(text_sequence)

# Train input and outputs
# An input sequece is a list of characters ( ex. [ '0' , '1' , ... ] )
# inputs['character'] will store a list of sequences
inputs = { 'character': [] }
# Outputs is a list of int's, to reduce complexity
outputs =  []

def prepare_train_sequences_length(seq_length : int):
    """
    Prepare sequences of a given length

    Args:
        lenght: Length of sequences to prepare
    """
    for i in range(0, len(text) - seq_length):
        sequence = text[i : i + seq_length]
        sequence_output = text[i + seq_length : i + seq_length+1]
        inputs['character'].append( pad_sequence(sequence) )
        outputs.append( int(sequence_output) )


# Prepare sequences of a range of lengths from 1 to 7 characters
for sequence_length in range(1, 8):
    prepare_train_sequences_length(sequence_length)

print("N. train sequences: ", len(inputs['character']))

def input_fn(n_repetitions = 1) -> tf.data.Dataset:
    """
    Returns the text as char array

    Args:
        n_repetitions: Number of times to repeat the inputs
    """

    # The dataset
    ds = tf.data.Dataset.from_tensor_slices( (inputs,outputs) )

    # Repeat inputs n times
    if n_repetitions > 1:
        ds = ds.repeat(n_repetitions)

    ds = ds.shuffle( 1000 )
    ds = ds.batch(4)
    
    return ds

def model_fn(
   features, # Doc says: "This is batch_features from input_fn". THEY ARE THE NET INPUTS
   labels,   # Doc says: "This is batch_labels from input_fn". THEY ARE THE EXPECTED NET OUTPUTS. I guess they are not feeded in 
             # prediction mode. TODO: Check it
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration
    """ Function that defines the model """
    
    
    # The input layer
    #input = tf.compat.v1.feature_column.input_layer(features, params['feature_columns'])
    sequence_input_layer = tf.keras.experimental.SequenceFeatures( params['feature_columns'] )
    sequence_input, sequence_length = sequence_input_layer(features)


    # Define a GRU layer
    rnn_layer = tf.keras.layers.GRU( params['hidden_units'] )(sequence_input)

    # Output layer. Compute logits (1 per class)
    logits = tf.keras.layers.Dense(params['n_classes'], activation=None)(rnn_layer)

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

# The single character sequence 
# character_column = contrib_feature_column.sequence_categorical_column_with_vocabulary_list( 'character' , vocabulary_list = vocabulary )
# indicator_column = feature_column.indicator_column( character_column )
# feature_columns = [ indicator_column ]

character_column = tf_feature_column.sequence_categorical_column_with_vocabulary_list( 'character', 
    vocabulary_list = vocabulary)
indicator_column = feature_column.indicator_column( character_column )
feature_columns = [ indicator_column ]

# The estimator
# estimator = RNNClassifier(
#     sequence_feature_columns=feature_columns,
#     num_units=[7], cell_type='lstm', model_dir='./model', 
#     n_classes=10, label_vocabulary=vocabulary)

# Create the estimator
estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    params={
        'feature_columns': feature_columns,
        # One hidden layers of 4 nodes
        'hidden_units': 7,
        # The model must choose between 2 classes.
        'n_classes': 10,
    })

for _ in range(10):
    estimator.train(input_fn=input_fn)

def predict( text : str ):
    """
    Predicts and print the next character after a given sequence

    Args:
        text: The input sequence text
    """

    result = estimator.predict( input_fn=lambda:tf.data.Dataset.from_tensors( ({ 'character' : [ pad_sequence(text) ] }) ) )
    print("-----")
    print("Input sequence: " , text )
    for r in result:
        #print("Prediction: " , r)
        print('Class output: ', r['class_ids'])
    print("-----")

# Some predictions in the train set (memory)
predict( '0123456' )
predict( '1234567' )
predict( '2345678' )
predict( '3456789' )
predict( '4567890' )
predict( '3' )
predict( '5678' )

# Some predictions out the train set (generalization)
predict( '2345678901' )
predict( '6789012345678' )
predict( '9012345678901234567890123456789012' )
predict( '0123456789012345678901234567890123456789' )
