
from model_data_definition import ModelDataDefinition
from column_info import ColumnInfo
import tensorflow as tf
import tensorflow.compat.v1.feature_column as tf_feature_column
import tensorflow.feature_column as feature_column

# TODO: Implement context columns (see rnn.py > _concatenate_context_input)

class _ClassifierHead:
    """ Classification head for the model """

    def __init__(self, rnn_layer, output_column:ColumnInfo, mode:str, output_labels):
        """ Constructor:
            rnn_layer: The model RNN layer
            output_column: The model output column for this classifier
            mode: The estimator work mode (a tf.estimator.ModeKeys.* value)
            output_labels: Tensor with expected output labels. Only for TRAIN or EVAL models
        """

        # The model output column definition
        self.output_column = output_column

        # Output layer. Compute logits (1 per class)
        self.logits = tf.keras.layers.Dense( len(output_column.labels), activation=None)(rnn_layer)

        # Compute predictions.
        self.predicted_classes = tf.argmax( self.logits , 1 )

        if mode != tf.estimator.ModeKeys.PREDICT:
            # mode == TRAIN or EVAL

            # Compute loss
            self.loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=output_labels, logits=self.logits)

            # The operation that will compute the predictions accuracy for metrics for Tensorboard
            self.accuracy_metric = tf.compat.v1.metrics.accuracy(labels=output_labels,
                predictions=self.predicted_classes,
                name='acc_op_' + output_column.name)

            # This is for Tensorboard...
            # tf.metrics.accuracy returns (values, update_ops). So accuracy[1] I guess is update_ops. Documentation says
            # "An operation that increments the total and count variables appropriately and whose value matches accuracy"
            # Sooo... I guess this is an operation that will accumulate accuracy across all batches feeded to the net
            # Documentation say this is for training
            tf.compat.v1.summary.scalar('accuracy/' + output_column.name, self.accuracy_metric[1])

            # Metric operation for each output loss
            # https://github.com/tensorflow/tensorflow/issues/14041
            self.loss_metric = tf.metrics.mean(values=self.loss, name='loss_metric_op_' + output_column.name)

            # Tensorboard metric for each output loss (mean loss)
            tf.compat.v1.summary.scalar('loss/' + output_column.name, self.loss_metric[1])

    def add_prediction(self, predictions:dict):
        """ Add this classifier prediction to the model predictions """
        # The predicted class
        predictions[ self.output_column.name + '/classes' ] = self.predicted_classes
        # Each class probability
        predictions[ self.output_column.name + '/probabilities' ] = tf.nn.softmax( self.logits )
        # Output logits
        predictions[ self.output_column.name + '/logits' ] = self.logits

    def add_metrics(self, metrics:dict):
        """ Add this classifier metrics to the model metrics """
        metrics[ 'accuracy/' + self.output_column.name ] = self.accuracy_metric
        metrics[ 'loss/' + self.output_column.name ] = self.loss_metric


class CustomRnnEstimator:
    """ Custom RNN estimator """

    def __init__(self, data_definition: ModelDataDefinition):

        self.data_definition = data_definition

        model_dir = self.data_definition.get_current_model_dir_path()
        print("Current train model dir:" , model_dir)

        # Create estimator
        self.estimator = tf.estimator.Estimator(
            model_fn=CustomRnnEstimator._model_fn,
            params={
                'sequence_columns': self._get_sequence_columns(),
                'context_columns' : self._get_context_columns(),
                'data_definition': self.data_definition
            },
            model_dir=model_dir
        )
        
    def _get_sequence_columns(self) -> list:
        """ Returns the model input features list definition for sequence columns """
        sequence_columns = []
        for col_name in self.data_definition.sequence_columns:
            def_column = self.data_definition.column_definitions[ col_name ]
            column = tf_feature_column.sequence_categorical_column_with_identity( col_name , len(def_column.labels) )
            indicator_column = feature_column.indicator_column( column )
            sequence_columns.append( indicator_column )
        return sequence_columns

    def _get_context_columns(self) -> list:
        """ Returns the model input features list definition for context columns """
        context_columns = []
        for col_name in self.data_definition.context_columns:
            def_column = self.data_definition.column_definitions[ col_name ]
            column = tf_feature_column.categorical_column_with_identity( col_name , len(def_column.labels) )
            indicator_column = feature_column.indicator_column( column )
            context_columns.append( indicator_column )
        return context_columns

    ##############################################################################################
    # This function has been copied (and modified) from Tensorflow source code
    # tensorflow_estimator/contrib/estimator/python/estimator/rnn.py > _concatenate_context_input
    # Tensorflow is under Apache license (http://www.apache.org/licenses/LICENSE-2.0)
    ##############################################################################################
    @staticmethod
    def _concatenate_context_input(sequence_input, context_input):
        """Replicates `context_input` across all timesteps of `sequence_input`.

        Expands dimension 1 of `context_input` then tiles it `sequence_length` times.
        This value is appended to `sequence_input` on dimension 2 and the result is
        returned.

        Args:
            sequence_input: A `Tensor` of dtype `float32` and shape `[batch_size,
            padded_length, d0]`.
            context_input: A `Tensor` of dtype `float32` and shape `[batch_size, d1]`.

        Returns:
            A `Tensor` of dtype `float32` and shape `[batch_size, padded_length,
            d0 + d1]`.

        Raises:
            ValueError: If `sequence_input` does not have rank 3 or `context_input` does
            not have rank 2.
        """
        seq_rank_check = tf.debugging.assert_rank(
            sequence_input,
            3,
            message='sequence_input must have rank 3',
            data=[tf.shape(sequence_input)])
        seq_type_check = tf.debugging.assert_type(
            sequence_input,
            tf.float32,
            message='sequence_input must have dtype float32; got {}.'.format(
                sequence_input.dtype))
        ctx_rank_check = tf.debugging.assert_rank(
            context_input,
            2,
            message='context_input must have rank 2',
            data=[tf.shape(context_input)])
        ctx_type_check = tf.debugging.assert_type(
            context_input,
            tf.float32,
            message='context_input must have dtype float32; got {}.'.format(
                context_input.dtype))
        with tf.control_dependencies(
            [seq_rank_check, seq_type_check, ctx_rank_check, ctx_type_check]):
            padded_length = tf.shape(sequence_input)[1]
            tiled_context_input = tf.tile(
                tf.expand_dims(context_input, 1),
                tf.concat([[1], [padded_length], [1]], 0))
        return tf.concat([sequence_input, tiled_context_input], 2)
        
    @staticmethod
    def _model_fn(
        features, # Doc says: "This is batch_features from input_fn". THEY ARE THE NET INPUTS, defined by the input_fn
        labels,   # Doc says: "This is batch_labels from input_fn". THEY ARE THE EXPECTED NET OUTPUTS, defined by the input_fn. 
                    # I guess they are not feeded in prediction mode. TODO: Check it
        mode,     # An instance of tf.estimator.ModeKeys
        params):  # Additional configuration
        """ Function that defines the model """
        
        data_definition = params['data_definition']

        # print("*********** features:", features )
        # print("*********** labels:", labels )

        # The input layer for sequence inputs
        sequence_input_layer = tf.keras.experimental.SequenceFeatures( params['sequence_columns'] )
        # TODO: Second returned value is "sequence_length". it should match data_definition.sequence_length
        sequence_input_tensor, _ = sequence_input_layer(features)

        #print("*********** sequence_input_tensor:", sequence_input_tensor )

        if len(data_definition.context_columns) > 0:
            # Append the context columns to each sequence timestep
            context_input_layer = tf.keras.layers.DenseFeatures( params['context_columns'] )
            context_input_tensor = context_input_layer( features )
            #print("*********** context_input_tensor:", context_input_tensor )
            sequence_input_tensor = CustomRnnEstimator._concatenate_context_input(sequence_input_tensor, context_input_tensor)
            #print("*********** Final sequence_input_tensor:", sequence_input_tensor )

        # Define a GRU layer
        rnn_layer = tf.keras.layers.GRU( data_definition.n_network_elements )(sequence_input_tensor)

        # Create a classifier for each output to predict
        i_output_labels = None
        classifiers = []
        for i in range(len(data_definition.output_columns)):
            output_column = data_definition.column_definitions[ data_definition.output_columns[i] ]

            # Output labels are only defined if we are training / evaluating:
            if mode != tf.estimator.ModeKeys.PREDICT:
                # Get expected outputs for this classifier
                i_output_labels = labels[ output_column.name ]
            classifiers.append( _ClassifierHead(rnn_layer , output_column, mode, i_output_labels) )

        if mode == tf.estimator.ModeKeys.PREDICT:
            # Accumulate each classifier prediction
            predictions = {}
            for classifier in classifiers:
                classifier.add_prediction(predictions)
            # Done
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)


        # The object that we will return as metrics
        # Documentation says this is only for training
        metrics = {}
        for classifier in classifiers:
            classifier.add_metrics(metrics)

        # Compute total loss

        # Version 1:
        # TODO: Compute mean total loss ???
        # TODO: This creates n ops. Seach a single "sum" op for all entries
        # total_loss = classifiers[0].loss
        # for i in range(1, len(classifiers)):
        #     total_loss = total_loss + classifier.loss
        # Version 2
        classifier_losses = [ c.loss for c in classifiers ]
        total_loss = tf.reduce_mean( classifier_losses )

        # If we are evaluating the model, we are done:
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=total_loss, eval_metric_ops=metrics)

        # If we still here, we are training
        # TODO: Allow to configure what optimizer run
        # Create the optimizer (AdagradOptimizer in this case)
        #optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.1)
        optimizer = tf.train.AdamOptimizer()

        # Create the "optimize weigths" operacion, based on the given optimizer
        # TODO: What is "global_step"?
        train_op = optimizer.minimize(total_loss, global_step=tf.compat.v1.train.get_global_step())

        # Done for training
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)