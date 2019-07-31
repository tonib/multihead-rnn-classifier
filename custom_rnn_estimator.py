
from model_data_definition import ModelDataDefinition
from column_info import ColumnInfo
import tensorflow as tf
import tensorflow.compat.v1.feature_column as tf_feature_column
import tensorflow.feature_column as feature_column

# TODO: Implement context columns

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
            model_fn=lambda: self._model_fn,
            params={
                'feature_columns': self._get_sequence_columns(),
            },
            model_dir=model_dir
        )
        
    def _get_sequence_columns(self) -> list:
        """ Returns the model input features list definition (sequence) """
        sequence_columns = []
        for col_name in self.data_definition.sequence_columns:
            def_column = self.data_definition.column_definitions[ col_name ]
            column = tf_feature_column.sequence_categorical_column_with_identity( col_name , len(def_column.labels) )
            indicator_column = feature_column.indicator_column( column )
            sequence_columns.append( indicator_column )
        return sequence_columns

    def _model_fn(
        self,
        features, # Doc says: "This is batch_features from input_fn". THEY ARE THE NET INPUTS, defined by the input_fn
        labels,   # Doc says: "This is batch_labels from input_fn". THEY ARE THE EXPECTED NET OUTPUTS, defined by the input_fn. 
                    # I guess they are not feeded in prediction mode. TODO: Check it
        mode,     # An instance of tf.estimator.ModeKeys
        params):  # Additional configuration
        """ Function that defines the model """
        
        #print("*********** labels shape:", labels.shape )

        # The input layer
        sequence_input_layer = tf.keras.experimental.SequenceFeatures( params['feature_columns'] )
        # TODO: Second returned value is "sequence_length". What is used for?
        sequence_input, _ = sequence_input_layer(features)


        # Define a GRU layer
        rnn_layer = tf.keras.layers.GRU( self.data_definition.n_network_elements )(sequence_input)

        # Create a classifier for each output to predict
        # TODO: Get the number of classifiers from the labels shape
        i_output_labels = None
        classifiers = []
        for i in range(len(self.data_definition.output_columns)):
            output_column = self.data_definition.column_definitions[ self.data_definition.output_columns[i] ]

            # Output labels are only defined if we are training / evaluating:
            if mode != tf.estimator.ModeKeys.PREDICT:
                # Explanation for "labels[:, i]": First dimension is the batch, keep it as is. Second is the output for the i-th output
                i_output_labels = labels[:, i]
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
        # TODO: Compute mean total loss ???
        # TODO: This creates n ops. Seach a single "sum" op for all entries
        total_loss = classifiers[0].loss
        for i in range(1, len(classifiers)):
            total_loss = total_loss + classifier.loss

        # If we are evaluating the model, we are done:
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=total_loss, eval_metric_ops=metrics)

        # If we still here, we are training
        # TODO: Allow to configure what optimizer run
        # Create the optimizer (AdagradOptimizer in this case)
        optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.1)

        # Create the "optimize weigths" operacion, based on the given optimizer
        # TODO: What is "global_step"?
        train_op = optimizer.minimize(total_loss, global_step=tf.compat.v1.train.get_global_step())

        # Done for training
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)