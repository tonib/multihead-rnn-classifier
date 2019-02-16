from tensorflow.contrib.estimator import RNNEstimator
from tensorflow.contrib.estimator import multi_head
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses

class Model:

    def __init__(self):
        self._create_estimator()

    def _create_estimator(self):
        
        # Input columns
        column1 = tf.feature_column.categorical_column_with_identity('column1', 4)
        column2 = tf.feature_column.categorical_column_with_identity('column2', 4)

        # Output heads
        head_classify_col1 = head_lib._multi_class_head_with_softmax_cross_entropy_loss(4, name='headcol1')
        head_classify_col2 = head_lib._multi_class_head_with_softmax_cross_entropy_loss(4, name='headcol2')
        head = multi_head( [ head_classify_col1 , head_classify_col2 ] )

        # The estimator
        self.estimator = RNNEstimator(
            head = head,
            sequence_feature_columns = [ column1 , column2 ],
            num_units=[64, 64], 
            cell_type='gru', 
            optimizer=tf.train.AdamOptimizer,
            model_dir='model'
        )