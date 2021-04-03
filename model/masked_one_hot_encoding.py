import tensorflow as tf

class MaskedOneHotEncoding(tf.keras.layers.Layer):
    """ Compute masked one hot encoding from an integer input. Mask value is 0. Input mask is ignored. """
    def __init__(self, input_n_labels: int, **kwargs):
        """
            Arguments: 
                input_n_labels: Number of labels expected in input, including the padding value (zero). Ex. {0, 1, 2} -> n.labels = 3
        """
        super().__init__(**kwargs)
        self.input_n_labels = input_n_labels

    def call(self, inputs):
        # -1 is to optimize the output size. As zero is reserved for padding, only 1+ values will be used as real inputs
        tf.debugging.assert_greater_equal(inputs, 0, "MaskedOneHotEncoding: inputs[i] < 0")
        tf.debugging.assert_less(inputs, self.input_n_labels, "MaskedOneHotEncoding: inputs[i] >= self.input_n_labels")
        return tf.one_hot(inputs - 1, self.input_n_labels - 1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"input_n_labels": self.input_n_labels})
        return config

    def compute_mask(self, inputs, mask=None):
        return tf.cast( inputs , tf.bool )