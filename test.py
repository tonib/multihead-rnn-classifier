import tensorflow as tf
from tensorflow.data import Dataset

sequence_length = 55
sequence_feature_names = [ "sequence" ]
context_feature_names = [ "sequence" ]
output_feature_names = [ "output" ]

data = { 'sequence': list(range(1,11)), 'context': list(range(21, 31)) , 'output': list(range(31, 41)) }
ds = Dataset.from_tensor_slices(data)
ds = ds.window(sequence_length + 1, shift=1, drop_remainder=False)

def flat_map_window(window_elements_dict):
    for key in window_elements_dict:
        window_elements_dict[key] = tf.data.experimental.get_single_element( window_elements_dict[key].batch(sequence_length + 1) )
    return window_elements_dict

ds = ds.map(flat_map_window)
print("--", ds)

# TODO: This will not allow to predict the empty sequence
@tf.function
def expand_first_sequence(window_elements_dict):
    """ Maps the first sequece to a dataset with initial incomplete subsequences. 
        Zero will be used for padding.
        Ex (padding element is zero, sequence_length = 2): [1, 2, 3] -> { "in":[ [1, 0], [1, 2] ], "out": [ 2 , 3 ] ] """
    expanded_sequence = {}
    for key in window_elements_dict:
        feature_values = window_elements_dict[key] # [1, 2, 3]
        if key in sequence_feature_names:
            # Sequence feature
            inputs_length = tf.shape(feature_values)[0] - 1
            inputs = feature_values[:-1] # [1, 2, 3] -> [1, 2]
            inputs = tf.reshape(inputs, (1, -1)) # [1, 2] -> [[1, 2]]
            inputs = tf.repeat(inputs, repeats=inputs_length, axis=0) # [[1, 2]] -> [[1, 2], [1, 2]]
            expanded_sequence[key] = tf.linalg.band_part( inputs , inputs_length , 0 ) # Keep lower subdiagonals: [[1, 2], [1, 2]] -> [[1, 0], [1, 2]]
        else:
            # Context or output
            expanded_sequence[key] = feature_values[1:] # [1, 2, 3] -> [2, 3]
    
    #return Dataset.from_tensor_slices(expanded_sequence)
    return expanded_sequence

first_element_ds = ds.take(1)
first_element_ds = first_element_ds.map(expand_first_sequence)
#first_element_ds = first_element_ds.flat_map(lambda x: Dataset.from_tensor_slices(x))

#ds = first_element_ds.concatenate( ds.skip(1) )
ds = first_element_ds

for example in ds:
  print(example)

