import tensorflow as tf
from tensorflow.data import Dataset

sequence_length = 55
sequence_feature_names = [ "sequence" ]
eos = -1

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
        Ex (padding element = 0, eos =-1, sequence_length = 2): 
        [1, 2, 3] -> { "in":[ [-1, 0, 0] [1, -1, 0], [1, 2, -1] ], "out": [ 1, 2 , 3 ] ] } """
    expanded_sequence = {}
    for key in window_elements_dict:
        if key in sequence_feature_names:
            inputs = window_elements_dict[key] # [1, 2, 3]
            # Sequence feature
            elements_length = tf.shape(inputs)[0]
            inputs = tf.reshape(inputs, (1, -1)) # [1, 2, 3] -> [[1, 2, 3]]
            inputs = tf.repeat(inputs, repeats=elements_length, axis=0) # [[1, 2, 3]] -> [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
            # Keep lower subdiagonals: [[1, 2, 3], [1, 2, 3], [1, 2, 3]] -> [[1, 0, 0], [1, 2, 0], [1, 2, 3]]
            inputs = tf.linalg.band_part( inputs , elements_length , 0 )
            # Assign EOS: [[1, 0, 0], [1, 2, 0], [1, 2, 3]] -> [[-1, 0, 0], [1, -1, 0], [1, 2, -1]]
            eos_vector = tf.repeat( eos, elements_length)
            expanded_sequence[key] = tf.linalg.set_diag(inputs, eos_vector)
        else:
            expanded_sequence[key] = window_elements_dict[key]
    
    #return Dataset.from_tensor_slices(expanded_sequence)
    return expanded_sequence

first_element_ds = ds.take(1)
first_element_ds = first_element_ds.map(expand_first_sequence)
first_element_ds = first_element_ds.flat_map(lambda x: Dataset.from_tensor_slices(x))

#ds = first_element_ds.concatenate( ds.skip(1) )
ds = first_element_ds

for example in ds:
  print(example)

