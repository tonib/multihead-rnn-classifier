import tensorflow as tf
from tensorflow.data import Dataset

sequence_length = 4 # This length includes the EOS symbol, so the real length is -1
sequence_feature_names = [ "sequence" ]
context_feature_names = [ "context" ]
output_features_names = [ "output" , "sequence"]
eos = -1

data = { 'sequence': list(range(1,11)), 'context': list(range(21, 31)) , 'output': list(range(31, 41)) }
ds = Dataset.from_tensor_slices(data)
ds = ds.window(sequence_length, shift=1, drop_remainder=False)

def flat_map_window(window_elements_dict):
    for key in window_elements_dict:
        # See https://github.com/tensorflow/tensorflow/issues/23581#issuecomment-529702702
        window_elements_dict[key] = tf.data.experimental.get_single_element( window_elements_dict[key].batch(sequence_length + 1) )
    return window_elements_dict

ds = ds.map(flat_map_window)

@tf.function
def expand_first_sequence(window_elements_dict):
    """ Maps the first sequence to a dataset with initial incomplete subsequences. 
        Zero will be used for padding.
        Ex (padding element = 0, eos =-1, sequence_length = 2): 
        [1, 2, 3] -> { "in":[ [-1, 0, 0] [1, -1, 0], [1, 2, -1] ], "out": [ 1, 2 , 3 ] ] } """

    # Inputs
    input_dict = {}
    for key in sequence_feature_names:
        inputs = window_elements_dict[key] # [1, 2, 3]
        elements_length = tf.shape(inputs)[0]
        inputs = tf.reshape(inputs, (1, -1)) # [1, 2, 3] -> [[1, 2, 3]]
        inputs = tf.repeat(inputs, repeats=elements_length, axis=0) # [[1, 2, 3]] -> [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        # Keep lower subdiagonals: [[1, 2, 3], [1, 2, 3], [1, 2, 3]] -> [[1, 0, 0], [1, 2, 0], [1, 2, 3]]
        inputs = tf.linalg.band_part( inputs , elements_length , 0 )
        # Assign EOS: [[1, 0, 0], [1, 2, 0], [1, 2, 3]] -> [[-1, 0, 0], [1, -1, 0], [1, 2, -1]]
        eos_vector = tf.repeat( eos, elements_length)
        inputs = tf.linalg.set_diag(inputs, eos_vector)
        #if elements_length < (sequence_length + 1):
            # Pad right up to sequence length
        input_dict[key] = inputs

    for key in context_feature_names:
        input_dict[key] = window_elements_dict[key]
    
    # Outputs
    output_dict = {}
    for key in output_features_names:
        output_dict[key] = window_elements_dict[key]
    
    return (input_dict, output_dict)

# def process_full_sequence(window_element_dict):


first_window_ds = ds.take(1)
first_window_ds = first_window_ds.map(expand_first_sequence)
first_window_ds = first_window_ds.flat_map( lambda x, y: Dataset.zip( (Dataset.from_tensor_slices(x), Dataset.from_tensor_slices(y)) ) )

# later_windows_ds = ds.skip(1)
# later_windows_ds.map()

#ds = first_element_ds.concatenate( later_windows_ds )
ds = first_window_ds

for example in ds:
  print(example)

