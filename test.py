from predictor import Predictor
from model_data_definition import ModelDataDefinition
from classifier_dataset import ClassifierDataset
from data_directory import DataDirectory
import tensorflow as tf
import numpy as np

data_definition = ModelDataDefinition()
predictor = Predictor(data_definition)

l = predictor.model.get_layer('one_hot_ctxParmType')

# print(l)
# print(l.input_n_labels)
# print(l(tf.constant[2]))
# exit()

def print_label_prob(output, label):
    label_idx = data_definition.column_definitions['outputTypeIdx'].labels.index(label)
    print(label, ".", output[label_idx])

all_data = DataDirectory.read_all(data_definition)
ds = ClassifierDataset(all_data, data_definition, shuffle=False, debug_columns=True)
for x in ds.dataset.batch(1).take(1):
    #print(x)
    input = x[0]
    output = x[1]
    #print(output['outputTypeIdx'])
    output = predictor.model(input)['outputTypeIdx'][0]
    output = tf.nn.softmax(output)

    print_label_prob(output, "_if")
    print_label_prob(output, "_endif")
    print_label_prob(output, "_for")
    print_label_prob(output, "_endfor")
    print()
    
    sorted_label_indices = np.argsort( -output.numpy() )
    labels = data_definition.column_definitions['outputTypeIdx'].labels
    for idx in sorted_label_indices[:10]:
        print( labels[idx], output[idx] )

    # if_idx = data_definition.column_definitions['outputTypeIdx'].labels.index("_if")
    # print( output[23] )

    exit()

    # print( predictor._predict_tf(input) )
    # print( predictor._predict_tf(input)['outputTypeIdx'][23] )

    #   dtype=int32)>, 'ctxParmType': <tf.Tensor: shape=(), dtype=int32, numpy=0>, 'ctxParmExtTypeHash': <tf.Tensor: shape=(), dtype=int32, numpy=0>, 
    #   'ctxParmLength': <tf.Tensor: shape=(), dtype=int32, numpy=18>, 'ctxParmDecimals': <tf.Tensor: shape=(), dtype=int32, numpy=10>, 
    #   'ctxParmCollection': <tf.Tensor: shape=(), dtype=int32, numpy=2>, 'ctxParmAccess': <tf.Tensor: shape=(), dtype=int32, numpy=0>, 
    #   'ctxIsVariable': <tf.Tensor: shape=(), dtype=int32, numpy=0>, 'objectType': <tf.Tensor: shape=(), dtype=int32, numpy=2>, 
    #   'partType': <tf.Tensor: shape=(), dtype=int32, numpy=2>, 
    #   '_file_path': <tf.Tensor: shape=(), dtype=string, numpy=b'data/PUAlcRdiNro.csv'>, 
    #   '_file_row': <tf.Tensor: shape=(), dtype=int32, numpy=2>, 
    #   'trainable': <tf.Tensor: shape=(), dtype=int32, numpy=1>}, 
    #   {'isCollection': <tf.Tensor: shape=(), dtype=int32, numpy=2>, 
    #   'lengthBucket': <tf.Tensor: shape=(), dtype=int32, numpy=18>, 
    #   'decimalsBucket': <tf.Tensor: shape=(), dtype=int32, numpy=10>, 
    #   'outputTypeIdx': <tf.Tensor: shape=(), dtype=int32, numpy=23>, 
    #   'outputExtTypeHash': <tf.Tensor: shape=(), dtype=int32, numpy=0>, 
    #   'textHash0': <tf.Tensor: shape=(), dtype=int32, numpy=0>, 
    #   'textHash1': <tf.Tensor: shape=(), dtype=int32, numpy=0>, 
    #   'textHash2': <tf.Tensor: shape=(), dtype=int32, numpy=0>, 
    #   'textHash3': <tf.Tensor: shape=(), dtype=int32, numpy=0>, 
    #   'isControl': <tf.Tensor: shape=(), dtype=int32, numpy=2>})

# input = {"wordType": [], "keywordIdx": [], "kbObjectTypeIdx": [], "dataTypeIdx": [], "dataTypeExtTypeHash": [], 
# "isCollection": [], "lengthBucket": [], "decimalsBucket": [], "textHash0": [], "textHash1": [], "textHash2": [], 
# "textHash3": [], "controlType": [], 

# "ctxParmType": 0, "ctxParmExtTypeHash": 0, "ctxParmLength": 18, 
# "ctxParmDecimals": 10, "ctxParmCollection": 2, "ctxParmAccess": 2, "ctxIsVariable": 0, "objectType": 2, "partType": 2}

# print( predictor.predict(input)['outputTypeIdx']['probabilities'][23])

# print( predictor._preprocess_input(input) )


