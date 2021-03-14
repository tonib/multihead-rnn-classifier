from predictor import Predictor
from model_data_definition import ModelDataDefinition
from model import MaskedOneHotEncoding
import tensorflow as tf

data_definition = ModelDataDefinition()
predictor = Predictor(data_definition)

def process(x):
    for k in x:
        x[k] = tf.constant(x[k], dtype=tf.int32)
    print(x)
    print( predictor._preprocess_input(x) )
    print()
    print()

process({ "wordType": [ 0 , 1 , 2 ] , "keywordIdx": [ 2 , 1 , 0 ] , "ctxParmType": 7 })
process({ "wordType": [ 0 , 1 , 2, 3, 4 ] , "keywordIdx": [ 2 , 1 , 0 , 3, 4 ] , "ctxParmType": 7 })
process({ "wordType": [ 0 , 1 , 2, 3, 4 , 5 , 6 ] , "keywordIdx": [ 2 , 1 , 0 , 3, 4 , 5 , 6 ] , "ctxParmType": 7 })
process({ "wordType": [] , "keywordIdx": [] , "ctxParmType": 7 })

# one_hot = MaskedOneHotEncoding(4)
# x = tf.constant( [ [1, 2] , [3, 0] ] )
# print(x)
# print( one_hot(x) )
# print( one_hot.compute_mask(x) )
