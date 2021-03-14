from predictor import Predictor
from model_data_definition import ModelDataDefinition
from model import MaskedOneHotEncoding
import tensorflow as tf

data_definition = ModelDataDefinition()
predictor = Predictor(data_definition)

"""
Input: {"_file_path": "data/AlbaranNet.csv", "_file_row": 214, "controlType": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1], "ctxIsVariable": 0, "ctxParmAccess": 0, "ctxParmCollection": 2, "ctxParmDecimals": 10, "ctxParmLength": 18, "ctxParmType": 0, "dataTypeIdx": [5, 5, 2, 5, 2, 5, 2, 2, 5, 5, 2, 5, 2, 5, 2, 1], "decimalsBucket": [2, 2, 12, 2, 12, 2, 12, 12, 2, 2, 12, 2, 12, 2, 12, 1], "isCollection": [2, 2, 4, 2, 4, 2, 4, 4, 2, 2, 4, 2, 4, 2, 4, 1], "kbObjectTypeIdx": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1], "keywordIdx": [2, 2, 6, 2, 19, 2, 8, 2, 2, 2, 6, 2, 19, 2, 8, 1], "lengthBucket": [2, 15, 20, 2, 20, 2, 20, 20, 2, 15, 20, 2, 20, 2, 20, 1], "objectType": 4, "partType": 0, "textHash0": [2, 8, 2, 2, 2, 2, 2, 2, 2, 8, 2, 2, 2, 2, 2, 1], "textHash1": [2, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1], "textHash2": [2, 18, 2, 2, 2, 2, 2, 2, 2, 18, 2, 2, 2, 2, 2, 1], "textHash3": [2, 18, 2, 2, 2, 2, 2, 2, 2, 18, 2, 2, 2, 2, 2, 1], "trainable": 1, "wordType": [4, 10, 3, 4, 3, 4, 3, 3, 4, 10, 3, 4, 3, 4, 3, 1]}
Output: {"decimalsBucket": 10, "isCollection": 2, "isControl": 2, "lengthBucket": 18, "outputTypeIdx": 0, "textHash0": 0, "textHash1": 0, "textHash2": 0, "textHash3": 0}
"""

input = {}
for col_name in data_definition.sequence_columns:
    input[col_name] = []
for col_name in data_definition.context_columns:
    input[col_name] = 0

print(predictor.predict(input))
