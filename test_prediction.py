from prediction_model import PredictionModel
from model_data_definition import ModelDataDefinition
from time import time
import json

# Read data definition
data_definition = ModelDataDefinition()

print("Reading latest exported model")
predictor = PredictionModel(data_definition)

# Sample input: First file word (sequence with all pad elements)
input = data_definition.get_empty_element()
#print(input)
json_test = json.dumps(input)

print( "Prediction:" , predictor.predict(input, data_definition) )

n_repetitions = 1000
print("Testing performance, n. repetitions:" , n_repetitions)
start = time()
for i in range(n_repetitions):
    #data_definition.input_sequence_to_tf_predict_format(input) # About 0.01 ms
    #predictor.predict(input, data_definition) # About 6.7 ms
    predictor.predict_json(json_test, data_definition) # About 6.9 ms
end = time()
print("Total time:" , end - start , "s")
print("Prediction performance:" , ((end - start) / n_repetitions) * 1000 , "ms")
