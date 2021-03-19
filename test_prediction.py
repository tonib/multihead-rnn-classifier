from predictor import Predictor
from model_data_definition import ModelDataDefinition
from time import time
import json

# Read data definition
data_definition = ModelDataDefinition()

print("Reading exported model")
predictor = Predictor(data_definition)

# Sample input: First file word (sequence with all pad elements)
input = data_definition.get_empty_element()
#print(input)
json_test = json.dumps(input)

print( "Prediction:" , predictor.predict(input) )

n_repetitions = 1000
print("Testing performance, n. repetitions:" , n_repetitions)
start = time()
for i in range(n_repetitions):
    # My house computer (Linux):
    # seq_len = 16, rnn size = 64 -> With GPU: 1.7 ms / With CPU: 0.85 ms
    # seq_len= 64, rnn_size = 256 -> With GPU: 2.76 ms / With CPU: 4.24 ms

    # Work computer (Windows 10): 
    # seq_len= 64, rnn_size = 256 -> With CPU: 13-14 ms
    
    predictor.predict_json(json_test) 
end = time()
print("Total time:" , end - start , "s")
print("Prediction performance:" , ((end - start) / n_repetitions) * 1000 , "ms")
