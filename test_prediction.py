import configure_tf_log # Must be FIRST import
from model_definition import ModelDefinition
from predict.predictor import Predictor
from time import time
import json

# Read model definition
model_definition = ModelDefinition()

print("Reading exported model")
predictor = Predictor(model_definition)

# Sample input: First file word (sequence with all pad elements)
input = predictor.get_empty_element()
print(input)
json_test = json.dumps(input)

print( "Prediction:" , predictor.predict(input) )

n_repetitions = 1000
print("Testing performance, n. repetitions:" , n_repetitions)
start = time()
for i in range(n_repetitions):
    predictor.predict_json(json_test) 
end = time()
print("Total time:" , end - start , "s")
print("Prediction performance:" , ((end - start) / n_repetitions) * 1000 , "ms")

# RNN: House computer (Linux):
# seq_len = 16, rnn size = 64 -> With GPU: 1.7 ms / With CPU: 0.85 ms
# seq_len= 64, rnn_size = 256 -> With GPU: 2.76 ms / With CPU: 4.24 ms

# RNN: Work computer (Windows 10): 
# seq_len= 64, rnn_size = 256 -> With CPU: 13-14 ms

# GPT / House / GPU / seq_len = 64, embedding = 128, 2 layers, n.heads = 2: 3.18 ms
# GPT / House / CPU / seq_len = 64, embedding = 128, 2 layers, n.heads = 2: 2.08 ms ???
