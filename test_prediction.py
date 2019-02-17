from prediction_model import PredictionModel
from model_data_definition import ModelDataDefinition
from time import time

# Read data definition
data_definition = ModelDataDefinition( 'data' )

print("Reading latest exported model")
predictor = PredictionModel()

# Sample input: First file word (sequence with all paddign elements)
input = [ data_definition.get_padding_element() ] * data_definition.sequence_length

print( "Prediction:" , predictor.predict(input, data_definition) )

print("Testing performance")
n_repetitions = 5000
start = time()
for i in range(n_repetitions):
    predictor.predict(input, data_definition)
end = time()
print("Total time:" , end - start)
print("Seconds/prediction:" , (end - start) / n_repetitions)
