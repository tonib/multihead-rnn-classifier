import tensorflow as tf
from model import generate_model
from model_data_definition import ModelDataDefinition

data_definition = ModelDataDefinition()

m = generate_model(data_definition)
m.summary()
