from model_definition import ModelDefinition

model_definition = ModelDefinition()
trainer = model_definition.trainer_class()
trainer.train()

# Get prediction now to test if after export the prediction will be the same
# predictor = model_definition.predictor_class(model_definition.data_definition, trainer.model)
# print( "Prediction:" , predictor.predict( predictor.get_empty_element() ) )
