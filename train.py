from data_directory import DataDirectory
from model_data_definition import ModelDataDefinition
from classifier_dataset import ClassifierDataset
from model import generate_model
import tensorflow as tf

#tf.compat.v1.disable_eager_execution()

# Read data definition
data_definition = ModelDataDefinition()
data_definition.print_summary()
print()

# Get train and evaluation source file paths
train_files, eval_files = DataDirectory.get_train_and_validation_sets(data_definition)
print("N. train files:", len(train_files.file_paths))
print("N. evaluation files:", len(eval_files.file_paths))

# Get sequences datasets
batch_size = 32
train_dataset = ClassifierDataset(train_files, data_definition, shuffle=True, batch_size=32)
eval_dataset = ClassifierDataset(eval_files, data_definition, shuffle=False, batch_size=128)

# Additional datasets configuration. TODO: Prefetch, shuffle, etc
# train_dataset.dataset = train_dataset.dataset.cache()
# eval_dataset.dataset = eval_dataset.dataset.cache()

# We need the batches number in evaluation dataset, so here is:
print("Getting n. batches in evaluation dataset")
n_eval_batches = eval_dataset.n_batches_in_dataset()
print("Getting n. batches in evaluation dataset - Done:", n_eval_batches)

# Create model
model = generate_model(data_definition)

# Losses for each output (sum of all will be minimized)
losses = {}
for output_column_name in data_definition.output_columns:
    losses[output_column_name] = loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=data_definition.learning_rate),
    loss = losses,
    metrics=['accuracy']
)
model.summary()

model.fit(train_dataset.dataset, 
        epochs=data_definition.max_epochs,
        validation_data=eval_dataset.dataset,
        validation_steps=n_eval_batches,
        verbose=2
)
