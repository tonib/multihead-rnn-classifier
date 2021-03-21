from data_directory import DataDirectory
from model_data_definition import ModelDataDefinition
from classifier_dataset import ClassifierDataset
from model import generate_model
from log_callback import LogCallback
import tensorflow as tf
import os
import time

# Read data definition
data_definition = ModelDataDefinition()
data_definition.print_summary()
print()

# Get train and evaluation source file paths
train_files, eval_files = DataDirectory.get_train_and_validation_sets(data_definition)
print("N. train files:", len(train_files.file_paths))
print("N. evaluation files:", len(eval_files.file_paths))
print()

if data_definition.cache_dataset:
    # Cache files for datasets will be created in dir. data/cache. If it does not exists, train will fail
    cache_dir_path = data_definition.get_data_dir_path("cache")
    if not os.path.exists(cache_dir_path):
        print("Creating directory " + cache_dir_path)
        os.mkdir(cache_dir_path)

# Train dataset
train_dataset = ClassifierDataset(train_files, data_definition, shuffle=True)
if data_definition.cache_dataset:
    train_cache_path = os.path.join(cache_dir_path, "train_cache")
    print("Caching train dataset in " + train_cache_path)
    train_dataset.dataset = train_dataset.dataset.cache(train_cache_path)
train_dataset.dataset = train_dataset.dataset.shuffle(1024).batch( data_definition.batch_size )
if data_definition.max_batches_per_epoch > 0:
    print("Train dataset limited to n. batches:", data_definition.max_batches_per_epoch)
    train_dataset.dataset = train_dataset.dataset.take( data_definition.max_batches_per_epoch )
train_dataset.dataset = train_dataset.dataset.prefetch(4)

# Evaluation dataset (shuffle=True -> Important for performance: It will enable files interleave)
eval_dataset = ClassifierDataset(eval_files, data_definition, shuffle=True)
eval_dataset.dataset = eval_dataset.dataset.batch( data_definition.batch_size )
if data_definition.cache_dataset:
    eval_cache_path = os.path.join(cache_dir_path, "eval_cache")
    print("Caching evaluation dataset in " + eval_cache_path)
    eval_dataset.dataset = eval_dataset.dataset.cache(eval_cache_path)
if data_definition.max_batches_per_epoch > 0:
    max_eval_batches = int( data_definition.max_batches_per_epoch * data_definition.percentage_evaluation )
    print("Evaluation dataset limited to n. batches:", max_eval_batches)
    eval_dataset.dataset = eval_dataset.dataset.take( max_eval_batches )
eval_dataset.dataset = eval_dataset.dataset.prefetch(4)

print()

# Create model
model = generate_model(data_definition)

# Losses for each output (sum of all will be minimized)
losses = {}
for output_column_name in data_definition.output_columns:
    losses[output_column_name] = loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Callbacks:
callbacks = []

# Tensorboard callback, each epoch
callbacks.append( tf.keras.callbacks.TensorBoard(log_dir=data_definition.get_data_dir_path( ModelDataDefinition.TBOARD_LOGS_DIR )) )

# Save checkpoints, each epoch
checkpoints_dir_path = data_definition.get_data_dir_path(ModelDataDefinition.CHECKPOINTS_DIR)
print("Storing checkpoints in " + checkpoints_dir_path)
checkpoint_path_prefix = checkpoints_dir_path + '/checkpoint-'
checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path_prefix + '{epoch:04d}.ckpt',
    save_weights_only=True,
    verbose=1
)
callbacks.append( checkpoints_callback )

if data_definition.log_each_epochs > 0:
    # Log each x epochs, to check peformance
    callbacks.append( LogCallback(data_definition) )

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=data_definition.learning_rate),
    loss = losses,
    metrics=['accuracy']
)
model.summary()

# Restore latest checkpoint
latest_cp = tf.train.latest_checkpoint( checkpoints_dir_path )
if latest_cp != None:
    # Get epoch number from checkpoint path
    l = len(checkpoint_path_prefix)
    initial_epoch = int(latest_cp[l:l+4])
    print("*** Continuing training from checkpoint " + latest_cp + ", epoch " + str(initial_epoch))
    model.load_weights(latest_cp)
else:
    initial_epoch = 0

model.fit(train_dataset.dataset, 
    initial_epoch=initial_epoch,
    epochs=initial_epoch + data_definition.max_epochs,
    validation_data=eval_dataset.dataset,
    #validation_steps=n_eval_batches,
    verbose=2,
    callbacks=callbacks
)
