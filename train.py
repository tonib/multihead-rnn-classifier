from data_directory import DataDirectory
from model_data_definition import ModelDataDefinition
from classifier_dataset import ClassifierDataset
from model import generate_model
import tensorflow as tf
import os

# Read data definition
data_definition = ModelDataDefinition()
data_definition.print_summary()
print()

# Get train and evaluation source file paths
train_files, eval_files = DataDirectory.get_train_and_validation_sets(data_definition)
print("N. train files:", len(train_files.file_paths))
print("N. evaluation files:", len(eval_files.file_paths))

# Cache files for datasets will be created in dir. data/cache. If it does not exists, train will fail
cache_dir_path = data_definition.get_data_dir_path("cache")
if not os.path.exists(cache_dir_path):
    print("Creating directory " + cache_dir_path)
    os.mkdir(cache_dir_path)

# Get sequences datasets
batch_size = 64
train_dataset = ClassifierDataset(train_files, data_definition, shuffle=True)
train_cache_path = os.path.join(cache_dir_path, "train_cache")
print("Caching train dataset in " + train_cache_path)
train_dataset.dataset = train_dataset.dataset.cache(train_cache_path).prefetch(4096).shuffle(4096).batch(batch_size)

eval_dataset = ClassifierDataset(eval_files, data_definition, shuffle=False)
eval_cache_path = os.path.join(cache_dir_path, "eval_cache")
print("Caching evaluation dataset in " + eval_cache_path)
eval_dataset.dataset = eval_dataset.dataset.batch(256).cache(eval_cache_path)

# Create model
model = generate_model(data_definition)

# Losses for each output (sum of all will be minimized)
losses = {}
for output_column_name in data_definition.output_columns:
    losses[output_column_name] = loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Tensorboard callback, each epoch
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=data_definition.get_data_dir_path( ModelDataDefinition.TBOARD_LOGS_DIR ))

# Save checkpoints, each epoch
checkpoints_dir_path = data_definition.get_data_dir_path(ModelDataDefinition.CHECKPOINTS_DIR)
print("Storing checkpoints in " + checkpoints_dir_path)
checkpoint_path_prefix = checkpoints_dir_path + '/checkpoint-'
checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path_prefix + '{epoch:04d}.ckpt',
    save_weights_only=True,
    verbose=1
)

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
    callbacks=[tensorboard_callback, checkpoints_callback]
)
