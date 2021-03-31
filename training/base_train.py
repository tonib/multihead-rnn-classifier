import configure_tf_log # Must be FIRST import

from typing import Type

from data_directory import DataDirectory
from model_data_definition import ModelDataDefinition
from model.rnn_model import generate_model
from training.log_callback import LogCallback
import tensorflow as tf
import os
import time

class BaseTrain:
    """ Base class for training """

    def __init__(self, dataset_class:Type):
        """
            Args:
                dataset_class: class type for datasets used in this train
        """
        self.dataset_class = dataset_class
        self.callbacks = []

        # Read data definition
        self.data_definition = ModelDataDefinition()
        self.data_definition.print_summary()
        print()

        self.create_datasets()
        self.model = self.create_model()
        self.create_losses()
        self.prepare_checkpoints()
        self.other_callbacks()
        self.compile()
        self.model.summary()

    def create_datasets(self):
        # Get train and evaluation source file paths
        self.train_files, self.eval_files = DataDirectory.get_train_and_validation_sets(self.data_definition)
        print("N. train files:", len(self.train_files.file_paths))
        print("N. evaluation files:", len(self.eval_files.file_paths))
        print()

        if self.data_definition.cache_dataset:
            # Cache files for datasets will be created in dir. data/cache. If it does not exists, train will fail
            cache_dir_path = self.data_definition.get_data_dir_path("cache")
            if not os.path.exists(cache_dir_path):
                print("Creating directory " + cache_dir_path)
                os.mkdir(cache_dir_path)

        # Train dataset
        self.train_dataset = self.dataset_class(self.train_files, self.data_definition, shuffle=True)
        if self.data_definition.cache_dataset:
            train_cache_path = os.path.join(cache_dir_path, "train_cache")
            print("Caching train dataset in " + train_cache_path)
            self.train_dataset.dataset = self.train_dataset.dataset.cache(train_cache_path)
        self.train_dataset.dataset = self.train_dataset.dataset.shuffle(1024).batch( self.data_definition.batch_size )
        if self.data_definition.max_batches_per_epoch > 0:
            print("Train dataset limited to n. batches:", self.data_definition.max_batches_per_epoch)
            self.train_dataset.dataset = self.train_dataset.dataset.take( self.data_definition.max_batches_per_epoch )
        self.train_dataset.dataset = self.train_dataset.dataset.prefetch(4)

        # Evaluation dataset (shuffle=True -> Important for performance: It will enable files interleave)
        self.eval_dataset = self.dataset_class(self.eval_files, self.data_definition, shuffle=True)
        self.eval_dataset.dataset = self.eval_dataset.dataset.batch( self.data_definition.batch_size )
        if self.data_definition.cache_dataset:
            eval_cache_path = os.path.join(cache_dir_path, "eval_cache")
            print("Caching evaluation dataset in " + eval_cache_path)
            self.eval_dataset.dataset = self.eval_dataset.dataset.cache(eval_cache_path)
        if self.data_definition.max_batches_per_epoch > 0:
            max_eval_batches = int( self.data_definition.max_batches_per_epoch * self.data_definition.percentage_evaluation )
            print("Evaluation dataset limited to n. batches:", max_eval_batches)
            self.eval_dataset.dataset = self.eval_dataset.dataset.take( max_eval_batches )
        self.eval_dataset.dataset = self.eval_dataset.dataset.prefetch(4)

        print()

    def create_model(self) -> tf.keras.Model:
        # To be defined in inherited classes
        return None

    def create_losses(self):
        # Losses for each output (sum of all will be minimized)
        self.losses = {}
        for output_column_name in self.data_definition.output_columns:
            self.losses[output_column_name] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def prepare_checkpoints(self):
        # Save checkpoints, each epoch
        checkpoints_dir_path = self.data_definition.get_data_dir_path(ModelDataDefinition.CHECKPOINTS_DIR)
        print("Storing checkpoints in " + checkpoints_dir_path)
        checkpoint_path_prefix = checkpoints_dir_path + '/checkpoint-'
        checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path_prefix + '{epoch:04d}.ckpt',
            save_weights_only=True,
            verbose=1
        )
        self.callbacks.append( checkpoints_callback )

        # Restore latest checkpoint
        latest_cp = tf.train.latest_checkpoint( checkpoints_dir_path )
        if latest_cp != None:
            # Get epoch number from checkpoint path
            l = len(checkpoint_path_prefix)
            self.initial_epoch = int(latest_cp[l:l+4])
            print("*** Continuing training from checkpoint " + latest_cp + ", epoch " + str(initial_epoch))
            self.model.load_weights(latest_cp)
        else:
            self.initial_epoch = 0

    def other_callbacks(self):
        # Tensorboard callback, each epoch
        self.callbacks.append( tf.keras.callbacks.TensorBoard(log_dir=self.data_definition.get_data_dir_path( ModelDataDefinition.TBOARD_LOGS_DIR )) )

        if self.data_definition.log_each_epochs > 0:
            # Log each x epochs, to check peformance
            self.callbacks.append( LogCallback(self.data_definition) )

    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.data_definition.learning_rate),
            loss = self.losses,
            metrics=['accuracy']
        )

    def train(self):
        self.model.fit(self.train_dataset.dataset, 
            initial_epoch=self.initial_epoch,
            epochs=self.initial_epoch + self.data_definition.max_epochs,
            validation_data=self.eval_dataset.dataset,
            #validation_steps=n_eval_batches,
            verbose=2,
            callbacks=self.callbacks
        )