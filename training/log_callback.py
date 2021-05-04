from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from model_data_definition import ModelDataDefinition

import tensorflow as tf
from time import time

class LogCallback(tf.keras.callbacks.Callback):
    def __init__(self, data_definition: ModelDataDefinition):
        super().__init__()
        self.data_definition = data_definition

    def on_train_begin(self, logs=None):
        print("Train started")
        self.start_time = time() 
        self.last_time = self.start_time
    
    def print_batch(self, train: bool, batch, logs=None):
        if batch > 0 and batch % self.data_definition.log_each_batches == 0:
            current = time()
            elapsed = current - self.last_time
            rate = ( self.data_definition.log_each_batches / elapsed ) if elapsed > 0 else 0
            total_elapsed = current - self.start_time
            if train:
                print("\n* Train - batch {} / {:.2f} s / {:.2f} batch/s: {}\n".format(batch, total_elapsed, rate, logs))
            else:
                print("* Evaluation - batch {} / {:.2f} s / {:.2f} batch/s".format(batch, total_elapsed, rate))
            self.last_time = current

    def on_train_batch_end(self, batch, logs=None):
        self.print_batch(True, batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        self.print_batch(False, batch, logs)