from datasets import load_dataset
from tensorflow import keras
import time
import json
import pandas as pd
import glob
import pdb
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import layers,losses,models,regularizers
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error
from unet_utils import *
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallback
import argparse
from pathlib import Path

def train_unet(config):
    unet = padding_model(start_neurons=config['start_neurons'], 
                        activation=config['activation'],
                        dropout=config['dropout'])
    unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr'],
                                                    amsgrad=config['amsgrad']), 
                loss='mae', 
                loss_weights=[config['loss_weights']],
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

    home = str(Path.home())

    X_data = np.load(f'{home}/GIT/mig_exp/mps/unet/input_data.npy')
    X_data = np.expand_dims(X_data, axis=-1)
    Y_data = np.load(f'{home}/GIT/mig_exp/mps/unet/target_data.npy')
    Y_data = np.expand_dims(Y_data, axis=-1)
    
    tensorslice = tf.data.Dataset.from_tensor_slices((X_data,Y_data)).shuffle(buffer_size=len(Y_data),reshuffle_each_iteration=False).batch(config['batch'])
    train_data = tensorslice.skip(int(len(tensorslice) * 0.25)).shuffle(buffer_size=len(Y_data),reshuffle_each_iteration=True)
    test_data = tensorslice.take(int(len(tensorslice) * 0.25)).shuffle(buffer_size=len(Y_data),reshuffle_each_iteration=True)

    if config['lr_reduce']:
        callbacks = [TuneReportCallback(metrics={'mean_error': 'val_loss'}, on='epoch_end'),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.2)]
    else:
        callbacks = [TuneReportCallback(metrics={'mean_error': 'val_loss'}, on='epoch_end')]

    unet.fit(train_data,
            validation_data=test_data,
            epochs=200,
            verbose=0,
            callbacks=callbacks
            )

def ray_tune(num_training_iterations, num_cpu, config):
    sched = AsyncHyperBandScheduler(
            time_attr="training_iteration",
#            metric='mean_error',
            max_t=200, 
            grace_period=10)

    analysis = tune.run(
                train_unet,
                name='tune_unet',
                scheduler=sched,
                metric='mean_error',
                mode='min',
#                stop={'mean_error': 0.01, 'training_iteration': num_training_iterations},
                stop={'training_iteration': num_training_iterations},
                num_samples=300,
                resources_per_trial={'cpu': num_cpu, 'gpu': 1},
                config=config,
                checkpoint_score_attr='min-mean_error',
                keep_checkpoints_num=10,
                checkpoint_freq=10,
                checkpoint_at_end=True
                )

    print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == '__main__':
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, help='number of iterations required by ray_tune')
    parser.add_argument('--num_cpu', type=int, help='number of cpu cores')
    args = parser.parse_args()

    ray.init()

    config = {
        'lr': tune.uniform(0.0001, 0.1),
        'amsgrad': tune.choice([True, False]),
        'batch': tune.randint(1, 64),
#        'activation': tune.choice([tf.keras.activations.relu,
#                                tf.keras.activations.sigmoid,
#                                tf.keras.activations.tanh]), 
        'activation': tf.keras.activations.relu,
        'start_neurons': tune.choice([2, 4, 8, 16, 32]),
        'loss_weights': 1, #tune.uniform(0.01, 1),
        'dropout': tune.choice([True, False]),
        'lr_reduce': tune.choice([True, False])
        }

    ray_tune(args.iteration, args.num_cpu, config)


