from os import sys
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling2D, MaxPooling1D, Dropout, AveragePooling2D, AveragePooling1D
import keras.backend as K
from keras import optimizers
import keras.callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import sys
import h5py
import os
import pandas as pd
import math
import json

from sklearn.model_selection import train_test_split
import pickle

from msms_keras.MSMS_Generator import MSMS_Generator
import utils

import tensorflow as tf

TRAIN = True
PREFIX = "3c_20x1"
SAVE_PERIOD = 10

class MetricHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.metrics = {"rmse": math.inf, "mean_absolute_error": math.inf}
        self.all_metrics = {"rmse": [], "mean_absolute_error": []}

    def on_epoch_end(self, epoch, logs={}):
        self.metrics['rmse'] = min(self.metrics['rmse'], logs.get('rmse'))
        self.metrics['mean_absolute_error'] = min(self.metrics['mean_absolute_error'],
                                            logs.get('mean_absolute_error'))

        self.all_metrics["rmse"].append(str(logs.get("rmse")))
        self.all_metrics["mean_absolute_error"].append(str(logs.get("mean_absolute_error")))

        if epoch % SAVE_PERIOD == 0:
            for key in self.metrics:
                self.metrics[key] = str(self.metrics[key])
            with open("models/{}_metrics.json".format(PREFIX), "w") as outfile:
                json.dump(self.metrics, outfile)
            for key in self.metrics:
                self.metrics[key] = float(self.metrics[key])

            with open("models/{}_all_metrics.json".format(PREFIX), "w") as outfile:
                json.dump(self.all_metrics, outfile)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def neural_network_3c(params):
    
    NUMEPOCHS = params.epochs
    BATCHSIZE = params.batchsize
    NUMTRAIN = params.total_sims
    CORES = params.cores
    l2_lambda = params.l2_lambda
    ksize = (params.k_height, params.k_width)
    poolsize = (params.pool_height, params.pool_width)
    
    msms_gen = MSMS_Generator(params.num_individuals, params.sequence_length, 
            params.length_to_pad_to, params.pop_min, params.pop_max)
    dims = msms_gen.dim

    
    model = Sequential()
     
    model.add(Conv2D(params.conv_2D_1_out_dim, kernel_size=ksize, 
        activation='relu',input_shape=dims, 
        kernel_regularizer=keras.regularizers.l2(l2_lambda), 
        data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=poolsize, data_format='channels_first'))
    model.add(Dropout(params.conv_2D_1_drop))
    
    model.add(Conv2D(params.conv_2D_2_out_dim, kernel_size=ksize, 
        activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda), 
        data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=poolsize, data_format='channels_first'))
    model.add(Dropout(params.conv_2D_2_drop))

    model.add(Conv2D(params.conv_2D_3_out_dim, kernel_size=ksize, 
        activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda), 
        data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=poolsize, data_format='channels_first'))
    model.add(Dropout(params.conv_2D_3_drop))

    model.add(Flatten())

    model.add(Dense(params.dense_1_dim, activation='relu', 
        kernel_initializer='normal',
        kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    model.add(Dropout(params.dense_1_drop))
    model.add(Dense(1))
    
    early_stop = EarlyStopping(monitor='mean_absolute_error', min_delta=20, patience=5, 
                                restore_best_weights=True)
    """
    check = ModelCheckpoint("models/{}_model.hdf5".format(PREFIX), 
            monitor='mean_absolute_error', save_best_only=True, 
            save_weights_only=False, period=1)
    """
    metric = MetricHistory()
    
    model.compile(loss='mean_squared_error',
                      optimizer=keras.optimizers.Adam(),
                      metrics=[rmse, 'mean_absolute_error'])

    print(model.summary())

    history = model.fit_generator(msms_gen.data_generator(BATCHSIZE), 
            steps_per_epoch=NUMTRAIN/NUMEPOCHS, epochs=NUMEPOCHS, 
            workers=CORES, use_multiprocessing=True, 
            callbacks=[early_stop, metric])
    
    """
    model.save(prefix + '_model.hdf5')
    model.save_weights(prefix + '_weights.hdf5')

    with open(prefix + '_trainhist.keras', 'wb') as f:
        pickle.dump(history.history, f)
    """

    return model, history

if __name__ == "__main__":
    from keras.backend import tensorflow_backend
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    if TRAIN:
            
        params = utils.Params("./configurations/20x1.json")
        
        neural_network_3c(params)
    
    
