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

from msms_keras.MSMS_Generator import MSMS_Generator, MSprime_Generator
import utils

import tensorflow as tf

TRAIN = False
PREFIX = "models/sumstats"
SAVE_PERIOD = 2

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
            with open("{}_metrics.json".format(PREFIX), "w") as outfile:
                json.dump(self.metrics, outfile)
            for key in self.metrics:
                self.metrics[key] = float(self.metrics[key])

            with open("{}_all_metrics.json".format(PREFIX), "w") as outfile:
                json.dump(self.all_metrics, outfile)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def neural_network_fc(params):
    
    NUMEPOCHS = params.epochs
    BATCHSIZE = params.batchsize
    NUMTRAIN = params.total_sims
    CORES = params.cores
    l2_lambda = params.l2_lambda
    
    msms_gen = MSprime_Generator(params.num_individuals, params.sequence_length, 
            params.length_to_pad_to, params.pop_min, params.pop_max, yield_summary_stats=1)
    dims = msms_gen.dim

    model = Sequential()
    
    model.add(Dense(params.dense_1_dim, activation='relu', 
        kernel_initializer='normal',
        kernel_regularizer=keras.regularizers.l2(l2_lambda), 
        input_dim=8))
    model.add(Dropout(params.dense_1_drop))
    
    model.add(Dense(params.dense_2_dim, activation='relu', 
        kernel_initializer='normal',
        kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    model.add(Dropout(params.dense_2_drop))    
    
    model.add(Dense(params.dense_3_dim, activation='relu', 
        kernel_initializer='normal',
        kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    model.add(Dropout(params.dense_3_drop))    
    
    model.add(Dense(3))
    
    early_stop = EarlyStopping(monitor='mean_absolute_error', min_delta=.1, patience=5, 
                                restore_best_weights=True)

    check = ModelCheckpoint("{}_model.hdf5".format(PREFIX), 
            monitor='mean_absolute_error', save_best_only=True, 
            save_weights_only=False, period=1)

    metric = MetricHistory()
    
    model.compile(loss='mean_squared_error',
                      optimizer=keras.optimizers.Adam(),
                      metrics=[rmse, 'mean_absolute_error'])

    print(model.summary())

    history = model.fit_generator(msms_gen.data_generator(BATCHSIZE), 
            steps_per_epoch=NUMTRAIN/NUMEPOCHS, epochs=NUMEPOCHS, 
            workers=CORES, use_multiprocessing=True, 
            callbacks=[early_stop, metric, check])
    
    """
    model.save(PREFIX + '_model.hdf5')
    model.save_weights(PREFIX + '_weights.hdf5')
    """

    with open(PREFIX + '_trainhist.keras', 'wb') as f:
        pickle.dump(history.history, f)
    
    return model, history

if __name__ == "__main__":
    from keras.backend import tensorflow_backend
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    if TRAIN:
            
        params = utils.Params("./configurations/sumstats.json")
        
        neural_network_fc(params)
    
    else:
        model = load_model(PREFIX + '_model.hdf5', custom_objects={"rmse": rmse})
        with open("sumstats_X.keras", 'rb') as f:
            X = pickle.load(f)
        with open("sumstats_y.keras", 'rb') as f:
            y = pickle.load(f)

        y_pred = model.predict(X, batch_size=32)
        
        diff = y_pred - y
        mean_diff = np.mean(diff, axis=0)
        print(mean_diff)
        print(np.mean(mean_diff))

