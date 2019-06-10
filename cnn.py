from os import sys
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling2D, MaxPooling1D, Dropout, AveragePooling2D, AveragePooling1D
from keras import optimizers

import numpy as np
import sys
import h5py
import os
import pandas as pd
import math

from sklearn.model_selection import train_test_split
import pickle

from msms_keras.MSMS_Generator import MSMS_Generator
import utils

train_file="train.h5py"
X_train_name = 'X_train'
y_train_name = 'y_train'

TRAIN = True

def neural_network_1fc(prefix, params, X_train, y_train):
    
    NUMEPOCHS = params.epochs
    BATCHSIZE = params.batchsize
    NUMTRAIN = params.total_sims
    NUMTEST = NUMTRAIN * .2
    CORES = params.cores
    l2_lambda = params.l2_lambda
    ksize = (params.k_height, params.k_width)
    print (ksize)

    msms_gen = MSMS_Generator(params.num_individuals, 
            params.sequence_length, params.length_to_pad_to, 
            params.total_sims, params.pop_min, params.pop_max, 
            params.T, params.rho_region)
    dims = msms_gen.dim
    print (dims)
    
    model = Sequential()
     
    model.add(Conv2D(params.conv_2D_1_out_dim, kernel_size=ksize, 
        activation='relu',input_shape=dims, 
        kernel_regularizer=keras.regularizers.l2(l2_lambda), 
        data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=ksize, data_format='channels_first'))
    model.add(Dropout(params.conv_2D_1_drop))
    
    model.add(Conv2D(params.conv_2D_2_out_dim, kernel_size=ksize,
        activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda), 
        data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=ksize, data_format='channels_first'))
    model.add(Dropout(params.conv_2D_2_drop))

    # model.add(Conv2D(params.conv_2D_3_out_dim, kernel_size=ksize, activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda), data_format='channels_first'))
    # model.add(MaxPooling2D(pool_size=ksize,data_format='channels_first'))
    # model.add(Dropout(params.conv_2D_2_drop))

    model.add(Flatten())

    model.add(Dense(params.dense_1_dim, activation='relu', 
        kernel_initializer='normal',
        kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    model.add(Dropout(params.dense_1_drop))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error',
                      optimizer=keras.optimizers.Adam(),
                      metrics=['mean_squared_error','mean_absolute_error'])

    print(model.summary())

    print (X_train)
    history = model.fit(x=X_train,y=y_train, batch_size=8, 
            epochs = params.epochs, shuffle=False)

    model.save(prefix + '_model.hdf5')
    model.save_weights(prefix + '_weights.hdf5')

    with open(prefix + '_trainhist.keras', 'wb') as f:
        pickle.dump(history.history, f)

    return model, history

def neural_network_2fc(prefix, params, X_train, y_train):

    NUMEPOCHS = params.epochs
    BATCHSIZE = params.batchsize
    NUMTRAIN = params.total_sims
    NUMTEST = NUMTRAIN * .2
    CORES = params.cores
    l2_lambda = params.l2_lambda
    ksize = (params.k_height, params.k_width)
    
    msms_gen = MSMS_Generator(params.num_individuals, params.sequence_length, 
            params.length_to_pad_to, params.total_sims, params.pop_min, 
            params.pop_max, params.T, params.rho_region)
    dims = msms_gen.dim
    
    model = Sequential()
    model.add(Conv2D(params.conv_2D_1_out_dim, kernel_size=ksize, 
        activation='relu',input_shape=(dims), 
        kernel_regularizer=keras.regularizers.l2(l2_lambda), 
        data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=ksize, data_format='channels_first'))
    model.add(Dropout(params.conv_2D_1_drop))

    model.add(Conv2D(params.conv_2D_2_out_dim, kernel_size=ksize, 
        activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda), 
        data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=ksize, data_format='channels_first'))
    model.add(Dropout(params.conv_2D_2_drop))

    # model.add(Conv2D(params.conv_2D_3_out_dim, kernel_size=ksize, activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda), data_format='channels_first'))
    # model.add(MaxPooling2D(pool_size=ksize,data_format='channels_first'))
    # model.add(Dropout(params.conv_2D_2_drop))

    model.add(Flatten())

    model.add(Dense(params.dense_1_dim, activation='relu', 
        kernel_initializer='normal',
        kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    model.add(Dropout(params.dense_1_drop))

    model.add(Dense(params.dense_2_dim, activation='relu', 
        kernel_initializer='normal',
        kernel_regularizer=keras.regularizers.l2(l2_lambda)))
    model.add(Dropout(params.dense_2_drop))    
    
    model.add(Dense(1))

    model.compile(loss='mean_squared_error',
                      optimizer=keras.optimizers.Adam(),
                      metrics=['mean_squared_error','mean_absolute_error'])

    print(model.summary())

    history = model.fit(x=X_train,y= y_train, epochs=NUMEPOCHS, batch_size=8, 
            shuffle=False)

    model.save(prefix + '_model.hdf5')
    model.save_weights(prefix + '_weights.hdf5')

    with open(prefix + '_trainhist.keras', 'wb') as f:
        pickle.dump(history.history, f)

    return model, history



if __name__ == "__main__":
    from keras.backend import tensorflow_backend

    if TRAIN:
        #configs = ["2x2.json", "1x2.json", "1x4.json", "4x4.json"]	
        configs = ["1x5.json"]
    with h5py.File(train_file) as train_f:
        X_train = train_f.get(X_train_name)
        print ("Got X")
        y_train= train_f.get(y_train_name)
        print ("Got y")

        for config in configs:
            params = utils.Params("./configurations/" + config)
        
            neural_network_1fc('cnn_1fc_{}'.format(config), params, X_train, 
                    y_train)
            neural_network_2fc('cnn_2fc_{}'.format(config), params, X_train, 
                    y_train)
    
    """
    hist = pickle.load(open('basic_size_{}'.format(NUMEPOCHS) + '_trainhist.keras', "rb"))
    print(hist)
    """
    
