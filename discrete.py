from os import sys
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling2D, MaxPooling1D, Dropout, AveragePooling2D, AveragePooling1D, Input, Activation
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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import pickle

from msms_keras.MSMS_Generator import Discrete_Generator
import utils

import tensorflow as tf

TRAIN = True 
PREFIX = "models/discrete"
SAVE_PERIOD = 2

class DiscreteNet:
    @staticmethod
    def build_N1_branch(num_buckets, params, inputs):
        l2_lambda = params.l2_lambda
        ksize = (params.k_height, params.k_width)
        poolsize = (params.pool_height, params.pool_width)
     
        x = Conv2D(params.conv_2D_1_out_dim, kernel_size=ksize, 
            activation='relu', 
            kernel_regularizer=keras.regularizers.l2(l2_lambda), 
            data_format='channels_first', strides=(1, params.stride))(inputs)
        x = MaxPooling2D(pool_size=poolsize, data_format='channels_first')(x)
        x = Dropout(params.conv_2D_1_drop)(x)
        
        x = Conv2D(params.conv_2D_2_out_dim, kernel_size=ksize, 
            activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda), 
            data_format='channels_first', strides=(1, params.stride))(x)
        x = MaxPooling2D(pool_size=poolsize, data_format='channels_first')(x)
        x = Dropout(params.conv_2D_2_drop)(x)

        x = Conv2D(params.conv_2D_3_out_dim, kernel_size=ksize, 
            activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda), 
            data_format='channels_first', strides=(1, params.stride))(x)
        x = MaxPooling2D(pool_size=poolsize, data_format='channels_first')(x)
        x = Dropout(params.conv_2D_3_drop)(x)
        
        x = Flatten()(x)

        x = Dense(params.dense_1_dim, activation='relu', 
            kernel_initializer='normal',
            kernel_regularizer=keras.regularizers.l2(l2_lambda))(x)
        x = Dropout(params.dense_1_drop)(x)

        x = Dense(num_buckets)(x)
        x = Activation('softmax', name="N1_branch")(x)

        return x

    @staticmethod
    def build_N2_branch(num_buckets, params, inputs):
        l2_lambda = params.l2_lambda
        ksize = (params.k_height, params.k_width)
        poolsize = (params.pool_height, params.pool_width)
     
        x = Conv2D(params.conv_2D_1_out_dim, kernel_size=ksize, 
            activation='relu', 
            kernel_regularizer=keras.regularizers.l2(l2_lambda), 
            data_format='channels_first', strides=(1, params.stride))(inputs)
        x = MaxPooling2D(pool_size=poolsize, data_format='channels_first')(x)
        x = Dropout(params.conv_2D_1_drop)(x)
        
        x = Conv2D(params.conv_2D_2_out_dim, kernel_size=ksize, 
            activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda), 
            data_format='channels_first', strides=(1, params.stride))(x)
        x = MaxPooling2D(pool_size=poolsize, data_format='channels_first')(x)
        x = Dropout(params.conv_2D_2_drop)(x)

        x = Conv2D(params.conv_2D_3_out_dim, kernel_size=ksize, 
            activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda), 
            data_format='channels_first', strides=(1, params.stride))(x)
        x = MaxPooling2D(pool_size=poolsize, data_format='channels_first')(x)
        x = Dropout(params.conv_2D_3_drop)(x)
        
        x = Flatten()(x)

        x = Dense(params.dense_1_dim, activation='relu', 
            kernel_initializer='normal',
            kernel_regularizer=keras.regularizers.l2(l2_lambda))(x)
        x = Dropout(params.dense_1_drop)(x)

        x = Dense(num_buckets)(x)
        x = Activation('softmax', name="N2_branch")(x)

        return x
    
    @staticmethod
    def build_N3_branch(num_buckets, params, inputs):
        l2_lambda = params.l2_lambda
        ksize = (params.k_height, params.k_width)
        poolsize = (params.pool_height, params.pool_width)
     
        x = Conv2D(params.conv_2D_1_out_dim, kernel_size=ksize, 
            activation='relu', 
            kernel_regularizer=keras.regularizers.l2(l2_lambda), 
            data_format='channels_first', strides=(1, params.stride))(inputs)
        x = MaxPooling2D(pool_size=poolsize, data_format='channels_first')(x)
        x = Dropout(params.conv_2D_1_drop)(x)
        
        x = Conv2D(params.conv_2D_2_out_dim, kernel_size=ksize, 
            activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda), 
            data_format='channels_first', strides=(1, params.stride))(x)
        x = MaxPooling2D(pool_size=poolsize, data_format='channels_first')(x)
        x = Dropout(params.conv_2D_2_drop)(x)

        x = Conv2D(params.conv_2D_3_out_dim, kernel_size=ksize, 
            activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda), 
            data_format='channels_first', strides=(1, params.stride))(x)
        x = MaxPooling2D(pool_size=poolsize, data_format='channels_first')(x)
        x = Dropout(params.conv_2D_3_drop)(x)
        
        x = Flatten()(x)

        x = Dense(params.dense_1_dim, activation='relu', 
            kernel_initializer='normal',
            kernel_regularizer=keras.regularizers.l2(l2_lambda))(x)
        x = Dropout(params.dense_1_drop)(x)

        x = Dense(num_buckets)(x)
        x = Activation('softmax', name="N3_branch")(x)

        return x
    
        """
    @staticmethod
    def build_N3_branch(num_buckets, params, dims):
        l2_lambda = params.l2_lambda
        ksize = (params.k_height, params.k_width)
        poolsize = (params.pool_height, params.pool_width)
        model = Sequential()
     
        model.add(Conv2D(params.conv_2D_1_out_dim, kernel_size=ksize, 
            activation='relu',input_shape=dims, 
            kernel_regularizer=keras.regularizers.l2(l2_lambda), 
            data_format='channels_first', strides=(1, params.stride)))
        model.add(MaxPooling2D(pool_size=poolsize, data_format='channels_first'))
        model.add(Dropout(params.conv_2D_1_drop))
        
        model.add(Conv2D(params.conv_2D_2_out_dim, kernel_size=ksize, 
            activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda), 
            data_format='channels_first', strides=(1, params.stride)))
        model.add(MaxPooling2D(pool_size=poolsize, data_format='channels_first'))
        model.add(Dropout(params.conv_2D_2_drop))

        model.add(Conv2D(params.conv_2D_3_out_dim, kernel_size=ksize, 
            activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda), 
            data_format='channels_first', strides=(1, params.stride)))
        model.add(MaxPooling2D(pool_size=poolsize, data_format='channels_first'))
        model.add(Dropout(params.conv_2D_3_drop))    
        
        model.add(Flatten())

        model.add(Dense(params.dense_1_dim, activation='relu', 
            kernel_initializer='normal',
            kernel_regularizer=keras.regularizers.l2(l2_lambda)))
        model.add(Dropout(params.dense_1_drop))

        model.add(Dense(num_buckets))
        model.add(Activation('softmax', name="N3_branch"))

        return model
    """

    @staticmethod
    def build(params, num_buckets):
        NUMEPOCHS = params.epochs
        BATCHSIZE = params.batchsize
        NUMTRAIN = params.total_sims
        CORES = params.cores
        msms_gen = Discrete_Generator(params.num_individuals, params.sequence_length, 
                params.length_to_pad_to, params.pop_min, params.pop_max)
        dims = msms_gen.dim
        
        inputs = Input(shape=dims)

        N1_branch = DiscreteNet.build_N1_branch(num_buckets, params, inputs)
        N2_branch = DiscreteNet.build_N2_branch(num_buckets, params, inputs)
        N3_branch = DiscreteNet.build_N3_branch(num_buckets, params, inputs)


        model = Model(inputs=inputs, outputs=[N1_branch, N2_branch, N3_branch],
                    name="DiscreteNet")

        return model

def late_fork(num_buckets, params):
    l2_lambda = params.l2_lambda
    ksize = (params.k_height, params.k_width)
    poolsize = (params.pool_height, params.pool_width)
    NUMEPOCHS = params.epochs
    BATCHSIZE = params.batchsize
    NUMTRAIN = params.total_sims
    CORES = params.cores
    msms_gen = Discrete_Generator(params.num_individuals, params.sequence_length, 
            params.length_to_pad_to, params.pop_min, params.pop_max)
    dims = msms_gen.dim
    
    inputs = Input(shape=dims)

    
    x = Conv2D(params.conv_2D_1_out_dim, kernel_size=ksize, 
        activation='relu', 
        kernel_regularizer=keras.regularizers.l2(l2_lambda), 
        data_format='channels_first', strides=(1, params.stride))(inputs)
    x = MaxPooling2D(pool_size=poolsize, data_format='channels_first')(x)
    x = Dropout(params.conv_2D_1_drop)(x)
    
    x = Conv2D(params.conv_2D_2_out_dim, kernel_size=ksize, 
        activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda), 
        data_format='channels_first', strides=(1, params.stride))(x)
    x = MaxPooling2D(pool_size=poolsize, data_format='channels_first')(x)
    x = Dropout(params.conv_2D_2_drop)(x)

    x = Conv2D(params.conv_2D_3_out_dim, kernel_size=ksize, 
        activation='relu',kernel_regularizer=keras.regularizers.l2(l2_lambda), 
        data_format='channels_first', strides=(1, params.stride))(x)
    x = MaxPooling2D(pool_size=poolsize, data_format='channels_first')(x)
    x = Dropout(params.conv_2D_3_drop)(x)
    
    x = Flatten()(x)

    N1_branch = Dense(params.dense_1_dim, activation='relu', 
        kernel_initializer='normal',
        kernel_regularizer=keras.regularizers.l2(l2_lambda))(x)
    N1_branch = Dropout(params.dense_1_drop)(N1_branch)
    N1_branch = Dense(num_buckets)(N1_branch)
    N1_branch = Activation('softmax', name="N1_branch")(N1_branch)
    
    N2_branch = Dense(params.dense_1_dim, activation='relu', 
        kernel_initializer='normal',
        kernel_regularizer=keras.regularizers.l2(l2_lambda))(x)
    N2_branch = Dropout(params.dense_1_drop)(N2_branch)
    N2_branch = Dense(num_buckets)(N2_branch)
    N2_branch = Activation('softmax', name="N2_branch")(N2_branch)
    
    N3_branch = Dense(params.dense_1_dim, activation='relu', 
        kernel_initializer='normal',
        kernel_regularizer=keras.regularizers.l2(l2_lambda))(x)
    N3_branch = Dropout(params.dense_1_drop)(N3_branch)
    N3_branch = Dense(num_buckets)(N3_branch)
    N3_branch = Activation('softmax', name="N3_branch")(N3_branch)

        
    model = Model(inputs=inputs, outputs=[N1_branch, N2_branch, N3_branch],
                name="DiscreteNet")

    return model

if __name__ == "__main__":
    from keras.backend import tensorflow_backend
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    if TRAIN:
        num_buckets = sys.argv[1]
        try:
            num_buckets = int(num_buckets)
        except:
            raise RuntimeError("Please use an integer")

        params = utils.Params("./configurations/discrete.json")
        NUMEPOCHS = params.epochs
        BATCHSIZE = params.batchsize
        NUMTRAIN = params.total_sims
        CORES = params.cores
        
        msms_gen = Discrete_Generator(params.num_individuals, params.sequence_length, 
                params.length_to_pad_to, params.pop_min, params.pop_max)
        #model = DiscreteNet.build(params, num_buckets)
        model = late_fork(num_buckets, params) 
        early_stop = EarlyStopping(monitor='loss', min_delta=.05, 
                patience=10, restore_best_weights=True)

        """
        check = ModelCheckpoint("{}_model.hdf5".format(PREFIX), 
                monitor='accuracy', save_best_only=True, 
                save_weights_only=False, period=1)
        """

        losses = {"N1_branch": "categorical_crossentropy",
                    "N2_branch": "categorical_crossentropy", 
                    "N3_branch": "categorical_crossentropy"} 
        loss_weights = {"N1_branch": 1.0,
                        "N2_branch": 1.0, 
                        "N3_branch": 2.0} 
        
        model.compile(loss=losses, loss_weights=loss_weights,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])

        print(model.summary())

        history = model.fit_generator(msms_gen.data_generator(BATCHSIZE, num_buckets), 
                steps_per_epoch=NUMTRAIN/NUMEPOCHS, epochs=NUMEPOCHS, 
                workers=CORES, use_multiprocessing=True, 
                callbacks=[early_stop])   

        model.save(PREFIX + str(num_buckets) + '_model.hdf5')
            
        
    else:
        model = load_model(PREFIX + '_model.hdf5')
        with open("snp_X3k.keras", 'rb') as f:
            X = pickle.load(f)
        with open("snp_y3k.keras", 'rb') as f:
            y = pickle.load(f)

        y_pred = model.predict(X, batch_size=32)
        diff = y_pred - y
        mean_diff = np.mean(diff, axis=0)
        print(mean_diff)
        print(np.mean(mean_diff))
        
        """
        t1 = 0
        t2 = 1786
        t3 = 3571
        x = [t1, t2, t2, t3, t3, 5000]
        y_real = y[0]
        y0 = [y_real[0], y_real[0], y_real[1]/10, y_real[1]/10, y_real[2], y_real[2]]
        y_pred1 = y_pred[0]
        y1 = [y_pred1[0], y_pred1[0], y_pred1[1]/10, y_pred1[1]/10, 
                y_pred1[2], y_pred1[2]]
        plt.plot(x, y0, "r-", label="real")
        plt.plot(x, y1, "b-", label="predicted")

        plt.xlabel("Time (generations)")
        plt.ylabel("Population Scaling Factor (x10,000)")

        plt.title("Prediction vs Truth")
        plt.legend()
        plt.show()
        """
        

