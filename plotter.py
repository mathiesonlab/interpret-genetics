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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import pickle

from msms_keras.MSMS_Generator import MSMS_Generator, MSprime_Generator
import utils

import tensorflow as tf

from run_example import rmse, MetricHistory
from range_shift import range_shift

def main():

    cnn_model = load_model('models/range_shift_model.hdf5', custom_objects={"rmse": rmse,
                                "range_shift": range_shift})
    fc_model = load_model('models/sumstats_model.hdf5', custom_objects={"rmse": rmse})
     
    msms_gen = MSprime_Generator(10, 1000000, 8000, 1000, 10000, yield_summary_stats=2)
    gen = msms_gen.data_generator(1)
    X, matrix, y_real = next(gen)
    #y_real = np.mean(y_real, axis=0)

    cnn_pred = cnn_model.predict(X, batch_size=1)
    cnn_pred = np.mean(cnn_pred, axis=0)
    #cnn_pred = cnn_pred.reshape((3,))
    fc_pred = fc_model.predict(matrix, batch_size=1)
    #fc_pred = np.mean(fc_pred, axis=0)
    fc_pred = fc_pred.reshape((3,))

    t1 = 0
    t2 = 1786
    t3 = 3571
    x = [t1, t2, t2, t3, t3, 5000]

    y_real = y_real.reshape((3,))
    y0 = [y_real[0], y_real[0], y_real[1]/10, y_real[1]/10, y_real[2], y_real[2]]
    
    y1 = [cnn_pred[0], cnn_pred[0], cnn_pred[1]/10, cnn_pred[1]/10, 
            cnn_pred[2], cnn_pred[2]]
    
    y2 = [fc_pred[0], fc_pred[0], fc_pred[1]/10, fc_pred[1]/10, 
            fc_pred[2], fc_pred[2]]
    

    plt.plot(x, y0, "r-", label="real")
    plt.plot(x, y1, "b-", label="cnn prediction")
    plt.plot(x, y2, "g-", label="fc prediction")

    plt.xlabel("Time (generations)")
    plt.ylabel("Population Scaling Factor (x10,000)")

    plt.title("Prediction vs Truth")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()
