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

from sklearn.ensemble import GradientBoostingRegressor

import pickle

#from msms_keras.MSMS_Generator import MSMS_Generator
import utils

import shap

def main():
    with h5py.File("test.h5py", "r") as f:
        y = f["y_test"]
        w = y[:100]
        X = pd.read_csv("summary_stats.csv")
        Z = X.values[:100] 

        model = GradientBoostingRegressor(n_estimators=25, learning_rate=.4)

        model.fit(Z, w)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X)
main()
