import vcf

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
from run_example import rmse, MetricHistory
from range_shift import range_shift

import tensorflow as tf


def main():
    
    #reader = vcf.Reader(open('/home/smathieson/public/cs68/1000g/EUR_135-136Mb.chr2.vcf', 'r'))
    reader = vcf.Reader(filename="EUR.chr2.vcf.gz")
    X = np.empty((2, 8000, 10), dtype=int) 
    X_lst = []
    stat_lst = []
    matrix = []
    distances = []
    record = next(reader)
    prev = record.POS
    #reader = vcf.Reader(open('/home/smathieson/public/cs68/1000g/EUR_135-136Mb.chr2.vcf', 'r'))
    reader = vcf.Reader(filename="EUR.chr2.vcf.gz")
    breakpoint = prev + 1000000
    count = 0
    for record in reader:
        if count >= 16:
            break
        lst = []
        for i, sample in enumerate(record.samples):
            if i >= 5:
                break
            lst.append(int(sample['GT'][0]))
            lst.append(int(sample['GT'][2]))

        if lst.count(0) != 10 and lst.count(1) != 10:
            matrix.append(lst)
            dist = record.POS - prev
            distances.append(dist)
            prev = record.POS
        
        if prev > breakpoint:
            count += 1

            matrix = np.array(matrix)
            matrix_padded = centered_padding(matrix, 8000)
            X[0] = matrix_padded  

            distance_matrix = np.array(distances)
            distance_matrix = distance_matrix.reshape((len(distances), 1))
            distance_matrix = np.tile(distance_matrix, (1, 10))
            distances_padded = centered_padding(distance_matrix, 8000)
            X[1] = distances_padded 
            X_lst.append(X)

            summstats = summary_stats(matrix)
            summstats = summstats.reshape((1, 8))
            stat_lst.append(summstats)
            
            X = np.empty((2, 8000, 10), dtype=int)
            matrix = []
            distances = []
            breakpoint = prev + 1000000

    X = np.stack(X_lst)
    summstats = np.concatenate(stat_lst)

    cnn_model = load_model('models/range_shift_model.hdf5', custom_objects={"rmse": rmse, 
                        "range_shift": range_shift})
    fc_model = load_model('models/sumstats_model.hdf5', custom_objects={"rmse": rmse})
    
    cnn_pred = cnn_model.predict(X, batch_size=16)
    fc_pred = fc_model.predict(summstats, batch_size=16)
    
    t1 = 1786
    t2 = 3571
    x = [0, t1, t1, t2, t2, 5000]
    range1 = [0, t1]
    range2 = [t1, t2]
    range3 = [t2, 5000] 
    
    plt.figure()
    plt.plot(range1, [1,1], "k-", linewidth=3)
    plt.plot(range1, [10,10], "k-", linewidth=3)
    plt.plot(range2, [1/10,1/10], "k-", linewidth=3)
    plt.plot(range2, [1,1], "k-", linewidth=3)
    plt.plot(range3, [1,1], "k-", linewidth=3)
    plt.plot(range3, [10,10], "k-", linewidth=3)
   
    cnn_mean = np.mean(cnn_pred, axis=0).reshape(3,)
    fc_mean = np.mean(fc_pred, axis=0).reshape(3,)

    y1 = [cnn_mean[0], cnn_mean[0], cnn_mean[1]/10, cnn_mean[1]/10, 
            cnn_mean[2], cnn_mean[2]]
    
    y2 = [fc_mean[0], fc_mean[0], fc_mean[1]/10, fc_mean[1]/10, 
            fc_mean[2], fc_mean[2]]
    
    plt.plot(x, y1, "b-", label="cnn prediction", linewidth=.5)
    plt.plot(x, y2, "g-", label="fc prediction", linewidth=.5)
    
 
    plt.xlabel("Time (generations)")
    plt.ylabel("Population Scaling Factor (x10,000)")

    plt.title("Real Data (CNN mean vs FC mean)")
    plt.legend()
    plt.savefig("mean.png")
    
    plt.figure()
    for i in range(cnn_pred.shape[0]):
        y1 = [cnn_pred[i][0], cnn_pred[i][0], cnn_pred[i][1]/10, cnn_pred[i][1]/10, 
            cnn_pred[i][2], cnn_pred[i][2]]
        plt.plot(x, y1, linewidth=.5)


    plt.plot(range1, [1,1], "k-", linewidth=3)
    plt.plot(range1, [10,10], "k-", linewidth=3)
    plt.plot(range2, [1/10,1/10], "k-", linewidth=3)
    plt.plot(range2, [1,1], "k-", linewidth=3)
    plt.plot(range3, [1,1], "k-", linewidth=3)
    plt.plot(range3, [10,10], "k-", linewidth=3)
 
    plt.xlabel("Time (generations)")
    plt.ylabel("Population Scaling Factor (x10,000)")

    plt.title("Real Data (CNN)")
    plt.savefig("cnn.png")
 
    plt.figure()
    for i in range(cnn_pred.shape[0]):
        y1 = [fc_pred[i][0], fc_pred[i][0], fc_pred[i][1]/10, fc_pred[i][1]/10, 
            fc_pred[i][2], fc_pred[i][2]]
        plt.plot(x, y1, linewidth=.5)

    plt.plot(range1, [1,1], "k-", linewidth=3)
    plt.plot(range1, [10,10], "k-", linewidth=3)
    plt.plot(range2, [1/10,1/10], "k-", linewidth=3)
    plt.plot(range2, [1,1], "k-", linewidth=3)
    plt.plot(range3, [1,1], "k-", linewidth=3)
    plt.plot(range3, [10,10], "k-", linewidth=3)
 
    plt.xlabel("Time (generations)")
    plt.ylabel("Population Scaling Factor (x10,000)")

    plt.title("Real Data (FC)")
    plt.savefig("fc.png")

def centered_padding(matrix, length_to_extend_to):
    
    diff = length_to_extend_to - matrix.shape[0]
    if diff >= 0:
        if diff % 2 == 0:
            zero1 = np.zeros((diff//2, matrix.shape[1]))
            zero2 = np.zeros((diff//2, matrix.shape[1]))
        else:
            zero1 = np.zeros((diff//2 + 1, matrix.shape[1]))
            zero2 = np.zeros((diff//2, matrix.shape[1]))
        return np.concatenate((zero1, matrix, zero2), axis=0)
    else:
        diff *= -1
        if diff % 2 == 0:
            arr = np.delete(matrix, range(diff//2), axis=0)
            arr = np.delete(arr, range(arr.shape[0] - diff//2, arr.shape[0]),
                    axis=0)
        else:
            arr = np.delete(matrix, range(diff//2), axis=0)
            arr = np.delete(arr, range(arr.shape[0] - (diff//2 + 1), arr.shape[0]), 
                    axis=0)
        return arr

def summary_stats(array):
    
    s = array.shape[0]
    X_new = array 

    lst = [0 for i in range(10)]
    for i in range(X_new.shape[0]):
        count = 0
        for j in range(X_new.shape[1]):
            if X_new[i, j] == 1:
                count += 1
        lst[count] += 1

    lst.pop(0)
    SFS = lst[:]

    lst = []
    for j in range(math.ceil(len(SFS)/2)):
        if j == len(SFS) - 1 - j:
            lst.append(SFS[j])
        else:
            n = SFS[j] + SFS[len(SFS) - 1 - j]
            lst.append(n)

    SFS_folded = lst[:]

    pi = 0
    for j in range(len(SFS_folded)):
        pi += SFS_folded[j] * (j+1) * (10-(j+1))
    pi *= 1 / (10 * (10-1)/2)
    
    a1 = 0
    for i in range(1, 10):
        a1 += 1/i

    tajd = pi - s/a1

    element = [s, pi, *SFS_folded, tajd]
    matrix = np.array(element)

    return matrix


main()
