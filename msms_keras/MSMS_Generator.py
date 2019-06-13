"""
* This serves as core msms generator routine.
"""
import time
import numpy as np
import subprocess
import sys
import h5py
import os
import pandas as pd
import io
import pickle
from contextlib import contextmanager
import msprime
import random
import libsequence  

NUMCHANNELS = 2 # Assume to always use both SNP and length matrix

class MSMS_Generator:
    def __init__(self, num_individuals, sequence_length, length_to_extend_to, 
            pop_min, pop_max, yield_summary_stats=0):
        """
        Params:
            - num_individuals: the number of individuals in the population
            - sequence_length: the length of the sequence to simulate
            - length_to_extend_to: the length to padd the SNP matrix to
        """
        self.num_individuals = num_individuals
        self.sequence_length = sequence_length
        self.length_to_extend_to = length_to_extend_to
        self.num_channels = NUMCHANNELS
        self.pop_min = pop_min
        self.pop_max = pop_max
        self.dim = (self.num_channels, self.length_to_extend_to, 
                self.num_individuals)
        self.summary_stats = yield_summary_stats


    def data_generator(self, batch_size):
        
        if self.summary_stats == 0: 
            while True:
                X = np.empty((batch_size, *self.dim))
                y = np.empty((batch_size), dtype=int)
                
                for i in range(batch_size):
                    pop_size = random.randrange(self.pop_min, self.pop_max)
                    tree_sequence = msprime.simulate(sample_size=self.num_individuals,
                            Ne=pop_size, length=self.sequence_length, 
                            recombination_rate=1e-8, mutation_rate=1e-8)

                    genotype = tree_sequence.genotype_matrix().astype(int)
                    genotype_padded = self.centered_padding(genotype)
                    X[i][0] = genotype_padded

                    variant_iter = tree_sequence.variants()
                    first = next(variant_iter)
                    prev_pos = first.site.position
                    pos_distances = [0]
                    for variant in variant_iter:
                        pos = variant.site.position
                        pos_distances.append(int(pos)-int(prev_pos))
                        prev_pos = pos
                    
                    distance_matrix = np.array(pos_distances)
                    distance_matrix = np.reshape(distance_matrix, (len(pos_distances), 1))
                    distance_matrix = np.tile(distance_matrix, (1, self.num_individuals))
                    distance_padded = self.centered_padding(distance_matrix)
                    X[i][1] = distance_padded
                    
                    y[i] = pop_size 
                    
                yield X, y

        elif self.summary_stats == 1:
            pass
            
    def centered_padding(self, matrix):
        
        diff = self.length_to_extend_to - matrix.shape[0]
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



   
if __name__ == "__main__":
    msms_gen = MSMS_Generator(10, 1000000, 800, 1000, 10000)
    generator = msms_gen.data_generator(8)
    X, y = next(generator)
    print(X.shape)
    print(y)
