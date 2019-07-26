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
import math
import pickle

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

class OnePop_Generator:
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
        
        while True:
            y = np.empty((batch_size), dtype=float)
            X = np.empty((batch_size, *self.dim))

            for i in range(batch_size):
                
                pop = np.random.choice(np.linspace(1, 10, 10000))

                tree_sequence = msprime.simulate(sample_size=self.num_individuals, 
                        length=self.sequence_length,
                        Ne=pop*self.pop_max,
                        recombination_rate=1e-8, mutation_rate=1e-8)

                genotype = tree_sequence.genotype_matrix().astype(int)
                genotype_padded = self.centered_padding(genotype)
                y[i] = pop
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

            if self.summary_stats == 0:
                
                yield X, y

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


class MSprime_Generator:
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
        
        while True:
            y = np.empty((batch_size, 3), dtype=float)
            X = np.empty((batch_size, *self.dim))
            s = []

            for i in range(batch_size):
                
                N1 = np.random.choice(np.linspace(1, 10, 10000))
                N2 = np.random.choice(np.linspace(1, 10, 10000))
                N3 = np.random.choice(np.linspace(1, 10, 10000))
                
                demo1 = msprime.PopulationParametersChange(time=0, 
                        initial_size=N1 * self.pop_max)
                demo2 = msprime.PopulationParametersChange(time=1786, 
                        initial_size=N2 * self.pop_min)
                demo3 = msprime.PopulationParametersChange(time=3571, 
                        initial_size=N3 * self.pop_max)
                demos = [demo1, demo2, demo3]

                tree_sequence = msprime.simulate(sample_size=self.num_individuals, 
                        length=self.sequence_length,
                        Ne=10000,
                        recombination_rate=1e-8, mutation_rate=1e-8,
                        demographic_events=demos)

                genotype = tree_sequence.genotype_matrix().astype(int)
                s.append(genotype.shape[0])
                genotype_padded = self.centered_padding(genotype)
                y[i] = np.array([N1, N2, N3])
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

            if self.summary_stats == 0:
                
                """
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
                """

                yield X, y

            elif self.summary_stats == 1 or self.summary_stats == 2:

                X_new = X[:,0]
                SFS = []

                for batch in range(batch_size):
                    lst = [0 for i in range(self.num_individuals)]
                    for i in range(X_new.shape[1]):
                        count = 0
                        for j in range(X_new.shape[2]):
                            if X_new[batch, i, j] == 1:
                                count += 1
                        lst[count] += 1

                    lst.pop(0)
                    SFS.append(lst)
                
                SFS_folded = [] 
                for i in range(len(SFS)):
                    lst = []
                    for j in range(math.ceil(len(SFS[i])/2)):
                        if j == len(SFS[i]) - 1 - j:
                            lst.append(SFS[i][j])
                        else:
                            n = SFS[i][j] + SFS[i][len(SFS[i]) - 1 - j]
                            lst.append(n)
                    SFS_folded.append(lst)

                pi_lst = []
                for i in range(len(SFS_folded)):
                    pi = 0
                    for j in range(len(SFS_folded[i])):
                        pi += SFS_folded[i][j] * (j+1) * (self.num_individuals-(j+1))
                    pi *= 1 / (self.num_individuals * (self.num_individuals-1)/2)
                    pi_lst.append(pi)
                
                a1 = 0
                for i in range(1, self.num_individuals):
                    a1 += 1/i

                tajd_lst = []
                for i in range(len(pi_lst)):
                    tajd_lst.append(pi_lst[i] - (s[i]/a1))

                matrix = []
                for i in range(batch_size):
                    element = [s[i], pi_lst[i], *SFS_folded[i], tajd_lst[i]]
                    matrix.append(element)
                matrix = np.array(matrix, dtype=float)
                
                if self.summary_stats == 1:
                    yield matrix, y

                
            if self.summary_stats == 2:
                yield X, matrix, y


                

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

class Schrider_Generator:
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
        self.pop_min = pop_min
        self.pop_max = pop_max
        self.dim = (1, self.length_to_extend_to, self.num_individuals)
        self.summary_stats = yield_summary_stats


    def data_generator(self, batch_size):
        
        while True:
            y = np.empty((batch_size, 3), dtype=float)
            X1 = np.empty((batch_size, *self.dim))
            X2 = np.empty((batch_size, *self.dim))
            s = []

            for i in range(batch_size):
                
                N1 = np.random.choice(np.linspace(1, 10, 10000))
                N2 = np.random.choice(np.linspace(1, 10, 10000))
                N3 = np.random.choice(np.linspace(1, 10, 10000))
                
                demo1 = msprime.PopulationParametersChange(time=0, 
                        initial_size=N1 * self.pop_max)
                demo2 = msprime.PopulationParametersChange(time=1786, 
                        initial_size=N2 * self.pop_min)
                demo3 = msprime.PopulationParametersChange(time=3571, 
                        initial_size=N3 * self.pop_max)
                demos = [demo1, demo2, demo3]

                tree_sequence = msprime.simulate(sample_size=self.num_individuals, 
                        length=self.sequence_length,
                        Ne=10000,
                        recombination_rate=1e-8, mutation_rate=1e-8,
                        demographic_events=demos)

                genotype = tree_sequence.genotype_matrix().astype(int)
                s.append(genotype.shape[0])
                genotype_padded = self.centered_padding(genotype)
                y[i] = np.array([N1, N2, N3])
                X1[i][0] = genotype_padded
                
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
                X2[i][0] = distance_padded

            if self.summary_stats == 0:
                
                """
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
                """

                yield ([X1, X2], y)

            elif self.summary_stats == 1 or self.summary_stats == 2:

                X_new = X[:,0]
                SFS = []

                for batch in range(batch_size):
                    lst = [0 for i in range(self.num_individuals)]
                    for i in range(X_new.shape[1]):
                        count = 0
                        for j in range(X_new.shape[2]):
                            if X_new[batch, i, j] == 1:
                                count += 1
                        lst[count] += 1

                    lst.pop(0)
                    SFS.append(lst)
                
                SFS_folded = [] 
                for i in range(len(SFS)):
                    lst = []
                    for j in range(math.ceil(len(SFS[i])/2)):
                        if j == len(SFS[i]) - 1 - j:
                            lst.append(SFS[i][j])
                        else:
                            n = SFS[i][j] + SFS[i][len(SFS[i]) - 1 - j]
                            lst.append(n)
                    SFS_folded.append(lst)

                pi_lst = []
                for i in range(len(SFS_folded)):
                    pi = 0
                    for j in range(len(SFS_folded[i])):
                        pi += SFS_folded[i][j] * (j+1) * (self.num_individuals-(j+1))
                    pi *= 1 / (self.num_individuals * (self.num_individuals-1)/2)
                    pi_lst.append(pi)
                
                a1 = 0
                for i in range(1, self.num_individuals):
                    a1 += 1/i

                tajd_lst = []
                for i in range(len(pi_lst)):
                    tajd_lst.append(pi_lst[i] - (s[i]/a1))

                matrix = []
                for i in range(batch_size):
                    element = [s[i], pi_lst[i], *SFS_folded[i], tajd_lst[i]]
                    matrix.append(element)
                matrix = np.array(matrix, dtype=float)
                
                if self.summary_stats == 1:
                    yield matrix, y

                
            if self.summary_stats == 2:
                yield X, matrix, y

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


class Discrete_Generator:
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


    def data_generator(self, batch_size, num_buckets):
        
        while True:
            X = np.empty((batch_size, *self.dim))
            s = []
            N1_lst = []
            N2_lst = []
            N3_lst = []
            for i in range(batch_size):
                
                N1 = np.random.choice(np.linspace(1, 10, 10000))
                N2 = np.random.choice(np.linspace(1, 10, 10000))
                N3 = np.random.choice(np.linspace(1, 10, 10000))
                
                demo1 = msprime.PopulationParametersChange(time=0, 
                        initial_size=N1 * self.pop_max)
                demo2 = msprime.PopulationParametersChange(time=1786, 
                        initial_size=N2 * self.pop_min)
                demo3 = msprime.PopulationParametersChange(time=3571, 
                        initial_size=N3 * self.pop_max)
                demos = [demo1, demo2, demo3]

                tree_sequence = msprime.simulate(sample_size=self.num_individuals, 
                        length=self.sequence_length,
                        Ne=10000,
                        recombination_rate=1e-8, mutation_rate=1e-8,
                        demographic_events=demos)

                genotype = tree_sequence.genotype_matrix().astype(int)
                s.append(genotype.shape[0])
                genotype_padded = self.centered_padding(genotype)
                
                N1 = self.discretize(float(N1), num_buckets)
                N2 = self.discretize(float(N2), num_buckets)
                N3 = self.discretize(float(N3), num_buckets)
                N1_lst.append(N1)
                N2_lst.append(N2)
                N3_lst.append(N3)

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

            if self.summary_stats == 0:
                
                """
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
                """
                N1_array = np.array(N1_lst)
                N2_array = np.array(N2_lst)
                N3_array = np.array(N3_lst)
                yield (X, {"N1_branch": N1_array, "N2_branch": N2_array, 
                    "N3_branch": N3_array})

            elif self.summary_stats == 1 or self.summary_stats == 2:

                X_new = X[:,0]
                SFS = []

                for batch in range(batch_size):
                    lst = [0 for i in range(self.num_individuals)]
                    for i in range(X_new.shape[1]):
                        count = 0
                        for j in range(X_new.shape[2]):
                            if X_new[batch, i, j] == 1:
                                count += 1
                        lst[count] += 1

                    lst.pop(0)
                    SFS.append(lst)
                
                SFS_folded = [] 
                for i in range(len(SFS)):
                    lst = []
                    for j in range(math.ceil(len(SFS[i])/2)):
                        if j == len(SFS[i]) - 1 - j:
                            lst.append(SFS[i][j])
                        else:
                            n = SFS[i][j] + SFS[i][len(SFS[i]) - 1 - j]
                            lst.append(n)
                    SFS_folded.append(lst)

                pi_lst = []
                for i in range(len(SFS_folded)):
                    pi = 0
                    for j in range(len(SFS_folded[i])):
                        pi += SFS_folded[i][j] * (j+1) * (self.num_individuals-(j+1))
                    pi *= 1 / (self.num_individuals * (self.num_individuals-1)/2)
                    pi_lst.append(pi)
                
                a1 = 0
                for i in range(1, self.num_individuals):
                    a1 += 1/i

                tajd_lst = []
                for i in range(len(pi_lst)):
                    tajd_lst.append(pi_lst[i] - (s[i]/a1))

                matrix = []
                for i in range(batch_size):
                    element = [s[i], pi_lst[i], *SFS_folded[i], tajd_lst[i]]
                    matrix.append(element)
                matrix = np.array(matrix, dtype=float)
                
                if self.summary_stats == 1:
                    yield matrix, y

                
            if self.summary_stats == 2:
                yield X, matrix, y


    def discretize(self, N, num_buckets):
        bucket_range = 9 / num_buckets
        diff = N - 1
        lst = [0] * num_buckets
        try:
            lst[int(diff / bucket_range)] = 1
        except:
            lst[-1] = 1
        
        return lst 

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

    """
    msms_gen = MSprime_Generator(10, 1000000, 8000, 1000, 10000, yield_summary_stats=0)
    generator = msms_gen.data_generator(1)
    X_lst = []
    y_lst = []
    for i in range(10000):
        X, y = next(generator)
        X_lst.append(X)
        y_lst.append(y)
    
    X = np.concatenate(X_lst, axis=0)
    y = np.concatenate(y_lst, axis=0)
    with open("snp_X.keras", 'wb') as f:
        pickle.dump(X, f)
    with open("snp_y.keras", 'wb') as f:
        pickle.dump(y, f)
    
    with open("snp_X.keras", 'rb') as f:
        X = pickle.load(f)
    with open("snp_y.keras", 'rb') as f:
        y = pickle.load(f)

    print(X.shape)
    print(y.shape)
    
    msms_gen = MSprime_Generator(10, 1000000, 8000, 1000, 10000, yield_summary_stats=1)
    generator = msms_gen.data_generator(1)
    X_lst = []
    y_lst = []
    for i in range(10000):
        X, y = next(generator)
        X_lst.append(X)
        y_lst.append(y)
    
    X = np.concatenate(X_lst, axis=0)
    y = np.concatenate(y_lst, axis=0)
    with open("sumstats_X.keras", 'wb') as f:
        pickle.dump(X, f)
    with open("sumstats_y.keras", 'wb') as f:
        pickle.dump(y, f)
    
    with open("sumstats_X.keras", 'rb') as f:
        X = pickle.load(f)
    with open("sumstats_y.keras", 'rb') as f:
        y = pickle.load(f)

    print(X.shape)
    print(y.shape)

    msms_gen = MSprime_Generator(10, 1000000, 8000, 1000, 10000, yield_summary_stats=1)
    gen = msms_gen.data_generator(8)
    X, y = next(gen)
    print(X)

    """
    """
    msms_gen = Discrete_Generator(10, 1000000, 8000, 1000, 10000)
    generator = msms_gen.data_generator(3000, 3)
    for i in range(1000):
        X, y = next(generator)
    
    with open("discrete3_X.keras", 'wb') as f:
        pickle.dump(X, f)
    with open("discrete3_y.keras", 'wb') as f:
        pickle.dump(y, f)
    
    with open("discrete3_X.keras", 'rb') as f:
        X = pickle.load(f)
    with open("discrete3_y.keras", 'rb') as f:
        y = pickle.load(f)

    print(X.shape)
    print(y)    
    
    msms_gen = Discrete_Generator(10, 1000000, 8000, 1000, 10000)
    generator = msms_gen.data_generator(3000, 5)
    for i in range(1000):
        X, y = next(generator)
    
    with open("discrete5_X.keras", 'wb') as f:
        pickle.dump(X, f)
    with open("discrete5_y.keras", 'wb') as f:
        pickle.dump(y, f)
    
    with open("discrete5_X.keras", 'rb') as f:
        X = pickle.load(f)
    with open("discrete5_y.keras", 'rb') as f:
        y = pickle.load(f)

    print(X.shape)
    print(y)

    """
    msms_gen = MSprime_Generator(10, 1000000, 8000, 1000, 10000, yield_summary_stats=0)
    generator = msms_gen.data_generator(3000)
    
    X, y = next(generator)
    
    with open("snp_X_test.keras", 'wb') as f:
        pickle.dump(X, f)
    with open("snp_y_test.keras", 'wb') as f:
        pickle.dump(y, f)
    
    with open("snp_X_test.keras", 'rb') as f:
        X = pickle.load(f)
    with open("snp_y_test.keras", 'rb') as f:
        y = pickle.load(f)

    print(X.shape)
    print(y.shape)
 
