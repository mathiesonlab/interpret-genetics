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

NUMCHANNELS = 2 # Assume to always use both SNP and length matrix
SUMMARYSTATNAMES = 0
SUMMARYSTATVALUES = 1

STATSZI_DIR = "statsZI"

# The stats ZI compilation command
# StatsZI gets compilied on run
STATSZI = "javac -d . -cp JSAP-2.1.jar statistics/*.java utility/*.java; jar cfm statsZI.jar manifest.txt utility/*.class statistics/*.class"
RUNSTATSZI = "java -jar -Xmx5G statsZI.jar"

class MSMS_Generator:
    def __init__(self, num_individuals, sequence_length, length_to_extend_to, 
            total_sims, pop_min, pop_max, T, rho_region, 
            yield_summary_stats=False):
        """
        Params:
            - num_individuals: the number of individuals in the population
            - sequence_length: the length of the sequence to simulate
            - length_to_extend_to: the length to padd the SNP matrix to
            - total_sims: the number of times to run the simulation
        """
        self.num_individuals = num_individuals
        self.sequence_length = sequence_length
        self.length_to_extend_to = length_to_extend_to
        self.total_sims = total_sims
        self.num_channels = NUMCHANNELS
        self.pop_min = pop_min
        self.pop_max = pop_max
        self.T = T
        self.rho_region = rho_region
        self.command = "msms -N %s -ms %s 1 -t %s -r %s %s"
        self.dim = (self.num_channels, self.num_individuals, 
                self.length_to_extend_to)
        self.summary_stats = yield_summary_stats

        if self.summary_stats:
            try:
                with cd(STATSZI_DIR):
                    out = subprocess.check_output(STATSZI, shell=True)
            except:
                raise NameError("Failed compiling statsZI on java compilation.")


    def yield_msms_command_line(self):
        """
        * Yield the next msms command
        """
        while True:
            pop = np.random.randint(self.pop_min, self.pop_max)
            command = (self.command % (pop, self.num_individuals, self.T,
                         self.rho_region, self.sequence_length))
            yield (command, pop)

    def data_generator(self, batch_size):
        """
        * This generator is used to yield generator data for keras fit_generator
          functionality.  Note, generation is done on the fly, so it is not
          necessary to use additional sequence functionality.
        * Parameters:
            - batch_size: the size of the batch
            - dim: the input dimension of the dataset
        * Out:
            - Yields another batch of msms data (X,y) if self.summary_stats is off
              otherwise, yields X, y, summary_stats dictionary if self.summary_stats
              is true
        """

        # It doesn't make much sense to allow for summary statistics but have a batch size
        # larger than 1. We will only yield the batch every b iterations, where b is batch
        # size but the summary stats will just be from one generation
        if self.summary_stats and batch_size > 1:
            raise UserWarning("Summary stats set to True but batch size is not 1.  " + \
                "Summary stats file will only include the last item in the batch.")

        X = np.empty((batch_size, *self.dim))
        y = np.empty((batch_size), dtype=int)
        for i, tup in enumerate(self.yield_msms_command_line()):
            out = subprocess.Popen(tup[0].split(), stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT)
            stdout, stderr = out.communicate()
            stdout = str(stdout, 'utf-8')
            buf = io.StringIO(stdout)

            summary_stat_dictionary = {}
            if self.summary_stats:
                temporary_file_name = os.path.join(STATSZI_DIR, 
                        "current_msms.msms")
                summary_stats_name = os.path.join(STATSZI_DIR, 
                        "stats_current_stats.txt")

                with open(temporary_file_name, "w+") as f:
                    f.write(stdout)

                # this changes the directory and then resets once it gets outside the scope
                with cd(STATSZI_DIR):
                    subprocess.check_output(RUNSTATSZI, shell=True)
                summary_stat_dictionary = self.get_summary_stats(summary_stats_name)

                os.remove(temporary_file_name)
                os.remove(summary_stats_name)

            X1, X2 = self.read_files(buf.readlines(), None, None, string=True)
            X[i % batch_size] = X1
            y[i % batch_size] = tup[1]
            if i % batch_size == batch_size-1:

                if not self.summary_stats:
                    yield X, y
                else:
                    yield X, y, summary_stat_dictionary

                # reset the batch
                X = np.empty((batch_size, *self.dim))
                y = np.empty((batch_size), dtype=int)


    def read_files(self, f, strengths_dict, sim_dir, string=False):
        """
        * PReads files and handles SNP generation
        * Attribution: Nhung
        """

        xSNPs = []
        xTMRCAs = []
        xPositions = []
        yNS_strengths = []

        strength, SNPs, positions, selection = \
            self.parse_sim(f,self.num_individuals,self.sequence_length,
                    strengths_dict, y_indx = 2, string = True)
        SNP_lengths = [mat.shape[1] for mat in SNPs]
        SNPs_matrices,  position_matrices = self.centered_padding(SNPs,
                positions,self.length_to_extend_to)
        xSNPs.extend(SNPs_matrices)
        xPositions.extend(position_matrices)

        xSNPs = np.array(xSNPs)
        xPositions = np.array(xPositions)
        yNS_strengths = np.array(yNS_strengths)
        dims = xSNPs.shape

        # 2 channel input
        snp_pos = np.concatenate((np.reshape(xSNPs,(dims[0],1,dims[1],dims[2])),
                            np.reshape(xPositions,(dims[0],1,dims[1],dims[2])),
                            ), axis=1)

        return snp_pos, selection

    def parse_sim(self, filename, n, L, strengths_dict, y_indx = None, 
            verbose = False, string=False):
        """
        * Parses the file and handles SNP matrix generation appropriately
        * Attribution: Nhung
        """

        SNPs_matrices = [] # each element is an n by [num sites] SNPs matrix
        position_matrices = [] # each element is an n by [num sites] matrix of the corresponding [num sites] positions PLUS an end zero column
        num_sites_list = [] # corresponds to num sites per position vector
        count = 0 # to track parsing progress

        ns_strength  = None
        if verbose:
            print("nat sel data: parsing SNPs matrices...")

        if not string:
            file_ = open(filename,'r')
            lines = file_.readlines()
        else:
            lines = filename

        ### This just provides a way to extract a potential y from the msms outputs if it's inluded in a certain spot in the msms output
        if y_indx:
            selection = float(lines[0].split()[y_indx])
        else:
            selection = None

        for i in range(3,len(lines),n+4): # excluding header, moving through n SNP seqs
            assert(lines[i].strip()=='//')
            num_sites = int(lines[i+1].split(' ')[1])
            num_sites_list.append(num_sites)
            if num_sites == 0: # no seg sites
                    position_matrices.append(np.zeros((n,0), dtype='int32'))
                    SNPs_matrices.append(np.zeros((n,0), dtype='int32'))
            else:
                    positions = lines[i+2].strip().split(' ')
                    position_vector = [int(float(p)*L) for p in positions[1:]]
                    position_vector = self.check_positions(position_vector)
                    position_dists = self.position_distances(position_vector)
                    position_matrix = np.tile(position_dists,(n,1))
                    position_matrices.append(position_matrix)
                    SNPs = []
                    for j in range(n):
                            str_SNPs = lines[i+3+j].strip()
                            int_SNPs = np.array(list(str_SNPs), dtype='int32')
                            SNPs.append(int_SNPs)
                    SNPs = np.array(SNPs)
                    assert(SNPs[0].shape[0]==num_sites==position_matrix[0].shape[0])
                    assert(SNPs.shape[0]==n==position_matrix.shape[0])
                    SNPs_matrices.append(SNPs)
            count += 1
            if count%100==0 and verbose: print(count)
        if not string:
            file_.close()
        if verbose:
                print("nat sel data: stats")
                print("min sites:",min(num_sites_list))
                print("max sites:",max(num_sites_list))
                print("avg sites:",sum(num_sites_list)/len(num_sites_list))
        over350 = 0
        for s in num_sites_list:
            if s > 350: over350 += 1
        if verbose:
            print("over 350 sites:", over350)

        return ns_strength, SNPs_matrices, position_matrices, selection

    def centered_padding(self, SNPs_matrices, position_matrices, length, 
            verbose = False):
        """
        * Pads the matrix appropriately.  This is needed because the matrices might not all
          be the same length
        * Attribution: Nhung
        """

        if verbose:
             print("nat sel data: padding sequence matrices...")

        count = 0
        uniform_SNPs = []
        uniform_TMRCAs = []
        uniform_pos_vecs = []
        for i in range(len(SNPs_matrices)):
            snp = SNPs_matrices[i]
            # tmrca = TMRCAs_matrices[i]
            position_matrix = position_matrices[i]
            h, w = snp.shape
            if w >= length:
                reduced_snp = snp[:,:length]
                uniform_SNPs.append(reduced_snp)
                # reduced_tmrca = tmrca[:,:length]
                # uniform_TMRCAs.append(reduced_tmrca)
                reduced_pos_vec = position_matrix[:,:length]
                uniform_pos_vecs.append(reduced_pos_vec)
            else:
                padding_width = length-w
                zeros = np.zeros((h,padding_width), dtype='int32')
                half = int(padding_width/2)
                padded_snp = np.concatenate((zeros[:,:half],snp,zeros[:,half:]),
                        axis=1)
                uniform_SNPs.append(padded_snp)
                # padded_tmrca = np.concatenate((zeros[:,:half],tmrca,zeros[:,half:]),axis=1)
                # uniform_TMRCAs.append(padded_tmrca)
                padded_pos_vec = np.concatenate((zeros[:,:half],position_matrix,
                    zeros[:,half:]),axis=1)
                uniform_pos_vecs.append(padded_pos_vec)
            if count%100==0 and verbose: print(count)
            count += 1
        # return uniform_SNPs, uniform_TMRCAs, uniform_pos_vecs
        return uniform_SNPs, uniform_pos_vecs

    def check_positions(self, position_vector):
        """
        * Rare, but if two mutations are really close to each other,
          they could be settling into the same position number. check
          for that and separate them by a value of 1
        * Attribution: Nhung
        """
        for p in range(len(position_vector)-1):
            while position_vector[p] >= position_vector[p+1]:
                position_vector[p+1] += 1
        return position_vector

    def position_distances(self, position_vector):
        """
        * Calculate distance between site positions
        * Attribution: Nhung
        """
        distances = []
        for i in range(len(position_vector)-1):
            distance = position_vector[i+1] - position_vector[i]
            distances.append(distance)
        distances.append(0) # last column zero padding
        return np.array(distances)

def get_num(string):
    r = ''
    for i in string:
        if i in "0123456789":
            r += i
    return int(r)

def build_summary_stats_dataset(datasetname, files_dir):
    for root, dirs, files in os.walk(files_dir, topdown=False):
       for name in files:
           yield get_summary_stats(os.path.join(root, name)), get_num(name)


def get_summary_stats(stats_file_name):
    """
    * This routine gets the summary statistics from a data_frame
      Unfortunately, we have to write the file temporarility to storage
      in order to be able to call it in the java routine.
    * Paramerters: msms_data, this is a formatted msms

    """
    with open(stats_file_name) as f:
        stat_data = f.read()
        split_data = stat_data.split("\n")

        names = split_data[SUMMARYSTATNAMES].split(" ")
        values = split_data[SUMMARYSTATVALUES].split(" ")

        summary_stat_dictionary = {names[i]: values[i] for i in range(len(names))}

    return summary_stat_dictionary

@contextmanager
def cd(newdir):
    """
    * A helper function used in the running of the statsZI to change directory
    * This comes from: http://bit.ly/2ZT4KeZ
    """
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

if __name__ == "__main__":
    msms_gen = MSMS_Generator(10, 1000000, 1500, 10000, 10000, 1000000, 50, 50, 
            yield_summary_stats = True)
    i = 0
    for X,y,stats_dict in msms_gen.data_generator(1):
        print (stats_dict)
        if i >= 1:
            break
        i += 1
