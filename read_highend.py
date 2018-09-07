import sys, os

import pandas as pd
import numpy as np

import scipy.signal

import matplotlib as mpl
import matplotlib.pyplot as plt

outpath='/Users/masahiroyamatani/Desktop/spark_sig/plots'
    
# File reading 
directory = './data/20180709/all.c300a300.2h'
fileID = '11422.CSV' 
dt = 1e-10
trigger = -0.01 
btrigger = -0.01

def FFT_AMP(data):
    data=np.hamming(len(data))*data
    data=np.fft.fft(data)
    data=np.abs(data)
    return data

if __name__ == "__main__":

    args = sys.argv
    if len(args)==3:
        directory = args[1]
        fileID = args[2]
    
    df = pd.read_csv(directory+'/'+fileID, header=None)
    df.columns = ['Param','nan','nan1','time','vol','nan3']

