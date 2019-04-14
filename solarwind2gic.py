#!/usr/bin/env python
'''
--------------------------------------------------------------------
solarwind2gic.py
Package for training a machine learning LSTM model to predict GIC
magnitudes from L1 solar wind data.
 
Useful reading:
 - Sequential: https://keras.io/models/sequential/
 - LSTM RNN: https://keras.io/layers/recurrent/#lstm
 - Example code: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
 - Batch size choice: https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network

Created 2017-09-14 by R Bailey, IWF Helio Group (Graz) / ZAMG Conrad Observatory (Vienna).
Last updated April 2019.
--------------------------------------------------------------------
'''

# Main imports:
import os
import sys

from datetime import datetime, timedelta
from dateutil import tz
import getopt
import math
import matplotlib.pyplot as plt
from matplotlib.dates import num2date, date2num, DateFormatter
import numpy as np
import operator
import pandas as pd
import pickle
import random
from scipy import optimize

# Machine-learning imports:
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals import joblib
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from keras.models import model_from_json
from keras import backend

# Special imports:
try:
    import IPython
except:
    pass
try:
    import seaborn as sns
except:
    pass

# *******************************************************************
# A. Useful functions:
# *******************************************************************

def root_mean_square(ar, interval=10):
    """Returns the root-mean-square in the interval of 10*input-t of
    the input array ar."""
    
    ar2 = ar * ar
    rms = np.zeros(len(ar)/interval)
    for i in range(len(ar)/interval):
        rms[i] = np.sqrt((1./float(interval)) * np.sum(ar2[i*interval:i*interval+interval]))
    return rms


def rsme(y_true, y_pred):
    """Returns the root-mean-square error between two arrays."""
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def mean_over_interval(ar, interval=10):
    """Returns the mean value of the input ar over the interval."""
    
    mean = np.zeros(len(ar)/interval)
    for i in range(len(ar)/interval):
        mean[i] = np.mean(ar[i*interval:i*interval+interval])
    return mean


def std_over_interval(ar, interval=10):
    """Returns the standard deviation of the input ar over the interval."""
    
    std = np.zeros(len(ar)/interval)
    for i in range(len(ar)/interval):
        std[i] = np.std(ar[i*interval:i*interval+interval])
    return std


def lin_interpolate_array(ar, condition, val=None):
    """Linearly interpolates values in array that fulfill condition
    given by variable condition.
    
    INPUT:
        - ar            (np.array) Array with numerical values
        - condition     (str) Operator to apply as condition. Values in ar
                        fulfilling the condition will be linearly interpolated.
                        Options: '>', '<', '>=', '<=', '=', 'isnan'
        - val           (float) Value for condition if condition!='isnan'.
    """
    
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '=': operator.eq,
           'isnan': np.isnan}
    
    if condition == 'isnan':
        inds = ops[condition](ar)
    else:
        inds = ops[condition](np.abs(ar), val)
    verboseprint("Linear interpolation over {:.1f}% of array for condition {}".format(
                 len(inds.nonzero()[0])/len(ar)*100., condition))
    ar[inds] = np.interp(inds.nonzero()[0], (~inds).nonzero()[0], ar[~inds])
    
    return ar


def verboseprint(*args):
    """Print each argument separately so caller doesn't need to
    stuff everything to be printed into a single string.
    """
    for arg in args:
        print("{} - {}".format(datetime.strftime(datetime.utcnow(), "%H:%M:%S"), arg)),
    print


# *******************************************************************
# B. Routines for data preparation:
# *******************************************************************


if __name__ == '__main__':

 	print("Nothing to see here.")




