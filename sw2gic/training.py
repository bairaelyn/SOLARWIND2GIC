#!/usr/bin/env python
'''
--------------------------------------------------------------------
sw2gic/training.py
Functions for training machine learning models to predict local GICs
from solar wind data.

Created July 2021 by R Bailey, ZAMG Conrad Observatory (Vienna).
Last updated Sept 2021.

TODO:
- Move find_folds into object
- Add function descriptions
--------------------------------------------------------------------
'''

import os, sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date
import numpy as np
import pickle
import random
from scipy.stats import pearsonr

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder

import keras
from keras.losses import BinaryCrossentropy
from keras.layers import Dense, LSTM, Layer, Dropout
from tensorflow.keras import backend as K
import tensorflow as tf

from .tools import calc_event_rates, compute_CM, compute_metrics, plot_on_time_series
from .tools import BOLD_s, BOLD_e

# *******************************************************************
#               OBJECTS USED IN MODEL TRAINING
# *******************************************************************

class DataStorer():
    '''
    This class holds all the data (training / testing / real-time) related to one variable.
    It also store the metrics calculated from the various arrays for each dataset as well
    as the scikit-learn scaling and encoding objects.
    
    Attributes:
    -----------
    varname :: str
        String name of variable.
    input_tr, output_tr, offset_tr : all int
        Timeranges for input and output (-offset) used for extracting samples.
    scaler :: sklearn MixMaxScaler
        Object scaled to ranges for variable.
    X_train, X_test, X_rt :: np.ndarrays
        Arrays containing training, testing and real-time application feature data.
    y_train, y_test, y_rt :: np.ndarrays
        Arrays containing target data (abs value of max. variable).
    y_train_sc, y_test_sc :: np.ndarrays
        Same as above but inverse-scaled using scaler (y_rt is already normal).
    ysign_train, ysign_test, ysign_rt :: np.ndarrays
        Arrays containing target data (class depicting sign).
    pred_train, pred_test, pred_rt :: np.ndarrays
        Arrays containing prediction target data (abs value of max. variable).
    pred_sign_train, pred_sign_test, pred_sign_rt :: np.ndarrays
        Arrays containing prediction target data (class depicting sign).
    pers_train, pers_test, pers_rt :: np.ndarrays
        Arrays containing persistence target data (abs value of max. variable).
    pers_sign_train, pers_sign_test, pers_sign_rt :: np.ndarrays
        Arrays containing persistence target data (class depicting sign).
    pred_metrics :: Dict
        Contains validation metrics for prediction model vs. target.
    pers_metrics :: Dict
        Contains validation metrics for persistence model vs. target.
        
    Functions:
    ----------
    load_training_data(self, path) 
        Loads data used for training and testing.
    load_realtime_data(self, path)
        Loads data in virtual real-time application.
    calc_pers_metrics
        Calculates metrics for persistence model (pers_) vs. real data (y_).
    make_predictions
        Takes keras NN model and makes predictions using X_train, X_test and X_rt.
    '''
    
    def __init__(self, varname, ip_tr, op_tr, os_tr, predict_sign=True):
        '''
        Parameters:
        -----------
        ip_tr :: int
            Length of history (time range) of input data, e.g. 120 (mins)
        op_tr :: int
            Time range over which max of output is taken, e.g. 40 (mins)
        os_tr :: int
            Offset applied to input, e.g. 0-20 (mins)
            '''
        self.varname = varname
        self.scaler = None
        self.predict_sign = predict_sign
        
        self.input_tr = ip_tr
        self.output_tr = op_tr
        self.offset_tr = os_tr
        suffix = '_{:04d}_{:04d}_{:04d}'.format(ip_tr, op_tr, os_tr)
        self.suffix = suffix
        

    def load_training_data(self, path, use_skip=10, enc=None):
        '''
        Parameters:
        -----------
        path :: str
            Directory where the data is stored.
        use_skip :: int (default=10)
            Uses input for every use_skip values
        enc :: OneHotEncoder()
            Encoder for encoding signs for classification problem.
        '''
        self.use_skip = use_skip
        
        # Load scaler from file:
        scaler = pickle.load(open(os.path.join(path,'scaler_{}.pkl'.format(self.varname)), 'rb'))
        self.scaler = scaler
        
        # Load training data arrays
        self.datapath_train = os.path.join(path, 'traindata_{}{}.npz'.format(self.varname, self.suffix))
        npzfile = np.load(self.datapath_train)
        X_train_all, y_train_all, pers_train_all = npzfile['X'], npzfile['y'], npzfile['pers']
        i_train = npzfile['i_clean']
        
        # Load testing data arrays
        self.datapath_test = os.path.join(path, 'testdata_{}{}.npz'.format(self.varname, self.suffix))
        npzfile = np.load(self.datapath_test)
        X_test_all, y_test_all, pers_test_all = npzfile['X'], npzfile['y'], npzfile['pers']
        i_test = npzfile['i_clean']

        # Reduce input to only use values every use_skip [10] minutes:
        self.X_train = X_train_all[:,::use_skip,:]
        self.X_test = X_test_all[:,::use_skip,:]
        
        # Reduce to just the regression problem (magnitude):
        self.y_train = y_train_all[:,0:1]
        self.y_test = y_test_all[:,0:1]
        
        # Encode the classification problem (sign):
        if enc == None:
            enc = OneHotEncoder()
            enc.fit(y_train_all[:,1:])
            self.enc = enc
        else:
            self.enc = enc
        self.ysign_train = enc.transform(y_train_all[:,1:]).toarray()
        self.ysign_test = enc.transform(y_test_all[:,1:]).toarray()
        
        # Define the persistence values:
        self.pers_train = pers_train_all[:,0:1]
        self.pers_test = pers_test_all[:,0:1]
        self.pers_sign_train = enc.transform(pers_train_all[:,1:]).toarray()
        self.pers_sign_test = enc.transform(pers_test_all[:,1:]).toarray()
        
        # Set sizes for tensors:
        self.seq_len = self.X_train.shape[1]
        self.input_size = self.X_train.shape[2]
        self.output_size = self.y_train.shape[1]
        
        self.i_train = i_train
        self.i_test = i_test
    
        y_train_sc = self.scaler.inverse_transform(self.y_train.reshape(-1,1))
        y_test_sc = self.scaler.inverse_transform(self.y_test.reshape(-1,1))
        
        self.y_train_sc = y_train_sc
        self.y_test_sc = y_test_sc
        

    def load_realtime_data(self, path, year=2017, varname=None, use_skip=10, enc=None):
        '''
        Loads virtual real-time data (e.g. 'rtdata{}_{}{}.npz') into objects.
        
        Parameters:
        -----------
        path :: str
            Directory where the data is stored.
        year :: int (default=2017)
            Decides which year of data to load. Can also be from ['all', 'allmeas', 'valid']
        use_skip :: int (default=10)
            Uses input for every use_skip values
        enc :: OneHotEncoder()
            Encoder for encoding signs for classification problem.
        '''
        
        if varname == None:
            vstr = self.varname
        else:
            vstr = varname
            
        if self.scaler == None:
            # Load scaler from file:
            scaler = pickle.load(open(os.path.join(path,'scaler_{}.pkl'.format(self.varname)), 'rb'))
            self.scaler = scaler
        
        # Load real-time data from file:
        if year.lower() in ['all', 'allmeas', 'valid']:
            self.year_lens = {}
            if year == 'valid':       # Data used for model selection
                rt_years = [2000, 2001]
            elif year == 'allmeas':   # All years with GIC measurements
                rt_years = [2017, 2019, 2020]
            else:                     # All testing data
                rt_years = [2000, 2001, 2017, 2019, 2020]
            print("Loading years {}...".format(rt_years))
            self.rt_years = rt_years
            X_rt_all, y_rt_all, pers_rt_all, i_nans = [], [], [], []
            for year_temp in rt_years:
                savepath_rt = os.path.join(path, 'rtdata{}_{}{}.npz'.format(year_temp, vstr, self.suffix))
                npzfile = np.load(savepath_rt)
                X_rt_temp, y_rt_temp, pers_rt_temp = npzfile['X'], npzfile['y'], npzfile['pers']
                i_nans_temp = npzfile['i_nans']
                if '5' in vstr and year_temp == 2017: # data only starts later in the year
                    i_cut = 23552
                    X_rt_temp = X_rt_temp[i_cut:]
                    y_rt_temp = y_rt_temp[i_cut:]
                    pers_rt_temp = pers_rt_temp[i_cut:]
                    i_nans_temp = i_nans_temp[i_nans_temp > i_cut] - i_cut
                X_rt_all.append(X_rt_temp)
                y_rt_all.append(y_rt_temp)
                pers_rt_all.append(pers_rt_temp)
                i_nans.append(i_nans_temp.reshape(-1,1))
                self.year_lens[year_temp] = len(y_rt_temp)
            X_rt_all, y_rt_all = np.vstack(X_rt_all), np.vstack(y_rt_all)
            pers_rt_all, i_nans = np.vstack(pers_rt_all), np.vstack(i_nans)
        else:
            self.rt_years = [year]
            savepath_rt = os.path.join(path, 'rtdata{}_{}{}.npz'.format(year, vstr, self.suffix))
            npzfile = np.load(savepath_rt)
            X_rt_all, y_rt_all, pers_rt_all = npzfile['X'], npzfile['y'], npzfile['pers']
            i_nans = npzfile['i_nans']
                
        self.i_nans_rt = i_nans
        
        # Encode the classification problem (sign):
        if enc == None:
            enc = OneHotEncoder()
            enc.fit(y_rt_all[:,1:])
            self.enc = enc
        else:
            self.enc = enc

        # Reduce to only use values every 10 minutes:
        self.X_rt = X_rt_all[:,::use_skip,:]
        # Reduce to just the regression problem:
        self.y_rt = y_rt_all[:,0:1]
        self.ysign_rt = enc.transform(y_rt_all[:,1:]).toarray()
        self.pers_rt = pers_rt_all[:,0:1]
        self.pers_sign_rt = enc.transform(pers_rt_all[:,1:]).toarray()
        
        y_rt_sc = self.scaler.inverse_transform(self.y_rt.reshape(-1,1))
        self.y_rt_sc = y_rt_sc
    
    
    def calc_pers_metrics(self):
        '''
        Calculates the RMSE and Pearson's Correlation Coefficient (PCC) for the
        persistence model vs. the actual data.
        
        Parameters:
        -----------
        None, but must be called after load_training_data() and load_realtime_data().
        
        Returns:
        --------
        (rmse_pers_train, rmse_pers_test, pcc_pers_train, pcc_pers_test) :: tuple of float
            Values for RMSE followed by PCC for train and test datasets.
            (These are also saved to self.pers_metrics.)
        '''
        
        rmse_pers_train = np.sqrt(mean_squared_error(self.pers_train, self.y_train_sc))
        rmse_pers_test = np.sqrt(mean_squared_error(self.pers_test, self.y_test_sc))
        rmse_pers_rt = np.sqrt(mean_squared_error(self.pers_rt, self.y_rt_sc))
        pcc_pers_train = pearsonr(np.squeeze(self.pers_train), np.squeeze(self.y_train_sc))[0]
        pcc_pers_test = pearsonr(np.squeeze(self.pers_test), np.squeeze(self.y_test_sc))[0]
        pcc_pers_rt = pearsonr(np.squeeze(self.pers_rt), np.squeeze(self.y_rt_sc))[0]
        
        # Classification problem
        if self.predict_sign:
            acc_pers_train = accuracy_score(self.ysign_train, self.pers_sign_train)
            acc_pers_test  = accuracy_score(self.ysign_test,  self.pers_sign_test)
            acc_pers_rt  = accuracy_score(self.ysign_rt,  self.pers_sign_rt)
        else:
            acc_pers_train, acc_pers_test, acc_pers_rt = 0, 0, 0
        
        self.pers_metrics = {'rmse_train': rmse_pers_train,
                             'rmse_test':  rmse_pers_test,
                             'rmse_rt':    rmse_pers_rt,
                             'pcc_train':  pcc_pers_train,
                             'pcc_test':   pcc_pers_test,
                             'pcc_rt':     pcc_pers_rt,
                             'acc_train':  acc_pers_train,
                             'acc_test':   acc_pers_test,
                             'acc_rt':     acc_pers_rt}
        
        return (rmse_pers_train, rmse_pers_test, pcc_pers_train, pcc_pers_test)

    
    def make_predictions(self, model, threshold=60, verbose=False, show_plot=True):
        '''
        Takes a model and makes predictions using X_train, X_test, and X_rt.
        
        Parameters:
        -----------
        model :: compiled keras LSTM
            The model should output two variables: (1) regression for abs target
            value, and (2) class depicting target sign.
        threshold :: float (default=60)
            Threshold above which to define events for metric calculation.
        verbose :: bool (default=False)
            Prints steps and details if True.
        show_plot :: bool (default=True)
            Plots some example data if True.
        
        Returns:
        --------
        None
        '''
        output_train = model.predict(self.X_train)
        output_test  = model.predict(self.X_test)
        output_rt    = model.predict(self.X_rt)
        self.evaluate_model_pred(output_train, output_test, output_rt)
        
        CM_train, CM_pers_train = compute_CM(self.y_train_sc, self.pred_train, self.pers_train, threshold=threshold, verbose=False)
        CM_test, CM_pers_test   = compute_CM(self.y_test_sc, self.pred_test, self.pers_test, threshold=threshold, verbose=False)
        CM_rt, CM_pers_rt       = compute_CM(self.y_rt_sc, self.pred_rt, self.pers_rt, threshold=threshold, verbose=False)
        
        for dstr, CMs in zip(['train', 'test', 'rt'], 
                             [[CM_train, CM_pers_train], [CM_test, CM_pers_test], [CM_rt, CM_pers_rt]]):
            pred_dict = calc_event_rates(CMs[0])
            for key, value in pred_dict.items():
                self.pred_metrics[key+'_'+dstr] = value
            pers_dict = calc_event_rates(CMs[1])
            for key, value in pers_dict.items():
                self.pers_metrics[key+'_'+dstr] = value
        
        if verbose:
            print("\nTRAIN / TEST DATASETS\n----------------------")
            for datastr in ['train', 'test']:
                self.print_metrics(datastr)

        if show_plot:
            fig = plt.figure(figsize=(6,6))
            plt.plot(self.y_train_sc, self.pred_train, 'x')
            plt.plot(self.y_test_sc, self.pred_test, 'x')
            plt.gca().set_aspect('equal', 'box')
            plt.show()

        if verbose:
            print("\nREAL-TIME APPLICATION\n----------------------")
            self.print_metrics('rt', metrics=['rmse', 'pcc', 'acc', 'TSS', 'BS'])
        
        if show_plot:
            test_year = (2017 if 2017 in self.rt_years else 2000)
            plot_on_time_series(self.y_rt_sc, self.pred_rt, self.pers_rt, year=test_year)
        

    def evaluate_model_pred(self, output_train, output_test, output_rt):
        '''
        Evaluates the model output and transforms it into data that can be used for metric evaluation.
        '''
        
        if self.predict_sign:
            pred_train_reg, pred_train_cla = output_train
            pred_test_reg, pred_test_cla   = output_test
            pred_rt_reg, pred_rt_cla       = output_rt
            pred_train, sign_train = self.scaler.inverse_transform(pred_train_reg), self.enc.inverse_transform(pred_train_cla)
            pred_test, sign_test   = self.scaler.inverse_transform(pred_test_reg),  self.enc.inverse_transform(pred_test_cla)
            pred_rt, sign_rt       = self.scaler.inverse_transform(pred_rt_reg),    self.enc.inverse_transform(pred_rt_cla)
        else:
            pred_train = self.scaler.inverse_transform(output_train)
            pred_test  = self.scaler.inverse_transform(output_test)
            pred_rt    = self.scaler.inverse_transform(output_rt)
            sign_train, sign_test, sign_rt = np.zeros(len(output_train)), np.zeros(len(output_test)), np.zeros(len(output_rt))
          
        self.pred_train = pred_train
        self.pred_test = pred_test
        self.pred_rt = pred_rt
        self.pred_sign_train = sign_train
        self.pred_sign_test = sign_test
        self.pred_sign_rt = sign_rt
    
        # Regression problem
        rmse_train = np.sqrt(mean_squared_error(self.pred_train, self.y_train_sc))
        rmse_test = np.sqrt(mean_squared_error(self.pred_test, self.y_test_sc))
        rmse_rt = np.sqrt(mean_squared_error(self.pred_rt, self.y_rt_sc))
        pcc_train = pearsonr(np.squeeze(self.pred_train), np.squeeze(self.y_train_sc))[0]
        pcc_test  = pearsonr(np.squeeze(self.pred_test), np.squeeze(self.y_test_sc))[0]
        pcc_rt  = pearsonr(np.squeeze(self.pred_rt), np.squeeze(self.y_rt_sc))[0]
            
        # Classification problem
        if self.predict_sign:
            self.pred_train_signed = pred_train * (sign_train*2 - 1)
            self.pred_test_signed  = pred_test * (sign_test*2 - 1)
            self.pred_rt_signed    = pred_rt * (sign_rt*2 - 1)

            self.y_train_signed = self.y_train_sc * (self.enc.inverse_transform(self.ysign_train)*2 - 1)
            self.y_test_signed  = self.y_test_sc  * (self.enc.inverse_transform(self.ysign_test)*2 - 1)
            self.y_rt_signed    = self.y_rt_sc    * (self.enc.inverse_transform(self.ysign_rt)*2 - 1)

            self.pers_train_signed = self.pers_train * (self.enc.inverse_transform(self.pers_sign_train)*2 - 1)
            self.pers_test_signed  = self.pers_test  * (self.enc.inverse_transform(self.pers_sign_test)*2 - 1)
            self.pers_rt_signed    = self.pers_rt    * (self.enc.inverse_transform(self.pers_sign_rt)*2 - 1)
            
            acc_train = accuracy_score(self.enc.inverse_transform(self.ysign_train), self.pred_sign_train)
            acc_test  = accuracy_score(self.enc.inverse_transform(self.ysign_test), self.pred_sign_test)
            acc_rt  = accuracy_score(self.enc.inverse_transform(self.ysign_rt), self.pred_sign_rt)
        else:
            acc_train, acc_test, acc_rt = 0, 0, 0
        
        self.pred_metrics = {'rmse_train': rmse_train,
                             'rmse_test':  rmse_test,
                             'rmse_rt':    rmse_rt,
                             'pcc_train':  pcc_train,
                             'pcc_test':   pcc_test,
                             'pcc_rt':     pcc_rt,
                             'acc_train':  acc_train,
                             'acc_test':   acc_test,
                             'acc_rt':     acc_rt}

        
    def print_metrics(self, datastr, metrics=['rmse', 'pcc', 'acc']):
        '''
        Prints a handy summary of the metrics RMSE, Pearson's correlation coefficient
        and accuracy score. datastr can only be one of ['train', 'test', 'rt']
        '''
        
        units = {'rmse': 'mV/km ', 'pcc': '', 'acc': '', 'TSS': '', 'BS': ''}
        # -1 for "smaller is better", +1 for "greater is better"
        mf = {'rmse': -1, 'pcc': +1, 'acc': +1, 'TSS': +1, 'BS': -1}
        
        res_str = '({})'.format(datastr)
        for m in metrics:
            pred_val = self.pred_metrics[m+'_'+datastr]*mf[m]
            pers_val = self.pers_metrics[m+'_'+datastr]*mf[m]
            if pred_val > pers_val:
                pred_str, pers_str = BOLD_s+"{:.2f}".format(pred_val*mf[m])+BOLD_e, "{:.2f}".format(pers_val*mf[m])
            else:
                pred_str, pers_str = "{:.2f}".format(pred_val*mf[m]), BOLD_s+"{:.2f}".format(pers_val*mf[m])+BOLD_e
            res_str += "\t{} = {} ({}) {}".format(m.upper(), pred_str, pers_str, units[m])
            
        print(res_str)


class BasicAttention(Layer):
    '''Basic Self-Attention Layer built using this resource:
    https://towardsdatascience.com/create-your-own-custom-attention-layer-understand-all-flavours-2201b5e8be9e
    '''
    
    def __init__(self, return_sequences=True, n_units=1, w_init='normal', b_init='zeros', **kwargs):
        self.return_sequences = return_sequences
        self.n_units = n_units
        self.w_init = w_init
        self.b_init = b_init
        super(BasicAttention,self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.n_features = input_shape[-1]
        self.seq_len = input_shape[-2]
        
        self.W=self.add_weight(name="att_weight", shape=(self.n_features,self.n_units),
                               initializer=self.w_init)
        self.b=self.add_weight(name="att_bias", shape=(self.seq_len,self.n_units),
                               initializer=self.b_init)
        
        super(BasicAttention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)
    
    def get_config(self):
        config = super(BasicAttention, self).get_config()
        config["return_sequences"] = self.return_sequences
        config["n_units"] = self.n_units
        config["w_init"] = self.w_init
        config["b_init"] = self.b_init
        #config["name"] = self.name
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
# *******************************************************************
#               FUNCTIONS USED IN MODEL TRAINING
# *******************************************************************

def get_model(seq_len=12, input_size=24, p_dropout=0.3, nhidden=256, nhidden_2=32, predict_sign=True):
    '''
    Loads a model object.
    
    Parameters:
    -----------
    seq_len :: int (default=12)
        Length of input sequence to LSTM.
    input_size :: int (default=24)
        Number of features.
    p_dropout :: float (default=0.3)
        Fraction of dropout used in second LSTM layer.
    nhidden :: int (default=256)
        Number of hidden LSTM blocks in initial layer.
    nhidden_2 :: int (default=32)
        Number of hidden LSTM blocks in second LSTM layer.
    predict_sign :: bool (default=True)
        If True, LSTM branches into two with reg and cla output (magnitude and sign).
        If False, LSTM only predicts magnitude.
        
    Returns:
    --------
    model :: keras.Model with LSTM and Attention layers. 
        Use e.g. model.summary() to see the structure.
    '''

    input_layer = keras.Input(shape=(seq_len, input_size), name="input_layer")

    # First LSTM for all:
    lstm_all = LSTM(nhidden, return_sequences=True, name='lstm_all')(input_layer) # 
    
    # Attention as it splits
    attn_reg = BasicAttention(return_sequences=True, name='attn_reg')(lstm_all)
    attn_cla = BasicAttention(return_sequences=True, name='attn_cla')(lstm_all)
    
    # Second LSTM layer
    lstm_reg = LSTM(nhidden_2, name='lstm_reg', dropout=p_dropout)(attn_reg)
    lstm_cla = LSTM(nhidden_2, name='lstm_cla', dropout=p_dropout)(attn_cla)
        
    # Feed-forward layer for output
    output_reg = Dense(1, activation = 'linear', name='output_reg')(lstm_reg)
    output_cla = Dense(2, activation = 'sigmoid', name='output_cla')(lstm_cla)
    
    if predict_sign:
        model = keras.Model(inputs=input_layer,outputs=[output_reg, output_cla])
    else:
        model = keras.Model(inputs=input_layer,outputs=output_reg)
    
    return(model)


def fit_model(v, pdict, i_split=None, use_callback=True, plot_loss=True, fit_verbose=2, seed=42, print_params=True):
    '''Fits the model defined in get_model() according to the parameters defined in pdict.
    
    i_split provides a split between training and validation data if the model should stop after
    validation loss stops improving.'''
    
    patience, batch_size, loss_weights, restore_best_weights = pdict['p'], pdict['bs'], pdict['lw'], pdict['rbw']
    epochs, p_dropout, nhidden = pdict['ep'], pdict['do'], pdict['nh']
    if print_params:
        param_str = "\n\t".join(["patience={}".format(patience), "loss_weights={}".format(loss_weights),
                                 "batch_size={}".format(batch_size), "restore_best_weights={}".format(restore_best_weights),
                                 "seed={}".format(seed), "epochs={}".format(epochs), "use_callback={}".format(use_callback),
                                 "p_dropout={}".format(p_dropout), "n_hidden={}".format(nhidden)])
        print("    Training with following parameters:\n\t{}".format(param_str))
    
    # Set all seeds so that results are reproducible
    os.environ['PYTHONHASHSEED'] = str(seed+3)
    random.seed(seed-3)
    np.random.seed(seed+2)
    tf.random.set_seed(seed)

    if use_callback:
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, min_delta=0,
                                                 restore_best_weights=restore_best_weights)
        callback = [callback]
        i_val, i_train = i_split[0], i_split[1]
        val_data = (v.X_train[i_val], [v.y_train[i_val], v.ysign_train[i_val]])
        X_train = v.X_train[i_train]
        y_train, sign_train = v.y_train[i_train], v.ysign_train[i_train]
    else:
        callback = None
        val_data = None
        X_train = v.X_train
        y_train, sign_train = v.y_train, v.ysign_train
        
    # Single or split LSTM:
    if v.predict_sign:
        y_output = {"output_reg": y_train, "output_cla": sign_train}
        loss_fn = [min_max_loss, keras.losses.BinaryCrossentropy()]
    else:
        y_output = y_train
        loss_fn = min_max_loss
        loss_weights = None
        
    # Create model
    model = get_model(p_dropout=p_dropout, nhidden=nhidden, predict_sign=v.predict_sign)
    model.compile(optimizer="adam",loss=loss_fn,loss_weights=loss_weights)
        
    # Fit
    history = model.fit(X_train, y_output, validation_data=val_data, epochs=epochs, 
                        batch_size=batch_size, callbacks=callback, verbose=fit_verbose)

    if plot_loss:
        plt.plot(history.history['loss'], label='train')
        if use_callback:
            plt.plot(history.history['val_loss'], label='val')
        plt.axvline(x=len(history.history['loss'])-patience-1, ls='--', c='grey')
        plt.legend()
        plt.show()
        
    return model
    

def min_max_loss(y_true, y_pred, loss_adjustment=0.1):
    N = tf.dtypes.cast(len(y_true), tf.float32)
    abs_diff = tf.abs(y_true - y_pred)
    error = tf.reduce_sum(abs_diff, axis=-1) / N
    adjustment = ((tf.reduce_max(y_true, axis=-1) - tf.reduce_min(y_true, axis=-1)) - 
                  (tf.reduce_max(y_pred, axis=-1) - tf.reduce_min(y_pred, axis=-1))) / N
    return error + 0.1*adjustment


