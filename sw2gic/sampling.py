#!/usr/bin/env python
'''
--------------------------------------------------------------------
sw2gic/sampling.py
Functions for extracting samples from many years of data.

Created July 2021 by R Bailey, ZAMG Conrad Observatory (Vienna).
Last updated August 2021.
--------------------------------------------------------------------
'''

import os, sys
from datetime import datetime, timedelta
import numpy as np
import random
from scipy.stats import norm

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

lt_vars = ['sin_DOY', 'cos_DOY', 'sin_LT', 'cos_LT']
sw_vars = ['speed', 'bz', 'by', 'btot', 'density']

# #########################################################################
#     ENCODER FOR FEATURE ENCODING                                        #
# #########################################################################

class SolarWindFeatureEncoder:
    '''Creates an object for encoding all SW variables in preparation for model training.
    
    Attributes:
    -----------
    self.Encoder_bz, self.Encoder_by, self.Encoder_btot, self.Encoder_speed, self.Encoder_density
        : TriangularValueEncoding objects for each SW variable
    self.encode_nodes : int (default=4)
        Number of overlapping triangles to encode variables in.
    '''
    
    def __init__(self, max_vals, min_vals, encode_nodes=4):
        '''Creates an object for encoding all SW variables into overlapping triangles
        (n of triangles=encode_nodes) - each triangle goes from 0 to 1.
    
        Parameters:
        -----------
        max_vals : Dict 
            Contains maximum values for each variable for scaling
        min_vals : Dict 
            Contains minimum values for each variable for scaling
        encode_nodes : int (default=4)
            Number of overlapping triangles - splits one var into four.
        '''
        # IMF Bz, By, Btot
        self.Encoder_bz = TriangularValueEncoding(min_value=min_vals['bz'], max_value=max_vals['bz'], 
                                                  n_nodes=encode_nodes)
        self.Encoder_by = TriangularValueEncoding(min_value=min_vals['by'], max_value=max_vals['by'], 
                                                  n_nodes=encode_nodes)
        self.Encoder_btot = TriangularValueEncoding(min_value=min_vals['btot'], max_value=max_vals['btot'], 
                                                    n_nodes=encode_nodes)
        # Solar wind speed
        self.Encoder_speed = TriangularValueEncoding(min_value=min_vals['speed'], max_value=max_vals['speed'], 
                                                     n_nodes=encode_nodes)
        # Density
        self.Encoder_density = TriangularValueEncoding(min_value=min_vals['density'], max_value=max_vals['density'], 
                                                       n_nodes=encode_nodes)
        # Number of encoding nodes
        self.encode_nodes = encode_nodes
        
    def encode_all(self, df, verbose=False):
        '''Uses the Encoder objects created in initialisation to encode all the SW
        variables in the DataFrame object provided.
        
        Local time variables are also returned, but these are already encoded in sine/cosine
        functions and don't undergo triangular encoding.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Should contain all SW variables under: ['bx', 'by', 'btot', 'speed', 'density']
            Should also contain local time variables: ['sin_DOY', 'cos_DOY', 'sin_LT', 'cos_LT']
        verbose : bool (default=False)
            Print the steps if True.
            
        Returns:
        --------
        omni_feat : np.ndarray, dtype=np.float32
            Returns an array of shape (len(df), 24). SW vars x encode_nodes + local-time vars = 24.
        '''
        bz_encoded = self.Encoder_bz.encode_values(df['bz'].to_numpy())
        by_encoded = self.Encoder_by.encode_values(df['by'].to_numpy())
        btot_encoded = self.Encoder_btot.encode_values(df['btot'].to_numpy())
        speed_encoded = self.Encoder_speed.encode_values(df['speed'].to_numpy())
        density_encoded = self.Encoder_density.encode_values(df['density'].to_numpy())
        # Don't encode these since sin/cos is already an encoding:
        lt_vars = df[['sin_DOY', 'cos_DOY', 'sin_LT', 'cos_LT']].to_numpy()
        
        # Stack them into a feature array:
        omni_feat = np.hstack((bz_encoded, by_encoded, btot_encoded, speed_encoded, density_encoded, lt_vars)).astype(np.float32)
        var_list = ['Bz', 'By', 'Btot', 'Speed', 'Density', 'LT']
            
        # Print locations in array:
        if verbose:
            pos_str = 'Positions in feature array: '
            for i_var, varname in enumerate(var_list):
                if varname != 'LT':
                    pos_str += ("{}=[{}:{}], ".format(varname, i_var*self.encode_nodes, (i_var+1)*self.encode_nodes))
                else:
                    pos_str += ("{}=[{}:{}],".format(varname, i_var*self.encode_nodes, omni_feat.shape[1]-1))
            print(pos_str)

        return omni_feat


class TriangularValueEncoding(object):
    def __init__(self, max_value, min_value, n_nodes: int, normalize: bool = False):
        """
        Code originally from M. Widrich, Python package widis-lstm-tools (2021):
        https://github.com/widmi/widis-lstm-tools/blob/master/widis_lstm_tools/preprocessing.py
        
        Encodes values in range [min_value, max_value] as array of shape (len(values), n_nodes)
        
        LSTM profits from having a numerical input with large range split into multiple input nodes; This class encodes
        a numerical input as n_nodes nodes with activations of range [0,1]; Each node represents a triangle of width
        triangle_span; These triangles are distributed equidistantly over the input value range such that 2 triangles
        overlap by 1/2 width and the whole input value range is covered; For each value to encode, the height of the
        triangle at this value is taken as node activation, i.e. max. 2 nodes have an activation > 0 for each input
        value, where both activations sum up to 1.
        
        Values are encoded via self.encode_value(value) and returned as float32 numpy array of length self.n_nodes;
        
        Parameters
        ----------
        max_value : float or int
            Maximum value to encode
        min_value : float or int
            Minimum value to encode
        n_nodes : int
            Number of nodes to use for encoding; n_nodes has to be larger than 1;
        normalize : bool
            Normalize encoded values? (default: False)
        """
        if n_nodes < 2:
            raise ValueError("n_nodes has to be > 1")
        
        # Set max_value to max value when starting from min value = 0
        max_value -= min_value
        
        # Calculate triangle_span (triangles overlap -> * 2)
        triangle_span = (max_value / (n_nodes - 1)) * 2
        
        self.n_nodes = int(n_nodes)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.triangle_span = float(triangle_span)
        self.normalize = normalize
    
    def encode_values(self, values):
        """Encode values as multiple triangle node activations
        Parameters
        ----------
        values : numpy.ndarray
            Values to encode as numpy array of shape (len(values),)
        
        Returns
        ----------
        float32 numpy array
            Encoded value as float32 numpy array of shape (len(values), n_nodes)
        """
        values = np.array(values, dtype=np.float)
        values[:] -= self.min_value
        values[:] *= ((self.n_nodes - 1) / self.max_value)
        encoding = np.zeros((len(values), self.n_nodes), np.float32)
        value_inds = np.arange(len(values))
        
        # low node
        node_ind = np.asarray(np.floor(values), dtype=np.int)
        node_activation = 1 - (values - node_ind)
        node_ind[:] = np.clip(node_ind, 0, self.n_nodes - 1)
        encoding[value_inds, node_ind] = node_activation
        
        # high node
        node_ind[:] += 1
        node_activation[:] = 1 - node_activation
        node_ind[:] = np.mod(node_ind, self.n_nodes)
        encoding[value_inds, node_ind] = node_activation
        
        # normalize encoding
        if self.normalize:
            encoding[:] -= (1 / self.n_nodes)
        
        return encoding
    

# *******************************************************************
#               FUNCTIONS FOR SAMPLING DATA SETS
# *******************************************************************

def extract_feature_samples(df, i_samples, SWEncoder, output_timerange, input_timerange, offset_timerange, nan_window=15, use_rand_offset=True):
    '''Extracts samples from DataFrame with list of indices provided.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame of time series solar wind data with time as the index.
    i_samples : np.array
        Indices of samples.
    SWEncoder : sw2gic.sampling.SolarWindEncoder
        Object that encodes all variables in overlapping triangles - splits one var into four.
    output_timerange : int
        Window over which to take the max. value of variable var_name for the target.
    input_timerange : int
        Window over which to extract the input solar wind data to make the features.
    offset_timerange : int
        Offset to apply to sample extraction.
    nan_window : int (default=15)
        Max. allowable number of consecutive nans in input_timerange.
        If exceeded, sample is excluded from analysis and training.
    use_rand_offset : bool (default=True)
        If True, applies a random offset in the range -10:10
        
    Returns:
    --------
    (X, i_clean) : (np.array, np.array)
        X : max(df[var_name]) for each sample in i_samples over output_timerange.
        i_clean : i_samples with nan-filled samples removed
    '''
    
    # Variables to extract:
    variables = sw_vars + lt_vars
    np.random.seed(123)
    
    # Define random offsets:
    if use_rand_offset:
        rand_offset = np.random.randint(-10, 10, i_samples.shape)
    else:
        rand_offset = np.zeros(i_samples.shape).astype(int)
    i_clean = []
    omni_seq = []
    nan_count = 0
    for i_i, i in enumerate(i_samples):
        # Subtract the offset and add a random offset so not all maxima are at omni_t=input_timerange
        ioff = i + rand_offset[i_i] - offset_timerange
        df_excerpt = df.iloc[ioff - input_timerange:ioff][variables]
        if len(df_excerpt) == 0:
            raise Exception("Empty DataFrame! Something wrong with indexing: i={}, ioff={}, offset_timerange={}".format(
                            i, ioff, offset_timerange))
        # If there is a period of nans longer than nan_window, ignore it:
        df_excerpt = df_excerpt.interpolate(method='linear', limit=int(nan_window/2), limit_direction='both')
        test_for_nans = np.count_nonzero(np.isnan(df_excerpt.values))
        if test_for_nans > 0:
            nan_count += 1
        else:
            omni_feat_excerpt = SWEncoder.encode_all(df_excerpt)
            omni_seq.append(omni_feat_excerpt)
            i_clean.append(i)
    print("Ignored {:.1f}% of data contaminated by NaNs in data set.".format(nan_count/len(i_samples)*100.))
    X = np.array(omni_seq).astype(np.float32)
    
    return X, np.array(i_clean)


def create_target_array(df, var_name, i_samples, output_timerange, scaler=None, apply_scaler=True, max_val=None):
    '''Creates an array of max. E taken from the past output_timerange to forecast the
    next value after the expansion time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame of time series data with time as the index.
    var_name : str
        Name of column variable in df to extract, e.g. 'Ex'.
    i_samples : np.array
        Indices of samples.
    output_timerange : int
        Window over which to take the max. value of variable var_name.
    scaler : sklearn.preprocessing.MinMaxScaler or None (default=None)
        Scaler to apply to df[var_name]. If None, new one is applied.
    max_val : float or None (default=None)
        Maximum value allowed. If df[var_name] exceeds this, it is set to max_val.
        If max_val is None, all values are allowed.
        
    Returns:
    --------
    (y, sign) : (np.array, np.array)
        y : max(df[var_name]) for each sample in i_samples over output_timerange.
        sign : class designating sign of each value in y_pers (0 for neg, 1 for pos)
    '''
        
    target_seq = []
    output_win = int(output_timerange/2)
    for i in i_samples:
        target_seq.append(df[var_name][i-output_win:i+output_win][np.argmax(
            np.abs(df[var_name][i-output_win:i+output_win]))])
    target = np.abs(np.array(target_seq)).astype(np.float32)
    if max_val != None:
        target[target > max_val] = max_val
    # Scale the values - this is only fit to the training data
    target_sc = scaler.transform(target.reshape(-1,1))
    y = target_sc.astype(np.float32)
    # Retrieve binary classes for sign:
    sign = np.zeros(len(target))
    sign[np.array(target_seq) >= 0.] = 1
    sign = sign.astype(int)
    
    return (y, sign)


def calc_t_pers_back(speed):
    '''Calculates the time it took solar wind to propagate from L1 to Earth at 'speed'.
    Returned in minutes as int.'''
    dist_to_L1 = 1496547.603
    return int(dist_to_L1 / speed / 60.)


def create_persistence_array(df, var_name, i_samples, output_timerange, offset_timerange):
    '''Creates an array of max. E taken from the past output_timerange to forecast the
    next value after the expansion time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame of time series data with time as the index.
    var_name : str
        Name of column variable in df to extract, e.g. 'Ex'.
    i_samples : np.array
        Indices of samples.
    output_timerange : int
        Window over which to take the max. value of variable var_name.
    offset_timerange : int
        Offset to apply to extraction of max from output timerange. (t-offset_timerange)
        
    Returns:
    --------
    (y_pers, sign) : (np.array, np.array)
        y_pers : max(df[var_name]) for each sample in i_samples over output_timerange.
        sign : sign of each value in y_pers (0 for neg, 1 for pos)
    '''
    
    target_seq = []
    for i in i_samples:
        # Calculate t-back (the time SW would have arrived at L1 at i.e. time forecast made at)
        avg_speed = df.iloc[i-output_timerange-offset_timerange:i-offset_timerange]['speed'].mean()
        t_back = calc_t_pers_back(avg_speed)
        # Cut the range from t_back - timerange : t_back
        i_cut = np.arange(i-t_back-output_timerange,i-t_back)
        target_seq.append(df[var_name][i_cut][np.argmax(np.abs(df[var_name][i_cut]))])
    y_pers = np.abs(np.array(target_seq)).astype(np.float32)
    sign = np.zeros(len(y_pers))
    sign[np.array(target_seq) >= 0.] = 1
    sign = sign.astype(int)
    
    return (y_pers, sign)


def extract_datasets_by_years(df, years, i_all, verbose=False):
    '''Extracts indices from a DataFrame according to lists of years.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame of time series data with time as the index.
    years : list of lists
        List of years split by start and end, e.g. [[2000, 2001], [2017], [2019, 2020]]
    i_all : np.array(dtype=int)
        The list of indices to extract.
    verbose : bool (default=False)
        Will print steps if true.
        
    Returns:
    --------
    (df_new, i_true, i_adj) : (pandas.DataFrame, np.array, np.array)
        df_new : DataFrame containing only the years in the 'years' variable.
        i_true : the original indices, extracted.
        i_adj : the extracted indices adjusted to the dataframe df_new.
    '''
    
    df_new = pd.DataFrame()
    i_true, i_adj = np.array([]), np.array([])
    len_subtracted_total, last_cut_end = 0, 0
    for dates in years:
        cut_start, cut_end = datetime(dates[0],1,1), datetime(dates[-1]+1,1,1)
        i_cut_start = np.where(df.index >= cut_start)[0][0]
        if cut_end > df.iloc[-1].name:
            i_cut_end = len(df)-1
        else:
            i_cut_end = np.where(df.index >= cut_end)[0][0]
        if verbose:
            print(dates, cut_start, cut_end)
            print("Length = {}, Last stop = {}, New = {}--{}".format(i_cut_end - i_cut_start, last_cut_end, i_cut_start, i_cut_end))
            print("Cutting dataframe between {} and {}".format(df.iloc[i_cut_start].name, df.iloc[i_cut_end].name))

        i_all_cuts = np.where(np.logical_and(np.array(i_all) >= i_cut_start, np.array(i_all) < i_cut_end))
        df_new = pd.concat((df_new, df.iloc[i_cut_start:i_cut_end]))
        i_true = np.hstack((i_true, i_all[i_all_cuts]))
        len_subtracted = (i_cut_start - last_cut_end)
        i_adj = np.hstack((i_adj, i_all[i_all_cuts] - (len_subtracted + len_subtracted_total)))
        len_subtracted_total += len_subtracted
        last_cut_end = i_cut_end
        
    return (df_new, i_true, i_adj)


def high_val_decay(x, a=8000.):
    return a/x


def sample_quiet_and_active_times(rawdata, input_timerange, min_level=100, binsize=5, a_gauss=80000., var_gauss=50, max_overlap=60, oversample_by={}, verbose=False):
    '''Samples a subset of the full dataset. This is done in reverse from most active times to least active
    because the algorithm also checks no two samples are within input_timerange/2 of each other.
    This means two samples will have a max. input overlap of 50% and quiet periods can't exclude active periods.
    ... The loop is a bit of a monster. Set verbose=True to debug and improve the resulting distribution.
    
    Parameters:
    -----------
    rawdata : np.array
        Raw time series data in an array.
    input_timerange : int
        Length of time range that will be extracted as input to the model.
        (This is needed to remove overlapping input sequences.)
    min_level : float (default=100)
        Level above which values are considered 'active'.
    binsize : int (default=5)
        Size of each level to sample from. min_level/binsize should be an integer.
        The smaller this number, the longer the runtime and the smoother the distribution.
    a_gauss : float (default=80000.)
        The bins will be sampled according to a rough Gaussian distribution.
        This value determines the peak height of the Gaussian/number of samples.
    var_gauss : float (default=50)
        This value determines the variance of the Gaussian/width of sample distribution.
    max_overlap : int (default=60)
        Maximum amount of overlap (in # of timesteps) between two samples.
    verbose : bool (default=False)
        Will print steps if true.
        
    Returns:
    --------
    (i_quiet, i_max, max_level_splits) : (np.array, np.array, int)
        Indices of active and quiet times, following by the maximum level in rawdata.
    '''
    
    level_splits = int(min_level / binsize)
    
    # Define Gaussian distribution as a PDF:
    x = np.arange(0, 400, binsize)
    norm_x = norm.pdf(x,0,var_gauss) * a_gauss
    i_quiet = []
    
    # --------------------------------------------------------------
    # SAMPLE FROM THE ACTIVE TIMES AND TAIL END OF THE DISTRIBUTION
    # --------------------------------------------------------------
    
    i_max = []
    # Define how often values above a certain level will be oversampled by:
    if len(oversample_by) == 0:
        oversample_by = {100: 2, 200: 3, 250: 5, 300: 10}
    max_level_splits = np.ceil((np.max(np.abs(rawdata)) - min_level) / binsize).astype(int)
    
    # Iterate through splits
    for i_split in range(max_level_splits-1, -1, -1):

        level_low = min_level + i_split * binsize
        level_high = min_level + (i_split+1) * binsize
        level_sampled = np.where(np.logical_and(np.abs(rawdata) > level_low, np.abs(rawdata) < level_high))[0]
        vstr = "Level {}/{} in range {:.1f}-{:.1f} with {} total samples. ".format(
            i_split, max_level_splits, level_low, level_high, len(level_sampled))

        # If there are samples at this level, make sure none are overlapping
        if len(level_sampled) > 0:
            high_val_decay_samples = int(high_val_decay((level_high+level_low)/2., a=a_gauss/10.))
            repeat_by_n = oversample_by[[x for x in list(oversample_by.keys()) if level_low >= x][-1]]
            if (high_val_decay_samples < len(level_sampled)):
                use_upsampling = False
                vstr += "Adding {} samples. ".format(high_val_decay_samples)
                n_samples_by_level = high_val_decay_samples
                i_max_level = sorted(random.sample(list(level_sampled), k=n_samples_by_level))
            else:
                vstr += "Upsampled {} times (n={}). ".format(repeat_by_n, repeat_by_n*len(level_sampled))
                use_upsampling = True
                if repeat_by_n * len(level_sampled) > high_val_decay_samples:
                    i_max_level = sorted(random.sample(list(np.repeat(level_sampled, repeat_by_n)), k=high_val_decay_samples))
                else:
                    n_samples_by_level = len(level_sampled) * repeat_by_n
                    i_max_level = list(np.repeat(level_sampled, repeat_by_n))
            i_max_level.sort()
            
            if use_upsampling: # just remove the doubles
                doubles = []
                for i, i_m in enumerate(i_max_level):
                    # Remove those too close to each other:
                    diffs = i_max - i_m
                    if len(np.where(np.abs(diffs) < max_overlap)[0]) > 0:
                        doubles.append(i_m)
                i_max_level = [x for x in i_max_level if x not in doubles]
                if verbose == True and len(doubles) > 0:
                    vstr += "Removed {} samples.".format(len(doubles))
            else:
                i_abort = 0
                while True:
                    i_abort += 1
                    no_overlap = True
                    doubles = []
                    # If there's nothing to check against, escape this loop:
                    if len(i_max) == 0:
                        break
                    for i, i_m in enumerate(i_max_level):
                        # Remove those too close to each other:
                        diffs = i_max - i_m
                        if len(np.where(np.abs(diffs) < max_overlap)[0]) > 0:
                            no_overlap = False
                            doubles.append(i_m)
                    i_max_level = [x for x in i_max_level if x not in doubles]
                    # Can get stuck trying to find samples. Aborting gets it close enough:
                    if i_abort >= 100:
                        if verbose:
                            print("Too many iterations to find values. Aborting at n={} values.".format(len(i_max_level)))
                        break
                    # Replace removed samples with new ones:
                    i_max_level += random.sample(list(level_sampled), k=n_samples_by_level-len(i_max_level))
                    i_max_level.sort()
                    if no_overlap == True:
                        break
            
            i_max += i_max_level
            i_max.sort()
            
        if verbose:
            print(vstr)
    
    # --------------------------------------
    #     SAMPLE FROM THE QUIET TIMES
    # --------------------------------------
    for i_split in range(level_splits-1, -1, -1):
        level_low, level_high = i_split * binsize, (i_split+1) * binsize
        level_sampled = np.where(np.logical_and(np.abs(rawdata) > level_low, np.abs(rawdata) < level_high))[0]

        n_samples_at_level = int(norm_x[i_split])
        if n_samples_at_level > len(level_sampled):
            if verbose:
                print("--- Not enough samples available ({}) to reach n={}. Lowering to n available.".format(len(level_sampled), 
                                                                                                             n_samples_at_level))
                print("--- (Reduce a_gauss variable to avoid this problem.)")
            n_samples_at_level = len(level_sampled)
        i_quiet_level = sorted(random.sample(list(level_sampled), k=n_samples_at_level))
        if verbose:
            print("Level {}/{} in range {:.1f}-{:.1f} with {} total samples, {} subsampled ({:.1f}%)".format(
                i_split, level_splits, level_low, level_high, len(level_sampled), n_samples_at_level,
                100.*n_samples_at_level/len(level_sampled)))
        i_abort = 0
        while True:
            i_abort += 1
            no_overlap = True
            doubles = []
            for i, i_m in enumerate(i_quiet_level):
                # Remove those too close to each other:
                diffs = sorted(i_quiet + i_max) - i_m
                if len(np.where(np.abs(diffs) < max_overlap)[0]) > 0:
                    no_overlap = False
                    doubles.append(i_m)
            i_quiet_level = [x for x in i_quiet_level if x not in doubles]
            
            # The loop can get stuck trying to find samples. Aborting gets it close enough:
            if i_abort >= 1000:
                if verbose:
                    print("Too many iterations to find values. Aborting at n={} values.".format(len(i_quiet_level)))
                break
                
            # Replace removed samples with new ones:
            i_quiet_level += random.sample(list(level_sampled), k=n_samples_at_level-len(i_quiet_level))
            i_quiet_level.sort()
            
            # Exit condition: no more overlapping samples
            if no_overlap == True:
                break

        i_quiet += i_quiet_level
        i_quiet.sort()
            
    return i_quiet, i_max, max_level_splits