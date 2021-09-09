#!/usr/bin/env python
'''
--------------------------------------------------------------------
sw2gic/tools.py
General tools for running the code.

Created July 2021 by R Bailey, ZAMG Conrad Observatory (Vienna).
Last updated July 2021.
--------------------------------------------------------------------
'''

import os, sys

import calendar
from datetime import datetime, timedelta
from dateutil import tz
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import pearsonr

from sklearn.metrics import mean_squared_error, confusion_matrix

BOLD_s, BOLD_e = '\033[1m', '\033[0m'

# *******************************************************************
#                TOOLS FOR DATA HANDLING
# *******************************************************************

def extract_local_time_variables(dtime):
    """Takes the UTC time in numpy date format and 
    returns local time and day of year variables, cos/sin.

    Parameters:
    -----------
    dtime : np.array
        Contains UTC timestamps in datetime format.

    Returns:
    --------
    sin_DOY, cos_DOY, sin_LT, cos_LT : np.arrays
        Sine and cosine of day-of-yeat and local-time.
    """

    utczone = tz.gettz('UTC')
    cetzone = tz.gettz('CET')
    # Original data is in UTC:
    dtimeUTC = [dt.replace(tzinfo=utczone) for dt in dtime]
    # Correct to local time zone (CET) for local time:
    dtimeCET = [dt.astimezone(cetzone) for dt in dtime]
    dtlocaltime = np.array([(dt.hour + dt.minute/60. + dt.second/3600.) for dt in dtimeCET]) / 24.
    # add_day calculation checks for leap years
    add_day = np.array([calendar.isleap(dt.year) for dt in dtimeCET]).astype(int)
    dtdayofyear = (np.array([dt.timetuple().tm_yday for dt in dtimeCET]) + dtlocaltime) / (365. + add_day)
    
    sin_DOY, cos_DOY = np.sin(2.*np.pi*dtdayofyear), np.cos(2.*np.pi*dtdayofyear)
    sin_LT, cos_LT = np.sin(2.*np.pi*dtlocaltime), np.cos(2.*np.pi*dtlocaltime)

    return sin_DOY, cos_DOY, sin_LT, cos_LT


def extract_time_from_pos(ar, start, end, zero_time=datetime(1995,1,1), axis=0):
    '''Given an array ar and start and end times, the function returns the array
    sliced within the time ranges.
    
    Parameters:
    -----------
    ar : np.array
        1D (axis=0) or 2D (axis=1) array with time series minute data.
    start : datetime.datetime
        Start time of slice.
    emd : datetime.datetime
        End time of slice.
    zero_time : datetime.datetime (default=datetime(1995,1,1))
        Time of index 0 of array.
    axis : int (default=0)
        0 for 1D arrays, 1 for 2D arrays (time series should be 2nd index).
        
    Returns:
    --------
    ar : np.array
        Slice of array within time range.
        '''
    
    n_min_pos_start = int((start - zero_time).total_seconds()/60.)
    n_min_pos_end = int((end - zero_time).total_seconds()/60.)
    if axis == 0:
        return ar[n_min_pos_start:n_min_pos_end]
    elif axis == 1:
        return ar[:,n_min_pos_start:n_min_pos_end]
    else:
        raise Exception("Incorrect value for 'axis' in extract_time_from_pos()!")

        
# *******************************************************************
#               FUNCTIONS FOR CALCULATING METRICS
# *******************************************************************
    
def calc_event_rates(confusion_matrix):
    '''Takes a confusion matrix as produced by sklearn and returns a dictionary
    with common event counts and the True Skill Score + Heidke Skill Score.'''
    
    TN, FP, FN, TP = confusion_matrix.ravel()
    
    metrics = {}
    
    metrics['TN'] = TN
    metrics['FP'] = FP
    metrics['FN'] = FN
    metrics['TP'] = TP
    
    TPR = TP / (TP + FN)
    metrics['TPR'] = TPR
    metrics['POD'] = TPR
    FPR = FP / (FP + TN)
    metrics['FPR'] = FPR
    metrics['POFD'] = FPR
    metrics['FNR'] = FN / (TP + FN)
    
    metrics['TS'] = TP / (TP + FP + FN)
    metrics['TSS'] = TPR - FPR
    metrics['BS'] = (TP + FP) / (TP + FN)
    metrics['HSS'] = 2*(TP*TN - FN*FP) / ( (TP + FN)*(FN + TN) + (TP + FP)*(FP + TN) )
    
    return metrics
    
    
def compute_CM(y_rt, y_pred, y_pers, threshold=40., verbose=True, enc=False):
    '''Calculates the confusion matrix for data after separating into events/non-events
    according to the defined threshold. All input arrays must have the same size.
    
    Parameters:
    -----------
    y_rt : np.array
        Array of observed values.
    y_pred : np.array
        Array of predicted values.
    y_pers : np.array
        Array of values from persistence model.
    threshold : float (default=40)
        Threshold by which to differentiate events from non-events (event is >= threshold).
    verbose : bool (default=True)
        Set to True to have nicely formatted results printed.
    env : bool (default=False)
        If set to True, data is assumed to be already encoded into event/non-events.
        
    Returns:
    --------
    CM, CM_pers : np.array, np.array
        CM : confusion matrix of predicted values with shape (2,2)
        CM_pers : confusion matrix of persistence values with shape (2,2)
        If threshold is set too high/low, the function returns nans.
    '''
    if enc == False:
        y_rt_events = np.zeros(y_rt.shape)
        y_pred_events = np.zeros(y_pred.shape)
        y_pers_events = np.zeros(y_pers.shape)
        y_rt_events[np.abs(y_rt) >= threshold] = 1
        y_pred_events[np.abs(y_pred) >= threshold] = 1
        y_pers_events[np.abs(y_pers) >= threshold] = 1
    else:
        y_rt_events = y_rt
        y_pred_events = y_pred
        y_pers_events = y_pers
    CM = confusion_matrix(y_rt_events, y_pred_events)
    CM_pers = confusion_matrix(y_rt_events, y_pers_events)
    
    try:
        TN, TP, FN, FP = CM[0,0], CM[1,1], CM[1,0], CM[0,1]
        TN_p, TP_p, FN_p, FP_p = CM_pers[0,0], CM_pers[1,1], CM_pers[1,0], CM_pers[0,1]
    except:
        if verbose:
            print("Cannot compute confusion matrix at threshold {}.".format(threshold))
        return np.full((2,2), np.nan), np.full((2,2), np.nan)
    
    if verbose:
        POD = TP / (TP + FN)
        POFD = FP / (FP + TN)
        POD_pers = TP_p / (TP_p + FN_p)
        POFD_pers = FP_p / (FP_p + TN_p)

        POD_str, POD_pers_str = "{:.2f}".format(POD), "{:.2f}".format(POD_pers)
        POFD_str, POFD_pers_str = "{:.2f}".format(POFD), "{:.2f}".format(POFD_pers)
        if POD > POD_pers: POD_str = BOLD_s+POD_str+BOLD_e
        else: POD_pers_str = BOLD_s+POD_pers_str+BOLD_e
        if POFD < POFD_pers: POFD_str = BOLD_s+POFD_str+BOLD_e
        else: POFD_pers_str = BOLD_s+POFD_pers_str+BOLD_e
        print("POD = {} ({})\nPOFD = {} ({})".format(POD_str, POD_pers_str, POFD_str, POFD_pers_str))
    
    return CM, CM_pers

    
def compute_metrics(y_rt, y_pred, y_pers):
    '''Calculates the RMSE and Pearson's Correlation Coefficient.
    
    Parameters:
    -----------
    y_rt : np.array
        Array of observed values.
    y_pred : np.array
        Array of predicted values.
    y_pers : np.array
        Array of values from persistence model.
        
    Returns:
    --------
    None, just prints a nice string.
    '''
    rmse = np.sqrt(mean_squared_error(y_rt, y_pred))
    pcc = pearsonr(np.squeeze(y_rt), np.squeeze(y_pred))[0]
    rmse_pers = np.sqrt(mean_squared_error(y_rt, y_pers))
    pcc_pers = pearsonr(np.squeeze(y_rt), np.squeeze(y_pers))[0]
    
    rmse_str = "{:.2f}".format(rmse)
    rmse_pers_str = "{:.2f}".format(rmse_pers)
    pcc_str = "{:.2f}".format(pcc)
    pcc_pers_str = "{:.2f}".format(pcc_pers)
    
    if rmse < rmse_pers: rmse_str = BOLD_s+rmse_str+BOLD_e
    else: rmse_pers_str = BOLD_s+rmse_pers_str+BOLD_e
    if pcc > pcc_pers: pcc_str = BOLD_s+pcc_str+BOLD_e
    else: pcc_pers_str = BOLD_s+pcc_pers_str+BOLD_e
                             
    print("RMSE = {} ({}) mV/km\nPCC = {} ({})".format(rmse_str, rmse_pers_str, pcc_str, pcc_pers_str))


def plot_on_time_series(y_rt, y_pred, y_pers, year=2017):
    '''For quick visualisation of results on a time series.
    
    Parameters:
    -----------
    y_rt : np.array
        Array of observed values.
    y_pred : np.array
        Array of predicted values.
    y_pers : np.array
        Array of values from persistence model.
        
    Returns:
    --------
    None, just prints a nice string.
    '''
    # Time range:
    if year == 2017:
        start_rt = datetime(2017,1,1)+timedelta(hours=2)
        xlims = (datetime(2017,8,15), datetime(2017,9,30))
        xlims_cut = (datetime(2017,9,6), datetime(2017,9,11))
        title_str = "Sept 2017"
    elif year == 2000:
        start_rt = datetime(2000,1,1)+timedelta(hours=2)
        xlims = (datetime(2000,6,26), datetime(2000,8,5))
        xlims_cut = (datetime(2000,7,13), datetime(2000,7,19))
        title_str = "July 2000"
    run_model_every = 15 # minutes

    # Find times for every point of run_model_every:
    t = [start_rt + timedelta(minutes=run_model_every*n) for n in range(len(y_rt))]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    for ax in (ax1, ax2):
        ax.plot_date(t, y_pred, 'b-', lw=1, label="Prediction")
        ax.plot_date(t, y_rt, 'r-', lw=1, label="True")
        ax.plot_date(t, y_pers, 'k--', lw=1, alpha=0.5, label="Persistence")
        ax.legend()
        ax.set_xlabel("Time [UTC]")
        ax.set_ylabel("E-field [mV/km]")
    ax1.set_xlim((t[0], t[-1]))
    ax1.set_xlim(xlims)
    ax2.set_xlim(xlims_cut)
    ax1.set_title("Geoelectric field vs. ML predictions for {} Storm".format(title_str))
    plt.show()