#!/usr/bin/env python
'''
--------------------------------------------------------------------
sw2gic/gic.py
Functions for handling GIC data and modelling.

Created July 2021 by R Bailey, ZAMG Conrad Observatory (Vienna).
Last updated August 2021.
--------------------------------------------------------------------
'''

import os, sys

from datetime import datetime, timedelta
import numpy as np
from matplotlib.dates import date2num, num2date

import pandas as pd

# *******************************************************************
#                 FUNCTIONS FOR MODELLING GICS
# *******************************************************************

def calc_ab_for_GIC_from_E(Ex, Ey, GIC):
    '''Taken from Pulkkinen et al. (2007), EPS, Eqs. (4-5):
    Determination of ground conductivity and system parameters for optimal modeling of 
    geomagnetically induced current flow in technological systems.
    All input arrays should have the same shape!
    (Provides basically the same numbers as a least-square linear fit.)
    
    Parameters:
    -----------
    Ex : np.array
        The geoelectric field, northward component
    Ey : np.array
        The geoelectric field, eastward component
    GIC : np.array
        Measured GICs

    Returns:
    --------
    (a, b) : (float, float)
        The parameters a and b for the equation GIC = a*Ex + b*Ey.
    '''
    ExEy = np.nanmean(Ex*Ey)
    ExEx = np.nanmean(Ex*Ex)
    EyEy = np.nanmean(Ey*Ey)
    denominator = (ExEy**2. - ExEx * EyEy)
    a = (np.nanmean(GIC*Ey) * ExEy - np.nanmean(GIC*Ex) * EyEy) / denominator
    b = (np.nanmean(GIC*Ex) * ExEy - np.nanmean(GIC*Ey) * ExEx) / denominator
    return (a, b)


def calc_E_using_plane_wave_method(mag_x, mag_y, resistivities, thicknesses, dt=60):
    """
    Calculation of geoelectric field for a 1D layered half-space going into the Earth.
    Takes minute data of horizontal magnetic field measurements and outputs minute values of the 
    geoelectric field magnitude. Vertical fields (z-direction) are ignored.

    Adapted in 2020 from original function by Greg Lucas:
    https://github.com/greglucas/bezpy/blob/master/bezpy/mt/site.py

    Parameters:
    -----------
    mag_x : np.array
        Contains Bx (northward) geomagnetic field measurements. Should not contain nans or data gaps.
    mag_y : np.array
        Contains By (eastward) geomagnetic field measurements. Should not contain nans or data gaps.
    resistivities : list (len=n)
        A list of resistivities in Ohm-m. The last value represents resis. into the Earth (infinity),
        which doesn't have a corresponding layer thickness.
    thicknesses : list (len=n-1)
        A list of the thicknesses (in m) of each resistive layer.
    dt : int (default=60)
        Sampling rate for the FFT in s.
        
    Returns:
    --------
    (Ex_t, Ey_t) : (np.array, np.array)
        Array of northward (x) and eastward (y) geoelectric field variables for the db/dt
        values provided over the same time range.
    """

    # Add buffer to reduce edge effects:
    buffer_len = 100000
    mag_x = np.hstack((np.full(buffer_len, mag_x[0]), mag_x, np.full(buffer_len, mag_x[-1])))
    mag_y = np.hstack((np.full(buffer_len, mag_y[0]), mag_y, np.full(buffer_len, mag_y[-1])))

    N0 = len(mag_x)
    # Round N to the next highest power of 2 (+1 (makes it 2) to prevent circular convolution)
    N = 2**(int(np.log2(N0))+2)

    freqs = np.fft.rfftfreq(N, d=dt)    # d is sample spacing in s

    # Z needs to be organized as: xx, xy, yx, yy
    Z_interp = _calc_Z(freqs, resistivities, thicknesses)

    # Take Fourier Transform of function:
    mag_x_fft = np.fft.rfft(mag_x, n=N)
    mag_y_fft = np.fft.rfft(mag_y, n=N)

    # Multiply each frequency component by the transfer function:
    Ex_fft = Z_interp[0, :]*mag_x_fft + Z_interp[1, :]*mag_y_fft
    Ey_fft = Z_interp[2, :]*mag_x_fft + Z_interp[3, :]*mag_y_fft

    # Inverse Fourier transform:
    Ex_t = np.real(np.fft.irfft(Ex_fft)[:N0])
    Ey_t = np.real(np.fft.irfft(Ey_fft)[:N0])

    # Remove buffers around edges:
    Ex_t = Ex_t[buffer_len:-buffer_len]
    Ey_t = Ey_t[buffer_len:-buffer_len]

    return Ex_t, Ey_t


def _calc_Z(freqs, resistivities, thicknesses):
    '''
    Calculates the Z array for contributions of the subsurface resistive layers
    and geomagnetic field components to the geoelectric field.

    Called by calc_E_using_plane_wave_method().
    Taken (in 2020) from https://github.com/greglucas/bezpy/blob/master/bezpy/mt/site.py
    '''

    MU = 1.2566370614*1e-6

    freqs = np.asarray(freqs)

    n = len(resistivities)
    nfreq = len(freqs)

    omega = 2*np.pi*freqs
    complex_factor = 1j*omega*MU

    # eq. 5
    k = np.sqrt(1j*omega[np.newaxis, :]*MU/resistivities[:, np.newaxis])

    # eq. 6
    Z = np.zeros(shape=(n, nfreq), dtype=np.complex)
    # DC frequency produces divide by zero errors
    with np.errstate(divide='ignore', invalid='ignore'):
        Z[-1, :] = complex_factor/k[-1, :]

        # eq. 7 (reflection coefficient at interface)
        r = np.zeros(shape=(n, nfreq), dtype=np.complex)

        for i in range(n-2, -1, -1):
            r[i, :] = ((1-k[i, :]*Z[i+1, :]/complex_factor) /
                       (1+k[i, :]*Z[i+1, :]/complex_factor))
            Z[i, :] = (complex_factor*(1-r[i, :]*np.exp(-2*k[i, :]*thicknesses[i])) /
                       (k[i, :]*(1+r[i, :]*np.exp(-2*k[i, :]*thicknesses[i]))))

    # Fill in the DC impedance as zero
    if freqs[0] == 0.:
        Z[:, 0] = 0.

    # Return a 3d impedance [0, Z; -Z, 0]
    Z_output = np.zeros(shape=(4, nfreq), dtype=np.complex)
    # Only return the top layer impedance
    # Z_factor is conversion from H->B, 1.e-3/MU
    Z_output[1, :] = Z[0, :]*(1.e-3/MU)
    Z_output[2, :] = -Z_output[1, :]

    return Z_output


def prepare_B_for_E_calc(df, xkey='Bx', ykey='By', return_time=False):
    '''Prepares geomagnetic field data from a DataFrame object with
    time series index by filling gaps and nan values as well as
    subtracting the mean.
    The values returned are interpolated to be periodic timesteps.
    
    Parameters:
    -----------
    df : pandas.DataFrame object
        DataFrame containing geomagnetic field measurements in x and y.
    xkey : str (default='Bx')
        Key in df for Bx measurements.
    ykey : str (default='By')
        Key in df for By measurements.
    return_time : bool (default=False
        A list of the thicknesses (in m) of each resistive layer.
        
    Returns:
    --------
    (mag_x, mag_y) : (np.array, np.array)
        Arrays of northward (x) and eastward (y) geomagnetic field values.
        If return_time=True, time is added to end in matplotlib date2num form.
    '''
    
    mag_x, mag_y = df[xkey].to_numpy(), df[ykey].to_numpy()
    # Remove nans and linearly interpolate over them:
    try:
        nan_inds = np.isnan(mag_x)
        mag_x[nan_inds] = np.interp(nan_inds.nonzero()[0], (~nan_inds).nonzero()[0], mag_x[~nan_inds])
    except:
        pass
    try:
        nan_inds = np.isnan(mag_y)
        mag_y[nan_inds] = np.interp(nan_inds.nonzero()[0], (~nan_inds).nonzero()[0], mag_y[~nan_inds])
    except:
        pass
    # Subtract the mean
    mag_x, mag_y = mag_x - np.mean(mag_x), mag_y - np.mean(mag_y)

    # Interpolate so that every point in time is covered:
    mag_time = np.array(df.index)
    mag_time_num = date2num(mag_time)
    n_mins = int((pd.to_datetime(mag_time[-1]) - pd.to_datetime(mag_time[0])).total_seconds()/60.) + 1
    mag_start = pd.to_datetime(mag_time[0])
    new_time = date2num(np.arange(mag_start, mag_start+timedelta(minutes=n_mins), timedelta(minutes=1)).astype(datetime))
    mag_x, mag_y = np.interp(new_time, mag_time_num, mag_x), np.interp(new_time, mag_time_num, mag_y)
    
    if return_time:
        return (mag_x, mag_y, new_time)
    else:
        return (mag_x, mag_y)