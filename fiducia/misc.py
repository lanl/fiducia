#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:25:05 2019

Miscellaneous utilities

@author: Pawel M. Kozlowski
"""

# python modules
import numpy as np
# import astropy.units as u
import copy
import matplotlib.pyplot as plt

# listing all functions declared in this file so that sphinx-automodapi
# correctly documents them and doesn't document imported functions.
__all__ = ["find_nearest",
           "areDataFramesCompatible",
           ]

# solid angles of channels 1 thru 18 (from C Sorce)
# This is in steradians (added by DHB 3/22/2019)
solidAngles = [4.48e-6,
               2.76e-6,
               2.76e-6,
               2.76e-6,
               7.07e-6,
               7.07e-6,
               7.07e-6,
               7.07e-6,
               7.07e-6,
               7.07e-6,
               7.07e-6,
               7.07e-6,
               7.07e-6,
               7.07e-6,
               7.07e-6,
               7.07e-6,
               7.07e-6,
               7.07e-6]

# radius of OMEGA chamber
chamberRadius = 122.2 #* u.cm

# standoff distance of detector area
standoff = 436.1

def find_nearest(array, value):
    r"""
    Find nearest value in array and return index, and value as a tuple.
    
    Parameters
    ----------
    array: list, numpy.ndarray
        Array of values to be searched.
        
    value: int, float
        Value for which this function will find the nearest value
        in the array.
    
    Returns
    -------
    idx: int
        Index at which nearest value to input value occurs in the array.
    
    array[idx]: int, float
        The nearest value to the input value.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


#TODO Add support for channels of type ndarray
def areDataFramesCompatible(channels, *frames):
    r"""
    Check DataFrame compatibility for specified channels.
    
    Checks if multiple pandas.core.frame.DataFrame objects are compatible and have
    the channels that are requested. Checks that the DataFrames span the same
    energy range. Returns true if the frames pass all checks, false otherwise.
    
    Parameters
    ----------
    channels: list
        List of relevant channels
    
    *frames: pandas.core.frame.DataFrame
        The DataFrames that you want to check for compatiblity with the relevant
        channels
        
    Returns
    -------
    bool
        True if frames are compatible with the requested channels, and False
        otherwise.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    #deep copy so we don't modify the passed channel list
    ch = copy.deepcopy(channels)
    ch.append('Energy(eV)')
    cols = []
    frameShapes = []
    energyCols = []
    for frame in frames:
        cols.append(set(frame.columns.values))
        frameShapes.append(frame.shape)
        #check if the frame has an Energy(eV) col
        if not 'Energy(eV)' in frame.columns:
            #if not stop here, it definitely wont pass the next tests
             return False
        energyCols.append(frame['Energy(eV)'].values)
     
    #check if all frames have 2d shape
    for shape in frameShapes:
        if len(shape) != 2:
            return False
    
    #check if the frames have the channels we need
    overlap = set.intersection(*cols)
    if not set(ch).issubset(overlap):
        return False
    
    #check if the energy columns are equal
    energyColToChangeAgainst = energyCols[0]
    for energyCol in energyCols:
        #check if the energy column lengths are equal
        if len(energyColToChangeAgainst) != len(energyCol):
            return False
        #check if the energy column elements are equal
        if not np.array_equal(energyColToChangeAgainst, energyCol):
            return False
    
    #no more tests to run, DataFrames are compatible with the relevant channels
    return True

def planckian(E, T, plot=False):
    r"""
    Calculate the Planckian spectral radiance.
    
    Parameters
    ----------
    E: numpy.ndarray
        Energy values (eV).
    T: float
        Temperature (eV).
    
    Returns
    -------
    numpy.ndarray
        Planckian spectral radiance values (W/sr/m^2/eV).
    
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    h = 4.135667696e-15 # eV*s
    c = 299792458 # m/s
    k = 8.617333262e-5 # eV/K
    E_J = E * 1.60218e-19 # convert eV to J
    # T_K = T / k # convert eV to K
    B_E = (2 * E**3)*1.60218e-19 / (h**3 * c**2) * 1/(np.exp(E/T) - 1)

    if plot:
        plt.figure(6)
        plt.plot(E, B_E, label = 'Planckian at 100 eV')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Spectral Radiance (W/sr/m^2/eV)')
        plt.title("Planckian Spectrum at 100 eV")
        # plt.xlim(0, 3000)
        # plt.ylim(0, 1.2e15)
        plt.show()

    return B_E