#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:20:37 2019

Utilities for loading DANTE measurement and response function data.

@author: Pawel M. Kozlowski
"""

# python modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sparse


# custom modules
from fiducia.misc import find_nearest
import fiducia.pltDefaults


# listing all functions declared in this file so that sphinx-automodapi
# correctly documents them and doesn't document imported functions.
__all__ = ["cleanupHeader",
           "loadResponses",
           "loadResponseUncertainty",
           "readDanProcessed",
           "signalsAtTime",
           "signalInt",
           "readDanteData",
           ]


def __responseName__(channelNum):
    r"""
    Convenience function for generating response function file name
    given the DANTE channel number.
    
    Parameters
    ----------
    channelNum: int
        DANTE channel number
    
    Returns
    -------
    fileName: str
        The file name of the response file for channel 'channelNum'
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    baseName = 'DanteDataS_1_'
    extension = '.dat'
    fileName = baseName + str(channelNum) + extension
    return fileName


def __readResponse__(channelNum, directory):
    r"""
    Read a single DANTE channel response function file given the channel
    number and path to the directory containing the response function files.
    
    Parameters
    ----------
    channelNum: int
        DANTE channel number
        
    directory: str
        Path to channel response function files
        
    Raises
    ------
    Exception
        If file does not exist.
    
    Returns
    -------
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    try:
        fileName = __responseName__(channelNum)
        responseArr = np.loadtxt(directory + fileName)
        return responseArr
    except IOError:
        # file doesnt exist, return dummy data
        Exception(f"File {fileName} does not exist!")


def cleanupHeader(dataFrame):
    r"""
    Strip whitespace and rename DataFrame headers.

    Parameters
    ----------
    dataFrame : pandas.core.frame.DataFrame
        DataFrame to be cleaned.

    Returns
    -------
    cleanedDataFrame : pandas.core.frame.DataFrame
        DataFrame with stripped and renamed channel headers.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------

    """
    # stripping whitespace from header names
    colNames = dataFrame.columns
    colNamesStripped = [header.strip() for header in colNames]
    renameStripDict = {colName: colNamesStripped[idx]
                       for idx, colName in enumerate(colNames)}
    df1 = dataFrame.rename(columns=renameStripDict)
    # renaming channel headers from e.g. Ch2 to just 2
    allChannels = np.arange(18) + 1
    renameDict = {}
    for channel in allChannels:
        renameDict['Ch' + str(channel)] =  channel

    cleanedDataFrame = df1.rename(columns=renameDict)
    return cleanedDataFrame


def loadResponses(channels, fileName, solid=True):
    r"""
    Load DANTE measurement data from files given the channels and path to the
    directory containing the response function files. Returns a dataframe
    with the data.
    
    Parameters
    ----------
    channels: list, numpy.ndarray
        List or array of relevant channels
        
    fileName: str
        Full path and filename of .csv file containing DANTE respones
        functions.
    
    solid: Bool, optional
        Includes solid angle in response function value if true. The default
        is True.
        
    Returns
    -------
    responseFrame:  pandas.core.frame.DataFrame
        DataFrame with the response function data for the 'channels' requested
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """

    solidAngles = fiducia.misc.solidAngles
    # loading all the response functions
    dataFrame = pd.read_csv(fileName)
    #clean headers
    cleanedFrame = cleanupHeader(dataFrame)
    # filtering for channels we care about
    # for this particular shot
    colFilter = ['Energy(eV)'] + channels
    responseFrame = cleanedFrame[colFilter].copy()
    # convert energy column from strings to floats (if necessary)
    if type(responseFrame['Energy(eV)'][0]) == str:
        energyFloats = responseFrame['Energy(eV)'].str.replace(',', '')
        responseFrame.loc[:,'Energy(eV)'] = energyFloats.astype(float)
    else:
        energyFloats = responseFrame['Energy(eV)'].astype(float)
        responseFrame.loc[:,'Energy(eV)'] = energyFloats

    if solid:
        for chan in channels:
            #multiply each element by the corresponding channel solid angle
            #convert V/GW to V/W for benchmarking against Mathematica Fiducia
            responseFrame.loc[:, chan] *= solidAngles[chan-1]*1e-9
            #save metadata that we already include solid angle
            responseFrame.solid = True

    return responseFrame


def loadResponseUncertainty(responseFrame, fileName):
    r"""
    Load uncertainty percentages into a DataFrame.
    
    Parameters
    ----------
    responseFrame: pandas.core.frame.DataFrame
        DataFrame to base the respones uncertainty frame on. 
        
    fileName: str
        Full path and filename of .csv file containing DANTE response
        uncertainty percentages functions.

    Returns
    -------
    responseUncertaintyFrame : pandas.core.frame.DataFrame
        DataFrame with each column being a channel and each element being the
        channel's uncertainty percentage. Extended to match the photon energy
        range in the response frame.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    #read raw data
    channelUncertaintyFrame = pd.read_csv(fileName)
    #clean headers
    cleanedFrame = cleanupHeader(channelUncertaintyFrame)
    #filtering for columns we care about
    channels = list(responseFrame.columns.values)
    channels.remove('Energy(eV)')
    responseUncertaintyFrame = responseFrame.copy()
    #fill each column with corresponding uncertainty
    for chan in channels:
        #multiply each responseFrame element by the percent uncertainty
        responseUncertaintyFrame.loc[:, chan] *= cleanedFrame.loc[0, chan]/100
    return responseUncertaintyFrame


def readDanProcessed(channels, directory):
    r"""
    Loads DANTE measurement data from files given the channels and path to the
    directory containing the reduced and aligned DANTE data. Returns a
    dataframe with the data.
    
    Note that this is *not* for raw data. It is for reading DANTE signals
    that have already been processed by Dan Barnak's scripts.
    
    Parameters
    ----------
    channels: list, numpy.ndarray
        List or array of relevant channels
        
    directory: str
        Path to channel response function files
    
    Returns
    -------
    dataFrame : pandas.core.frame.DataFrame
        Dataframe of aligned signals from Dan's analysis.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    
    """
    # initialize finder for length of data array for longest channel
    longestChLen = 0
    # load all the arrays into a dict
    responseDict = {}
    for channel in channels:
        responseData = __readResponse__(channel, directory)
        responseDict[channel] = responseData
        # getting length of array for longest channel
        responseLen = len(responseData)
        if responseLen > longestChLen:
            longestChLen = responseLen
    # initialize pandas dataframe
    indices = np.arange(longestChLen)
    dataFrame = pd.DataFrame(index=indices)
    # add index axis label
    dataFrame.index.name = 'indices'
    # write channel response data into dataframe
    for channel in channels:
        chData = responseDict[channel]
        # photon energy
        dataFrame['Time' + str(channel)] = chData[:,0]
        # channel response
        dataFrame['Signal' + str(channel)] = chData[:,1]
    return dataFrame


def signalsAtTime(time,
                  timesFrame,
                  df,
                  channels,
                  plot=False,
                  method="interp"):
    r"""
    Get DANTE signals from each channel at a particular time. Default is
    to return an interpolated value of the signal at the given time.
    Alternatively, this function can return the nearest value in
    the signal data array for the given time.
    
    
    Parameters
    ----------
    time: float
        Time for which we want DANTE signals (in ns).
        
    timesFrame: pandas.core.frame.DataFrame
        Dataframe containing time axis corresponding to dante signals in
        df dataframe. See timesScope() and bkgCorrect().
        
    df: pandas.core.frame.DataFrame
        Dante dataframe with background corrected values and scaled
        to units of volts. See readDanteData(), bkgCorrect() and
        voltageScale().
        
    plot: Bool
        When True, plots DANTE signals vs channel index at a particular time.
        
    method: str
        Either 'nearest' or 'interp'. 'nearest' finds the nearest point in the
        DANTE signal to the given time. 'interp' returns an interpolated
        signal value for the given time. Default is 'interp'.
    
    Returns
    -------
    signals : numpy.ndarray
        Dante signals for each channel at a particular time step.
    
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    chLen = len(channels)
    signals = np.zeros(chLen)
    for idx, channel in enumerate(channels):
        if method == "nearest":
            timeIdx, _ = find_nearest(array=timesFrame[channel],
                                      value=time*1e-9)
            signals[idx] = df[channel][timeIdx]
        elif method == "interp":
            signals[idx] = np.interp(x=time*1e-9,
                                     xp=timesFrame[channel],
                                     fp=df[channel])
        else:
            raise Exception(f"Method {method} not found!")
        
    if plot:
        plt.scatter(channels, signals)
        plt.xticks(channels)
        plt.xlabel('DANTE channel')
        plt.ylabel('Signal (V)')
        plt.title(f'DANTE signals @ t = {time} ns')
        plt.show()
    return signals


def signalInt(channels,
              timesFrame,
              df, 
              tStart, 
              tEnd, 
              npts=1000, 
              method = 'interp'):
    r"""
    Get time-integrated Dante signals for a specified time interval. Used in
    getting time-integrated spectrum from the unfold.
    
    Parameters
    ----------
    timesFrame: pandas.core.frame.DataFrame
        Dataframe containing time axis corresponding to dante signals in
        df dataframe. See timesScope() and bkgCorrect().
        
    df: pandas.core.frame.DataFrame
        Dante dataframe with background corrected values and scaled
        to units of volts. See readDanteData(), bkgCorrect() and
        voltageScale().
        
    tStart: float
        Lower bound for time integration (in ns).
        
    tEnd: float
        Upper bound for time integration (in ns).
        
    npts: int
        Number of points used in computing the integral. The default is 1000
        
    method: str
        Either 'nearest' or 'interp'. 'nearest' finds the nearest point in the
        DANTE signal to the given time. 'interp' returns an interpolated
        signal value for the given time. Default is 'interp'.
    
    Returns
    -------
    signalInt: numpy.ndarray
        Time integrated Dante signals for each channel (in V*s).
    
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    from scipy import integrate
    
    chLen = len(channels)
    signalInt = np.zeros(chLen)
    for idx, channel in enumerate(channels):
        timeseries = timesFrame[channel]*1
        chanseries = df[channel]
        if method == 'nearest':
            timeIdx1, _ = find_nearest(array=timeseries, value=tStart*1e-9)
            timeIdx2, _ = find_nearest(array=timeseries, value=tEnd*1e-9)
            print(timeIdx1, timeIdx2)
            signalInt[idx] = integrate.simps(y=chanseries[timeIdx1:timeIdx2],
                                             x=timeseries[timeIdx1:timeIdx2])
        elif method == 'interp':
            interpTime = np.linspace(tStart*1e-9, tEnd*1e-9, npts)
            signalInterp = np.interp(interpTime, timeseries, chanseries)
            signalInt[idx] = integrate.simps(y=signalInterp,
                                             x=interpTime)
        else:
            raise Exception(f"Method {method} not found!")
                
        
    return signalInt


def readDanteData(filePath):
    r"""
    Reads Dante .dat file and returns header info and channel signals
    as two separate pandas dataframes.
    
    Parameters
    ----------
    filePath: str
        Full path to the Dante .dat file.
    
    Returns
    -------
    headerFrame: pandas.core.frame.DataFrame
        Header of Dante data file. This typically include information
        about the various components used in each Dante channel, such
        as oscilloscopes, XRDs, etc.
    
    dataFrame: pandas.core.frame.DataFrame
        Dante data.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    # reading the entire dante file
    dataAndHeaderFrame = pd.read_csv(filePath, sep='\t', header=None)
    # generating dante dataframe header names
    headerNames1st = [str(num) for num in np.arange(18) + 1]
    headerNames2nd = [str(num) + ' bkg' for num in np.arange(18) + 1]
    headerName = headerNames1st + headerNames2nd
    # replacing dataframe column header names for more intuitive access
    dataAndHeaderFrame.columns = headerName
    # splitting into header frame and measurement data frame
    headerLen = 18
    headerFrame = dataAndHeaderFrame[:][:headerLen]
    dataFrame = dataAndHeaderFrame[:][headerLen:]
    # replacing row names for header frame
    indexNamesReplace = {0:'Signal Cable',
                         1:'Attenuator 1',
                         2:'Attenuator 2',
                         3:'Attenuator 3',
                         4:'Attenuator 4',
                         5:'Jumper Cable',
                         6:'XRD SN',
                         7:'Mirror SN',
                         8:'Filter 1 SN',
                         9:'Filter 2 SN',
                         10:'Filter 3 SN',
                         11:'Fiducial T',
                         12:'Scope type',
                         13:'Full scale Hor time',
                         14:'#Hor Pts',
                         15:'Full Scale Vert mV',
                         16:'HV bias for XRDs',
                         17:'(unused field)'}
    headerFrame.rename(index=indexNamesReplace, inplace=True)
    return headerFrame, dataFrame

def readDanteDataOld(filePath):
    r"""
    Reads Dante .dat file for old format files (no bkg) and returns header
    info and channel signals as two separate pandas dataframes. Produces 
    artificial background dataframe of just zeros.
    
    Parameters
    ----------
    filePath: str
        Full path to the Dante .dat file.
    
    Returns
    -------
    headerFrame: pandas.core.frame.DataFrame
        Header of Dante data file. This typically include information
        about the various components used in each Dante channel, such
        as oscilloscopes, XRDs, etc.
    
    dataFrame: pandas.core.frame.DataFrame
        Dante data.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    # reading the entire dante file
    dataAndHeaderFrame = pd.read_csv(filePath, sep='\t', header=None)
    
    # make background data frame with duplicate header
    bkg = np.zeros_like(dataAndHeaderFrame)
    bkgFrame = pd.DataFrame(bkg)
    headerLen = 17 # old header does not include XRD voltages
    headerFrame = dataAndHeaderFrame[:][:headerLen]
    bkgFrame[:][:headerLen] = headerFrame
    
    master = pd.concat((dataAndHeaderFrame, bkgFrame), axis = 1)
    # generating dante dataframe header names
    headerNames1st = [str(num) for num in np.arange(18) + 1]
    
    # take out non-existant background traces
    headerNames2nd = [str(num) + ' bkg' for num in np.arange(18) + 1]
    headerName = headerNames1st + headerNames2nd
    
    # headerName = headerNames1st
    # replacing dataframe column header names for more intuitive access
    master.columns = headerName
    # splitting into header frame and measurement data frame
    
    headerFrame = master[:][:headerLen]
    dataFrame = master[:][headerLen:]
    
    # replacing row names for header frame
    indexNamesReplace = {0:'Signal Cable',
                         1:'Attenuator 1',
                         2:'Attenuator 2',
                         3:'Attenuator 3',
                         4:'Attenuator 4',
                         5:'Jumper Cable',
                         6:'XRD SN',
                         7:'Mirror SN',
                         8:'Filter 1 SN',
                         9:'Filter 2 SN',
                         10:'Filter 3 SN',
                         11:'Fiducial T',
                         12:'Scope type',
                         13:'Full scale Hor time',
                         14:'#Hor Pts',
                         15:'Full Scale Vert mV',
                         16:'(unused field)'}
    headerFrame.rename(index=indexNamesReplace, inplace=True)
    return headerFrame, dataFrame