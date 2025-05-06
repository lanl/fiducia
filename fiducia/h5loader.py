# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 15:42:39 2025

Module for importing newest h5 formatted Dante shots

@author: Daniel H. Barnak
"""

# python modules
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from itertools import chain

def h5_import(filepath):
    """
    imports h5 file and checks for the proper formatting by reading data keys
    and attributes. Searches for attributes will be hard-coded to this module,
    so this is the first check to see if there are errors associated with it.

    Parameters
    ----------
    filepath : str or Path object
        Location and name of the H5 Dante file.

    Returns
    -------
    data : h5py object
        The H5 file heirarchy and datasets.

    """
    with h5py.File(filepath) as f:
        dataImp = f
        try:
            # get detector positions
            keys = np.array(list(dataImp.keys()), dtype = 'str') 
            attSearch = dataImp[keys[0]][keys[0] + " Baseline"]
            # get all attributes of first position
            attrs = np.array(list(attSearch.attrs.keys()))
        except:
            print("Data keys and attributes location may have changed! Check file"+
                  "formatting")
    
    return dataImp
    
def make_frames(filepath):
    """
    Function that returns the raw attributes and scope traces of each detector
    position. This makes it easier to call and access the raw H5 data in
    Pandas.

    Parameters
    ----------
    h5File : h5py object
        The imported H5 file containing the raw Dante data.

    Returns
    -------
    attrsFrame : DataFrame
        Pandas array containing all of the attributes (scope settings) for each
        detector position.
    dataFrame : TYPE
        Pandas array containing the raw digitized data for both the shot data
        and preshot backgrounds.

    """
    with h5py.File(filepath, 'r') as f:
        # uses the same commands as the import function
        h5File = f
        # import will throw an exception if these don't work
        keys = np.array(list(h5File.keys()), dtype = 'str') 
        attSearch = h5File[keys[0]][keys[0] + " Baseline"]
        # get all attributes of first position
        attrs = np.array(list(attSearch.attrs.keys()))
        header = np.full((len(attrs), len(keys)), "x" * 1000)
        # set data array dims
        recLen = attSearch.shape[0]
        shotData = np.full((recLen, len(keys)), 1.1)
        baseData = np.full((recLen, len(keys)), 1.1)
        chanOrder = []
        indexer = []
        shotCols = np.full(2*len(keys), "x"*1000)
        baseCols = np.full(2*len(keys), "x"*1000)
        for idx1, pos in enumerate(h5File.keys()):
            # get attributes for position (taken from baseline)
            posAttrs1 = h5File[pos][pos + " Baseline"].attrs
            # get channel number for labeling/sorting
            chanNum = int(posAttrs1["NIF"])
            chanOrder.append(chanNum)
            indexer.append(f"Channel {chanNum}")
            # get data values
            base = h5File[pos][pos + " Baseline"][:]
            shot = h5File[pos][pos + " Shot"][:]
            shotData[:, idx1] = shot
            baseData[:, idx1] = base
            shotCols[idx1] = f"Channel {chanNum} Shot"
            baseCols[idx1] = f"Channel {chanNum} Baseline"
            for idx2, att in enumerate(posAttrs1.keys()):
                header[idx2, idx1] = posAttrs1.get(att)
        # sort baseline and shot data by channel
        shotDataSort = shotData[:, np.argsort(chanOrder)]
        baseDataSort = baseData[:, np.argsort(chanOrder)]
        data = np.concatenate((shotDataSort, baseDataSort), axis = 1)
        # sort indices similarly
        shotCols = shotCols[np.argsort(chanOrder)]
        baseCols = baseCols[np.argsort(chanOrder)]
        cols = np.concatenate((shotCols, baseCols))
        dataFrame = pd.DataFrame(data, columns = cols)
        # construct attrs frame and sort by channels
        attrsFrame = pd.DataFrame(np.array(header), columns = indexer, index = attrs)
        # sort attributes and data by channel number
        attrsFrame = attrsFrame.iloc[:, np.argsort(chanOrder)]
        
        return attrsFrame, dataFrame   

def preamble_regex(position, verbose = False):
    """
    Function for reading the preamble of one detector position and parsing the
    preamble string into useable numbers to reduce the raw digitized data.

    Parameters
    ----------
    position : h5py key
        The position for which to read the preamble from. Each preamble may be
        different for each position.
    verbose : str, optional
        Turn this flag to True for debugging. This will print whether the
        partiuclar part of the string is found and what value was returned from
        the regex search/match. The default is False.

    Returns
    -------
    headerVals : list
        A list of all of the important scope parameters from the preamble. This
        is later read into the headerFrame function to generate the header. The
        header is then used to reduce the data.

    """
    preamble = position.loc["Preamble"]
    # Get values for vertical axis
    # Regular expression to find the value after 'YMULT'
    match = re.search(r'YMULT\s([-+]?\d*\.\d+([eE][-+]?\d+)?)', preamble)
    
    # If a match is found, print the extracted value
    if match:
        yMult = float(match.group(1))
        if verbose:
            print(f"yMult value: {yMult}")
    else:
        if verbose:
            print("yMult value not found.")
    
    # Regular expression to find the value after 'YOFF'
    match = re.search(r'YOFF\s([-+]?\d*\.\d+([eE][-+]?\d+)?)', preamble)
    
    # If a match is found, print the extracted value
    if match:
        yOff = float(match.group(1))
        if verbose:
            print(f"yOff value: {yOff}")
    else:
        if verbose:
            print("yOff value not found.")
        
    # # Regular expression to find y axis units
    # match = re.search(r'YUNIT\s+"([^"]+)"', preamble)
    
    # # If a match is found, print the extracted value
    # if match:
    #     yUnit = match.group(1)
    #     if verbose:
    #         print(f"yUnit value: {yUnit}")
    # else:
    #     if verbose:
    #         print("yUnit value not found.")
        
    # Regular expression to find the value after 'YOFF'
    match = re.search(r'YZERO\s([-+]?\d*\.\d+([eE][-+]?\d+)?)', preamble)
    
    # If a match is found, print the extracted value
    if match:
        yZero = float(match.group(1))
        if verbose:
            print(f"yZero value: {yZero}")
    else:
        if verbose:
            print("yZero value not found.")
        
    # do the same for horizontal axis
    # Regular expression to find the value after 'XINCR'
    match = re.search(r'XINCR\s([-+]?\d*\.\d+([eE][-+]?\d+)?)', preamble)

    # If a match is found, print the extracted value
    if match:
        xIncr = float(match.group(1))
        if verbose:
            print(f"xIncr value: {xIncr}")
    else:
        if verbose:
            print("xIncr value not found.")

    # Regular expression to find the value after 'YOFF'
    match = re.search(r"PT_OFF (\d+)", preamble)

    # If a match is found, print the extracted value
    if match:
        ptOff = int(match.group(1))
        if verbose:
            print(f"ptOff value: {ptOff}")
    else:
        if verbose:
            print("ptOff value not found.")
        
    # # Regular expression to find y axis units
    # match = re.search(r'XUNIT\s+"([^"]+)"', preamble)

    # # If a match is found, print the extracted value
    # if match:
    #     xUnit = match.group(1)
    #     if verbose:
    #         print(f"xUnit value: {xUnit}")
    # else:
    #     if verbose:    
    #         print("xUnit value not found.")
        
    # Regular expression to find the value after 'YOFF'
    match = re.search(r'XZERO\s([-+]?\d*\.\d+([eE][-+]?\d+)?)', preamble)

    # If a match is found, print the extracted value
    if match:
        xZero = float(match.group(1))
        if verbose:
            print(f"xZero value: {xZero}")
    else:
        if verbose:
            print("xZero value not found.")
        
    # Regular expression to find the value after 'NR_PT'
    match = re.search(r"(?<=NR_PT\s)(\d+)", preamble)

    # If a match is found, print the extracted value
    if match:
        pts = int(match.group(1))
        if verbose:
            print(f"Points value: {pts}")
    else:
        if verbose:
            print("Points value not found.")
        
    headerVals = yMult, yOff, yZero, xIncr, ptOff, xZero, ptOff*xIncr
    return headerVals

def total_attenuation(position):
    """
    Function that uses regex to obtain both the dB and multiplier of the
    attenuation applied to each channel. This is then written to the header in
    the headerFrame function.

    Parameters
    ----------
    position : h5py key
        The position for which to read the preamble from. Each preamble may be
        different for each position.

    Returns
    -------
    dbAtten : int
        Integer value of the total attenuation in dB. Not used in data
        reduction.
    xAtten : int
        Integer value of the total attenuation multiplier. This is the value
        that is applied tot he raw traces.

    """
    
    input_string = position.loc["Attenuation Value"]

    # Regular expression to match both dB value and x value
    pattern = r"(-?\d+)\s+dB\s+\((\d+)x\)"

    # Find all matches
    matches = re.findall(pattern, input_string)
    dbAtten, xAtten = (0, 1)
    # Print the extracted values
    for match in matches:
        dbAtten += int(match[0])
        xAtten *= int(match[1])
        
    return dbAtten, xAtten

def make_header(attrsFrame):
    """
    Master function for generating the header file from both the preamble and
    the total attenuation read in from the attributes DataFrame.

    Parameters
    ----------
    attrsFrame : DataFrame
        Pandas array containing all of the attributes for each detector
        position.

    Returns
    -------
    headerFrame : DataFrame
        Pandas array containing all of the important scope parameters extracted
        from the attributes.

    """
    posNum = attrsFrame.shape[1]
    rowNum = 9 # fixed by data elements required
    header = np.zeros((rowNum, posNum))
    for idx, pos in enumerate(attrsFrame.keys()):
        position = attrsFrame[pos]
        header[2:, idx]  = preamble_regex(position)
        _, header[0, idx] = total_attenuation(position)
        header[1, idx] = position["PointsReceived"]
    # make header dataframe
    cols = attrsFrame.columns
    # create row names for header frame
    indexNames = ['Total attenuation',
                  'Record length',
                  'Vertical Scale',
                  'Vertical Offset',
                  'Vertical Zero',
                  'Horizontal Scale',
                  'Horizontal Pts Offset',
                  'Horizontal Zero',
                  'Horizontal Offset']
    headerFrame = pd.DataFrame(header, columns = cols, index = indexNames)
    
    return headerFrame

def voltage_scale(df, headerFrame, plot = False):
    """
    Function for subtracting the baseline from the shot data and rescaling the
    digitized values to diode voltages using the parameters in the header.

    Parameters
    ----------
    df : DataFrame
        Pandas array containg both the preshot background and shot data.
    headerFrame : DataFrame
        Pandas array containing all of the important scope parameters extracted
        from the attributes.
    plot : bool, optional
        Turn this on to plor the results of the data reduction. The default is
        False.

    Returns
    -------
    dfScaled : DataFrame
        Pandas array containing the background subtracted and rescaled diode
        voltages for each channel of Dante.

    """
    # initialize scaled dataframe
    recLen = int(headerFrame.loc["Record length"][0]) # length of time record
    chans = headerFrame.shape[1] # number of channels
    dfScaled = pd.DataFrame(np.zeros((recLen, chans)), columns=headerFrame.columns)
    # pick a position
    for idx, key in enumerate(headerFrame.keys()):
        head1 = headerFrame[key]
        yMult = head1["Vertical Scale"]
        yOff = head1["Vertical Offset"]
        yZero = head1["Vertical Zero"]
        # posBase = dataFrame[key + " Baseline"]
        unscaledBase = df[key + " Baseline"]
        unscaledShot = df[key + " Shot"]
        
        scaleBase = yZero + yMult*(unscaledBase - yOff)
        scaleShot = yZero + yMult*(unscaledShot - yOff)
        
        scaledVolt = scaleShot - scaleBase
    
        dfScaled[key] = scaledVolt

    if plot:
        fig, ax = plt.subplots(1 ,2)
    
        #plot the unscaled data
        ax[0].plot(df)
        ax[0].set_title("Unscaled")
        ax[0].set_xlabel("pts")
        ax[0].set_ylabel("Counts")
    
        # plot scaled data
        ax[1].plot(dfScaled)
        ax[1].set_title("Scaled no bkg")
        ax[1].set_xlabel("pts")
        ax[1].set_ylabel("Volts")
    
        fig.tight_layout()
        plt.show()
        
    return dfScaled

def apply_attenuation(df, headerFrame):
    """
    Function for applying the total attenuation to the background subtracted
    and scaled diode voltages. Please note that you must call this function
    last in the data processing pipeline.

    Parameters
    ----------
    df : DataFrame
        Pandas array of shot data that has been background subtracted and
        scaled to diode voltages. Need to call this data reduction function
        last in the order of operations.
    headerFrame : DataFrame
        Pandas array containing scope information and attenuation values.

    Returns
    -------
    df : DataFrame
        The attenuation corrected shot data. After this step, the DataFrame
        will contain the diode voltages required for unfolds.

    """
    for idx, key in enumerate(headerFrame.keys()):
        head1 = headerFrame[key]
        atten = head1["Total attenuation"]
        df[key] = df[key]*atten
    return df

def times_frame(headerFrame):
    """
    Generates the time base for each channel based on scope parameters passed
    to the header frame.

    Parameters
    ----------
    headerFrame : pandas.core.frame.DataFrame
        Pandas array containing scope information and attenuation values.

    Returns
    -------
    timesFrame : pandas.core.frame.DataFrame
        Pandas array containing the time axis coordinates for each channel of
        Dante. This frame together with the DataFrame returned by voltage_scale
        gives the diode votlages vs time.

    """
    # calculate time offset using time increment and number of offset points
    recLen = int(headerFrame.loc["Record length"][0]) # length of time record
    chans = headerFrame.shape[1] # number of channels
    timeBase = np.zeros((recLen, chans))
    for idx, key in enumerate(headerFrame.keys()):
        head1 = headerFrame[key]
        xIncr = head1["Horizontal Scale"]
        ptOff = head1["Horizontal Pts Offset"]
        xOff = ptOff*xIncr
        # pts = head1["Record length"]
        timeBase[:, idx] = np.linspace(0 - xOff, (recLen * xIncr) - xOff, recLen)
    timesFrame = pd.DataFrame(timeBase, columns=headerFrame.columns)
    
    return timesFrame

def h5_rawProcess(filepath):
    """
    Master function for loading the fully reduced H5 data from files after
    January 1 2025. 

    Parameters
    ----------
    filepath : Path object
        The file path of the H5 file you wish to analyze.

    Returns
    -------
    timesFrame : pandas.core.frame.DataFrame
        The time coordinate for each channel of Dante.
    dfAtten : pandas.core.frame.DataFrame
        The diode voltages of each channel of Dante.
    headerFrame : pandas.core.frame.DataFrame
        An array containing important scope parameters for reducing the raw 
        digitized data.

    """
    # import file
    # h5File = h5_import(filepath)
    
    # import and parse input data
    attrsFrame, df = make_frames(filepath)
    
    # make header file of scope specs
    headerFrame = make_header(attrsFrame)
    
    # test voltage scale function
    dfScale = voltage_scale(df, headerFrame, plot = False)
    
    # test timesFrame function
    timesFrame = times_frame(headerFrame)
    
    # calculate the fully reduced Dante data
    dfAtten = apply_attenuation(dfScale, headerFrame)

    return  timesFrame, dfAtten, headerFrame

def reindex_integers(frame):
    """
    Reindexes Pandas DataFrames with integer values in columns.This provides
    compatibility with other FIDUCIA functions. Use when there are key error
    instances in downstream functions (cspline and pchipSpline).

    Parameters
    ----------
    frame : pandas.core.frame.DataFrame
        Input DataFrame to be converted to integer column labels.

    Returns
    -------
    frame : pandas.core.frame.DataFrame
        Output DataFrame now with integer values column labels.

    """

    strCols = frame.columns
    intCols = strCols.map(lambda x: [int(num) for num in re.findall(r'\d+', x)])
    flattened = pd.Index(list(chain.from_iterable(intCols)))
    frame.columns = flattened
    
    return frame

#############DEVELOPMENT GRAVEYARD#################################
# def attrs_frame(h5File): #DEPRECATED
#     """
#     Generates a dataframe of all attributes associated with each position
#     (Dante channel) from the dataset.

#     Parameters
#     ----------
#     h5File : h5py file object
#         The H5 file imported by the h5_import function.

#     Returns
#     -------
#     attrsFrame : DataFrame
#         A Pandas DataFrame containing all of the attributes associated with
#         each position (Dante channel) .

#     """
#     # uses the same commands as the import function
#     # import will throw an exception if these don't work
#     keys = np.array(list(h5File.keys()), dtype = 'str') 
#     attSearch = h5File[keys[0]][keys[0] + " Baseline"]
#     # get all attributes of first position
#     attrs = np.array(list(attSearch.attrs.keys()))
#     header = np.full((len(attrs), len(keys)), "x" * 1000)
#     # posArr = []
#     chanOrder = []
#     indexer = []
#     for idx1, pos in enumerate(h5File.keys()):
#         # get attributes for position
#         posAttrs1 = h5File[pos][pos + " Baseline"].attrs
#         # get channel number for labeling/sorting
#         chanNum = int(posAttrs1["NIF"])
#         chanOrder.append(chanNum)
#         indexer.append(f"Channel {chanNum}")
#         for idx2, att in enumerate(posAttrs1.keys()):
#             header[idx2, idx1] = posAttrs1.get(att)
#     attrsFrame = pd.DataFrame(np.array(header), columns = indexer, index = attrs)
#     # sort attributes by channel number
#     attrsFrameSort = attrsFrame.iloc[:, np.argsort(chanOrder)]
    
#     return attrsFrameSort
    
# def data_frame(h5File): #DEPRECATED
#     """
#     Function that retrieves both the baseline and shot data in one dataframe.

#     Parameters
#     ----------
#     h5File : h5py object
#         The imported Dante H5 file.
#     attrsFrame : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     dataFrame : DataFrame
#         Pandas dataframe containing both shot data and preshot backgrounds.

#     """
#     # use function to get attrsFrame
#     attrsFrame = attrs_frame(h5File)
#     # make an array of keys
#     keys = np.array(list(h5File.keys()), dtype = 'str') 
#     # get the length of records from the header info
#     recLen = int(np.max(attrsFrame.loc["PointsReceived"]))
#     # set array dims.
#     data = np.full((recLen, 2*len(keys)), 1.1)
#     # set columns for dataframe
#     chanOrder = []
#     indexer = []
#     for idx1, pos in enumerate(h5File.keys()):
#         # create array for sorting positions
#         chanNum = attrsFrame.loc["NIF"]
#         chanOrder.append(int(chanNum))
#         # print(f"Analyzing Channel {chanNum}")
#         indexer.append(f"Channel {chanNum}")
#         base = h5File[pos][pos + " Baseline"][:]
#         shot = h5File[pos][pos + " Shot"][:]
#         data[:, chanNum] = shot
#         data[:, chanNum+len(keys)] = base
#     dataFrame = pd.DataFrame(data, columns = indexer)
#     dataFrame = dataFrame.iloc[:, np.argsort(chanOrder)]
    
#     return dataFrame 