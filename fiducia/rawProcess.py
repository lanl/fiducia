#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:43:39 2019

Utilities for processing raw DANTE data. Typical steps include:
    - attenuator correction
    - background shot subtraction
    - channel alignment (via e.g. peak finding)
    - temporal axis calibration

@author: Pawel M. Kozlowski
"""

# python modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
#from skimage import feature

# custom modules
from fiducia.misc import find_nearest
from fiducia.loader import readDanteData
import fiducia.pltDefaults


# listing all functions declared in this file so that sphinx-automodapi
# correctly documents them and doesn't document imported functions.
__all__ = ["noScope",
           "noXRD",
           "onChannels",
           "timesScope",
           "voltageScale",
           "bkgCorrect",
           "offsetCorrect",
           "attenuationFactors",
           "attenuationCorrect",
           "timeAvgBkg",
           "avgBkgCorrect",
           "polyBkg",
           "signalEdges",
           "polyBkgFrame",
           "highestPeak",
           "highestN",
           "getPeaks",
           "alignPeaks",
           "constructMeasurementFrame",
           "loadCorrected",
           "hysteresisCorrect",
           "align",
           ]


def noScope(hf):
    r"""
    Given a header frame, return a list of channels with no scope. These
    are the channels that are off.
    
    Parameters
    ----------
    hf: pandas.core.frame.DataFrame
        Header frame from DANTE measurement data. See readDanteData().
    
    Returns
    -------
    set
        Set of channels corresponding to oscilloscopes marked as "off"
        in the Dante data file header.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    totalChannels = int(np.shape(hf)[1] / 2)
    offScope = []
    for idx in range(totalChannels):
        ch = idx + 1
        chStr = str(ch)
        if hf[chStr]['Scope type'] == 9999:
            offScope.append(ch)
    return set(offScope)


def noXRD(hf):
    r"""
    Given a header frame, return a list of channels with no XRD. If there is
    a scope, then these channels may still register a signal!
    
    Parameters
    ----------
    hf: pandas.core.frame.DataFrame
        Header frame from DANTE measurement data. See readDanteData().
    
    Returns
    -------
    set
        Set of channels corresponding to no XRDs marked in the Dante data file
        header.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    totalChannels = int(np.shape(hf)[1] / 2)
    offXRD = []
    for idx in range(totalChannels):
        ch = idx + 1
        chStr = str(ch)
        if hf[chStr]['XRD SN'] == 0:
            offXRD.append(ch)
    return set(offXRD)


def onChannels(hf):
    r"""
    Given a header frame, return a list of which dante channels
    were on for the shot.
    
    Parameters
    ----------
    hf: pandas.core.frame.DataFrame
        Header frame from DANTE measurement data. See readDanteData().
    
    Returns
    -------
    set
        Set of channels corresponding oscilloscopes and XRDs marked as on
        within the Dante data file header.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    offXRDSet = noXRD(hf)
    offScopeSet = noScope(hf)
    allOff = offXRDSet.union(offScopeSet)
    allChSet = set(np.arange(18) + 1)
    onChSet = allChSet - allOff
    return onChSet


def chStatus(hf):
    r"""
    Generates a list of statuses for each channel (on, no scope, or no XRD).
    
    Parameters
    ----------
    hf: pandas.core.frame.DataFrame
        Header frame from DANTE measurement data. See readDanteData().
    
    Returns
    -------
    status: dict
        Statuses of Dante channels according to data file header.
    
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    offXRDSet = noXRD(hf)
    offScopeSet = noScope(hf)
    allOff = offXRDSet.union(offScopeSet)
    allChArr = np.arange(18) + 1
    allChSet = set(allChArr)
    onChSet = allChSet - allOff
    status = {}
    for chNum in allChArr:
        if chNum in onChSet:
            status[chNum] = "On"
        elif chNum in offXRDSet:
            status[chNum] = "No XRD"
        elif chNum in offScopeSet:
            status[chNum] = "No Scope"
        else:
            status[chNum] = "Off?"
    return status


def timesScope(hf):
    r"""
    Given a headerFrame, returns a timesFrame containing an array of
    oscilloscope times for each channel and background shot in the
    headerFrame.
    
    Parameters
    ----------
    hf: pandas.core.frame.DataFrame
        Header frame from DANTE measurement data. See readDanteData().
    
    Returns
    -------
    timesFrame: pandas.core.frame.DataFrame
        Returns a timesFrame containing the corresponding times for each
        oscilloscope trace contained in the header frame.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    timesFrame = pd.DataFrame(index=np.arange(1024), columns=hf.keys())
    for key in hf.keys():
        timeCode = str(hf[key]['Full scale Hor time'])
        # if the time code is 99, it means the scope was off
        if timeCode == '99':
            print(f"No timespan for channel {key}.")
        # extracting total time spanned by the oscilloscope trace based on the
        # time code. For example, a time code of 208 means 2.0e-8 seconds.
        else:
            # assuming decimal point comes after first number, and all but the last
            # number are part of the significand.
            timeSpan = float(timeCode[0] + '.' + timeCode[1:-1] + 'e-' + timeCode[-1])
            # getting total number of points recorded by oscilloscope
            scopePts = hf[key]['#Hor Pts']
            # getting length of time steps between points by dividing the total time
            # over which oscilloscope trace is recorded by the total number of
            # time steps. There is one fewer step than number of points.
            timeStep = timeSpan / (scopePts - 1)
            # constructing array of times for convience in analysis/plotting
            timesFrame[key] = np.arange(scopePts) * timeStep

    return timesFrame


def voltageScale(hf, df):
    r"""
    Scales voltage (vertical) axis of dante signals based on information
    contained in the header. Returns a dataframe with the dante signals
    in units of volts. Also returns an errors/uncertainties frame
    in units of volts, where the uncertainty due to the 11-bit ADC
    converter has been calculated.
    
    
    Parameters
    ----------
    hf: pandas.core.frame.DataFrame
        Header dataframe from dante .dat file. See readDanteData().
        
    df: pandas.core.frame.DataFrame
        Dante dataframe. See readDanteData().
    
    Returns
    -------
    dfScaled: pandas.core.frame.DataFrame
        Dante dataframe with signals in units of volts.
        
    errFrame: pandas.core.frame.DataFrame
        Correspoding errors for dfScaled. Also in units of volts.
    
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    # number of bits in ADC converter
    bits = 11
    # number of bins/segments that voltage range is divided into. This
    # defines the uncertainty in the measurements due to the oscilloscope
    # and is one less than the total number of points spanning the range
    # of the oscilloscope's ADC.
    bitRange = 2 ** bits - 1
    # initializing scaled dataframe and error frame with same structure
    # as input dataframe.
    dfScaled = pd.DataFrame().reindex_like(df)
    errFrame = pd.DataFrame().reindex_like(df)
    for key in hf.keys():
        # channel voltage scale in units of volts
        maxVoltage = hf[key]['Full Scale Vert mV'] / 1e3
        # scale conversion factor
        countsToVolts = maxVoltage / bitRange
        # converting oscilloscope traces from counts to volts
        dfScaled[key] = df[key] * countsToVolts
        # uncertainties are constant and based on the ADC resolution
        # for each channel
        errFrame[key] = np.ones_like(df[key]) * countsToVolts
    return dfScaled, errFrame


def bkgCorrect(df, timesFrame):
    r"""
    Give a Dante data frame containing measurement data and background
    shot data, remove the background from the data and return the corrected
    data as a dataframe. Note that the returned dataframe is different
    in a few ways from the input dataframe. First, the returned dataframe
    is assumed to have strings as column headers, whereas the returned
    dataframe will have integers (corresponding to dante channel
    number) as the column headers. In addition, the input dataframe
    will start indexing at some number above 0 (usually 18, due to the
    header length), whereas the returned dataframe is re-indexed to
    begin at 0.
    
    A dataframe with corresponding time scales to df is also passed to this
    function for reindexing from strings to integers. This also acts as a
    placeholder in case it is necessary to interpolate values if the
    background shot and measurement shot timescales are not the same. Though
    this type of interpolation is not currently implemented.
    
    
    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        Dataframe of raw dante data. This should contain both the shot
        measurement and the shot background as columns. See readDanteData().
        The columns in this dataframe are assumed to be strings.
        
    timesFrame: pandas.core.frame.DataFrame
        Dataframe containing time axis corresponding to dante signals in
        df dataframe. See timesScope().
    
    Returns
    -------
    timesBkg: pandas.core.frame.DataFrame
        Returns a dataframe of times corresponding to dfCorrected signals.
        The columns in this dataframe are integers corresponding to Dante
        channel number.
    
    dfCorrected: pandas.core.frame.DataFrame
        Returns a dataframe of background subtracted dante signals. The
        columns in this dataframe are integers corresponding to Dante
        channel number.
    
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    # reassign index (dataframe currently starts at 18 or so due to header)
    dfnew = df.set_index(np.arange(np.shape(df)[0]))
    # get total number of channels in dataframe
    totalChannels = int(np.shape(dfnew)[1] / 2)
    # produce a list of all channels
    channels = np.arange(totalChannels) + 1
    # get the total number of datapoints in the dataframe
    dataRange = np.arange(np.shape(dfnew)[0])
    # initialize an empty dataframe for storing background corrected values.
    dfCorrected = pd.DataFrame(index=dataRange, columns=channels)
    # initializing a new timesframe
    timesBkg = pd.DataFrame(index=dataRange, columns=channels)
    for ch in channels:
        # subtract shot background from shot measurement.
        dfCorrected[ch] = dfnew[str(ch)] - dfnew[str(ch) + ' bkg']
        # passing through time values to new frame
        timesBkg[ch] = timesFrame[str(ch)]
    return timesBkg, dfCorrected


def offsetCorrect(df, timesFrame, offsetsFile):
    r"""
    Reads given offset correction file (.xls) and applies offsets to
    dante measurement data given in dataframe. The input dataframe
    should already be background corrected and scaled to units of volts, 
    see bkgCorrect() and voltageScale().
    Note that although timing offsets are also applied, they are not as
    relevant since timing should be realigned to a fiducial peak
    anyway.
    Additional attenuation is not implemented and an error will be raised if
    the offsets file contains attenuation values other than 1.
    Returns a dataframe with applied offsets.
    
    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        Dante dataframe with background corrected values and scaled
        to units of volts. See readDanteData(), bkgCorrect() and
        voltageScale().
        
    timesFrame: pandas.core.frame.DataFrame
        Dataframe containing time axis corresponding to dante signals in
        df dataframe. See timesScope() and bkgCorrect().
    
    
    Returns
    -------
    offsetsFile: str
        Full path to .xls file containing dante channel offsets.
    
    dfOffset: pandas.core.frame.DataFrame
        Dante dataframe with applied offset corrections
    
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    # read excel file with offsets
    offsetsFrame = pd.read_excel(offsetsFile)
    offsets = offsetsFrame.set_index(np.arange(np.shape(offsetsFrame)[0]) + 1)
    # initialize dataframe for storing offset corrected values
    dfOffset = pd.DataFrame().reindex_like(df)
    timeOffset = pd.DataFrame().reindex_like(timesFrame)
    for key in df.keys():
        # correct for channel offset
        # Offsets are added to signal
        dfOffset[key] = df[key] + offsets['OffsetV (V)'][key]
        if offsets['additional attenuation'][key] != 1:
            raise NotImplementedError("Non-zero attenuation in offsets "
                                      "file not implemented!")
        # applying time offsets
        # need to scale by 1e-9 since timesFrame is given in seconds, whereas
        # offsets are given in nanoseconds.
        timeOffset[key] = timesFrame[key] + 1e-9 * offsets['OffsetT (ns)'][key]
    return timeOffset, dfOffset


def __getAttenuation__(attenuatorsFrame, serialNum):
    r"""
    Given dataframe with attenuator info and a serial number,
    returns the attenuation factor
    
    Parameters
    ----------
    attenuatorsFrame: 
        
        
    serialNum: 
    
        
    Returns
    -------
    factorVal: 
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    factor = attenuatorsFrame[attenuatorsFrame['#'] == serialNum]['Factor']
    factorVal = factor.values
    return factorVal


def attenuationFactors(hf, channels, attenuatorsPath):
    r"""
    Given a header frame, return the attenuation factors applied to each
    channel.
    
    Parameters
    ----------
    hf: pandas.core.frame.DataFrame
        Pandas dataframe containing dante header information. See
        readDanteData().
    
    channels: set
        Set of channels to be analyzed.
        
    attenuatorsPath: str
        Full path to excel file containing attenuator serial numbers and
        corresponding attenuation factors.
    
    Returns
    -------
    chFactors
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    # load attenuators dataframe
    attenuatorsFrame = pd.read_excel(attenuatorsPath)
    # list of pandas dataframe indices containing attenuator serial numbers.
    attenuatorIdxs = ['Attenuator 1',
                      'Attenuator 2',
                      'Attenuator 3',
                      'Attenuator 4']
    # initializing array to store attenuation factors corresponding to each
    # channel.
    chFactors = pd.DataFrame(1, index=['Attenuation Factor'], columns=channels)
    # cycling over channels
    for ch in channels:
        # getting attenuator serial numbers
        serials = hf[str(ch)][attenuatorIdxs]
        # filtering null values (no attenuator)
        realSerials = serials[serials != 0]
        # fetching attenuation factors
        factors = [__getAttenuation__(attenuatorsFrame, sn) for sn in realSerials]
        # product sum of attenuation factors and saving to pandas
        # dataframe. If the list is empty then prod will return a value
        # of 1, which is exactly what we want!
        chFactors[ch] = np.prod(factors)
    return chFactors


def attenuationCorrect(attenuatorsFile, hf, df, channels):
    r"""
    Given a Dante data frame and header frame, return a data frame with
    attenuation corrections applied to each channel.
    
    Parameters
    ----------
    attenuatorsFile: str
        Full path to .xls file containing attenuator serial numbers
        and corresponding attenuation factors. See attenuationFactors().
    
    hf: pandas.core.frame.DataFrame
        Header dataframe from dante .dat file. See readDanteData().
        
    df: pandas.core.frame.DataFrame
        Dante dataframe. This frame should already be voltage scaled,
        background corrected, and offset corrected. See readDanteData().
        
    channels: list
        List of dante channels in df to be analyzed.
    
    Returns
    -------
    dfAtten: pandas.core.frame.DataFrame
        Returns dataframe with attenuation corrected signal values for
        the given channels.
    
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    
    """
    # getting attenuation factors
    attFactors = attenuationFactors(hf, channels, attenuatorsFile)
    # initializing dataframe for attenuation corrected values
    dfAtten = pd.DataFrame(index=np.arange(len(df)),
                           columns=channels)
    for ch in channels:
        # applying attenuation factors
        dfAtten[ch] = df[ch] * attFactors[ch]['Attenuation Factor']
    return dfAtten


def timeAvgBkg(times, signals, timeStart, timeEnd):
    r"""
    Calculates time averaged background for given data.
    
    Parameters
    ----------
    times:
        
        
    signals:
        
        
    timeStart:
        
        
    timeEnd:
        
    
    Returns
    -------
    avg: 
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    # getting nearest indices to start and stop times
    idxStart, valueStart = find_nearest(times, timeStart)
    idxEnd, valueEnd = find_nearest(times, timeEnd)
    # averaging
    avg = np.mean(signals[idxStart:idxEnd])
    return avg


def avgBkgCorrect(timesFrame, df, channels, timeLength=1e-9):
    r"""
    Applies background correction to bring the signal down to zero, based
    on averaging the signal background over a section of time from earliest
    time contained in timesFrame to earliest time plus timeLength.
    
    Parameters
    ----------
    timesFrame: pandas.core.frame.DataFrame
        Dataframe of time axis values corresponding to signals in df. These
        should be in units of seconds.
        
    df: pandas.core.frame.DataFrame
        Dataframe of dante signals. These should already be attenuation
        corrected and in units of volts.
        
    channels: set
        Set of channels to be analyzed.
    
    timeLength: float
        Duration of time from initial time over which to take the average. In
        units of seconds.
        
    dfAvg: pandas.core.frame.DataFrame
        Returns a dataframe containing average background corrected
        signals.
    
    Returns
    -------
    dfAvg:
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    dfAvg = pd.DataFrame().reindex_like(df)
    for ch in channels:
        # getting earliest time in frame.
        timeStart = timesFrame[ch][0]
        # added duration to earliest time to get end time for average.
        timeEnd = timeStart + timeLength
        # calculating average background by grabbing a section of the signal
        avgBkg = timeAvgBkg(times=timesFrame[ch].values,
                            signals=df[ch].values,
                            timeStart=timeStart,
                            timeEnd=timeEnd)
        # applying correction
        dfAvg[ch] = df[ch] - avgBkg
    return dfAvg


def polyBkg(time,
            signal,
            lowerEdge,
            upperEdge,
            order=3,
            lowerLength=None,
            upperLength=None,
            plot=False):
    r"""
    Fit polynomial function to ends of the signal as an estimate of the
    background signal + hyesteresis. Default is cubic fit.
    
    Parameters
    ----------
    time: numpy.ndarray
        array of times corresponding to signal
        
    signal: numpy.ndarray
        array of signal values for a single dante channel
        
    lowerEdge: int
         Index of time array corresponding to lower edge of detected
         signal. See signalEdges().
        
    upperEdge: int
        Index of time array corresponding to upper edge of detected
        signal. See signalEdges().
        
    order: int
        Order of polynomial to be fitted to estimated background/hysteresis.
        
    lowerLength: int
        Length over which to take the polynomial background fit on the
        lower end (earlier in time) segment of the signal, with respect
        to lowerEdge. Default is None, which just takes the first point
        in the signal.
        
    upperLength: int
        Length over which to take the polynomial background fit on the
        upper end (later in time) segment of the signal, with respect
        to upperEdge. Defualt is None, which then just picks the second
        to last point in the signal.
        
    plot: bool
        Flag for plotting polynomial fitted background signal. Default
        is False.
        
    Returns
    -------
    time:
        
        
    fitSignal: 
        
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    if lowerLength == None:
        lowerMin = 0
    else:
        lowerMin = lowerEdge - lowerLength
    lowerMax = lowerEdge
    upperMin = upperEdge
    if upperLength == None:
        upperMax = -1
    else:
        upperMax = upperEdge + upperLength
    dataEdgeSignal = np.concatenate((signal[lowerMin:lowerMax], signal[upperMin:upperMax]))
    dataEdgeTime = np.concatenate((time[lowerMin:lowerMax], time[upperMin:upperMax]))
    # fit polynomial
    fit = np.polyfit(dataEdgeTime, dataEdgeSignal, order)
    polyObj = np.poly1d(fit)
    fitSignal = polyObj(time)
    
    # plotting
    if plot:
        plt.plot(time, signal, label='signal')
        plt.plot(time, fitSignal, '--', label='fit')
        plt.scatter(dataEdgeTime, dataEdgeSignal, label='bkg')
        plt.legend()
        plt.show()
    
    return time, fitSignal



def signalEdges(timesFrame,
                df,
                channels,
                sigmaMult=3,
                plot=False,
                prominence=0.1,
                width=10,
                avgMult=1):
    r"""
    Determines locations and widths of peaks above the mean of the signal
    for each dante channel. Edges of the signal containing region are then
    obtained by moving sigmaMult peak widths away from the earliest and
    latest peaks. Returns these lower and upper bound edges of the signal
    containing region as a dataframe. These edges are useful for fitting
    and removing the background/hysteresis.
    
    Parameters
    ----------
    timesFrame: pandas.core.frame.DataFrame
        A dataframe containing time axis values corresponding to signals in
        df.
    
    df: pandas.core.frame.DataFrame
        A dataframe of corrected/calibrated dante signal measurements.
    
    channels: list
        List of channels in df for which edges will be determined.
    
    sigmaMult: float
        Multiplier factor by which the lower and upper bounds of the signal
        containing region are determined. The lower bound is determined
        by sigmaMult times the width of the earliest peak away from the
        earliest peak. The upper bound is determined by sigmaMult times the
        width of the latest peak away from the latest peak. Default is 3
        for approximately 3*sigma away from each peak.
        
    plot: bool
        Flag for plotting peak locations and widths. Default is False.
        
    edgesFrame: pandas.core.frame.DataFrame
        Lower and upper bound edges of the signal containing region for
        each dante channel. The lower bound is in 0 index and the upper
        bound is in 1 index. The bounds are given in index coordinates
        and they have been rounded to the nearest point.
        
    prominence: float
        Prominence threshold for identifying peaks in scipy's find_peaks().
        
    width: int
        Width in index units for identifying peaks in scipy's find_peaks().
        
    avgMult: float
        Multiplicative factor for setting minimum intensity threshold
        for indentifying peaks in scipy's find_peaks(). This is a multiple
        of the signal average.
    
    Returns
    -------
    edgesFrame: 
        
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    edgesFrame = pd.DataFrame(index=[0,1], columns=df.keys())
    for ch in channels:
        times = timesFrame[ch]
        signal = df[ch]
        # finding peaks which are above the mean of the signal
        peaks, properties = find_peaks(signal,
                                       height=avgMult * np.mean(signal),
                                       prominence=prominence,
                                       width=width)
        # determining lower and upper bounds of signal region by taking first
        # and last peaks and shifting by distance of peak width
        width1 = properties["widths"][0]
        # width / 2 is kind of like sigma, and then that multiplied by 3 is
        # like 3 sigma away.
        lowerBnd = peaks[0] - sigmaMult * width1 / 2
        width2 = properties["widths"][-1]
        upperBnd = peaks[-1] + sigmaMult * width2 / 2
        # saving these edges to the dataframe which is to be returned
        edgesFrame[ch][0] = int(round(lowerBnd))
        edgesFrame[ch][1] = int(round(upperBnd))
        if plot:
            # plotting
            plt.plot(times, signal)
            plt.plot(times[peaks], signal[peaks], "x")
            plt.vlines(x=times[peaks],
                       ymin=signal[peaks] - properties["prominences"],
                       ymax = signal[peaks],
                       color = "C1")
            plt.hlines(y=properties["width_heights"],
                       xmin=times[properties["left_ips"]],
                       xmax=times[properties["right_ips"]],
                       color = "C1")
            plt.hlines(y=avgMult * np.mean(signal),
                       xmin=times[0],
                       xmax=times[len(signal) - 1],
                       color = "C2",
                       linestyles='--')
            plt.scatter(times[edgesFrame[ch][0]],
                        signal[edgesFrame[ch][0]],
                        color="C3")
            plt.scatter(times[edgesFrame[ch][1]],
                        signal[edgesFrame[ch][1]],
                        color="C3")
            plt.title(f"Ch {ch}")
            plt.xlabel("Time (s)")
            plt.ylabel("Signal (V)")
            plt.show()
    
    return edgesFrame


def polyBkgFrame(timesFrame,
                 df,
                 edgesFrame,
                 channels,
                 order=3,
                 plot=False):
    r"""
    
    Parameters
    ----------
    timesFrame: pandas.core.frame.DataFrame
        A dataframe containing time axis values corresponding to signals in
        df.
        
    df: pandas.core.frame.DataFrame
        A dataframe of corrected/calibrated dante signal measurements.
        
    edgesFrame: pandas.core.frame.DataFrame
        Dataframe describing edges of the signal containing region, outside
        of which should be just background. This function will fit to these
        two early time and late time background containing regions. See
        signalEdges().
        
    channels: list
        A list of channels for which to apply analyis.
        
    order: int
        Order of polynomial to be fitted to estimated background/hysteresis.
        
    plot: bool
        Flag for plotting fitted background and background subtracted
        signal. Default is False.
        
    Returns
    -------
    dfPoly: 
        
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    dfPoly = pd.DataFrame().reindex_like(df)
    for ch in channels:
        # getting polynomial fit of background/hysteresis
        fitTimes, fitSignals = polyBkg(time=timesFrame[ch],
                                       signal=df[ch],
                                       lowerEdge=edgesFrame[ch][0],
                                       upperEdge=edgesFrame[ch][1],
                                       order=order,
                                       lowerLength=None,
                                       upperLength=None,
                                       plot=plot)
        # subtracting hysteresis/background and saving to dataframe
        dfPoly[ch] = df[ch] - fitSignals
        
        if plot:
            plt.plot(timesFrame[ch], dfPoly[ch])
            plt.title(f'Channel {ch}')
            plt.xlabel('Time (s)')
            plt.ylabel('Signal (V)')
            plt.show()
    return dfPoly


def highestPeak(signal, peakIdxs):
    r"""
    Find the highest peak, and return list of peaks with the highest
    peak removed from the list.
    
    Parameters
    ----------
    signal: pandas.core.series.Series
        A data series consisting of signals from a single dante channel.
        
    peakIdxs: list
        A list of indices corresponding to peaks identified in the signal
        by using scipy's find_peaks() function.
    
    Returns
    -------
    peakHighestIdx: int
        Returns the index corresponding to the highest peak.
        
    peakIdxs2: list
        Returns a list of peak index locations with the highest peak
        removed from the list. This makes it easier for highestN() to
        apply highestPeak() iteratively to find the N highest peaks.
    
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    # search for first tallest peak
    peakSignals = signal[peakIdxs]
    peakHighestIdx = pd.Series.idxmax(peakSignals)
    # finding index of tallest peak in peakIdxs array (since this is a
    # numpy array and is indexed differently from peakSignals).
    delIdx = np.where(peakIdxs==peakHighestIdx)
    # remove first tallest peak
    peakIdxs2 = np.delete(peakIdxs, delIdx)
    return peakHighestIdx, peakIdxs2 


def highestN(signal, peakIdxs, peaksNum=2):
    r"""
    Select the N tallest peaks.
    
    Parameters
    ----------
    signal: pandas.core.series.Series
        A data series consisting of signals from a single dante channel.
        
    peakIdxs: list
        A list of indices corresponding to peaks identified in the signal
        by using scipy's find_peaks() function.
        
    peaksNum: int
        Number of peaks to grab from peakIdxs. This function will grab
        just the N tallest peaks where N=peaksNum.
        
    Returns
    -------
    highestPeaks: numpy.ndarray
        Array of indices corresponding to the highest peaks. Peaks are
        ordered from highest to lowest.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    if peaksNum > len(peakIdxs):
        raise Exception("Number of peaks to grab should be less than or "
                        "equal to total number of peaks provided!")
    # initialize input to find highest peak first time around
    peakIdxsInput = peakIdxs
    # initialize array containing highest peaks
    highestPeaks = np.zeros(peaksNum, dtype=int)
    # loop over finding the N highest peaks
    for idx in range(peaksNum):
        # find the highest peak in the given list of peaks
        peakHighestIdx, peakIdxs2 = highestPeak(signal=signal,
                                                peakIdxs=peakIdxsInput)
        # reassign input to be the list of peaks without the most
        # recently found peak.
        peakIdxsInput = peakIdxs2
        # saving highest peak
        highestPeaks[idx] = peakHighestIdx
    return highestPeaks


def getPeaks(timesFrame,
             df,
             channels,
             peaksNum=2,
             plot=False,
             prominence=0.1,
             width=10,
             avgMult=1):
    r"""
    
    Parameters
    ----------
    timesFrame: pandas.core.frame.DataFrame
        A dataframe containing time axis values corresponding to signals in
        df.
        
    df: pandas.core.frame.DataFrame
        A dataframe of corrected/calibrated dante signal measurements.
    
    channels: list
        A list of channels for which to apply analyis.
        
    peaksNum: int
        Number of peaks to grab from peakIdxs. This function will grab
        just the N tallest peaks where N=peaksNum.
    
    peaksNum: int
        Number of peaks to grab from peakIdxs. This function will grab
        just the N tallest peaks where N=peaksNum.
        
    plot: bool
        Flag for plotting identified peaks, with prominences, and widths,
        overlaid with the corresponding dante signal, and the average
        dante signal.
        
    peaksFrame: pandas.core.frame.DataFrame
        Returns a dataframe containing indices of the identified peaks sorted
        from the peak that occurs earliest in time to the latest in time.
        
    prominence: float
        Prominence threshold for identifying peaks in scipy's find_peaks().
        
    width: int
        Width in index units for identifying peaks in scipy's find_peaks().
        
    avgMult: float
        Multiplicative factor for setting minimum intensity threshold
        for indentifying peaks in scipy's find_peaks(). This is a multiple
        of the signal average.
        
    
    
    Returns
    -------
    peaksFrame:
        
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    peaksFrame = pd.DataFrame(index=np.arange(peaksNum), columns=df.keys())
    for ch in channels:
        times = timesFrame[ch]
        signal = df[ch]
        # finding peaks which are above the mean of the signal
        peaks, properties = find_peaks(signal,
                                       height=avgMult * np.mean(signal),
                                       prominence=prominence,
                                       width=width)
        # selecting the N tallest peaks
        highestPeakIdxs = highestN(signal=signal,
                                   peakIdxs=peaks,
                                   peaksNum=peaksNum)
        # ordering peaks by which comes earliest
        peaksOrdered = np.sort(highestPeakIdxs)
        # saving these peaks to the dataframe which is to be returned
        peaksFrame[ch] = peaksOrdered
        if plot:
            # plotting
            plt.plot(times, signal)
            plt.plot(times[peaks], signal[peaks], "x")
            plt.vlines(x=times[peaks],
                       ymin=signal[peaks] - properties["prominences"],
                       ymax = signal[peaks],
                       color = "C1")
            plt.hlines(y=properties["width_heights"],
                       xmin=times[properties["left_ips"]],
                       xmax=times[properties["right_ips"]],
                       color = "C1")
            plt.hlines(y=avgMult * np.mean(signal),
                       xmin=times[0],
                       xmax=times[len(signal) - 1],
                       color = "C2",
                       linestyles='--')
            plt.title(f"Ch {ch}")
            plt.xlabel("Time (s)")
            plt.ylabel("Signal (V)")
            plt.show()
    return peaksFrame


def alignPeaks(timesFrame,
               df,
               peaksFrame,
               channels,
               peakAlignIdx=0,
               referenceTime=1e-9,
               plot=False):
    r"""
    Parameters
    ----------
    timesFrame: pandas.core.frame.DataFrame
        A dataframe containing time axis values corresponding to signals in
        df.
        
    df: pandas.core.frame.DataFrame
        A dataframe of corrected/calibrated dante signal measurements.
        
    peaksFrame: pandas.core.frame.DataFrame
        Dataframe containing positions of N highest peaks and sorted
        from earliest in time to latest in time. See getPeaks().
    
    peakAlignIdx: int
        Picks which peak to align to. 0 is first peak, 1 is second peak in
        peaksFrame, etc.
    
    referenceTime: float
        Time in s to which align peaks. Default is 1e-9 s or 1 ns.
        
    plot: bool
        Flag for plotting aligned dante signals. Default is False.
        
    Returns
    -------
    timesAligned: pandas.core.frame.DataFrame
        Returns a dataframe identical in shape to timesFrame, but with
        the times for each dante channel offset such that the selected
        peaks are temporally aligned.
    
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    peakAlignIdx = 0 # 0 is first peak, 1 is 2nd peak, etc.
    referenceTime = 1e-9 # set peaks to occur at 1 ns
    timesAligned = pd.DataFrame().reindex_like(timesFrame)
    for ch in channels:
        time = timesFrame[ch]
        # of the two highest peaks, grab the one with the lowest index (earliest
        # in time)
        # align all signals to reference time, by setting first peak to be at
        # reference time.
        # Get index of peak relative to signals dataframe
        peakIdx = peaksFrame[ch][peakAlignIdx]
        # get time at which peak occurs
        peakTime = time[peakIdx]
        # calculate necessary shift in peak to bring it to the reference time
        offsetTime = peakTime - referenceTime
        # generate new time scales for shifted peaks
#        print(f'times: \n{time}')
#        print(f'offset time: {offsetTime}')
        timesAligned[ch] = time - offsetTime
    # plotting
    if plot:
        for ch in channels:
            plt.plot(timesAligned[ch], df[ch], label=ch)
        plt.xlabel('Time (s)')
        plt.ylabel('Signal (V)')
        plt.title('Aligned')
        plt.legend(frameon=False,
                   labelspacing=0.001,
                   borderaxespad=0.1)
        plt.show()
    return timesAligned


def constructMeasurementFrame(timesFrame, df, channels):
    r"""
    Takes out put timesFrame and dataFrame from rawProcess.py functions
    and generates a measurementFrame that can be passed to analyzeStreak()
    and other main.py functions.
    
    Converts units from seconds to nanoseconds.
    
    Parameters
    ----------
    timesFrame: pandas.core.frame.DataFrame
        A dataframe containing time axis values corresponding to signals in
        df.
        
    df: pandas.core.frame.DataFrame
        A dataframe of corrected/calibrated dante signal measurements.
        
    channels: list
        A list of channels for which to apply analyis.
        
    Returns
    -------
    measurementFrame: pandas.core.frame.DataFrame
        Returns a measurementFrame which can be passed to main.py functions
        such as analyzeSpectrum() and analyzeStreak().
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    # generating list of column names for measurementFrame
    colsList= []
    for ch in channels:
        timeStr = 'Time' + str(ch)
        signalStr = 'Signal' + str(ch)
        colsList.append(timeStr)
        colsList.append(signalStr)
    # initialize dataframe
    measurementFrame = pd.DataFrame(index=np.arange(len(df)), columns=colsList)
    for ch in channels:
        # writing data into measurement frame columns
        measurementFrame['Time' + str(ch)] = timesFrame[ch] * 1e9
        measurementFrame['Signal' + str(ch)] = df[ch]
    return measurementFrame


def loadCorrected(danteFile,
                  attenuatorsFile,
                  offsetsFile,
                  cut=None,
                  plot=False,
                  addCh=[]):
    r"""
    Given a dante data file, an attenuators file, and an offsets file, reads
    the file and applies background correction, attenuation correction, and
    channel offset correction. Returns the corrected data traces as a pandas
    dataframe. The row indices of this dataframe also contain the 
    correct time scaling given the oscilloscope settings, but note that
    the channels are not aligned. User must apply alignment correction
    using some measured signal as a temporal fiducial.
    
    
    Parameters
    ----------
    danteFile: str
        Full path to .dat file containing raw dante traces.
        
    attenuatorsFile: str
        Full path to the .xls file containing attenuator serial numbers and
        corresponding attenuation factors.
        
    offsetsFile: str
        Full path to .xls file containing dante channel offsets.
        
    cut: int
        Number of points to cut from leading and trailing end of each
        Dante channel trace. This is used to remove noise that occurs
        at the edges of the signal. Default is None, which means no
        cut is applied.
        
    plot: bool
        Flag for plotting data after each calibration/correction step.
        Default is False.
        
    addCh: list
        Add channels to analyze. This is used to override which channels
        are listed as on in the header of the data dante data file.
    
    Returns
    -------
    timeOffset: 
        
        
    dfAvg:
        
        
    onChList:
        
        
    hf:
        
        
    dfVolt:
        
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    # load data
    hf, df = readDanteData(danteFile)
    # get set of all dante channels with useful data
    onCh = onChannels(hf)
    print(f'Analyzing channels {onCh}')
    # converting from set to list because some functions are implemented
    # assuming a list of channels instead of a set.
    if addCh:
        onChList = list(onCh) + addCh
    else:
        onChList = list(onCh)
    
    # apply voltage scaling
    dfVolt, errVolt = voltageScale(hf, df)
    
    # time axis calibration
    timesFrame = timesScope(hf)
    
    # shot background correction
    # Note: Here is where we switch from string indices to integer indices
    # since we no longer need the bkg columns in our dataframe.
    timesBkg, dfBkg = bkgCorrect(df=dfVolt,
                                 timesFrame=timesFrame)
    
    # Applying offset correction
    timeOffset, dfOffset = offsetCorrect(df=dfBkg,
                                         timesFrame=timesBkg,
                                         offsetsFile=offsetsFile)
    
    # Applying attenuation correction
    dfAtten = attenuationCorrect(attenuatorsFile, hf, dfOffset, onChList)
    
    # applying an averaged background correction
    dfAvg = avgBkgCorrect(timesFrame=timeOffset,
                          df=dfAtten,
                          channels=onCh,
                          timeLength=1e-9)
    
    # plotting each calibration/correction step
    if plot:
        plt.plot(df)
        plt.title('raw dante data, fresh off the load')
        plt.xlabel('Time (counts)')
        plt.ylabel('Signal (counts)')
        plt.show()
        
        plt.plot(dfVolt)
        plt.title('Voltage scaled')
        plt.xlabel('Time (counts)')
        plt.ylabel('Signal (V)')
        plt.show()
        
        for ch in onChList:
            plt.plot(timesFrame[str(ch)], dfVolt[str(ch)], label=ch)
        plt.title('time and voltage scaled')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal (V)')
        plt.legend(frameon=False,
                   labelspacing=0.001,
                   borderaxespad=0.1)
        plt.show()
        
        for ch in onChList:
            plt.plot(timesBkg[ch], dfBkg[ch], label=ch)
        plt.title('bkg corrected')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal (V)')
        plt.legend(frameon=False,
                   labelspacing=0.001,
                   borderaxespad=0.1)
        plt.show()
        
        for ch in onChList:
            plt.plot(timeOffset[ch], dfOffset[ch], label=ch)
        plt.title('offset corrected')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal (V)')
        plt.legend(frameon=False,
                   labelspacing=0.001,
                   borderaxespad=0.1)
        plt.show()
        
        for ch in onChList:
            plt.plot(timeOffset[ch], dfAtten[ch], label=ch)
        plt.title('attenuation corrected')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal (V)')
        plt.legend(frameon=False,
                   labelspacing=0.001,
                   borderaxespad=0.1)
        plt.show()
        
        for ch in onChList:
            plt.plot(timeOffset[ch], dfAvg[ch], label=ch)
        plt.title('averaged background corrected')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal (V)')
        plt.legend(frameon=False,
                   labelspacing=0.001,
                   borderaxespad=0.1)
        plt.show()
    
    return timeOffset, dfAvg, onChList, hf, dfVolt


def hysteresisCorrect(timesFrame,
                      df,
                      channels,
                      order=5,
                      prominence=0.2,
                      width=10,
                      avgMult=1):
    r"""
    Corrects for hysteresis by detecting edges of signal containing region
    and fitting a polynomial background to regions that do not belong to 
    signal. This background is then subtracted.
    
    
    Parameters
    ----------
    timesFrame: pandas.core.frame.DataFrame
        Time corresponding to df
        
    df: pandas.core.frame.DataFrame
        Dataframe of dante signals. See loadCorrected().
        
    channels: list
        A list of channels for which to apply analyis.
        
    order: int
        Polynomial order to be fitted to hysteresis/background.
        
    prominence: float
        Prominence threshold for identifying peaks in scipy's find_peaks().
        
    width: int
        Width in index units for identifying peaks in scipy's find_peaks().
        
    avgMult: float
        Multiplicative factor for setting minimum intensity threshold
        for indentifying peaks in scipy's find_peaks(). This is a multiple
        of the signal average.
        
        
    Returns
    -------
    dfPoly: pandas.core.frame.DataFrame
        Returns a dataframe of hysteresis corrected dante signals.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    
    """
    # getting edges of signal containing region
    edgesFrame = signalEdges(timesFrame=timesFrame,
                             df=df,
                             channels=channels,
                             plot=False,
                             prominence=prominence,
                             width=width,
                             avgMult=avgMult)
    
    # removing hysteresis and background with a polynomial fit
    dfPoly = polyBkgFrame(timesFrame=timesFrame,
                          df=df,
                          edgesFrame=edgesFrame,
                          channels=channels,
                          order=order,
                          plot=False)
    return dfPoly


def align(timesFrame,
          df,
          channels,
          peaksNum=1,
          peakAlignIdx=0,
          referenceTime=1e-9,
          prominence=0.01,
          width=10,
          avgMult=1.5):
    r"""
    Aligns dante signals based on peak finding.
    
    Parameters
    ----------
    timesFrame: pandas.core.frame.DataFrame
        Time corresponding to df
        
    df: pandas.core.frame.DataFrame
        Dataframe of dante signals. See loadCorrected().
        
    channels: list
        A list of channels for which to apply analyis.
        
    peaksNum: int
        Number of peaks to grab from peakIdxs. This function will grab
        just the N tallest peaks where N=peaksNum.
        
    peakAlignIdx: int
        Picks which peak to align to. 0 is first peak, 1 is second peak in
        peaksFrame, etc.
        
    referenceTime: float
        Time in s to which align peaks. Default is 1e-9 s or 1 ns.
        
    prominence: float
        Prominence threshold for identifying peaks in scipy's find_peaks().
        
    width: int
        Width in index units for identifying peaks in scipy's find_peaks().
        
    avgMult: float
        Multiplicative factor for setting minimum intensity threshold
        for indentifying peaks in scipy's find_peaks(). This is a multiple
        of the signal average.
        
    Returns
    -------
    timesAligned: pandas.core.frame.DataFrame
        Returns a dataframe of times corresponding to signals in df, such
        that the signals are now aligned to the given peak.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    
    """
    if peakAlignIdx > peaksNum - 1:
        raise Exception(f"Cannot align to {peakAlignIdx}th peak if there"
                        f" are only {peaksNum} peaks.")
    # finding tallest peaks in the signal
    peaksFrame = getPeaks(timesFrame=timesFrame,
                          df=df,
                          channels=channels,
                          peaksNum=peaksNum,
                          plot=True,
                          prominence=prominence,
                          width=width,
                          avgMult=avgMult)
    
    # aligning to selected peak
    timesAligned = alignPeaks(timesFrame=timesFrame,
                              df=df,
                              peaksFrame=peaksFrame,
                              channels=channels,
                              peakAlignIdx=peakAlignIdx,
                              referenceTime=referenceTime,
                              plot=True)
    return timesAligned

#-----------------------------------------
# PURGATORY FOR DEPRECATED METHOD ATTEMPTS
#-----------------------------------------

#def signalEdges(time,
#                signal,
#                sigma,
#                windowMin=0,
#                windowMax=1,
#                title='',
#                plot=False):
#    """
#    Gets the signal edges via Canny edge detection method
#    
#    time: numpy.ndarray
#        Array of times corresponding to signal.
#        
#    signal: numpy.ndarray
#        Array of signal intensities.
#        
#    sigma: float
#        Width of Gaussian filter used in Canny edge detection. This should
#        be close to the width of the features you want to detect.
#        
#    windowMin: int
#        Lower bound index of window over which to search for edges in signal.
#        Default is 0.
#        
#    windowMax: int
#        Distance in index units from end of signal to upper bound of window
#        over which to search for edges in signal. Default is 1.
#        
#    plot: bool
#        Flag for plotting signal with detected edges. Default is False.
#    """
#    timeWin = time[windowMin:-windowMax]
#    signalWin = signal[windowMin:-windowMax]
#    # stacking signals to form an "image" so that scikit-image can process it.
#    sigStack = np.vstack((signalWin, signalWin, signalWin))
#    edgeCanny = feature.canny(sigStack,
#                              sigma=sigma)
#    # getting the actual edge points instead of just the indices
#    edgeTimes = timeWin[edgeCanny[1,:]]
#    edges = signalWin[edgeCanny[1,:]]
#    
#    if plot:
#        # plotting detected edges
#        plt.plot(time, signal, label='signal')
#        plt.scatter(edgeTimes, edges, color='red', label='edges')
#        plt.legend()
#        plt.title(title)
#        plt.show()
#    
#    return edgeTimes, edges
#
#
#
#
#
#def highestTwo(signal, peakIdxs):
#    """
#    Get the two highest peaks given signal, and peak
#    indices found using a peaking finder such as find_peaks_cwt().
#    """
#    # search for first tallest peak
#    peakSignals1 = signal[peakIdxs]
#    peak1Idx = np.argmax(peakSignals1)
#    # convert back into index for signals
#    peak1SigIdx = peakIdxs[peak1Idx]
#    # remove first tallest peak
#    peakIdxs2 = np.delete(peakIdxs, peak1Idx)
#    peakSignals2 = signal[peakIdxs2]
#    # search for second tallest peak
#    peak2Idx = np.argmax(peakSignals2)
#    # convert back into index for signals
#    peak2SigIdx = peakIdxs2[peak2Idx]
#    # return positions of two tallest peaks in order of which is first
#    if peak1SigIdx < peak2SigIdx:
#        return peak1SigIdx, peak2SigIdx
#    else:
#        return peak2SigIdx, peak1SigIdx