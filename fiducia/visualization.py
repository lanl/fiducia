#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:49:17 2019

Utilities for visualizing DANTE data.

@author: Pawel M. Kozlowski
"""

# python modules
import numpy as np
import matplotlib.pyplot as plt

# custom modules
import fiducia.pltDefaults


# listing all functions declared in this file so that sphinx-automodapi
# correctly documents them and doesn't document imported functions.
__all__ = ["plotResponse",
           "plotTraces",
           "plotStreak",
           "signalImg"]


def plotResponse(channels,
                 responseFrame,
                 knots,
                 solid=True,
                 title='Dante Response Functions'):
    r"""
    Plots response function curves with knot locations identified as
    vertical dashed lines.
    
    channels: list, numpy.ndarray
        List or array of relevant channels
    
    responseFrame: pandas.core.frame.DataFrame
        Pandas dataFrame containing response functions for each DANTE
        channel. See loadResponses().
        
    knots: list, numpy.ndarray
        List or array of knot point photon energy value. See knotFind().
        
    solid: Bool
        Includes solid angle in response function value if true.
        Necessary for plotting responses with correct units.
        
    Parameters
    ----------
    
    Returns
    -------
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
        
    """
    # getting sensible figure bounds
    yMax = np.max(np.max(responseFrame[channels]))
    yMin = yMax / 1e11
    xMin = 1e1
    xMax = 1e5
    # plotting figure
    fig = plt.figure()
    for idx, channel in enumerate(channels):
        plt.loglog(responseFrame['Energy(eV)'],
                   responseFrame[channel],
                   label=str(channel))
    for idx, _ in enumerate(knots):
        plt.loglog((knots[idx], knots[idx]), (yMin, yMax), '--', color='grey')
    plt.xlim((xMin, xMax))
    plt.ylim((yMin, yMax))
    plt.legend(frameon=False,
               labelspacing=0.001,
               borderaxespad=0.1)
    plt.xlabel('Energy (eV)')
    if solid:
        plt.ylabel('Response (V/GW/sr)')
    else:
        plt.ylabel('Response (V/GW)')        
    plt.title(title)
    plt.show()
    return


def plotTraces(channels, measurementFrame, scale='regular'):
    r"""
    Given a dataframe of Dante channel data, plot all the signal
    traces onto a single plot.
    
    channels: list, numpy.ndarray
        List or array of relevant channels
        
    measurementFrame: pandas.core.frame.DataFrame
        Pandas dataframe containing DANTE measurement data. See
        readDanteData() and readDanProcessed().
        
    Parameters
    ----------
    
    Returns
    -------
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    if scale == 'regular':
        for channel in channels:
            plt.plot(measurementFrame['Time' + str(channel)],
                     measurementFrame['Signal' + str(channel)],
                     label=str(channel))
        plt.legend(frameon=False,
                   labelspacing=0.001,
                   borderaxespad=0.1)
        plt.xlabel('Time (ns)')
        plt.ylabel('Signal (V)')
        plt.title('Dante Measurement Data')
        plt.show()
    elif scale == 'log':
        for channel in channels:
            plt.semilogy(measurementFrame['Time' + str(channel)],
                         measurementFrame['Signal' + str(channel)],
                         label=str(channel))
        plt.legend(frameon=False,
                   labelspacing=0.001,
                   borderaxespad=0.1)
        plt.xlabel('Time (ns)')
        plt.ylabel('Signal (V)')
        plt.title('Dante Measurement Data')
        plt.show()
    else:
        raise Exception(f"No such method for scale: {scale}")
    return


def uniformStreak(times, energies, spectra):
    r"""
    Convert streak spectra from an irregularly spaced energy sampling
    to uniformly spaced energies via linear interpolation. This is
    primarily useful for visualizing the streaked spectrum in plotStreak().
    
    times : numpy.ndarray
        1D array of times corresponding to streaked spectrum
        
    energies : numpy.ndarray
        2D array of energies which are nonuniformly spaced. Each column
        corresponds to the photon energy axis for a particularly time.
        Typically all the columns are identical.
        
    spectra : numpy.ndarray
        2D array of streaked spectra corresponding to energies array.
    """
    # generate a uniform energy axis
    energyMin = int(np.min(energies)) # eV
    energyMax = int(np.max(energies)) # eV
    energyStep = 1 # eV
    energyNew = np.arange(start=energyMin, stop=energyMax, step=energyStep)
    newShape = (len(energyNew), len(times))
    energiesNew = np.zeros(newShape)
    spectraNew = np.zeros(newShape)
    for idx, time in enumerate(times):
        energyOld = energies[:, idx]
        spectrumOld = spectra[:, idx]
        spectrumNew = np.interp(x=energyNew, xp=energyOld, fp=spectrumOld)
        # storing values
        energiesNew[:, idx] = energyNew
        spectraNew[:, idx] = spectrumNew
    return energiesNew, spectraNew


def plotStreak(times, energies, spectra):
    r"""
    Plot streak of unfolded Dante spectra. See analyzeStreak().
    
    Parameters
    ----------
    times: numpy.ndarray
        Array of times for which the unfold was analyzed.
        
    energies: numpy.ndarray
        Array of photon energies corresponding to the unfolded spectra.
        
    spectra: numpy.ndarray
        The unfolded spectral intensities as a 2D array. See analyzeStreak().
    
    Returns
    -------
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    energiesInterp, spectraInterp = uniformStreak(times, energies, spectra)
    extent=[energiesInterp[0,0],
            energiesInterp[-1,0],
            times[0],
            times[-1]]
    
    plt.imshow(spectraInterp.transpose(),
               extent=extent,
               origin='lower',
               aspect='auto')
    plt.ylabel('Time (ns)')
    plt.xlabel('Photon Energy (eV)')
    cbar = plt.colorbar()
    cbar.set_label("Spectrum (GW/sr/eV)")
    plt.show()
    return


def signalImg(signalsArr):
    r"""
    Visualize dante signals as an image.
    
    Parameters
    ----------
    signalsArr: numpy.ndarray
    
    Returns
    -------
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    plt.imshow(signalsArr.transpose(), aspect='auto')
    plt.xlabel('Time')
    plt.ylabel('channel')
    cbar = plt.colorbar()
    cbar.set_label('Signal')
    plt.show()
    return


def individualChPlot(channels, timesFrame, signalsFrame, cut=None):
    r"""
    Plot individual channels on separate plots.
    
    Parameters
    ----------
    
    Returns
    -------
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    if cut:
        timesFrame = timesFrame.iloc[cut:-cut]
        signalsFrame = signalsFrame.iloc[cut:-cut]
    for chNum in channels:
        plt.plot(timesFrame[chNum] * 1e9, signalsFrame[chNum])
        plt.title(f"Ch {chNum}")
        plt.xlabel("Time  (ns)")
        plt.ylabel("Signal (V)")
        plt.show()
    return


def traceGrid(channels,
              timesFrame,
              signalsFrame,
              chStatus=None,
              saveName="",
              cut=None,
              shotNum=""):
    r"""
    Plot individual channels on separate plots arranged into a grid.
    
    Parameters
    ----------
    
    Returns
    -------
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    # cutting leading and trailing edges of signal where noise exists
    if cut:
        timesFrame = timesFrame.iloc[cut:-cut]
        signalsFrame = signalsFrame.iloc[cut:-cut]
        
    # setting up subplots for all 18 channels
    fig, ax = plt.subplots(6, 3, sharex=True, sharey=True, figsize=(15,10))
    fig.subplots_adjust(hspace=0, wspace=0)
    
    # creating array of all possible dante channels
    allCh = np.arange(18) + 1
    # annotating each subplot window with the channel number
    for chNum in allCh:
        col = int((chNum - 1)/6)
        row = (chNum - 1) % 6
        if not chStatus:
            ax[row][col].annotate(f'Ch {chNum}', xy=(0, 0.8), xytext=(0, 0.8))
        else:
            ax[row][col].annotate(f'Ch {chNum} - {chStatus[chNum]}', xy=(0, 0.8), xytext=(0, 0.8))
    # plotting traces for only user specified channels
    for chNum in channels:
        col = int((chNum - 1)/6)
        row = (chNum - 1) % 6
        ax[row][col].plot(timesFrame[chNum] * 1e9,
                          signalsFrame[chNum] / np.max(signalsFrame[chNum]))
        ax[row][col].set_ylim((-0.1, 1.1))
#        ax[row][col].annotate(f'Ch {chNum}', xy=(0, 0.8), xytext=(0, 0.8))
    fig.suptitle(f"Shot {shotNum}")
    fig.text(0.5, 0.04, 'Time (ns, arbitrary offset)', ha='center')
    fig.text(0.04, 0.5, 'Signal (normalized to ch max)', va='center', rotation='vertical')
    
    if saveName:
        # if there is a non-empty name, then try to save
        plt.savefig(saveName)
    else:
        # no name, then just show
        plt.show()
    return