# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 18:12:23 2021

@author: Daniel Barnak
"""

from fiducia import loader, visualization, rawProcess, response

# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sgfilter

from scipy.signal import savgol_filter
#%% import raw data, attenuators, and offsets
# path info
path = 'C:\\Users\\barna\\Desktop\\Git_Projects\\fiducia\\fiducia\\'
path2 = 'C:\\Users\\barna\\Desktop\\XFOL-22A\\Dante\\'
file = 'DANTE-dante104175.dat'
resp = 'do20220419_RspFcts_2022-05-02_1721.csv'
attenFile = 'TableAttenuators.xls'
offsetFile = 'Offset.xls'
atten = path + attenFile
offset = path + offsetFile
danteFile = path2 + file
responseFile = path2 + resp
channels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
#%% import data using loadCorrected
times2, dfAvg, on, hf, dfVolt = rawProcess.loadCorrected(danteFile,
                                                         atten,
                                                         offset,
                                                         plot = True,
                                                         dataFormat='new')
#%% load in response functions
responseFrame = loader.loadResponses(channels, responseFile)
knots = response.knotFind(channels, responseFrame)
visualization.plotResponse(channels, responseFrame, knots)
#%% filter sample signal for find_peaks function
dfSavgol = pd.DataFrame().reindex_like(dfAvg)
for ch in on:
    signal = dfAvg[ch]
    nOpt = sgfilter.n_opt(signal, sigma = 'auto')
    noise = sgfilter.noise(signal)
    dfSavgol[ch] = sgfilter.savgol_filter(signal, nOpt, 2)
#%% plot the filtered signal
for ch in on:
    plt.figure(ch)
    plt.plot(dfAvg[ch])
    plt.plot(dfSavgol[ch])
#%%
signal = dfSavgol[12]
peaks, properties = rawProcess.find_peaks(signal,
                                height=1 * np.mean(signal),
                                prominence=0.01,
                                width=10)
pHeights = properties["peak_heights"]
highest = np.where(pHeights == max(pHeights))[0]
pHigh = peaks[highest]
#%% plot channel 12 width
plt.plot(dfSavgol[12])
#%%
edgesFrame = rawProcess.signalEdges(timesFrame=times2,
                                     df=dfSavgol,
                                     channels=on,
                                     plot=True,
                                     prominence=0.01,
                                     width=5,
                                     avgMult=2)
#%% hysteresis correction for all channels
dfPoly = rawProcess.hysteresisCorrect(times2,
                                      dfSavgol,
                                      on,
                                      order = 3,
                                      prominence = 0.02,
                                      width = 5,
                                      sigmaMult = 3,
                                      plot=True)
#%% plot the hysteresis corrections
plt.figure(20)
plt.plot(dfPoly[4])
plt.plot(dfAvg[4])
#%%plot the signal and found peaks
plt.figure(10)
plt.plot(signal)
plt.plot(peaks[0], signal[peaks[0]], marker = 'o')