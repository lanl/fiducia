# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 18:12:23 2021

@author: Daniel Barnak
"""

from fiducia import loader, visualization, rawProcess, response, cspline, main

# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import xarray as xr
import sgfilter

# from scipy.signal import savgol_filter, find_peaks, peak_prominences
# from scipy.interpolate import pchip_interpolate as pchip
# from scipy.integrate import quad
# from scipy.optimize import minimize, differential_evolution, Bounds
#%% import raw data, attenuators, and offsets
# path info
shotNum = 108928
path = 'C:\\Users\\barna\\Desktop\\Git_Projects\\fiducia\\fiducia\\'
path2 = 'C:\\Users\\barna\\Desktop\\XFOL-23B\\Data\\Dante\\'
file = 'DANTE-dante'+str(shotNum)+'.dat'
resp = 'do20230710_RspFcts_2023-07-10_0844.csv'
resp2 = 'new_response_file.csv'
attenFile = 'TableAttenuators.xls'
offsetFile = 'Offset.xls'
atten = path + attenFile
offset = path + offsetFile
bitFile = path + 'bitness.xlsx'
danteFile = path2 + file
responseFile = path2 + resp2
channels = [3,4,5,7,8,9,10,12,13,14,18]
detChan = [3,4,5,6,7,8,9,10,11,12,13,14,18]
# discriminate good channels by eye
goodChan = [3,4,5,7,8,9,10,12,13,14]
#%% import data using loadCorrected
times2, dfAvg, on, hf, dfVolt = rawProcess.loadCorrected(danteFile,
                                                         atten,
                                                         offset,
                                                         bitFile,
                                                         plot = True,
                                                         dataFormat='new')
#%% filter sample signal for find_peaks function
dfSavgol = pd.DataFrame().reindex_like(dfAvg)
for ch in goodChan:
    print(ch)
    signal = dfAvg[ch]
    nOpt = sgfilter.n_opt(signal, sigma = 'auto')
    noise = sgfilter.noise(signal)
    dfSavgol[ch] = sgfilter.savgol_filter(signal, nOpt, 2)
#%% plot the filtered signal
for ch in on:
    plt.figure(ch)
    plt.plot(dfAvg[ch])
    plt.plot(dfSavgol[ch])
    plt.title("Channel" + str(ch))
    plt.show()
#%% test viability of force edges in edgeFrame
forceEdges = np.array([[3, 300, 800],
                       [4, 400, 800],
                       [5, 200, 800],
                       [7, 200, 600],
                       [8, 400, 800],
                       [9, 400, 800],
                       [10, 400, 800],
                       [12, 200, 800],
                       [13, 200, 800],
                       [14, 400, 800]])
# forceEdges = None
#%% hysteresis correction for all channels
dfPoly = rawProcess.hysteresisCorrect(times2,
                                      dfSavgol, 
                                      goodChan,
                                      order = 3,
                                      prominence = 0.021,
                                      width = 5,
                                      sigmaMult = 2,
                                      forceEdges=forceEdges,
                                      plot = True)
#%% align signal peaks
timesAlign = rawProcess.align(times2,
                              dfPoly,
                              goodChan,
                              peaksNum = 2,
                              peakAlignIdx = 1,
                              referenceTime = 1.2e-9,
                              prominence = 0.01,
                              width = 5,
                              avgMult = 1)
#%% manual data alignment
timesGood=timesAlign[goodChan]
dfGood=dfPoly[goodChan]
# make adjsutments in the times
# timesGood = timesAlign
# timesGood[1] = timesAlign[1] + 0.40e-9
# timesGood[2] = timesAlign[2] + 0.4e-9
# timesGood[3] = timesAlign[3] - 2.35e-9
# timesGood[4] = timesAlign[4] - 2.4e-9
# timesGood[5] = timesAlign[5] - 2.3e-9
# timesGood[6] = timesAlign[6] + 0.4e-9
# timesGood[7] = timesAlign[7] + 0.4e-9
# timesGood[8] = timesAlign[8] + 0.4e-9
# timesGood[9] = timesAlign[9] + 0.3e-9
# timesGood[10] = timesAlign[10] + 0.2e-9
# timesGood[11] = timesAlign[11] + 0.8e-9
timesGood[12] = timesAlign[12] + 2.45e-9
timesGood[13] = timesAlign[13] + 2.4e-9
timesGood[14] = timesAlign[14] + 1.95e-9
# timesGood = timesGood+0.4e-9
for ch in goodChan:
    # plt.yscale("log")
    plt.plot(timesGood[ch], dfGood[ch], label=ch)
plt.xlabel('Time (s)')
plt.ylabel('Signal (V)')
plt.title('Shot '+str(shotNum)+' Aligned')
plt.legend(frameon=False,
           labelspacing=0.001,
           borderaxespad=0.1)
plt.xlim((0,5e-9))
plt.yscale('log')
# plt.ylim((1e-4, 1e1))
plt.show()
#%% export the reduced aligned data from the load processing
expPath="C:\\Users\\barna\\Desktop\\XFOL-23B\\Data\\Dante\\reduced_data\\"
dfGood.to_csv(expPath+str(shotNum)+"_channels_reduced_good.csv")
timesGood.to_csv(expPath+str(shotNum)+"_time_reduced_good.csv")