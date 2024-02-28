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
import xarray as xr
import sgfilter

from scipy.signal import savgol_filter, find_peaks, peak_prominences
from scipy.interpolate import pchip_interpolate as pchip
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution, Bounds
#%% import raw data, attenuators, and offsets
# path info
shotNum = 46456
path = 'C:\\Users\\barna\\Desktop\\Git_Projects\\fiducia\\fiducia\\'
path2 = 'C:\\Users\\barna\\Desktop\\Barnak\\Dante\\'
file = 'dante'+str(shotNum)+'.dat'
resp = 'do20061218_RspFcts_2007-01-11_1721_kluge.csv'
resp2 = 'new_response_file.csv'
attenFile = 'TableAttenuators.xls'
offsetFile = 'Offset.xls'
atten = path + attenFile
offset = path + offsetFile
danteFile = path2 + file
responseFile = path2 + resp2
bitFile = path + 'bitness.xlsx'
channels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
detChan = [1,2,3,5,6,7,8,9,11,12,13,14,15]
# discriminate good channels by eye
goodChan = [1, 2, 3, 5, 6, 7, 8, 9, 11]
#%% import data using loadCorrected
times2, dfAvg, on, hf, dfVolt = rawProcess.loadCorrected(danteFile,
                                                         atten,
                                                         offset,
                                                         bitFile,
                                                         plot = True,
                                                         dataFormat='old')
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
forceEdges = np.array([[1, 160,420],[3, 100, 360],[13, 180, 400],[14, 300, 500]])
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
                              peaksNum = 1,
                              peakAlignIdx = 0,
                              referenceTime = 3.0e-9,
                              prominence = 0.01,
                              width = 5,
                              avgMult = 1)
#%% clean up data frame nans
timesGood=timesAlign[goodChan]
dfGood=dfPoly[goodChan]
# make adjsutments in the times
# timesGood = timesAlign
# timesGood[1] = timesAlign[1] + 0.40e-9
# timesGood[2] = timesAlign[2] + 0.4e-9
# timesGood[3] = timesAlign[3] + 0.4e-9
# timesGood[5] = timesAlign[5] - 0.3e-9
# timesGood[6] = timesAlign[6] + 0.4e-9
# timesGood[7] = timesAlign[7] + 0.4e-9
# timesGood[8] = timesAlign[8] + 0.4e-9
# timesGood[9] = timesAlign[9] + 0.3e-9
# timesGood[11] = timesAlign[11] + 0.8e-9
# timesGood[12] = timesAlign[12] + 0.9e-9
# timesGood[13] = timesAlign[13] - 0.3e-9
# timesGood[14] = timesAlign[14] + 0.1e-9
for ch in goodChan:
    # plt.yscale("log")
    plt.plot(timesGood[ch], dfGood[ch], label=ch)
plt.xlabel('Time (s)')
plt.ylabel('Signal (V)')
plt.title('Aligned')
plt.legend(frameon=False,
           labelspacing=0.001,
           borderaxespad=0.1)
plt.xlim((0,5e-9))
# plt.ylim((1e-4, 1e1))
plt.show()
#%% export the reduced aligned data from the load processing
expPath="C:\\Users\\barna\\Desktop\\Barnak\Dante\\reduced_data\\"
dfGood.to_csv(expPath+str(shotNum)+"_channels_reduced_good.csv")
timesGood.to_csv(expPath+str(shotNum)+"_time_reduced_good.csv")