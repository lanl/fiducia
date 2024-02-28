# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 19:04:30 2023

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

from scipy.signal import savgol_filter
from scipy.interpolate import pchip_interpolate as pchip
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution, Bounds
#%% import reduced data by shot number
channels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
detChan = [1,2,3,5,6,7,8,9,11,12,13,14,15]
# discriminate good channels by eye
goodChan = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
root = "C:\\Users\\barna\\Desktop\\Barnak\\Dante\\reduced_data\\good_only\\"
path = 'C:\\Users\\barna\\Desktop\\Git_Projects\\fiducia\\fiducia\\'
shotNum = 46456
timePath = root + str(shotNum) + "_time_reduced_good.csv"
voltPath = root + str(shotNum) + "_channels_reduced_good.csv"
times = pd.read_csv(timePath, index_col=[0], header = 0)
voltages = pd.read_csv(voltPath, index_col = [0], header = 0)

# make sure indexes are in int64 type and not object type
times.columns = times.columns.astype("int64")
voltages.columns = voltages.columns.astype("int64")



# set goodChan based on csv export from data reduction
goodChan = voltages.columns.to_numpy()

# fix times for some reason
timesGood = times[goodChan]

# import data using loadCorrected
path2 = 'C:\\Users\\barna\\Desktop\\Barnak\\Dante\\'
file = 'dante'+str(shotNum)+'.dat'
danteFile = path2 + file
attenFile = 'TableAttenuators.xls'
offsetFile = 'Offset.xls'
atten = path + attenFile
offset = path + offsetFile
# times2, dfAvg, on, hf, dfVolt = rawProcess.loadCorrected(danteFile,
#                                                          atten,
#                                                          offset,
#                                                          plot = True,
#                                                          dataFormat='old')
#%% plot input data to check quality
for ch in goodChan:
    print(ch)
    plt.plot(times[ch], voltages[ch], label=ch)
    plt.title('averaged background corrected')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal (V)')
    plt.legend(frameon=False,
               labelspacing=0.001,
               borderaxespad=0.1)
    plt.xlim((0,5e-9))
plt.show()
#%% load response frame
path2 = 'C:\\Users\\barna\\Desktop\\Barnak\\Dante\\'
resp2 = 'new_response_file.csv'
responseFile = path2 + resp2
force16 = np.array([[10, 1838],[11, 2500],[15, 5500]])
responseFrame = loader.loadResponses(detChan, responseFile)
knots = response.knotFind(goodChan, responseFrame, forceKnot=force16)
# plot the resposne functions
visualization.plotResponse(goodChan,
                           responseFrame,
                           knots,
                           solid=True,
                           title='Dante Response Functions')
#%% get signals at time
timeStart = 0.0
timeStop = 5.0
timeStep = 0.1
runTimes = np.arange(timeStart, timeStop + timeStep, timeStep)
splineVals = np.empty((len(runTimes), len(goodChan)+1))
#%% run linespline for all time steps
signals = loader.signalsAtTime(1.0, times, voltages, goodChan)
testLine = cspline.linespline(signals,
                              responseFrame,
                              goodChan,
                              knots)
#%% run knotSolve for all time steps
for idx, time in enumerate(runTimes):
    print(time)
    signals = loader.signalsAtTime(time, times, voltages, goodChan)
    splineVals[idx] = cspline.knotSolvePCHIP(signals,
                                             responseFrame,
                                             goodChan,
                                             knots,
                                             plot=False)
#%% plot the spline solution
interpLen = responseFrame.shape[0]*10
xInt = np.linspace(min(knots), max(knots), num = interpLen)
splineConstruct = pchip(knots, splineVals[1], xInt)
plt.figure(1)
plt.plot(xInt, splineConstruct)
plt.show()
#%% tack on the times for the file export
splinePlusTime = np.append(runTimes[:, None], splineVals, axis = 1)
#%% export the run
path = "C:\\Users\\barna\\Desktop\\Barnak\\Dante\\PCHIP_solutions\\"
fname = str(shotNum)+"_" + str(goodChan) + "_PCHIP_unfold_solutions_vcmster.csv"
np.savetxt(path + fname, splinePlusTime, delimiter=',')
#%% overwrite the data if any adjustments occured
# clean up data frame nans
timesGood=times[goodChan]
dfGood=voltages[goodChan]
expPath="C:\\Users\\barna\\Desktop\\Barnak\Dante\\unfolded_data\\"
dfGood.to_csv(expPath+str(shotNum)+"_channels_reduced_good.csv")
timesGood.to_csv(expPath+str(shotNum)+"_time_reduced_good.csv")