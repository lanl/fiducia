# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 19:04:30 2023

@author: Daniel Barnak
"""

from fiducia import loader, response, cspline

# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import xarray as xr
# import sgfilter

# from scipy.signal import savgol_filter
from scipy.interpolate import pchip_interpolate as pchip
# from scipy.integrate import quad
# from scipy.optimize import minimize, differential_evolution, Bounds

import time
#%% import reduced data by shot number
channels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
detChan = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
shotList = [108917, 108924, 108925, 108926, 108927, 108928]
# discriminate good channels by eye
# goodChan = [1, 2, 5, 6, 7, 8, 9]
# for shotNum in shotList:
shotNum = shotList[1]
root = "C:\\Users\\barna\\Desktop\\XFOL-23B\\Data\Dante\\reduced_data\\"
path = 'C:\\Users\\barna\\Desktop\\Git_Projects\\fiducia\\fiducia\\'
# shotNum = 108917
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
# path2 = 'C:\\Users\\barna\\Desktop\\Barnak\\Dante\\'
# file = 'dante'+str(shotNum)+'.dat'
# danteFile = path2 + file
# attenFile = 'TableAttenuators.xls'
# offsetFile = 'Offset.xls'
# atten = path + attenFile
# offset = path + offsetFile
# times2, dfAvg, on, hf, dfVolt = rawProcess.loadCorrected(danteFile,
#                                                          atten,
#                                                          offset,
#                                                          plot = True,
#                                                          dataFormat='old')
# plot input data to check quality
for ch in goodChan:
    print(ch)
    plt.plot(times[ch], voltages[ch], label=ch)
    plt.title('Shot '+str(shotNum) + ' reduced data')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal (V)')
    plt.legend(frameon=False,
               labelspacing=0.001,
               borderaxespad=0.1)
    plt.xlim((0,5e-9))
plt.show()
#%% load response frame
path2 = 'C:\\Users\\barna\\Desktop\\XFOL-23B\\Data\\Dante\\'
resp2 = 'do20230710_RspFcts_2023-07-10_0844.csv'
responseFile = path2 + resp2
force16 = np.array([[4, 511],
                    [7, 1020],
                    [10, 1838],
                    [12, 2821],
                    [14, 4966]])
responseFrame = loader.loadResponses(detChan, responseFile)
responseFrame.fillna(0, inplace=True)
# run knot find function
knots = response.knotFind(goodChan, responseFrame, forceKnot=force16)
# plot the resposne functions
plotChan = [10]
knotChan = [knots[7]]
yMax = responseFrame[channels].max().max()*2
yMin = yMax / 1e3
xMin = 1e2
xMax = 1e4
# plotting figure
fig = plt.figure()
for idx, channel in enumerate(plotChan):
    plt.loglog(responseFrame['Energy(eV)'],
               responseFrame[channel],
               label=str(channel))
for idx, _ in enumerate(knotChan):
    plt.loglog((knotChan[idx], knotChan[idx]), (yMin, yMax), '--', color='grey')
plt.xlim((xMin, xMax))
plt.ylim((yMin, yMax))
plt.legend(frameon=False,
           labelspacing=0.001,
           borderaxespad=0.1)
plt.xlabel('Energy (eV)')
# if solid:
plt.ylabel('Response (V/W/cm^2/sr)')
# else:
#     plt.ylabel('Response (V/GW)')     
title = "Dante Response Functions"   
plt.title(title)
plt.show()
# visualization.plotResponse(plotChan,
#                            responseFrame,
#                            knotChan,
#                            solid=True,
#                            title='Dante Response Functions')
#%% get signals at time
timeStart = 0.0
timeStop = 2.0
timeStep = 0.1
runTimes = np.arange(timeStart, timeStop + timeStep, timeStep)
splineVals = np.empty((len(runTimes), len(goodChan)+1))
# run linespline for all time steps
signals = loader.signalsAtTime(1.2, times, voltages, goodChan)
testLine = cspline.linespline(signals,
                              responseFrame,
                              goodChan,
                              knots,
                              plot = True)
#%% plot the linear spline with some goddamn labels you heathen
plt.plot(knots[1:], testLine, marker = 'o')
plt.xlabel("Photon Energy (eV)")
plt.ylabel("Spectral Flux (W/cm^2/ster/eV)")
#%%
# run knotSolve for all time steps
# for idx, ti in enumerate(runTimes):
#     print(ti)
#     signals = loader.signalsAtTime(ti, times, voltages, goodChan)
#     splineVals[idx] = cspline.knotSolvePCHIP(signals,
#                                              responseFrame,
#                                              goodChan,
#                                              knots,
#                                              plot=False)
# plot the spline solution
# interpLen = responseFrame.shape[0]*10
# xInt = np.linspace(min(knots), max(knots), num = interpLen)
# splineConstruct = pchip(knots, splineVals[1], xInt)
# plt.figure(1)
# plt.plot(xInt, splineConstruct)
# plt.show()
# tack on the times for the file export
splinePlusTime = np.append(runTimes[:, None], splineVals, axis = 1)
# export the run
# path = "C:\\Users\\barna\\Desktop\\XFOL-23B\\Data\\Dante\\PCHIP_solutions\\"
# fname = str(shotNum)+"_" + str(goodChan) + "_PCHIP_unfold_solutions_vcmster.csv"
# np.savetxt(path + fname, splinePlusTime, delimiter=',')
# overwrite the data if any adjustments occured
# clean up data frame nans
timesGood=times[goodChan]
dfGood=voltages[goodChan]
# expPath="C:\\Users\\barna\\Desktop\\XFOL-23B\\Data\\Dante\\unfolded_data\\"
# dfGood.to_csv(expPath+str(shotNum)+"_channels_reduced_good.csv")
# timesGood.to_csv(expPath+str(shotNum)+"_time_reduced_good.csv")
#%% use signals at time
signals = loader.signalsAtTime(1.2, times, voltages, goodChan)
#%% check seeding of knotsolvePCHIP
tic = time.time()
splineVals = cspline.knotSolvePCHIP(signals,
                                    responseFrame,
                                    goodChan,
                                    knots,
                                    initial = [])
elapse1 = time.time()-tic
#%% run spline vals again with solution as the seed
tic = time.time()
splineVals2 = cspline.knotSolvePCHIP(signals,
                                     responseFrame,
                                     goodChan,
                                     knots,
                                     initial = splineVals[1:])
elapse2 = time.time()-tic
#%% plot the differences in the results
xInt = np.linspace(knots[0], knots[-1], 5000)
pchipIntSol1 = pchip(knots, splineVals, xInt)
# pchipIntSol2 = pchip(knots, splineVals2, xInt)
plt.scatter(knots, splineVals)
plt.plot(xInt, pchipIntSol1)
plt.xlabel("Photon Energy (eV)")
plt.ylabel("Spectral Flux (W/cm^2/ster/eV)")
# plt.scatter(knots, splineVals2)
# plt.plot(xInt, pchipIntSol2)

plt.show()
#%% run the jitted function
responseArray = cspline.get_responses(responseFrame, goodChan)
tic = time.time()
splineJit = cspline.knotSolvePCHIP_jit(signals, responseArray, knots)
elapse3 = time.time()-tic
#%% rerun PCHIP_jit with solution seeding
tic = time.time()
splineJit2 = cspline.knotSolvePCHIP_jit(signals,
                                        responseArray,
                                        knots,
                                        initial=splineJit[1:])
elapse4 = time.time()-tic
#%%
pchipIntSol3 = pchip(knots, splineJit2, xInt)
pchipIntSol4 = pchip(knots, splineJit, xInt)
plt.plot(xInt, pchipIntSol1)
plt.plot(xInt, pchipIntSol2)
plt.plot(xInt, pchipIntSol3)
plt.plot(xInt, pchipIntSol4)
plt.show()
#%% test running initial guesses through the MC function
test1, test2, test3 = cspline.run_MC_wrapper(responseFrame,
                                             signals,
                                             goodChan,
                                             knots,
                                             samples = 1,
                                             initial = splineJit[1:])