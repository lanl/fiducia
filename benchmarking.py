# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:32:34 2022

Benchmark integration and array construction for FIDUCIA

@author: Daniel Barnak
"""

from fiducia import loader, visualization, rawProcess, response, cspline

# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import sgfilter
import time

from scipy.signal import savgol_filter

from scipy.interpolate import pchip_interpolate as pchip

from numba import jit

#%% import reduced data by shot number
channels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
detChan = [1,2,3,5,6,7,8,9,11,12,13,14,15]
# discriminate good channels by eye
# goodChan = [1, 2, 5, 6, 7, 8, 9]
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
#%% benchmark jitted linear spline function
timeStart = 0.0
timeStop = 2.0
timeStep = 0.2
runTimes = np.arange(timeStart, timeStop + timeStep, timeStep)
splineVals = np.empty((len(runTimes), len(goodChan)+1))
splineVals_jit = np.empty((len(runTimes), len(goodChan)+1))
#%% generate the responseArray for the jitted knotSolve function
responseArray = cspline.get_responses(responseFrame, goodChan)
#%% run knotSolve for all time steps
for idx, time in enumerate(runTimes):
    print(time)
    signals = loader.signalsAtTime(time, times, voltages, goodChan)
    splineVals[idx] = cspline.knotSolvePCHIP(signals,
                                             responseFrame,
                                             goodChan,
                                             knots,
                                             plot=False)
    splineVals_jit[idx] = cspline.knotSolvePCHIP_jit(signals,
                                                     responseArray,
                                                     knots)
#%% calculate differences
print(splineVals-splineVals_jit)
#%% calculate the linear splines instead for benchmarking
linesplineVals = np.empty((len(runTimes), len(goodChan)))
linesplineVals_jit = np.empty((len(runTimes), len(goodChan)))
for idx, time in enumerate(runTimes):
    print(time)
    signals = loader.signalsAtTime(time, times, voltages, goodChan)
    linesplineVals[idx] = cspline.linespline(signals, 
                                             responseFrame, 
                                             goodChan, 
                                             knots)
    linesplineVals_jit[idx] = cspline.linespline_jit(signals,
                                                     responseArray, 
                                                     knots)
#%% pretty sure I opted to do numpy interpolation for speed (check this)
# @jit(nopython=True)
def linespline_jit(signals,
                   responseArray, 
                   knots):
    chanNum = responseArray.shape[1]
    # integrate channel responses to build response matrix
    linesplineTab = np.empty((chanNum-1, len(knots)-1))
    energy = responseArray[:, 0]
    for idx in range(0, chanNum-1):
        toIntegrate = responseArray[:, idx+1]
        for idx2 in range(0, len(knots)-1):
            print(idx2)
            lb = knots[idx2]
            rb = knots[idx2 + 1]
            uInterp = np.arange(lb, rb, 0.001)
            interpResp = np.interp(uInterp, energy, toIntegrate)
            integResponse = np.trapz(interpResp, uInterp)
            # pchipResponse = cspline.pchip_1d(energy, toIntegrate, uInterp)
            # integResponse = np.trapz(pchipResponse, uInterp)
            linesplineTab[idx, idx2] = integResponse
    # take the inverse of the integrated response matrix
    lineInverse = np.linalg.inv(linesplineTab)
    # take dot product and clip solution to positive values
    dotProd = np.dot(lineInverse, signals)
    linespline = np.clip(dotProd, 0, np.inf)
            
    # return linespline
    return None

def linespline(signals,
               responseFrame, 
               channels, 
               knots,
               plot=False):
    
    # integrate channel responses to build response matrix
    linesplineTab = np.zeros((len(channels), len(knots)-1))
    for idx, chan in enumerate(channels):
        energy = responseFrame["Energy(eV)"]
        toIntegrate = responseFrame[chan]
        for idx2 in range(0, len(knots)-1):
            lb = knots[idx2]
            rb = knots[idx2 + 1]
            uInterp = np.arange(lb, rb, 0.001)
            # pchipResponse = pchip(energy, toIntegrate, uInterp)
            # integResponse = np.trapz(pchipResponse, uInterp)
            interpResp = np.interp(uInterp, energy, toIntegrate)
            integResponse = np.trapz(interpResp, uInterp)
            linesplineTab[idx, idx2] = integResponse
    # take the inverse of the integrated response matrix
    lineInverse = np.linalg.inv(linesplineTab)
    # take dot product and clip solution to positive values
    dotProd = np.dot(lineInverse, signals)
    linespline = np.clip(dotProd, 0, np.inf)
    
    return linespline

#%% find initialization bug in pchip_1d
energy = responseFrame["Energy(eV)"].to_numpy()
lb = knots[1]
rb = knots[2]
uInterp = np.arange(lb, rb, 0.001)
lowSeg = lb < uInterp
upSeg = uInterp <= rb
segBool = lowSeg & upSeg
seg = uInterp[segBool]
for idx, knot in enumerate(energy[:-1]):
    # isolate each segment of the spline
    lowSeg = energy[idx] < uInterp
    upSeg = uInterp <= energy[idx + 1]
    segBool = lowSeg & upSeg
    seg = uInterp[segBool]
    if len(seg)>0:
        print(idx, len(seg))
        # break
    if idx == 72:
        break
    # break
#%% do a pchip interpolation of a response function
pchipResp = cspline.pchip_1d(energy, responseArray[:, 1], uInterp)
pchipRespScipy = pchip(energy, responseArray[:, 1], uInterp)
# integResponse = np.trapz(pchipResp, uInterp)
integScipy = np.trapz(pchipRespScipy, uInterp)
#%% run the workspace version and then the cspline version and check speed
tic = time.time()
linSpeed = linespline_jit(signals,
                          responseArray,
                          knots)
print(time.time()-tic)