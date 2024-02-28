# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:18:18 2023

@author: Daniel Barnak
"""

from fiducia import loader, visualization, rawProcess, response, cspline, main
from fiducia.cspline import knotSolvePCHIP, minFunc, linespline
# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import xarray as xr
import sgfilter

# from scipy.signal import savgol_filter
from scipy.interpolate import pchip_interpolate as pchip
# from scipy.integrate import quad
# from scipy.optimize import minimize

from numba import jit
import time


#%%all of the necessary data processing steps
# import raw data, attenuators, and offsets
# path info
shotNum = 46475
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
channels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
detChan = [1,2,3,5,6,7,8,9,11,12,13,14,15]
# discriminate good channels by eye
goodChan = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
# plot data imports?
plot = False
#%% import data using loadCorrected
times2, dfAvg, on, hf, dfVolt = rawProcess.loadCorrected(danteFile,
                                                         atten,
                                                         offset,
                                                         plot = True,
                                                         dataFormat='old')
#%%
# load in response functions
# force knot point of channel 16 to be last photon energy
force16 = np.array([[10, 1838],[11, 2500],[16, 5500]])
responseFrame = loader.loadResponses(detChan, responseFile)
knots = response.knotFind(goodChan, responseFrame, forceKnot=force16)
# visualization.plotResponse(goodChan, responseFrame, knots)

# filter sample signal for find_peaks function
dfSavgol = pd.DataFrame().reindex_like(dfAvg)
for ch in on:
    signal = dfAvg[ch]
    nOpt = sgfilter.n_opt(signal, sigma = 'auto')
    noise = sgfilter.noise(signal)
    dfSavgol[ch] = sgfilter.savgol_filter(signal, nOpt, 2)

# plot the filtered signal
if plot == True:
    for ch in on:
        plt.figure(ch)
        plt.plot(dfAvg[ch])
        # plt.plot(dfSavgol[ch])
        plt.title("Channel" + str(ch))
        plt.show()

#%% hysteresis correction for all channels
dfPoly = rawProcess.hysteresisCorrect(times2,
                                      dfSavgol, 
                                      goodChan,
                                      order = 3,
                                      prominence = 0.03,
                                      width = 5.5,
                                      sigmaMult = 3.3,
                                      plot = plot)
# align signal peaks
timesAlign = rawProcess.align(times2,
                              dfPoly,
                              goodChan,
                              peaksNum = 1,
                              peakAlignIdx = 0,
                              referenceTime = 3.0e-9,
                              prominence = 0.01,
                              width = 5,
                              avgMult = 1)
#%% make manual adjustments to the timing of all of the channels
adjust = np.zeros_like(goodChan)
change = np.array([1]) #array of channels to change
adjust = -1e-9
timesAlign[1] = timesAlign[1] + adjust
#%% plot the adjustment
for ch in goodChan:
    plt.plot(timesAlign[ch], dfPoly[ch], label=ch)
plt.xlabel('Time (s)')
plt.ylabel('Signal (V)')
plt.title('Aligned')
plt.legend(frameon=False,
           labelspacing=0.001,
           borderaxespad=0.1)
# plt.vlines(1.5e-9, 0,8)
plt.show()
#%% PCHIP knot solve demo
timeTest = 2.2
signals = loader.signalsAtTime(timeTest, timesAlign, dfPoly, goodChan)
splineVals = cspline.knotSolvePCHIP(signals,
                                    responseFrame,
                                    goodChan,
                                    knots,
                                    plot=True)
#%% export the reduced aligned data from the load processing
# dfPoly.to_csv("C:\\Users\\barna\\Desktop\\Barnak\Dante\\reduced_data\\"+str(shotNum)+"_channels_reduced.csv")
# timesAlign.to_csv("C:\\Users\\barna\\Desktop\\Barnak\\Dante\\reduced_data\\"+str(shotNum)+"_time_reduced.csv")
#%% ####PCHIP PART#### rewrite MC routine using cspline routines
# errors from K.M. Campbell RSI paper (2004)
# @jit(nopython = False, parallel = True)
# def MC_jit(timesAlign, dfPoly, goodChan, detChan):
def MC_jit():

    randErrors = np.array([7.8, 7.8, 18., 13.2, 8.3, 7.1, 7.1, 7.1, 7.1,
                           5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4])
    # sysErrors = np.array([17.4, 8.2, 11.5, 6.0, 3.8, 2.3, 2.3, 2.3, 2.3,
    #                       2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3])
    # chanErrorsSelect = randErrors[np.array(goodChan)-1]
    samples = 1 #number of samples in MC
    responseFrameMC = responseFrame.copy()
    
    timeTest = 2.2
    signals = loader.signalsAtTime(timeTest, timesAlign, dfPoly, goodChan)
    
    # initialize storage arrays
    splineVals = np.zeros((samples, len(goodChan)+1))
    # randVoltages = np.zeros((samples, len(goodChan)))
    fidelityVals = np.zeros((samples, len(detChan)))
    deltaVals = np.zeros((samples, len(goodChan)))
    
    interpLen = responseFrame.shape[0]*10
    xInterp = np.linspace(min(knots), max(knots), num = interpLen)
    
    for idx in range(0, samples):
        # randVoltages[idx] = np.random.normal(vSignals, 0.05)
        # contruct random response functions
        print(idx)
        print("generate responses")
        for idx2, chan in enumerate(detChan):
            responseTest = responseFrame[chan]
            randNums = randErrors[chan - 1]*1e-2*responseTest
            
            randResponse = np.random.normal(responseTest, randNums)
            # respMult = np.random.normal(0, sysErrors[chan - 1])*1e-2
            # responseFrameMC[chan] = randResponse*(1 + respMult)
            responseFrameMC[chan] = randResponse
        print("calculation complete")   
        print("start solver")
        splineVals[idx] = cspline.knotSolvePCHIP(signals,
                                                 responseFrameMC,
                                                 goodChan,
                                                 knots,
                                                 plot=False)
        # splineVals[idx] = np.insert(fiduciaSolve.x, 0, y0)
        print( "solver completed")
        pchipSpline = pchip(knots, splineVals[idx], xInterp)
        fidelityVals[idx] = cspline.checkFidelity(signals,
                                                  goodChan,
                                                  xInterp,
                                                  pchipSpline,
                                                  responseFrameMC,
                                                  plot = False)
        
        deltaVals[idx] = fidelityVals[idx][:10] - signals
        print("step "+str(idx) + " complete")

    return splineVals, fidelityVals, deltaVals
#%% call MC function with jit decorator
# splineVals, fidelityVals, deltaVals = MC_jit(timesAlign,
#                                              dfPoly,
#                                              goodChan,
#                                              detChan)

splineVals, fidelityVals, deltaVals = MC_jit()

#%% export run
root = "C:\\Users\\barna\\Desktop\\Fiducia_MC\\"
# np.savetxt(root+
#            "shot"+str(shotNum)+
#            "_time"+str(timeTest)+
#            "_samp"+str(samples)+
#            "_splines.csv", splineVals, delimiter=",")    
# np.savetxt(root+
#            "shot"+str(shotNum)+
#            "_time"+str(timeTest)+
#            "_samp"+str(samples)+
#            "_fidelity.csv", fidelityVals, delimiter=",")
# np.savetxt(root+
#            "shot"+str(shotNum)+
#            "_time"+str(timeTest)+
#            "_samp"+str(samples)+
#            "_deltas.csv", deltaVals, delimiter=",") 
#%% IPYPARALLEL IMPLEMENTATION FOR THE MC FUNCTION
import time
import ipyparallel as ipp
cluster = ipp.Cluster(n=100)
cluster.start_cluster_sync()

rc = cluster.connect_client_sync()
dview = rc[:]
# let's put the MC code into an ipyparallel engine
def MC_jit(samples, responseFrame, times, df, goodChan, detChan, knots):
    import numpy as np
    import pandas as pd
    from fiducia.loader import signalsAtTime
    from fiducia.cspline import knotSolvePCHIP, checkFidelity
    
    from scipy.interpolate import pchip_interpolate as pchip

    randErrors = np.array([7.8, 7.8, 18., 13.2, 8.3, 7.1, 7.1, 7.1, 7.1,
                           5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4])
    # sysErrors = np.array([17.4, 8.2, 11.5, 6.0, 3.8, 2.3, 2.3, 2.3, 2.3,
    #                       2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3])
    # chanErrorsSelect = randErrors[np.array(goodChan)-1]
    # samples = 1 #number of samples in MC
    responseFrameMC = responseFrame.copy()
    timeTest = 2.2
    timesAlign = times
    dfPoly = df
    signals = signalsAtTime(timeTest, timesAlign, dfPoly, goodChan)
    
    # initialize storage arrays
    splineVals = np.zeros((samples, len(goodChan)+1))
    # randVoltages = np.zeros((samples, len(goodChan)))
    fidelityVals = np.zeros((samples, len(detChan)))
    deltaVals = np.zeros((samples, len(goodChan)))
    
    interpLen = responseFrame.shape[0]*10
    xInterp = np.linspace(min(knots), max(knots), num = interpLen)
    for idx in range(0, samples):
        # randVoltages[idx] = np.random.normal(vSignals, 0.05)
        # contruct random response functions
        print(idx)
        print("generate responses")
        for idx2, chan in enumerate(detChan):
            responseTest = responseFrame[chan]
            randNums = randErrors[chan - 1]*1e-2*responseTest
            
            randResponse = np.random.normal(responseTest, randNums)
            # respMult = np.random.normal(0, sysErrors[chan - 1])*1e-2
            # responseFrameMC[chan] = randResponse*(1 + respMult)
            responseFrameMC[chan] = randResponse
        print("calculation complete")   
        print("start solver")
        splineVals[idx] = knotSolvePCHIP(signals,
                                         responseFrameMC,
                                         goodChan,
                                         knots,
                                         plot=False)
        # splineVals[idx] = np.insert(fiduciaSolve.x, 0, y0)
        print( "solver completed")
        pchipSpline = pchip(knots, splineVals[idx], xInterp)
        fidelityVals[idx] = checkFidelity(signals,
                                          goodChan,
                                          xInterp,
                                          pchipSpline,
                                          responseFrameMC,
                                          plot = False)
        
        deltaVals[idx] = fidelityVals[idx][:10] - signals
        print("step "+str(idx) + " complete")

    return splineVals, fidelityVals, deltaVals

ar = dview.apply_async(MC_jit, 
                       8, 
                       responseFrame, 
                       timesAlign, 
                       dfPoly, 
                       goodChan,
                       detChan,
                       knots)

tic = time.time()
ar.get()
toc = time.time()
elapse = toc-tic
