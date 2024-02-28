# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:09:25 2023

@author: Daniel Barnak
"""

from fiducia import loader, visualization, rawProcess, response, cspline, main
from fiducia.cspline import knotSolvePCHIP, linespline
# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import xarray as xr
import sgfilter
import copy

from scipy.signal import savgol_filter
from scipy.interpolate import pchip_interpolate as pchip
from scipy.integrate import quad
from scipy.optimize import minimize

from numba import jit
import time
#%% import reduced data by shot number
channels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
detChan = [1,2,3,5,6,7,8,9,11,12,13,14,15]
root = "C:\\Users\\barna\\Desktop\\Barnak\\Dante\\reduced_data\\good_only\\"
path = 'C:\\Users\\barna\\Desktop\\Git_Projects\\fiducia\\fiducia\\'
shotNum = 46472
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
    plt.plot(times[ch], voltages[ch], label=ch)
    plt.title('averaged background corrected')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal (V)')
    plt.legend(frameon=False,
               labelspacing=0.001,
               borderaxespad=0.1)
    plt.xlim((0,5e-9))
plt.show()
#%% import solutions and times
# shotNums = [46456, 46458]
filepath = "C:\\Users\\barna\\Desktop\\Barnak\\Dante\\PCHIP_solutions\\"
# splinesImpArr = []
splinesImp = np.genfromtxt(filepath + str(shotNum)+"_"+str(goodChan)+"_PCHIP_unfold_solutions_vcmster.csv", delimiter=",")
solTimes = splinesImp[:, 0]
solVals = splinesImp[:, 1:]
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
#%% run the MC
# for idx, ti  in enumerate(solTimes[0]):
idx = 0
ti = solTimes[idx]
timeStr = str(np.round(ti,1))
print("starting run "+ timeStr + " ns")
samples = 800
initial = solVals[idx, 1:]
signals = cspline.signalsAtTime(ti, timesGood, voltages, goodChan)
sVals, fVals, dVals = cspline.run_MC_wrapper(responseFrame,
                                             signals,
                                             goodChan,
                                             knots,
                                             samples,
                                             initial = initial)
    # export the run
    
    # print("export " + timeStr + " ns")
    # root = "C:\\Users\\barna\\Desktop\\Barnak\\Dante\\MC_results\\"
    # np.savetxt(root+
    #             "shot"+str(shotNum)+
    #             "_time"+timeStr+
    #             "_samp"+str(samples)+
    #             "_splines.csv", sVals, delimiter=",")    
    # np.savetxt(root+
    #             "shot"+str(shotNum)+
    #             "_time"+timeStr+
    #             "_samp"+str(samples)+
    #             "_fidelity.csv", fVals, delimiter=",")
    # np.savetxt(root+
    #             "shot"+str(shotNum)+
    #             "_time"+timeStr+
    #             "_samp"+str(samples)+
    #             "_deltas.csv", dVals, delimiter=",") 

#%% plot the spline solutions for an arbitrary sample
sampNum = 100
sampSpline = sVals[sampNum]
xInt = np.linspace(knots[0], knots[-1], 5500)
pchipRecon = cspline.pchip_1d(knots, sampSpline, xInt)
plt.plot(xInt, pchipRecon)
plt.show()
#%% are they all the same??
for idx in range(800):
    sampSpline = sVals[idx]
    pchipRecon = cspline.pchip_1d(knots, sampSpline, xInt)
    plt.plot(xInt, pchipRecon)
plt.show()
#%% check histograms I guess??