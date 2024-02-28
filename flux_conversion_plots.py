# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:23:13 2023

@author: Daniel Barnak
"""
from fiducia import loader, visualization, rawProcess, response, cspline, main
from fiducia.cspline import knotSolvePCHIP, linespline
# import matplotlib
# matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit
import time
knots= np.array([1.0e+00,
                 7.2e+01,
                 1.88e+02,
                 2.91e+02,
                 7.71e+02,
                 9.26e+02,
                 1.0210e+03,
                 1.3030e+03,
                 1.5510e+03,
                 2.50e+03,
                 2.82e+03,
                 3.3510e+03,
                 4.9610e+03,
                 5.50e+03])

#%% import Dante data by shot number
channels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
detChan = [1,2,3,5,6,7,8,9,11,12,13,14,15]
root = "C:\\Users\\barna\\Desktop\\Barnak\\Dante\\reduced_data\\good_only\\"
path = 'C:\\Users\\barna\\Desktop\\Git_Projects\\fiducia\\fiducia\\'
voltImp = []
timeImp = []
chanImp = []
shotNum = [46456, 46458, 46459, 46460, 46462, 46465, 46466, 46468, 
           46469, 46472, 46474, 46475]
for idx, shot in enumerate(shotNum):
    timePath = root + str(shot) + "_time_reduced_good.csv"
    voltPath = root + str(shot) + "_channels_reduced_good.csv"
    times = pd.read_csv(timePath, index_col=[0], header = 0)
    voltages = pd.read_csv(voltPath, index_col = [0], header = 0)
    
    # make sure indexes are in int64 type and not object type
    times.columns = times.columns.astype("int64")
    voltages.columns = voltages.columns.astype("int64")
    
    # set goodChan based on csv export from data reduction
    goodChan = voltages.columns.to_numpy()
    
    # fix times for some reason
    timesGood = times[goodChan]
    
    # append import arrays with data
    voltImp.append(voltages)
    timeImp.append(timesGood)
    chanImp.append(goodChan)
#%% import spline solutions and times
filepath = "C:\\Users\\barna\\Desktop\\Barnak\\Dante\\PCHIP_solutions\\"
splinesImpArr = []
splineTimesArr = []
for idx, shot in enumerate(shotNum):
    chans = chanImp[idx]
    file = str(shot)+"_"+str(chans)+"_PCHIP_unfold_solutions_vcmster.csv"
    
    splinesImp = np.genfromtxt(filepath + file, delimiter=",")
    timesImp = splinesImp[:, 0]
    
    splinesImpArr.append(splinesImp[:, 1:])
    splineTimesArr.append(timesImp)
#%% import MC solutions
# shotNum = [46456]
filepath = "C:\\Users\\barna\\Desktop\\Barnak\\Dante\\MC_results\\"
samp = 800
splinesMCImp = []
voltagesMCImp = []
fidelityMCImp = []
deltasMCImp = []
for idx1, shot in enumerate(shotNum):
    timeSteps = splineTimesArr[idx1]
    chans = chanImp[idx1]
    times = timeImp[idx1]
    volts = voltImp[idx1]
    splineErrArr = np.empty((len(timeSteps), len(chans)+1))
    errAllTimes = []
    for idx2, ti in enumerate(timeSteps):
        rt = np.round(ti, 1)
        file = "shot"+str(shot)+"_time"+str(rt)+"_samp"+str(samp)
        splines = np.genfromtxt(filepath + file + "_splines.csv", delimiter=",")
        fidelity = np.genfromtxt(filepath + file + "_fidelity.csv", delimiter=",")
        deltas = np.genfromtxt(filepath + file + "_deltas.csv" , delimiter=",")
        
        #calculate variances of splines and residuals
        splineVars = np.var(splines, axis = 0)
        deltaVars = np.var(deltas, axis = 0)
        
        _, voltErrors = cspline.get_errors(chans)
        
        solTime = splineTimesArr[idx1][idx2]
        sol = splinesImpArr[idx1][idx2]
        voltAtTimes = loader.signalsAtTime(solTime, times, volts, chans)
        voltDag = (voltAtTimes*voltErrors/100)**2+deltaVars
        splineErr = ((sol[1:]**2/voltAtTimes**2)*voltDag + splineVars[1:])**(1/2)
        splineErr = np.insert(splineErr, 0, splineVars[0])
        splineErrArr[idx2] = splineErr
    splineErrArr = np.append(timeSteps[:, None], splineErrArr, axis = 1)
    np.savetxt(filepath +
                "shot"+str(shot)+
                "_errors.csv", splineErrArr, delimiter=",") 