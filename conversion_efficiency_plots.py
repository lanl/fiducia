# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:05:01 2023

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

#%% combine the errors and solutions and generate plots
pathData = "C:\\Users\\barna\\Desktop\\Barnak\\Dante\\reduced_data\\good_only\\"
pathMC = "C:\\Users\\barna\\Desktop\\Barnak\\Dante\\MC_results\\"
pathSol = "C:\\Users\\barna\\Desktop\\Barnak\\Dante\\PCHIP_solutions\\"
shotPlot = 46456

# import the data for the shot in question
timePath = pathData + str(shotPlot) + "_time_reduced_good.csv"
voltPath = pathData + str(shotPlot) + "_channels_reduced_good.csv"
times = pd.read_csv(timePath, index_col=[0], header = 0)
voltages = pd.read_csv(voltPath, index_col = [0], header = 0)

# make sure indexes are in int64 type and not object type
times.columns = times.columns.astype("int64")
voltages.columns = voltages.columns.astype("int64")

# set goodChan based on csv export from data reduction
goodChan = voltages.columns.to_numpy()

# fix times for some reason
timesGood = times[goodChan]

# import spline solutions
file = str(shotPlot)+"_"+str(goodChan)+"_PCHIP_unfold_solutions_vcmster.csv"

splinesImp = np.genfromtxt(pathSol + file, delimiter=",")
splineSols = splinesImp[:, 1:]
timesImp = splinesImp[:, 0]

# import spline errors
splineErr = np.genfromtxt(pathMC + 
                          "shot" + str(shotPlot) + 
                          "_errors.csv",
                          delimiter=',')

# print("times available for plotting are " + str(timesImp))

# plotTime = input("please select a time to plot")
plotTime = 1.0
#%% plot the reduced data used in the unfold
for idx, chan in enumerate(goodChan):
    plt.plot(timesGood[chan], voltages[chan], label = str(chan))
    plt.xlim(0, 2e-9)
plt.legend(loc = "upper right")
plt.show()
#%% load response frame
path2 = 'C:\\Users\\barna\\Desktop\\Barnak\\Dante\\'
resp2 = 'new_response_file.csv'
detChan = [1,2,3,5,6,7,8,9,11,12,13,14,15]
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
#%% import LILAC Spect3D files (f = 0.06 runs)
lilPath = "C:\\Users\\barna\\Desktop\\Barnak\\LILAC\\readable\\"
lilFile = "e"+str(shotPlot)+"_spect_streak_readable.csv"
lilImp = np.genfromtxt(lilPath + lilFile, delimiter=",")
# lilTest = lilImp.iloc[:, 0].to_numpy()
#%% select only the time steps we want
lilTime = 1.02
sel = np.where(lilImp[:,0]==lilTime)
lilEnergy = lilImp[sel][:,1]*1000
lilSpect = lilImp[sel][:,2]*1e-16
#%% plot the LILAC spectrum
plt.plot(lilEnergy, lilSpect)
plt.xscale("log")
plt.yscale("log")
plt.xlim(10, 10000)
plt.ylim(1e-3, 10)
plt.show()
#%% take appropriate slices of arrays at locations
xInt = np.linspace(knots[0], knots[-1],int(knots[-1]))
solIdx = np.where(timesImp == plotTime)[0][0]
solVals = splineSols[solIdx]
errVals = splineErr[:, 1:][solIdx]
pchipSol = cspline.pchip_1d(knots, solVals, xInt)
#%% plot the LILAC spectrum and the FIDUCIA spectrum on top of each other
# plt.plot(lilEnergy, lilSpect, label = "LILAC")
# plt.plot(xInt, pchipSol*1e-9, label = 'FIDUCIA')
# plt.errorbar(knots, solVals*1e-9, errVals*1e-9, fmt = 'o', color = 'orange')
rSource = 0.043 #estimated source radius in cm

plt.plot(lilEnergy, lilSpect*1e9/(4*np.pi), label = "LILAC")
plt.plot(xInt, pchipSol, label = 'FIDUCIA')
plt.errorbar(knots, solVals, errVals, fmt = 'o', color = 'orange')
plt.xscale("log")
plt.yscale("log")
plt.xlim(10, 10000)
plt.ylim(1e-3*1e9, 10*1e9)
plt.xlabel("Photon Energy (eV)")
# plt.ylabel("Spectral Intensity (GW/cm^2/ster/eV)")
plt.ylabel("Spectral Intensity (W/cm^2/ster/eV)")
plt.legend()
plt.gca().set_aspect(0.7)
plt.show()
#%% get the input voltage signals to the unfold
signals = loader.signalsAtTime(plotTime, timesGood, voltages, goodChan)
#%% convolve LILAC/FIDUCIA solutions with Dante responses
lilFid = cspline.checkFidelity(signals,
                               detChan,
                               lilEnergy,
                               lilSpect*1e9,
                               responseFrame,
                               plot = False)
fiduciaFid = cspline.checkFidelity(signals,
                                   detChan,
                                   xInt,
                                   pchipSol,
                                   responseFrame,
                                   plot = False)
#%% plot both the lilac and fiducia fidelity
plt.scatter(detChan, lilFid, label = "LILAC")
plt.scatter(detChan, fiduciaFid, label = "FIDUCIA")
plt.scatter(goodChan, signals, label = "Dante")
plt.xlabel("Channels")
plt.title("Fit fideltiy for shot "+str(shotPlot) + " at t="+str(plotTime)+"ns")
plt.gca().set_aspect(0.2)
plt.legend()
plt.show()