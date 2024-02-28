# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 06:20:43 2021

@author: Daniel Barnak
"""

from fiducia import loader, visualization, rawProcess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate as pchip
#%%
filepath = "C:\\Users\\barna\\Desktop\\Fiducia_MC\\"
splines4 = np.genfromtxt(filepath + "splines4.csv", delimiter=",")
voltages4 = np.genfromtxt(filepath + "voltages4.csv", delimiter=",")
fidelity4 = np.genfromtxt(filepath + "fidelity4.csv", delimiter=",")
deltas4 = np.genfromtxt(filepath + "deltas4.csv", delimiter=",")

splines5 = np.genfromtxt(filepath + "splines5.csv", delimiter=",")
voltages5 = np.genfromtxt(filepath + "voltages5.csv", delimiter=",")
fidelity5 = np.genfromtxt(filepath + "fidelity5.csv", delimiter=",")
deltas5 = np.genfromtxt(filepath + "deltas5.csv", delimiter=",")

splines6 = np.genfromtxt(filepath + "splines6.csv", delimiter=",")
voltages6 = np.genfromtxt(filepath + "voltages6.csv", delimiter=",")
fidelity6 = np.genfromtxt(filepath + "fidelity6.csv", delimiter=",")
deltas6 = np.genfromtxt(filepath + "deltas6.csv", delimiter=",")

splines7 = np.genfromtxt(filepath + "splines7.csv", delimiter=",")
voltages7 = np.genfromtxt(filepath + "voltages7.csv", delimiter=",")
fidelity7 = np.genfromtxt(filepath + "fidelity7.csv", delimiter=",")
deltas7 = np.genfromtxt(filepath + "deltas7.csv", delimiter=",")

splines8 = np.genfromtxt(filepath + "splines8.csv", delimiter=",")
voltages8 = np.genfromtxt(filepath + "voltages8.csv", delimiter=",")
fidelity8 = np.genfromtxt(filepath + "fidelity8.csv", delimiter=",")
deltas8 = np.genfromtxt(filepath + "deltas8.csv", delimiter=",")

# splines78 = np.concatenate((splines7, splines8), axis=0)
# splines784 = np.concatenate((splines7, splines8, splines4), axis = 0)
# splines7845 = np.concatenate((splines7, splines8, splines4, splines5), axis = 0)
splinesAll = np.concatenate((splines4, splines5, splines6, splines7, splines8), axis=0)
deltaAll = np.concatenate((deltas4, deltas5, deltas6), axis=0)
#%% read in response function files
path2 = 'C:\\Users\\barna\\Desktop\\Barnak\\Dante\\'
resp2 = 'new_response_file.csv'
responseFile = path2 + resp2
channels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
detChan = [1,2,3,5,6,7,8,9,11,12,13,14,16]
# discriminate good channels by eye
goodChan = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12]
responseFrame = loader.loadResponses(detChan, responseFile)
#%% histogram of residuals
for idx in range(0, 10):
    deltaCounts, deltaBins = np.histogram(deltas4[:, idx], bins =20)
    plt.stairs(deltaCounts, deltaBins)
plt.show()
for idx in range(0, 10):
    deltaCounts, deltaBins = np.histogram(deltas5[:, idx], bins =20)
    plt.stairs(deltaCounts, deltaBins)
plt.show()
#%% histogram of all runs in MC
for idx in range(0, 10):
    deltaCounts, deltaBins = np.histogram(deltaAll[:, idx], bins =20)
    plt.stairs(deltaCounts, deltaBins)
plt.show()
#%% calculate residuals for all steps
residual4 = np.sum(deltas4**2, axis=1)
residual5 = np.sum(deltas5**2, axis=1)
residual6 = np.sum(deltas6**2, axis=1)
#%% join individual residuals
residualsAll = np.concatenate((residual4, residual5, residual6))
rescounts, resbins = np.histogram(residualsAll, bins = 100)
#%% plot residuals
plt.figure(5)
plt.stairs(rescounts, resbins)
plt.show()
#%% reconstruct a spline from a given MC iteration
knots = np.array([1.0e+00,
                  7.2e+01,
                  1.88e+02,
                  2.91e+02,
                  7.73e+02,
                  9.26e+02,
                  1.021e+03,
                  1.303e+03,
                  1.551e+03,
                  2.50e+03,
                  2.82e+03])
interpLen = responseFrame.shape[0]*10
xInterp = np.linspace(min(knots), max(knots), num = interpLen)
# randIdx = np.random.randint(0, len(splinesAll))
for idx in range(0, 800):
# randIdx = 1
    testSpline = pchip(knots, splinesAll[idx], xInterp)
    # plot the spline
    plt.figure(6)
    plt.plot(xInterp, testSpline)
    plt.scatter(knots, splinesAll[idx])
plt.show()
#%% take statistics of each y point
ySTD = np.std(splinesAll, axis = 0)
yMean = np.mean(splinesAll, axis=0)
percentError = ySTD/yMean
#%% fuck it lets make some error bars and see what happens
exactSol = np.array([1.198770603603514843e+06,
                    1.593549811032719165e+07,
                    3.713001090405823290e+07,
                    4.114268359756814316e+06,
                    1.636576340972094238e+07,
                    3.935510901513358112e+05,
                    1.380652388268260658e+07,
                    1.027400869585571345e+05,
                    3.102405516428953037e+06,
                    1.696480156291684136e+06,
                    1.221822507037961976e-03])
exactPCHIP = pchip(knots, exactSol, xInterp)
plt.figure(21)
plt.errorbar(knots, exactSol, ySTD, fmt = 'o', ms = 5)
plt.plot(xInterp, exactPCHIP)
plt.show()
#%% looks pretty good let's export MC results
# np.savetxt("C:\\Users\\barna\\Desktop\\Fiducia_MC\\MC_stds.csv", stds, delimiter=",")
#%% look at convergence of std values over sample sizes
# totalSamp = splinesAll.shape[0]
totalSamp = 1000
sampleSizes = np.arange(0, totalSamp)
sampSTDs = np.zeros((totalSamp, len(knots)))
for idx in range(1, totalSamp):
    samp = splinesAll[:idx+1]
    sampSTDs[idx] = np.std(samp, axis = 0)

for idx2 in range(0, len(knots)):
    plt.figure(20)
    plt.plot(sampSTDs[:, idx2], label = "knot " + str(idx2))
    plt.legend()
#%% import runs from timestep scan
splines1p5 = np.genfromtxt(filepath + "shot46462_time1.5_samp800_splines.csv", delimiter=",")
splines1p8 = np.genfromtxt(filepath + "shot46462_time1.8_samp800_splines.csv", delimiter=",")
splines2p0 = np.genfromtxt(filepath + "shot46462_time2.0_samp800_splines.csv", delimiter=",")
splines2p2 = np.genfromtxt(filepath + "shot46462_time2.2_samp800_splines.csv", delimiter=",")

fidelity1p5 = np.genfromtxt(filepath + "shot46462_time1.5_samp800_fidelity.csv", delimiter=",")
fidelity1p8 = np.genfromtxt(filepath + "shot46462_time1.8_samp800_fidelity.csv", delimiter=",")
fidelity2p0 = np.genfromtxt(filepath + "shot46462_time2.0_samp800_fidelity.csv", delimiter=",")
fidelity2p2 = np.genfromtxt(filepath + "shot46462_time2.2_samp800_fidelity.csv", delimiter=",")

deltas1p5 = np.genfromtxt(filepath + "shot46462_time1.5_samp800_deltas.csv", delimiter=",")
deltas1p8 = np.genfromtxt(filepath + "shot46462_time1.8_samp800_deltas.csv", delimiter=",")
deltas2p0 = np.genfromtxt(filepath + "shot46462_time2.0_samp800_deltas.csv", delimiter=",")
deltas2p2 = np.genfromtxt(filepath + "shot46462_time2.2_samp800_deltas.csv", delimiter=",")

#%% take statistics of each y point at each time
ySTD1p5 = np.std(splines1p5, axis = 0)
ySTD1p8 = np.std(splines1p8, axis = 0)
ySTD2p0 = np.std(splines2p0, axis = 0)
ySTD2p2 = np.std(splines2p2, axis = 0)
# yMean = np.mean(splinesAll, axis=0)
# percentError = ySTD/yMean
#%% check STD convergence for each time step
totalSamp = 800
sampleSizes = np.arange(0, totalSamp)
sampSTDs = np.zeros((totalSamp, len(knots)))
for idx in range(1, totalSamp):
    samp = splines1p5[:idx+1]
    sampSTDs[idx] = np.std(samp, axis = 0)

for idx2 in range(0, len(knots)):
    plt.figure(21)
    plt.plot(sampSTDs[:, idx2], label = "knot " + str(idx2))
    plt.legend()
#%% plot ySTDs for each time step and each knot
plt.figure(21)
plt.plot(ySTD1p5)
plt.plot(ySTD1p8)
plt.plot(ySTD2p0)
plt.plot(ySTD2p2)
plt.show()
#%% plot the knot value for 1.5 ns time
plt.figure(22)
plt.plot(splines1p5)
#%% reconstruct a spline from a given MC iteration
knots = np.array([1.0e+00,
                  7.2e+01,
                  1.88e+02,
                  2.91e+02,
                  7.73e+02,
                  9.26e+02,
                  1.021e+03,
                  1.303e+03,
                  1.551e+03,
                  2.50e+03,
                  2.82e+03])
interpLen = responseFrame.shape[0]*10
xInterp = np.linspace(min(knots), max(knots), num = interpLen)
chkIdx = 144
testSpline = pchip(knots, splines1p5[chkIdx], xInterp)
# plot the spline
plt.figure(6)
plt.plot(xInterp, testSpline)
plt.scatter(knots, splines1p5[chkIdx])
plt.show()