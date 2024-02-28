# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:22:41 2024

@author: Daniel Barnak

Calculatring the efficacy of using Dant for rho R measurements
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit
import time

from scipy import integrate
from scipy.interpolate import pchip_interpolate as pchip

import h5py

from fiducia import loader, response, visualization, cspline
channels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
detChan = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

#%% import the csv files from spect3D
path = "Z:\\Desktop\\Dante_rhoR\\readable\\"
rhoRTab = [0, 100, 150, 200]
spect_read = []
for idx, rhoR, in enumerate(rhoRTab):
    spect_imp = np.genfromtxt(path+"rhoR_"+str(rhoR)+"mg_readable.csv",
                              delimiter=',')
    spect_read.append(spect_imp)
#%% plot the rho R fluxes
for idx, spect in enumerate(spect_read):
    plt.loglog(spect[:, 0], spect[:, 1])
plt.xlim((1e1, 1e5))
# plt.ylim((yMin, yMax))
plt.show()
#%% import dante response functions
path2 = 'C:\\Users\\barna\\Desktop\\Git_projects\\fiducia\\fiducia\\'
resp2 = 'do00000000_RspFcts_2024-01-11_1115.csv'
responseFile = path2 + resp2
force16 = np.array([[4, 511],
                    [7, 1020],
                    [10, 1838],
                    [11, 2500],
                    [12, 2821],
                    [14, 4966]])
responseFrame = loader.loadResponses(detChan, responseFile)
responseFrame.fillna(0, inplace=True)
# run knot find function
knots = response.knotFind(detChan, responseFrame, forceKnot=force16)
# plot the resposne functions
plotChan = [5, 6, 7]
knotChan = [knots[5], knots[6], knots[7]]
yMax = responseFrame[channels].max().max()*2
yMin = yMax / 1e3
xMin = 1e1
xMax = 1e5
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
#%% plot channels 1 and 2
plt.plot(responseFrame["Energy(eV)"],responseFrame[1])
plt.show()
#%%
visualization.plotResponse(detChan,
                            responseFrame,
                            knots,
                            solid=True,
                            title='Dante Response Functions')
#%% calculate the voltages for each rho R calculation
rhoRVolts = np.empty((len(detChan),len(rhoRTab)))
for idx1, spect in enumerate(spect_read):
    # print(spect)
    responseEnergy = responseFrame["Energy(eV)"]
    responseOnly = responseFrame.drop("Energy(eV)", axis=1)
    # initialize fidelity to number of channel responses
    photonEnergy = spect[:, 0]
    intensity = spect[:, 1]*1e-7
    fidelity = np.zeros(responseOnly.shape[1]) 
    # keep track of channels checked for fidelity
    fidChan = responseOnly.columns.to_numpy()
    for idx2, channel in enumerate(fidChan):
        chanResponse = responseFrame[channel]
        responseInterp = np.interp(photonEnergy, responseEnergy, chanResponse)
        convolve = intensity*responseInterp
        integral = integrate.simps(y=convolve, x=photonEnergy)
        rhoRVolts[idx2, idx1] = integral
#%% Plot a bar plot of the voltages for each 
width = 0.25  # the width of the bars
multiplier = 0
x = np.arange(len(detChan))*(width+1)  # the label locations
fig, ax = plt.subplots(layout='constrained')
for idx, rhoR in enumerate(rhoRTab):
    offset = width * (multiplier)
    volts = rhoRVolts[:, idx]
    rects = ax.bar(x + offset, volts, width, label=str(rhoR)+" mg/cm^2")
    multiplier += 1
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Voltage (V)')
ax.set_title('Peak channel voltage changes vs rho R (mg/cm^2)')
ax.set_xticks(x + (len(rhoRTab)-1)/2*width , detChan)
ax.legend(loc='upper right', ncols=2)
ax.set_ylim(0, 100)
ax.set_xlim(4,14)
# ax.set_yscale('log')

plt.show()
#%%
linesplineTab = np.zeros((len(channels), len(knots)-1))
for idx, chan in enumerate(channels):
    energy = responseFrame["Energy(eV)"]
    toIntegrate = responseFrame[chan]
    for idx2 in range(0, len(knots)-1):
        lb = knots[idx2]
        rb = knots[idx2 + 1]
        uInterp = np.arange(lb, rb, 0.001)
        pchipResponse = pchip(energy, toIntegrate, uInterp)
        integResponse = np.trapz(pchipResponse, uInterp)
        # interpResp = np.interp(uInterp, energy, toIntegrate)
        # integResponse = np.trapz(interpResp, uInterp)
        linesplineTab[idx, idx2] = integResponse
#%% perform linespline test
Linespline = cspline.linespline(rhoRVolts[:, 0],
                                responseFrame,
                                detChan,
                                knots)
#%% perform unfolds for rhoR measurements
rhoRKnots = np.empty((len(knots), len(rhoRTab)))
for idx in range(0, len(rhoRTab)):
    inputVolts = rhoRVolts[:, idx]
    rhoRKnots[:, idx] = cspline.knotSolvePCHIP(rhoRVolts[:, idx],
                                               responseFrame, 
                                               detChan, 
                                               knots)
#%% reconstruct the spline
interpLen = responseFrame.shape[0]*10
xInt = np.linspace(min(knots), max(knots), num = interpLen)
splineConstruct = pchip(knots, rhoRKnots, xInt)
plt.figure(1)
plt.plot(xInt, splineConstruct[:, 0], label = "spline unfold")
for idx in range(0, 1):
    plt.plot(spect_read[idx][:, 0], spect_read[idx][:, 1]*1e-7, label="model")
    # plt.loglog(spect_read[idx][:, 0], spect_read[idx][:, 1]*1e-7)
plt.yscale('log')
plt.legend()
plt.ylabel("Spectral Intensity (ergs/cm^2/ster/s/eV")
plt.xlabel("Photon Energy (eV)")
plt.ylim(1e6, 1e9)
plt.xlim(1, 5000)
plt.show()