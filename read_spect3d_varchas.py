# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 09:05:09 2023

@author: Daniel H. Barnak
"""

# import matplotlib
# matplotlib.use('TkAgg')
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit
import time

from fiducia import loader, cspline, visualization, response

from scipy import integrate

import h5py
#%% import spect3D simulations in using pandas
shotNum = 46456
impPath="C:\\Users\\barna\\Desktop\\Barnak\\Dante\\SPECT3D\\" + str(shotNum)+"\\"
#%%
fileList = listdir(impPath)
#%%
impList=[]
for idx, file in enumerate(fileList):
    impRead = np.genfromtxt(impPath+file, delimiter=',')
    impList.append(impRead)
#%% load response frame
channels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
detChan = [1,2,3,5,6,7,8,9,11,12,13,14,15]
#%% load response frame
path2 = 'C:\\Users\\barna\\Desktop\\Barnak\\Dante\\'
resp2 = 'new_response_file.csv'
responseFile = path2 + resp2
force16 = np.array([[10, 1838],[11, 2500],[15, 5500]])
responseFrame = loader.loadResponses(detChan, responseFile)
knots = response.knotFind(detChan, responseFrame, forceKnot=force16)
# plot the resposne functions
visualization.plotResponse(detChan,
                           responseFrame,
                           knots,
                           solid=True,
                           title='Dante Response Functions')
#%% convolve LILAC spectra with Dante response functions
lilacVolts = np.empty((len(detChan),len(fileList)))
for idx1, spect in enumerate(impList):
    # print(spect)
    responseEnergy = responseFrame["Energy(eV)"]
    responseOnly = responseFrame.drop("Energy(eV)", axis=1)
    # initialize fidelity to number of channel responses
    photonEnergy = spect[:, 0]
    ergToJoule = 1e-7 #ergs to Joules conversion
    distToDet = 122.2**2 #distance to detector in cm squared
    intensity = spect[:, 1]*ergToJoule*distToDet
    # keep track of channels checked for fidelity
    fidChan = responseOnly.columns.to_numpy()
    for idx2, channel in enumerate(fidChan):
        chanResponse = responseFrame[channel]
        responseInterp = np.interp(photonEnergy, responseEnergy, chanResponse)
        convolve = intensity*responseInterp
        integral = integrate.simps(y=convolve, x=photonEnergy)
        lilacVolts[idx2, idx1] = integral
#%% plot LILAC simulated voltage traces
