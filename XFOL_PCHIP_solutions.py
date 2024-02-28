# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:58:39 2023

@author: Daniel Barnak
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.interpolate import pchip_interpolate as pchip

#%% import solutions and times
shotNums = [108917, 108924, 108925, 108926, 108927, 108928]
filepath = "C:\\Users\\barna\\Desktop\\XFOL-23B\\Data\\Dante\\PCHIP_solutions\\"
splinesImpArr = []
for idx, shot in enumerate(shotNums):
    splinesImp = np.genfromtxt(filepath + str(shot)+"_[ 3  4  5  7  8  9 10 12 13 14]_PCHIP_unfold_solutions_vcmster.csv", delimiter=",")
    splinesImpArr.append(splinesImp)
#%% separate out the time stamps
times1 = []
splines1 = []
for idx in range(len(shotNums)):
    times1.append(splinesImpArr[idx][:, 0])
    splines1.append(splinesImpArr[idx][:, 1:])
#%% generate a streaked spectrum from the spline points
knots = np.array([1., 
                  284., 
                  511., 
                  779., 
                  1020., 
                  1303., 
                  1560., 
                  1838., 
                  2821.,
                  3352.,
                  4966.])
interpLen = 50000
xInterp = np.linspace(min(knots), max(knots), num = interpLen)
streak1 = []
streakInt = []
for idx1 in range(len(shotNums)):
    for idx2 in range(len(times1[idx1])):
        streakInt.append(pchip(knots, splines1[idx1][idx2], xInterp))
    streak1.append(streakInt)
    streakInt = []
#%% Put each streak kin a table and export the correct shot number
path = "C:\\Users\\barna\\Desktop\\XFOL-23B\\Data\\Dante\\PCHIP_solutions\\Tables\\"
for idx1 in range(len(shotNums)):
    streakExp=np.empty((interpLen+1, len(times1[idx1])+1))
    streakExp[0, 1:] = times1[idx1]
    streakExp[1:, 0] = xInterp
    streakExp[0, 0] = None
    for idx2 in range(len(times1[idx1])):
        streakExp[1:, idx2+1] = streak1[idx1][idx2]
    fname = str(shotNums[idx1]) + "_PCHIP_unfold_table_vcmster.csv"
    np.savetxt(path + fname, streakExp, delimiter=',')
#%% plot the entire streak
shot = 1
timeShot = times1[shot]
streakShot = streak1[shot]
X, Y = np.meshgrid(xInterp, timeShot)
tmin = np.clip(np.min(streakShot),0, np.inf)
# tmin = 1
tmax = np.max(streakShot)
tmaxMag = int(np.log10(tmax))+1
levels = 20
mylevels = np.linspace(tmin, tmax, levels)
ourplot = plt.contourf(X,Y, 
                       streakShot,
                       levels = mylevels)
plt.xlabel('Photon energy (eV)')
plt.ylabel('Time (ns)')
# plt.xlim()
plt.title("Streaked spectrum shot " + str(shotNums[shot]))
mycb = plt.colorbar(ourplot, label="Intensity (W/cm^2/ster/eV)")
plt.show()
#%% plot the lineouts
timePlot = 10 # time index of plot
for idx in range(len(shotNums)):
    plt.plot(xInterp, streak1[idx][timePlot], label = str(shotNums[idx]))
plt.title("Lineout at t= " +str(np.round(times1[idx][timePlot],1))+" ns")
plt.xlabel('Photon energy (eV)')
plt.ylabel('Spectral Intensity (W/cm^2/ster/eV)')
plt.yscale("log")
plt.ylim(1.0e6, 1e9)
plt.xlim(0, 3000)
plt.fill([0, 0, 5000, 5000], [0, 1e6, 1e6, 0], 'lightgray')
    # plt.show()
plt.legend()
plt.show()
#%% import the data used in the unfolds
for shotNum in shotNums:
    root = "C:\\Users\\barna\\Desktop\\XFOL-23B\\Data\Dante\\unfolded_data\\"
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
    
    # plot input data to check quality
    for ch in goodChan:
        print(ch)
        plt.plot(times[ch], voltages[ch], label=ch)
        plt.title('Shot ' +str(shotNum) + ' traces')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal (V)')
        plt.legend(frameon=False,
                   labelspacing=0.001,
                   borderaxespad=0.1)
        plt.xlim((0,5e-9))
    plt.show()
#%% calculate the power as a function of time
streakInt = np.zeros((len(shotNums), len(times1[0])))
for idx1 in range(len(shotNums)):
    print(idx1)
    for idx2, ti in enumerate(times1[idx]):
        print(idx2)
        streakCut = streak1[idx1][idx2]
        streakInt[idx1, idx2] = np.trapz(streakCut, xInterp)

#%% plot the spectral power as a function of time
for idx in range(len(shotNums)):
    plt.plot(times1[idx], streakInt[idx])
plt.xlabel("Time (ns)")
plt.ylabel("Integrated Intensity (W/cm^2/ster)")
plt.title("XFOL-23B shots")
plt.legend()
plt.show()
plt.show()
#%% now calculate the radiation temp as a function of time
degtorad = 25 * np.pi / 180
SB = 5.67e-8 # Stefan-Boltzmann constant
kB = 8.617e-5
area = np.pi*(4.50e-4)**2
cos = np.cos(degtorad)
radTemp = (streakInt/(area*SB*cos))**(1/4)*kB
#%% plot the rad temp
for idx in range(len(shotNums)):
    plt.plot(times1[idx], radTemp[idx], label = "shot " +str(shotNums[idx]))
plt.xlabel("Time (ns)")
plt.ylabel("Trad (eV)")
plt.title("XFOL-23B shots")
plt.legend()
plt.show()