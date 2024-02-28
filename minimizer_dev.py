# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:15:00 2023

@author: Daniel Barnak
"""
from fiducia import loader, visualization, rawProcess, response, cspline, main
from fiducia.cspline import knotSolvePCHIP, linespline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
#%% x-interpolant
interpLen = responseFrame.shape[0]*10
xInt = np.linspace(min(knots), max(knots), num = interpLen)
#%% DEAD CELL TO SEPARATE DANTE DATA FROM METHOD DEVELOPMENT##################
#%% jitable pchip interpolator
@jit(nopython = True)
def pchip_1d(xData, yData, xInterp, order = 3):
    # xData = knots
    # yData = splineVals
    if yData.ndim == 1:
        # So that _edge_case doesn't end up assigning to scalars
        x = xData[:, None]
        y = yData[:, None]
    hk = x[1:] - x[:-1]
    mk = (y[1:] - y[:-1]) / hk
    
    if y.shape[0] == 2:
        # edge case: only have two points, use linear interpolation
        dk = np.zeros_like(y)
        dk[0] = mk
        dk[1] = mk
        dydx = dk.reshape(y.shape)
    else:
        # needs special handling for jit compatibility and div-by-zero cases
        smk = np.sign(mk)
        condition = (smk[1:] != smk[:-1]) | (mk[1:] == 0) | (mk[:-1] == 0)
        boolConv = condition*1
        boolConvNeg = ~condition*1
        w1 = 2*hk[1:] + hk[:-1]
        w2 = hk[1:] + 2*hk[:-1]
        mkNonzero1 = mk[:-1] + boolConv
        mkNonzero2 = mk[1:] + boolConv
        whmean = ((w1/mkNonzero1 + w2/mkNonzero2) / (w1 + w2))
        
        # calculate derivatives of middle points
        dk = np.zeros_like(y)
        # set instances where mk is 0 to 0
        dk[1:-1] = (1.0/whmean)*boolConvNeg
        
        # calculate derivatives of edge points
        # left-most edge
        h0 = hk[0]
        h1 = hk[1]
        m0 = mk[0]
        m1 = mk[1]
        d0 = ((2*h0 + h1)*m0 - h0*m1) / (h0 + h1)
    
        # try to preserve shape
        mask = np.sign(d0) != np.sign(m0)
        mask2 = (np.sign(m0) != np.sign(m1)) & (np.abs(d0) > 3.*np.abs(m0))
        mmm = (~mask) & mask2
    
        d0[mask] = 0.
        d0[mmm] = 3.*m0[mmm]
        dk[0] = d0
        # right-most edge
        h0 = hk[-1]
        h1 = hk[-2]
        m0 = mk[-1]
        m1 = mk[-2]
        d0 = ((2*h0 + h1)*m0 - h0*m1) / (h0 + h1)
    
        # try to preserve shape
        mask = np.sign(d0) != np.sign(m0)
        mask2 = (np.sign(m0) != np.sign(m1)) & (np.abs(d0) > 3.*np.abs(m0))
        mmm = (~mask) & mask2
    
        d0[mask] = 0.
        d0[mmm] = 3.*m0[mmm]
        dk[-1] = d0
        # dk[0] = PchipInterpolator._edge_case(hk[0], hk[1], mk[0], mk[1])
        # dk[-1] = PchipInterpolator._edge_case(hk[-1], hk[-2], mk[-1], mk[-2])
        dydx = dk
    # # Calculate spline coefficients using derivative values
    dx = np.diff(xData)
    dxr = dx[:, None]
    dy = np.diff(yData)
    dyr = dy[:, None]
    slope = dyr / dxr
    t = (dydx[:-1] + dydx[1:] - 2 * slope) / dxr
    
    c = np.empty((4, len(x) - 1) + y.shape[1:], dtype=t.dtype)
    c[0] = t / dxr
    c[1] = (slope - dydx[:-1]) / dxr - t
    c[2] = dydx[:-1]
    c[3] = y[:-1]
    
    # now construct the spline using framework described in Scipy: class PPoly
    k = order # order of the polynomial
    interp = np.array([yData[0]], dtype = "float64")
    for idx, knot in enumerate(xData[:-1]):
        # isolate each segment of the spline
        lowSeg = knot < xInterp
        upSeg = xInterp <= xData[idx + 1]
        segBool = lowSeg & upSeg
        seg = xInterp[segBool]
        # # construct the spline using sum from scipy PPoly
        c0 = c[0, idx]
        c1 = c[1, idx]
        c2 = c[2, idx]
        c3 = c[3, idx]
        poly = seg - knot
        S = c0*poly**3 + c1*poly**2 + c2*poly + c3
        # S = sum(c[m, idx] * (seg - knot)**(k-m) for m in range(k+1))
        interp = np.append(interp, S)
        
    return interp

@jit(nopython=True, parallel = False)
def minFunc_jit(yKnotVals,
                responseArray,
                knots,
                voltages,
                y0,
                xInterp = 'default'):
    chanNum = responseArray.shape[1]
    solution = np.zeros(chanNum-1, dtype="float64")
    yKnotValsGuess = np.zeros(len(yKnotVals)+1, dtype="float64")
    yKnotValsGuess[0] = y0
    yKnotValsGuess[1:] = yKnotVals
    if xInterp == "default":
        interpLen = responseArray.shape[0]*10
        xInt = np.linspace(min(knots), max(knots), interpLen)
    for idx in range(chanNum-1):
        pchipSol = pchip_1d(knots, yKnotValsGuess, xInt)
        # pchipResp = pchip_1d(responseArray[:, 0],
        #                       responseArray[:, idx+1],
        #                       xInt)
        interpResp = np.interp(xInt, responseArray[:, 0], responseArray[:, idx+1])
        # convolve = pchipResp*pchipSol
        convolve = interpResp*pchipSol        
        integCon = np.trapz(convolve, xInt)
        solution[idx] = integCon
    residual = np.sum((solution-voltages)**2)**(1/2)
    
    return residual

@jit(nopython = True)
def neldermead(func, x0, args=(),
               maxiter=800, maxfev=np.inf,
               xatol=1e-6, fatol=1e-6, adaptive=False, bounds = None):
    """
    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm. Adapted from Scipy to work with numba by DH Barnak.

    Options
    -------
    maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.
        Will default to ``N*200``, where ``N`` is the number of
        variables, if neither `maxiter` or `maxfev` is set. If both
        `maxiter` and `maxfev` are set, minimization will stop at the
        first reached.
    initial_simplex : array_like of shape (N + 1, N)
        Initial simplex. If given, overrides `x0`.
        ``initial_simplex[j,:]`` should contain the coordinates of
        the jth vertex of the ``N+1`` vertices in the simplex, where
        ``N`` is the dimension.
    xatol : float, optional
        Absolute error in xopt between iterations that is acceptable for
        convergence.
    fatol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
    adaptive : bool, optional
        Adapt algorithm parameters to dimensionality of problem. Useful for
        high-dimensional minimization [1]_.
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.

        Note that this just clips all vertices in simplex based on
        the bounds.

    References
    ----------
    .. [1] Gao, F. and Han, L.
       Implementing the Nelder-Mead simplex algorithm with adaptive
       parameters. 2012. Computational Optimization and Applications.
       51:1, pp. 259-277

    """
    # _check_unknown_options(unknown_options)
    maxfun = maxfev

    x0 = np.asfarray(x0).flatten() #make sure x0 is 1d array of floats
    
    # unjitable class Bounds
    # bounds = standardize_bounds(bounds)

    if adaptive:
        dim = float(len(x0))
        rho = 1
        chi = 1 + 2/dim
        psi = 0.75 - 1/(2*dim)
        sigma = 1 - 1/dim
    else:
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5

    nonzdelt = 0.05
    zdelt = 0.00025

    if bounds is not None:
        # lower_bound, upper_bound = bounds.lb, bounds.ub #old bounds routine
        lower_bound = bounds[:, 0]
        upper_bound = bounds[:, 1]


    if bounds is not None:
        x0 = np.clip(x0, lower_bound, upper_bound)
    # this version does not allow users to supply an initial simplex
    # if initial_simplex is None:
    N = len(x0)
    
    # calculate initial simplex from starting point
    sim = np.empty((N + 1, N), dtype=x0.dtype)
    sim[0] = x0
    for k in range(N):
        # y = np.array(x0, copy=True) #copy array not jit supported
        y = x0
        if y[k] != 0:
            y[k] = (1 + nonzdelt)*y[k]
        else:
            y[k] = zdelt
        sim[k + 1] = y
    # this version does not allow users to supply an initial simplex
    # else:
    #     sim = np.asfarray(initial_simplex).copy()
    #     if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
    #         raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
    #     if len(x0) != sim.shape[1]:
    #         raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
    #     N = sim.shape[1]

    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = N * 200
        maxfun = N * 200
    elif maxiter is None:
        # Convert remaining Nones, to np.inf, unless the other is np.inf, in
        # which case use the default to avoid unbounded iteration
        if maxfun == np.inf:
            maxiter = N * 200
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 200
        else:
            maxfun = np.inf
    if bounds is not None:
        sim = np.clip(sim, lower_bound, upper_bound)
    one2np1 = list(range(1, N + 1))
    fsim = np.full((N + 1,), np.inf, dtype=float)
    # # fcalls, func = _wrap_scalar_function_maxfun_validation(func, args, maxfun)

    for k in range(N + 1):
        fsim[k] = func(sim[k], args[0], args[1], args[2], args[3])
    ind = np.argsort(fsim)
    fsim = np.take(fsim, ind)
    # # fsim = np.sort(fsim)
    # # # sort so sim[0,:] has the lowest function value
    # sim = np.take(sim, ind, 0)
    sim = sim[ind]
    # # sim = np.sort(sim)
    
    iterations = 1
    # while (fcalls[0] < maxfun and iterations < maxiter):
    while (iterations < maxiter):
            if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol and
                    np.max(np.abs(fsim[0] - fsim[1:])) <= fatol):
                print("break from xatol fatol")
                break

            # xbar = np.add.reduce(sim[:-1], 0) / N
            xbar = np.sum(sim[:-1], 0) / N
            xr = (1 + rho) * xbar - rho * sim[-1]
            if bounds is not None:
                xr = np.clip(xr, lower_bound, upper_bound)
            # fxr = func(xr)
            fxr = func(xr, args[0], args[1], args[2], args[3])
            doshrink = 0

            if fxr < fsim[0]:
                xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
                if bounds is not None:
                    xe = np.clip(xe, lower_bound, upper_bound)
                # fxe = func(xe)
                fxe = func(xe, args[0], args[1], args[2], args[3])

                if fxe < fxr:
                    sim[-1] = xe
                    fsim[-1] = fxe
                else:
                    sim[-1] = xr
                    fsim[-1] = fxr
            else:  # fsim[0] <= fxr
                if fxr < fsim[-2]:
                    sim[-1] = xr
                    fsim[-1] = fxr
                else:  # fxr >= fsim[-2]
                    # Perform contraction
                    if fxr < fsim[-1]:
                        xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                        if bounds is not None:
                            xc = np.clip(xc, lower_bound, upper_bound)
                        # fxc = func(xc)
                        fxc = func(xc, args[0], args[1], args[2], args[3])

                        if fxc <= fxr:
                            sim[-1] = xc
                            fsim[-1] = fxc
                        else:
                            doshrink = 1
                    else:
                        # Perform an inside contraction
                        xcc = (1 - psi) * xbar + psi * sim[-1]
                        if bounds is not None:
                            xcc = np.clip(xcc, lower_bound, upper_bound)
                        # fxcc = func(xcc)
                        fxcc = func(xcc, args[0], args[1], args[2], args[3])

                        if fxcc < fsim[-1]:
                            sim[-1] = xcc
                            fsim[-1] = fxcc
                        else:
                            doshrink = 1
                    # shrink
                    if doshrink:
                        for j in one2np1:
                            sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                            if bounds is not None:
                                sim[j] = np.clip(sim[j], lower_bound, upper_bound)
                            # fsim[j] = func(sim[j])
                            fsim[j] = func(sim[j], 
                                            args[0], 
                                            args[1], 
                                            args[2], 
                                            args[3])
            iterations += 1
            ind = np.argsort(fsim)
            sim = sim[ind]
            fsim = np.take(fsim, ind)
    #         sim = np.sort(sim)
    #         fsim = np.sort(fsim)

    x = sim[0]
    # x = x0
    return x
    # return sim, fsim

@jit(nopython=True)
def linespline_jit(signals,
                   responseArray, 
                   knots):
    chanNum = responseArray.shape[1]
    # integrate channel responses to build response matrix
    linesplineTab = np.zeros((chanNum-1, len(knots)-1))
    energy = responseArray[:, 0]
    for idx in range(0, chanNum-1):
        toIntegrate = responseArray[:, idx+1]
        for idx2 in range(0, len(knots)-1):
            lb = knots[idx2]
            rb = knots[idx2 + 1]
            uInterp = np.arange(lb, rb, 0.001)
            interpResp = np.interp(uInterp, energy, toIntegrate)
            integResponse = np.trapz(interpResp, uInterp)
            linesplineTab[idx, idx2] = integResponse
    # take the inverse of the integrated response matrix
    lineInverse = np.linalg.inv(linesplineTab)
    # take dot product and clip solution to positive values
    dotProd = np.dot(lineInverse, signals)
    linespline = np.clip(dotProd, 0, np.inf)
            
    return linespline

@jit(nopython=True)
def knotSolvePCHIP_jit(signals,
                       responseArray, 
                       knots,
                       xInterp = 'default'):
    
    # solve linear spline first to provide seeding for Nelder-Mead simplex
    Linespline = linespline_jit(signals,
                                responseArray,
                                knots)
    
    # initial point guess based upon initial linear spline point
    y0 = Linespline[0]*1e-1

    # list of bounds. Accept only physical soltuions (knotsY>0)
    bounds = np.array([(0, np.inf)]*len(signals))
    
    # run minimization using Nelder-Mead and scipy minimize function
    fiduciaSolve = neldermead(minFunc_jit,
                              Linespline,
                              args = (responseArray,
                                      knots,
                                      signals,
                                      y0,
                                      xInterp),
                              bounds = bounds)
    
    # return knot values from minimization
    init = np.array([y0])
    knotsY = np.concatenate((init, fiduciaSolve))
    
    return knotsY

def get_errors(channels):
    randErrors = np.array([7.8, 7.8, 18., 13.2, 8.3, 7.1, 7.1, 7.1, 7.1,
                            5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4])
    sysErrors = np.array([17.4, 8.2, 11.5, 6.0, 3.8, 2.3, 2.3, 2.3, 2.3,
                          2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3])
    get_rand = randErrors[np.array(channels)-1]
    get_sys = sysErrors[np.array(channels)-1]
    
    return get_rand, get_sys

def get_responses(responseFrame, channels):
    responses = responseFrame[channels].to_numpy()
    energies = responseFrame["Energy(eV)"].to_numpy()
    energies = energies[:, None]
    resp = np.concatenate((energies, responses), axis = 1)
    
    return resp

@jit(nopython=True)
def checkFidelity_jit(responseArray,
                      photonEnergy,
                      intensity):
    
    responseEnergy = responseArray[:, 0]
    responseOnly = responseArray[:, 1:]
    chanNum = responseOnly.shape[1]
    
    # initialize fidelity to number of channel responses
    fidelity = np.zeros(chanNum) 
    
    #calculate fidelity by reconvolving solution with Dante responses
    for idx in range(chanNum):
        chanResponse = responseOnly[:, idx]
        responseInterp = np.interp(photonEnergy, responseEnergy, chanResponse)
        convolve = intensity*responseInterp
        fidelity[idx] = np.trapz(convolve, photonEnergy)
    return fidelity

def responseArrayMC(responseFrame, channels, randErrors, samples):
    responseArray = get_responses(responseFrame, channels)
    responseArrayMCshape = ((samples),) + responseArray.shape
    responseArrayMC = np.empty(responseArrayMCshape)
    chanNum = len(channels)
    for idx in range(samples):
        responseArrayMC[idx, :, 0] = responseFrame["Energy(eV)"]
        for idx2 in range(chanNum):
            responseTest = responseArray[:, idx2 + 1]
            randNums = randErrors[idx2]*1e-2*responseTest
            
            # selects a random number for each element of the response array
            randResponse = np.random.normal(responseTest, randNums)
            
            # unused response multiplier for "systematic errors"
            # systematic errors are handled as errors on the voltage values
            # respMult = np.random.normal(0, sysErrors[chan - 1])*1e-2
            # responseFrameMC[chan] = randResponse*(1 + respMult)
            
            responseArrayMC[idx, :, idx2+1] = randResponse
            
    return responseArrayMC

@jit(nopython = True, parallel = False)
def MC_jit(responseArray,
           signals,
           knots):    
    # generate an interpolation array for pchip
    interpLen = responseArray.shape[0]*10
    xInterp = np.linspace(min(knots), max(knots), num = interpLen)
    
    # solve for knots with perturbed responseArray
    splineVals = knotSolvePCHIP_jit(signals,
                                    responseArray,
                                    knots)
    # generate the spline using the pchip interpolater
    pchipSpline = pchip_1d(knots, splineVals, xInterp)
    
    # reconvolve solution with the response functions and compare to inputs
    fidelityVals = checkFidelity_jit(responseArray,
                                     xInterp,
                                     pchipSpline)
    
    # calculate the difference between the reconvolution and data                                              
    deltaVals = fidelityVals - signals

    return splineVals, fidelityVals, deltaVals

def run_MC_wrapper(responseFrame, signals, channels, knots, samples):
    
    # initialize storage arrays
    splineVals = np.zeros((samples, len(channels)+1))
    fidVals = np.zeros((samples, len(channels)))
    deltaVals = np.zeros((samples, len(channels)))
    
    # get random errors from input channels
    randErrors, _ = get_errors(channels)
    
    # generate arrays of random responsed functions
    responseMC = responseArrayMC(responseFrame, 
                                 channels, 
                                 randErrors, 
                                 samples)
    for idx in range(samples):
        responseArray = responseMC[idx]
        splineVals[idx], fidVals[idx], deltaVals[idx] = MC_jit(responseArray,
                                                               signals,
                                                               knots)
    return splineVals, fidVals, deltaVals
#%% run the knot solver with inputs from the MC
tic = time.time()
timeTest = 1.0
samples = 1
signals = cspline.signalsAtTime(timeTest, timesGood, voltages, goodChan)
splineVals, fidVals, deltaVals = run_MC_wrapper(responseFrame,
                                                signals,
                                                goodChan,
                                                knots,
                                                samples)
print(time.time() - tic)
#%% export run
root = "C:\\Users\\barna\\Desktop\\Fiducia_MC\\jit_solver\\"
np.savetxt(root+
            "shot"+str(shotNum)+
            "_time"+str(timeTest)+
            "_samp"+str(samples)+
            "_splines.csv", splineVals, delimiter=",")    
np.savetxt(root+
            "shot"+str(shotNum)+
            "_time"+str(timeTest)+
            "_samp"+str(samples)+
            "_fidelity.csv", fidVals, delimiter=",")
np.savetxt(root+
            "shot"+str(shotNum)+
            "_time"+str(timeTest)+
            "_samp"+str(samples)+
            "_deltas.csv", deltaVals, delimiter=",") 