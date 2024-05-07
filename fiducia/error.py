#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues June 16 13:48:21 2020

Utilities for calculating response uncertainty

@author: Myles Brophy
"""

# python modules
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import datetime
#custom modules
from fiducia.cspline import splineCoordsInv, segmentsArr, dToyArr, yChiCoeffArrEnergies, responseInterp, detectorArr
from fiducia.misc import areDataFramesCompatible
from fiducia.stats import trapzVariance, gradientVariance, interpVariance
from fiducia.response import knotFind
from fiducia.loader import loadResponses, loadResponseUncertainty, signalsAtTime
# listing all functions declared in this file so that sphinx-automodapi
# correctly documents them and doesn't document imported functions.
__all__ = ["detectorErrMC",
           "knotVarianceFind",
           "responseInterpVariance",
           "fancyTrapz2Variance",
           "detectorArrVariance",
           "detectorUncertainty"
           ]

# errors from K.M. Campbell RSI paper (2004)
def defaultErrors():
    """
    Generates the random and systematic errors for each channel response
    function from the 2004 paper by K.M. Campbell for existing Dante equipment.
    The random errors represent the error bar of each measured point along the
    response function curve. The systematic error acts like a multiplier to
    the entire response function.
    
    For Monte Carlo sampling of the response functions, you draw a random
    number for each point along the response curve using a normal distribution:
        i.e. channel 3 has response `R_3(e*)` at energy `e*` with a random error of
        18%. The value drawn for the Monte Carlo would then be (using numpy)
        `np.random.normal(R_3(e*), 0.18*R_3(e*))`. This can be done as a vector of
        all points in the channel: `responseMC = np.random.normal(R_3, sig)` 
        where `sig = randErrors*R_3`
    
    To apply the systematic error, simply draw a random number using a mean of
    zero and multiply it by the response:
        i.e. channel 3 has a systematic error of 11.5% Draw a random number
        using numpy `rand = np.random.normal(0, 0.115)` and then multiply it 
        to the response function `responseMC = R_3*(1+rand)`
    Systematic errors propagate forward to being uncertainties on the voltage
    reading for each channel. This can be added in quadrature to the
    digitizer/cable chain noise for a true error bar of the voltages. You can
    prove this yourself with an MC of your very own!

    Returns
    -------
    randErrors : numpy.ndarray
        The random errors of each channel in order from 1 to 18.
    sysErrors : numpy.ndarray
        The systematic errors of each channel in order from 1 to 18.

    """
    randErrors = np.array([7.8, 7.8, 18., 13.2, 8.3, 7.1, 7.1, 7.1, 7.1,
                           5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4])
    sysErrors = np.array([17.4, 8.2, 11.5, 6.0, 3.8, 2.3, 2.3, 2.3, 2.3,
                          2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3])
    return randErrors, sysErrors
    
def detectorErrMC(detArr, detArrVariance, samples=10000,
                  boundary="y0", MChistogram=False):
    r"""
    Monte Carlo simulation and statistics to determine cubic spline uncertainty.
    
    Calculate the cubic spline matrix uncertainty using a Monte Carlo simulation
    and statistics on the MC's output.
    
    Parameters
    ----------
    detArr: numpy.ndarray
        Matrix representing the spectrally integrated folding of the detector
        response with a cubic spline interpolation of the x-ray spectrum. See
        'cspline.detectorArr()'.
    
    detArrVariance:  numpy.ndarray
        A DataFrame containing the uncertainty for each Dante channel for the
        photon energy range that detArr spans.
    
    samples : int, optional
        Number of MC samples to run. Default is `10000`.
    
    boundary : str, optional
        Choose whether yGuess corresponds to :math:`y_0` (lowest photon energy) or
        :math:`y_{n+1}` (highest photon energy) boundary condition. This should
        correspond to the photon energy value given in knots. Options are `y0`
        or `yn+1`. Default is 'y0'.
    
    MChistogram : bool, optional
        Plot histograms corresponding to each variant of detArr generated with
        Monte Carlo uncertainty propagation.
        Default is `False`.
    
    Returns
    -------
    stdErrorMatrix : numpy.ndarray
         A numpy.ndarray with the standard deviation of the inverted matrices
         generated using random weights based on the channel uncertainty.
        
    Raises
    ------
    Exception
        If `boundary` doesn't equal `y0` or `yn+1`.
    ValueError
        If the shapes of `detArr` and `detUncertaintyArr` aren't equal.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    
    if detArr.shape != detArrVariance.shape:
        raise ValueError(f"Shape of the cubic spline matrix ({detArr.shape}) is not equal to"+
                         f" the shape of the detector uncertainty matrix ({detArrVariance.shape})")
    #square root the uncertainty matrix because it is sigma^2 and we want sigma
    detArrUncertainty = np.sqrt(detArrVariance)
    #make a 3d matrix, where each 2d slice is an error matrix generated by MC
    print("Generating "+str(samples) + " MC samples...")
    error = xr.DataArray(np.zeros((samples,)+detArr.shape), dims=['sample', 'channel', 'knot_point'])
    for s in range(samples):
        error[s] = detArr + np.random.normal(loc=0, scale=1, size=detArr.shape)*detArrUncertainty

    print("Finished generating MC samples.")
    
    detArrInv = xr.DataArray(np.linalg.inv(detArr), dims=['channel', 'knot_point'], attrs={'boundary':boundary})  
    invError = xr.DataArray(np.zeros(error.shape), dims=['sample', 'channel', 'knot_point'])
    
    # #invert each 2d slice of the error matrix
    for sample in range(samples):
        invErrSlice = np.linalg.inv(error[sample])
        invError[sample] = invErrSlice
            
    # #find std of each element compared to other samples
    stdDetArrInv = invError.std(dim="sample")
    stdDetArrInv.attrs['boundary'] = boundary
    
    if MChistogram:
        for ch in invError['channel']:
            for knot in invError['knot_point']:
                # print(len(invError.sel(channel=ch, knot_point=knot).values))
                plt.hist(invError.sel(channel=ch, knot_point=knot).values,
                         samples,
                         alpha=1,
                         label=f'Channel {ch.values} knot {knot.values}')
                plt.axvline(detArrInv.sel(channel=ch, knot_point=knot).values,
                            label='DetArrInv',
                            color='red')
                plt.axvline(detArrInv.sel(channel=ch,knot_point=knot).values + stdDetArrInv.sel(channel=ch, knot_point=knot).values,
                            label='DetArrInv+sigma',
                            color='purple')
                plt.axvline(detArrInv.sel(channel=ch, knot_point=knot).values - stdDetArrInv.sel(channel=ch, knot_point=knot).values,
                            label='DetArrInv-sigma',
                            color='green')
                plt.legend(loc='upper right')
                plt.title("Detector MC")
                plt.show()
                
    return detArrInv, stdDetArrInv
    

#TODO check to see if this has any effect on final error bars. Pawel thinks not.
def knotVarianceFind(channels,
                     responseUncertaintyFrame=None,
                     forceKnot=np.array([]), 
                     knotBoundaryY=1e-77,
                     boundary='y0'):
    r"""
    Modification of response.knotFind()

    Parameters
    ----------
    channels: numpy.ndarray
        Array of DANTE channel numbers. 

    responseUncertaintyFrame: pandas.core.frame.DataFrame, optional
        DataFrame holding percent uncertainties of DANTE channel responses as
        a function of photon energy (not normalized). The default is `None`.

    forceKnot : TYPE, optional
        DESCRIPTION. The default is `np.array([])`.
    
    knotBoundaryY : float, optional
        Guess for position of y_0 or y_{n+1} knot point. Default is 1e-77.
    
    boundary: str, optional
        Choose whether yGuess corresponds to :math:`y_0` (lowest photon
        energy) or :math:`y_{n+1}` (highest photon energy) boundary condition.
        This should correspond to the photon energy value given in knots.
        Options are `y0` or `yn+1`. Default `y0`.

    Returns
    -------
     knotUncertainty: numpy.ndarray
         An array of uncertainty in knot points, with each element
         corresponding to a channel or boundary condition. 
         See :func:`response.knotFind`.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    #modified from response.knotFind
    knotsUncertainty = np.zeros(len(channels))
    
    #check if uncertainty frame is passed.
    if responseUncertaintyFrame is None:
        #return ndarray of length channels + 1 for boundary condition
        return  np.zeros(len(channels)+1)
    
    for idx, channel in enumerate(channels):
        # if forceKnot isn't an empty list, then we go about forcing the user
        # provided values.
        if forceKnot.size != 0:
            if channel in forceKnot[:,0]:
                # if the channel is forced by the user then put the user provided
                # photon energy into the knots array
                forceIdx = np.where(forceKnot[:,0] == channel)
                knotsUncertainty[idx] = forceKnot[forceIdx, 1]
            else:
                # finding largest negative gradient, which should correpond to the
                # K-edge of the DANTE channel filter.
                grad = -gradientVariance(responseUncertaintyFrame[channel])
                maxIndex = np.argmax(grad)
                knotsUncertainty[idx] = responseUncertaintyFrame['Energy(eV)'][maxIndex]
        else:
            # finding largest negative gradient, which should correpond to the
            # K-edge of the DANTE channel filter.
            grad = -gradientVariance(responseUncertaintyFrame[channel])
            maxIndex = np.argmax(grad)
            knotsUncertainty[idx] = responseUncertaintyFrame['Energy(eV)'][maxIndex]
        
    if boundary == 'y0':
        knotsUncertaintyAppend = np.append([knotBoundaryY], knotsUncertainty)
    elif boundary == 'yn+1':
        knotsUncertaintyAppend = np.append(knotsUncertainty, [knotBoundaryY])
    else:
        raise Exception(f"No method found for boundary {boundary}.")
    return knotsUncertaintyAppend


def responseInterpVariance(energyNorm,
                           energyMin,
                           energyMax,
                           responseUncertaintyFrame,
                           channels):
    r"""
    Given a DANTE detector response as a function of energy, convert the
    response to normalized photon energy, t, over a given spline segment, and
    return interpolated response values for a given value of t. Returns an
    array of interpolated responses corresponding to the number of channels.
    
    Parameters
    ----------
    energyNorm: float, numpy.ndarray
        normalized photon energy
        
    energyMin: float
        Lower bound photon energy of the spline segment over which we are
        normalizing.
        
    eneryMax: float
        Upper bound photon energy of the spline segment over which we are
        normalizing.
        
    responseFrame: pandas.core.frame.DataFrame
        DANTE channel responses as a function of photon energy (not normalized).

    channels: numpy.ndarray
        numpy array of DANTE channel numbers.
    
    Returns
    -------
    responsesInterpdVariance: numpy.ndarray
        Returns a matrix of (energyNorms, channels) of response functions.
    
    Notes
    -----
    
    See also
    --------
    cspline.repsonseInterp
    
    Examples
    --------

    """
    chLen = len(channels)
    if np.shape(energyNorm):
        # if energyNorm is a vector, then get the length of that vector
        energyLen = len(energyNorm)
    else:
        # otherwise the length is just 1
        energyLen = 1
    # construct an array of interpolated response functions based on
    # number of photon energy points we want and number of DANTE channels
    responsesInterpdVariance = np.zeros((energyLen, chLen))
    # fetch the original array of energy values (not normalized)
    energyArr = responseUncertaintyFrame['Energy(eV)']
    # converting normalized energy values to un-normalized energy values,
    # since our response functions are in terms of absolute photon energy
    energyReg = splineCoordsInv(energyNorm, energyMin, energyMax)
    for idx, channel in enumerate(channels):
        responseUncertaintyArr = responseUncertaintyFrame[channel]
        responsesInterpdVariance[:,idx] = interpVariance(x=energyReg,
                                                         xp=energyArr,
                                                         fpUnc=responseUncertaintyArr)
    return responsesInterpdVariance


def fancyTrapz2Variance(energyNorms,
                        yChis,
                        segments,
                        responseUncertaintyFrame,
                        channels,
                        interpProp=True):
    r"""
    Calculate the variance when propogating uncertainties
    through :func:`fiducia.cspline.fancyTrapz2`.

    Parameters
    ----------
    energyNorms : numpy.ndarray
        Array of normalized energies over which the integral is computed.
    
    yChis : numpy.ndarray
        3D array corresponding to the :math:`M_{y \chi}` coefficients.
        Array shape corresponds to (`energyNorms`, `chLen`, `dToY`). 
        See :func:`fiducia.error.detectorArrVariance`
    
    segments : numpy.ndarray
        Array of segments produced by :func:`segmentsArr` with the knots
   
    responseUncertaintyFrame : pandas.core.frame.DataFrame
        DataFrame holding uncertainty percentages of DANTE channel responses
        as a function of photon energy (not normalized).
        
    channels : numpy.ndarray
        Array of DANTE channel numbers.
        
    interpProp : bool, optional
        Boolean to decide if :func:`error.responseInterpVariance` should be
        used. If `False, :func:`cspline.responseInterp()` is used, speeding
        up the calculation. Note that the uncertainty is would not be
        propagated correctly if `False`. With future optimizations, this
        option to choose may be removed. Default is `True`.
    
    Returns
    -------
    integArrVariance : xarray.Dat
        A matrix containing the folded integration of the :math:`M_{y \chi}` 
        matrix and response function uncertainty matrix, with respect to
        normalized photon  energy.
        Has shape (`len(channels)`, `len(segments)`, `len(knotIndex)`).
        
    Notes
    -----
    
    See also
    --------
    cspline.fancyTrapz2()
    
    Examples
    --------
        
    """
    shape = np.shape(yChis)
    segmentsLen = np.shape(segments)[0]
    knotsLen = shape[2]
    chLen = segmentsLen
    # initialize integArr for storing photon energy integrated values
    integArrVariance = xr.DataArray(np.zeros((chLen, segmentsLen, knotsLen)),
                                    dims=['channel', 'segment', 'knot_point'],
                                    coords={'channel':channels})  

    # loop over relevant DANTE channels for analysis
    for channelIdx in np.arange(chLen):
        # loop over photon energy segments (between knot points)
        for segmentNum, segment in enumerate(segments):
            energyMin, energyMax = segment
            # loop over knot points
            for knotNum in np.arange(knotsLen):
                # multiplication of response by spline matrix
                #use interpolation propagation by default, but runs slower
                #might not have much an effect. For now we'll let the user decide
                #need to optimize reponseInterpVariance
                if interpProp:
                    responsesVariance = responseInterpVariance(energyNorms,
                                                               energyMin,
                                                               energyMax,
                                                               responseUncertaintyFrame,
                                                               channels)
                    responsesUncertainty = np.sqrt(responsesVariance)
                else:
                    responsesVariance = responseInterp(energyNorms,
                                                       energyMin,
                                                       energyMax,
                                                       responseUncertaintyFrame,
                                                       channels)
                    
                multArr = np.abs(yChis[:, segmentNum, knotNum] * responsesUncertainty[:, channelIdx])
                integVar = (energyMax - energyMin) ** 2 * trapzVariance(multArr,
                                                                        x=energyNorms)
                integArrVariance[channelIdx, segmentNum, knotNum] = integVar
    return integArrVariance


def detectorArrVariance(channels, knots, responseUncertaintyFrame, boundary="y0", npts=1000):
    r"""
    Propagates uncertanity through :func:`cspline.detectorArr() to find the variance in :math:`M_{int}`.
    
    Parameters
    ----------
    channels : numpy.ndarray
        Array of DANTE channel numbers.
    
    knots : numpy.ndarray
        Array of photon energies describing positions of spline knots.
    
    responseUncertaintyFrame : pandas.core.frame.DataFrame
        DataFrame holding uncertainty percentages of DANTE channel responses
        as a function of photon energy (not normalized).
        
    npts : int, optional
        Number of points used in computing the integral. The default is 1000.
        
    Returns
    -------
    detArrVariance : xarray.DataArray
        2D array of channels and knot points uncertainties of shape `(n, n+1)`.

    detArrVarianceBoundaryCol : xarray.DataArray
        Column of variances in the cublic spline matrix corresponding to the 
        knots at the boundary chosen with `boundary`.
        
    Notes
    -----
    Covariances between segments is not currently accounted for. This
    covariance should be small compared to the other uncertainties, but should
    be noted.
    
    See also
    --------
    cspline.detectorArr()
    
    Examples
    --------
    
    """
    # number of DANTE channels where we have useful measurements
    chLen = len(channels)
    # initialize normalized energies array
    # array of normalized energies over which we do the integral
    energyNorms = np.linspace(0, 1, num=npts)
    # producing segments from knotsUncertainty. Use the same segmentsArr
    #because there is no calculation done
    segments = segmentsArr(knots)
    # calculating array for converting from values of D_i to y_i. This
    # is an optimization as this array is constant!
    dToY = dToyArr(chLen)
    # M_{y \chi} coefficients array corresponding to given normalized
    # energies. Array shape is (energyNorms, segments, knotIndex).
    yChis = yChiCoeffArrEnergies(energyNorms, chLen, dToY)
    print("running fancytrapz2Variance")
    integFoldArrVariance = fancyTrapz2Variance(energyNorms,
                               yChis,
                               segments,
                               responseUncertaintyFrame,
                               channels)
    print("finished fancytrapz2")
    # sum along segment axis, as each segment must contribute to the
    # overall signal.
    #detArrVariance = np.sum(integFoldArrVariance, axis=1)
    detArrVariance = integFoldArrVariance.sum(dim="segment")
    detArrVariance.attrs['boundary'] = boundary
    if boundary == "y0":
        # extracting column corresponding to y0
        detArrVarianceBoundaryCol = detArrVariance.isel(knot_point=0)
        detArrVariance = detArrVariance.isel(knot_point=slice(1, None))
    elif boundary == "yn+1":
        # extracting column corresponding to y_{n+1}
        detArrVarianceBoundaryCol = detArrVariance.isel(knot_point=-1)
        detArrVariance = detArrVariance.isel(knot_point=slice(None, -1))
    else:
        raise Exception(f"No method found for boundary {boundary}.")
    return detArrVariance, detArrVarianceBoundaryCol


def detectorUncertainty(channels,
                        responseFile,
                        responseUncertaintyFile=None,
                        boundary="y0",
                        npts=1000,
                        samples=1000,
                        MChistogram=False,
                        saveDataset=True,
                        csplineDatasetFile=''):
    r"""
    Finds the cspline detector matrix, it`s inverse matrix and std matrix
    using Monte Carlo uncertainty propagation.

    Propagates response uncertainties through 

    Parameters
    ----------
    channels: numpy.ndarray
        Array of DANTE channel numbers.
        
    responseFile: str
        Path to the `.csv` holding DANTE channel responses as a function of 
        photon energy (not normalized).
    
    responseUncertaintyFile: str, optional
        Path to the `.csv` holding DANTE channel response uncertainties as a 
        function of photon energy. Uncertainty values provided as percentages.

    boundary: str, optional
        Choose whether yGuess corresponds to :math:`y_0` (lowest photon
        energy) or :math:`y_{n+1}` (highest photon energy) boundary condition.
        This should correspond to the photon energy value given in knots.
        Options are `y0` or `yn+1`. Default 'y0'.
    
    npts: int, optional
        Number of points used in computing the integral. Default is 1000.
    
    samples: int, optional
        Number of samples to generate during Monte Carlo propagation.
        See :func:`error.detectorErrMC`. Default is 1000.
        
    Returns
    -------
    detArr : xarray.DataArray
        Matrix representing the spectrally integrated folding of the detector
        response with a cubic spline interpolation of the x-ray spectrum.
        2D array of channels and knot points of shape (n, n).
 
    detArrBoundaryCol : xarray.DataArray
        Column of cublic spline matrix corresponding to the knots at the
        boundary chosen with `boundary`.
        
    detArrVarianceBoundaryCol: xarray.DataArray
        Column of variances in the cublic spline matrix corresponding to the 
        knots at the boundary chosen with `boundary`.
    
    detArrInv : xarray.DataArray
        Inversion of detArr, with the column corresponding to boundary removed
        so detArr is invertible.
   
    stdDetArrInv : xarray.DataArray
        Array of the standard deviation of each element in detArrInv based on
        variance using the `responseUncertaintyFrame` propagated with Monte
        Carlo. 
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    #load response functions
    responseFrame = loadResponses(channels, responseFile)
    
     #create responseUncertainty with 0s if not given.
    if responseUncertaintyFile is None:
        responseUncertaintyFrame = responseFrame.copy()
        cols = list(responseFrame.columns.values)
        cols.remove('Energy(eV)')
        for col in cols:
            responseUncertaintyFrame.loc[:,col] = 0
    else:
        responseUncertaintyFrame = loadResponseUncertainty(responseFrame, responseUncertaintyFile)

    knots = knotFind(channels, responseFrame)
    
    #knotsVariance = knotVarianceFind(channels, responseUncertaintyFrame)
    if not areDataFramesCompatible(channels, responseFrame, responseUncertaintyFrame):
        raise ValueError("Response frame and response uncertainty frame are not compatible. Check formats.")
    detArr, detArrBoundaryCol = detectorArr(channels,
                                            knots,
                                            responseFrame,
                                            boundary,
                                            npts)
    detArrVariance, detArrVarianceBoundaryCol = detectorArrVariance(channels,
                                                                    knots,
                                                                    responseUncertaintyFrame,
                                                                    boundary,
                                                                    npts)
    detArrInv, stdDetArrInv = detectorErrMC(detArr,
                                            detArrVariance,
                                            samples,
                                            boundary,
                                            MChistogram)
    #combine into Dataset
    csplineDataset = xr.Dataset(data_vars={'detArr' : detArr,
                                           'detArrBoundaryCol' : detArrBoundaryCol,
                                           'detArrVarianceBoundaryCol' : detArrVarianceBoundaryCol,
                                           'detArrInv' : detArrInv,
                                           'stdDetArrInv' : stdDetArrInv}, 
                                attrs={'boundary' : boundary,
                                       'channels' : np.asarray(channels), 
                                       'yGuess' : 0,
                                       'responseFile' : responseFile, 
                                       'responseUncertaintyFile' : responseUncertaintyFile,
                                       'generatedDatetime' : str(datetime.datetime.now())})
    if saveDataset:
        #save as netCDF as recommended by http://xarray.pydata.org/en/stable/io.html#netcdf
        if not csplineDatasetFile:
            csplineDatasetFile = 'csplineDataset_' + str(datetime.datetime.now().date()) + '.nc'
        csplineDataset.to_netcdf(csplineDatasetFile)
    return csplineDataset 

def pchipMC(responseFrame,
            channels,
            timesFrame,
            df,
            time,
            randErrors = 'default',
            sysErrors = 'default',
            samples = 1000):
    chanErrorsSelect = randErrors[np.array(channels)-1]
    # samples = 1000 #number of samples in MC
    responseFrameMC = responseFrame.copy()
    vSignals = signalsAtTime(time,
                             timesFrame,
                             df,
                             channels)
    initial = np.clip(Linespline[10], 0, np.inf)
    # bounds = [(0, np.inf),(0, np.inf),(0, np.inf),(0, np.inf),
    #           (0, np.inf),(0, np.inf),(0, np.inf),(0, np.inf),
    #           (0, np.inf),(0, np.inf)]
    bounds = [(0, np.inf) for _ in range(len(channels))]
    
    y0 = 1e-1*initial[0]
    
    interpLen = responseFrame.shape[0]*10
    xInterp = np.linspace(min(knots), max(knots), num = interpLen)
    
    # initialize storage arrays
    splineVals = np.zeros((samples, len(goodChan)+1))
    randVoltages = np.zeros((samples, len(goodChan)))
    fidelityVals = np.zeros((samples, len(detChan)))
    deltaVals = np.zeros((samples, len(goodChan)))
    
    
    for idx in range(0, samples):
        randVoltages[idx] = np.random.normal(vSignals, 0.05)
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
        fiduciaSolve = minimize(minFunc,
                                initial,
                                args = (responseFrameMC,
                                        goodChan,
                                        knots,
                                        vSignals,
                                        y0),
                                method = "Nelder-Mead",
                                bounds = bounds)
        splineVals[idx] = np.insert(fiduciaSolve.x, 0, y0)
        print( "solver completed")
        pchipSpline = pchip(knots, splineVals[idx], xInterp)
        fidelityVals[idx] = cspline.checkFidelity(vSignals,
                                                  goodChan,
                                                  xInterp,
                                                  pchipSpline,
                                                  responseFrameMC,
                                                  plot = False)
        
        deltaVals[idx] = fidelityVals[idx][:10] - vSignals
        print("step "+str(idx) + " complete")
    
        return None
