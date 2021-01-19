
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 12:22:01 2019

FIDUCIA: Filtered Diode Unfolder (using) Cubic Spline Algorithm

DANTE spectrum deconvolver based on cubic splines method [1]. Translated
from Dan Barnak's Mathematica code.

DANTE channels are bounded by edge absorption feature (knot point) due to
filter for the respective channel. Cubic splines representing the estimated
spectrum are fitted in each spectral region bounded by knot points. The
detector signal for each channel is then equal to the response function of
the detector folded with the matrix representation of the cubic spline. A
triadiagonal matrix representation of the cubic spline equation is used to
make the problem numerically tractable. This way a matrix inversion can be
used to solve for the unknown coefficients in the cubic spline equation,
using the measured signals. These coefficients are then plugged back into
the cubic spline equation over each interval (between knot points) to
make a piecewise reconstruction of the x-ray spectrum at each time step.
    

References
----------

Cubic spline deconvolution method
[1] J. P. Knauer and N. C. Gindele. Temporal and spectral deconvolution of
data from diamond, photoconductive devices. Rev. Sci. Instrum. 75, 3714 (2004)
https://doi.org/10.1063/1.1785274

Error propagation for cubic spline deconvolution method
[2] D. L. Fehl and F. Briggs. Verification of unfold error estimates in the
unfold operator code. Rev. Sci. Instrum. 68, 890 (1997)
https://doi.org/10.1063/1.1147713
    
Useful description of cubic spline matrix representation
[3] http://mathworld.wolfram.com/CubicSpline.html

Paper comparing cubic splines unfolds to other methods
[4] D. H. Barnak, J. R. Davies, J. P. Knauer, and P. M. Kozlowski. Soft
x-ray spectrum unfold of K-edge filtered x-ray diode arrays using
cubic splines. Submitted to Review of Scientific Instruments in 2020.

@author: Pawel M. Kozlowski
"""

# python modules
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import sigma_sb, k_B
import xarray as xr

# custom modules
from fiducia.rawProcess import (loadCorrected,
                                hysteresisCorrect,
                                align,
                                constructMeasurementFrame)
from fiducia.loader import signalsAtTime, loadResponses
from fiducia.cspline import knotSolve, reconstructSpectrum
from fiducia.visualization import plotStreak, plotTraces, plotResponse
from fiducia.response import knotFind
from fiducia.error import trapzVariance
import fiducia.pltDefaults


# listing all functions declared in this file so that sphinx-automodapi
# correctly documents them and doesn't document imported functions.
__all__ = ["simulateSignal",
           "inferRadTemp",
           "inferPower",
           "analyzeSpectrum",
           "analyzeStreak",
           "feelingLucky",
           ]


def simulateSignal():
    """
    Takes the inferred spectrum and folds it with the instrument
    function to retrieve the forward propagated signal for each
    DANTE channel.
    
    Parameters
    ----------
    
    Returns
    -------
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    return


def inferRadTemp(power, area, angle, powerUncertainty=None):
    r"""
    Gets the inferred radiation temperature by calculating in from radiated
    power through the Stefan-Boltzmann Law.
    
    Parameters
    ----------
    power: float, np.ndarray
        Total radiated power as a function of time calculated from unfolded
        spectra. See :func:`main.inferPower()`.

    area: float
        Area of emitting surface in units of mm^2. For hohlraums/halfraums,
        this is the area of the LEH.

    angle: float
        Angle between the surface area normal and the Dante line of sight in 
        degrees. Usually 37.4 degrees for hohlraums/halfraums. Must be
        between 0 and 90 degrees.

    powerUncertainty: float, np.ndarray, optional
        Uncertainty in total radiated power as a function of time calculated from unfolded
        spectra. See :func:`main.inferPower()`. The default is None.

    Returns
    -------
    tRad : numpy.ndarray
        Radiation temperature of the blackbody emitter.
        
    tRadVariance: numpy.ndarray
        Variance :math:`\sigma^2` on the radiation temperature.

    Notes
    -----
    Total x-ray flux (power) from a black body emitter is given by:
        
    .. math::
        P = \sigma_{SB} A \cos(\theta) T^4
        
        
    Where P = power, :math:`\sigma_{SB}` = Stefan-Boltzmann constant, A is the
    area of radiating surface, :math:`\theta` is the viewing angle between the
    surface area normal and the Dante line-of-sight, T is the radiation
    temperature of the black body emitter.
    
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    # limit viewing angles to 90 degrees.
    if not 0 <= angle <= 90:
        raise Exception("Angle must be between 0 and 90 degrees.")
    
    # when power uncertainty is not given, it is set to zero.
    if powerUncertainty is None:
        if type(power) is np.ndarray:
            powerUncertainty = np.zeros(power.shape)
        else:
            powerUncertainty = 0
    
    cosAngle = np.cos(np.deg2rad(angle))
    #Stefan-Boltzmann constant
    sb = sigma_sb.to(u.GW/(u.mm ** 2 * u.K ** 4))
    sigma_sb_unitless = sb.value
    kb = k_B.to(u.eV/u.K).value
    denom = area * cosAngle * sigma_sb_unitless
    # value to be quadratic rooted for getting radiation temperature
    val2Root = power / denom
    # quadratic rooting to get Trad in Kelvins, and converting from
    # Kelvins to electronvolts.
    tRad = np.power(val2Root, 0.25) * kb
    tRadVariance = 0.0
    if np.shape(powerUncertainty) == np.shape(power):
#        tRadVariance = kb ** 2 / np.sqrt(denom) * (0.25 * power ** (-0.75) * powerUncertainty) ** 2
        tRadVariance = (0.25 * tRad * powerUncertainty / power) ** 2
        # cleaning up cases where we get nan because input power is zero
        tRadVariance = np.nan_to_num(tRadVariance)
    return tRad, tRadVariance


def inferPower(energies, spectra, spectraUncertainty=None):
    r"""
    Gets the inferred total radiation power as a function of time.
    
    Parameters
    ----------
    energies: numpy.ndarray
        Photon energies corresponding to input spectrum
        
    spectra: numpy.ndarray
        Spectral Flux values as a function of photon energy in units of
        (GW/sr/eV)
    
    Returns
    -------
    power: numpy.ndarray
        Total x-ray power (flux) as a function of time.
        
    powerVariance: numpy.ndarray
        Variance :math:`\sigma^2` on total x-ray power.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    if spectraUncertainty is None:
        spectraUncertainty = np.zeros(spectra.shape)

    timesLength = np.shape(spectra)[1]
    power = np.zeros(timesLength)
    powerVariance = np.zeros(timesLength)
    for idx in range (0, timesLength):
        power[idx] = np.trapz(y = spectra[:, idx], x=energies[:, idx])
        powerVariance[idx] = trapzVariance(spectraUncertainty[:, idx],
                                           x=energies[:, idx])
    return power, powerVariance


def analyzeSpectrum(channels,
                    knots,
                    detArr,
                    detArrBoundaryCol,
                    detArrVarianceBoundaryCol,
                    detArrInv,
                    stdDetArrInv,
                    measurementFrame,
                    time,
                    signalsUncertainty = None,
                    yGuess=0,
                    boundary="y0",
                    nPtsIntegral=100,
                    nPtsSpectrum=100,
                    plotSignal=False,
                    plotKnots=False,
                    plotSpectrum=True):
    r"""
    Given the response function file and the DANTE measurement data file,
    run cubic spline analysis to reconstruct spectrum for a given time.
    
    
    Parameters
    ----------
    channels: list, numpy.ndarray
        List or array of relevant DANTE channel numbers.
    
    responseFrame: pandas.core.frame.DataFrame
        Pandas dataFrame containing response functions for each DANTE
        channel. See loadResponses().
        
    knots: list, numpy.ndarray
        List or array of knot point photon energy value. See knotFind().
    
    detArr : xarray.DataArray
        Matrix representing the spectrally integrated folding of the detector
        response with a cubic spline interpolation of the x-ray spectrum.
        2D array of channels and knot points of shape (n, n).
 
    detArrBoundaryCol : xarray.DataArray
        Column of cublic spline matrix corresponding to the knots at the boundary
        chosen with `boundary`.
 
    detArrVarianceBoundaryCol: xarray.DataArray
        Column of variances in the cublic spline matrix corresponding to the 
        knots at the boundary chosen with `boundary`.   
 
    detArrInv : xarray.DataArray
        Inversion of detArr, with the column corresponding to boundary removed so detArr is invertible.
   
    stdDetArrInv : xarray.DataArray
        Array of the standard deviation of each element in detArrInv based on variance
        using the `responseUncertaintyFrame` propagated with Monte Carlo. 
    
    measurementFrame: pandas.core.frame.DataFrame
        Pandas dataframe containing DANTE measurement data. See
        readDanteData() and readDanProcessed().
        
    time: float
        Time for which we want DANTE signals (in ns).

    signalsUncertainty: numpy.ndarray, optional
        One dimensional array with each element corresponding to the uncertainty
        each signal. The default is None.        
        
    yGuess: float, optional
        Guess for position of boundary knot point. Default 0.
        
    boundary: str, optional
        Choose whether yGuess corresponds to :math:`y_0` (lowest photon energy) or
        :math:`y_{n+1}` (highest photon energy) boundary condition. This should
        correspond to the photon energy value given in knots. Options are `y0`
        or `yn+1`. Default 'y0'.
    
    nPtsIntegral: int, optional
        Number of points used in computing the integral. Default is 100.
        
    nPtsSpectrum: int, optional
        Number of points to use in reconstructing the spectrum. Default is 100.
    
    plotKnots: Bool, optional
        Flag for plotting the Dante signal at the given time across all
        channels. Default is False.
    
    plotKnots: Bool, optional
        Flag for plotting just the solved knot points. Default is False.
        
    plotSpectrum: Bool, optional
        Flag for plotting the unfolded spectrum. Default is True.
    
    Returns
    -------    
    knotsYAll: numpy.ndarray
        Array of knot point intensity values with yGuess appended.
        See knotSolve() and analyzeSpectrum().

    knotsYVariance: numpy.ndarray
        Array of knot point intensity uncertainty values with yGuess appended.
        See knotSolve() and analyzeSpectrum().
    
    photonEnergies: numpy.ndarray
        Photon energy axis of unfolded spectrum.
        
    intensities: numpy.ndarray
        Spectral intensity axis of unfolded spectrum.
        
    intensitiesVariance: numpy.ndarray
        Uncertaitny (1 :math:`\sigma`) on spectral intensity values.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    # Select a particular time step for generating x-ray spectrum from DANTE
    signals = signalsAtTime(time=time,
                            measurementFrame=measurementFrame,
                            channels=channels,
                            plot=plotSignal,
                            method="interp")
    # solve knots and plot
    # if signalsUncertainty is None:
    #     signalsUncertainty = np.ones(signals.shape)
        
    knotsY, knotsYVariance = knotSolve(signals,
                                       detArr, detArrBoundaryCol,
                                       detArrVarianceBoundaryCol,
                                       detArrInv, stdDetArrInv,
                                       signalsUncertainty,
                                       yGuess=yGuess,
                                       npts=nPtsIntegral)
    #Root the variance to get the uncertainty (sigma)
    knotsYUncertainty = np.sqrt(knotsYVariance)
    
    # prepending y0 guess to form full array of y knot values equal in
    # length to knot energy values
    if boundary == "y0":
        knotsYAll = np.append([yGuess], knotsY)
        knotsYUncertaintyAll = np.append([yGuess], knotsYUncertainty)

    elif boundary == "yn+1":
        knotsYAll = np.append(knotsY, [yGuess])
        knotsYUncertaintyAll = np.append(knotsYUncertainty, [yGuess])
    else:
        raise Exception(f"No method found for boundary {boundary}.")
    
    # making simple plot at this intermediate step to get rough idea
    # of the shape of the spectrum
    if plotKnots:
            plt.plot(knots, knotsYAll)
            plt.xlabel("Energy (eV)")
            plt.ylabel("X-ray spectrum")
            plt.title("cubic spline knots")
            plt.show()
    
    # reconstruct spectrum using inferred knot points
    chLen = len(channels)
    photonEnergies, intensities, intensitiesVariance = reconstructSpectrum(chLen,
                                                                           knots,
                                                                           knotsYAll,
                                                                           knotsYUncertaintyAll,
                                                                           nPtsSpectrum,
                                                                           plot=plotSpectrum)
    photonEnergiesPlus, intensitiesPlus, _ = reconstructSpectrum(chLen,
                                                                 knots,
                                                                 knotsYAll+knotsYUncertaintyAll,
                                                                 knotsYUncertaintyAll,
                                                                 nPtsSpectrum,
                                                                 plot=False)
    photonEnergiesMinus, intensitiesMinus, _ = reconstructSpectrum(chLen,
                                                                   knots,
                                                                   knotsYAll-knotsYUncertaintyAll,
                                                                   knotsYUncertaintyAll,
                                                                   nPtsSpectrum,
                                                                   plot=False)
    #subtract to just get the delta from intensities. plot_line_shaded just wants the difference
    intensitiesPosDiff = intensitiesPlus-intensities
    intensitiesNegDiff = intensities - intensitiesMinus
    if plotSpectrum:
        fiducia.pltDefaults.plot_line_shaded(photonEnergies, 
                                             intensities, 
                                             intensitiesPosDiff, 
                                             intensitiesNegDiff,)
        plt.xlabel("Photon Energy (eV)")
        plt.ylabel("Spectrum (GW/sr/eV)")
        plt.title("Spectra with error bars")
        plt.show()    
    return knotsYAll, knotsYVariance, photonEnergies, intensities, intensitiesVariance


def analyzeStreak(channels,
                  responseFrame,
                  knots,
                  detArr,
                  detArrBoundaryCol,
                  detArrVarianceBoundaryCol,
                  detArrInv, 
                  stdDetArrInv,
                  measurementFrame,
                  timeStart,
                  timeStop,
                  timeStep,
                  signalsUncertainty=None,
                  yGuess=0,
                  boundary="y0",
                  nPtsIntegral=100,
                  nPtsSpectrum=100):
    r"""
    Given the response function file and the DANTE measurement data file,
    run cubic spline analysis to reconstruct spectrum for a given time.
    
    Parameters
    ----------
    channels: list, numpy.ndarray
        List or array of relevant DANTE channel numbers.
    
    responseFrame: pandas.core.frame.DataFrame
        Pandas dataFrame containing response functions for each DANTE
        channel. See loadResponses().
        
    knots: list, numpy.ndarray
        List or array of knot point photon energy value. See knotFind().
    
    detArr : xarray.DataArray
        Matrix representing the spectrally integrated folding of the detector
        response with a cubic spline interpolation of the x-ray spectrum.
        2D array of channels and knot points of shape (n, n).
 
    detArrBoundaryCol : xarray.DataArray
        Column of cublic spline matrix corresponding to the knots at the boundary
        chosen with `boundary`.
 
    detArrVarianceBoundaryCol: xarray.DataArray
        Column of variances in the cublic spline matrix corresponding to the 
        knots at the boundary chosen with `boundary`.   
        
    detArrInv : xarray.DataArray
        Inversion of detArr, with the column corresponding to boundary removed so detArr is invertible.
   
    stdDetArrInv : xarray.DataArray
        Array of the standard deviation of each element in detArrInv based on variance
        using the `responseUncertaintyFrame` propagated with Monte Carlo. 
    
    measurementFrame: pandas.core.frame.DataFrame
        Pandas dataframe containing DANTE measurement data. See
        :func:`loader.readDanteData` and :func:`readDanProcessed`.
        
    timeStart: float
        Start time for producing temporally streaked DANTE spectra (in ns).
        
    timeStop: float
        End time for producing temporally streaked DANTE spectra (in ns).
        
    timeStep: float
        Time step size for producing temporally streaked DANTE spectra (in ns).
        
    signalsUncertainty: numpy.ndarray, optional
        One dimensional array with each element corresponding to the uncertainty
        each signal. The default is None. 
    
    yGuess: float, optional
        Guess for position of boundary knot point. Default is 1e-77.
        
    boundary: str, optional
        Choose whether yGuess corresponds to :math:`y_0` (lowest photon energy) or
        :math:`y_{n+1}` (highest photon energy) boundary condition. This should
        correspond to the photon energy value given in knots. Options are `y0`
        or `yn+1`. Default 'y0'.
    
    nPtsIntegral: int, optional
        Number of points used in computing the integral. Default is 100.
        
    nPtsSpectrum: int, optional
        Number of points to use in reconstructing the spectrum. Default is 100.
    
    Returns
    -------
    times
    
    
    energies
    
    
    spectra
    
    
    spectraVariance
        
    
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
        
    """
    chLen = len(channels)
    # generating evenly spaced timesteps
    times = np.arange(timeStart, timeStop + timeStep, timeStep)
    timesLen = len(times)
    # initializing array for holding spectral values
    spectralPts = chLen * nPtsSpectrum - chLen
    energies = np.zeros((spectralPts, timesLen))
    spectra = np.zeros((spectralPts, timesLen))
    #TODO implement spectraVariance once DotVariance works on N-D arrays
    spectraVariance = np.zeros((spectralPts, timesLen))
    
    for idt, time in enumerate(times):
        # process the spectrum for the given time
        results = analyzeSpectrum(channels,
                                  knots,
                                  detArr, 
                                  detArrBoundaryCol,
                                  detArrVarianceBoundaryCol,
                                  detArrInv, 
                                  stdDetArrInv,
                                  measurementFrame,
                                  time,
                                  signalsUncertainty=signalsUncertainty,
                                  yGuess=yGuess,
                                  boundary=boundary,
                                  nPtsIntegral=100,
                                  nPtsSpectrum=100,
                                  plotKnots=False,
                                  plotSpectrum=False)
        knotsYAll, knotsYVariance, photonEnergies, intensities, intensitiesVariance = results
        # including unfolding spectrum for time step into array of spectra
        # for all time steps
        energies[:, idt] = photonEnergies
        spectra[:, idt] = intensities
        spectraVariance[:, idt] = intensitiesVariance
        print(f"Completed time step {time} ns.")    
    # plotting streaked spectrum
    plotStreak(times, energies, spectra)
    return times, energies, spectra, spectraVariance


def feelingLucky(dataFile,
                 attenuatorsFile,
                 offsetsFile,
                 responseFile,
                 csplineDatasetFile,
                 channels,
                 area,
                 angle,
                 signalsUncertainty=None,
                 peaksNum=2):
    r"""
    Attempt processing dante signals given dante data file and calibration
    files using sensible defaults.
    
    
    Parameters
    ----------
    dataFile: str
        Full path to the Dante .dat file containing dante signals from LLE
        site.
        
    attenuatorsFile: str
        Full path to file containing attenuator serial numbers and
        corresponding attenuation factors.
        
    offsetsFile: str
        Full path to file containing oscilloscope channel offsets in
        time and voltage.
        
    responseFile: str
        Full path and filename of .csv file containing DANTE respones
        functions corresponding to dataFile.
    
    csplineDatasetFile : str
        File pointing to the path of the saved dataset containing ''detArr'',
        ''detArrBoundaryCol'',  ''detArrInv'', and ''stdDetArrInv''. 
        See :func:'error.analyzeSpectrumUncertainty()'.
  
    channels: list, numpy.ndarray
        List or array of relevant channels for which to apply analysis.
        
    area: float
        Area of emitting surface in units of mm^2. For hohlraums/halfraums,
        this is the area of the LEH. Used in Trad calculation.
        
    angle: float
        Angle between the surface area normal and the Dante line of sight in 
        degrees. Usually 37.4 degrees for hohlraums/halfraums. Used in Trad 
        calculation.
    
    signalsUncertainty: numpy.ndarray, optional
        One dimensional array with each element corresponding to the uncertainty
        each signal. The default is None.
    
    Returns
    -------
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    #load cspline matrix related files
    csplineDataset =  xr.open_dataset(csplineDatasetFile).load()
    detArr = csplineDataset['detArr']
    detArrBoundaryCol = csplineDataset['detArrBoundaryCol']
    detArrVarianceBoundaryCol = csplineDataset['detArrVarianceBoundaryCol']
    detArrInv = csplineDataset['detArrInv']
    stdDetArrInv = csplineDataset['stdDetArrInv']
    #close out of the files
    csplineDataset.close()
    
    #find what boundary was used when calculating cspline matrices
    boundary = csplineDataset.attrs['boundary']
    
    # loading data and applying corrections.
    timesFrame, dfAtten, onChList, hf, dfVolt = loadCorrected(danteFile=dataFile,
                                                              attenuatorsFile=attenuatorsFile,
                                                              offsetsFile=offsetsFile,
                                                              plot=True)
    
    # removing hysteresis and background with a polynomial fit
    dfPoly = hysteresisCorrect(timesFrame=timesFrame,
                               df=dfAtten,
                               channels=channels,
                               order=5,
                               prominence=0.2,
                               width=10,
                               avgMult=1)
    
    # aligning signals to peak
    # aligning to 1e-9 seconds by default.
    # looking for 2 peaks and aligning to the first (0th index) peak.
    timesAligned = align(timesFrame=timesFrame,
                         df=dfPoly,
                         channels=channels,
                         peaksNum=peaksNum,
                         peakAlignIdx=0,
                         referenceTime=1e-9,
                         prominence=0.01,
                         width=10,
                         avgMult=1.5)
    
    # constructing dataframe for passing to analyzeStreak()
    measurementFrame = constructMeasurementFrame(timesFrame=timesAligned,
                                                 df=dfPoly,
                                                 channels=channels)
        
    # testing plot traces
    plotTraces(channels, measurementFrame, scale='log')
    
    # loading dante responses
    responseFrame = loadResponses(channels, responseFile)

    # getting knots from response function edges
    # currently not forcing any knots
    knots = knotFind(channels=channels,
                      responseFrame=responseFrame,
                      forceKnot=np.array([]),
                      knotBoundary=0,
                      boundary=boundary)
    
    # print(f"Channels: {channels}")
    # print(f"Channels length: {len(channels)}")
    # print(f"Knot energies (eV): {knots}")
    # print(f"Knot energies length: {len(knots)}")
    
    # plotting all relevant response functions
    # plotResponse(channels=channels, 
    #              responseFrame=responseFrame,
    #              knots=knots)
    
    # unfolding the spectrum for a particular time step
    # currently unfolding at 1 ns
    time = 1.0
    spectrumResults = analyzeSpectrum(channels, 
                                      knots, 
                                      detArr,
                                      detArrBoundaryCol,
                                      detArrVarianceBoundaryCol,
                                      detArrInv,
                                      stdDetArrInv,
                                      measurementFrame,
                                      signalsUncertainty=signalsUncertainty,
                                      time=time,
                                      yGuess=0,
                                      boundary=boundary,
                                      plotKnots=True)
    knotsYAll, knotsYVariance, photonEnergies, intensities, intensitiesVariance = spectrumResults
    
    # unfolding time resolved spectra
    timeStart = -1
    timeStop = 4
    timeStep = 0.1
    streakResults = analyzeStreak(channels,
                                  responseFrame,
                                  knots,
                                  detArr, 
                                  detArrBoundaryCol,
                                  detArrVarianceBoundaryCol,
                                  detArrInv, 
                                  stdDetArrInv,
                                  measurementFrame,
                                  timeStart,
                                  timeStop,
                                  timeStep,
                                  signalsUncertainty=signalsUncertainty,
                                  yGuess=0,
                                  boundary=boundary,
                                  nPtsIntegral=100,
                                  nPtsSpectrum=100)
    times, energies, spectra, spectraVariance = streakResults


    spectraUncertainty = np.sqrt(spectraVariance)
    power, powerVariance = inferPower(energies, spectra, spectraUncertainty)
    plt.scatter(times, power)
    plt.xlabel("Time (ns)")
    plt.ylabel("Radiated Power (GW/sr)")
    plt.show()
    powerUncertainty = np.sqrt(powerVariance)
    fiducia.pltDefaults.plot_line_shaded(times, power, powerUncertainty)
    plt.xlabel("Time (ns)")
    plt.ylabel("Radiated Power (GW/sr)")
    axes = plt.gca()
    axes.set_ylim([0,None])
    plt.show()
    
    tRad, tRadVariance = inferRadTemp(power, area, angle, powerUncertainty)
    plt.scatter(times, tRad)
    plt.xlabel("Time (ns)")
    plt.ylabel("Radiation Temperature (eV)")
    plt.show()
    tRadUncertainty = np.sqrt(tRadVariance)
    fiducia.pltDefaults.plot_line_shaded(times, tRad, tRadUncertainty)
    plt.xlabel("Time (ns)")
    plt.ylabel("Radiation Temperature (eV)")
    axes = plt.gca()
    axes.set_ylim([0,None])
    plt.show()
#    return None
    return times, energies, spectra, power, tRad
