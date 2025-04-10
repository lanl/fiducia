"""
Created on Fri Mar  8 09:41:36 2019

Functions for working with cubic spline equation in matrix form.

@author: Pawel M. Kozlowski
"""

# python modules
import numpy as np
import scipy.sparse as sparse
from scipy import integrate
from scipy.interpolate import pchip_interpolate as pchip
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
# custom modules
import fiducia.pltDefaults
import fiducia.loader
from fiducia.stats import dotVariance
from fiducia.loader import signalsAtTime
from numba import jit

# listing all functions declared in this file so that sphinx-automodapi
# correctly documents them and doesn't document imported functions.
__all__ = ["splineCoords",
           "splineCoordsInv",
           "yCoeffArr",
           "dCoeffArr",
           "dToyArr",
           "responseInterp",
           "yChiCoeffArr",
           "yChiCoeffArrEnergies",
           "fancyTrapz2",
           "segmentsArr",
           "detectorArr",
           "knotSolve",
           "reconstructSpectrum",
           ]


def splineCoords(energy, energyStart, energyEnd):
    r"""
    Convert photon energy value into normalized coordinates for a particular
    spline region.
    
    Parameters
    ----------
    energy: float, numpy.ndarray
        Energy value to be converted into normalized spline coordinate.
        
    energyStart: float
        Lower bound energy for the spline region based on knot points.
        
    energyEnd: float
        Upper bound energy for the spline region based on knot points.
    
    normCoord: float
        Return value of energy converted into normalized spline coordinates.
    
    Returns
    -------
    normCoord: float, numpy.ndarray
        Normalized energy coordinate(s).
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    
    """
#    if energyStart > energyEnd:
#        raise ValueError(f"Lower bound energy {energyStart} must be less than "
#                        f"upper bound energy {energyEnd}.")
#    if np.all(energy < energyStart) or np.all(energy > energyEnd):
#        raise ValueError(f"Given energy value {energy} is outside spline "
#                        f"coordinate bounds {energyStart} to {energyEnd}.")
    normCoord = (energy - energyStart) / (energyEnd - energyStart)
    return normCoord


def splineCoordsInv(energyNorm, energyStart, energyEnd):
    r"""
    Given a normalized energy value and the bounds of a spline segment, 
    return the un-normalized photon energy value. This is the inverse of
    splineCoords().
    
    Parameters
    ----------
    energyNorm: float, numpy.ndarray
        Normalized photon energy
        
    energyStart: float
        Lower bound energy for the spline region based on knot points.
        
    energyEnd: float
        Upper bound energy for the spline region based on knot points.
    
    Returns
    -------
    energy: float, numpy.ndarray
        Absolute photon energy (un-normalized).
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    energy = (energyEnd - energyStart) * energyNorm + energyStart
    return energy


def yCoeffArr(energyNorm, chLen):
    r"""
    Returns the matrix M_y(t) for a given value of t in:
        
    .. math::
        Y_i(t) = M_y(t) y_i + M_D(t) D_i
        
    Parameters
    ----------
    energyNorm: float
        normalized photon energy for a spline section.
        
    chLen: int
        Number of DANTE channels (equal to number of spline knots).
    
    Returns
    -------
    mArr: scipy.sparse.lil.lil_matrix
        Sparse matrix :math:`M_y(t)`.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    coeff1 = 1 - 3 * energyNorm ** 2 + 2 * energyNorm **3 
    coeff2 = 3 * energyNorm ** 2 - 2 * energyNorm ** 3
    mArr = sparse.diags([coeff1, coeff2],
                        [0,1],
                        shape=(chLen, chLen + 1),
                        format='lil')
    #First and last row elements in y array are different 
    #(added by DHB 3/25/19)
    mArr[0,0] = 1 - energyNorm ** 3
    mArr[0,1] = energyNorm ** 3
    mArr[chLen-1,chLen-1] = 1 - energyNorm ** 3
    mArr[chLen-1,chLen] = energyNorm ** 3
    return mArr.tocsc()


def dCoeffArr(energyNorm, chLen):
    r"""
    Returns the matrix M_D(t) for a given value of t in:
        
    .. math::
        Y_i(t) = M_y(t) y_i + M_D(t) D_i
    
    Parameters
    ----------
    energyNorm: float
        normalized photon energy for a spline section.
        
    chLen: int
        Number of DANTE channels (equal to number of spline knots).
    
    Returns
    -------
    dArr: scipy.sparse.lil.lil_matrix
        Sparse matrix :math:`M_D(t)`.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    coeff1 = energyNorm - 2 * energyNorm ** 2 + energyNorm ** 3
    coeff2 = -1 * energyNorm ** 2 + energyNorm ** 3
    dArr = sparse.diags([coeff1, coeff2],
                        [0,1],
                        shape=(chLen, chLen + 1), 
                        format='lil')
    #First and last row elements in D array are different
    #(added by DHB 3/25/19)
    dArr[0,0] = energyNorm - energyNorm ** 3
    dArr[0,1] = 0
    dArr[chLen-1,chLen-1] = energyNorm - energyNorm ** 2
    dArr[chLen-1,chLen] = energyNorm ** 2 - energyNorm ** 3
    return dArr.tocsc()


def __chi1Arr__(chLen):
    r"""
    Parameters
    ----------
    chLen: int
        Number of DANTE channels (equal to number of spline knots).
    
    Returns
    -------
    chi1: scipy.sparse.lil.lil_matrix
        Sparse matrix :math:`\chi_1`.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    # chi1array in Dan Barnak's notation = TDy'' in Jim's paper 
    chi1 = sparse.diags([1, 4, 1], 
                        [-1,0,1],
                        shape=(chLen + 1, chLen + 1),
                        format='lil')
    chi1[0, 0] = 2
    chi1[-1, -1] = 2
    return chi1.tocsc()


def __chi3Arr__(chLen):
    r"""
    Parameters
    ----------
    chLen: int
        Number of DANTE channels (equal to number of spline knots).
        
    Returns
    -------
    chi3: scipy.sparse.lil.lil_matrix
        Sparse matrix :math:`\chi_3`.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    # chi2array in Dan Barnak's notation = TDy in Jim's paper
    chi3 = sparse.diags([-1, 0, 1],
                        [-1,0,1],
                        shape=(chLen + 1, chLen + 1),
                        format='lil')
    chi3[0, 0] = -1
    chi3[-1, -1] = 1
    return chi3.tocsc()


def dToyArr(chLen):
    r"""
    Construct matrix for converting from :math:`D_i` to :math:`y_i` vector.
    
    Parameters
    ----------
    chLen: int
        Number of DANTE channels (equal to number of spline knots).
        
    Returns
    -------
    diToyi: numpy.ndarray
        Matrix for converting from :math:`D_i` to :math:`y_i` vector.
        
    Notes
    -----
    The matrix is given by:
        
    .. math::
        D_i = 3 \chi_1^{-1} \chi_3 y_i
    
    
    See also
    --------
    
    Examples
    --------
    """
    chi1Arr = __chi1Arr__(chLen)
    chi3Arr = __chi3Arr__(chLen)
    chi1ArrInv = np.linalg.inv(chi1Arr.toarray())
    # constructing array for converting values from Di to yi
    diToyi = 3 * np.dot(chi1ArrInv, chi3Arr.toarray())
    return diToyi


def responseInterp(energyNorm, energyMin, energyMax, responseFrame, channels):
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
        DANTE channel responses as a function of photon energy (not 
        normalized).

    channels: numpy.ndarray
        numpy array of DANTE channel numbers.
    
    Returns
    -------
    responsesInterpd: numpy.ndarray
        Returns a matrix of (energyNorms, channels) of response functions.
        
    Notes
    -----
    
    See also
    --------
    
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
    responsesInterpd = np.zeros((energyLen, chLen))
    # fetch the original array of energy values (not normalized)
    energyArr = responseFrame['Energy(eV)']
    # converting normalized energy values to un-normalized energy values,
    # since our response functions are in terms of absolute photon energy
    energyReg = splineCoordsInv(energyNorm, energyMin, energyMax)
    for idx, channel in enumerate(channels):
        responseArr = responseFrame[channel]
        responsesInterpd[:,idx] = np.interp(x=energyReg,
                                            xp=energyArr,
                                            fp=responseArr)
    return responsesInterpd


def yChiCoeffArr(energyNorm, chLen, dToY):
    r"""
    This is the matrix corresponding to:
        
    .. math::
        M_y(t) + 3 M_D(t) \chi_1^{-1} \chi_3
    
    which describes the cubic spline interpolation of the x-ray
    spectrum.
    
    Parameters
    ----------
    energyNorm: float
        normalized photon energy
        
    chLen: int
        Number of DANTE channels (equal to number of spline knots).
        
    dToY: numpy.ndarray
        Matrix for converting from :math:`D_i` to :math:`y_i` values in
        cubic spline interpolation. See dToyArr().
    
    Returns
    -------
    yChiArr: numpy.ndarray
        Returns a 2D matrix for a particular value of energyNorm.
        
    Notes
    -----
    The matrix is given by:
        
    .. math::
        M_y(t) + 3 M_D(t) \chi_1^{-1} \chi_3
    
    See also
    --------
    
    Examples
    --------
    """
    # M_y(t)
    yCoeff = yCoeffArr(energyNorm, chLen)
    # M_D(t)
    dCoeff = dCoeffArr(energyNorm, chLen)
    # folded x-ray spectrum and detector response array
    yChiArr = yCoeff.toarray() + np.dot(dCoeff.toarray(), dToY)
    return yChiArr


def yChiCoeffArrEnergies(energyNorms, chLen, dToY):
    r"""
    This is the matrix corresponding to:
        
    .. math::
        M_y(t) + 3 M_D(t) \chi_1^{-1} \chi_3
    
    which describes the cubic spline interpolation of the x-ray
    spectrum.
    
    energyNorms: numpy.ndarray
        Vector of normalized photon energies.
        
    chLen: int
        Number of DANTE channels (equal to number of spline knots).
        
    dToY: numpy.ndarray
        Matrix for converting from :math:`D_i` to :math:`y_i` values in cubic
        spline interpolation. See dToyArr().
    
    Returns
    -------
    yChiArrEnergies: numpy.ndarray
        Returns a 3D matrix composed of a series of 2D yChiCoeff matrices
        corresponding to the given energyNorm values. This matrix is
        indexed as (energyNorms, knotIndex, knotIndex).
        
    Notes
    -----
    The matrix is given by:
        
    .. math::
        M_y(t) + 3 M_D(t) \chi_1^{-1} \chi_3
    
    See also
    --------
    
    Examples
    --------
    """
    yChiArrEnergies = np.array([yChiCoeffArr(energyNorm, chLen, dToY) for energyNorm in energyNorms])
    return yChiArrEnergies
    

def fancyTrapz2(energyNorms, yChis, segments, responseFrame, channels):
    r"""
    Trap rule integration of the folding between our :math:`M_{y \chi}` matrix
    and response function matrix, with respect to normalized photon energy,
    for each channel. The result should be a matrix with shape
    (`len(channels)`, `len(segments)`, `len(knotIndex)`).
    
    Parameters
    ----------
    energyNorms: numpy.ndarray
        1D array of normalized photon energies
        
    yChis: numpy.ndarray
        3D array of :math:`M_{y \chi}` values corresponding to
        (energyNorms, segments, knotIndex).
        
    responses: numpy.ndarray
        2D array of DANTE channel response functions corresponding to
        (energyNorms, channels).
        
    channels: numpy.ndarray
        Array of DANTE channel numbers.      
        
    Returns
    -------
    integArr : xarray.DataArray
        A matrix containing the folded integration of the :math:`M_{y \chi}` 
        matrix and response function matrix, with respect to normalized photon 
        energy. Has shape (`len(channels)`, `len(segments)`, `len(knotIndex)`).
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    shape = np.shape(yChis)
    segmentsLen = np.shape(segments)[0]
    knotsLen = shape[2]
    chLen = segmentsLen
    # initialize integArr for storing photon energy integrated values
    integArr = xr.DataArray(np.zeros((chLen, segmentsLen, knotsLen)),
                            dims=['channel', 'segment', 'knot_point'],
                            coords={'channel':channels})
    # loop over relevant DANTE channels for analysis
    for channelIdx in np.arange(integArr['channel'].size):
        # loop over photon energy segments (between knot points)
#        print(f"Calculating for channel {channels[channelIdx]}.")
        for segmentNum, segment in enumerate(segments):
            energyMin, energyMax = segment
            # print(f"Calculating for segment {segments[segmentNum]}.")
            # loop over knot points
            for knotNum in np.arange(integArr['knot_point'].size):
                # multiplication of response by spline matrix
                responses = responseInterp(energyNorms,
                                           energyMin,
                                           energyMax,
                                           responseFrame,
                                           channels)
#                plt.plot(energyNorms, responses[:, channelIdx])
#                plt.title(f"Response channel {channels[channelIdx]}, segment {segmentNum}, knot {knotNum}")
#                plt.show()
                multArr = yChis[:, segmentNum, knotNum] * responses[:, channelIdx]
#                integArr[channelIdx, segmentNum, knotNum] = np.trapz(y=multArr,
#                                                                     x=energyNorms)
                integVal = (energyMax - energyMin) * integrate.simps(y=multArr, x=energyNorms)
                integArr[channelIdx, segmentNum, knotNum] = integVal
    return integArr


def segmentsArr(knots):
    r"""
    Returns the bounds of each spline segment, given the spline knot points.
    
    Returns an array of tuples of (energyMin, energyMax) describing the bounds
    of each spline segment, given an array of spline knots (photon energies
    corresponding to K-edges).
    
    Parameters
    ----------
    knots: numpy.ndarray
        numpy array of photon energies describing positions of spline knots.
        
    Returns
    -------
    segments : numpy.ndarray
        A 1D array of tuples of (energyMin, energyMax), corresponding to the 
        bounds of each spline segment. Has length of 'len(knots) - 1'.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    # initializing array of segments
    segments = np.zeros(len(knots) - 1, dtype=object)
    for idx, knot in enumerate(knots):
        if idx == len(segments):
            # skipping the last knot since there are 1 fewer segments than knots
            continue
        segments[idx] = knots[idx], knots[idx + 1]
    return segments
    

def detectorArr(channels, knots, responseFrame, boundary="y0", npts=1000):
    r"""
    Matrix representing the spectrally integrated folding of the detector
    response with a cubic spline interpolation of the x-ray spectrum. This
    is applied to the measured DANTE channel signals to recover knot points
    :math:`y_i` of the cubic spline, which can then be used to reconstruct the
    inferred x-ray spectrum.
    
    Parameters
    ----------
    channels: numpy.ndarray
        Array of DANTE channel numbers.
    
    knots: numpy.ndarray
        Array of photon energies describing positions of spline knots.
    
    responseFrame: pandas.core.frame.DataFrame
        DANTE channel responses as a function of photon energy (not 
        normalized).
    
    boundary: str, optional
        Choose whether yGuess corresponds to :math:`y_0` (lowest photon 
        energy) or :math:`y_{n+1}` (highest photon energy) boundary condition.
        This should correspond to the photon energy value given in knots.
        Options are 'y0' or 'yn+1'. Default 'y0'.
        
    npts: int
        Number of points used in computing the integral. The default is 1000.
        
    Returns
    -------
    detArr: numpy.ndarray
        Matrix representing the spectrally integrated folding of the detector
        response with a cubic spline interpolation of the x-ray spectrum.
        2D array of channels and knot points of shape (n, n).
        
    Notes
    -----
    For each spline segment we have:
        
    .. math::
        M_{stuff} = \int_0^{1} (M_y(t) + 3 M_D(t) \chi_1^{-1} \chi_3) R_d(t) dt
    
    Each spline is then summed to form the full detector matrix for recovering
    the knot points.
    
    See also
    --------
    
    Examples
    --------
    """
    if boundary == "yn+1":
        print("yn+1 boundary is deprecated. Using y0 instead.")
        boundary = "y0"
    elif boundary == "y0":
        pass
    else:
        raise Exception(f"No method found for boundary {boundary}.")
        
    # number of DANTE channels where we have useful measurements
    chLen = len(channels)
    # initialize normalized energies array
    # array of normalized energies over which we do the integral
    energyNorms = np.linspace(0, 1, num=npts)
    # producing segments from knots
    segments = segmentsArr(knots)
    # calculating array for converting from values of D_i to y_i. This
    # is an optimization as this array is constant!
    dToY = dToyArr(chLen)
    # M_{y \chi} coefficients array corresponding to given normalized
    # energies. Array shape is (energyNorms, segments, knotIndex).
    yChis = yChiCoeffArrEnergies(energyNorms, chLen, dToY)

    integFoldArr = fancyTrapz2(energyNorms,
                               yChis,
                               segments,
                               responseFrame,
                               channels)
    # sum along segment axis, as each segment must contribute to the
    # overall signal.
    #detArr = np.sum(integFoldArr, axis=1)
    detArr = integFoldArr.sum(dim="segment")
    detArr.attrs['boundary'] = boundary
    return detArr


def knotSolve(signals,
              detArr,
              boundary='y0',
              yGuess=1e-10,
              npts=1000):
    r"""
    Get knot points :math:`y_i` from measured DANTE signals :math:`S_d`.
    
    Parameters
    ----------
    signals: numpy.ndarray
        numpy array of DANTE measured signal for each channel at a particular
        point in time.
        
    detArr : numpy.ndarray
        Matrix representing the spectrally integrated folding of the detector
        response with a cubic spline interpolation of the x-ray spectrum.
        2D array of channels and knot points of shape (n, n+1).

    yGuess: float, optional
        Guess for position of boundary knot point. Default is 1e-10.
    
    npts: int, optional
        Number of points used in computing the integral. Default is 1000.
        
    Returns
    -------
    knotsY : numpy.ndarray
        Array of knot point intensity values with yGuess appended.
        
    knotsYVariance: numpy.ndarray
        Array with each element corresponding to the variance of the same
        element in 'knotsY'.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    
    """ 
    #subtract boundary col from signals
    if boundary == "y0":
        # extracting column corresponding to y0
        detArrBoundaryCol = detArr.isel(knot_point=0)
        detArr = detArr.isel(knot_point=slice(1, None))
    elif boundary == "yn+1":
        # extracting column corresponding to y_{n+1}
        # detArrBoundaryCol = detArr.isel(knot_point=-1)
        # detArr = detArr.isel(knot_point=slice(None, -1))
        print("yn+1 boundary is deprecated. Using y0 instead.")
        boundary = "y0"
        detArrBoundaryCol = detArr.isel(knot_point=0)
        detArr = detArr.isel(knot_point=slice(1, None))
    else:
        raise Exception(f"No method found for boundary {boundary}.")
        
    signalsy0 = signals - yGuess * detArrBoundaryCol
    detArrInv = xr.DataArray(np.linalg.inv(detArr), 
                             dims=['channel', 'knot_point'], 
                             attrs={'boundary':boundary})  
    #huge difference between xarray.dot and np.dot. See issue in Gitlab
    knotsY = np.dot(detArrInv, signalsy0)
    knotsY = np.insert(knotsY, 0, yGuess)
    return knotsY


def reconstructSpectrum(knots,
                        knotsY,
                        npts=1000,
                        plot=False):
    r"""
    Reconstruct the inferred DANTE spectrum given the knot points
    :math:`y_i` obtained from knotSolve().
    
    Parameters
    ----------
    chLen: int
        Number of DANTE channels (equal to number of spline knots).
        
    knots: list, numpy.ndarray
        List or array of knot point photon energy value. See knotFind().
        
    knotsY: numpy.ndarray
        Array of knot point intensity values with yGuess appended.
        See knotSolve() and analyzeSpectrum().
    
    npts: int
        Number of points used in computing the integral. The default is 1000.
        
    plot: Bool
        Flag for plotting unfolded spectrum. The default is False.
        
    Returns
    -------
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
    chLen = len(knots) - 1
    # initialize normalized energies array
    energyNorms = np.linspace(0, 1, num=npts)
    # producing segments from knots
    segments = segmentsArr(knots)
    # getting cubic spline matrices for reconstructing spline
    dToY = dToyArr(chLen)
    yChis = yChiCoeffArrEnergies(energyNorms, chLen, dToY)
    #set a constant simple uncertainty for the yChis.
    #TOOD replace with something physical
    # yChisUncertainty = np.zeros(yChis.shape)
    # recovering the spectral values
    spectrum = np.dot(yChis, knotsY)
    # spectrumVariance = dotVariance(yChis,
    #                                knotsY,
    #                                yChisUncertainty,
    #                                knotsYUncertainty)
    # print(spectrumVariance)
    
    # initialize arrays for holding photon energies and corresponding
    # intensities of stitched spectrum.
    photonEnergies = np.array([])
    intensities = np.array([])
    # intensitiesVariance = np.array([])
    # loop over segments to reconstruct spectrum
    for segNum, segment in enumerate(segments):
        energyMin, energyMax = segment
        energyRegs = splineCoordsInv(energyNorms, energyMin, energyMax)
        # appending each segment to form a single array of spectral
        # values. We need to cut the last element from each array as the
        # last element overlaps the first element of the next segment.
        photonEnergies = np.append(photonEnergies, energyRegs[:-1])
        intensities = np.append(intensities, spectrum[:-1, segNum])
        # intensitiesVariance = np.append(intensitiesVariance,
        #                                 spectrumVariance[:-1, segNum])
        if plot:
            plt.plot(energyRegs, spectrum[:, segNum])
            
    if plot:
        plt.ylabel('Spectrum (GW/sr/eV)')
        plt.xlabel('Photon Energy (eV)')
        plt.title('Spectrum from cubic spline knots')
        plt.show()
    # return photonEnergies, intensities, intensitiesVariance
    return photonEnergies, intensities

def checkFidelity(signals, 
                  channels,
                  photonEnergy, 
                  intensity, 
                  responseFrame, 
                  plot=False):
    r"""
    Integrate constructed spectrum with the response functions to check if the
    recovered cubic spline is self-consistent with the input Dante signals

    Parameters
    ----------
    signals : numpy.ndarray
        Dante signals for each channel at a particular time step.
        
    channels: numpy.ndarray
        Array of DANTE channel numbers. 
        
    photonEnergies: numpy.ndarray
        Photon energy axis of unfolded spectrum.
        
    intensities: numpy.ndarray
        Spectral intensity axis of unfolded spectrum.
        
    responseFrame: pandas.core.frame.DataFrame
        DANTE channel responses as a function of photon energy (not 
        normalized).
        
    plot: Bool
        Flag for plotting the self-consistency check. The default is False.

    Returns
    -------
    fidelity : numpy.ndarray
        The recalculated voltage values from convolving the cubic spline
        solution with the Dante response functions.

    """
    from scipy import integrate
    responseEnergy = responseFrame["Energy(eV)"]
    responseOnly = responseFrame.drop("Energy(eV)", axis=1)
    # initialize fidelity to number of channel responses
    fidelity = np.zeros(responseOnly.shape[1]) 
    # keep track of channels checked for fidelity
    fidChan = responseOnly.columns.to_numpy()
    idx=0
    for idx, channel in enumerate(fidChan):
        # print(channel)
        chanResponse = responseFrame[channel]
        responseInterp = np.interp(photonEnergy, responseEnergy, chanResponse)
        convolve = intensity*responseInterp
        fidelity[idx] = integrate.simps(y=convolve, x=photonEnergy)
        idx+=1
    if plot:
        plt.plot(channels,
                 signals,
                 label = 'Measurement',
                 marker = 'o', 
                 lw = 2, 
                 ms = 12)
        plt.plot(fidChan,
                 fidelity, 
                 label = 'Spline', 
                 marker = 'o', 
                 lw = 1,
                 ms = 10)
        plt.ylabel('Signal (V)')
        plt.xlabel('Channels')
        plt.title('Cubic Spline Fidelity')
        plt.legend()
        plt.show()
    return fidelity

def checkFidelityStreak(timeCheck,
                        timesFrame, 
                        signalsFrame, 
                        channels,
                        responseFrame,
                        energies,
                        intensities):
    """
    

    Parameters
    ----------
    timeCheck : numpy.ndarray
        The times at which to check the Fiducia calculated x-ray flux against
        the measured Dante signals.
        
    timesFrame: pandas.core.frame.DataFrame
        Dataframe containing time axis corresponding to dante signals in
        df dataframe. See timesScope() and bkgCorrect().
        
    signalsFrame: pandas.core.frame.DataFrame
        Dante dataframe with background corrected values and scaled
        to units of volts. See readDanteData(), bkgCorrect() and
        voltageScale().
        
    channels: list, numpy.ndarray
        List or array of relevant DANTE channel numbers.
        
    responseFrame: pandas.core.frame.DataFrame
        DANTE channel responses as a function of photon energy (not 
        normalized).
        
    energies: numpy.ndarray
        Photon energy axis of unfolded streaked spectra.
        
    intensities: numpy.ndarray
        Spectral intensity axis of unfolded streaked spectra.

    Returns
    -------
    fidelityFrame : pandas.core.frame.DataFrame
        DataFrame containing the Dante signal values calculated from
        reconvolving the cubic spline solution with the Dante response 
        functions.

    """
    # use a for loop over select times to check fidelity of streak
    # timeCheck = np.array([1e-9, 2e-9]) #array of times to be checked
    responseOnly = responseFrame.drop("Energy(eV)", axis=1)
    fidelityFrame = pd.DataFrame(columns = timeCheck,
                                 index = responseOnly.columns)
    
    timeStep = timesFrame[1]-timesFrame[0]
    for idx, time, in enumerate(timeCheck):
        timeIdx = np.where((timesFrame <= time + timeStep/2) & 
                           (timesFrame >= time - timeStep/2))[0][0]
        getTime = timesFrame[timeIdx]
        signalStep = signalsAtTime(getTime, 
                                   timesFrame, 
                                   signalsFrame, 
                                   channels)
        energyStep = energies[:, timeIdx]
        intensityStep = intensities[:, timeIdx]
        fidelityFrame[time] = checkFidelity(signalStep,
                                            channels,
                                            energyStep,
                                            intensityStep,
                                            responseFrame,
                                            plot=True)
    return fidelityFrame

def linespline(signals,
               responseFrame, 
               channels, 
               knots,
               plot=False):
    
    # integrate channel responses to build response matrix
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
    # take the inverse of the integrated response matrix
    lineInverse = np.linalg.inv(linesplineTab)
    # take dot product and clip solution to positive values
    dotProd = np.dot(lineInverse, signals)
    linespline = np.clip(dotProd, 0, np.inf)
    # plot the result of each time step
    if plot == True:
        # plt.figure(idx)
        plt.plot(knots[1:], linespline, )
        plt.title("linear spline solution")
        plt.show()
            
    return linespline

def minFunc(yKnotVals,
            responseFrame,
            chanUsed,
            knots,
            voltages,
            y0, 
            xInterp = 'default'):
    solution = np.zeros_like(chanUsed, dtype="float64")
    yKnotValsGuess = np.insert(yKnotVals, 0, y0)
    if xInterp == "default":
        interpLen = responseFrame.shape[0]*10
        xInterp = np.linspace(min(knots), max(knots), num = interpLen)
    for idx, chan in enumerate(chanUsed):
        # print(chan, interpLen)
        pchipSol = pchip(knots, yKnotValsGuess, xInterp)
        pchipResp = pchip(responseFrame["Energy(eV)"],
                          responseFrame[chan],
                          xInterp)
        convolve = pchipResp*pchipSol
        integCon = np.trapz(convolve, xInterp)
        solution[idx] = integCon
    residual = np.sum((solution-voltages)**2)**(1/2)
    
    return residual

def knotSolvePCHIP(signals,
                   responseFrame, 
                   channels, 
                   knots,
                   initial = [],
                   xInterp = 'default',
                   plot=False):
    
    # solve linear spline first to provide seeding for Nelder-Mead simplex
    if len(initial) == 0:
        Linespline = linespline(signals=signals,
                                responseFrame=responseFrame,
                                channels=channels,
                                knots=knots,
                                plot=plot)
    else:
        Linespline = initial
    
    # initial point guess based upon initial linear spline point
    y0 = Linespline[0]*1e-1

    # list of bounds. Accept only physical soltuions (knotsY>0)
    bounds = [(0, np.inf) for _ in range(len(channels))]
    
    # run minimization using Nelder-Mead and scipy minimize function
    fiduciaSolve = minimize(minFunc,
                            Linespline,
                            args = (responseFrame,
                                    channels,
                                    knots,
                                    signals,
                                    y0,
                                    xInterp),
                            method = "Nelder-Mead",
                            bounds = bounds,
                            tol = 1e-6,
                            options = {'maxiter' : 600})
    
    # return knot values from minimization
    knotsY = np.insert(fiduciaSolve.x, 0, y0)
    
    # reconstruct the spline solution using PCHIP with identical interpolation
    if xInterp == "default":
        interpLen = responseFrame.shape[0]*10
        xInterp = np.linspace(min(knots), max(knots), num = interpLen)
    pchipSpline = pchip(knots, knotsY, xInterp)
    
    # plot the spline exact values
    if plot:
        # plt.figure(20)
        plt.plot(xInterp, pchipSpline)
        plt.scatter(knots, knotsY)
        plt.title("PCHIP solution")
        plt.show()
        
    return knotsY

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
    # interp = np.empty(0, dtype = "float64")
    for idx, knot in enumerate(xData[:-1]):
        # isolate each segment of the spline
        lowSeg = xData[idx] < xInterp
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
               xatol=1e-6, fatol=1e-1, adaptive=False, bounds = None):
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
            if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol or
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
            # interpResp = pchip_1d(energy, toIntegrate, uInterp)
            # integResponse = np.trapz(interpResp, uInterp)
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
                       initial = np.array([]),
                       xInterp = 'default'):
    
    # solve linear spline first to provide seeding for Nelder-Mead simplex
    if len(initial) == 0:
        Linespline = linespline_jit(signals,
                                    responseArray,
                                    knots)
    else:
        print("use custom seed")
        Linespline = initial
    
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
    """
    Converts responseFrame into numpy arrays to read into jitted functions

    Parameters
    ----------
    responseFrame : TYPE
        DESCRIPTION.
    channels : TYPE
        DESCRIPTION.

    Returns
    -------
    resp : TYPE
        DESCRIPTION.

    """
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
           knots,
           initial = np.array([])):    
    # generate an interpolation array for pchip
    interpLen = responseArray.shape[0]*10
    xInterp = np.linspace(min(knots), max(knots), num = interpLen)
    
    # solve for knots with perturbed responseArray
    splineVals = knotSolvePCHIP_jit(signals,
                                    responseArray,
                                    knots,
                                    initial=initial)
    # generate the spline using the pchip interpolater
    pchipSpline = pchip_1d(knots, splineVals, xInterp)
    
    # reconvolve solution with the response functions and compare to inputs
    fidelityVals = checkFidelity_jit(responseArray,
                                     xInterp,
                                     pchipSpline)
    
    # calculate the difference between the reconvolution and data                                              
    deltaVals = fidelityVals - signals

    return splineVals, fidelityVals, deltaVals

def run_MC_wrapper(responseFrame,
                   signals, 
                   channels, 
                   knots, 
                   samples, 
                   initial=np.array([])):
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
        print(idx)
        responseArray = responseMC[idx]
        splineVals[idx], fidVals[idx], deltaVals[idx] = MC_jit(responseArray,
                                                               signals,
                                                               knots,
                                                               initial=initial)
    return splineVals, fidVals, deltaVals