"""
Created on Fri Mar  8 09:41:36 2019

Functions for working with cubic spline equation in matrix form.

@author: Pawel M. Kozlowski
"""

# python modules
import numpy as np
import scipy.sparse as sparse
from scipy import integrate
import matplotlib.pyplot as plt
import xarray as xr
# custom modules
import fiducia.pltDefaults
from fiducia.stats import dotVariance

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
    #First and last row elements in y array are different (added by DHB 3/25/19)
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
    #First and last row elements in D array are different (added by DHB 3/25/19)
    dArr[0,0] = energyNorm - energyNorm ** 3
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
        DANTE channel responses as a function of photon energy (not normalized).

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
#            print(f"Calculating for segment {segments[segmentNum]}.")
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
        Number of points used in computing the integral
        
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
    if boundary == "y0":
        # extracting column corresponding to y0
        detArrBoundaryCol = detArr.isel(knot_point=0)
        detArr = detArr.isel(knot_point=slice(1, None))
    elif boundary == "yn+1":
        # extracting column corresponding to y_{n+1}
        detArrBoundaryCol = detArr.isel(knot_point=-1)
        detArr = detArr.isel(knot_point=slice(None, -1))
    else:
        raise Exception(f"No method found for boundary {boundary}.")
    return detArr, detArrBoundaryCol


def knotSolve(signals,
              detArr,
              detArrBoundaryCol,
              detArrVarianceBoundaryCol,
              detArrInv,
              stdDetArrInv,
              signalsUncertainty=None,
              yGuess=1e-77,
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
    
    stdDetArrInv: xarray.DataArray
        Array of the standard deviation of each element in detArrInv based on
        variance using the 'responseUncertaintyFrame' propagated with Monte Carlo. 

    signalsUncertainty: xarray.DataArray, optional
        numpy array of the uncertainty of the DANTE measured signal for each 
        channel at a particular point in time. The default is None.

    yGuess: float, optional
        Guess for position of boundary knot point. Default is 1e-77.
    
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
    signalsy0 = signals - yGuess * detArrBoundaryCol
    signalsy0Variance = signalsUncertainty**2  + yGuess**2 * detArrVarianceBoundaryCol
    signalsy0Uncertainty = np.sqrt(signalsy0Variance)
    # applying inverted array to signals to recover knot points
    #huge difference between xarray.dot and np.dot. See issue in Gitlab
    knotsY = np.dot(detArrInv, signalsy0)
    #knotsYother = detArrInv.dot(signalsy0, {subscripts:'i,i'})
    #print("knot diff", knotsYother-knotsY)
    knotsYVariance = dotVariance(detArrInv, signalsy0, stdDetArrInv, signalsy0Uncertainty)
    return knotsY, knotsYVariance


def reconstructSpectrum(chLen,
                        knots,
                        knotsY,
                        knotsYUncertainty=None,
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

    knotsYUncertainty: numpy.ndarray
        Array of knot point intensity uncertainty values with yGuess appended.
        See knotSolve() and analyzeSpectrum(). The default is None.
    
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
    #check for no uncertainty provided.
    if knotsYUncertainty is None:
        #give them uncertainty of 0
        knotsYUncertainty = np.zeros(knotsY.shape)
    # initialize normalized energies array
    energyNorms = np.linspace(0, 1, num=npts)
    # producing segments from knots
    segments = segmentsArr(knots)
    # getting cubic spline matrices for reconstructing spline
    dToY = dToyArr(chLen)
    yChis = yChiCoeffArrEnergies(energyNorms, chLen, dToY)
    #set a constant simple uncertainty for the yChis.
    #TOOD replace with something physical
    yChisUncertainty = np.zeros(yChis.shape)
    # recovering the spectral values
    spectrum = np.dot(yChis, knotsY)
    spectrumVariance = dotVariance(yChis,
                                   knotsY,
                                   yChisUncertainty,
                                   knotsYUncertainty)
    # print(spectrumVariance)
    
    # initialize arrays for holding photon energies and corresponding
    # intensities of stitched spectrum.
    photonEnergies = np.array([])
    intensities = np.array([])
    intensitiesVariance = np.array([])
    # loop over segments to reconstruct spectrum
    for segNum, segment in enumerate(segments):
        energyMin, energyMax = segment
        energyRegs = splineCoordsInv(energyNorms, energyMin, energyMax)
        # appending each segment to form a single array of spectral
        # values. We need to cut the last element from each array as the
        # last element overlaps the first element of the next segment.
        photonEnergies = np.append(photonEnergies, energyRegs[:-1])
        intensities = np.append(intensities, spectrum[:-1, segNum])
        intensitiesVariance = np.append(intensitiesVariance,
                                        spectrumVariance[:-1, segNum])
        if plot:
            plt.plot(energyRegs, spectrum[:, segNum])
            
    if plot:
        plt.ylabel('Spectrum (GW/sr/eV)')
        plt.xlabel('Photon Energy (eV)')
        plt.title('Spectrum from cubic spline knots')
        plt.show()
    return photonEnergies, intensities, intensitiesVariance