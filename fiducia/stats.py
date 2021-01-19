 # -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:11:12 2020

Common statistic operations

@author: Myles T. Brophy
"""
# python modules
import numpy as np
 
# listing all functions declared in this file so that sphinx-automodapi
# correctly documents them and doesn't document imported functions.
__all__ = ["simpsVariance",
           "trapzVariance",
           "gradientVariance",
           "dotVariance",
           "interpVariance",
           ]


def simpsVariance(yUnc, x=None, dx=1.0):
    r"""
    Propagates variance (:math:`\sigma^2`) through Simpson's rule
    numerical integration.
    
    NOTE: THIS FUNCTION IS INCOMPLETE AND HAS NOT BEEN VERIFIED
    2020-08-21 PMK.
    
    Parameters
    ----------
    yUnc: list, numpy.ndarray
        Uncertainties in the vertical axis.
        
    x: list, numpy.ndarray, optional
        Horizontal coordinates corresponding to yUnc. Default is None,
        which generates a uniformly spaced linear array of horizontal
        coordinates based on the length of yUnc and the value of dx.
        
    dx: float, optional
        Uniform spacing between horizontal coordinates corresponding to
        yUnc. Default is 1.0.
        
    Returns
    -------
    variance: float
        Variance (:math:`\sigma^2`) on value of integral from Simpson's
        rule numerical integration.
        
    Notes
    -----
    Based on a modified verison of:
        https://en.wikipedia.org/wiki/Simpson's_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
    
    See also
    --------
    
    Examples
    --------
    
    """
    N = len(yUnc) - 1
    yUnc = np.asarray(yUnc)
    if x is None:
        d = np.full(N, dx)
    else:
        d = np.diff(np.asarray(x))

    variance = 0.0
    #TODO validate with hand calculations
    for i in range(1, N, 2):
        hph = d[i] + d[i - 1]
        variance += yUnc[i]**2 * ( ( d[i]**3 + d[i - 1]**3
                           + 3. * d[i] * d[i - 1] * hph )\
                     / ( 6 * d[i] * d[i - 1] ) )**2
        variance += yUnc[i - 1]**2 * ( ( 2. * d[i - 1]**3 - d[i]**3
                              + 3. * d[i] * d[i - 1]**2)\
                     / ( 6 * d[i - 1] * hph) )**2
        variance += yUnc[i + 1]**2 * ( ( 2. * d[i]**3 - d[i - 1]**3
                              + 3. * d[i - 1] * d[i]**2)\
                     / ( 6 * d[i] * hph ) )**2

    if (N + 1) % 2 == 0:
        variance += yUnc[N]**2 * ( ( 2 * d[N - 1]**2
                          + 3. * d[N - 2] * d[N - 1])\
                     / ( 6 * ( d[N - 2] + d[N - 1] )) )**2
        variance += yUnc[N - 1]**2 * ( ( d[N - 1]**2
                           + 3*d[N - 1]* d[N - 2] )\
                     / ( 6 * d[N - 2] ) )**2
        variance -= yUnc[N - 2]**2 * (d[N - 1]**3\
                     / ( 6 * d[N - 2] * ( d[N - 2] + d[N - 1] )))**2
    return variance


def trapzVariance(yUnc, x=None, dx=1.0):
    r"""
    Error propogation for Trapezoidal rule integration using uniform or non-uniform grids.
        
    Parameters
    ----------
    yUnc : list, numpy.ndarray
        The list of uncertainities, referenced as :math:`\sigma_i`.
        
    x : list, numpy.ndarray, optional
        The sampling points for which the uncertainites ''y'' were found. Must be the
        same length as ''y''. If none are provided, then the step size will be uniform 
        and set with ''dx''. The default is None.
    dx : int, float, optional
        Step size. Only applies if sampling points aren't specified
        with ''x''. The default is 1.0.

    Returns
    -------
    variance : float
        The total variance (:math:`\sigma^2`) found by propagating ''y''.
        
    Notes
    -----
    Trap rule integration with non uniform spacing takes the form
    
    .. math::
        \sum_{k=1}^N \frac{\Delta x_i}{2} \left(f(x)_{i-1} + f(y)_i \right)
    
    Propogating the uncertainties through this integration results in
    
    .. math::
        \sigma^2 = \frac{1}{4} \left(\sum_{k=1}^N \Delta x_i \sigma_{i-1}^2 + \sigma_i^2 + 2\sum_{k=1}^{N-1} \Delta x_i \Delta x_i+1 \sigma_i^2 \right)
    
    The equation is generalized and applies to uniform and non-uniform step sizes.
    
    See also
    --------
    
    Examples
    --------
    """
    N = len(yUnc)
    yUnc = np.asarray(yUnc)
    if x is None:
        d = np.full(N-1, dx)
    else:
        d = np.diff(np.asarray(x))
    variance = 0.0
    #add variance of indiviual trapezoid variances
    for i in range(1, N):
        variance += d[i-1] ** 2 * 0.25 * (yUnc[i-1] ** 2 + yUnc[i] ** 2) 
    #add covariant terms because of edge overlap
    for i in range(1, N-1):
        variance += 0.5 * d[i-1] * d[i] * yUnc[i] ** 2
    return variance


def gradientVariance(yUnc, x=None, dx=1.0):
    r"""
    Propogates uncertainty for the gradient operator of an array of a given step size.

    Parameters
    ----------
    yUnc : list, numpy.ndarray
        The list of uncertainities, referenced as :math:'\sigma_i'.
        
    x : list, numpy.ndarray, optional
        The sampling points for which the uncertainites ''y'' were found. Must be the
        same length as ''y''. If none are provided, then the step size will be uniform 
        and set with ''dx''. The default is None.
        
    dx : list, numpy.ndarray, optional
        Step size. Only applies if sampling points aren't specified
        with ''x''. The default is 1.0.

    Returns
    -------
    variance : float
        The total variance (:math:`\sigma^2`) found by propagating ''y''.

    Notes
    -----
    
    .. math::
        \operatorname{Var}(\nabla y_i) = \frac{h_{i-1}^2 \sigma_{i+1}^2 + (h_i^2 + h_{i-1}^2)^2 \sigma_i^2 - h_i^4 \sigma_{i-1}^2}{(h_i h_{i-1}(h_i + h_{i-1}))^2}

    At the boundaries
    
    .. math::
        \operatorName{Var}(\nabla y_0) = \frac{\sigma_1^2 - \sigma_0^2}{h_0^2}, \operatorName{Var}(\nabla y_{N-1}) = \frac{\sigma_{N-1}^2 - \sigma_{N-2}^2}{h_{N-2}^2} 
   
    
    See also
    --------
    
    Examples
    --------
    """
    N = len(yUnc)
    np.asarray(yUnc)
    if x is None:
        h = np.full(N-1, dx)
    else:
        h = np.diff(np.asarray(x))
    
    grad = np.zeros(N)
    #square the uncertainites, cleaner to do it here than in the loop
    yUnc = yUnc**2
    grad[0] == (yUnc[1]-yUnc[0])/(h[0]**2)
    grad[N-1] == (yUnc[N-1]-yUnc[N-2])/(h[N-2]**2)
    for i in range(1, N-1):
        firstTerm = h[i-1]**4*yUnc[i+1]
        secondTerm = (h[i]**2 - h[i-1]**2)**2 * yUnc[i]
        thirdTerm = h[i]**4*yUnc[i-1]
        denom = (h[i]*h[i-1]*(h[i]+h[i-1]))**2
        
        grad[i] = (firstTerm + secondTerm - thirdTerm) / denom  

    return grad


def dotVariance(a, b, aUncertainty=None, bUncertainty=None):
    r"""
    Propogate uncertainty for the dot product of matrix a and 1D vector b.
    
    Propogate uncertainty for the dot product of a matrix and a 1D vector.
    Assumes no covariance between `a` and `b`. Methodology is similar to :func:`numpy.dot` where:
        
    - If both `a` and `b` are 1D, the uncertainty of the inner product of vectors
      is returned.
    - If 'a' is N dimensional (Where :math: `N>=2`) and `b` is 1D, the uncertainty of the sum
      product of the last axis of a with b is returned.
    
    0-D (scalar) arrays are not supported.
    `b` arrays that have more than one axis are not supported.
    `a` and `b` must have the same shape as `aUncertainty` and `bUncertainty`,
    respectively.
    
    Parameters
    ----------
    a : numpy.ndarray, list
        Matrix or vector to dot with 'b'.
  
    b : numpy.ndarray, list
        Vector that 'a' will be dotted with. Must be the same size as 
        the last axis of a.
   
    aUncertainty : numpy.ndarray, optional
        Uncertainty of each element in 'a'. The default is None.
   
    bUncertainty : numpy.ndarray, optional
        Uncertainty of each element in 'b'. The default is None.

    Returns
    -------
    variance : float
    
    Notes
    -----
    
    .. math::
        \operatorname{Var}(A \cdot B) = \sum_{i=1}^N \operatorname{Var}(a_i b_i)
    
    Assuming covariance between independent variables
    
    .. math:: 
        \sum_{i=1}^N (a_i\sigma_{b_i})^2 + (b_i\sigma_{a_i})^2
    
    See also
    --------
    
    Examples
    --------

    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    if aUncertainty is None:
        aUncertainty = np.zeros(a.shape)
    else:
        aUncertainty = np.asarray(aUncertainty)
        
    if bUncertainty is None:
        bUncertainty = np.zeros(b.shape)
    else:
        bUncertainty = np.asarray(bUncertainty)
    
    #first see if the shapes are compatible 
    #also checks that 
    try:
        if b.ndim != 1:
            raise NotImplementedError("Propogating uncertainty not implemented for b arguments that aren't 1D.")
        if b.shape != bUncertainty.shape or a.shape != aUncertainty.shape:
            raise ValueError("Array does not have the same shape as it's uncertainty array.")
        np.dot(a, b)
        np.dot(a, bUncertainty)
        np.dot(aUncertainty, b)
    except ValueError:
        raise
    else:
        #uncertainty propagation for dotting vectors
        if a.ndim == 1:
            variance = np.dot(b ** 2, aUncertainty ** 2) + np.dot(a ** 2, bUncertainty ** 2)
            return variance
        #uncertainty propagation for dotting matrix with vector
        elif a.ndim >=2:
            variance = np.zeros(a.shape[:-1])
            for i in range(a.shape[0]):
                variance[i] = dotVariance(a[i], b, aUncertainty[i], bUncertainty)
            return variance
    

def interpVariance(x, xp, fpUnc, leftVar=None, rightVar=None, period=None):
    r"""
    Propagate uncertainty for linear interpolation.
    

    Parameters
    ----------
    x : numpy.ndarray, list
        The x-coordinates at which to evaluate the interpolated values.
        
    xp : numpy.ndarray, list
        The 1D x-coordinates of the data points, must be increasing order
        
    fpUnc : numpy.ndarray, list
        The uncertainty in the y-coordinates of the data points, same length as `xp`.
        
    leftVar : float, optional
        Variance to return for `x < xp[0]`. If not given, the first `yUnc` element 
        will be used. Default is `None`
        
    rightVar : float, optional
        Variance to return for `x > xp[-1]`. If not given, the last `yUnc` element 
        will be used. Default is `None`


    Returns
    -------
    yVar : numpy.ndarray
        1D array containing the variance for each interpolated `x`.

    Notes
    -----
    Variance of interpolated point, assuming no uncertainty in `x` and `xp`,
    and no covariance between y-coordinates, is given by
    
    .. math::
        \operatorname{Var}(y) = \frac{1}{(x_1 - x_0)^2} ( (x_1 - x)^2 \sigma_{y_0}^2 + (x-x_0)^2 \sigma_{y_1}^2 )
        
    
    See also
    --------
    
    Examples
    --------
    """
    #conver to ndarray
    x = np.asarray(x)
    xp = np.asarray(xp)
    fpUnc = np.asarray(fpUnc)
    
    #make space for final answer
    yVar = np.zeros(x.shape)
    
    #find the indexs where x is between xp_i and xp_{i+1}
    indices = np.searchsorted(xp, x, side='left')
    
    for i, index in enumerate(indices):
        #if index is 0 and is not equal to the first data point
        if index == 0 and x[i] != xp[0]: 
            #set to default value  
            if leftVar:
                variance = leftVar
            else:
                variance = fpUnc[0]**2
        #if the index is after the last datapoint
        elif index >= len(xp): 
            #set to default value
            if rightVar:
                variance = rightVar
            else:
                variance = fpUnc[-1]**2
        else:
           rightIndex = index
           leftIndex = index-1
           #if we already have this value
           if xp[rightIndex] == x[i]:
               
               variance = fpUnc[rightIndex] ** 2
           elif xp[leftIndex] == x[i]:
               variance = fpUnc[leftIndex] ** 2
           else:
               #new value, actually need to do interpolation propagation
               term1 = 1 / (xp[rightIndex] - xp[leftIndex]) ** 2
               term2 = (xp[rightIndex] - x[i]) ** 2 * fpUnc[leftIndex] ** 2 + (x[i] - xp[leftIndex]) ** 2 *fpUnc[rightIndex] ** 2 
               variance = term1 * term2

        yVar[i] = variance
        
    return yVar
        
        
    