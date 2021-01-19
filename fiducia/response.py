#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:23:02 2019

Utilities for working with DANTE response functions (e.g. plotting, locating
edges).

@author: Pawel M. Kozlowski
"""

# python modules
import numpy as np


# listing all functions declared in this file so that sphinx-automodapi
# correctly documents them and doesn't document imported functions.
__all__ = ["knotFind"]


def knotFind(channels,
             responseFrame,
             forceKnot=np.array([]),
             knotBoundary=0,
             boundary='y0'):
    r"""Find knot points.
    
    Find knot points for cubic splines based on positions of K-edges
    of each DANTE channel filter.
    
    Parameters
    ----------
    channels: list, numpy.ndarray
        List or array of relevant channels
    
    responseFrame: pandas.core.frame.DataFrame
        Pandas dataFrame containing response functions for each DANTE
        channel. See loadResponses().
        
    forceKnot: numpy.ndarray
        Numpy array where first column is channelNumber and second column
        is the corresponding photonEnergy we wish to force. Use this for
        channels that do not have a distinct K-edge.
        
    knotBoundary: float
        Photon energy value for either y_0 or y_{n+1} boundary condition.
        This value gets appended to the array of photon energies otherwise
        found by knotFind().
        
    Returns 
    -------
    knotsAppend: numpy.ndarray
        An array of knot points, with each element corresponding to a channel
        or boundary condition.
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    knots = np.zeros(len(channels))
    for idx, channel in enumerate(channels):
        # if forceKnot isn't an empty list, then we go about forcing the user
        # provided values.
        if forceKnot.size != 0:
            if channel in forceKnot[:,0]:
                # if the channel is forced by the user then put the user provided
                # photon energy into the knots array
                forceIdx = np.where(forceKnot[:,0] == channel)
                knots[idx] = forceKnot[forceIdx, 1]
            else:
                # finding largest negative gradient, which should correpond to the
                # K-edge of the DANTE channel filter.
                grad = -np.gradient(responseFrame[channel])
                maxIndex = np.argmax(grad)
                knots[idx] = responseFrame['Energy(eV)'][maxIndex]
        else:
            # finding largest negative gradient, which should correpond to the
            # K-edge of the DANTE channel filter.
            grad = -np.gradient(responseFrame[channel])
            maxIndex = np.argmax(grad)
            knots[idx] = responseFrame['Energy(eV)'][maxIndex]
        
    if boundary == 'y0':
        knotsAppend = np.append([knotBoundary], knots)
    elif boundary == 'yn+1':
        knotsAppend = np.append(knots, [knotBoundary])
    else:
        raise Exception(f"No method found for boundary {boundary}.")
    return knotsAppend