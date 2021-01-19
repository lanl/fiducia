#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 02:37:12 2017

Default plotting parameters

@author: Pawel M. Kozlowski
"""

# importing Python modules
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from itertools import cycle


# listing all functions declared in this file so that sphinx-automodapi
# correctly documents them and doesn't document imported functions.
__all__ = ["plot_line_shaded",
           "plot_scatter_bars"]


# setting default plot properties
plotFont = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 16}
plt.rc('font', **plotFont)
plt.rc('lines', linewidth=2)
plt.rc('lines', markersize=8)
#plt.rc('text', usetex=True)
#plt.rc('text', usetex=False)
# setting default figure size
matplotlib.rcParams['figure.figsize'] = 10, 6

validMathTextFonts = ['dejavusans',
                      'dejavuserif',
                      'cm',
                      'stix',
                      'stixsans',
                      'custom']

# setting latex rendered fonts to be same as regular fonts
try:
    matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
    matplotlib.rcParams['mathtext.rm'] = 'DejaVu Serif'
    matplotlib.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
    matplotlib.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'
except:
    print("Couldn't load dejavuserif fonts for plot defaults."
          "Falling back to stix fonts.")
    try:
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'
    except:
        print("Couldn't load stix fonts for plot defaults.")

# for testing whether matplotlib and python fonts match
#plt.title(r'cm$\rm cm^{-3}$')


#%% convenience functions for plotting

# shaded error bars for line plot
def plot_line_shaded(xData, yData, yErrsPos, yErrsNeg=[], label="", **kwargs):
    """
    Generate a line plot with shaded region representing y-error bars.
    Can be run multiple times before plt.show(), to plot multiple data
    sets on the same axes.
    
    Parameters
    ----------
    xData: numpy.ndarray
        X-axis data to be plotted.
        
    yData: numpy.ndarray
        Y-axis data to be plotted.
        
    yErrsPos: numpy.ndarray
        Errors on yData.
        
    yErrsPos: numpy.ndarray
        When errors on yData are asymmetric, these are the positive
        side errors.
        
    yErrsNeg: numpy.ndarray
        When errors on yData are asymmetric, these are the negative
        side errors.
    
    Returns
    -------
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    #make errors symmetric unless specified otherwise
    if yErrsNeg == []:
        yErrsNeg = yErrsPos
    # check that arrays are 1D
    
    # check that arrays are equal length
    if not len(xData) == len(yData) == len(yErrsPos) ==len(yErrsNeg):
        raise ValueError("Arrays must of equal length!")

    # line plot
    ax = kwargs.pop('ax', plt.gca())
    base_line, = ax.plot(xData, yData, label=label, **kwargs)
    # shaded error region
    plt.fill_between(xData,
                     yData - yErrsNeg,
                     yData + yErrsPos,
                     alpha=0.5,
                     edgecolor=base_line.get_color(),
                     facecolor=base_line.get_color())
    return


def plot_scatter_bars(xData, yData, yErrsPos, yErrsNeg=[], label="", **kwargs):
    """
    Generate a scatter plot with y-error bars.
    Can be run multiple times before plt.show(), to plot multiple data
    sets on the same axes.
    
    Parameters
    ----------
    xData: numpy.ndarray
        X-axis data to be plotted.
        
    yData: numpy.ndarray
        Y-axis data to be plotted.
        
    yErrsPos: numpy.ndarray
        Errors on yData.
        
    yErrsPos: numpy.ndarray
        When errors on yData are asymmetric, these are the positive
        side errors.
        
    yErrsNeg: numpy.ndarray
        When errors on yData are asymmetric, these are the negative
        side errors.
    
    Returns
    -------
        
    Notes
    -----
    
    See also
    --------
    
    Examples
    --------
    """
    #make errors symmetric unless specified otherwise
    if yErrsNeg == []:
        yErrsNeg = yErrsPos
    # check that arrays are 1D
    
    # check that arrays are equal length
    if not len(xData) == len(yData) == len(yErrsPos) == len(yErrsNeg):
        raise ValueError("Arrays must of equal length!")
        
    # line plot
    ax = kwargs.pop('ax', plt.gca())
    ax.errorbar(xData,
                yData,
                yerr=[yErrsNeg, yErrsPos],
                label=label,
                fmt='o',
                fillstyle='none',
                capsize=4,
                elinewidth=1,
                **kwargs)
    return

#%% testing custom plotting tools

#xData = np.arange(10)
#yData = xData ** 2
#yData2 = xData ** 3
#yErrs = np.ones_like(xData) * 30
#plot_line_shaded(xData, yData, yErrs, label="#1", color='r')
#plot_line_shaded(xData, yData2, yErrs, label="#2")
#plt.ylabel("y")
#plt.xlabel("x")
#plt.legend(loc='upper left', frameon=False, labelspacing=0.001,
#           fontsize=14, borderaxespad=0.4)
#plt.show()
#
#
#xData = np.arange(10)
#yData = xData ** 2
#yData2 = xData ** 3
#yErrs = np.ones_like(xData) * 30
#plot_scatter_bars(xData, yData, yErrs, label="#1", color='r')
#plot_scatter_bars(xData, yData2, yErrs, label="#2")
#plt.ylabel("y")
#plt.xlabel("x")
#plt.legend(loc='upper left', frameon=False, labelspacing=0.001,
#           fontsize=14, borderaxespad=0.4)
#plt.show()

#%% style cycling
# printing default colors
default_colors = matplotlib.colors.cnames.keys()
#print(f"Colors: {default_colors}")
# printing default linestyles
default_lines = matplotlib.lines.lineStyles.keys()
#print(f"Lines: {default_lines}")
# printing default marker styles
default_markers = matplotlib.markers.MarkerStyle.markers.keys()
#print(f"Markers: {default_markers}")

# custom list of linestyles (excluding blank line styles)
lines = ["-","--","-.",":"]
linecycler = cycle(lines)
#plt.figure()
#for i in range(10):
#    x = range(i,i+10)
#    plt.plot(range(10),x,next(linecycler))
#plt.show()
    

#%% Custom colors based on https://towardsdatascience.com/making-matplotlib-beautiful-by-default-d0d41e3534fd
#CB91_Blue = '#2CBDFE'
#CB91_Green = '#47DBCD'
#CB91_Pink = '#F3A0F2'
#CB91_Purple = '#9D2EC5'
#CB91_Violet = '#661D98'
#CB91_Amber = '#F5B14C'
#
#color_list = [CB91_Blue,
#              CB91_Pink,
#              CB91_Green,
#              CB91_Amber,
#              CB91_Purple,
#              CB91_Violet]
#plt.rcParams['axes.prop_cycle'] = plt.cycler(color="seaborn-colorblind")


#%% built-in custom color cycles
#print(plt.style.available)
#plt.style.use('seaborn')