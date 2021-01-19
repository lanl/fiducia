#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 18:10:57 2020

Place for setting up fixtures to be used across pytests.
Loading example dante datasets should be done here so that they
don't have to be repeated across the various test modules.

@author: Pawel M. Kozlowski
"""

import pytest
import numpy as np


@pytest.fixture(scope="session")
def example_info(tmpdir_factory):
    r"""
    Loads example information on Dante data (calibration files, and raw data)
    from Omega-60, which can be used across multiple pytests.
    """
    # response function curves for Dante channels.
    responseFile = 'example_response.csv'
    # attenuator information loopkup by ID number.
    attenuatorsFile = 'TableAttenuators.xls'
    # Dante oscilloscope voltage axis offsets.
    offsetsFile = 'Offset.xls'
    # Raw Dante data file from Omega-60.
    dataFile = 'dante_dante86455.dat'
    # emitting area of blackbody in mm^2.
    area = np.pi * 0.6 ** 2
    # dante viewing angle relative to normal of emitting surface, in degrees.
    angle = 69.09
    # collecting info into a dict which can be passed to various pytests.
    info = {"Response File" : responseFile,
            "Attenuators File" : attenuatorsFile,
            "Offsets File" : offsetsFile,
            "Data File" : dataFile,
            "Area" : area,
            "Angle" : angle}
    return info



@pytest.fixture(scope="session")
def load_example_data(example_info):
    r"""
    Loads the example Dante data from example_info() so that it only
    has to be done once across the session.
    """
    # load raw dante data into corrected/aligned format
    from fiducia.rawProcess import loadCorrected
    loaded = loadCorrected(danteFile=example_info["Data File"],
                           attenuatorsFile=example_info["Attenuators File"],
                           offsetsFile=example_info["Offsets File"],
                           plot=False)
    timesFrame, dfAtten, onChList, hf, dfVolt = loaded
    dataDict = {"Times Frame" : timesFrame,
                "Attenuators Frame" : dfAtten,
                "On Channels" : onChList,
                "Header Frame" : hf,
                "Signal Frame" : dfVolt}
    return dataDict