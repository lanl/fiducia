#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 12:22:22 2019

Tests for main.py

Tests should be split and marked as:
    -type check tests, which check function execution and error handling
    against inputs of different types.
    - numerical tests, which check whether given input values produce
    expected output values.
    - property tests, which check mathematical properties of functions,
    such as associativity, commutativity, identity, etc.
    - physical tests, which check physical intution, such as higher power
    should produce larger radiation temperature, or larger uncertainties
    on the inputs should produce larger uncertainties on the outputs.


@author: Pawel M. Kozlowski, Myles T. Brophy
"""

# python modules
import numpy as np
import pytest
from hypothesis import given
import hypothesis.strategies as s
import hypothesis.extra.numpy as hnp
import hypothesis.extra.pandas as hpd

# custom modules
import fiducia.main as main


# categorize this test as part of the group of tests which check
# expected values of functions.
@pytest.mark.value
# parametrize the test to run multiple times over different inputs.
@pytest.mark.parametrize("power,area,angle,expectedTRad",
                         [(0, 0.6, 30, 0),
                          (1, 0.6, 30, 36.98703164858843),
                          (10, 0.6, 30, 65.7732768191428),
                          (100, 0.6, 30, 116.963263898272)])
def test_inferRadTemp_val(power, area, angle, expectedTRad):
    """Testing for expected return value."""
    methodVals, methodValsVariance = main.inferRadTemp(power, 
                                                       area,
                                                       angle)
    testTrue = np.allclose(methodVals,
                           expectedTRad, 
                           rtol=1e-7,
                           atol=0.0)
    errStr = (f"Infer radiation temperture test gives {methodVals} but "
              f"should be {expectedTRad}")
    assert testTrue, errStr
        

@given(power=s.floats(min_value=0),
       powerUncertainty=s.floats(min_value=0))
def test_inferRadTemp_prop(power, powerUncertainty):
    """
    Property-based testing of inferRadTemp() using hypothesis.
    Given a physical value of power (>=0),
    expect radiation temperature to be phyiscal (>=0).
    """
    area = 0.6
    angle = 30
    methodVals, methodValsVariance = main.inferRadTemp(power=power, 
                                                       area=area,
                                                       angle=angle,
                                                       powerUncertainty=powerUncertainty)
    assert methodVals >= 0 and methodValsVariance >= 0