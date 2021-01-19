#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:25:23 2019

Tests for misc.py

@author: Pawel M. Kozlowski, Myles T. Brophy
"""

# python modules
import numpy as np
import pytest

# custom modules
import fiducia.misc as misc


def test_find_nearest():
    """
    Testing for expected return value.
    """
    ex = [1,10,100,500,1000];
    assert misc.find_nearest(ex, 40) == (1,10)
    assert misc.find_nearest(ex, 5000) == (4,1000)
