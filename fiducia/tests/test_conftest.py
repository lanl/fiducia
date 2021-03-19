#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 19:17:24 2020

A couple of tests to ensure that fixtures from conftest.py are loading
correctly for the pytest sessions, and that the returned data are
behaving as expected.

@author: Pawel M. Kozlowski
"""

import pytest

def test_info_fixture(example_info):
    """Testing whether conftest.py fixture is working as expected."""
    assert example_info["Angle"] == 69.09
    

# def test_data_fixture(load_example_data):
#     """Testing whether conftest.py fixture is working as expected."""
#     expectedChOn = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]
#     testTrue = load_example_data["On Channels"] == expectedChOn
#     errStr = (f"Header loaded incorrectly. Expected on channels are "
#               f"{expectedChOn}, but got {load_example_data['On Channels']} "
#               "instead.")
#     assert testTrue, errStr