# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:16:31 2020

Testing for stats.py

@author: Myles T. Brophy
"""

# python modules
import numpy as np
import pytest

# custom modules
import fiducia.stats as stats

class Test_integrationError(object):
    def setup_class(self):
        """Initialize parameters for tests."""
        self.rectangle = np.full(2,5)
        self.rectangleHeightUncertainty = np.full(2, 1) 
        
        self.diagonal = np.arange(5)*3 + 1
    
    def test_trapz(self):
        #check that the trap integration of a rectangle equals width*height
        assert np.trapz(self.rectangle) == (len(self.rectangle) -1)*self.rectangle[0]
        #check that the trap integration of a diagonal equals the area of a right trapezoid.
        assert np.trapz(self.diagonal) == 0.5 * (len(self.diagonal)-1) * (self.diagonal[0] + self.diagonal[-1])
    
    def test_trapzUncertainty(self):
        """Testing for expected return value."""
        assert stats.trapzVariance([1,2,3]) == 6.5
        assert stats.trapzVariance([1,2,3], x=[5,7,10]) == 46.25
        assert stats.trapzVariance([1,2], x=[1,3]) == 5.0 
        assert stats.trapzVariance([1], x=[2]) == 0.0
        #assert misc.trapzVariance(self.rectangleHeightUncertainty) == self.rectangleHeightUncertainty[0]*(len(self.rectangleHeightUncertainty)-1)
    
    def test_interpVariance(self):
        """Testing for expected return value."""
        #checked by hand
        assert np.array_equal(stats.interpVariance([1,4,6], [1,2,3,4,5,6,7,8], [2,4,3,5,2,5,1,2]),[4.0, 25.0,25.0])
        assert np.array_equal(stats.interpVariance([0,4,4.5,6,9], [1,2,3,4,5,6,7,8], [2,4,3,5,2,5,1,2]),[4.0, 25.0, 7.25,25.0,4.0])