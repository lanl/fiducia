#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:42:24 2019

Tests for cspline.py

@author: Pawel M. Kozlowski, Myles T. Brophy
"""

# python modules
import numpy as np
import pytest
import os
import inspect

# custom modules
import fiducia
import fiducia.cspline as cspline
import fiducia.loader as loader
import scipy.sparse as sparse


class Test_splineCoords(object):
    """
    Testing class for the splineCoords and splineCorrdsInv functions
    """
    def setup_class(self):
        """Initialize parameters for tests."""
        self.energyMin = 10
        self.energyMax = 20
        self.values1 = np.arange(9) + 1
        self.energy1 = self.values1 + self.energyMin
        self.known1 = self.values1 / 10
            
    def test_splineCoords(self):
        """
        Testing for expected return value.
        """
        methodVals = cspline.splineCoords(self.energy1, self.energyMin, self.energyMax)
        testTrue = np.allclose(methodVals, self.known1, rtol=1e-16, atol=0.0)
        errStr = (f"Spline coordinate value test gives {methodVals} but "
                  f"should be {self.known1}")
        assert testTrue, errStr
        
    def test_splineCoordsInv(self):
        """Testing for expected return value."""
        methodVals = cspline.splineCoordsInv(self.known1, self.energyMin, self.energyMax)
        testTrue = np.allclose(methodVals, self.energy1,
                               rtol=1e-16, atol=0.0)
        errStr = (f"Spline coordinate inverted value test gives {methodVals} but "
                  f"should be {self.energy1}")
        assert testTrue, errStr


class Test_coeffArr(object):
    """
    Testing class for the yCoeffArr and dCoeffArr functions
    """
    def setup_class(self):
        """Initialize parameters for tests."""
        self.energyNorm = 0.5;
        self.chLen = 9
        self.trueYCoeffArr = sparse.csc_matrix([
            [0.875,0.125,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.5,0.5,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.5,0.5,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.5,0.5,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.5,0.5,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.5,0.5,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.5,0.5,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.5,0.5,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.875,0.125]])
        self.trueDCoeffArr = sparse.csc_matrix([
            [0.375,-0.125,0.,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.125,-0.125,0.,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.125,-0.125,0.,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.125,-0.125,0.,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.125,-0.125,0.,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.125,-0.125,0.,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.125,-0.125,0.,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.125,-0.125,0.],
            [0.,0.,0.,0.,0.,0.,0.,0.,0.25,0.125]])
        
        
    def test_yCoeffArr(self):
        """Testing for expected value"""
        methodVals = cspline.yCoeffArr(self.energyNorm, self.chLen)
        testTrue = np.allclose(methodVals.toarray(), 
                               self.trueYCoeffArr.toarray(),
                               rtol=1e-16, atol=0.0)
        errStr = (f"Y coefficient array test gives {methodVals.toarray()} but "
                  f"should be {self.trueYCoeffArr.toarray()}")
        assert testTrue, errStr
        
    def test_dCoeffArr(self):
        """Testing for expected value"""
        methodVals = cspline.dCoeffArr(self.energyNorm, self.chLen)
        testTrue = np.allclose(methodVals.toarray(),
                               self.trueDCoeffArr.toarray(),
                               rtol=1e-16, atol=0.0)
        errStr = (f"D coefficient array test gives {methodVals.toarray()} but "
                  f"should be {self.trueDCoeffArr.toarray()}")
        assert testTrue, errStr


class Test_chArr(object):
    """Testing class for the __chi1Arr__ and __chi3Arr__ functions"""
    def setup_class(self):
        """Initialize parameters for test"""
        self.chLen = 6
        self.trueChi1 = sparse.csc_matrix([
            [2.,1.,0.,0.,0.,0.,0.],
            [1.,4.,1.,0.,0.,0.,0.],
            [0.,1.,4.,1.,0.,0.,0.],
            [0.,0.,1.,4.,1.,0.,0.],
            [0.,0.,0.,1.,4.,1.,0.],
            [0.,0.,0.,0.,1.,4.,1.],
            [0.,0.,0.,0.,0.,1.,2.]])
        self.trueChi3 = sparse.csc_matrix([
            [-1.,1.,0.,0.,0.,0.,0.],
            [-1.,0.,1.,0.,0.,0.,0.],
            [0.,-1.,0.,1.,0.,0.,0.],
            [0.,0.,-1.,0.,1.,0.,0.],
            [0.,0.,0.,-1.,0.,1.,0.],
            [0.,0.,0.,0.,-1.,0.,1.],
            [0.,0.,0.,0.,0.,-1.,1.]])
    
    def test___chi1Arr__(self):
        """Testing for expected return value."""
        methodVals = cspline.__chi1Arr__(self.chLen)
        testTrue = np.allclose(methodVals.toarray(), self.trueChi1.toarray(),
                               rtol=1e-16, atol=0.0)
        errStr = (f"Chi1 array test gives {methodVals.toarray()} but "
                  f"should be {self.trueChi1.toarray()}")
        assert testTrue, errStr
        
    def test___chi3Arr__(self):
        """Testing for expected return value."""
        methodVals = cspline.__chi3Arr__(self.chLen)
        testTrue = np.allclose(methodVals.toarray(), self.trueChi3.toarray(),
                               rtol=1e-16, atol=0.0)
        errStr = (f"Chi3 array test gives {methodVals.toarray()} but "
                  f"should be {self.trueChi3.toarray()}")
        assert testTrue, errStr


class Test_dToyArr(object):
    """Testing class for the dToyArr function"""
    def setup_class(self):
        self.chLen = 4
        self.trueDToyArr = [[-1.26785714,  1.60714286, -0.42857143,  0.10714286, -0.01785714],
       [-0.46428571, -0.21428571,  0.85714286, -0.21428571,  0.03571429],
       [ 0.125     , -0.75      ,  0.        ,  0.75      , -0.125     ],
       [-0.03571429,  0.21428571, -0.85714286,  0.21428571,  0.46428571],
       [ 0.01785714, -0.10714286,  0.42857143, -1.60714286,  1.26785714]]
        
    def test_dToyArr(self):
        """Testing for expected return value."""
        methodVals = cspline.dToyArr(self.chLen)
        testTrue = np.allclose(methodVals, self.trueDToyArr, 
                               rtol=1e-6, atol=0.0)
        errStr = (f"D to Y test gives {methodVals} but "
                  f"should be {self.trueDToyArr}")
        assert testTrue, errStr
        
        
# class Test_responseInterp(object):
#     """
#     Testing class for the responseInterp function.
#     """
#     def setup_class(self):
#         "Initialize parameters for testing"
#         self.energyNorm = 0.2
#         self.energyMin = 10
#         self.energyMax = 80
#         #Have to set path this way in order to work locally on pytest and
#         #on the GitLab pipeline. Not pretty.
#         self.fiduciaPath = os.path.abspath(os.path.dirname(inspect.getfile(fiducia)))
#         self.responseFilePath = os.path.join(self.fiduciaPath, "tests", "example_response.csv")
        
#         self.channels= [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14]
#         self.responseFrame = loader.loadResponses(self.channels, self.responseFilePath)
#         self.trueResponsesInterpolated = [[8.19720e-39,0.00000e+00,0.00000e+00,0.00000e+00,2.82800e-38,
#                                            7.03465e-25,2.07858e-09,6.20039e-08,0.00000e+00,0.00000e+00,
#                                            0.00000e+00]]
        
#     def test_responseInterp(self):
#         """Testing for expected return value."""
#         methodVals = cspline.responseInterp(self.energyNorm,
#                                             self.energyMin, 
#                                             self.energyMax,
#                                             self.responseFrame,
#                                             self.channels)
#         testTrue = np.allclose(methodVals, self.trueResponsesInterpolated, 
#                                rtol=1e-13, atol=0.0)
#         errStr = (f"Response interpolation test gives {methodVals} but "
#                   f"should be {self.trueResponsesInterpolated}")
#         assert testTrue, errStr
    
    
class Test_yChiCoeff(object):
    """
    Testing class for the yChiCoeffArr and yChiCoeffArrEnergies functions
    """
    def setup_class(self):
        "Initialize parameters for testing"
        self.chLen = 4
        self.energyNorm = 0.2
        self.energyNorms = [0.2,0.3,0.5, 0.9]
        self.dToYArr = [[-1.26785714,1.60714286,-0.42857143,0.10714286,-0.01785714],
                        [-0.46428571,-0.21428571,0.85714286,-0.21428571,0.03571429],
                        [0.125,-0.75,0.,0.75,-0.125],
                        [-0.03571429,0.21428571,-0.85714286,0.21428571,0.46428571],
                        [0.01785714,-0.10714286,0.42857143,-1.60714286,1.26785714]]
    
        self.trueyChiCoeffArr = [[0.76342857,0.32342857,-0.10971429,0.02742857,-0.00457143],
                                 [-0.06342857,0.89257143,0.21371429,-0.05142857,0.00857143],
                                 [0.01714286,-0.10285714,0.92342857,0.19314286,-0.03085714],
                                 [-0.00514286,0.03085714,-0.12342857,0.97485714,0.12285714]]
        self.trueyChiCoeffArrEnergies = [[[0.76342857,0.32342857,-0.10971429,0.02742857,
                                           -0.00457143],
                                          [-0.06342857,0.89257143,0.21371429,-0.05142857,
                                           0.00857143],
                                          [0.01714286,-0.10285714,0.92342857,0.19314286,
                                           -0.03085714],
                                          [-0.00514286,0.03085714,-0.12342857,0.97485714,
                                           0.12285714]],

                                         [[0.656125,0.47925,-0.171,0.04275,
                                           -0.007125],
                                          [-0.076125,0.79975,0.342,-0.07875,
                                           0.013125],
                                          [0.020625,-0.12375,0.838,0.31275,
                                           -0.047625],
                                          [-0.006375,0.03825,-0.153,0.91675,
                                           0.204375]],

                                         [[0.45758929,0.75446429,-0.26785714,0.06696429,
                                           -0.01116071],
                                          [-0.07366071,0.56696429,0.60714286,-0.12053571,
                                           0.02008929],
                                          [0.02008929,-0.12053571,0.60714286,0.56696429,
                                           -0.07366071],
                                          [-0.00669643,0.04017857,-0.16071429,0.72767857,
                                           0.39955357]],

                                        [[0.09180357,1.02117857,-0.14271429,0.03567857,
                                          -0.00594643],
                                         [-0.01430357,0.08682143,0.97971429,-0.06267857,
                                          0.01044643],
                                         [0.00401786,-0.02410714,0.09742857,0.96139286,
                                          -0.03873214],
                                         [-0.00176786,0.01060714,-0.04242857,0.16010714,
                                          0.87348214]]]
        
    def test_yChiCoeffArr(self):
        """Testing for expected return value."""
        methodVals = cspline.yChiCoeffArr(self.energyNorm, self.chLen, self.dToYArr)
        testTrue = np.allclose(methodVals, self.trueyChiCoeffArr, 
                               rtol=1e-6, atol=0.0)
        errStr = (f"y Chi Coefficient Array test gives {methodVals} but "
                  f"should be {self.trueyChiCoeffArr}")
        assert testTrue, errStr
    
    def test_yChiCoeffArrEnergies(self):
        """Testing for expected return value."""
        methodVals = cspline.yChiCoeffArrEnergies(self.energyNorms, self.chLen, self.dToYArr)
        testTrue = np.allclose(methodVals, self.trueyChiCoeffArrEnergies, 
                               rtol=1e-5, atol=0.0)
        errStr = (f"y Chi Coefficient Energies Array test gives {methodVals} but "
                  f"should be {self.trueyChiCoeffArrEnergies}")
        assert testTrue, errStr
        
        
        
        