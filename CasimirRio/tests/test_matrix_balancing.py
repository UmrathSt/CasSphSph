"""Tests for matrix_balancing

"""
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import unittest
from matrix_balancing import balance


class matrix_balancing(unittest.TestCase):

    def test_det_invariance(self):
        """test wether the balancing leaves the 
           determinant invariant

        """
        M = np.random.rand(100,100)
        M[10:15, 40] = np.exp(np.arange(5))
        M[90, 2:7] = np.exp(-np.arange(5))
        expected_detM = np.linalg.det(M)
        scaledM = balance(M)
        result_detM = np.linalg.det(scaledM)
        
        assert_array_almost_equal(expected_detM/result_detM, 1) 
