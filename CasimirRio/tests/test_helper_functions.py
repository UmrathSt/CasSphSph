""" tests for helper functions

"""
import numpy as np
from scipy.special import gammaln
from math import factorial
import unittest
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
from helper_functions import ln_2dbl, lfac
from math import log, factorial


class mie_helper_functions(unittest.TestCase):

    def test_ln_double_factorial(self):
        """ test the natural logarithm of (2*l-1)!!
    
        """
        L = 10
        result = ln_2dbl(L)
        l = np.array([19, 17, 15, 13, 11, 9,
                      7, 5, 3, 1])
        expected = log(l.prod())
        assert_almost_equal(result,expected)
    
    def test_log_factorial(self):
        """ test the logarithm of a factorial
        
        """
        l = np.arange(4)
        result = lfac(l)
        expected = np.array([log(factorial(l)) for l in l])
        assert_array_almost_equal(result, expected)
