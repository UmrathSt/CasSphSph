"""Tests for thermodynamics

"""
import numpy as np
from numpy.testing import assert_array_equal
import unittest
import thermodynamics


class ThermodynamicsHelperFunctions(unittest.TestCase):

      def test_slicewise_eval_0_0(self):
          """test slicewise_eval for nmin=nmax=0

          """
          result = thermodynamics.slicewise_eval(lambda x: x+1, 0, 0, 10)
          expected = np.array([1])
          assert_array_equal(result, expected)

      def test_slicewise_eval_5_5(self):
          """test slicewise_eval for nmin=nmax=0

          """
          result = thermodynamics.slicewise_eval(lambda x: x+1, 5, 5, 10)
          expected = np.array([6])
          assert_array_equal(result, expected)

      def test_slicewise_eval_0_5(self):
          """test slicewise_eval for nmin=nmax=0

          """
          result = thermodynamics.slicewise_eval(lambda x: x+1, 0, 5, 10)
          expected = np.arange(1, 7)
          assert_array_equal(result, expected)

      def test_slicewise_eval_1_6(self):
          """test slicewise_eval for nmin=nmax=0

          """
          result = thermodynamics.slicewise_eval(lambda x: x+1, 1, 6, 10)
          expected = np.arange(2, 8)
          assert_array_equal(result, expected)

      def test_slicewise_eval_0_24(self):
          """test slicewise_eval for nmin=nmax=0

          """
          result = thermodynamics.slicewise_eval(lambda x: x+1, 0, 24, 10)
          expected = np.arange(1, 26)
          assert_array_equal(result, expected)

      def test_slicewise_eval_1_25(self):
          """test slicewise_eval for nmin=nmax=0

          """
          result = thermodynamics.slicewise_eval(lambda x: x+1, 1, 25, 10)
          expected = np.arange(2, 27)
          assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
