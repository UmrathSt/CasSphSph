"""Tests for bessel_sk

"""
import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
import unittest
import mpmath
import bessel_sk
from helper_functions import ln_2dbl


class Bessel_check_args(unittest.TestCase):

    def test_knu_nonint_order(self):
        """float values for lmax should not be accepted

        """
        x = np.array([1, 2])
        lmax = 1.5
        with self.assertRaises(TypeError):
            bessel_sk.lnknu(x, lmax)

    def test_iknu_nonint_order(self):
        """float values for lmax should not be accepted

        """
        x = np.array([1, 2])
        lmax = 1.5
        with self.assertRaises(TypeError):
            bessel_sk.lniknu(x, lmax)


class Bessel_small_args(unittest.TestCase):

    def setUp(self):
        self.orders = (0, 1, 10, 100, 1000)
        self.maxorder = self.orders[-1]
        self.x = np.array([1e-6])

    def test_iknu(self):
        """test whether the scaled Bessel functions I and K reach their
           correct values in the limit of vanishing argument
           numerical "expected" results taken from http://www.wolframalpha.com/input

        """
        x = self.x
        result_i, result_k = bessel_sk.lniknu(x, self.maxorder)
        i_exp = [1.66666666666e-13, -14.914122846, -161.4993600994,
                -1817.03225428, -20424.360951764]
        k_exp = [14.2670922632, 28.082603821, 172.7219309,
                1825.996042640, 20431.0266426932]
        for i, l in enumerate(self.orders):
            assert_almost_equal(result_i[0, l]/i_exp[i], 1, decimal=4)
            assert_almost_equal(result_k[0, l]/k_exp[i], 1, decimal=4)

    def test_knu(self):
        """test whether the scaled Bessel function K reaches its
           correct value in the limit of vanishing argument

        """
        x = self.x
        result = bessel_sk.lnknu(x, self.maxorder)
        k_exp = [14.2670922632, 28.082603821, 172.7219309,
                1825.996042640, 20431.0266426932]
        for i, l in enumerate(self.orders):
            assert_almost_equal(result[0,l]/k_exp[i], 1)

    def test_diff_inu(self):
        """test difference of Bessel functions I appearing in the
           numerator of the Mie coefficients

        """
        l = 1
        x = np.array([1e-9])
        n = 10
        bessel_i = bessel_sk.lniknu(x, l)[0]
        bessel_i_n = bessel_sk.lniknu(n*x, l)[0]
        result =  (-np.exp(bessel_i_n[:, -1] + bessel_i[:, -2])
                  + np.exp(bessel_i[:, -1] + bessel_i_n[:, -2]))
        expected = ( (-n+1)*x/3+
                     (-n/3 - n**3/5 + n**2/3 + 1/5)*x**3/6)
        assert_almost_equal(result/expected, 1, decimal=7)


class Bessel_generic(unittest.TestCase):

    def test_k_small_nu(self):
        """test small orders of scaled K against analytical results

        """
        x = np.array([1, 3])
        result = bessel_sk.lnknu(x, 2)
        pih = np.log(0.5*np.pi)
        expected = np.array([pih -x - 1*np.log(x),
                            pih -x - 2*np.log(x) + np.log(x+1),
                            pih -x - 3*np.log(x) + np.log(x**2+3*x+3)])
        assert_almost_equal(result, expected.T)

    def test_ik_small_nu(self):
        """test small orders of scaled I and K against analytical results

        """
        x = np.array([0.01, 1, 3])
        result = bessel_sk.lniknu(x, 2)
        expected_i = np.array([
            np.log(np.sinh(x)/x),
            np.log((np.cosh(x)*x-np.sinh(x))/x**2),
            np.log((np.sinh(x)*(x**2+3)-3*np.cosh(x)*x)/x**3)
                               ]).T
        expected_k = np.array([
            np.log(np.pi/(2*x)) - x,
            np.log((x+1)*np.pi/(2*x**2)) - x,
            np.log((x**2+3*x+3)*np.pi/(2*x**3)) - x
                             ]).T
        assert_array_almost_equal(result[0]/expected_i, 1)
        assert_array_almost_equal(result[1]/expected_k, 1)

    def test_iknu_generic(self):
        """test intermediate regime for I and K where order is large but
           argument still relatively small

        """
        l = np.array([500])
        x = np.array([1e2])
        result_i, result_k = bessel_sk.lniknu(x, l[0])
        expected_i = (mpmath.log(mpmath.besseli(l[0]+0.5, x[0]))
                                +0.5*mpmath.log(mpmath.pi/(2*x[0]))  
                                )
        expected_k = (mpmath.log(mpmath.besselk(l[0]+0.5, x[0]))
                                +0.5*mpmath.log(mpmath.pi/(2*x[0]))
                            )  
        self.assertAlmostEqual(result_i[0, -1]/expected_i, 1)
        self.assertAlmostEqual(result_k[0, -1]/expected_k, 1)

    def test_knu_generic(self):
        """test intermediate regime for K where order is large but
           argument still relatively small

        """
        l = np.array([500])
        x = np.array([1e2])
        result = bessel_sk.lnknu(x, l[0])[0, -1]
        expected = (mpmath.log(mpmath.besselk(l[0]+0.5, x[0]))
                                    +0.5*mpmath.log(mpmath.pi/(2*x[0]))  
                              )
        assert_almost_equal(result/expected, 1)

    def test_I_fraction(self):
        """test the behaviour of i_fraction(x, nu) which is itended to
           return the ratio I(nu, x)/I(nu+1, x)
        """
        nu = np.array([0, 1, 10, 101, 450, 1001])+0.5
        x = np.array([1e-4, 1, 1e2, 1e3])
        result = bessel_sk.i_fraction(x, nu)
        expected = np.zeros((len(nu), len(x)))
        for i in range(len(nu)):
            for j in range(len(x)):
                X = x[j]
                NU = nu[i]
                expected[i,j] = mpmath.besseli(NU, X)/mpmath.besseli(NU+1, X)
        assert_almost_equal(result/expected, 1)


class Bessel_large_args(unittest.TestCase):

    def test_inu(self):
        """test I and K for large arguments against asymptotic
           behavior DLMF10.40.1 and 10.40.2

        """
        lmax = 3
        x = np.array([5000])
        result_i, result_k = bessel_sk.lniknu(x, lmax)
        pih = np.log(0.5*np.pi)
        expP = (1+np.exp(-2*x))
        expM = (1-np.exp(-2*x))
        expected_i = np.array([
           -np.log(2*x**1) + x + np.log(expM),
           -np.log(2*x**2) + x + np.log(expM*(x+1)+x-1),
           -np.log(2*x**3) + x + np.log((3+x**2)*expM-3*x*expP),
           -np.log(2*x**4) + x + np.log((15*x+x**3)*expP-(15+6*x**2)*expM)           
                               ])
        expected_k = np.array([pih -x - 1*np.log(x),
                               pih -x - 2*np.log(x) + np.log(x+1),
                               pih -x - 3*np.log(x) + np.log(x**2+3*x+3),
                               pih -x - 4*np.log(x) + np.log(x**3+6*x**2+15*x+15)
                              ])
        assert_almost_equal(result_i[0]/expected_i.T, 1, decimal=4)
        assert_almost_equal(result_k[0]/expected_k.T, 1, decimal=4)

    def test_knu(self):
        """test K for large arguments against asymptotic
           behavior 10.40.2

        """
        lmax = 3
        x = np.array([500])
        result = bessel_sk.lnknu(x, lmax)
        pih = np.log(0.5*np.pi)
        expected = np.array([pih -x - 1*np.log(x),
                             pih -x - 2*np.log(x) + np.log(x+1),
                             pih -x - 3*np.log(x) + np.log(x**2+3*x+3),
                             pih -x - 4*np.log(x) + np.log(x**3+6*x**2+15*x+15)
                             ])
        assert_almost_equal(result[0]/expected.T, 1)


class Helperfunctions(unittest.TestCase):

    def test_mie_length(self):
        """verify length of array in angular momentum axis

        """
        kr = np.array([1, 2])
        lmin = 5
        lmax = 10
        x, y = bessel_sk.mie_bessels(kr, lmin, lmax)
        self.assertEqual(x.shape, (len(kr), lmax-lmin+1))
        self.assertEqual(y.shape, (len(kr), lmax-lmin+1))

    def test_mie_lmin_gt_lmax(self):
        """check raising of IndexError for lmin>lmax

        """
        x = np.array([1, 2])
        lmin = 5
        lmax = lmin-1
        with self.assertRaises(ValueError):
            bessel_sk.mie_bessels(x, lmin, lmax)

    def test_trans_extflag0(self):
        """check correct handling of ext_flag=0, this should
           result in a scaled Bessel function I

        """
        x = np.array([1, 2])
        lmax = 0
        ext_flag = 0
        expected = -np.log(x) + np.log(np.sinh(x))
        result = bessel_sk.trans_bessels(x, lmax, ext_flag)
        assert_almost_equal(result[:, 0], expected.T)

    def test_trans_extflag1(self):
        """check correct handling of ext_flag=1, this should
           result in a scaled Bessel function K

        """
        x = np.array([1, 2])
        lmax = 0
        ext_flag = 1
        expected = np.log(np.pi/(2*x)) - x
        result = bessel_sk.trans_bessels(x, lmax, ext_flag)
        assert_almost_equal(result[:, 0], expected.T)


if __name__ == "__main__":
    unittest.main()
