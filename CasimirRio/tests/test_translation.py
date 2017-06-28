"""Tests for translation.py

"""
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import unittest
import translation
import materials


class TranslationCoefficients(unittest.TestCase):

    def test_k0_translation_dipole(self):
        l1, l2 = 1, 1
        m = [0, 1]
        lmin = 1
        lmax = 1
        lssmax = 1
        L = 1
        k = np.array([0])
        material = materials.Vacuum
        ext_flag = 1
        Tm0 = translation.translation(lmax, lssmax, 0, material, L, k, ext_flag,
                 forceFlag=0)
        Tm1 = translation.translation(lmax, lssmax, 1, material, L, k, ext_flag,
                 forceFlag=0)
        T0 = Tm0.trans_ab()
        T1 = Tm1.trans_ab()
        result_m0 = T0[0]*np.exp(T0[1])
        result_m1 = T1[0]*np.exp(T1[1])
        expected_m0 = 3
        expected_m1 = -1.5
        assert_almost_equal(result_m0, expected_m0)
        assert_almost_equal(result_m1, expected_m1)

    def test_external_dipole_translation(self):
        """test against the analytic dipole limit of
           the translation coefficients

        """
        k = np.logspace(-2,2,5)
        L = 1
        lmin = 1
        lmax = 1
        lssmax = 1
        ext_flag = 1
        m = 0
        material = materials.Vacuum
        Tm0 = translation.translation(lmax, lssmax, m, material, L, k, ext_flag,
                 forceFlag=0)
        result_m0 = Tm0.trans_ab()

        result_vEEm0ab = result_m0[0]*np.exp(result_m0[4])
        result_vEEm0ba = result_m0[1]*np.exp(result_m0[4])
        result_vEMm0ab = result_m0[2]*np.exp(result_m0[4])
        result_vEMm0ba = result_m0[3]*np.exp(result_m0[4])
        m = 1
        Tm1 = translation.translation(lmax, lssmax, m, material, L, k, ext_flag,
                 forceFlag=0)
        result_m1 = Tm1.trans_ab()
        result_vEEm1ab = result_m1[0]*np.exp(result_m0[4])
        result_vEEm1ba = result_m1[1]*np.exp(result_m0[4])
        result_vEMm1ab = result_m1[2]*np.exp(result_m0[4])
        result_vEMm1ba = result_m1[3]*np.exp(result_m0[4])
        expected_vEEm0 = 3*np.exp(-k*L)*((k*L)**(-2)+(k*L)**(-3))
        expected_vEEm1 = -1.5*np.exp(-k*L)*((k*L)**(-1)+(k*L)**(-2)+(k*L)**(-3))
        expected_VEMm1ab = 1.5*np.exp(-k*L)*((k*L)**(-1)+(k*L)**(-2))
        expected_VEMm1ba = -expected_VEMm1ab
        assert_array_almost_equal(result_vEEm0ab[:,0,0]/expected_vEEm0,np.ones(len(k)))
        assert_array_almost_equal(result_vEEm0ba[:,0,0]/expected_vEEm0,np.ones(len(k)))
        assert_array_almost_equal(result_vEEm1ab[:,0,0]/expected_vEEm1,np.ones(len(k)))
        assert_array_almost_equal(result_vEEm1ba[:,0,0]/expected_vEEm1,np.ones(len(k)))
        assert_array_almost_equal(result_vEMm1ab[:,0,0]/expected_VEMm1ab,np.ones(len(k)))
        assert_array_almost_equal(result_vEMm1ba[:,0,0]/expected_VEMm1ba,np.ones(len(k)))

