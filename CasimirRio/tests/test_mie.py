"""tests for Mie coefficients

"""
import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
import scattering
import unittest
import materials
import stratified_sphere_scattering as strat_scat
from math import pi, sqrt, log, exp, factorial
from scipy.misc import factorial2


class mie_k0(unittest.TestCase):
    
    def test_perfect_conductor(self):
        pec = materials.PerfectConductor
        vac = materials.Vacuum
        R = 1
        lmin = 1
        lmax = 3
        TE, TM, ln_scale = scattering.scattering(pec, vac).mie_k0(lmin, lmax, R)
        expectedTE = np.array([-1/3, +1/45, -1/1575])
        expectedTM = np.array([+2/3, -1/30, +4/4725])
        resultTE = TE*np.exp(ln_scale)
        resultTM = TM*np.exp(ln_scale)
        assert_array_almost_equal(resultTE, expectedTE)
        assert_array_almost_equal(resultTM, expectedTM)

    def test_dielectrica(self):
        polystyrene = materials.Altern1Ps
        modifiedWater = materials.modifiedWater
        Vacuum = materials.Vacuum
        Gold = materials.Gold
        Gold_plasma = materials.Gold_plasma
        dielectrica = [polystyrene, modifiedWater, Vacuum]
        non_dielectrica = [Gold, Gold_plasma]
        for material in dielectrica:
            assert(material.is_dielectric == True)
        for material in non_dielectrica:
            assert(material.is_dielectric == False)

    def test_dielectric_sphere(self):
        matSp = materials.modifiedWater
        matMd = materials.Altern1Ps
        k0 = np.array([0])
        muSp = matSp.mu(k0)
        muMd = matMd.mu(k0)
        nSp = np.sqrt(matSp.epsilon(k0)*muSp)
        nMd = np.sqrt(matMd.epsilon(k0)*muMd)
        R = 1
        lmin = 1
        lmax = 3
        TE, TM, ln_scale = scattering.scattering(matSp, 
                        matMd).mie_k0(lmin, lmax, R)
        resultTM = TM*np.exp(ln_scale)
        expectedTM = np.ones(lmax+1-lmin)
        for i, l in enumerate(range(lmin, lmax+1)):
            prefac = (-1)**l*(-l-1)/(factorial2(2*l-1)**2*(2*l+1))
            expectedTM[i] *= ((muMd*nSp**2 - muSp*nMd**2)
                               /(l*muMd*nSp**2 + (l+1)*muSp*nMd**2)
                               )*prefac
        assert_array_almost_equal(resultTM, expectedTM)
        

    def test_Drude_limit(self):
        gold = materials.Gold
        vac = materials.Vacuum
        R = 1
        lmin = 1
        lmax = 2
        TE, TM, ln_scale = scattering.scattering(gold, vac).mie_k0(lmin, lmax, R)
        expectedTE = np.array([0, 0])
        expectedTM = np.array([+2/3, -1/30])
        resultTE = TE*np.exp(ln_scale)
        resultTM = TM*np.exp(ln_scale)
        assert_array_almost_equal(resultTE, expectedTE)
        assert_array_almost_equal(resultTM, expectedTM)

    def test_Plasma_limit(self):
        """test the low-frequency limit of the Mie-coefficients 
           versus results calculated in Maple
        """
        gold_plasma = materials.Gold_plasma
        vac = materials.Vacuum
        R = 1
        lmin = 1
        lmax = 2
        TE, TM, ln_scale = scattering.scattering(gold_plasma, vac).mie_k0(lmin, lmax, R)
        expectedTE = np.array([-0.3119289967, 0.01989598050])
        expectedTM = np.array([+2/3, -1/30])
        resultTE = TE*np.exp(ln_scale)
        resultTM = TM*np.exp(ln_scale)
        assert_array_almost_equal(resultTE, expectedTE)
        assert_array_almost_equal(resultTM, expectedTM)

    def test_Magnetic_limit(self):
        gold_magnetic = materials.Gold_magnetic
        vac = materials.Vacuum
        R = 1
        lmin = 1
        lmax = 2
        TE, TM, ln_scale = scattering.scattering(gold_magnetic, vac).mie_k0(lmin, lmax, R)
        expectedTE = np.array([+1/3,  -1/55])
        expectedTM = np.array([+2/3, -1/30])
        resultTE = TE*np.exp(ln_scale)
        resultTM = TM*np.exp(ln_scale)
        assert_array_almost_equal(resultTE, expectedTE)
        assert_array_almost_equal(resultTM, expectedTM)


class mie_small_arg(unittest.TestCase):
    def setUp(self):
        self.k = np.array([1e-6])
        self.lmax = (1, 1)
        self.r = 1
        self.matMd = materials.Ethanol
        self.matSp = materials.Altern1Ps#LorentzMaterial([(1.37*1e16, 0, 5.77e13)], ()) 

    def test_mie_small(self):
        """Check whether Mie-Coefficients boil down to desired low argument
           behaviour:

           MieTE(l=1, ka, nMd, nSp), ka --> 0:
               -2/(45*pi) * nMd^3 * (kr)^5 * (nMd^2 - nSp^2)

           MieTM(l=1, ka, nMd, nSp), ka --> 0:
               4/(3*pi) * (nMd*kr)^3 * (nMd^2 - nSp^2) / (2 nMd^2 + nSp^2)

        """
        nMd = sqrt(self.matMd.epsilon(self.k))
        nSp = sqrt(self.matSp.epsilon(self.k))
        expectedTE = 1/45*nMd**3*(nMd**2-nSp**2)*(self.k)**5
        expectedTM = -2/3*nMd**3*(nMd**2-nSp**2)/(2*nMd**2+nSp**2)*(self.k)**3

        for l in self.lmax:
            s = scattering.scattering(self.matSp, self.matMd)
            scaledresultTE, scaledresultTM, scaling = s.mie(
                self.k, l, l, self.r)
            resultTM = (scaledresultTM * exp(scaling)).flatten()
            resultTE = (scaledresultTE * exp(scaling)).flatten()
            self.assertAlmostEqual(np.round(resultTM/expectedTM), 1)
            self.assertAlmostEqual(np.round(resultTE/expectedTE), 1)

    def test_zeroInnerRadius(self):
        matcore = materials.Altern1Ps
        matCoating = materials.Gold
        matMedium = materials.Vacuum
        ra = 2
        ri = 1e-2
        lmax = (1, 10)
        for l in self.lmax:
            expected = scattering.scattering(matCoating, matMedium)
            expected = expected.mie(self.k, l, l, ra)
            result = scattering.coated_miescattering(matcore, matCoating,
                                                     matMedium)
            result = result.mie_coated(self.k, l, l, ri, ra)
            rTE = result[0][0, 0]
            rTM = result[1][0, 0]
            rs = result[2][0,0]
            eTE = expected[0][0, 0]
            eTM = expected[1][0, 0]
            es = expected[2][0,0]
            self.assertAlmostEqual(rTE/eTE*np.exp(rs-es), 1)
            self.assertAlmostEqual(rTM/eTM*np.exp(rs-es), 1)

    def test_transition_homogeneous_sphere1(self):
        matcore = materials.Polystyrene
        matCoating = matcore
        matMedium = materials.Ethanol
        ra = 1
        ri = 0.7
        lmax = (1, 10)
        ks = (np.array([1e-4]), np.array([1]), np.array([1e3]))
        for k in ks:
            for l in lmax:
                expected = scattering.scattering(matCoating, matMedium)
                expected = expected.mie(k, l, l, ra)
                result = scattering.coated_miescattering(matcore, matCoating,
                                                         matMedium)
                result = result.mie_coated(k, l, l, ri, ra)
                rTE = result[0][0, 0]
                rTM = result[1][0, 0]
                eTE = expected[0][0, 0]
                eTM = expected[1][0, 0]
                self.assertAlmostEqual(rTE/eTE, 1)
                self.assertAlmostEqual(rTM/eTM, 1)

    def test_transition_homogeneous_sphere2(self):
        matcore = materials.Polystyrene
        matCoating = materials.Ethanol
        matMedium = materials.Ethanol
        ra = 1
        ri = 0.7
        lmax = (1, 10)
        ks = (np.array([1e-4]), np.array([1]), np.array([1e3]))
        for k in ks:
            for l in lmax:
                expected = scattering.scattering(matcore, matMedium)
                expected = expected.mie(k, l, l, ri)
                result = strat_scat.StratifiedMieScattering([matcore, matCoating, matMedium],
                                    [ri, ra])
                result = result.mie_stratified(k, l, l)
                rTE = result[0][0, 0]
                rTM = result[1][0, 0]
                eTE = expected[0][0, 0]
                eTM = expected[1][0, 0]
                self.assertAlmostEqual(rTE/eTE*np.exp(-expected[2][0, 0]
                                                      + result[2][0, 0]),
                                       1)
                self.assertAlmostEqual(rTM/eTM*np.exp(-expected[2][0, 0]
                                                      + result[2][0, 0]),
                                       1)
    def test_dielectric_lowfreq_limit_interal(self):
        """check whether the internal scattering coefficients boil down to the analytic
           limit for small frequencies in case of dielectric materials
        """
        r = 1
        lmin = 1
        lmax = 5
        l = np.arange(lmin, lmax+1)
        matSp = materials.Vacuum
        matMd = materials.Polystyrene
        k0 = np.array([1e-3])
        nSp2 = matSp.epsilon(k0)*matSp.mu(k0)
        nMd2 = matMd.epsilon(k0)*matMd.mu(k0)
        S = scattering.scattering(matSp, matMd)
        b, a, s = S.mie_internal(k0, lmin, lmax, r)
        result_a = (a*np.exp(s))[0, :]
        be, ae, se = S.mie_internal_k0(lmin, lmax, r)
        assert_array_almost_equal(a[0,:]/ae*np.exp(s[0,:]-se+(2*l+1)*np.log(k0)), np.ones(result_a.shape))

    def test_mie_inverse_plasma(self):
        """ check whether the internal scattering coefficients for a plasma conductor
            in the exterior region approaches its limiting value for plasma frequency
            to infinity, which is just 1/(the  exterior scattering coefficient)
            for a perfectly conducting sphere
        """
        r = 10
        lmin = 1
        lmax = 5
        k = np.array([1e-6])
        pec = materials.PerfectConductor
        plasma = materials.Gold_plasma
        plasma.e_properties = [(5e16, 0, 0)]
        vac = materials.Vacuum
        b_ext, a_ext, s_ext = scattering.scattering(pec, vac).mie(k, lmin, lmax, r)
        b_int, a_int, s_int = scattering.scattering(vac, plasma).mie_internal(k, lmin, lmax, r)
        assert_array_almost_equal(a_ext*a_int*np.exp(s_ext+s_int), np.ones(b_ext.shape))

class mie_helper_functions(unittest.TestCase):

    def test_ln_double_factorial(self):
        """ test the natural logarithm of (2*l-1)!!
    
        """
        L = 10
        result = scattering.ln_2dbl(L)
        l = np.array([19, 17, 15, 13, 11, 9,
                      7, 5, 3, 1])
        expected = log(l.prod())
        assert_almost_equal(result,expected)
    
    def test_log_factorial(self):
        """ test the logarithm of a factorial
        
        """
        l = np.arange(4)
        result = scattering.lfac(l)
        expected = np.array([log(factorial(l)) for l in l])
        assert_array_almost_equal(result, expected)

if __name__ == "__main__":
    unittest.main()
