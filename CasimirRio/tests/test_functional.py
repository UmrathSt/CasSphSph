"""Functional tests

"""
import numpy as np
from numpy.testing import assert_array_almost_equal
import unittest
import thermodynamics
import materials
import geometry


class DipoleLimit(unittest.TestCase):

    def setUp(self):
        """choose L/r1 and L/r2 sufficiently large so that the large-distance
           approximation works reasonably well

        """
        pec = materials.PerfectConductor
        vac = materials.Vacuum
        r1 = 0.9
        r2 = 0.6
        L = 40
        lmax = 1
        lssmax =1
        forceFlag = 0
        l_offset = 0
        precision = 1e-5
        analytic_n0 = True
        self.geom = geometry.SphereSphere(r1, r2, L, pec, pec, vac, lmax, 
                                        l_offset, lssmax, precision, forceFlag)
        deltaT = 0.01
        Tmax = 1
        nmax = 500
        self.td = thermodynamics.finiteT(deltaT, nmax, Tmax, self.geom, analytic_n0)
        nu = np.linspace(deltaT, Tmax, 100)*self.geom.L
        self.nu = nu
        self.g = nu/np.sinh(nu)
        self.c = np.cosh(nu)

    def test_free_energy(self):
        """compare free energy with analytical results in the large-distance
           limit

        """
        result = self.td.free_energy()
        g = self.g
        c = self.c
        ftmtm0 = 2*g*c+2*g**2+g**3*c
        ftmtm1 = g*c+g**2+1.5*g**3*c+0.5*g**4*(2*c**2+1)+0.5*g**5*c*(c**2+2)
        ftmte1 = 0.25*(g**3*c+g**4*(2*c**2+1)+g**5*c*(c**2+2))
        expected = -(self.geom.r1*self.geom.r2/self.geom.L**2)**3*(
                                           1.25*ftmtm0+1.25*ftmtm1+2*ftmte1)
        expected = expected/self.geom.L
        assert_array_almost_equal(result/expected, 1, decimal=4)

    @unittest.expectedFailure
    def test_entropy(self):
        """compare entropy with analytical results in the large-distance
           limit

        """
        result = self.td.entropy()
        g = self.g
        c = self.c
        nu = self.nu
        stmtm0 = (2*g*c+2*g**2-g**3*c-g**4*(2*c**2+1))/nu
        stmtm1 = (g*c+g**2+2.5*g**3*c+0.5*g**4*(2*c**2+1)
                  + 0.5*g**5*c*(c**2+2)-0.5*g**6*(2*c**4+11*c**2+2))/nu
        stmte1 = 0.25*(3*g**3*c+3*g**4*(2*c**2+1)+g**5*c*(c**2+2)
                       - g**6*(2*c**4+11*c**2+2))/nu
        expected = (self.geom.r1*self.geom.r2/self.geom.L**2)**3*(
                                           1.25*stmtm0+1.25*stmtm1+2*stmte1)
        assert_array_almost_equal(result/expected, 1, decimal=4)

    def test_evaluation_n0_1(self):
        """test the analytical evaluation of the zeroth Matsubara fequency
           against a numerical approximation

        """
        r1 = 9
        r2 = 6
        L = 19
        lmax = "automatic"
        lssmax = "automatic"
        l_offset = 0
        precision = 1e-4
        forceFlag = 0
        deltaT = 0.1
        nmax = "automatic"
        Tmax = 10
        mat1 = [materials.Gold, materials.Gold_plasma, 
                materials.Altern1Ps, materials.Gold_magnetic]
        mat2 = materials.Gold
        mat3 = materials.Ethanol
        for m1 in mat1:
            geom = geometry.SphereSphere(r1, r2, L, m1, mat2, mat3, lmax, 
                                            l_offset, lssmax, precision, forceFlag)
            td = thermodynamics.finiteT(deltaT, nmax, Tmax, geom)
            result1 = td.F_alle(np.array([0]))
            result2 = td.F_alle(np.array([1e-12]))
            self.assertAlmostEqual(float(result1/result2), 0.5)



if __name__ == "__main__":
    unittest.main()
