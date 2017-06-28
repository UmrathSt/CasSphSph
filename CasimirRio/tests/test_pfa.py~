"""Tests for pfa

"""

import unittest
import numpy as np
import PFA
import materials


class pfa(unittest.TestCase):
    def setUp(self):
        self.a = 0.1
        self.b = 0.7
        self.L = 0.19
        self.matMd = materials.Vacuum
        self.matSpa = materials.PerfectConductor
        self.matSpb = materials.PerfectConductor
        self.g = PFA.SphereSpherePFA(self.matSpa, self.matSpb, self.matMd)

    def test_distance(self):
        """ return the surface to surface distance
        """
        self.assertAlmostEqual(self.g.distance(self.a,self.b,self.a+self.b+self.L,0),
                               self.L)

    def test_pec_planeplaneE0(self):
        expected = -np.pi**3 / (360*self.L**3)
        result = self.g.planeplaneFreeEnergy(T=0, L=self.L)
        self.assertAlmostEqual(expected, result)

    def test_pec_planeplaneHT(self):
        expected = -1.202056903 / (8*np.pi*self.L**2)
        result = float(self.g.planeplaneFreeEnergy(T=np.inf, L=self.L))
        self.assertAlmostEqual(expected, result)


if __name__ == "__main__":
    unittest.main()
