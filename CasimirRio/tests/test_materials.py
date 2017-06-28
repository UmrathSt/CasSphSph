"""Tests for materials

"""

import unittest
import materials
import numpy as np

class materials_check_args(unittest.TestCase):
    def test_conductivity(self):
        """ test whether conductivity is finite or not
            for non-magnetic materials
    
        """
        e1 = [(1e16, 0, 1e13), 
              (0, 0, 0),
              (1e14, 0, 0)] 
        e2 = [(1e16, 0, 1e13), 
              (0, 0, 0),
              (1e14, 0, 1e13)] 
        b = ()
        assert(materials.LorentzMaterial(e1, b).is_sigma_finite == False)
        assert(materials.LorentzMaterial(e2, b).is_sigma_finite == True)
    
    def test_eps_mu_shape(self):
        """test the return shapes of epsilon(k) and mu(k)

        """
        k1 = 1.3
        k2 = np.arange(1,6)
        k3 = np.linspace(0.2, 10, 100).reshape(20,5)
        material = materials.Gold
        assert(type(material.epsilon(k1)) == type(k1))
        assert(material.epsilon(k2).shape == k2.shape)
        assert(material.epsilon(k3).shape == k3.shape)
        assert(material.mu(k2).shape == k2.shape)
        assert(material.mu(k3).shape == k3.shape)

    def test_vacuum(self):
        """test the behaviour of material Vacuum with
           empty e_properties and empty b_properties

        """
        assert(materials.Vacuum.is_sigma_finite == True)
        assert(materials.Vacuum.is_magnetic == False)

    def test_magnetism(self):
        """ test whether a LorentzMaterial is magnetic or not

        """
        e = [(1e16, 0, 1e13), 
             (0, 0, 0),
             (1e14, 0, 0)] 
        b1 = ()
        b2 = (1,1e9)
        assert(materials.LorentzMaterial(e,b1).is_magnetic == False)
        assert(materials.LorentzMaterial(e,b2).is_magnetic == True)

    def test_PerfectConductor(self):
        """test the behaviour of PerfectConductor

        """
        pec = materials.PerfectConductor
        assert(pec.is_sigma_finite == False)
        assert(pec.is_perfect_conductor == True)
        assert(pec.is_magnetic == False)
        with self.assertRaises(NotImplementedError):
            pec.epsilon(1.24)
            
if __name__ == "__main__":
    unittest.main()
