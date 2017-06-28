"""materials.py provides magnetic and electric responses of 
   pre-defined materials for the description of the Casimir 
   effect within imaginary frequency Matsubara formalism.

"""

import numpy as np
__version__ = "SVN Revision "+str("".join(list(filter(str.isdigit, "$Revision: 952 $"))))

class Material(object):
    is_perfect_conductor = False

    def __init__(self, l_normalization=1e-6):
        self.omega_norm = 2.998e8/l_normalization  

    def epsilon(self, omega):
        """determine the dielectric susceptibility at frequency omega

           This method should be implemented by a class representing
           a definite material.

        """
        raise NotImplementedError

    def mu(self, omega):
        """determine the magnetic susceptibility at frequency omega

           This method should be implemented by a class representing
           a definite material.

        """
        raise NotImplementedError


class PerfectConductor(Material):
    is_perfect_conductor = True
    is_sigma_finite = False
    is_magnetic = False

    def mu(self, k):
        return np.ones(k.shape)

def is_sigma_finite(e_properties):
    """ determine whether the static conductivity deriving from e_properties
        is finite or not
        
        Parameters
        ----------
        e_properties : list of len() = 3 tuples specifying (w_Pl, w_R, gamma)
        
        Returns
        -------
        True :  if no zero-frequency oscillator w_R = 0 with gamma=0 is contained
        False : otherwise

    """
    try:
        sigma_finite = not(min([params[2] for params in e_properties if params[1]==0
                                                                and not params[0]==0]))==0 
    except ValueError:
        sigma_finite = True
    return sigma_finite

def is_dielectric(e_properties):
    """ determine whether e_properties includes a resonance at zero frequency
        which would require free charge carrieres
        
        Parameters
        ----------
        e_properties : list of len() = 3 tuples specifying (w_Pl, w_R, gamma)
        
        Returns
        -------
        True :  if no zero-frequency oscillator w_R = 0 is contained
        False : otherwise

    """
    for resonator in e_properties:
        if not resonator[0]==0 and resonator[1] == 0:
            return False
    return True

def is_magnetic(b_properties):
    """ determine whether the static permeability is zero or not

        Parameters
        ----------
        b_properties : len() = 2 tuple containing mu(omega=0) and omega_M

        Returns
        -------
        True : if mu(0) = 0 or tuple has length zero
        False : otherwise

    """
    if len(b_properties) == 0:
        return False
    if not b_properties[0] == 0:
        return True

class LorentzSmithMaterial(Material):
    """ lorentz-smith material defining epsilon(k) like described in 
        and mu(omega) in a different manner

        Parameters
        ----------
        e_properties : list of len = 3 tuples specifying (w_Pl, w_R, gamma)
        b_properties : tuple of len = 2 specifying (mu(k=0), w_M)

        Returns
        -------
        An instance of LorentzMaterial

    """
    def __init__(self, e_properties, b_properties):
    
        self.e_properties = e_properties
        self.b_properties = b_properties
        self.is_sigma_finite = is_sigma_finite(e_properties)
        self.is_magnetic = is_magnetic(b_properties)
        self.omega_norm = Material().omega_norm
        self.is_dielectric = is_dielectric(e_properties)

    def epsilon(self, k):
        """determine the relative dielectric permittivity at imaginary 
           wavenumber k as a sum over Lorentz peaks, with plasma-
           frequency wP, resonance frequency wR and disspation frequency
           gamma.

           Parameters
           ----------
           k : imaginary wavenumber multiplied by micro meter
               as np.nd_array
           Return
           ------
           epsilon(k), the same shape as input array k

        """
        epsilon = 1
        assert len(self.e_properties)==1
        for params in self.e_properties:
            wP, wR, gamma = [p/self.omega_norm for p in params]
            lorentz = wP**2/( k**2 + gamma*k)*(1-0.49*gamma/(k+gamma))
            epsilon = epsilon+lorentz
        return epsilon

    def mu(self, k):
        """determine the relative magnetic permeability at imaginary
           wavenumber k*1e-6 micro meter according to [1]
           
           References:
           [1] Guerot R. et. al. arXiv:1508.01659v1 [quant-ph] (2015)

        """
        if type(k) == float:
            mu = 1
        else:
            mu = np.ones(k.shape)
        if self.b_properties == ():
            return mu
        for params in [self.b_properties]:
            mu0 = params[0]
            wM = params[1]/self.omega_norm
            mu += (mu0-1)/(1+k/wM)
        return mu


class LorentzMaterial(Material):
    """ lorentz material defining epsilon(k) a sum of lorentz osciallators
        and mu(omega) in a different manner

        Parameters
        ----------
        e_properties : list of len = 3 tuples specifying (w_Pl, w_R, gamma)
        b_properties : tuple of len = 2 specifying (mu(k=0), w_M)

        Returns
        -------
        An instance of LorentzMaterial

    """
    def __init__(self, e_properties, b_properties):
    
        self.e_properties = e_properties
        self.b_properties = b_properties
        self.is_sigma_finite = is_sigma_finite(e_properties)
        self.is_magnetic = is_magnetic(b_properties)
        self.omega_norm = Material().omega_norm
        self.is_dielectric = is_dielectric(e_properties)

    def epsilon(self, k):
        """determine the relative dielectric permittivity at imaginary 
           wavenumber k as a sum over Lorentz peaks, with plasma-
           frequency wP, resonance frequency wR and disspation frequency
           gamma.

           Parameters
           ----------
           k : imaginary wavenumber multiplied by micro meter
               as np.nd_array
           Return
           ------
           epsilon(k), the same shape as input array k

        """
        if not type(k) == np.ndarray:
            epsilon = 1
        else:
            epsilon = np.ones(k.shape)
        for params in self.e_properties:
            wP, wR, gamma = [p/self.omega_norm for p in params]
            lorentz = wP**2/(wR**2 + k**2 + gamma*k)
            epsilon = epsilon+lorentz
        return epsilon

    def mu(self, k):
        """determine the relative magnetic permeability at imaginary
           wavenumber k*1e-6 micro meter according to [1]
           
           References:
           [1] Guerot R. et. al. arXiv:1508.01659v1 [quant-ph] (2015)

        """
        if type(k) == np.ndarray:
            mu = np.ones(k.shape)
        else:
            mu = 1
        if self.b_properties == ():
            return mu
        for params in [self.b_properties]:
            mu0 = params[0]
            wM = params[1]/self.omega_norm
            mu += (mu0-1)/(1+k/wM)
        return mu

e_Gold = [(1.37e16, 0, 5.77e13)]
e_Gold_Bimonte = [(1.37e16, 0, 5.32e13)]
e_Silver = [(1.46e16, 0, 3.46e13)] # Blaber
e_Silver_plasma = [(e_Silver[0][0], 0, 0)]
b_Silver = ()
e_Gold_plasma = [(e_Gold[0][0], 0, 0)]
b_Gold = ()
e_Gold_plasma_wp1 = [(1.37e16, 0, 0)]
e_Gold_plasma_wp01 = [(1.37e15, 0, 0)]
e_Gold_plasma_wp10 = [(1.37e17, 0, 0)]
b_Gold_magnetic = (4, 1e9)
e_Mercury = [(5.4e15, 0, 2.5e14)]
e_Mercury_plasma = [(e_Mercury[0][0], 0, 0)]
e_MercurySmith = [(1.98e16, 0, 1.647e15)]
e_MercurySmith_plasma = [(e_MercurySmith[0][0], 0, 0)]
e_SiC = [(7e14, 0, 1.5e14)]
e_SiC_plasma = [(e_SiC[0][0], 0, 0)]
b_SiC = ()
e_Water = [(4.16e13, 3.48e13, 0),
           (4.16e12, 1.33e12, 0),
           (1.10e13, 7.49e12, 0),
           (1.14e14, 1.56e14, 0),
           (9.00e15, 1.44e16, 0),
           (1.63e16, 3.18e16, 0),
           (1.48e16, 4.01e16, 0)]
e_artificial = [(1e18, 0, 0)]
e_modWater = [(8e11, 1e11, 0)]
e_modWater.extend(e_Water)
e_Ethanol = [(23.84**0.5*6.6e14,  6.6e14, 0.1e13),
             (0.852**0.5*1.14e16, 1.14e16, 0.1e13)]
e_AlternEth = [(2.41e12, 2.46e12, 0),
               (8.35e12, 6.56e12, 0),
               (6.37e13, 1.70e14, 0),
               (3.70e15, 1.04e16, 0),
               (1.49e16, 2.31e16, 0),
               (1.17e16, 2.37e16, 0),
               (1.77e16, 6.65e16, 0)]
e_Ptfe = [(4.40e10, 4.56e11, 0),
          (1.56e12, 1.15e13, 0),
          (3.15e13, 8.46e13, 0),
          (6.41e13, 1.91e14, 0),
          (4.50e15, 1.02e16, 0),
          (1.87e16, 2.83e16, 0),
          (2.08e16, 6.40e16, 0),
          (2.32e16, 1.18e17, 0)]
e_Bromobenzene = [(1.78e12, 7.63e12, 0),
                  (6.37e12, 4.69e13, 0),
                  (3.68e13, 1.69e14, 0),
                  (7.48e15, 1.03e16, 0),
                  (1.62e16, 2.02e16, 0),
                  (1.79e16, 3.65e16, 0),
                  (1.46e16, 1.52e17, 0)]
e_Altern2Ps = [(3.17e13, 1.79e14, 0),
               (1.48e11, 1.37e12, 0),
               (2.66e12, 1.81e13, 0),
               (2.27e14, 2.37e15, 0),
               (5.03e15, 9.30e15, 0),
               (1.24e16, 1.53e16, 0),
               (1.98e16, 3.07e16, 0),
               (1.52e16, 1.04e17, 0)]
e_Altern1Ps = [(1.67e11, 1.52e12, 0),
              (2.97e12, 2.01e13, 0),
              (7.89e14, 5.89e15, 0),
              (3.48e13, 1.99e14, 0),
              (5.01e15, 9.10e15, 0),
              (1.22e16, 1.55e16, 0),
              (1.63e16, 2.86e16, 0),
              (1.42e16, 7.82e16, 0)]
e_Polystyrene =  [(0.2*5.54e14, 5.54e14, 0),
                  (1.56*1.35e16, 1.35e16, 0)]
e_Silicon = [(5.53e13, 6.24e13, 0),
             (7.67e13, 1.70e14, 0),
             (1.10e14, 1.70e14, 0),
             (1.06e14, 1.69e14, 0),
             (4.93e15, 2.20e16, 0),
             (2.34e16, 2.58e16, 0),
             (5.76e15, 1.24e16, 0),
             (3.26e16, 1.39e17, 0)]



Gold = LorentzMaterial(e_Gold, b_Gold)
Gold.name = "Gold"
Gold_Bimonte = LorentzMaterial(e_Gold_Bimonte, b_Gold)
Gold_Bimonte.name = "Gold_Bimonte"
Gold_plasma = LorentzMaterial(e_Gold_plasma, b_Gold)
Gold_plasma.name = "Gold_plasma"
Gold_plasma_wp1 = LorentzMaterial(e_Gold_plasma_wp1, b_Gold)
Gold_plasma_wp1.name = "Gold_plasma_wp1"
Gold_plasma_wp01 = LorentzMaterial(e_Gold_plasma_wp01, b_Gold)
Gold_plasma_wp01.name = "Gold_plasma_wp01"
Gold_plasma_wp10 = LorentzMaterial(e_Gold_plasma_wp10, b_Gold)
Gold_plasma_wp10.name = "Gold_plasma_wp10"
Gold_magnetic = LorentzMaterial(e_Gold, b_Gold_magnetic)
Gold_magnetic.name = "Gold_magnetic"
Silver = LorentzMaterial(e_Silver, b_Silver)
Silver_plasma = LorentzMaterial(e_Silver_plasma, b_Silver)
Silver.name = "Silver"
Silver_plasma.name = "Silver_plasma"
SiC = LorentzMaterial(e_SiC, b_SiC)
SiC.name = "SiC"
SiC_plasma = LorentzMaterial(e_SiC_plasma, b_SiC)
SiC_plasma.name = "SiC_plasma"
Mercury = LorentzMaterial(e_Mercury, ())
Mercury.name = "Mercury"
Mercury_plasma = LorentzMaterial(e_Mercury_plasma, ())
Mercury_plasma.name = "Mercury_plasma"
MercurySmith = LorentzSmithMaterial(e_MercurySmith, ())
MercurySmith.name = "MercurySmith"
MercurySmith_plasma = LorentzMaterial(e_MercurySmith_plasma, ())
MercurySmith_plasma.name = "MercurySmith_plasma"
Water = LorentzMaterial(e_Water, ())
Water.name = "Water"
modifiedWater = LorentzMaterial(e_modWater, ())
modifiedWater.name = "modifiedWater"
Ethanol = LorentzMaterial(e_Ethanol, ())
Ethanol.name = "Ethanol"
AlternEth = LorentzMaterial(e_AlternEth, ())
AlternEth.name = "AlternEth"
Altern1Ps = LorentzMaterial(e_Altern1Ps, ())
Altern1Ps.name = "Altern1Ps"
Altern2Ps = LorentzMaterial(e_Altern2Ps, ())
Altern2Ps.name = "Altern2Ps"
Polystyrene = LorentzMaterial(e_Polystyrene, ())
Polystyrene.name = "Polystyrene"
Bromobenzene = LorentzMaterial(e_Bromobenzene, ())
Bromobenzene.name = "Bromobenzene"
Ptfe = LorentzMaterial(e_Ptfe, ())
Ptfe.name = "Ptfe"
Vacuum = LorentzMaterial((), ())
Vacuum.name = "Vacuum"
PerfectConductor = PerfectConductor()
PerfectConductor.name = "PerfectConductor"
Artificial = LorentzMaterial(e_artificial, ())
Artificial.name = "Artificial"
Silicon = LorentzMaterial(e_Silicon, ())
Silicon.name = "Silicon"


