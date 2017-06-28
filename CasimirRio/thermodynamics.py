"""thermodynamics.py implements functionality for the calculation of 
   Casimir forces, free energies and entropies in a sphere-sphere
   configuration.

"""

import numpy as np
import geometry
import scipy.integrate



class finiteT:# ??? Welche Methoden sind fuer den Benutzer gedacht?
    """finite temperature Casimir calculations for a given geometry
       for temperatures nu = 2pik_B T microns /hbar c
    
       - deltaT: lowest desired scaled temperature nu for which 
                 thermodynamic quantities will be returned.
       - nmax: maximum number of Matsubara frequencies to be
               calculated
       - Tmax: maximum temperature for which thermodynamics quantities
               will be returned
       - geom: a given geometry instance

    """
    def __init__(self, deltaT, nmax, Tmax, geom, analytic_n0=True):
        self.__version__ = "SVN Revision "+str("".join(list(filter(str.isdigit, 
                                                                "$Revision: 952 $"))))
        self.deltaT = deltaT
        self.analytic_n0 = analytic_n0
        if type(geom.mat1) == list or type(geom.mat2) == list:
            self.analytic_n0 = False
        self.nmax = nmax# ??? Was ist wenn nmax<maxstep gewählt ist?
        self.Tmax = Tmax
        if self.nmax == "automatic":
            gr1_out, gr2_out = geom.r1, geom.r2
            if type(gr1_out) == list:
                gr1_out = gr1_out[-1]
            if type(gr2_out) == list:
                gr2_out = gr2_out[-1]                    
            radii = [gr1_out, gr2_out]
            radii.sort()
            rs, rg = radii
            L = geom.L-rs-rg
            if rs + rg > geom.L:
                L = rg-rs-geom.L
            c = 0.4
            offset = 5
            if L/rs <=0.5:
                c = 0.1 + 2.21*np.exp(-rs/L)
            LT = L*deltaT
            if LT >=7:
                offset = 1
            exp = np.exp(-2*LT)
            self.nmax = int(c*geom.lmax/(LT)*(exp*(L+rs)/rs+1+LT/geom.lmax)+offset)
            self.nmax = max(self.nmax, 1)
            assert self.nmax >=1
        self.forceFlag = 0
        if (isinstance(geom, geometry.SphereSphere) or
            isinstance(geom, geometry.CoatedSphereSphere)):
            self.geometry = geom
        else:
            raise TypeError("geometry is not an instance of class "
                            + "SphereSphere")
        self.f_matsubaras = None
        self.force_matsubaras = None
        self.maxstep = int(round(self.Tmax / self.deltaT))

    def free_energy(self):
        """compute the Casimir free energy times 2π/ℏc in units of micrometers

        """
        if self.f_matsubaras is None:
            self.f_matsubaras = self.get_f_matsubaras(0)
        Ts = self.deltaT*np.arange(1, self.maxstep+1)
        F_T = Ts*np.array([self.f_matsubaras[::i].sum()
                           for i in range(1, self.maxstep+1)])
        return F_T

    def force(self):
        """compute the Casimir force times 2π/ℏc in units of micrometers squared

        """
        self.forceFlag = 1
        if self.force_matsubaras is None:
            self.force_matsubaras = self.get_f_matsubaras(0)
        Ts = self.deltaT*np.arange(1, self.maxstep+1)
        Force_T = Ts*np.array([self.force_matsubaras[::i].sum()
                               for i in range(1, self.maxstep+1)])
        return Force_T

    def entropy(self):
        """compute the Casimir entropy in units of the Boltzmann-constant.

        """
        if self.f_matsubaras is None:# ??? Was ist wenn zunächst force berechnet wurde und forceFlag jetzt auf 1 steht?
            self.f_matsubaras = self.get_f_matsubaras(0)
        S_T = - np.gradient(self.free_energy(), self.deltaT)
        return S_T

    def e0(self, a=0, b=np.inf):
        """compute the Casimir T=0 energy times 2π/ℏc in units of micrometers squared
           by the evaluation of a numeric wavenumber integral with lower bound a
           and upper bound b. 

        """
        if b == None:
            b = self.nmax * self.deltaT
        F_zeroT = scipy.integrate.quad(lambda x:
                                       self.geometry.mSumme(np.array([x])), a, b)
        return np.array([[F_zeroT[0], F_zeroT[1]]])

    def get_f_matsubaras(self, nmin, schrittweite=50):# ??? Festlegung der schrittweite in free_energy, force,...
        """computes the ln det(1-M(k_n)) of the round-trip operator M(k_n) which depends 
           on the Matsubara wavenumbers k_n for nmin to self.nmax in slices of length
           schrittweite. 

        """
        f_matsubaras = slicewise_eval(lambda x: self.F_alle(self.deltaT*x),
                                      nmin, self.nmax, schrittweite)
        return f_matsubaras

    def F_tempcorr(self):
        n = 1
        beta = 1/self.deltaT
        w = 1
        freeenergy = 0
        relchange = 1
        while relchange > 1e-6:
            df, error = scipy.integrate.quad(
                lambda x: self.geometry.mSumme(x), 0, np.inf,
                weight="cos", wvar=w)
            freeenergy = freeenergy+df/np.pi
            relchange = abs(df/(np.pi*freeenergy))
            n = n+1
            w = n*beta
        return freeenergy

    def F_alle(self, matsubaras):
        """create an array with elements F(matsubaras)

        """
        if matsubaras[0] == 0:
            if not self.forceFlag and self.analytic_n0:
                F_n_0 = 0.5 * self.geometry.mSumme_k0()
                if len(matsubaras) == 1:
                    return F_n_0
                F_n_alle = self.geometry.mSumme(matsubaras[1:])
            else:
                F_n_0 = 0.5 * self.geometry.mSumme(np.array([1e-10]))
                F_n_alle = self.geometry.mSumme(matsubaras[1:])
            return np.append(F_n_0, F_n_alle)
        else:
            if self.forceFlag:
                return self.geometry.mSum_trace_dot(matsubaras)
            else:
                return self.geometry.mSumme(matsubaras)

    def l_konvergenz(self, r1, r2, L, mat1, mat2, mat3,
                     lssmax1, lssmax2, lmax1, lmax2):
        """calculate T=0 Casimir energy for two different values of
           lmax: lm_1, lm_2 for perfect conductors and returns:
           (Fm_1-Fm_2) / Fm_1 as a measure of convergence

        """
        g_lm1 = geometry.SphereSphere(r1, r2, L, mat1, mat2, mat3,
                                       lmax1, lssmax1)
        g_lm2 = geometry.SphereSphere(r1, r2, L, mat1, mat2, mat3,
                                       lmax2, lssmax2)
        f_lm1 = self.F_zero_T(g_lm1)[0]
        f_lm2 = self.F_zero_T(g_lm2)[0]
        return [(f_lm1-f_lm2) / f_lm2, f_lm1]

class oneRoundtrip(finiteT):
    def __init__(self, deltaT, nmax, Tmax, geom):
        finiteT.__init__(self, deltaT, nmax, Tmax, geom)
        self.geometry = geom
    def F_alle(self, matsubaras):
        """create an array with elements F(matsubaras)
           using roundtrips up to order = 1

        """
        if matsubaras[0] == 0.:
            if not self.forceFlag:
                F_n_0 = 0.5 * self.geometry.mSum_oneRoundtrip(np.array([1e-10]))
                F_n_alle = self.geometry.mSum_oneRoundtrip(matsubaras[1:])
            else:
                F_n_0 = 0.5 * self.geometry.mSum_trace_dot_oneRoundtrip(np.array([1e-10]))
                F_n_alle = self.geometry.mSum_trace_dot_oneRoundtrip(matsubaras[1:])
            return np.append(F_n_0, F_n_alle)
        else:
            if self.forceFlag:
                return self.geometry.mSum_trace_dot(matsubaras)
            else:
                return self.geometry.mSum_oneRoundtrip(matsubaras)


def slicewise_eval(func, nmin, nmax, stepsize=200):
    """evaluate function func for integer arguments from nmin to nmax in

       func should accept an ndarray of type int as argument.

    """
    try:
        stepsize = int(stepsize)
    except ValueError:
        raise ValueError("stepsize has to be a positive integer, got %s"
                         % stepsize)
    if not stepsize > 0:
        raise ValueError("stepsize has to be a positive integer, got %s"
                         % stepsize)
    noffset = -nmin
    result = np.empty(nmax-nmin+1)
    while nmin <= nmax:
        nmax_sl = min(nmin+stepsize-1, nmax)
        result[noffset+nmin:noffset+nmax_sl+1] = func(np.arange(nmin, nmax_sl+1))
        nmin = nmax_sl+1
    return result 

if __name__ == "__main__":
    from materials import PerfectConductor as pec, Vacuum, Gold, modifiedWater, Altern1Ps
    from materials import Gold_plasma, MercurySmith, Mercury_plasma, Mercury
    from matplotlib import pyplot as plt
    from thermodynamics import finiteT
    from materials import PerfectConductor as pec
    from geometry import SphereSphere
    import time
    mat1 = Gold
    mat2 = Vacuum
    mat3 = modifiedWater
    mat1list = [Vacuum, Gold, Altern1Ps, modifiedWater]
    r1 = 1
    r2 = 4
    L = 2.9
    k = np.append(np.array([0]), np.logspace(-3 ,0, 50))
    lmax = "automatic"
    G1 = SphereSphere(r1, r2, L, Gold, Vacuum, modifiedWater, lmax=lmax,
                      l_offset=10, lssmax="automatic", precision=1e-4, 
                      forceFlag=0, evaluation="lndet")
    print("lmax=%i" %G1.lmax)
    f = finiteT(deltaT=0.01, nmax="automatic", Tmax=2, geom=G1, analytic_n0=False)
    T = np.linspace(f.deltaT, f.Tmax, f.Tmax/f.deltaT)/0.804*293
    plt.plot(T, f.free_energy())
    plt.show()








