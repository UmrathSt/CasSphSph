import numpy as np
import scattering
import materials
from scipy.integrate import quad
from math import sqrt, exp, log, pi

class SphereSpherePFA:
    def __init__(self, mat1, mat2, matMd):
        self.mat1 = mat1
        self.mat2 = mat2
        self.matMd= matMd
    
    def roundTrips(self, k_perp, kn, L):
        s1 = scattering.scattering(self.mat1, self.matMd)
        s2 = scattering.scattering(self.mat2, self.matMd)
        rTE1, rTM1 = s1.fresnel(k_perp, kn)
        rTE2, rTM2 = s2.fresnel(k_perp, kn)
        nMd2 = self.matMd.epsilon(kn)*self.matMd.mu(kn)
        k = sqrt(k_perp**2 + nMd2*kn**2)
        efactorTE = rTE1*rTE2 * np.exp(-2*k*L)
        efactorTM = rTM1*rTM2 * np.exp(-2*k*L)
        return (log(1 - efactorTE), log(1 - efactorTM))

    def pressureRoundTrips(self, k_perp, kn, L):
        s1 = scattering.scattering(self.mat1, self.matMd)
        s2 = scattering.scattering(self.mat2, self.matMd)
        rTE1, rTM1 = s1.fresnel(k_perp, kn)
        rTE2, rTM2 = s2.fresnel(k_perp, kn)
        nMd2 = self.matMd.epsilon(kn)*self.matMd.mu(kn)
        k = sqrt(k_perp**2 + nMd2*kn**2)
        efactorTE = rTE1*rTE2 * np.exp(-2*k*L)
        efactorTM = rTM1*rTM2 * np.exp(-2*k*L)
        return (-k*efactorTE / (1 - efactorTE), 
                -k*efactorTM / (1 - efactorTM)
               )

    def planeplaneFreeEnergyHT(self, L, n0_shielding = False):
        """ Returns the high-temperature Casimir free energy divided by T per surface area resolved
            into polarizations divided by [hbar c / 2 pi micron**3] 
            e.g. V. Esteso, S. Carretero-Palacios, H. Miguez, J. Phys. Chem. C 119 5663
        """
        matsubaras = np.array([1e-20])
        norm = 2*np.pi
        integralTE = lambda kn: quad(
                        lambda k_perp: k_perp/norm*self.roundTrips(k_perp, kn, L)[0], 
                                      1e-10, np.inf)[0]
        integralTM = lambda kn: quad(
                        lambda k_perp: k_perp/norm*self.roundTrips(k_perp, kn, L)[1], 
                                      1e-10, np.inf)[0]
        f_matsubarasTE =  np.array([integralTE(matsubaras)])
        f_matsubarasTM =  np.array([integralTM(matsubaras)])
        f_matsubarasTE[0] = 0.5*f_matsubarasTE[0]
        f_matsubarasTM[0] = 0.5*f_matsubarasTM[0]
        if n0_shielding:
            f_matsubarasTE[0] = 0
            f_matsubarasTM[0] = 0
        return f_matsubarasTE+f_matsubarasTM

    def planeplaneFreeEnergy(self, T, L, n0_shielding = False):
        """ Returns the Casimir free energy per surface area resolved
            into polarizations divided by [hbar c / 2 pi micron**3] 
            e.g. V. Esteso, S. Carretero-Palacios, H. Miguez, J. Phys. Chem. C 119 5663
        """
        if T == 0:
            return self.planeplane_e0(L, n0_shielding)
        nmax = int(100/(T*L))
        matsubaras = np.append(np.array([1e-10]), np.arange(1, nmax+1)*T)
        norm = 2*np.pi
        integralTE = lambda kn: quad(
                        lambda k_perp: k_perp/norm*self.roundTrips(k_perp, kn, L)[0], 
                                      0, np.inf)[0]
        integralTM = lambda kn: quad(
                        lambda k_perp: k_perp/norm*self.roundTrips(k_perp, kn, L)[1], 
                                      0, np.inf)[0]
        f_matsubarasTE = T * np.array([integralTE(matsubara) for matsubara in matsubaras])
        f_matsubarasTM = T * np.array([integralTM(matsubara) for matsubara in matsubaras])
        f_matsubarasTE[0] = 0.5*f_matsubarasTE[0]
        f_matsubarasTM[0] = 0.5*f_matsubarasTM[0]
        if n0_shielding:
            f_matsubarasTE[0] = 0
            f_matsubarasTM[0] = 0
        return f_matsubarasTE.sum()+f_matsubarasTM.sum()

    def planeplane_e0(self, L, n0_shielding = False):
        """ Returns the Casimir free energy per surface area resolved
            into polarizations divided by [hbar c / 2 pi micron**3] 
            e.g. V. Esteso, S. Carretero-Palacios, H. Miguez, J. Phys. Chem. C 119 5663
        """

        norm = 2*np.pi
        integralTE = lambda kn: quad(
                        lambda k_perp: k_perp/norm*self.roundTrips(k_perp, kn, L)[0], 
                                      0, np.inf)[0]
        integralTM = lambda kn: quad(
                        lambda k_perp: k_perp/norm*self.roundTrips(k_perp, kn, L)[1], 
                                      0, np.inf)[0]
        f_matsubarasTE = quad(integralTE, 1e-20, np.inf)[0]
        f_matsubarasTM = quad(integralTM, 1e-20, np.inf)[0]

        return f_matsubarasTE+f_matsubarasTM


    def get_f_matsubaras(self, T, L, matsubaras, n0_shielding = False):
        """ Returns the Casimir free energy per surface area resolved
            into polarizations divided by [hbar c / 2 pi micron**3] 
            e.g. V. Esteso, S. Carretero-Palacios, H. Miguez, J. Phys. Chem. C 119 5663
        """
        assert type(matsubaras) == np.ndarray
        if matsubaras[0] == 0:
            matsubaras[0] = 1e-20
        norm = 2*np.pi
        integralTE = lambda kn: quad(
                        lambda k_perp: k_perp/norm*self.roundTrips(k_perp, kn, L)[0], 
                                      0, np.inf)[0]
        integralTM = lambda kn: quad(
                        lambda k_perp: k_perp/norm*self.roundTrips(k_perp, kn, L)[1], 
                                      0, np.inf)[0]
        f_matsubarasTE =  np.array([integralTE(matsubara) for matsubara in matsubaras])
        f_matsubarasTM =  np.array([integralTM(matsubara) for matsubara in matsubaras])
        f_matsubarasTE[0] = f_matsubarasTE[0]
        f_matsubarasTM[0] = f_matsubarasTM[0]
        if n0_shielding:
            f_matsubarasTE[0] = 0
            f_matsubarasTM[0] = 0
        return f_matsubarasTE+ f_matsubarasTM


    def planeplanePressure(self, T, L, n0_shielding=False):
        """ Returns the Casimir pressure contributions resolved into 
            polarizations divided by [hbar c / (2 pi micron**4)]
            at scaled temperature t = T * [ 2 pi k_b  / (hbar c)] x 1 micron
        """
            
        nmax = int(100/(T*L))
        matsubaras = np.append(np.array([1e-20]), np.arange(1, nmax+1)*T)
        norm = np.pi*2
        integralTE = lambda kn: quad(
             lambda k_perp: k_perp/norm*self.pressureRoundTrips(k_perp, kn, L)[0], 
                                      0, np.inf)[0]
        integralTM = lambda kn: quad(
             lambda k_perp: k_perp/norm*self.pressureRoundTrips(k_perp, kn, L)[1], 
                                      0, np.inf)[0]
        p_matsubarasTE = T * np.array([integralTE(matsubara) for matsubara in matsubaras])
        p_matsubarasTM = T * np.array([integralTM(matsubara) for matsubara in matsubaras])
        p_matsubarasTE[0] = 0.5*p_matsubarasTE[0]
        p_matsubarasTM[0] = 0.5*p_matsubarasTM[0]
        if n0_shielding:
            p_matsubarasTE[0] = 0
            p_matsubarasTM[0] = 0         
        return p_matsubarasTE.sum()+ p_matsubarasTM.sum()

    def spheresphereForce(self, r1, r2, T, L, n0_shielding = False, ext=True):
        """ Returns the Casimir PFA force resolved into contributions of
            polarizations divided by [hbar c / (2 pi micron**2)]
            at scaled temperature t = T * [ 2 pi k_b  / (hbar c)] x 1 micron
            Consistent with Emig Phys. Rev. A 82, 052507, 2010
        """
        pref = 2*np.pi*(r1*r2)/(r1+r2)
        if not ext:
            pref = 2*np.pi*(r1*r2)/(r2-r1)
        if T == 0:
            return pref * self.planeplane_e0(L, n0_shielding)
        if T == np.inf:
            return pref * self.planeplaneFreeEnergyHT(L, n0_shielding)
        return pref* self.planeplaneFreeEnergy(T, L, n0_shielding)
 
    def spheresphereEnergy(self, r1, r2, T, L, n0_shielding = False, ext=True):
        """ Returns the Casimir PFA free energy resolved into contributions of
            polarizations divided by [hbar c / (2 pi* micron)]
            at scaled temperature t = T * [ 2 pi k_b  / (hbar c)] x 1 micron
        """
        pref = L*np.pi*r1*r2/(r1+r2)
        if not ext:
            pref = L*np.pi*r1*r2/(r2-r1)

        if T == 0:
            return pref * self.planeplane_e0(L, n0_shielding)
        if T == np.inf:
            return pref * self.planeplaneFreeEnergyHT(L, n0_shielding)
        return pref* self.planeplaneFreeEnergy(T, L, n0_shielding)



if __name__ == "__main__":
    from matplotlib import pyplot as plt
    pec = materials.PerfectConductor
    vac = materials.Vacuum
    gold = materials.Gold
    water = materials.modifiedWater
    mat1 = pec
    mat2 = pec
    matMd = vac
    r1, r2, L = 50, 50, 0.3
    rs = [50]
    T = 0.804
#    print("mat1, matMd, mat2: ", mat1.name, matMd.name, mat2.name)
    for r in rs:
        r1, r2 = r, r
        L = 1
        P = SphereSpherePFA(mat1, mat2, matMd)
        F = P.spheresphereEnergy(r1, r2, T, L)
        print("KK R=%.2f FreeE=%.17f" %(r1, F))










