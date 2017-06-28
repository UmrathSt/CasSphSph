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

    def roundTrip(self, k_perp, kn, L):
        s1 = scattering.scattering(self.mat1, self.matMd)
        s2 = scattering.scattering(self.mat2, self.matMd)
        rTE1, rTM1 = s1.fresnel(k_perp, kn)
        rTE2, rTM2 = s2.fresnel(k_perp, kn)
        nMd2 = self.matMd.epsilon(kn)*self.matMd.mu(kn)
        k = sqrt(k_perp**2 + nMd2*kn**2)
        efactorTE = rTE1*rTE2 * np.exp(-2*k*L)
        efactorTM = rTM1*rTM2 * np.exp(-2*k*L)
        return log((1 - efactorTE)*(1 - efactorTM))

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

    def pressureRoundTrip(self, k_perp, kn, L):
        s1 = scattering.scattering(self.mat1, self.matMd)
        s2 = scattering.scattering(self.mat2, self.matMd)
        rTE1, rTM1 = s1.fresnel(k_perp, kn)
        rTE2, rTM2 = s2.fresnel(k_perp, kn)
        nMd2 = self.matMd.epsilon(kn)*self.matMd.mu(kn)
        k = sqrt(k_perp**2 + nMd2*kn**2)
        efactorTE = rTE1*rTE2 * np.exp(-2*k*L)
        efactorTM = rTM1*rTM2 * np.exp(-2*k*L)
        return (-k*efactorTE / (1 - efactorTE)+
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
                                      0, np.inf)[0]
        integralTM = lambda kn: quad(
                        lambda k_perp: k_perp/norm*self.roundTrips(k_perp, kn, L)[1], 
                                      0, np.inf)[0]
        f_matsubarasTE =  np.array([integralTE(matsubaras)])
        f_matsubarasTM =  np.array([integralTM(matsubaras)])
        f_matsubarasTE[0] = 0.5*f_matsubarasTE[0]
        f_matsubarasTM[0] = 0.5*f_matsubarasTM[0]
        if n0_shielding:
            f_matsubarasTE[0] = 0
            f_matsubarasTM[0] = 0
        return f_matsubarasTE+ f_matsubarasTM

    def planeplaneFreeEnergy(self, T, L, n0_shielding = False):
        """ Returns the Casimir free energy per surface area resolved
            into polarizations divided by [hbar c / 2 pi micron**3] 
            e.g. V. Esteso, S. Carretero-Palacios, H. Miguez, J. Phys. Chem. C 119 5663
        """
        if T == 0:
            return self.planeplane_e0(L, n0_shielding)
        if T == np.inf:
            return self.planeplaneFreeEnergyHT(L, n0_shielding)
        nmax = int(20/(T*L))
        matsubaras = np.append(np.array([1e-10]), np.arange(1, nmax+1)*T)
        norm = 2*np.pi
        integral = lambda kn: quad(
                        lambda k_perp: k_perp/norm*self.roundTrip(k_perp, kn, L), 
                                      0, np.inf)[0]

        f_matsubaras= T * np.array([integral(matsubara) for matsubara in matsubaras])

        f_matsubaras[0] = 0.5*f_matsubaras[0]
        if n0_shielding:
            f_matsubaras[0] = 0
        return f_matsubaras.sum()

    def planeplane_e0(self, L, n0_shielding = False):
        """ Returns the Casimir free energy per surface area resolved
            into polarizations divided by [hbar c / 2 pi micron**3] 
            e.g. V. Esteso, S. Carretero-Palacios, H. Miguez, J. Phys. Chem. C 119 5663
        """

        norm = 2*np.pi
        integral = lambda kn: quad(
                        lambda k_perp: k_perp/norm*self.roundTrip(k_perp, kn, L), 
                                      0, np.inf)[0]

        f_matsubaras = quad(integral, 1e-20, np.inf)[0]

        return f_matsubaras


    def get_f_matsubaras(self, T, L, matsubaras, n0_shielding = False):
        """ Returns the Casimir free energy per surface area resolved
            into polarizations divided by [hbar c / 2 pi micron**3] 
            e.g. V. Esteso, S. Carretero-Palacios, H. Miguez, J. Phys. Chem. C 119 5663
        """
        assert type(matsubaras) == np.ndarray
        norm = 2*np.pi
        integralTE = lambda kn: quad(
                        lambda k_perp: k_perp/norm*self.roundTrips(k_perp, kn, L)[0], 
                                      0, np.inf)[0]
        integralTM = lambda kn: quad(
                        lambda k_perp: k_perp/norm*self.roundTrips(k_perp, kn, L)[1], 
                                      0, np.inf)[0]
        f_matsubarasTE =  np.array([integralTE(matsubara) for matsubara in matsubaras])
        f_matsubarasTM =  np.array([integralTM(matsubara) for matsubara in matsubaras])
        f_matsubarasTE[0] = 0.5*f_matsubarasTE[0]
        f_matsubarasTM[0] = 0.5*f_matsubarasTM[0]
        if n0_shielding:
            f_matsubarasTE[0] = 0
            f_matsubarasTM[0] = 0
        return np.array([f_matsubarasTE, f_matsubarasTM])


    def planeplanePressure(self, T, L, n0_shielding=False):
        """ Returns the Casimir pressure contributions resolved into 
            polarizations divided by [hbar c / (2 pi micron**4)]
            at scaled temperature t = T * [ 2 pi k_b  / (hbar c)] x 1 micron
        """
            
        nmax = int(20/(T*L))
        matsubaras = np.append(np.array([1e-20]), np.arange(1, nmax+1)*T)
        norm = np.pi*2
        integral = lambda kn: quad(
             lambda k_perp: k_perp/norm*self.pressureRoundTrip(k_perp, kn, L), 
                                      0, np.inf)[0]

        p_matsubaras = T * np.array([integral(matsubara) for matsubara in matsubaras])
        p_matsubaras[0] = 0.5*p_matsubaras[0]
        if n0_shielding:
            p_matsubaras[0] = 0
       
        return p_matsubaras .sum()

 
    def spheresphereEnergy(self, r1, r2, T, d, n0_shielding = False):
        """ Returns the Casimir PFA free energy resolved into contributions of
            polarizations divided by [hbar c / (2 pi* micron)]
            at scaled temperature t = T * [ 2 pi k_b  / (hbar c)] x 1 micron
            d is the center-to-center distance
        """
        Lrho = lambda rho: self.distance(r1, r2, d, rho)
        freeE = quad(lambda rho: 2*np.pi*rho*self.planeplaneFreeEnergy(T, Lrho(rho),
                                           n0_shielding), 0, r1)[0]
        return freeE

    def spheresphereForce(self, r1, r2, T, d, n0_shielding = False):
        """ Returns the Casimir PFA Force divided by [hbar c / (2 pi* micron)]
            at scaled temperature t = T * [ 2 pi k_b  / (hbar c)] x 1 micron
            d is the center-to-center distance
        """
        Lrho = lambda rho: self.distance(r1, r2, d, rho)
        nenner = r1+r2
        if d < r1+r2:
            nenner = max(r1, r2)-min(r1, r2)
        force = 2*np.pi*r1*r2/nenner*self.planeplaneFreeEnergy(T, Lrho(0))
        return force

    def distance(self, r1, r2, d, rho):
        """ return the surface to surface distance of two spheres
            of radii r1 and r2 at center-to-center distance d,
            at a point which is at the coordinate rho in a cylindrical
            frame of reference, where rho**2 = x**2+y**2, if the 
            spheres are displaced along the z-axis
        """
        if d > r1+r2:
            delta_z = d-r1*sqrt(1-(rho/r1)**2)-r2*sqrt(1-(rho/r2)**2)   
        else:
            delta_z = r2*sqrt(1-(rho/r2)**2)-d-r1*sqrt(1-(rho/r1)**2)   
        return delta_z



if __name__ == "__main__":
    from matplotlib import pyplot as plt
    mercury = materials.MercurySmith
    water = materials.modifiedWater
    pec = materials.PerfectConductor
    ps = materials.Altern1Ps
    silicon = materials.Silicon
    gold = materials.Gold
    gold_plasma = materials.Gold_plasma
    vac = materials.Vacuum
    mat1 = silicon
    mat2 = mercury
    matMd = water
    T = 0.804

    r1, r2 = 2, 7
    d = 9.25
        #print("mat1, matMd, mat2 ", mat1.name, matMd.name, mat2.name)
    P = SphereSpherePFA(mat1, mat2, matMd)
    F = P.spheresphereForce(r1, r2, T, d)
    print("Energy(R=%.2f" %r1, F)










