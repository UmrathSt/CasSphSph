"""stratified_sphere_scattering.py provides scaled scattering
   coefficients of spheres with multiple coatings at imaginary
   wavenumbers.

"""

import numpy as np
from scattering import scattering
from scattering import coated_miescattering
import materials
import bessel_sk
import scipy.special as spec


class StratifiedMieScattering(scattering):
    """The analog of the Mie scattering coefficients for solid
       spheres but for spheres which are built out of multiple concentric
       shells of different materials.

       - mat_list: is a list of material instances consting of the core, the shell
                   and the material in which the sphere is placed from left to right.
       - r_list: is a list of radii describing the core and shell radii from
                 left to right.
    
    """
    def __init__(self, mat_list, r_list):  # nCore, nCoating, nMedium
        try:
            if type(mat_list) == list and type(r_list) == list:
                self.mat_list = mat_list
                self.r_list = r_list
                assert(len(r_list)==len(mat_list)-1)
                scattering.__init__(self, mat_list[0], mat_list[1])
        except:
            raise TypeError("Materials building up the stratified sphere "+
                            "have to be provided in a list")

    def mie_stratified(self, k, lmin, lmax):
        """calculates the scattering coefficients of a stratified sphere
           recursively.

        """
        scInnermost = scattering(self.mat_list[0], self.mat_list[1])
        inner_Mie = scInnermost.mie(k, lmin, lmax, self.r_list[0])
        if self.mat_list[-1] == self.mat_list[-2] and len(self.mat_list) >=3:
            self.mat_list.pop(-1)
            self.r_list.pop(-1)
            return self.mie_stratified(k, lmin, lmax) 
            
        for i in range(1, len(self.r_list)):
            ri, ra = self.r_list[i-1], self.r_list[i]
            mat2, mat3 = self.mat_list[i], self.mat_list[i+1]
            inner_Mie = self.mie_coated(k, lmin, lmax, ri, ra, mat2, mat3, inner_Mie)
        return inner_Mie

    def mie_coated(self, k, lmin, lmax, ri, ra, mat2, mat3, MieInner):
        """calculates the scattering coefficient of a sphere coated with a
           single concentric coating, where MieInner would be the scattering
           coefficient of the uncoated core-sphere.
           
           - k: is a numpy.nd_array of imaginary Matsubara wavenumbers
           - lmin: the minimum angular momentum of the desired coefficients
           - lmax: the maximum "        "                    "
           - ri: the core radius
           - ra: the radius of the coating (i.e. the total radius of the coated
                 sphere)
           - mat2: the shell material
           - mat3: the material surrounding the coated sphere
           - MieInner: the scattering coefficient of the core sphere if it was
                       placed in a surrounding medium mat2. 

        """
        mieTE_i, mieTM_i, inner_scale = MieInner
        eps_i, mu_i = mat2.epsilon(k), mat2.mu(k)
        ni = np.sqrt(eps_i*mu_i)
        eps_e, mu_e = mat3.epsilon(k), mat3.mu(k)
        ne = np.sqrt(eps_e*mu_e)
        kr = k*ra
        iv_e, kv_e = bessel_sk.mie_bessels(kr*ne, lmin-1, lmax)
        iv_i, kv_i = bessel_sk.mie_bessels(kr*ni, lmin-1, lmax)
        if not type(eps_i) == int:
            eps_i = eps_i[:,np.newaxis]
        if not type(eps_i) == int:
            eps_e = eps_e[:,np.newaxis]
        if not type(mu_i) == int:
            mu_i = mu_i[:,np.newaxis]
        if not type(mu_e) == int:
            mu_e = mu_e[:,np.newaxis]
        l = np.arange(lmin, lmax+1)[np.newaxis, ...]
        kr = kr[:, np.newaxis]
        ni = ni[..., np.newaxis]
        ne = ne[..., np.newaxis]
        log_scaling_factor = (iv_e - kv_e)[:, 1:]
        sign = (-1)**l
        pih = np.pi/2
        pihI = 1/pih
        exp1 = np.exp((kv_i - iv_i)[:, 1:] + inner_scale)

        id_i = -l + kr*ni*np.exp(iv_i[:,0:-1] - iv_i[:,1:])
        id_e = -l + kr*ne*np.exp(iv_e[:,0:-1] - iv_e[:,1:])
        kd_i = -l - kr*ni*np.exp(kv_i[:,0:-1] - kv_i[:,1:])
        kd_e = -l - kr*ne*np.exp(kv_e[:,0:-1] - kv_e[:,1:])

        z = (kd_i*mu_e - id_e*mu_i)*pihI*sign*exp1*mieTE_i - (id_i*mu_e - id_e*mu_i)
        n = (kd_i*mu_e - kd_e*mu_i)*pihI*sign*exp1*mieTE_i - (id_i*mu_e - kd_e*mu_i)

        mie_te = sign*pih*z/n

        z = (kd_i*eps_e - id_e*eps_i)*pihI*sign*exp1*mieTM_i - (id_i*eps_e - id_e*eps_i)
        n = (kd_i*eps_e - kd_e*eps_i)*pihI*sign*exp1*mieTM_i - (id_i*eps_e - kd_e*eps_i)

        mie_tm = sign*pih*z/n

        return np.append(np.append(mie_te[np.newaxis], mie_tm[np.newaxis],
                         axis=0), log_scaling_factor[np.newaxis], axis=0)



if __name__ == "__main__":
    from matplotlib import pyplot as plt
    m1 = materials.Vacuum
    m2 = materials.Gold
    m3 = materials.Altern1Ps
    m4 = materials.MercurySmith
    m5 = materials.modifiedWater
    lmin, lmax = 1, 3
    k = np.logspace(-8, 1.5, 1000)
    mat_list1 = [m2, m2, m5]
    mat_list2= [m2, m3, m5]

    r_list1 = [0.5, 1.015]
    r_list2 = [1, 1.015]
    C0 = StratifiedMieScattering(mat_list1, r_list1)
    C2 = StratifiedMieScattering(mat_list2, r_list2)
    Mie0 = C0.mie_stratified(k, lmin, lmax)
    Mie2 = C2.mie_stratified(k, lmin, lmax)
    pol = 0
    l = 0
    plt.semilogx(k, np.log(abs(Mie0[pol,:,l]))+Mie0[2,:,l], "r-", label="solid TE")
    plt.semilogx(k, np.log(abs(Mie2[pol,:,l]))+Mie2[2,:,l], "b--", label="coated TE")
    pol = 1
    plt.semilogx(k, np.log(abs(Mie0[pol,:,l]))+Mie0[2,:,l], "m-", label="solid TM")
    plt.semilogx(k, np.log(abs(Mie2[pol,:,l]))+Mie2[2,:,l], "c--", label="coated TM")
    plt.legend(loc="best").draw_frame(False)
#    plt.ylim([-2.2, 2.2])
    plt.show()
