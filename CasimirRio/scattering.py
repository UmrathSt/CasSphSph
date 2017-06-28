"""scattering.py provides Mie and Fresnel scattering coefficients
   for a sphere/plate made of material mat1 immersed in a medium
   mat2.

"""

import numpy as np
from scipy.special import gammaln
import bessel_sk
from helper_functions import ln_2dbl, lfac

class scattering:
    def __init__(self, mat1, mat2):
        """Mie scattering at a solid sphere of material mat1 which is 
           embedded in a medium mat2
        
        """
        if mat1.is_perfect_conductor:
            self.mie = self.miePEC
            self.fresnel = self.fresnelPEC
        if mat2.is_perfect_conductor:
            self.mie_internal = self.miePEC_internal
        self.mat1 = mat1
        self.mat2 = mat2
        self.__version__ = "SVN Revision "+str("".join(list(filter(str.isdigit, 
                                                                    "$Revision: 888 $"))))
    
    def mie_k0(self, lmin, lmax, R):
        """ calculates scaled Mie scattering coefficients
            divided by (n*R*k)^(2*l+1) in the limit of k -> 0

            Parameters
            ----------
            lmin : integer setting the minimum order of the Mie-coefficients
                   to be returned
            lmax : integer setting the maximum order of Mie-coefficients to be
                   returned
            R : float or integer specifying the radius of the sphere in microns
           
            Returns
            ------
            a numpy.nd_array of shape (3, lmax-lmin+1). On the first and second
            axis are the TE and TM Mie-coefficients whereas the last axis contains
            the logarithm of their common scaling factors.

        """
        l = np.arange(lmin, lmax+1)
        k = np.array([0])
        mu0_md = self.mat2.mu(k)
        if not self.mat2.is_sigma_finite:
            raise NotImplementedError("The material in which the scattering-sphere "+
                                      " is embedded in is supposed to be non-dissipative")
        if not self.mat2.is_dielectric:
            raise NotImplementedError("intervening medium is supposed to be a dielectric.")
        nMd = np.sqrt(self.mat2.epsilon(k)*mu0_md)        
        rMd = R * nMd
        common_ln =  -np.log(2*l+1)-2*ln_2dbl(l)
        sgn = (-1)**l
        bTM = (l+1)/l
        if not self.mat1.is_perfect_conductor:
            mu0_sp = self.mat1.mu(k)
            wPR_c = np.array([0])
            try:
                wPR_c = np.array([self.mat1.e_properties[0][0]*R/2.998e14]) 
            except IndexError:
                wPR_c = np.array([0])
            if not self.mat1.is_sigma_finite:
                ifraction = bessel_sk.i_fraction(wPR_c, l-0.5)[:, 0]
                i_frac_term = (-l + wPR_c * ifraction)
                bTE = ((mu0_md*i_frac_term - mu0_sp*(l+1))
                       /(mu0_sp*l+mu0_md*i_frac_term)
                      )
            if self.mat1.is_sigma_finite:
                bTE = -(l+1)*(mu0_sp-mu0_md)/(l*(mu0_sp+mu0_md)+mu0_md)

                if self.mat1.is_dielectric:
                    n0sp = np.sqrt(mu0_sp*self.mat1.epsilon(k))
                    bTM = (l+1) * ((mu0_md*n0sp**2 - mu0_sp*nMd**2)/
                                    (l*mu0_md*n0sp**2 + (l+1)*mu0_sp*nMd**2)
                                   )
        else:
            bTE = np.ones(l.shape)
            bTM = bTM

        return np.append(np.append((sgn*bTE)[np.newaxis], (-sgn*bTM)[np.newaxis], 
                                    axis=0,), common_ln[np.newaxis], axis=0)
        

    def mie(self, k, lmin, lmax, r):
        """Scaled TE and TM Mie coefficients for imaginary wavenumbers k 
           are returned together with the logarithm of the scaling factor.
           The scaling is necessary since Mie-coefficients are numerically
           ill-behaved at imaginary arguments. 
           k has to be a one-dimensional numpy nd.array, lmin, lmax integers
           and is generally supposed to be a double
           The non-scaled Mie-coefficients for external scattering go to zero
           like:
           (k_mat2 * r)**(2*l+1) if the kr << l
           and grow like exp(2k_mat2 r), if kr >> l 
           The problem stems from I(nu, x) and K(nu, x) which will therefore
           always be calculated by bessel_sk.py as scaled quantities healing the
           numerical problems.
           mie() returns a 3 dimentionsl numpy array, where the first axis
           contains [Mie_TE, Mie_TM, scaling] and all three items themselves
           are 2 dimensional matrices of shape (len(k), lmax-lmin+1).

        """
        eps_i, mu_i = self.mat1.epsilon(k), self.mat1.mu(k)
        ni = np.sqrt(eps_i*mu_i)
        eps_e, mu_e = self.mat2.epsilon(k), self.mat2.mu(k)
        ne = np.sqrt(eps_e*mu_e)
        kr = k*r
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
        id_i = -l + kr*ni*np.exp(iv_i[:,0:-1]- iv_i[:,1:])
        id_e = -l + kr*ne*np.exp(iv_e[:,0:-1]- iv_e[:,1:])
        kd_e = -l - kr*ne*np.exp(kv_e[:,0:-1]- kv_e[:,1:])
        log_scaling_factor = (iv_e[:, 1:]-kv_e[:, 1:])


        prefak = (-1)**l*np.pi*0.5

        mie_te = ( (mu_e*id_i - mu_i*id_e) /
                   (mu_e*id_i - mu_i*kd_e) 
                  )*prefak

        mie_tm = ( (eps_e*id_i - eps_i*id_e) /
                   (eps_e*id_i - eps_i*kd_e) 
                  )*prefak
        return np.append(np.append(mie_te[np.newaxis], mie_tm[np.newaxis],
                         axis=0), log_scaling_factor[np.newaxis], axis=0)

    def mie_internal(self, k, lmin, lmax, r):
        """Scaled TE and TM Mie coefficients for imaginary wavenumbers k 
           are returned together with the logarithm of the scaling factor.
           The scaling is necessary since Mie-coefficients are numerically
           ill-behaved at imaginary arguments. 
           k has to be a one-dimensional numpy nd.array, lmin, lmax integers
           and is generally supposed to be a double
           The non-scaled Mie-coefficients for external scattering go to zero
           like:
           (k_mat2 * r)**(2*l+1) if the kr << l
           and grow like exp(2k_mat2 r), if kr >> l 
           The problem stems from I(nu, x) and K(nu, x) which will therefore
           always be calculated by bessel_sk.py as scaled quantities healing the
           numerical problems.
           mie() returns a 3 dimentionsl numpy array, where the first axis
           contains [Mie_TE, Mie_TM, scaling] and all three items themselves
           are 2 dimensional matrices of shape (len(k), lmax-lmin+1).

        """
        eps_e, mu_e = self.mat2.epsilon(k), self.mat2.mu(k)
        ne = np.sqrt(eps_e*mu_e)
        eps_i, mu_i = self.mat1.epsilon(k), self.mat1.mu(k)
        ni = np.sqrt(eps_i*mu_i)
        kr = k*r
        iv_i, kv_i = bessel_sk.mie_bessels(kr*ni, lmin-1, lmax)
        iv_e, kv_e = bessel_sk.mie_bessels(kr*ne, lmin-1, lmax)
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
        ne = ne[..., np.newaxis]
        ni = ni[..., np.newaxis]
        id_i = -l + kr*ni*np.exp(iv_i[:,0:-1]- iv_i[:,1:])
        kd_i = -l - kr*ni*np.exp(kv_i[:,0:-1]- kv_i[:,1:])
        kd_e = -l - kr*ne*np.exp(kv_e[:,0:-1]- kv_e[:,1:])
        log_scaling_factor = (kv_i - iv_i)[:, 1:]

        prefak = (-1)**l*2/np.pi

        mie_te = ( (mu_e*kd_i - mu_i*kd_e) /
                   (mu_e*id_i - mu_i*kd_e) 
                  )*prefak

        mie_tm = ( (eps_e*kd_i - eps_i*kd_e) /
                   (eps_e*id_i - eps_i*kd_e) 
                  )*prefak
        return np.append(np.append(mie_te[np.newaxis], mie_tm[np.newaxis],
                         axis=0), log_scaling_factor[np.newaxis], axis=0)


    def mie_internal_k0(self, lmin, lmax, R):
        """ calculates scaled Mie scattering coefficients
            multiplied by k^(2*l+1) in the limit of k -> 0

            Parameters
            ----------
            lmin : integer setting the minimum order of the Mie-coefficients
                   to be returned
            lmax : integer setting the maximum order of Mie-coefficients to be
                   returned
            R : float or integer specifying the radius of the sphere in microns
           
            Returns
            ------
            a numpy.nd_array of shape (3, lmax-lmin+1). On the first and second
            axis are the TE and TM Mie-coefficients whereas the last axis contains
            the logarithm of their common scaling factors.

        """
        l = np.arange(lmin, lmax+1)
        k = np.array([0])
        mu0_i = self.mat1.mu(k)
        mu0_e = self.mat2.mu(k)
        if not self.mat1.is_sigma_finite:
            raise NotImplementedError("The interior material is supposed to"+
                                      "to be non-dissipative for internal reflections")
        if not self.mat1.is_dielectric:
            raise NotImplementedError("The material a sphere is supposed to "+
                                      "be a dielectric.")
        ni = np.sqrt(self.mat1.epsilon(k)*mu0_i)        
        common_ln =  np.log(2*l+1) + 2*ln_2dbl(l)
        sgn = (-1)**(l)
        bTM = l/(l+1)
        if not self.mat2.is_perfect_conductor:
            try:
                wPR_c = np.array([self.mat2.e_properties[0][0]*R/2.998e14]) 
            except IndexError:
                wPR_c = np.array([0])
            if not self.mat2.is_sigma_finite:
                k_wp = bessel_sk.lnknu(wPR_c, lmax)[0, :]
                k_ratio = np.exp(k_wp[0:-1]-k_wp[1:])*wPR_c/(2*l-1)#k_wp[0:-1]/k_wp[1:]*wPR_c/(2*l-1)
                k_fracterm = -l-wPR_c*k_ratio
                bTE = (-(mu0_e*l + mu0_i*k_fracterm )
                       /( (l+1)*mu0_e - mu0_i*k_fracterm )
                      )
            if self.mat2.is_sigma_finite:
                bTE = (-mu0_e+mu0_i)*l / (l*(mu0_i+mu0_e)+mu0_e)

                if self.mat2.is_dielectric:
                    eps_e = self.mat2.epsilon(k) 
                    eps_i = self.mat1.epsilon(k)
                    bTM =  ( l*(eps_e - eps_i) /
                            ((l+1)*eps_e + eps_i*l)
                           )
        else:
            bTE = np.ones(l.shape)

        return np.append(np.append((sgn*bTE)[np.newaxis], (-sgn*bTM)[np.newaxis], 
                                    axis=0,), common_ln[np.newaxis], axis=0)

    def miePEC(self, k, lmin, lmax, r):
        """Returns the limit of the general Mie-coefficients, when
           perfectly electrically conducting spheres are considered,
           which zeroize the tangential electric field components.
           The scaling is analog to mie().

        """
        ne = np.sqrt(self.mat2.epsilon(k)*self.mat2.mu(k))
        knr = ne*k*r
        iv_e, kv_e = bessel_sk.mie_bessels(knr, lmin-1, lmax)
        l = np.arange(lmin, lmax+1)[np.newaxis, ...]
        knr = knr[:, np.newaxis]
        log_scaling_factor = (iv_e - kv_e)[:, 1:]
        id_e = -l + knr*np.exp(iv_e[:,0:-1]- iv_e[:,1:])
        kd_e = -l - knr*np.exp(kv_e[:,0:-1]- kv_e[:,1:])
        prefak = (-1)**l*np.pi*0.5 # a break with the conventions of Bohren Huffman
        mieTM = prefak * id_e/kd_e
        mieTE = prefak * np.ones(mieTM.shape)
        return np.append(np.append(mieTE[np.newaxis], 
            mieTM[np.newaxis], axis=0), log_scaling_factor[np.newaxis], axis=0)

    def miePEC_internal(self, k, lmin, lmax, r):
        """Returns the limit of the general Mie-coefficients, when
           perfectly electrically conducting spheres are considered,
           which zeroize the tangential electric field components.
           The scaling is analog to mie().

        """
        ni = np.sqrt(self.mat1.epsilon(k)*self.mat1.mu(k))
        knr = ni*k*r
        iv_i, kv_i = bessel_sk.mie_bessels(knr, lmin-1, lmax)
        l = np.arange(lmin, lmax+1)[np.newaxis, ...]
        knr = knr[:, np.newaxis]
        log_scaling_factor = (kv_i - iv_i)[:, 1:]
        id_i = -l + knr*np.exp(iv_i[:,0:-1]- iv_i[:,1:])
        kd_i = -l - knr*np.exp(kv_i[:,0:-1]- kv_i[:,1:])
        prefak = (-1)**l*2/np.pi # a break with the conventions of Bohren Huffman
        mieTM = prefak * (kd_i/id_i)
        mieTE = prefak * np.ones(mieTM.shape)
        return np.append(np.append(mieTE[np.newaxis], 
            mieTM[np.newaxis], axis=0), log_scaling_factor[np.newaxis], axis=0)

    def fresnel(self, kperp, kn):
        """Fresnel-coefficients for parallel wave-number k
           and imaginary Matsubara frequency xi [1]

        References:
        [1] Milton et. al., J. Phys. A: Math. Theor. 45 374006 (2012),
            http://dx.doi.org/10.1088/1751-8113/45/37/374006
        
        """
        epsP = self.mat1.epsilon(kn)
        epsM = self.mat2.epsilon(kn)
        muP = self.mat1.mu(kn)
        muM = self.mat2.mu(kn)
        nP, nM = np.sqrt(epsP*muP), np.sqrt(epsM*muM)
        sqrtM = np.sqrt(kperp**2 + (kn*nM)**2)
        sqrtP = np.sqrt(kperp**2 + (kn*nP)**2)
        fresnelTE = (muP*sqrtM - muM*sqrtP) / (muP*sqrtM + muM*sqrtP)
        fresnelTM = (epsP*sqrtM-epsM*sqrtP) / (epsP*sqrtM+ epsM*sqrtP)
        return fresnelTE, fresnelTM

    def fresnelPEC(self, kperp, kn):
        """Limit of the Fresnel coefficients for perfect electric
           conductors.
        
        """
        return -1, 1


class coated_miescattering(scattering):
    """Mie scattering at a coated sphere with core material
       mat1 and coating mat2 immersed in a memdium mat3.
    
    """
    def __init__(self, mat1, mat2, mat3):
        scattering.__init__(self, mat1, mat2)
        self.mat1 = mat1
        self.mat2 = mat2
        self.mat3 = mat3

    def mie_coated(self, k, lmin, lmax, ri, ra):
        """Scaled scattering coefficients of a coated sphere for imaginary 
           wavenumbers k and for angular momenta lmin <= l <= lmax of a 
           coated sphere with core radius ri and outer radius ra.
           Returns a numpy.nd_array consisting of TE, TM and the natural 
           logarithm of the applied scaling of the coefficients on the zeroth
           axis. The other dimensions are len(k), len(l) of the provided
           array of Matsubara wavenumbers and angular momenta.  

        """
        mieTE_i, mieTM_i, inner_scale = self.mie(k, lmin, lmax, ri)
        eps_i, mu_i = self.mat2.epsilon(k), self.mat2.mu(k)
        ni = np.sqrt(eps_i*mu_i)
        eps_e, mu_e = self.mat3.epsilon(k), self.mat3.mu(k)
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
    import materials
    from matplotlib import pyplot as plt
    plasma = materials.Gold_plasma
    plasma.e_properties = [(1e16, 0, 0)]
    from materials import PerfectConductor as pec, Vacuum as vac
    from materials import Gold_plasma as gold, Ethanol as eth
    gold.e_properties = [(1e18, 0,0)]
    k = np.logspace(-4,0,100)
    S1 = scattering(pec, eth).mie(k, 1, 1, 1)
    plt.loglog(k, -S1[0][:,0]*np.exp(S1[2][:,0])/(k*np.sqrt(25.69))**3)
    plt.show()








