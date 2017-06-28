"""translation.py provides translation coefficients for electromagnetic
   multipole fieldes connecting electromagnetic fields in frames of 
   reference shifted a distance L along the z-axis.

"""

import numpy as np
import materials
from cython_gaunt import gaunt
import bessel_sk
import scipy.special as spec
from math import lgamma
from helper_functions import lfac, ln_2dbl



class translation:
    """provides translation coefficients describing vector spherical waves (VSW)
       characterized by "angular momentum" numbers (l,m) and imaginary frequncy
       k in one frame of reference (A) in another frame of reference (B), which 
       is shifted a distance L with respect to (A).  
       - lmin: the minimum angular momentum in frame (A)
       - lmax: the maximum   "        "     "   "     "
       - lssmax: the maximum "        "     "   "    (B)
       - m: the z-projection of the angular momentum l
       - material: a material instance specifying the medium which separates
                   the origin of (A) and (B)
       - L: the absolute distance of the two frames of reference in microns
       - k: a numpy_nd array of imaginary Matsubra wavenumbers for which the 
            translation coefficients shall be calculated.
       - ext_flag: an integer specifying whether or not the origin of (B)
                   is contained in the region, where the field is expanded 
                   (ext_flag=0) or expcluded (ext_flag=1)
       
    """
    def __init__(self, lmax, lssmax, m, material, L, k, ext_flag,
                 forceFlag=0):
        self.__version__ = "SVN Revision "+str("".join(list(filter(str.isdigit, 
                                                              "$Revision: 703 $"))))
        self.lmin = max(1, m)
        self.lmax = lmax
        self.lssmax = lssmax
        self.m = m
        self.L = L# in microns
        self.medium = material
        self.k = k
        self.ext_flag = ext_flag
        if ext_flag:
            self.trans_k0 = self.trans_k0_ext
        else:
            self.trans_k0 = self.trans_k0_int
        if not (k == 0).any():
            self.bessels = bessel_sk.trans_bessels(self.k*self.L*
                 np.sqrt(self.medium.epsilon(self.k)), self.lmax+self.lssmax+1, self.ext_flag)
        self.forceFlag = forceFlag
        if self.forceFlag:
            self.get_vl1l2 = self.get_vl1l2_force

    def gnt(self, l1, l2, m):
        """Returns Gaunt coefficients for all nonzero l' in the range
           l' = |l1-l2|..l1+l2 for the z-projection of l1 being -m and the
           z-projection of l2 beeing m. Every second coefficient is zero and
            not returned.
        """
        return gaunt(l1, l2, m)

    def get_scaled_bessels(self, l1, l2, lpp):
        """ Since the modified bessel functions (I(nu, x), K(nu, x)) are 
            calculated only as ln(I) and ln(K) they have to be scaled
            for the largest value of lpp which is: 
            lpp = l1+l2 if ext_flag = 1
            lpp = |l1-l2| if ext_flag = 0

        """
        sphB = self.bessels[:, lpp.flatten()]
        norm = sphB[:, 0:1]
        if self.ext_flag:
            norm = sphB[:,-1:]

        sphB = np.exp(sphB - norm) 

        skalierung = norm[:, 0]

        return [sphB, skalierung]
    
    def trans_k0_int(self, l1, l2):
        """ Polarization conserving translation coefficients divided by
            (kd)**(|l1-l2|) and multiplied by (-+1j)**|l1-l2|*(-)**|l1-l2|

        """
        m = self.m
        sign = 1
        assert len(l1.shape) == 2 and len(l2.shape) == 2
        sh = (l1.shape[0], l2.shape[1])
        ones = np.ones(sh)
        l1, l2 = l1*ones, l2*ones
        lg = l1.copy()
        ls = l1.copy()
        l2gl1 = l2>l1
        nl2gl1 = np.logical_not(l2gl1)
        lg[l2gl1] = l2[l2gl1]
        ls[nl2gl1] = l2[nl2gl1]
        ln1 = 0.5*(
            ( lfac(lg+m) + lfac(lg-m) + np.log(2*lg+1) + np.log(2*ls+1) + np.log(ls*(lg+1))
             -lfac(ls+m) - lfac(ls-m) - np.log(lg*(ls+1))
            )
                    )
        ln2 = ( lfac(2*(lg-ls)+1) + lfac(2*ls) + lfac(lg)
               -ln_2dbl(lg-ls+1) - lfac(2*lg+1) - 2*lfac(lg-ls) - lfac(ls)
              )

        ln = ln1+ln2
        return [sign*ones, ln]
        

    def trans_k0_ext(self, l1, l2):
        """" Polarization conserving translation coefficients multiplied by 
             (kd)**(l1+l2+1) * (+-1j)**(l1-l2) where (+-) refers
             to translations in either positive or negative z direction 

        """
        m = self.m
        sign = (-1)**m
        ln_z1= (0.5*(np.log(2*l1+1)+np.log(2*l2+1)
                  -(np.log(l1*(l1+1))+np.log(l2*(l2+1)))
                      )+np.log(2*l1+2*l2+1)+ np.log(l1)+np.log(l2))
        ln_z2 = 2*lfac(l1+l2) + lfac(2*l2) + lfac(2*l1) + ln_2dbl(l1+l2)
        ln_n = (0.5*(lfac(l1-m)+lfac(l2-m)+lfac(l1+m)+lfac(l2+m))+
                    lfac(2*l1+2*l2+1)+lfac(l1)+lfac(l2)
               )
        ln = ln_z1+ln_z2-ln_n
        return [sign*np.ones(ln.shape), ln]

    def trans_ab(self):
        """ Returns a list of 4 scaled translation coefficients together with 
            the natural logarithm of the scaling factor. The latter is the same 
            for all translation coefficients.
            vEEab = vMMab is the translation of transverse electric vector 
            multipoles e. g. transverse magnetic vector multipoles due to 
            translation along the z-axis.
            VEMab = VMEab are the mode mixing coefficients which describe the 
            transverse magnetic/electric character of a multipole field in 
            frame of reference (B) which was incidentially electric/magnetic
            in the frame of reference (A).
            
        """
        l1 = np.arange(self.lmin, self.lmax+1)
        l1_len = l1.shape[0]
        l2 = np.arange(self.lmin, self.lssmax+1)
        l2_len = l2.shape[0]
        if len(self.k) == 1 and self.k[0] == 0:
            l1 = l1[:,np.newaxis]
            l2 = l2[np.newaxis,:]
            return self.trans_k0(l1, l2)
        sh_tupel = (len(self.k), l1_len, l2_len)
        vEEab = np.empty(sh_tupel)
        vEEba = np.empty(sh_tupel)
        vEMab = np.empty(sh_tupel)
        vEMba = np.empty(sh_tupel)
        skalierung = np.empty(sh_tupel)
        for i in range(0, l1_len):
            for j in range(0, l2_len):
                tmp = self.get_vl1l2(l1[i], l2[j])
                vEEab[:, i, j] = tmp[0]
                vEEba[:, i, j] = tmp[1]
                vEMab[:, i, j] = tmp[2]
                vEMba[:, i, j] = tmp[3]
                skalierung[:, i, j] = tmp[4]
        return [vEEab, vEEba, vEMab, vEMba, skalierung]


    def get_vl1l2(self, l1, l2):
        """ This method provides scaled translation coefficients for VSW of order "l1" in
            the frame of reference (A) to VSW of order "l2" frame of reference "B"
            shifted a distance "L" along the z-axis.
            If the translation was for regular -> regular VSW the scaling factor is
            ln((2/x)**nu*Gamma(nu+1)*exp(-2*x/3)) with nu = |l1-l2|+1/2, x = kd
            For regular -> outgoing the scaling is:
            ln((x/2)**nu/Gamma(nu)*exp(2*x/3)) with nu = l1+l2+1/2, x = kd
        """
        lppmax = l1+l2
        lpp = np.arange(abs(l1-l2), lppmax+1, 2)[np.newaxis, :]
        gaunt = self.gnt(l1, l2, self.m)
        nL = (self.k*self.L*np.sqrt(self.medium.epsilon(self.k)))[:, np.newaxis]
        sphB, skalierung = self.get_scaled_bessels(l1, l2, lpp)
        sphB_gnt = sphB * gaunt
        sgn_ba = (-1)**lpp
        sgn_int_ext = (-1)**(lpp-(lpp-1)*self.ext_flag)
        pi = np.pi
        m = self.m
        l1p = l1*(l1+1)
        l2p = l2*(l2+1)
        common_ab = (sgn_int_ext * (-1)**m * np.sqrt(pi) * sphB_gnt
                     * np.sqrt((2*lpp+1) / (l1p*l2p)))*(2/np.pi)**(self.ext_flag)
        vEE_ab = (l1p + l2p -lpp*(lpp+1)) * common_ab
        vEM_ab = 2 * self.m * nL * common_ab
        vEE_ba = vEE_ab * sgn_ba
        vEM_ba = - vEM_ab * sgn_ba
        return [vEE_ab.sum(axis=-1), vEE_ba.sum(axis=-1),
                vEM_ab.sum(axis=-1), vEM_ba.sum(axis=-1), skalierung]


