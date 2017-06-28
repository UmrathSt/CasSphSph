"""geometry.py provides sphere-sphere specific functionality
   for the description of the Casimir round-trip operator at
   imaginary frequencies for two solid or coated spheres.

"""
from distutils.version import StrictVersion
import numpy as np
if StrictVersion('1.8.0') > StrictVersion(np.__version__):
    raise ImportError("NumPy version needs to be at least 1.8.0")
from translation import translation
from scattering import scattering, coated_miescattering
from stratified_sphere_scattering import StratifiedMieScattering as strat_scat
import numexpr as ne
#from scale import balance
from matrix_balancing import balance



class SphereSphere:
    """provide Mie coefficients and elements for switching reference frames in
       respective geometry:

       - r1, r2: radii of the two spheres
       - L: distance of sphere centers
       - lmax >= 1:  maximum angular momentum for scattering
       - lssmax >=lmax: maximum  "       "    for translation
       - mat1, mat2, mat3 are the materials of sphere1, sphere2 and the medium
         in between
       - if lmax and/or lssmax are set to "automatic" the function get_lmax()
         is called with geometry parameters (r1, r2, L) and (precison, l_offset)
         such that the specified precision is approximately obtained.
       - l_offset specifies a manual offset which is added on top of the 
         automatically determined maximum angular momentum. It is intended for
         the use at rather larged distances L > 2*min(r1, r2), where get_lmax()
         might underestimate the necessry maximum angular momentum.
       * internal |r1-r2| < L       and
       * external  r1+r2  > L
       geometry

       # ??? Was ist für den Benutzer vorgesehen, nur roundTrip_elements,
       #     mSumme und mSum_trace_det?

    """
    def __init__(self, r1, r2, L, mat1, mat2, mat3, lmax="automatic",
                 l_offset=4, lssmax="automatic", precision=1e-4, 
                 forceFlag=0, evaluation="lndet"):
        self.__version__ = "SVN Revision "+str("".join(list(filter(str.isdigit, 
                                                                "$Revision: 740 $"))))
        self.r1 = r1
        self.r2 = r2
        self.L = L
        self.K = None
        self.mat1 = mat1
        self.mat2 = mat2
        self.mat3 = mat3
        self.lmax = lmax
        self.lssmax = lssmax
        self.forceFlag = forceFlag
        self.ext_flag = 0
        self.l_offset = l_offset
        self.roundTrip_dct = {}
        self.evaluation = evaluation
        self.precision = precision
        self.conv = 0.9
        if evaluation == "trace":
            self.mSumme = self.mSum_roundtrips
        if not evaluation == "trace" and not evaluation == "lndet":
            raise ValueError("Type of evaluation must be either lndet or \n"
                            +"trace")       
        if type(r1) == list:
            r1 = r1[-1]
        if type(r2) == list:
            r2 = r2[-1]
        if lmax == "automatic":
            self.lmax = get_lmax(r1, r2, L, precision, self.l_offset)
        if lssmax == "automatic":
            self.lssmax = int(max(r1,r2)/min(r1,r2)*self.lmax)

        if self.L > r1 + r2:
            self.ext_flag = 1.
            self.blockmatrices_k0 = self.blockmatrices_k0_ext
        else: 
            self.blockmatrices_k0 = self.blockmatrices_k0_int

    def roundTrip_elements(self, m, k):
        """Collect all necessary Mie-coefficients and translation matrices
           to describe the round-trip block-matrix in a subspace of m, the 
           z-projection of the angular momentum l in a spherical multipole 
           basis for a numpy array of wavenumbers k.

         """
        lmin = max(1, m)
        ext_VZ = -1+2*self.ext_flag
        if np.shape(self.K) == np.shape(k) and (self.K == k).all():
            mieA_1_lmax, mieB_1_lmax = self.MieA, self.MieB
            V = translation(self.lmax, self.lssmax, m, self.mat3, self.L, k,
                            self.ext_flag, 0).trans_ab()
            V = [V[0][:, :, np.newaxis, ...], V[1][:, np.newaxis, ...],
                 V[2][:, :, np.newaxis, ...], V[3][:, np.newaxis, ...], V[4]]
            if self.forceFlag:
                Vdiff = translation(self.lmax, self.lssmax, m, self.mat3,
                                    self.L, k, self.ext_flag, self.forceFlag
                                    ).trans_ab()
                Vdiff = [Vdiff[0][:, :, np.newaxis, ...],
                         Vdiff[1][:, np.newaxis, ...],
                         Vdiff[2][:, :, np.newaxis, ...],
                         Vdiff[3][:, np.newaxis, ...]
                         ]

                V.extend(Vdiff)
            return [mieA_1_lmax[:, :, lmin-1::, :, :], mieB_1_lmax[:, :, :, :, lmin-1::], V]

            V = translation(self.lmax, self.lssmax, m, self.mat3, self.L, k,
                            self.ext_flag, 0).trans_ab()
            V = [V[0][:, :, np.newaxis, ...], V[1][:, np.newaxis, ...],
                 V[2][:, :, np.newaxis, ...], V[3][:, np.newaxis, ...], V[4]]
            if self.forceFlag:
                Vdiff = translation(self.lmax, self.lssmax, m, self.mat3,
                                    self.L, k, self.ext_flag, self.forceFlag
                                    ).trans_ab()
                Vdiff = [Vdiff[0][:, :, np.newaxis, ...],
                         Vdiff[1][:, np.newaxis, ...],
                         Vdiff[2][:, :, np.newaxis, ...],
                         Vdiff[3][:, np.newaxis, ...]
                         ]

                V.extend(Vdiff)
            return [mieA[:, :, lmin-1::, :, :], mieB[:, :, :, :, lmin-1::], V]
        else:
            if type(self.mat1)== list:
                self.MieA = strat_scat(self.mat1, self.r1
                 ).mie_stratified(k, 1, self.lmax)[..., np.newaxis, np.newaxis]
            else:
                self.MieA = scattering(self.mat1, self.mat3
                   ).mie(k, 1, self.lmax, self.r1)[..., np.newaxis, np.newaxis]
            if type(self.mat2) == list:
                st = strat_scat(self.mat2, self.r2
                 ).mie_stratified(k, 1, self.lssmax)
                self.MieB = strat_scat(self.mat2, self.r2
                 ).mie_stratified(k, 1, self.lssmax)
            else:
                if not self.ext_flag:
                    self.MieB = scattering(self.mat3, self.mat2
                     ).mie_internal(k, 1, self.lssmax, self.r2)
                else:
                    self.MieB = scattering(self.mat2, self.mat3
                     ).mie(k, 1, self.lssmax, self.r2)
  
            self.MieB = np.append(self.MieB[0:2][:, :, np.newaxis,
                np.newaxis, :], self.MieB[2][np.newaxis, :, np.newaxis,
                                               np.newaxis, :],axis=0)          
            mieA = self.MieA
            mieB = self.MieB
            self.K = k.copy()
        V = translation(self.lmax, self.lssmax, m, self.mat3, self.L, k,
                        self.ext_flag, 0).trans_ab()
        V = [V[0][:, :, np.newaxis, ...], V[1][:, np.newaxis, ...],
             V[2][:, :, np.newaxis, ...], V[3][:, np.newaxis, ...], V[4]]
        if self.forceFlag:
            Vdiff = translation(self.lmax, self.lssmax, m, self.mat3,
                                self.L, k, self.ext_flag, self.forceFlag
                                ).trans_ab()
            Vdiff = [Vdiff[0][:, :, np.newaxis, ...],
                     Vdiff[1][:, np.newaxis, ...],
                     Vdiff[2][:, :, np.newaxis, ...],
                     Vdiff[3][:, np.newaxis, ...]
                     ]

            V.extend(Vdiff)
        return [mieA, mieB, V]

    def roundTrip_elements_k0(self, m, k):
        """Collect all necessyary Mie-coefficients and translation matrices
           to describe the round-trip block-matrix in a subspace of m, the 
           z-projection of the angular momentum l in a spherical multipole 
           basis at zero frequency.

         """
        if not type(k)==np.ndarray and len(k) == 1 and k[0] == 0:
            raise ValueError("k has to be np.array([0]), given %s" %(str(k)))
        lmin = max(1, m)
        ext_VZ = -1+2*self.ext_flag
        try:
            mieA_k0 = self.MieA_k0
            mieB_k0 = self.MieB_k0
            V_k0 = translation(self.lmax, self.lssmax, m, self.mat3, self.L, k,
                            self.ext_flag, 0).trans_ab()
            V_k0 = [V_k0[0][:, np.newaxis, :], V_k0[0][:, np.newaxis,:].transpose(1,2,0),
                 V_k0[1][:, np.newaxis, :]]
            if self.forceFlag:
                raise NotImplementedError("Force evaluation of zero frequency "+
                                          "is not implemented, yet.")

                V.extend(Vdiff)
            return [mieA_k0[:, lmin-1::, :, :], mieB_k0[:, :, :, lmin-1::], V_k0]

        except AttributeError:
            if type(self.mat1)==list or type(self.mat2)==list:
                raise NotImplementedError("Evaluation of the zeroth Matsubara frequency"+
                                          " is not yet implemented for stratified spheres")
            self.MieA_k0 = scattering(self.mat1, self.mat3
                                   ).mie_k0(lmin, self.lmax, self.r1
                                         )[:, :, np.newaxis, np.newaxis]
            if not self.ext_flag:
                self.MieB_k0 = scattering(self.mat3, self.mat2
                    ).mie_internal_k0(lmin, self.lssmax, self.r2)[:, np.newaxis, np.newaxis,:]
            else:
                self.MieB_k0 = scattering(self.mat2, self.mat3
                             ).mie_k0(lmin, self.lssmax, self.r2)[:,np.newaxis, np.newaxis,:]
            mieA_k0 = self.MieA_k0
            mieB_k0 = self.MieB_k0
        V_k0 = translation(self.lmax, self.lssmax, m, self.mat3, self.L, k,
                        self.ext_flag, 0).trans_ab()
        V_k0 = [V_k0[0][:, np.newaxis, :], V_k0[0][:, np.newaxis,:].transpose(1,2,0),
             V_k0[1][:, np.newaxis, :]]
        if self.forceFlag:
            raise NotImplementedError("Force evaluation of zero frequency "+
                                      "is not implemented, yet.")

            V.extend(Vdiff)
        return [mieA_k0, mieB_k0, V_k0]

    def blockmatrices_k0_int(self, m):
        """ Returns a list of two blockmatrices corresponding to a subspace
            of m at zero frequency

        """
        MieA, MieB, V = self.roundTrip_elements_k0(m, np.array([0]))
        r1, r2, L = self.r1, self.r2, self.L
        lmin = max(m, 1)
        l1 = np.arange(lmin, self.lmax+1)[:, np.newaxis, np.newaxis]
        l2 = l1.transpose(1,0,2)
        lss = np.arange(lmin, self.lssmax+1)[np.newaxis, np.newaxis, :]
        sgn = (-1)**np.float(lss-l1)        
        a_TEsgn, a_TMsgn, a_ln = MieA
        b_TEsgn, b_TMsgn, b_ln = MieB
        distance_dep = np.log(r1/r2) + l1*np.log(r1/L)+2*lss*np.log(L/r2) + l2*np.log(r1/L)
        vab_sgn, vba_sgn, v_ln = V
        TMsgn = sgn * vab_sgn * vba_sgn.transpose(0,2,1) * a_TMsgn * b_TMsgn
        TEsgn = sgn * vab_sgn * vba_sgn.transpose(0,2,1) * a_TEsgn * b_TEsgn
        ln_TMmag = a_ln+b_ln+v_ln.transpose(1,0,2)+v_ln+distance_dep
        ln_TEmag = a_ln+b_ln+v_ln.transpose(1,0,2)+v_ln+distance_dep
        condition = np.logical_not(np.logical_and(lss>=l1, lss>=l2))

        if ln_TMmag.shape[0] > 1:
            TMmag = np.exp(single_k_scale(ln_TMmag))
            TEmag = np.exp(single_k_scale(ln_TEmag))
        else:
            TMmag = np.exp(ln_TMmag)
            TEmag = np.exp(ln_TEmag)
        TEmag[condition], TMmag[condition] = 0, 0
        return [(TEsgn*TEmag).sum(axis=-1), (TMsgn*TMmag).sum(axis=-1)]

    def blockmatrices_k0_ext(self, m):
        """ Returns a list of two blockmatrices corresponding to a subspace
            of m at zero frequency

        """

        MieA, MieB, V = self.roundTrip_elements_k0(m, np.array([0]))
        r1, r2, L = self.r1, self.r2, self.L
        lmin = max(m, 1)
        l1 = np.arange(lmin, self.lmax+1)[:, np.newaxis, np.newaxis]
        l2 = l1.transpose(1,0,2)
        lss = np.arange(lmin, self.lssmax+1, dtype = np.float)[np.newaxis, np.newaxis, :]
        sgn = (-1)**(lss-l2)        
        a_TEsgn, a_TMsgn, a_ln = MieA
        b_TEsgn, b_TMsgn, b_ln = MieB
        distance_dep = (l1+l2+1)*np.log(r1/L)+(2*lss+1)*np.log(r2/L)
        vab_sgn, vba_sgn, v_ln = V
        TMsgn = sgn * vab_sgn * vba_sgn.transpose(0,2,1) * a_TMsgn * b_TMsgn
        TEsgn = sgn * vab_sgn * vba_sgn.transpose(0,2,1) * a_TEsgn * b_TEsgn
        ln_TMmag = a_ln+b_ln+v_ln+v_ln.transpose(1,0,2)+distance_dep
        ln_TEmag = a_ln+b_ln+v_ln+v_ln.transpose(1,0,2)+distance_dep
        if ln_TMmag.shape[0] > 1:
            TMmag = np.exp(single_k_scale(ln_TMmag))
            TEmag = np.exp(single_k_scale(ln_TEmag))
        else:
            TMmag = np.exp(ln_TMmag)
            TEmag = np.exp(ln_TEmag)
        return [(TEsgn*TEmag).sum(axis=-1), (TMsgn*TMmag).sum(axis=-1)]

    def blockmatrices(self, m, matsubaras):
        """Returns a list of matrices of dimension 
           (len(matsubaras), (lmax-max(m-1,0),lmax-max(m-1,0))
           which correspond to the entries of the round-trip matrix 
           M in the subspace of lz = m: M(m) = [[M_EE, MEM],[M_ME, M_MM]],
           for a numpy array of matsubara wavenumbers.

        """
        k0_evaluation = False
        if matsubaras[0] == 0:
            k0_evaluation = True
            matsubaras = matsubaras[1:]
        MieA, MieB, V = self.roundTrip_elements(m, matsubaras)
        V[4] = V[4][..., np.newaxis]
        dct = {"aTE": MieA[0], "V0": V[0], "bTE": MieB[0], "V1": V[1],
               "aTM": MieA[1], "V2": V[2], "bTM": MieB[1], "V3": V[3],
               "V4": V[4].transpose(0, 1, 3, 2),
               "V4t": V[4].transpose(0, 3, 1, 2), "a2": MieA[2],
               "b2": MieB[2], "V01": 1, "V23": 1, "V03": 1, "V21": 1}
        log_skalierung = ne.evaluate("a2+b2+V4+V4t", dct)
        skal = dynamic_scaling(log_skalierung)
        dct["skal"] = skal
        if self.forceFlag:
            if m == 0:
                V83 = 0
                V72 = 0
            else:
                V83 = V[8]/V[3]
                V72 = V[7]/V[2]
            dct["V01"] = V[5]/V[0] + V[6]/V[1]
            dct["V23"] = V72 + V83
            dct["V03"] = V[5]/V[0] + V83
        mEE = ne.evaluate("skal * aTE*(V0*bTE*V1*V01\
                                       + V2*bTM*V3*V23)", dct).sum(axis=3)
        mEM = ne.evaluate("skal * aTE*(V0*bTE*V3*V03\
                                       + V2*bTM*V1*V21)", dct).sum(axis=3)
        mME = ne.evaluate("skal * aTM*(V2*bTE*V1*V21\
                                       + V0*bTM*V3*V03)", dct).sum(axis=3)
        mMM = ne.evaluate("skal * aTM*(V2*bTE*V3*V23\
                                       + V0*bTM*V1*V01)", dct).sum(axis=3)
        if k0_evaluation:
            mEE_k0, mMM_k0 = self.blockmatrices_k0(m)
            mEE_k0, mMM_k0 = mEE_k0[np.newaxis,...], mMM_k0[np.newaxis,...]
            mEM_k0 = np.zeros(mEE_k0.shape)
            mEE = np.append(mEE_k0, mEE, axis=0)
            mEM = np.append(mEM_k0, mEM, axis=0)
            mME = np.append(mEM_k0, mME, axis=0)
            mMM = np.append(mMM_k0, mMM, axis=0)

        return [mEE, mEM, mME, mMM]

    def blockmatrices_channels(self, m, matsubaras):
        """Returns a list of matrices of dimension 
           (len(matsubaras), (lmax-max(m-1,0),lmax-max(m-1,0))
           which correspond to the entries of the round-trip matrix 
           M in the subspace of lz = m: M(m) = [[M_EE, MEM],[M_ME, M_MM]],
           for a numpy array of matsubara wavenumbers.

        """
        k0_evaluation = False
        if matsubaras[0] == 0:
            k0_evaluation = True
            matsubaras = matsubaras[1:]
        MieA, MieB, V = self.roundTrip_elements(m, matsubaras)
        V[4] = V[4][..., np.newaxis]
        dct = {"aTE": MieA[0], "V0": V[0], "bTE": MieB[0], "V1": V[1],
               "aTM": MieA[1], "V2": V[2], "bTM": MieB[1], "V3": V[3],
               "V4": V[4].transpose(0, 1, 3, 2),
               "V4t": V[4].transpose(0, 3, 1, 2), "a2": MieA[2],
               "b2": MieB[2], "V01": 1, "V23": 1, "V03": 1, "V21": 1}
        log_skalierung = ne.evaluate("a2+b2+V4+V4t", dct)
        skal = dynamic_scaling(log_skalierung)
        dct["skal"] = skal
        if self.forceFlag:
            if m == 0:
                V83 = 0
                V72 = 0
            else:
                V83 = V[8]/V[3]
                V72 = V[7]/V[2]
            dct["V01"] = V[5]/V[0] + V[6]/V[1]
            dct["V23"] = V72 + V83
            dct["V03"] = V[5]/V[0] + V83
        mEE1 = ne.evaluate("skal * aTE*(V0*bTE*V1*V01)", dct).sum(axis=3)
        mEE2 = ne.evaluate("skal * aTE*(V2*bTM*V3*V23)", dct).sum(axis=3)
        mEM1 = ne.evaluate("skal * aTE*(V0*bTE*V3*V03)", dct).sum(axis=3)
        mEM2 = ne.evaluate("skal * aTE*(V2*bTM*V1*V21)", dct).sum(axis=3)
        mME1 = ne.evaluate("skal * aTM*(V2*bTE*V1*V21)", dct).sum(axis=3)
        mME2 = ne.evaluate("skal * aTM*(V0*bTM*V3*V03)", dct).sum(axis=3)
        mMM1 = ne.evaluate("skal * aTM*(V2*bTE*V3*V23)", dct).sum(axis=3)
        mMM2 = ne.evaluate("skal * aTM*(V0*bTM*V1*V01)", dct).sum(axis=3)
        if k0_evaluation:
            mEE_k0, mMM_k0 = self.blockmatrices_k0(m)
            sh = mEE_k0.shape
            assert len(sh) == 2
            assert sh[0] == sh[1]
            zeros = np.zeros((1, sh[0], sh[1]))
            mEE1 = np.append(mEE_k0[np.newaxis], mEE1, axis=0)
            mMM1 = np.append(mMM_k0[np.newaxis], mMM1, axis=0)
            mEE2 = np.append(zeros, mEE2, axis=0)
            mMM2 = np.append(zeros, mMM2, axis=0)
            mEM1 = np.append(zeros, mEM1, axis=0)
            mEM2 = np.append(zeros, mEM2, axis=0)
            mME1 = np.append(zeros, mME1, axis=0)
            mME2 = np.append(zeros, mME2, axis=0)
            return [mEE1, mEE2, mMM1, mMM2, mEM1, mEM2, mME1, mME2]

        return [mEE1, mEE2, mMM1, mMM2, mEM1, mEM2, mME1, mME2]

    def blockmatrices_debug(self, m, matsubaras):
        """Returns a list of matrices of dimension 
           (len(matsubaras), (lmax-max(m-1,0),lmax-max(m-1,0))
           which correspond to the entries of the round-trip matrix 
           M in the subspace of lz = m: M(m) = [[M_EE, MEM],[M_ME, M_MM]],
           for a numpy array of matsubara wavenumbers.

        """
        k0_evaluation = False
        if matsubaras[0] == 0:
            k0_evaluation = True
            matsubaras = matsubaras[1:]
        MieA, MieB, V = self.roundTrip_elements(m, matsubaras)
        V[4] = V[4][..., np.newaxis]
        aTE= MieA[0]
        V0 = V[0]
        bTE= MieB[0]
        V1 = V[1]
        aTM= MieA[1]
        V2 = V[2]
        bTM= MieB[1]
        V3 = V[3]
        V4 = V[4].transpose(0, 1, 3, 2)
        V4t= V[4].transpose(0, 3, 1, 2)
        a2 = MieA[2]
        b2 = MieB[2]
        V01 = 1 
        V23 = 1
        V03 = 1
        V21 = 1
        log_skalierung = a2+b2+V4+V4t

        M = aTE*(V0*bTE*V1*V01 + V2*bTM*V3*V23)
        sgn_m = np.sign(M)
        abslog_m = np.log(abs(M))
        skal = dynamic_scaling(log_skalierung + abslog_m)
        resultEE = (sgn_m*skal).sum(axis=-1)

        M = aTE*(V0*bTE*V3*V03 + V2*bTM*V1*V21)
        sgn_m = np.sign(M)
        abslog_m = np.log(abs(M))
        skal = dynamic_scaling(log_skalierung+abslog_m)
        resultEM = (sgn_m*skal).sum(axis=-1)

        M = aTM*(V2*bTE*V1*V21 + V0*bTM*V3*V03)
        sgn_m = np.sign(M)
        abslog_m = np.log(abs(M))
        skal = dynamic_scaling(log_skalierung+abslog_m)
        resultME = (sgn_m*skal).sum(axis=-1)

        M = aTM*(V2*bTE*V3*V23 + V0*bTM*V1*V01)
        sgn_m = np.sign(M)
        abslog_m = np.log(abs(M))
        skal = dynamic_scaling(log_skalierung+abslog_m)
        resultMM = (sgn_m*skal).sum(axis=-1)

        if k0_evaluation:
            resultEE_k0, resultMM_k0 = self.blockmatrices_k0(m)
            resultEE_k0, resultMM_k0 = mEE_k0[np.newaxis,...], mMM_k0[np.newaxis,...]
            resultEM_k0 = np.zeros(resultEE_k0.shape)
            resultEE = np.append(resultEE_k0, resultEE, axis=0)
            resultEM = np.append(resultEM_k0, resultEM, axis=0)
            resultME = np.append(resultEM_k0, resultME, axis=0)
            resultMM = np.append(resultMM_k0, resultMM, axis=0)

        return [resultEE, resultEM, resultME, resultMM]

    def single_m_k0(self, m):
        A, B = self.blockmatrices_k0(m)
        lnDet_matsubaras_m = sln_det_blockdiagonal(A, B)
        return lnDet_matsubaras_m

    def mSumme_k0(self):
        A, B = self.blockmatrices_k0(0)
        lnDet_matsubaras_m = sln_det_blockdiagonal(A, B)
        for m in range(1, self.lmax+1):
            vorher = lnDet_matsubaras_m.copy()
            A, B = self.blockmatrices_k0(m)
            lnDet_matsubaras_m += 2*sln_det_blockdiagonal(A, B)
            rel_change = np.abs(np.abs(vorher/lnDet_matsubaras_m)-1) 
            if rel_change < self.precision/10:
                return lnDet_matsubaras_m
        return lnDet_matsubaras_m

    def single_m(self, m, matsubaras):
        """The same as mSumme, but it calculates the logdet directly
           without the blockdeterminant formula
        """
        conv = self.conv
        if matsubaras[0] == 0:
            m_k0 = self.single_m_k0(m)
            if len(matsubaras) >1:
                return np.append(m_k0, self.single_m(m, matsubaras[1:]))
            else:
                return m_k0
        M = matrix_append(self.blockmatrices(m, matsubaras))
        eins = np.eye(M.shape[1])[np.newaxis,:,:]
        lnDet_matsubaras_m = balanced_slogdet(M, conv)
        return lnDet_matsubaras_m

    def mSumme(self, matsubaras):
        """The same as mSumme, but it calculates the logdet directly
           without the blockdeterminant formula
        """
        conv = self.conv
        if matsubaras[0] == 0:
            mSumme_k0 = self.mSumme_k0()
            if len(matsubaras) >1:
                return np.append(mSumme_k0, self.mSumme(matsubaras[1:]))
            else:
                return mSumme_k0
        M = matrix_append(self.blockmatrices(0, matsubaras))
        eins = np.eye(M.shape[1])[np.newaxis,:,:]
        lnDet_matsubaras_m = balanced_slogdet(M, conv)
        for m in range(1, self.lmax+1):
            vorher = lnDet_matsubaras_m.copy()
            max_id = np.abs(vorher).argmax() 
            if vorher[max_id] == 0 and m>=1:
                return lnDet_matsubaras_m  
            M = matrix_append(self.blockmatrices(m, matsubaras))
            eins = np.eye(M.shape[1])[np.newaxis,:,:]
            lnDet_matsubaras_m += 2*balanced_slogdet(M, conv)
            rel_change = np.abs(np.abs(lnDet_matsubaras_m[max_id]/vorher[max_id])-1)
            if rel_change < self.precision/10:
                return lnDet_matsubaras_m
        return lnDet_matsubaras_m


    def mSum_oneRoundtrip(self, matsubaras, mmax="automatic"):# ??? Was hat das mit der Geometrie zu tun?
        """The analog of mSumme but calculating ln det(1-M(matsubaras)) in 
           single round-trip approximation as Trace(M(matsubaras))

        """
        A, B, C, D = self.blockmatrices(0, matsubaras)
        TrM_matsubaras_m = np.array([np.trace(A[i,:,:]+D[i,:,:]) 
                    for i in xrange(len(matsubaras))])
        th_m_max = round(np.log(self.precision)/np.log(0.5))
        if mmax == "automatic":
            mmax = round(np.log(self.precision)/np.log(0.5))
        else:
            assert type(mmax)== int and mmax <= self.lmax            
        for m in range(1, self.lmax+1):
            vorher = TrM_matsubaras_m.copy()
            A, B, C, D = self.blockmatrices(m, matsubaras)
            sh = np.shape(A)
            if sh[1] == 1:
                TrM_matsubaras_m += 2*np.array([np.trace(A[i]+D[i]) 
                            for i in xrange(len(matsubaras))])
                return TrM_matsubaras_m

            TrM_matsubaras_m += 2*np.array([np.trace(A[i]+D[i]) 
                                    for i in xrange(len(matsubaras))])
            if m >= 1 and m >= mmax:
                return TrM_matsubaras_m
        return TrM_matsubaras_m

    def mSum_roundtrips(self, matsubaras, order="automatic", forceFlag=0):
        m = 0
        bigM = self.supermatrix(m, matsubaras, forceFlag)
        TrM_matsubaras_m = self.arbitrary_roundtrips(bigM, order, m)
        for m in range(1, self.lmax+1):
            vorher = TrM_matsubaras_m.copy()
            bigM = self.supermatrix(m, matsubaras, forceFlag)
            TrM_matsubaras_m += 2*self.arbitrary_roundtrips(bigM, order, m)
            rel_change = abs(TrM_matsubaras_m/vorher)
            max_id = abs(TrM_matsubaras_m).argmax()
            if m >= 1 and rel_change[max_id] < self.precision:
                return TrM_matsubaras_m
        return TrM_matsubaras_m

    def arbitrary_roundtrips(self, Matrices, order, m):
        traceSum = 0
        Matrix = Matrices.copy()
        if type(order)== int:
            for n in range(1,order+1):
                traceSum += np.trace(Matrix, axis1=1,axis2=2)/n
                Matrix = np.einsum("ijk, ikl -> ijl", Matrices, Matrix)
        elif order == "automatic":
            n = 1.0
            relchange = 1
            traceSum = np.trace(Matrix,axis1=1,axis2=2)/n
            n += 1
            while relchange > self.precision:
                Matrix = np.einsum("ijk, ikl -> ijl", Matrices, Matrix)
                to_add = np.trace(Matrix,axis1=1,axis2=2)/n
                traceSum += to_add
                relchange = max(abs(to_add/traceSum))
                n += 1  
        self.roundTrip_dct[m] = n
        return -traceSum

    def supermatrix(self, m, matsubaras, forceFlag=0):
        """builds up a large blockmatrix consisting of the four
           matrices returned by the blockmatrices() method.

        """
        self.forceFlag = forceFlag
        Ms = self.blockmatrices(m, matsubaras)
        bigM = np.append(Ms[0], Ms[1], axis=2)
        bigM = np.append(bigM, np.append(Ms[2], Ms[3], axis=2), axis=1)
        return bigM

    def mSum_trace_dot(self, matsubaras):
        bigM = self.supermatrix(0, matsubaras, 0)
        bigM_force = self.supermatrix(0, matsubaras, 1)
        mSum = trace_dot_3d(bigM, bigM_force)
        for m in range(1, self.lmax+1):
            vorher = mSum.copy()
            bigM = self.supermatrix(m, matsubaras, 0)
            bigM_force = self.supermatrix(m, matsubaras, 1)
            mSum += 2*trace_dot_3d(bigM, bigM_force)
            if abs(abs(vorher/mSum)-1).all() < 1e-5:
                return mSum
        return mSum


class CoatedSphereSphere(SphereSphere):
    def __init__(self, r1_i, r1_a, mat1_core, mat1_coat, r2_i, r2_a,
                 mat2_core, mat2_coat, L, matMd, lmax, lssmax, forceFlag=0):
        SphereSphere.__init__(self, r1_a, r2_a, L, mat1_coat, mat2_coat,
                               matMd, lmax, lssmax, forceFlag)
        self.r1_i = r1_i
        self.r2_i = r2_i
        self.r1_a = r1_a
        self.r2_a = r2_a
        self.mat1core = mat1_core
        self.mat2core = mat2_core
        self.mat1coat = mat1_coat
        self.mat2coat = mat2_coat
        self.matMd = matMd# ??? wieso nicht konsistent mat3 oder matMd
        self.r1_a = self.r1
        self.r2_a = self.r2

    def roundTrip_elements(self, m, k):
        """collect all necessyary Mie-coefficients and translation matrices
           to describe the round-trip block-matrix in a subspace of m, the z-projection
           of the angularmomentum l in a spherical multipole basis for a numpy array of 
           wavenumbers k.

        """
        lmin = max(1, m)
        ext_VZ = -1+2*self.ext_flag
        if np.shape(self.K) == np.shape(k) and (self.K == k).all():
            mieA, mieB = self.MieA, self.MieB
        else:
            self.MieA = coated_miescattering(
                self.mat1core, self.mat1coat, self.matMd
                ).mie_coated(k, self.lmax, self.r1_i,
                             self.r1_a)[..., np.newaxis, np.newaxis]
            self.MieB = coated_miescattering(self.mat2core, self.mat2coat,
                                             self.matMd).mie_coated(
                k, self.lssmax, self.r2_i, self.r2_a)
            self.MieB = np.append(self.MieB[0:2][:, :, np.newaxis,
                                                 np.newaxis, :]**ext_VZ,
                                  self.MieB[2][np.newaxis, :, np.newaxis,
                                               np.newaxis, :]*ext_VZ,
                                  axis=0)
            mieA = self.MieA
            mieB = self.MieB
            self.K = k.copy()
        V = translation(self.lmax, self.lssmax, m, self.matMd, self.L, k,
                        self.ext_flag, 0).trans_ab()
        V = [V[0][:, :, np.newaxis, ...], V[1][:, np.newaxis, ...],
             V[2][:, :, np.newaxis, ...], V[3][:, np.newaxis, ...], V[4]]
        if self.forceFlag:
            Vdiff = translation(self.lmax, self.lssmax, m, self.matMd,
                                self.L, k, self.ext_flag, self.forceFlag
                                ).trans_ab()
            Vdiff = [Vdiff[0][:, :, np.newaxis, ...],
                     Vdiff[1][:, np.newaxis, ...],
                     Vdiff[2][:, :, np.newaxis, ...],
                     Vdiff[3][:, np.newaxis, ...]
                     ]

            V.extend(Vdiff)
        return [mieA[:, :, lmin-1::, :, :], mieB[:, :, :, :, lmin-1::], V]


def dot_3d_matrices(a, b):
    """determine the matrix product of three-dimensional matrices a and b
       with respect to axes 1 and 2; axis 0 is a spectator

    """
    if len(np.shape(a)) != 3 or len(np.shape(b)) != 3:
        raise ValueError("Error in geometry.dot_3d_matrices, "
                         + "one of the shapes %s and %s is "
                         + "not three-dimensional" % (len(np.shape(a)),
                                                      len(np.shape(b))))
    return np.einsum("ijk, ikl -> ijl", a, b)


def inv_3d_matrix(matrix):
    """determine the inverse of a three-dimensional matrix
       with respect to the axes 1 and 2; axis 0 is a spectator

       This functions should not be used with a matrix of type integer
       because the inverse matrix will also be of this type while the
       corrct inverse matrix will in general be of type float.

    """
    shM = np.shape(matrix)
    if not len(shM) == 3:
        raise ValueError("Error in geometry.inv_3d_matrix, "
                         + "shape %s is not three-dimensional" % shM)
    i_matrices = np.empty_like(matrix)
    for i in range(0, shM[0]):
        i_matrices[i] = np.linalg.inv(matrix[i])
    return i_matrices


def inv_3d_blockmatrix(matrix):
    """determine the inverse of a three-dimensional matrix
       with respect to the axes 1 and 2; axis 0 is a spectator

       The matrices to be inverted have the block structure
           [A11 A12]
       A = [       ] .
           [A21 A22]
       They should be of type float because inv_3d_matrix is used
       to invert matrices.

       ??? Dokumentation warum nicht die volle Matrix invertiert wird

    """
    shM = np.shape(matrix)
    if not len(shM) == 3:
        raise ValueError("Error in geometry.inv_3d_blockmatrix, "
                         + "shape %s is not three-dimensional" % str(shM))
    if shM[1] % 2 or shM[1] != shM[2]:
        raise ValueError("Error in geometry.inv_3d_blockmatrix, "
                         + "matrix with shape %s cannot be " % str(shM)
                         + "decomposed into quadratic blockmatrices")
    lmax = shM[1]//2
    A11 = matrix[:, 0:lmax, 0:lmax]
    A12 = matrix[:, 0:lmax, lmax::]
    A21 = matrix[:, lmax::, 0:lmax]
    A22 = matrix[:, lmax::, lmax::]
    iA11 = inv_3d_matrix(A11)
    schur = A22-dot_3d_matrices(dot_3d_matrices(A21, iA11), A12)
    ischur = inv_3d_matrix(schur)
    fak1 = dot_3d_matrices(A21, iA11)
    fak2 = dot_3d_matrices(iA11, A12)
    i11 = iA11+dot_3d_matrices(dot_3d_matrices(fak2, ischur), fak1)
    i12 = -dot_3d_matrices(fak2, ischur)
    i21 = -dot_3d_matrices(ischur, fak1)
    return np.append(np.append(i11, i12, axis=2),
                     np.append(i21, ischur, axis=2), axis=1)


def sln_det_3d(A, B, C, D, conv=0.9):
    """compute the logarithm of the # ??? absolute value of the # ???
       determinant det(1-Ms) with a block matrix
       Ms = [[1-A, B], [C, 1-D]]   # ???
       The matrices are three-dimensional with the axis 0 as a spectator.

       This function returns a 1d ndarray.

       slogdet for three-dimensional arrays requires at least NumPy 1.8.0.

    """
    if not A.shape == B.shape == C.shape == D.shape:
        raise ValueError("Error in geometry.lnDet, "
                         + "shapes %s, %s, %s, and %s are not equal" %
                         (A.shape, B.shape, C.shape, D.shape))
    M = matrix_append([A,B,C,D])
    one = np.eye(M.shape[1])[np.newaxis,:,:]
    return np.array([np.linalg.slogdet(balance((one-M)[i,:,:], conv))[1] 
                                                        for i in range(A.shape[0])])


def sln_det_blockdiagonal(A, B, conv = 0.9):
    if not A.shape == B.shape:
        raise ValueError("Error in geometry.ln_det_blockdiagonal. The shapes "+
                         "%s and %s of A and B are not equal" %(A.shape, B.shape))
    assert len(A.shape) == 2
    if A.shape == (1,1):
        return np.log((1-A)*(1-B))
    eins = np.eye(A.shape[0])
    bB, bA = balance(eins-B, conv), balance(eins-A, conv)
    return np.linalg.slogdet(bB)[1]+np.linalg.slogdet(bA)[1]

def trace_dot_3d(m1, m2):
    """compute the trace of (1-m1)^-1 times m2 for three-dimensional
       matrices m1 and m2 with the axis 0 as spectator

       This function returns a 1d ndarray.

    """
    sh = np.shape(m1)
    idy = np.identity(sh[1])[np.newaxis, :, :]
    return np.einsum("ijk, ikj -> i", inv_3d_blockmatrix(idy-m1), m2)

def balanced_slogdet(matrix, conv = 0.9):
    """ returns the log det of the matrix
        after balancing the matrix with a diagonal matrix

    """
    sh = matrix.shape
    assert len(sh) == 3
    assert sh[1] == sh[2]
    eins = np.eye(sh[1])
    balancedM = np.array([balance(eins-matrix[i,:,:], conv) for i in 
                        range(sh[0])])
    return np.linalg.slogdet(balancedM)[1]

def single_k_scale(M):
    """balances in a trace-invariant manner a given matrix where the maximum
       matrix elements are expected in the upper right or lower left corner

       This is intended for use with the l1 x l2 x lss sized blockmatrices
       containing logarithms of scaling factors.

    """
    shM = np.shape(M)
    assert len(shM) == 3
    if shM[1] == 1:
        return M
    order = M[0,-1,:].argmax()
    A = M[0, 0, order]
    B = M[0, -1, order]
    C = M[-1, 0, order]
    l1, l2 = np.ogrid[1:shM[1]+1, 1:shM[1]+1]
    fak = 1.0/(shM[1]-1)
    K = abs(A-max(B, C))*fak*(l1-l2)
    if C > B:
        K = -K
    return K[..., np.newaxis] + M


def dynamic_scaling(M):
    """rescales the matrix M in a trace-invariant manner by means 
       of single_k_scale. The axis 0 is only a spectator.

    """
    shM = np.shape(M)
    assert len(shM) == 4
    ergM = np.empty(shM)
    for i in range(0, shM[0]):
        ergM[i, :, :, :] = np.exp(single_k_scale(M[i, :, :, :]))
    return ergM

def matrix_append(matrix_list):
    """fügt Polarisationsmatrizen aneinander, um die große 
       Blockmatrix:
       M = [[A, B], 
            [C, D]]
       zu erhalten
    """
    assert(len(matrix_list) == 4)
    for i in range(3):
        assert(matrix_list[i].shape == matrix_list[i+1].shape)
    A, B, C, D = matrix_list
    z1 = np.append(A, B, axis=2)
    z2 = np.append(C, D, axis=2)
    return np.append(z1,z2, axis=1)


def get_lmax(r1, r2, L, precision, l_offset=1):
    """determines necessary angular momenta to achieve a given precision
       for a sphere-sphere scattering situation with two different sphere
       radii and center-to-center-distance L.
    """
    r = [r1, r2]
    r.sort()
    surf_to_surf =  abs(r[1]- L) - r[0]
    if surf_to_surf/r[0] >=30:
        return 1
    eta = -np.log10(precision)*0.33+3
    r_o_L = r[0]/surf_to_surf
    return max(int(np.exp(eta)*r_o_L/13.0)+l_offset, 1)

if __name__ == "__main__":
    from materials import PerfectConductor as pec, Vacuum, Gold, modifiedWater, Altern1Ps
    from materials import Gold_plasma, MercurySmith, Mercury_plasma, Mercury
    from matplotlib import pyplot as plt
    from thermodynamics import finiteT
    from materials import PerfectConductor as pec
    #import time
    mat1 = Gold
    mat2 = Vacuum
    mat3 = modifiedWater
    mat1list = [Vacuum, Gold, Altern1Ps, modifiedWater]
    r1 = 1
    r2 = 1
    L = 20
    k = np.logspace(-3,3,7)
    lmax = 60
    G1 = SphereSphere(r1, r2, L, Vacuum, Vacuum, modifiedWater, lmax=lmax,
                      l_offset=10, lssmax="automatic", precision=1e-5, 
                      forceFlag=0, evaluation="lndet")
    print("G1.lmax", G1.lmax)
    print("G1.msumme", G1.mSumme(k))

