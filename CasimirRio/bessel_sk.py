"""bessel_sk.py defines scaled modified Bessel functions.
   The helper functions mie_bessels() and trans_bessels()
   provide an interface for the use with scattering.py and
   translation.py. 

"""
import numpy as np
__version__ = "SVN Revision "+str("".join(list(filter(str.isdigit, "$Revision: 684 $"))))
def lnknu(x, lmax):
    """ natural log of the irregular modified spherical bessel
        function for all orders l = 0 .. lmax
    """
    if not (type(lmax) == int or type(lmax) == np.int64):
        raise TypeError('lmax must be an integer')
    lnk = np.empty((len(x), lmax+1))
    ln_pih = np.log(np.pi*0.5)
    k0 = ln_pih -np.log(x) - x
    k1 = ln_pih -2*np.log(x) - x + np.log(x+1)
    lnk[:, 0] = k0
    if lmax == 0:
        return lnk
    lnk[:, 1] = k1
    if lmax == 1:
        return lnk
    for l in range(2, lmax+1):
        lnk[:, l] = lnk[:,l-1] + np.log(np.exp(lnk[:,l-2]-lnk[:,l-1]) + (2*l-1)/x)
    return lnk

def fraction(x):
    """calculates continued fractions which are needed for the ratios
       I_(nu+1) / I_(nu) [1]

       References:
       [1] W. Gautschi, J. Slavik, Math. Comp. 32, 865 (1978),
           http://dx.doi.org/10.2307/2006491

    """
    if len(x[0, :]) == 1:
        return x[:, 0]
    return x[:, 0] + 1 / fraction(x[:, 1::])

def fnu(x, nu):
    """returns I(nu, x)/I(nu+1, x)
       nu = l+0.5 has to be a scalar, while x has to be a one-dimensional
       numpy.nd_array

    """
    Nmax = 80
    if x[-1] > 100:
        Nmax = 140
    n = np.arange(1, Nmax+1).reshape(1, Nmax)
    x = x[..., np.newaxis]
    an = 2 * (nu + n) / x
    return fraction(an)



def lniknu(x, lmax):
    """ natural log of the irregular modified spherical bessel
        function for all orders l = 0 .. lmax
    """
    if not (type(lmax) == int or type(lmax) == np.int64):
        raise TypeError('lmax must be an integer')
    lni = np.empty((len(x), lmax+1))
    sm = x < 1e-3
    lg = x >= 1e-3
    xsm = x[sm]
    xlg = x[lg]
    if not xsm.shape[0] == 0 and not xlg.shape[0] == 0:
        i0sm = xsm**2/6 - xsm**4/180 + xsm**6/2835
        i0lg = -np.log(2*xlg) + np.log(1-np.exp(-2*xlg)) + xlg
        i1sm = -np.log(3)+np.log(xsm)+0.1*xsm**2-xsm**4/700
        i1lg = -np.log(2*xlg) -np.log(xlg) + xlg + np.log(np.exp(-2*xlg)*(xlg+1)+xlg-1)
        i0 = np.append(i0sm, i0lg, axis=0)
        i1 = np.append(i1sm, i1lg, axis=0)
    else:
        if xsm.shape[0] == 0:
            i0lg = -np.log(2*xlg) + np.log(1-np.exp(-2*xlg)) + xlg
            i1lg = -np.log(2*xlg) -np.log(xlg) + xlg + np.log(np.exp(-2*xlg)*(xlg+1)+xlg-1)
            i0, i1 = i0lg, i1lg
        if xlg.shape[0] == 0:
            i0sm = xsm**2/6-xsm**4/180
            i1sm = -np.log(3)+np.log(xsm)+0.1*xsm**2
            i0, i1 = i0sm, i1sm
    lni[:, 0] = i0
    lnk = lnknu(x, lmax+1)
    if lmax == 0:
        return lni, lnk[:,0:1]
    lni[:, 1] = i1
    if lmax == 1:
        return lni, lnk[:,0:2]

    for l in range(2, lmax+1):
        f = fnu(x, l+0.5)
        lni[:, l] = np.log(np.pi/(2*x**2)) - lnk[:,l+1] \
                    -np.log(np.exp(lnk[:,l]-lnk[:,l+1])/f + 1)
    return lni, lnk[:,0:-1]



def mie_bessels(kr, lmin, lmax):
    """calculate scaled Bessel functions I and K for angular momenta
       between lmin and lmax

       kr is expected to be a one-dimensional ndarray, while lmin and 
       lmax are expected to be integers specifying the minimum and 
       maximum order of I and K needed in the computation of the Mie
       coefficients defined in scattering.py.

    """
    if lmin > lmax:
        raise ValueError("lmin should be less or equal lmax, found: "
                         + "lmin=%s, lmax=%s" % (lmin, lmax))
    lninu, lnknu = lniknu(kr, lmax)
    return lninu[:, lmin::], lnknu[:, lmin::]


def trans_bessels(kr, lmax, ext_flag):
    """trans_bessels calculates scaled Bessel function for all orders 
       up to lmax depending on geometry

       ext_flag=0: interior geometry -> Bessel function I
       ext_flag=1: exterior geometry -> Bessel function K

       kr is expected to be a one-dimensional ndarray.

    """
    if ext_flag:
        return lnknu(kr, lmax)
    else:
        return lniknu(kr, lmax)[0]

def i_fraction(x, nu):
    """ returns I(nu, x)/I(nu+1, x)
        nu = l+0.5 has to be a one-dimensional numpy.nd_array and x has 
        to be a one-dimensional numpy.nd_array

    """
    return np.array([fnu(x, nu) for nu in nu])
