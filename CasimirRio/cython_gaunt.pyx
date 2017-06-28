import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double exp(double x)

cdef extern from "math.h":
    double lgamma(double x)

cdef extern from "math.h":
    double sqrt(double x)

cdef extern from "math.h":
    double log(double x)

@cython.cdivision(True)
@cython.boundscheck(False)
def gaunt(int l, int lp, int m):

    cdef int lppmax = l+lp
    cdef int lppmin = abs(l-lp)
    cdef int alphadim = lppmax-lppmin
    cdef int gauntdim = (lppmax-lppmin)//2+1
    cdef double a2
    cdef double b2  
    cdef np.ndarray[np.float64_t, ndim=1] alphas = np.zeros(alphadim, np.float64)
    cdef int idx
    cdef double p
    cdef double amax
    cdef double M_PI = 3.14159265358979328462
    cdef np.ndarray[np.float64_t, ndim=1] gaunts = np.zeros(gauntdim, np.float64)
        
    if abs(m) > min(l, lp):
        return gaunts
    # alphas according to Xu (3)
    a2 = (l+lp+1)**2
    b2 = (l-lp)**2
    for idx in range(lppmax-lppmin):
        p = idx+lppmin+1.0
        alphas[idx] = (p*p-a2)*(p*p-b2)/(4.0*p*p-1)
    # Gaunt coefficient for maximal lpp according to Xu (4)
    amax = exp(lgamma(2*l+1)+lgamma(2*lp+1)+2*lgamma(lppmax+1) -
               lgamma(l+1)-lgamma(lp+1)-lgamma(2*lppmax+1) -
               lgamma(l-m+1)-lgamma(lp+m+1)+
                0.5*(lgamma(l-m+1)+lgamma(lp+m+1) -
                                   lgamma(l+m+1)-lgamma(lp-m+1))
               +0.5*(log(2*l+1)+log(2*lp+1)-log(4*M_PI))
                )
    # normalized Gaunt coefficients as defined in Bruning (3.8) and Xu (1)
    gaunts = np.zeros(gauntdim)
    gaunts[gauntdim-1] = amax
    if gauntdim > 1:
        gaunts[gauntdim-2] = amax*((2.0*lppmax-3.0)*
                                   (l*lp-m**2.0*(2.0*lppmax-1.0))/
                                   ((2.0*l-1.0)*(2.0*lp-1.0)*lppmax))
    for idx in range(gauntdim-3, -1, -1):
        gaunts[idx] = (((4.0*m*m+alphas[2*idx+1] +
                        alphas[2*idx+2])*gaunts[idx+1] -
                       alphas[2*idx+3]*gaunts[idx+2])/
            (alphas[2*idx]))
    # Gaunt coefficients Y^(lpp, l, lp)_(0, m, -m) as defined in
    # Rasch et al. (4.1)
    for idx in range(gauntdim):
        gaunts[idx] = gaunts[idx]/sqrt(2.0*(lppmin+2.0*idx)+1.0)

    return gaunts
