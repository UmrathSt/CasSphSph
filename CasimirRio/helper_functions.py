"""tests for helper_functions.py

"""

import numpy as np
from scipy.special import gammaln
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

def ln_2dbl(x):
    """return the natural logarithm of (2*x-1)!!,
       with !! meaning a double faculty

    """
    return gammaln(2*x+1)-x*np.log(2)-gammaln(x+1)

def lfac(x):
    """returns the natural logarithm of the factorial

    """
    return gammaln(x+1)


def num_derivative(f, x, interpolate=False, multiplier=10):
    """calculates a first order derivative of the numpy array
       f with equidistant spacing h via the 5-point formula
       at the edges lower-order differences are taken
    """
    h = x[1]-x[0]
    if not h>0:
        raise ValueError("x has to be increasing")
    if interpolate:
        xnew = np.linspace(x[0], x[-1], (x.size-1)*multiplier+1)
        signf = np.sign(f)
        h = h/multiplier
        if (signf[0] == signf).all():
            signf = signf[0]
            fip = interp1d(np.log(x), np.log(abs(f)))
            f = signf*np.exp(fip(np.log(xnew)))
        else:
            fip = interp1d(np.log(x), np.log(f+abs(np.min(f))+1))         
            plt.loglog(x, f+31, "k-")
            f = np.exp(fip(np.log(xnew)))
    i_max = f.shape[0]-1
    f_prime = np.zeros(f.shape)
    deriv_5th_order = lambda i : (-f[i+2]+8*f[i+1]-8*f[i-1]+f[i-2]) / (12*h)
    deriv_2nd_order = lambda i : (f[i+1]-f[i-1]) / (2*h)
    for i in range(2, i_max-1):
        f_prime[i] = deriv_5th_order(i)
    f_prime[1], f_prime[i_max-1] = deriv_2nd_order(1), deriv_2nd_order(i_max-1)
    f_prime[0], f_prime[i_max] = (f[1]-f[0]) / h, (f[i_max]-f[i_max-1]) / h
    if interpolate:
        f_prime = f_prime[::multiplier]
    return f_prime 


    
