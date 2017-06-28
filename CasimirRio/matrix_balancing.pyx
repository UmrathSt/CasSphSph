import numpy as np
cimport numpy as np
cimport cython

@cython.cdivision(True)
@cython.boundscheck(False)
def balance(np.ndarray[np.float64_t, ndim=2] matrix, double conv = 0.95):
    cdef int n = matrix.shape[0]
    cdef int beta = 2
    cdef int i = 0
    cdef int j = 0
    cdef double r = 0
    cdef double c = 0
    cdef double f = 0
    cdef double s = 0
    cdef int converged = 0
    while converged == 0:
        converged = 1
        for i in range(n):
            r, c = 0.0, 0.0
            for j in range(n):
                if not j == i:
                    c += abs(matrix[j, i])**2
                    r += abs(matrix[i, j])**2
            if not(abs(r) <= 1e-320  or abs(c) <= 1e-320):
                s = c + r
                c = c**0.5
                r = r**0.5
                f = 1.0
                while c < r/beta:
                    f *= beta 
                    c *= beta
                    r /= beta
                while c >= r*beta:
                    f /= beta
                    c /= beta
                    r *= beta
                if (c**2+r**2) < conv*s:
                    converged = 0
                    matrix[:, i] *= f
                    matrix[i, :] /= f 
    return matrix
