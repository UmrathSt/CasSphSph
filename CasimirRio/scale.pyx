import numpy as np
cimport numpy as np
cimport cython
@cython.boundscheck(False)

def balance(np.ndarray[np.float64_t, ndim=2] matrix):
    cdef int n = matrix.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] scale = np.ones(n, dtype=np.float64)
    cdef int radix = 2
    cdef int i = 0
    cdef int j = 0
    cdef double r = 0
    cdef double c = 0
    cdef double g = 0
    cdef double f = 0
    cdef double s = 0
    cdef int done = 0
    cdef double sqrdx = radix**2
    while not done:
        done = 1
        for i in range(n):
            r, c = 0.0, 0.0
            for j in range(n):
                if not j == i:
                    c += abs(matrix[j, i])
                    r += abs(matrix[i, j])
                if not c == 0.0 and not r == 0.0:
                    g = r/radix
                    f = 1.0
                    s = c+r
                    while c<g:
                        f *= radix
                        c *= sqrdx
                    g = r*radix
                    while c>g:
                        f /= radix
                        c /= sqrdx
                    if (c+r)/f < 0.95*s:
                        done = 1
                        g = 1.0/f
                        scale[i] *= f
                        for j in range(0, n): 
                            matrix[i, j] *= g
                        for j in range(0, n):
                            matrix[j, i] *= f
    return matrix
