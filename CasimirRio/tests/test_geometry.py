"""Tests for geometry

"""
import math
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import unittest
import geometry


class MatrixOps3d(unittest.TestCase):

    def test_dot_3d_matrices(self):
        """test product of matrices with axis 0 as spectator

        """
        a = np.arange(12.).reshape(3, 2, 2)
        b = np.arange(12., 24.).reshape(3, 2, 2)
        result = geometry.dot_3d_matrices(a, b)
        expected = np.array([[[ 14.,  15.],
                              [ 66.,  71.]],
                             [[154., 163.],
                              [222., 235.]],
                             [[358., 375.],
                              [442., 463.]]])
        assert_array_equal(result, expected) 

    def test_ln_det_blockdiagonal(self):
        """thest the ln det evaluation for the zeroth Matusbara
           frequency, where the blockmatrix becomes diagonal
           ln_det_blockdiagonal(A, B) should return:
           lndet(1-A)+ lndet(1-B)

        """
        A = np.array([[0.17, 2.7], [3, 0.4]])
        B = np.array([[2, 1.47], [1.1, -0.4]])
        eins = np.eye(A.shape[0])
        expected = np.log(np.linalg.det(eins-A)*np.linalg.det(eins-B))
        result = geometry.sln_det_blockdiagonal(A, B)
        assert_array_equal(expected, result)

    def test_balanced_slogdet(self):
        """thest the behaviour of the balanced slogdet of a matrix.
           balanced_slogdet should take a 3-d matrix M and return the 
           logdet of (1-M[i,:,:]) for all i

        """
        A = np.array([[-0.17, -2.7], [3, 0.4]])
        B = np.array([[2, 1.47], [-1.1, -0.4]])
        M = np.append(A[np.newaxis], B[np.newaxis], axis=0)
        eins = np.eye(M.shape[1])[np.newaxis]
        expected = np.log(np.linalg.det(eins-M))
        result = geometry.balanced_slogdet(M)
        assert_array_equal(expected, result)

    def test_inv_3d_matrix(self):
        """test inverse of a matrix with axis 0 as spectator

        """
        matrix = np.arange(12.).reshape(3, 2, 2)
        result = geometry.inv_3d_matrix(matrix)
        expected = np.array([[[-1.5, 0.5],
                              [1.,   0. ]],
                             [[-3.5, 2.5],
                              [3.,  -2. ]],
                             [[-5.5, 4.5],
                              [5.,  -4. ]]])
        assert_array_almost_equal(result, expected, decimal=10)

    def test_inv_3d_blockmatrix(self):
        """test inverse of a matrix using its block structure with
           axis 0 as spectator

        """
        matrix = np.random.rand(3, 4, 4)
        inversematrix = geometry.inv_3d_blockmatrix(matrix)
        id = np.identity(4)
        for i in range(matrix.shape[0]):
            assert_array_almost_equal(np.dot(matrix[i],
                                             inversematrix[i]),
                                      id)

    def test_inv_3d_blockmatrix_odd(self):
        """check whether exception is raised if no decomposition
           in blockmatrices is possible

        """
        matrix = np.random.rand(3, 5, 5)
        with self.assertRaises(ValueError):
            inversematrix = geometry.inv_3d_blockmatrix(matrix)

    def test_inv_3d_blockmatrix_nonquadratic(self):
        """check whether exception is raised if the blockmatrices
           are not quadratic

        """
        matrix = np.random.rand(3, 4, 6)
        with self.assertRaises(ValueError):
            inversematrix = geometry.inv_3d_blockmatrix(matrix)

    def test_ln_3d_idmatrix1(self):
        """test the behavior for an identity matrix where the blockmatrices
           are scalar

        """
        a = np.zeros((3, 1, 1))
        b = np.zeros((3, 1, 1))
        c = np.zeros((3, 1, 1))
        d = np.zeros((3, 1, 1))
        result = geometry.sln_det_3d(a, b, c, d)
        expected = np.array([0., 0., 0.])
        assert_array_almost_equal(result, expected)

    def test_ln_3d_1(self):
        """test the behavior where the blockmatrices are scalar

        """
        a = np.array([[[0.1]], [[0.3]], [[0.7]]])
        b = np.array([[[0.2]], [[0.3]], [[0.4]]])
        c = np.array([[[0.7]], [[0.5]], [[0.1]]])
        d = np.array([[[0.3]], [[0.2]], [[0.6]]])
        result = geometry.sln_det_3d(a, b, c, d)
        expected = np.array([math.log(0.49),
                             math.log(0.41),
                             math.log(0.08)])
        assert_array_almost_equal(result, expected)

    def test_ln_3d_idmatrix2(self):
        """test the behavior for an identity matrix where the blockmatrices
           are not scalar

        """
        a = np.zeros((3, 2, 2))
        b = np.zeros((3, 2, 2))
        c = np.zeros((3, 2, 2))
        d = np.zeros((3, 2, 2))
        result = geometry.sln_det_3d(a, b, c, d)
        expected = np.array([0., 0., 0.])
        assert_array_almost_equal(result, expected)

    def test_ln_3d_2(self):
        """test the behavior where the blockmatrices are not scalar

        """
        a = 0.001*np.arange(12).reshape(3, 2, 2)
        b = 0.001*np.arange(12, 24).reshape(3, 2, 2)
        c = 0.001*np.arange(24, 36).reshape(3, 2, 2)
        d = 0.001*np.arange(36, 48).reshape(3, 2, 2)
        result = geometry.sln_det_3d(a, b, c, d)
        m = np.empty((3, 4, 4))
        m[:, 0:2, 0:2] = a
        m[:, 0:2, 2:4] = b
        m[:, 2:4, 0:2] = c
        m[:, 2:4, 2:4] = d
        expected = np.log(np.abs(np.linalg.det(np.identity(4)-m)))
        assert_array_almost_equal(result, expected)

    def test_trace_dot_3d(self):
        m1 = np.arange(12).reshape(3, 2, 2)
        m2 = np.arange(12, 24).reshape(3, 2, 2)
        result = geometry.trace_dot_3d(m1, m2)
        expected = np.array([-7.75, -3.25, -2.35])
        assert_array_almost_equal(result, expected)


class Scaling(unittest.TestCase):

    def test_single_k_scale_1(self):
        """test trace-invariant scaling for axes 0 and 1 of length 1

        """
        m = np.arange(1, 4).reshape(1, 1, 3)
        result = geometry.single_k_scale(m)
        assert_array_equal(result, m)

    def test_single_k_scale_B(self):
        """test trace-invariant scaling for B > C

        """
        m = -np.arange(1, 13).reshape(2, 2, 3)
        result = geometry.single_k_scale(m)
        expected = np.array([[[-1, -2, -3], [-7, -8, -9]],
                             [[-4, -5, -6], [-10, -11, -12]]])
        assert_array_equal(result, expected)

    def test_single_k_scale_C(self):
        """test trace-invariant scaling for C > B

        """
        m = np.arange(1, 13).reshape(2, 2, 3)
        result = geometry.single_k_scale(m)
        expected = np.array([[[1, 2, 3], [10, 11, 12]],
                             [[1, 2, 3], [10, 11, 12]]])
        assert_array_equal(result, expected)

    def test_dynamic_scaling_B(self):
        """test trace-invariant scaling with axis 0 as spectator
           for B > C

        """
        m = -np.arange(1, 13).reshape(2, 2, 3)[np.newaxis, ...]
        m = np.concatenate((m, m, m, m), axis=0)
        expected = np.array([[[-1, -2, -3], [-7, -8, -9]],
                             [[-4, -5, -6], [-10, -11, -12]]])
        expected = np.exp(expected)[np.newaxis, ...]
        expected = np.concatenate((expected, expected,
                                   expected, expected), axis=0)
        result = geometry.dynamic_scaling(m)
        assert_array_equal(result, expected)

    def test_dynamic_scaling_C(self):
        """test trace-invariant scaling with axis 0 as spectator
           for C > B

        """
        m = np.arange(1, 13).reshape(2, 2, 3)[np.newaxis, ...]
        m = np.concatenate((m, m, m, m), axis=0)
        expected = np.array([[[1, 2, 3], [10, 11, 12]],
                             [[1, 2, 3], [10, 11, 12]]])[np.newaxis, ...]
        expected = np.exp(expected)
        expected = np.concatenate((expected, expected,
                                   expected, expected), axis=0)
        result = geometry.dynamic_scaling(m)
        assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
