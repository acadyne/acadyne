import unittest
from acadyne.tensores.dynamic_tensor import DynamicTensor
from acadyne.tensores.set_tensor import SetTensor
import numpy as np
import sympy as sp

class TestDynamicTensor(unittest.TestCase):
    def test_matrix_multiplication(self):
        t1 = SetTensor([[1, 2, 3], [4, 5, 6]])
        t2 = SetTensor([[7, 8], [9, 10], [11, 12]])
        result = t1.multiply(t2)
        expected = SetTensor([[58, 64], [139, 154]])
        self.assertTrue(np.array_equal(result.components, expected.components))

    def test_scalar_multiplication(self):
        t = SetTensor([[1, 2], [3, 4]])
        scalar = 3
        result = t.multiply(scalar)
        expected = SetTensor([[3, 6], [9, 12]])
        self.assertTrue(np.array_equal(result.components, expected.components))

    def test_determinant_of_symbolic_tensor(self):
        a, b, c, d = sp.symbols('a b c d')
        t = SetTensor([[a, b], [c, d]])
        determinant = t.determinant()
        expected = a*d - b*c
        self.assertEqual(determinant, expected)

    def test_inverse_of_symbolic_tensor(self):
        a, b, c, d = sp.symbols('a b c d')
        t = SetTensor([[a, b], [c, d]])
        inverse = t.inverse()
        expected = SetTensor([[d/(a*d - b*c), -b/(a*d - b*c)], [-c/(a*d - b*c), a/(a*d - b*c)]])
        for i in range(2):
            for j in range(2):
                self.assertEqual(inverse.components[i, j].simplify(), expected.components[i, j].simplify())

    def test_apply_nonlinear_transformation(self):
        t = DynamicTensor([[1, 2], [3, 4]])
        result = t.nonlinear_transformation(lambda x: x**2)
        expected = DynamicTensor([[1, 4], [9, 16]])
        self.assertTrue(np.array_equal(result.components, expected.components))

    def test_singular_value_decomposition(self):
        t = SetTensor([[1, 0], [0, 1]])
        u, s, v = t.singular_value_decomposition()
        self.assertTrue(np.allclose(u @ np.diag(s) @ v, np.array([[1, 0], [0, 1]])))

    def test_lu_decomposition(self):
        t = SetTensor([[4, 3], [6, 3]])
        L, U, perm = t.lu_decomposition()
        L_expected = [[1, 0], [1.5, 1]]
        U_expected = [[4, 3], [0, -1.5]]
        self.assertTrue(np.allclose(L, L_expected))
        self.assertTrue(np.allclose(U, U_expected))

    def test_qr_decomposition(self):
        t = SetTensor([[1, 2], [3, 4]])
        q, r = t.qr_decomposition()
        self.assertTrue(np.allclose(q @ r, np.array([[1, 2], [3, 4]])))

    def test_evaluate_symbolic_tensor(self):
        x = sp.symbols('x')
        t = SetTensor([[x, x**2], [x**3, 1]])
        result = t.evaluate({x: 2})
        expected = SetTensor([[2, 4], [8, 1]])
        self.assertTrue(np.array_equal(result.components, expected.components))

    def test_differentiate_symbolic_tensor(self):
        x = sp.symbols('x')
        t = SetTensor([[x, x**2], [sp.sin(x), sp.cos(x)]])
        result = t.differentiate(x)
        expected = SetTensor([[1, 2*x], [sp.cos(x), -sp.sin(x)]])
        self.assertTrue(np.array_equal(result.components, expected.components))

if __name__ == '__main__':
    unittest.main()
