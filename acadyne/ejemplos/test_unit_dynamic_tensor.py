import unittest
import sympy as sp

from tensores.dynamic_tensor import BaseTensor

class TestSymbolicTensor(unittest.TestCase):

    def setUp(self):
        self.x, self.y, self.z = sp.symbols('x y z')
        self.tensor = BaseTensor([
            [self.x**2 + self.y, sp.sin(self.x) * sp.cos(self.y), self.z**2],
            [sp.exp(self.y), sp.log(self.x + 1), sp.sin(self.z)],
            [self.x * self.y, self.y * self.z, self.z * self.x]
        ])

    def test_tensor_evaluation(self):
        subs = {self.x: 1, self.y: 2, self.z: 3}
        evaluated_tensor = self.tensor.evaluate(subs)
        expected_result = sp.Matrix([
            [3, sp.sin(1) * sp.cos(2), 9],
            [sp.exp(2), sp.log(2), sp.sin(3)],
            [2, 6, 3]
        ])
        print("Evaluated Tensor:", evaluated_tensor)
        print("Expected Result:", expected_result)
        for i in range(3):
            for j in range(3):
                self.assertTrue(sp.simplify(evaluated_tensor.components[i, j] - expected_result[i, j]).is_zero)

    def test_rotation_3d(self):
        theta_x, theta_y, theta_z = sp.symbols('theta_x theta_y theta_z')
        rotation_matrix = sp.Matrix([
            [sp.cos(theta_y) * sp.cos(theta_z), -sp.cos(theta_y) * sp.sin(theta_z), sp.sin(theta_y)],
            [sp.sin(theta_x) * sp.sin(theta_y) * sp.cos(theta_z) + sp.cos(theta_x) * sp.sin(theta_z), -sp.sin(theta_x) * sp.sin(theta_y) * sp.sin(theta_z) + sp.cos(theta_x) * sp.cos(theta_z), -sp.sin(theta_x) * sp.cos(theta_y)],
            [-sp.cos(theta_x) * sp.sin(theta_y) * sp.cos(theta_z) + sp.sin(theta_x) * sp.sin(theta_z), sp.cos(theta_x) * sp.sin(theta_y) * sp.sin(theta_z) + sp.sin(theta_x) * sp.cos(theta_z), sp.cos(theta_x) * sp.cos(theta_y)]
        ])
        rotation_tensor = BaseTensor(rotation_matrix.tolist())
        rotation_tensor_T = BaseTensor(rotation_matrix.T.tolist())

        rotated_tensor = rotation_tensor.multiply(self.tensor).multiply(rotation_tensor_T)
        rotated_tensor_evaluated = rotated_tensor.evaluate({theta_x: sp.pi / 4, theta_y: sp.pi / 4, theta_z: sp.pi / 4})

        self.assertIsNotNone(rotated_tensor_evaluated)  # This is just a placeholder check
        # Use actual expected values for a real test

    def test_integration_xyz(self):
        tensor_integrated_xyz = self.tensor.integrate([self.x, self.y, self.z])
        expected_result = sp.Matrix([
            [self.z * (self.x**3 * self.y / 3 + self.x * self.y**2 / 2), -self.z * sp.sin(self.y) * sp.cos(self.x), self.x * self.y * self.z**3 / 3],
            [self.x * self.z * sp.exp(self.y), self.y * self.z * (self.x * sp.log(self.x + 1) - self.x + sp.log(self.x + 1)), -self.x * self.y * sp.cos(self.z)],
            [self.x**2 * self.y**2 * self.z / 4, self.x * self.y**2 * self.z**2 / 4, self.x**2 * self.y * self.z**2 / 4]
        ])
        print("Integrated Tensor:", tensor_integrated_xyz)
        print("Expected Result:", expected_result)
        for i in range(3):
            for j in range(3):
                self.assertTrue(sp.simplify(tensor_integrated_xyz.components[i, j] - expected_result[i, j]).is_zero)

    def test_cross_differentiation(self):
        tensor_diff_xy = self.tensor.differentiate(self.x).differentiate(self.y)
        expected_result = sp.Matrix([
            [0, -sp.sin(self.y) * sp.cos(self.x), 0],
            [0, 0, 0],
            [1, 0, 0]
        ])
        print("Cross Differentiated Tensor:", tensor_diff_xy)
        print("Expected Result:", expected_result)
        for i in range(3):
            for j in range(3):
                self.assertTrue(sp.simplify(tensor_diff_xy.components[i, j] - expected_result[i, j]).is_zero)

if __name__ == '__main__':
    unittest.main()
