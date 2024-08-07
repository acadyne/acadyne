import unittest
import sympy as sp
from acadyne.tensor.set_tensor import SetTensor, InvalidElementStructureError

class TestSetTensor(unittest.TestCase):
    def test_initialize_with_list(self):
        tensor = SetTensor([1, 2, 3])
        self.assertEqual(tensor.elements, sp.FiniteSet(1, 2, 3))

    def test_initialize_with_finiteset(self):
        finite_set = sp.FiniteSet(1, 2, 3)
        tensor = SetTensor(finite_set)
        self.assertEqual(tensor.elements, finite_set)

    def test_initialize_with_interval(self):
        interval = sp.Interval(0, 5)
        tensor = SetTensor(interval)
        self.assertEqual(tensor.elements, interval)

    def test_union(self):
        tensor1 = SetTensor([1, 2, 3])
        tensor2 = SetTensor([3, 4, 5])
        union_tensor = tensor1.union(tensor2)
        self.assertEqual(union_tensor.elements, sp.FiniteSet(1, 2, 3, 4, 5))

    def test_intersect(self):
        tensor1 = SetTensor([1, 2, 3])
        tensor2 = SetTensor([3, 4, 5])
        intersection_tensor = tensor1.intersect(tensor2)
        self.assertEqual(intersection_tensor.elements, sp.FiniteSet(3))

    def test_difference(self):
        tensor1 = SetTensor([1, 2, 3])
        tensor2 = SetTensor([3, 4, 5])
        difference_tensor = tensor1.difference(tensor2)
        self.assertEqual(difference_tensor.elements, sp.FiniteSet(1, 2))

    def test_symmetric_difference(self):
        tensor1 = SetTensor([1, 2, 3])
        tensor2 = SetTensor([3, 4, 5])
        sym_diff_tensor = tensor1.symmetric_difference(tensor2)
        self.assertEqual(sym_diff_tensor.elements, sp.FiniteSet(1, 2, 4, 5))

    def test_invalid_initialization(self):
        with self.assertRaises(InvalidElementStructureError):
            SetTensor(42)  # No es un tipo válido

if __name__ == '__main__':
    unittest.main()
