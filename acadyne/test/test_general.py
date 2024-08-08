import unittest
import sympy as sp

from tensores.set_tensor import InvalidElementStructureError, SetTensor

class TestSetTensor(unittest.TestCase):

    def setUp(self):
        # Inicializa algunos conjuntos de prueba
        self.set_a = SetTensor([1, 2, 3])
        self.set_b = SetTensor([3, 4, 5])
        self.universal_set = SetTensor([1, 2, 3, 4, 5, 6])
        self.empty_set = SetTensor([])

    def test_initialization(self):
        # Prueba la inicialización con diferentes tipos de elementos
        print("Prueba de inicialización de conjuntos...")
        self.assertEqual(self.set_a.elements, sp.FiniteSet(1, 2, 3))
        with self.assertRaises(InvalidElementStructureError):
            SetTensor("invalid_type")

    def test_union(self):
        # Prueba la unión de dos conjuntos
        print("Prueba de unión de conjuntos...")
        result = self.set_a.union(self.set_b)
        expected = SetTensor([1, 2, 3, 4, 5])
        print(f"Resultado: {result.elements}, Esperado: {expected.elements}")
        self.assertEqual(result.elements, expected.elements)

    def test_intersection(self):
        # Prueba la intersección de dos conjuntos
        print("Prueba de intersección de conjuntos...")
        result = self.set_a.intersect(self.set_b)
        expected = SetTensor([3])
        print(f"Resultado: {result.elements}, Esperado: {expected.elements}")
        self.assertEqual(result.elements, expected.elements)

    def test_difference(self):
        # Prueba la diferencia de dos conjuntos
        print("Prueba de diferencia de conjuntos...")
        result = self.set_a.difference(self.set_b)
        expected = SetTensor([1, 2])
        print(f"Resultado: {result.elements}, Esperado: {expected.elements}")
        self.assertEqual(result.elements, expected.elements)
        
        # Prueba la diferencia que da un conjunto vacío
        result_empty = self.set_a.difference(SetTensor([1, 2, 3]))
        print(f"Resultado para conjunto vacío: {result_empty.elements}, Esperado: {self.empty_set.elements}")
        self.assertEqual(result_empty.elements, self.empty_set.elements)

    def test_is_subset(self):
        # Prueba si un conjunto es un subconjunto de otro
        print("Prueba de subconjunto...")
        print(f"¿self.set_a es subconjunto de universal_set? {self.set_a.is_subset(self.universal_set)}")
        self.assertTrue(self.set_a.is_subset(self.universal_set))
        print(f"¿self.set_b es subconjunto de self.set_a? {self.set_b.is_subset(self.set_a)}")
        self.assertFalse(self.set_b.is_subset(self.set_a))

    def test_symmetric_difference(self):
        # Prueba la diferencia simétrica
        print("Prueba de diferencia simétrica...")
        result = self.set_a.symmetric_difference(self.set_b)
        expected = SetTensor([1, 2, 4, 5])
        print(f"Resultado: {result.elements}, Esperado: {expected.elements}")
        self.assertEqual(result.elements, expected.elements)

    def test_complement(self):
        # Prueba el complemento de un conjunto respecto al conjunto universal
        print("Prueba de complemento de conjuntos...")
        result = self.set_a.complement(self.universal_set)
        expected = SetTensor([4, 5, 6])
        print(f"Resultado: {result.elements}, Esperado: {expected.elements}")
        self.assertEqual(result.elements, expected.elements)

    def test_power_set(self):
        # Prueba el conjunto potencia
        print("Prueba de conjunto potencia...")
        power_set = self.set_a.power_set()
        expected_elements = [
            SetTensor([]), 
            SetTensor([1]), 
            SetTensor([2]), 
            SetTensor([3]), 
            SetTensor([1, 2]), 
            SetTensor([1, 3]), 
            SetTensor([2, 3]), 
            SetTensor([1, 2, 3])
        ]
        power_set_elements = sorted([s.elements for s in power_set], key=lambda x: len(x))
        expected_elements_elements = sorted([s.elements for s in expected_elements], key=lambda x: len(x))
        print(f"Resultado: {power_set_elements}, Esperado: {expected_elements_elements}")
        self.assertEqual(power_set_elements, expected_elements_elements)

    def test_apply_function(self):
        # Prueba la aplicación de una función simbólica
        print("Prueba de aplicación de función...")
        func_lambda = lambda x: x**2
        result = self.set_a.apply_function(func_lambda)
        expected = SetTensor([1, 4, 9])
        print(f"Resultado: {result.elements}, Esperado: {expected.elements}")
        self.assertEqual(result.elements, expected.elements)

    def test_simplify_elements(self):
        # Prueba la simplificación de elementos simbólicos
        print("Prueba de simplificación de elementos...")
        set_c = SetTensor([sp.sin(sp.pi/2), sp.cos(sp.pi)])
        result = set_c.simplify_elements()
        expected = SetTensor([1, -1])
        print(f"Resultado: {result.elements}, Esperado: {expected.elements}")
        self.assertEqual(result.elements, expected.elements)

    def test_solve_equations(self):
        # Prueba la resolución de ecuaciones simbólicas
        print("Prueba de resolución de ecuaciones...")
        eq1 = sp.Eq(sp.Symbol('x') + 2, 0)
        solutions = self.set_a.solve_equations([eq1], [sp.Symbol('x')])
        expected = [sp.FiniteSet(-2)]
        print(f"Resultado: {solutions}, Esperado: {expected}")
        self.assertEqual(solutions, expected)

    def test_integrate_elements(self):
        # Prueba la integración de elementos simbólicos
        print("Prueba de integración de elementos...")
        set_d = SetTensor([sp.Symbol('x')])
        result = set_d.integrate_elements(sp.Symbol('x'))
        expected = SetTensor([sp.Rational(1, 2) * sp.Symbol('x')**2])
        print(f"Resultado: {result.elements}, Esperado: {expected.elements}")
        self.assertEqual(result.elements, expected.elements)

    def test_differentiate_elements(self):
        # Prueba la derivación de elementos simbólicos
        print("Prueba de derivación de elementos...")
        set_e = SetTensor([sp.Symbol('x')**2])
        result = set_e.differentiate_elements(sp.Symbol('x'))
        expected = SetTensor([2 * sp.Symbol('x')])
        print(f"Resultado: {result.elements}, Esperado: {expected.elements}")
        self.assertEqual(result.elements, expected.elements)

    def test_expand_elements(self):
        # Prueba la expansión de elementos simbólicos
        print("Prueba de expansión de elementos...")
        set_f = SetTensor([sp.Symbol('x') * (sp.Symbol('x') + 1)])
        result = set_f.expand_elements()
        expected = SetTensor([sp.Symbol('x')**2 + sp.Symbol('x')])
        print(f"Resultado: {result.elements}, Esperado: {expected.elements}")
        self.assertEqual(result.elements, expected.elements)

    def test_factor_elements(self):
        # Prueba la factorización de elementos simbólicos
        print("Prueba de factorización de elementos...")
        set_g = SetTensor([sp.Symbol('x')**2 - 1])
        result = set_g.factor_elements()
        expected = SetTensor([(sp.Symbol('x') - 1) * (sp.Symbol('x') + 1)])
        print(f"Resultado: {result.elements}, Esperado: {expected.elements}")
        self.assertEqual(result.elements, expected.elements)

    def test_solve_system(self):
        # Prueba la resolución de un sistema de ecuaciones
        print("Prueba de resolución de sistema de ecuaciones...")
        eq1 = sp.Eq(sp.Symbol('x') + sp.Symbol('y'), 2)
        eq2 = sp.Eq(sp.Symbol('x') - sp.Symbol('y'), 0)
        solutions = self.set_a.solve_system([eq1, eq2], [sp.Symbol('x'), sp.Symbol('y')])
        expected = [sp.FiniteSet(1, 1)]
        print(f"Resultado: {solutions}, Esperado: {expected}")
        self.assertEqual(solutions, expected)

    def test_harmonic_oscillator(self):
        # Prueba del oscilador armónico
        print("Prueba de oscilador armónico...")
        try:
            m, k, x0, v0 = 1, 1, 1, 0
            result = self.set_a.harmonic_oscillator(m, k, x0, v0)
            t = sp.symbols('t')
            # Usar simplificación para normalizar las expresiones
            result = result.simplify()
            expected = sp.cos(t)
            print(f"Resultado: {result}, Esperado: {expected}")
            self.assertEqual(result, expected)
        except Exception as e:
            self.fail(f"Error en harmonic_oscillator: {e}")

    def test_harmonic_oscillator(self):
        # Prueba del oscilador armónico
        print("Prueba de oscilador armónico...")
        try:
            # Definimos la función del oscilador
            result = self.set_a.harmonic_oscillator(m=1, k=1, x0=1, v0=0)
            t = sp.symbols('t')
            expected = sp.cos(t)
            # Simplificar para asegurar comparabilidad simbólica
            result = result.simplify()
            expected = expected.simplify()
            print(f"Resultado: {result}, Esperado: {expected}")
            if not result.equals(expected):
                raise AssertionError(f"La comparación falló: Resultado {result} no es igual a Esperado {expected}")
            self.assertTrue(result.equals(expected))
        except Exception as e:
            self.fail(f"Error en harmonic_oscillator: {e}")

    def test_lorentz_transformation(self):
        # Prueba de transformación de Lorentz
        print("Prueba de transformación de Lorentz...")
        try:
            beta = 0.5
            gamma = 1 / sp.sqrt(1 - beta**2)
            result = self.set_a.lorentz_transformation(beta, 'x').evalf()
            expected = sp.Matrix([
                [gamma, -gamma * beta, 0, 0],
                [-gamma * beta, gamma, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]).evalf()

            # Comparar cada elemento de las matrices con una pequeña tolerancia
            matrices_equal = all(
                [abs(result[i] - expected[i]) < 1e-10 for i in range(result.shape[0] * result.shape[1])]
            )
            print(f"Resultado: {result}, Esperado: {expected}")
            self.assertTrue(matrices_equal)
        except Exception as e:
            self.fail(f"Error en lorentz_transformation: {e}")


    def test_generate_fibonacci_series(self):
        # Prueba de generación de serie de Fibonacci
        print("Prueba de generación de serie de Fibonacci...")
        try:
            result = self.set_a.generate_fibonacci_series(5)
            x = sp.symbols('x')
            # La serie generada comienza desde el término x, no incluye un término constante
            expected = x + x**2 + 2*x**3 + 3*x**4 + 5*x**5
            print(f"Resultado: {result}, Esperado: {expected}")
            self.assertEqual(result.simplify(), expected.simplify())
        except Exception as e:
            self.fail(f"Error en generate_fibonacci_series: {e}")

    def test_matrix_fib(self):
        # Prueba del cálculo de Fibonacci con matrices
        print("Prueba de cálculo de Fibonacci con matrices...")
        try:
            result = self.set_a.matrix_fib(5)
            expected = 5
            print(f"Resultado: {result}, Esperado: {expected}")
            self.assertEqual(result, expected)
        except Exception as e:
            self.fail(f"Error en matrix_fib: {e}")

    def test_binet_formula(self):
        # Prueba de la fórmula de Binet
        print("Prueba de fórmula de Binet...")
        try:
            result = self.set_a.binet_formula(5)
            expected = 5
            print(f"Resultado: {result}, Esperado: {expected}")
            self.assertEqual(result, expected)
        except Exception as e:
            self.fail(f"Error en binet_formula: {e}")

    def test_derivative_of_generator_function(self):
        # Prueba de derivada de la función generadora de Fibonacci
        print("Prueba de derivada de función generadora de Fibonacci...")
        try:
            result = self.set_a.derivative_of_generator_function()
            x = sp.symbols('x')
            expected = x * (2 * x + 1) / (-x**2 - x + 1)**2 + 1 / (-x**2 - x + 1)
            print(f"Resultado: {result.simplify()}, Esperado: {expected.simplify()}")
            self.assertEqual(result.simplify(), expected.simplify())
        except Exception as e:
            self.fail(f"Error en derivative_of_generator_function: {e}")

if __name__ == '__main__':
    unittest.main()
