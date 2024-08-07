# set_tensor.py

import numpy as np
import sympy as sp
from typing import Dict, Union, List, Callable
from scipy.optimize import minimize

class InvalidElementStructureError(Exception):
    """Excepción para manejar errores en la estructura de elementos del conjunto."""
    def __init__(self, message: str):
        super().__init__(message)

class SetTensor:
    def __init__(self, elements: Union[List[Union[sp.Basic, str, float, int]], sp.FiniteSet, sp.Interval, sp.Union]):
        """
        Inicializa el SetTensor con los elementos dados.

        :param elements: Elementos del conjunto que pueden ser una lista, FiniteSet, Interval o Union.
        :raises InvalidElementStructureError: Si los elementos no son de un tipo válido.
        """
        self._elements = self._initialize_elements(elements)

    @property
    def elements(self):
        """Propiedad para obtener los elementos del SetTensor."""
        return self._elements

    def _initialize_elements(self, elements: Union[List[Union[sp.Basic, str, float, int]], sp.FiniteSet, sp.Interval, sp.Union]) -> Union[sp.FiniteSet, sp.Interval, sp.Union]:
        """
        Inicializa los elementos del conjunto basado en el tipo de entrada.

        :param elements: Elementos del conjunto en diferentes formatos.
        :return: Elementos como un FiniteSet, Interval o Union.
        :raises InvalidElementStructureError: Si el tipo de elementos es inválido.
        """
        if isinstance(elements, list):
            return sp.FiniteSet(*elements)
        elif isinstance(elements, (sp.FiniteSet, sp.Interval, sp.Union)):
            return elements
        else:
            raise InvalidElementStructureError("Elements must be a list, a FiniteSet, Interval, or Union.")

    def union(self, other: 'SetTensor') -> 'SetTensor':
        """
        Realiza la unión de dos conjuntos.

        :param other: Otro objeto SetTensor para unir.
        :return: Nuevo objeto SetTensor con la unión de los conjuntos.
        """
        return SetTensor(self._elements.union(other.elements))

    def intersect(self, other: 'SetTensor') -> 'SetTensor':
        """
        Realiza la intersección de dos conjuntos.

        :param other: Otro objeto SetTensor para intersectar.
        :return: Nuevo objeto SetTensor con la intersección de los conjuntos.
        """
        intersection = self._elements.intersect(other.elements)
        return SetTensor(intersection)

    def difference(self, other: 'SetTensor') -> 'SetTensor':
        """
        Calcula la diferencia entre dos conjuntos.

        :param other: Otro objeto SetTensor para calcular la diferencia.
        :return: Nuevo objeto SetTensor con la diferencia de los conjuntos.
        """
        difference = self._elements - other.elements
        
        # Verifica si la diferencia es un conjunto vacío
        if difference.is_empty:
            return SetTensor([])  # Devuelve un conjunto vacío
        elif isinstance(difference, (sp.FiniteSet, sp.Interval, sp.Union)):
            return SetTensor(difference)
        else:
            raise InvalidElementStructureError("Difference operation returned an unsupported type.")

    def is_subset(self, other: 'SetTensor') -> bool:
        """
        Verifica si el conjunto actual es un subconjunto de otro conjunto.

        :param other: Otro objeto SetTensor para verificar.
        :return: True si es un subconjunto, False de lo contrario.
        """
        return self._elements.is_subset(other.elements)

    def symmetric_difference(self, other: 'SetTensor') -> 'SetTensor':
        """
        Calcula la diferencia simétrica entre dos conjuntos.

        :param other: Otro objeto SetTensor para calcular la diferencia simétrica.
        :return: Nuevo objeto SetTensor con la diferencia simétrica de los conjuntos.
        """
        symmetric_diff = self._elements.symmetric_difference(other.elements)
        return SetTensor(symmetric_diff)

    def complement(self, universal_set: 'SetTensor') -> 'SetTensor':
        """
        Calcula el complemento del conjunto actual respecto a un conjunto universal.

        :param universal_set: El conjunto universal respecto al cual calcular el complemento.
        :return: Nuevo objeto SetTensor con el complemento del conjunto.
        """
        complement_set = universal_set.elements - self._elements
        return SetTensor(complement_set)

    def power_set(self) -> List['SetTensor']:
        """
        Calcula el conjunto potencia del conjunto actual.

        :return: Lista de objetos SetTensor que representan cada subconjunto.
        """
        elements_list = list(self._elements)
        power_set_list = []

        # Genera todas las combinaciones posibles de subconjuntos
        for i in range(2**len(elements_list)):
            subset = [elements_list[j] for j in range(len(elements_list)) if (i & (1 << j))]
            power_set_list.append(SetTensor(subset))

        return power_set_list

    def apply_function(self, func: Callable[[sp.Basic], sp.Basic]) -> 'SetTensor':
        """
        Aplica una función simbólica a cada elemento del conjunto.

        :param func: Función simbólica para aplicar.
        :return: Nuevo objeto SetTensor con los resultados.
        """
        if isinstance(self._elements, sp.FiniteSet):
            transformed_elements = sp.FiniteSet(*[func(e) for e in self._elements])
            return SetTensor(transformed_elements)
        raise NotImplementedError("Function application is only supported for FiniteSet.")

    def simplify_elements(self) -> 'SetTensor':
        """
        Simplifica cada elemento simbólico del conjunto.

        :return: Nuevo objeto SetTensor con los elementos simplificados.
        """
        if isinstance(self._elements, sp.FiniteSet):
            simplified_elements = sp.FiniteSet(*[sp.simplify(e) for e in self._elements])
            return SetTensor(simplified_elements)
        raise NotImplementedError("Simplification is only supported for FiniteSet.")

    def solve_equations(self, equations: List[sp.Eq], symbols: List[sp.Symbol]) -> List[sp.FiniteSet]:
        """
        Resuelve un conjunto de ecuaciones simbólicas donde los elementos del conjunto son incógnitas.

        :param equations: Lista de ecuaciones simbólicas.
        :param symbols: Lista de símbolos que actúan como incógnitas.
        :return: Lista de conjuntos de soluciones.
        """
        solutions = []
        for equation in equations:
            solution = sp.solve(equation, *symbols)
            solutions.append(sp.FiniteSet(*solution))
        return solutions

    def integrate_elements(self, symbol: sp.Symbol) -> 'SetTensor':
        """
        Calcula la integral indefinida de cada elemento del conjunto.

        :param symbol: El símbolo de integración.
        :return: Nuevo objeto SetTensor con las integrales calculadas.
        """
        if isinstance(self._elements, sp.FiniteSet):
            integrated_elements = sp.FiniteSet(*[sp.integrate(e, symbol) for e in self._elements])
            return SetTensor(integrated_elements)
        raise NotImplementedError("Integration is only supported for FiniteSet.")

    def differentiate_elements(self, symbol: sp.Symbol) -> 'SetTensor':
        """
        Calcula la derivada de cada elemento del conjunto.

        :param symbol: El símbolo de derivación.
        :return: Nuevo objeto SetTensor con las derivadas calculadas.
        """
        if isinstance(self._elements, sp.FiniteSet):
            differentiated_elements = sp.FiniteSet(*[sp.diff(e, symbol) for e in self._elements])
            return SetTensor(differentiated_elements)
        raise NotImplementedError("Differentiation is only supported for FiniteSet.")

    def expand_elements(self) -> 'SetTensor':
        """
        Expande cada elemento simbólico del conjunto.

        :return: Nuevo objeto SetTensor con los elementos expandidos.
        """
        if isinstance(self._elements, sp.FiniteSet):
            expanded_elements = sp.FiniteSet(*[sp.expand(e) for e in self._elements])
            return SetTensor(expanded_elements)
        raise NotImplementedError("Expansion is only supported for FiniteSet.")

    def factor_elements(self) -> 'SetTensor':
        """
        Factoriza cada elemento simbólico del conjunto.

        :return: Nuevo objeto SetTensor con los elementos factorizados.
        """
        if isinstance(self._elements, sp.FiniteSet):
            factored_elements = sp.FiniteSet(*[sp.factor(e) for e in self._elements])
            return SetTensor(factored_elements)
        raise NotImplementedError("Factorization is only supported for FiniteSet.")

    # ===========================================================================

    def solve_system(self, equations: List[sp.Eq], symbols: List[sp.Symbol]) -> List[sp.FiniteSet]:
        """
        Resuelve un sistema de ecuaciones simbólicas.

        :param equations: Lista de ecuaciones simbólicas.
        :param symbols: Lista de símbolos que actúan como incógnitas.
        :return: Lista de conjuntos de soluciones.
        """
        solutions = sp.solve(equations, symbols, dict=True)
        return [sp.FiniteSet(*(sol[sym] for sym in symbols)) for sol in solutions]
    
    def solve_ode(self, equation: sp.Eq, function: sp.Function) -> sp.Expr:
        """
        Resuelve una ecuación diferencial ordinaria (EDO).

        :param equation: Ecuación diferencial simbólica.
        :param function: Función dependiente.
        :return: Solución simbólica de la EDO.
        """
        solution = sp.dsolve(equation, function)
        return solution
    
    def matrix_operations(self, matrix: sp.Matrix) -> dict:
        """
        Realiza operaciones simbólicas en matrices.

        :param matrix: Matriz simbólica.
        :return: Diccionario con determinante, traza y matriz inversa.
        """
        det = matrix.det()
        trace = matrix.trace()
        inverse = matrix.inv() if matrix.det() != 0 else "No invertible"
        return {'det': det, 'trace': trace, 'inverse': inverse}


    def harmonic_oscillator(self, m: Union[sp.Symbol, int], k: Union[sp.Symbol, int], x0: Union[sp.Symbol, int], v0: Union[sp.Symbol, int]) -> sp.Expr:
        """
        Modela un oscilador armónico.

        :param m: Masa del oscilador.
        :param k: Constante de resorte.
        :param x0: Posición inicial.
        :param v0: Velocidad inicial.
        :return: Solución simbólica de la posición en función del tiempo.
        """
        t = sp.symbols('t')
        # Convertir a enteros simbólicos para garantizar el manejo simbólico
        m = sp.Rational(m)
        k = sp.Rational(k)
        omega = sp.sqrt(k/m).simplify()  # Asegúrate de que omega se simplifique simbólicamente
        x = x0 * sp.cos(omega * t) + (v0/omega) * sp.sin(omega * t)
        return x.simplify()  # Simplificar el resultado final para manejar correctamente los tipos




    def optimize_function(self, function: sp.Expr, symbol: sp.Symbol) -> dict:
        """
        Optimiza una función simbólica para encontrar máximos y mínimos.

        :param function: Función simbólica a optimizar.
        :param symbol: Símbolo respecto al cual se optimiza.
        :return: Diccionario con puntos críticos y tipo de extremos.
        """
        critical_points = sp.solve(sp.diff(function, symbol), symbol)
        second_derivatives = [sp.diff(function, symbol, 2).subs(symbol, cp) for cp in critical_points]
        extrema_types = ['max' if sd < 0 else 'min' if sd > 0 else 'inflection' for sd in second_derivatives]
        return {'critical_points': critical_points, 'types': extrema_types}


    # =====================================================================

    def jacobian(self, functions: List[sp.Expr], variables: List[sp.Symbol]) -> sp.Matrix:
        """
        Calcula el Jacobiano de un conjunto de funciones respecto a las variables dadas.

        :param functions: Lista de funciones simbólicas.
        :param variables: Lista de variables simbólicas.
        :return: Matriz Jacobiana.
        """
        return sp.Matrix([[f.diff(var) for var in variables] for f in functions])


    def hessian(self, function: sp.Expr, variables: List[sp.Symbol]) -> sp.Matrix:
        """
        Calcula el Hessiano de una función respecto a las variables dadas.

        :param function: Función simbólica.
        :param variables: Lista de variables simbólicas.
        :return: Matriz Hessiana.
        """
        return sp.hessian(function, variables)


    def solve_nonlinear_system(self, equations: List[sp.Eq], variables: List[sp.Symbol]) -> List[dict]:
        """
        Resuelve un sistema de ecuaciones no lineales.

        :param equations: Lista de ecuaciones simbólicas.
        :param variables: Lista de variables simbólicas.
        :return: Lista de soluciones.
        """
        solutions = sp.nonlinsolve(equations, variables)
        return [sol for sol in solutions]
    

    def stability_analysis(self, functions: List[sp.Expr], variables: List[sp.Symbol], equilibrium_points: List[List[float]]) -> List[str]:
        """
        Analiza la estabilidad de puntos de equilibrio de un sistema dinámico.

        :param functions: Lista de funciones simbólicas.
        :param variables: Lista de variables simbólicas.
        :param equilibrium_points: Lista de puntos de equilibrio.
        :return: Lista de resultados de estabilidad por punto.
        """
        jacobian_matrix = self.jacobian(functions, variables)
        stability_results = []

        for point in equilibrium_points:
            evaluated_matrix = jacobian_matrix.subs({var: val for var, val in zip(variables, point)})
            eigenvalues = evaluated_matrix.eigenvals()
            real_parts = [ev.as_real_imag()[0] for ev in eigenvalues.keys()]

            if all(rp < 0 for rp in real_parts):
                stability_results.append("Estable")
            elif any(rp > 0 for rp in real_parts):
                stability_results.append("Inestable")
            else:
                stability_results.append("Indeterminado")

        return stability_results
    
    def transform_coordinates(self, points: List[sp.Matrix], transformation_matrix: sp.Matrix) -> List[sp.Matrix]:
        """
        Transforma puntos usando una matriz de transformación.

        :param points: Lista de puntos como vectores columna.
        :param transformation_matrix: Matriz de transformación.
        :return: Lista de puntos transformados.
        """
        return [transformation_matrix * point for point in points]    

    # =======================================================================

    def simulate_dynamics(self, functions, variables, initial_conditions, time_symbol, duration, step_size, params):
            """
            Simula un sistema dinámico a lo largo del tiempo.

            :param functions: Lista de funciones simbólicas que representan el sistema.
            :param variables: Lista de variables del sistema.
            :param initial_conditions: Diccionario con condiciones iniciales para cada variable.
            :param time_symbol: Símbolo que representa el tiempo.
            :param duration: Duración de la simulación.
            :param step_size: Tamaño del paso de tiempo.
            :param params: Diccionario de parámetros numéricos del sistema.
            :return: Lista de estados del sistema a lo largo del tiempo.
            """
            from scipy.integrate import odeint
            import numpy as np
            
            def system(state, t):
                subs = {var: val for var, val in zip(variables, state)}
                subs[time_symbol] = t
                subs.update(params)  # Asegúrate de incluir todos los parámetros

                # Asegura que todas las variables y parámetros están sustituidos
                try:
                    derivatives = [float(f.subs(subs).evalf()) for f in functions]
                except Exception as e:
                    print("Error al evaluar las funciones:")
                    print(f"Subs: {subs}")
                    print(f"Error: {e}")
                    raise TypeError(f"No se puede convertir la expresión a float. Funciones: {functions}, Sustituciones: {subs}")

                return derivatives
            
            initial_state = [initial_conditions[var] for var in variables]
            time_points = np.arange(0, duration, step_size)
            solution = odeint(system, initial_state, time_points)
            
            return [{'time': t, 'state': dict(zip(variables, s))} for t, s in zip(time_points, solution)]

    def explore_dynamics_with_variations(self, functions: List[sp.Expr], variables: List[sp.Symbol], initial_conditions_list: List[dict], time_symbol: sp.Symbol, duration: float, step_size: float) -> None:
        """
        Explora la dinámica del sistema con diferentes condiciones iniciales.

        :param functions: Lista de funciones simbólicas del sistema.
        :param variables: Lista de variables del sistema.
        :param initial_conditions_list: Lista de diccionarios con diferentes condiciones iniciales.
        :param time_symbol: Símbolo que representa el tiempo.
        :param duration: Duración de la simulación.
        :param step_size: Tamaño del paso de tiempo.
        """
        for initial_conditions in initial_conditions_list:
            dynamics = self.simulate_dynamics(functions, variables, initial_conditions, time_symbol, duration, step_size, {})
            print(f"\nCondiciones iniciales: {initial_conditions}")
            for point in dynamics[:5]:  # Mostrar solo los primeros 5 puntos
                print(point)

    def simulate_discrete_system(self, update_rule: Callable[[List[float]], List[float]], initial_state: List[float], steps: int) -> List[List[float]]:
        """
        Simula un sistema en tiempo discreto.

        :param update_rule: Función que define la regla de actualización del sistema.
        :param initial_state: Estado inicial del sistema.
        :param steps: Número de pasos de simulación.
        :return: Lista de estados a lo largo de los pasos.
        """
        state = initial_state
        states = [state]
        
        for _ in range(steps):
            state = update_rule(state)
            states.append(state)
        
        return states


    def global_optimization(self, function: Callable[[np.ndarray], float], initial_guess: np.ndarray) -> dict:
        """
        Realiza la optimización global de una función.

        :param function: Función a optimizar.
        :param initial_guess: Estimación inicial para la optimización.
        :return: Resultado de la optimización.
        """
        result = minimize(function, initial_guess, method='L-BFGS-B')
        return {'point': result.x, 'value': result.fun, 'success': result.success}


    def simulate_sir(self, beta: float, gamma: float, initial_conditions: dict, duration: float, step_size: float) -> List[dict]:
        """
        Simula un modelo epidemiológico SIR.

        :param beta: Tasa de transmisión.
        :param gamma: Tasa de recuperación.
        :param initial_conditions: Diccionario con condiciones iniciales para S, I, y R.
        :param duration: Duración de la simulación.
        :param step_size: Tamaño del paso de tiempo.
        :return: Lista de estados del sistema a lo largo del tiempo.
        """
        from scipy.integrate import odeint

        if beta < 0 or gamma < 0:
            raise ValueError("Las tasas de transmisión y recuperación deben ser no negativas.")

        def sir_model(y, t, beta, gamma):
            S, I, R = y
            dSdt = -beta * S * I
            dIdt = beta * S * I - gamma * I
            dRdt = gamma * I
            return [dSdt, dIdt, dRdt]

        initial_state = [initial_conditions['S'], initial_conditions['I'], initial_conditions['R']]
        
        if sum(initial_state) != 1.0:
            raise ValueError("La suma de las condiciones iniciales S, I y R debe ser igual a 1.")

        time_points = np.arange(0, duration + step_size, step_size)  # Asegúrate de incluir el punto final
        solution = odeint(sir_model, initial_state, time_points, args=(beta, gamma))

        return [{'time': t, 'state': {'S': s, 'I': i, 'R': r}} for t, (s, i, r) in zip(time_points, solution)]



    def sensitivity_analysis(self, functions: List, variables: List, params: List, param_values: List[float], initial_conditions: Dict, time_symbol, duration: float, step_size: float) -> dict:
        """
        Realiza un análisis de sensibilidad variando los parámetros uno por uno.
        """
        sensitivity_results = {}

        for i, param in enumerate(params):
            baseline_value = param_values[i]
            perturbed_value = baseline_value * 1.01  # Incremento del 1%

            try:
                # Diccionario de parámetros para simulación
                param_dict = {params[0]: param_values[0], params[1]: param_values[1]}

                # Simulación con valor base
                baseline_conditions = initial_conditions.copy()
                param_dict[param] = baseline_value  # Asegúrate de incluir el valor correcto para el parámetro
                baseline_dynamics = self.simulate_dynamics(functions, variables, baseline_conditions, time_symbol, duration, step_size, param_dict)

                # Simulación con valor perturbado
                perturbed_conditions = initial_conditions.copy()
                param_dict[param] = perturbed_value
                perturbed_dynamics = self.simulate_dynamics(functions, variables, perturbed_conditions, time_symbol, duration, step_size, param_dict)

                # Evaluar sensibilidad
                sensitivity_results[param] = {
                    'baseline': baseline_dynamics[-1]['state'],
                    'perturbed': perturbed_dynamics[-1]['state'],
                    'change': {var: (perturbed_dynamics[-1]['state'][var] - baseline_dynamics[-1]['state'][var]) for var in variables}
                }
            except TypeError as e:
                print(f"Error de tipo: {e}")
                print(f"Faltan parámetros o sustituciones para {param}.")
            except Exception as e:
                print(f"Ocurrió un error durante la simulación: {e}")
                raise

        return sensitivity_results

    def bifurcation_analysis(self, functions: List, variables: List, param, param_range: List[float], initial_conditions: Dict, time_symbol, duration: float, step_size: float) -> List[dict]:
        """
        Realiza un análisis de bifurcaciones variando un parámetro sobre un rango.
        """
        bifurcation_results = []

        for value in param_range:
            conditions = initial_conditions.copy()
            param_dict = {param: value, sp.symbols('gamma'): 0.1}  # Asegúrate de incluir todos los parámetros necesarios
            dynamics = self.simulate_dynamics(functions, variables, conditions, time_symbol, duration, step_size, param_dict)
            final_state = dynamics[-1]['state']
            bifurcation_results.append({'param_value': value, 'final_state': final_state})

        return bifurcation_results
    
    def add_element(self, element):
        self._elements.append(element)

    def lorentz_transformation(self, beta_value, direction='x'):
        """
        Calcula la matriz de transformación de Lorentz en la dirección especificada.

        :param beta_value: Velocidad relativa (v/c).
        :param direction: Dirección de la transformación ('x', 'y', 'z').
        :return: Matriz de Lorentz para la dirección dada.
        """
        beta = sp.Symbol('beta')
        gamma = 1 / sp.sqrt(1 - beta**2)
        if direction == 'x':
            L = sp.Matrix([
                [gamma, -gamma * beta, 0, 0],
                [-gamma * beta, gamma, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        elif direction == 'y':
            L = sp.Matrix([
                [gamma, 0, -gamma * beta, 0],
                [0, 1, 0, 0],
                [-gamma * beta, 0, gamma, 0],
                [0, 0, 0, 1]
            ])
        elif direction == 'z':
            L = sp.Matrix([
                [gamma, 0, 0, -gamma * beta],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [-gamma * beta, 0, 0, gamma]
            ])
        else:
            raise ValueError("Direction must be one of 'x', 'y', or 'z'")
        
        return L.subs(beta, beta_value)

    def generate_fibonacci_series(self, n_terms):
        """
        Genera la serie de Fibonacci utilizando la función generadora.

        :param n_terms: Número de términos de la serie de Fibonacci a generar.
        :return: Serie de Fibonacci.
        """
        x = sp.symbols('x')
        G = x / (1 - x - x**2)
        fibonacci_series = G.series(x, 0, n_terms + 1).removeO()
        return fibonacci_series

    def matrix_fib(self, n):
        """
        Calcula el enésimo número de Fibonacci utilizando la exponenciación de matrices.

        :param n: Índice del número de Fibonacci a calcular.
        :return: El enésimo número de Fibonacci.
        """
        F = np.array([[1, 1], [1, 0]], dtype=object)
        return np.linalg.matrix_power(F, n-1)[0, 0]

    def binet_formula(self, n):
        """
        Calcula una aproximación rápida del enésimo número de Fibonacci usando la fórmula de Binet.

        :param n: Índice del número de Fibonacci a calcular.
        :return: Aproximación del enésimo número de Fibonacci.
        """
        phi = (1 + np.sqrt(5)) / 2
        psi = (1 - np.sqrt(5)) / 2
        fib_n = (phi**n - psi**n) / np.sqrt(5)
        return int(round(fib_n))

    def derivative_of_generator_function(self):
        """
        Calcula la derivada de la función generadora de Fibonacci.

        :return: Derivada de la función generadora.
        """
        x = sp.symbols('x')
        G = x / (1 - x - x**2)
        G_derivative = sp.diff(G, x)
        return G_derivative
