Documentación de set_tensor.py
Descripción General
La librería set_tensor.py proporciona una clase llamada SetTensor que facilita la manipulación y operaciones avanzadas con conjuntos de elementos simbólicos utilizando las bibliotecas sympy y numpy. Además, incluye diversas herramientas para cálculos matemáticos y análisis simbólico.

Clases y Métodos
Clase: InvalidElementStructureError
Excepción personalizada utilizada para manejar errores relacionados con la estructura de elementos en la clase SetTensor.

Método:
__init__(self, message: str): Inicializa la excepción con un mensaje específico.
Clase: SetTensor
Esta clase permite la creación y manipulación de conjuntos de elementos simbólicos, proporcionando métodos para realizar diversas operaciones matemáticas y simbólicas.

Constructor
__init__(self, elements): Inicializa un objeto SetTensor con los elementos dados, que pueden ser una lista, FiniteSet, Interval o Union.
Parámetros:
elements: Elementos del conjunto, aceptando varios tipos de estructuras simbólicas.
Excepciones:
InvalidElementStructureError: Lanzada si los elementos no son de un tipo válido.
Métodos
elements: Propiedad para obtener los elementos del SetTensor.

union(other): Realiza la unión de dos conjuntos SetTensor.

Parámetros:
other: Otro objeto SetTensor.
Retorno: Nuevo SetTensor con la unión de los conjuntos.
intersect(other): Realiza la intersección de dos conjuntos.

Parámetros:
other: Otro objeto SetTensor.
Retorno: Nuevo SetTensor con la intersección de los conjuntos.
difference(other): Calcula la diferencia entre dos conjuntos.

Parámetros:
other: Otro objeto SetTensor.
Retorno: Nuevo SetTensor con la diferencia de los conjuntos.
Excepciones:
InvalidElementStructureError: Lanzada si la operación devuelve un tipo no soportado.
is_subset(other): Verifica si el conjunto actual es un subconjunto de otro conjunto.

Parámetros:
other: Otro objeto SetTensor.
Retorno: True si es un subconjunto, False de lo contrario.
symmetric_difference(other): Calcula la diferencia simétrica entre dos conjuntos.

Parámetros:
other: Otro objeto SetTensor.
Retorno: Nuevo SetTensor con la diferencia simétrica.
complement(universal_set): Calcula el complemento respecto a un conjunto universal.

Parámetros:
universal_set: Conjunto universal de referencia.
Retorno: Nuevo SetTensor con el complemento.
power_set(): Calcula el conjunto potencia del conjunto actual.

Retorno: Lista de objetos SetTensor representando cada subconjunto.
apply_function(func): Aplica una función simbólica a cada elemento del conjunto.

Parámetros:
func: Función simbólica a aplicar.
Retorno: Nuevo SetTensor con los elementos transformados.
Excepciones:
NotImplementedError: Lanzada si la operación no es soportada para el tipo de conjunto actual.
simplify_elements(): Simplifica cada elemento simbólico del conjunto.

Retorno: Nuevo SetTensor con los elementos simplificados.
Excepciones:
NotImplementedError: Lanzada si la operación no es soportada para el tipo de conjunto actual.
solve_equations(equations, symbols): Resuelve un conjunto de ecuaciones simbólicas.

Parámetros:
equations: Lista de ecuaciones simbólicas.
symbols: Lista de símbolos que actúan como incógnitas.
Retorno: Lista de conjuntos de soluciones.
integrate_elements(symbol): Calcula la integral indefinida de cada elemento.

Parámetros:
symbol: Símbolo de integración.
Retorno: Nuevo SetTensor con las integrales.
Excepciones:
NotImplementedError: Lanzada si la operación no es soportada para el tipo de conjunto actual.
differentiate_elements(symbol): Calcula la derivada de cada elemento del conjunto.

Parámetros:
symbol: Símbolo de derivación.
Retorno: Nuevo SetTensor con las derivadas.
Excepciones:
NotImplementedError: Lanzada si la operación no es soportada para el tipo de conjunto actual.
expand_elements(): Expande cada elemento simbólico del conjunto.

Retorno: Nuevo SetTensor con los elementos expandidos.
Excepciones:
NotImplementedError: Lanzada si la operación no es soportada para el tipo de conjunto actual.
factor_elements(): Factoriza cada elemento simbólico del conjunto.

Retorno: Nuevo SetTensor con los elementos factorizados.
Excepciones:
NotImplementedError: Lanzada si la operación no es soportada para el tipo de conjunto actual.
solve_system(equations, symbols): Resuelve un sistema de ecuaciones simbólicas.

Parámetros:
equations: Lista de ecuaciones simbólicas.
symbols: Lista de símbolos que actúan como incógnitas.
Retorno: Lista de conjuntos de soluciones.
solve_ode(equation, function): Resuelve una ecuación diferencial ordinaria (EDO).

Parámetros:
equation: Ecuación diferencial simbólica.
function: Función dependiente.
Retorno: Solución simbólica de la EDO.
matrix_operations(matrix): Realiza operaciones simbólicas en matrices.

Parámetros:
matrix: Matriz simbólica.
Retorno: Diccionario con determinante, traza y matriz inversa.
harmonic_oscillator(m, k, x0, v0): Modela un oscilador armónico.

Parámetros:
m: Masa del oscilador.
k: Constante de resorte.
x0: Posición inicial.
v0: Velocidad inicial.
Retorno: Solución simbólica de la posición en función del tiempo.
optimize_function(function, symbol): Optimiza una función simbólica.

Parámetros:
function: Función simbólica a optimizar.
symbol: Símbolo respecto al cual se optimiza.
Retorno: Diccionario con puntos críticos y tipo de extremos.
jacobian(functions, variables): Calcula el Jacobiano de funciones respecto a variables.

Parámetros:
functions: Lista de funciones simbólicas.
variables: Lista de variables simbólicas.
Retorno: Matriz Jacobiana.
hessian(function, variables): Calcula el Hessiano de una función respecto a variables.

Parámetros:
function: Función simbólica.
variables: Lista de variables simbólicas.
Retorno: Matriz Hessiana.
solve_nonlinear_system(equations, variables): Resuelve un sistema de ecuaciones no lineales.

Parámetros:
equations: Lista de ecuaciones simbólicas.
variables: Lista de variables simbólicas.
Retorno: Lista de soluciones.
stability_analysis(functions, variables, equilibrium_points): Analiza la estabilidad de puntos de equilibrio.

Parámetros:
functions: Lista de funciones simbólicas.
variables: Lista de variables simbólicas.
equilibrium_points: Lista de puntos de equilibrio.
Retorno: Lista de resultados de estabilidad por punto.
transform_coordinates(points, transformation_matrix): Transforma puntos usando una matriz de transformación.

Parámetros:
points: Lista de puntos como vectores columna.
transformation_matrix: Matriz de transformación.
Retorno: Lista de puntos transformados.
simulate_dynamics(functions, variables, initial_conditions, time_symbol, duration, step_size, params): Simula un sistema dinámico a lo largo del tiempo.

Parámetros:
functions: Lista de funciones simbólicas que representan el sistema.
variables: Lista de variables del sistema.
initial_conditions: Diccionario con condiciones iniciales para cada variable.
time_symbol: Símbolo que representa el tiempo.
duration: Duración de la simulación.
step_size: Tamaño del paso de tiempo.
params: Diccionario de parámetros numéricos del sistema.
Retorno: Lista de estados del sistema a lo largo del tiempo.
explore_dynamics_with_variations(functions, variables, initial_conditions_list, time_symbol, duration, step_size): Explora la dinámica con diferentes condiciones iniciales.

Parámetros:
functions: Lista de funciones simbólicas del sistema.
variables: Lista de variables del sistema.
initial_conditions_list: Lista de diccionarios con diferentes condiciones iniciales.
time_symbol: Símbolo que representa el tiempo.
duration: Duración de la simulación.
step_size: Tamaño del paso de tiempo.
simulate_discrete_system(update_rule, initial_state, steps): Simula un sistema en tiempo discreto.

Parámetros:
update_rule: Función que define la regla de actualización.
initial_state: Estado inicial del sistema.
steps: Número de pasos de simulación.
Retorno: Lista de estados a lo largo de los pasos.
global_optimization(function, initial_guess): Realiza la optimización global de una función.

Parámetros:
function: Función a optimizar.
initial_guess: Estimación inicial para la optimización.
Retorno: Resultado de la optimización.
simulate_sir(beta, gamma, initial_conditions, duration, step_size): Simula un modelo epidemiológico SIR.

Parámetros:
beta: Tasa de transmisión.
gamma: Tasa de recuperación.
initial_conditions: Diccionario con condiciones iniciales para S, I, y R.
duration: Duración de la simulación.
step_size: Tamaño del paso de tiempo.
Retorno: Lista de estados del sistema a lo largo del tiempo.
sensitivity_analysis(functions, variables, params, param_values, initial_conditions, time_symbol, duration, step_size): Realiza un análisis de sensibilidad.

Parámetros:
functions: Lista de funciones simbólicas del sistema.
variables: Lista de variables del sistema.
params: Lista de parámetros a analizar.
param_values: Lista de valores para cada parámetro.
initial_conditions: Diccionario con condiciones iniciales.
time_symbol: Símbolo que representa el tiempo.
duration: Duración de la simulación.
step_size: Tamaño del paso de tiempo.
Retorno: Diccionario con resultados del análisis de sensibilidad.
bifurcation_analysis(functions, variables, param, param_range, initial_conditions, time_symbol, duration, step_size): Realiza un análisis de bifurcaciones.

Parámetros:
functions: Lista de funciones simbólicas del sistema.
variables: Lista de variables del sistema.
param: Parámetro a variar.
param_range: Rango de valores para el parámetro.
initial_conditions: Diccionario con condiciones iniciales.
time_symbol: Símbolo que representa el tiempo.
duration: Duración de la simulación.
step_size: Tamaño del paso de tiempo.
Retorno: Lista de resultados de bifurcación.
add_element(element): Añade un elemento al conjunto.

Parámetros:
element: Elemento a añadir.
lorentz_transformation(beta_value, direction='x'): Calcula la matriz de transformación de Lorentz.

Parámetros:
beta_value: Velocidad relativa (v/c).
direction: Dirección de la transformación ('x', 'y', 'z').
Retorno: Matriz de Lorentz para la dirección dada.
generate_fibonacci_series(n_terms): Genera la serie de Fibonacci utilizando la función generadora.

Parámetros:
n_terms: Número de términos de la serie de Fibonacci a generar.
Retorno: Serie de Fibonacci.
matrix_fib(n): Calcula el enésimo número de Fibonacci utilizando la exponenciación de matrices.

Parámetros:
n: Índice del número de Fibonacci a calcular.
Retorno: El enésimo número de Fibonacci.
binet_formula(n): Calcula una aproximación rápida del enésimo número de Fibonacci usando la fórmula de Binet.

Parámetros:
n: Índice del número de Fibonacci a calcular.
Retorno: Aproximación del enésimo número de Fibonacci.
derivative_of_generator_function(): Calcula la derivada de la función generadora de Fibonacci.

Retorno: Derivada de la función generadora.