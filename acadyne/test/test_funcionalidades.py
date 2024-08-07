from acadyne.tensor.set_tensor import SetTensor
from acadyne.core.tensor_bionico import BionicTensor
import sympy as sp


def demonstrate_set_tensor_operations():
    # Crear instancias de SetTensor con diferentes elementos
    set1 = SetTensor([1, 2, 3])
    set2 = SetTensor([3, 4, 5])

    # Realizar operaciones de conjuntos
    union_set = set1.union(set2)
    intersection_set = set1.intersect(set2)
    difference_set = set1.difference(set2)

    print("Operaciones de Conjuntos:")
    print(f"Unión: {union_set.elements}")
    print(f"Intersección: {intersection_set.elements}")
    print(f"Diferencia (set1 - set2): {difference_set.elements}\n")

def solve_symbolic_equations():
    # Definir un símbolo
    x = sp.symbols('x')
    
    # Crear un conjunto con el símbolo
    set1 = SetTensor([x])
    
    # Definir ecuaciones y resolverlas
    equations = [sp.Eq(x**2, 1)]
    solutions = set1.solve_equations(equations, [x])

    print("Solución de Ecuaciones Simbólicas:")
    print(f"Soluciones de las ecuaciones x^2 = 1: {solutions}\n")

def demonstrate_lorentz_transformation():
    # Valor de beta para la transformación de Lorentz
    beta_value = 0.5  # Velocidad relativa (v/c)

    # Crear una instancia de BionicTensor y calcular la matriz de Lorentz
    bionic_tensor = BionicTensor()
    lorentz_matrix = bionic_tensor.lorentz_transformation(beta_value, direction='x')

    print("Transformación de Lorentz:")
    print("Matriz de Transformación de Lorentz en la dirección x:")
    print(lorentz_matrix, "\n")

def simulate_sir_model():
    # Parámetros del modelo SIR
    beta = 0.3  # Tasa de transmisión
    gamma = 0.1  # Tasa de recuperación

    # Condiciones iniciales y configuración de la simulación
    initial_conditions = {'S': 0.99, 'I': 0.01, 'R': 0.0}
    duration = 100  # Días
    step_size = 1.0

    # Ejecutar simulación
    sir_model = SetTensor([])
    sir_results = sir_model.simulate_sir(beta, gamma, initial_conditions, duration, step_size)

    print("Simulación del Modelo SIR:")
    for result in sir_results[:5]:  # Mostrar los primeros 5 días
        print(result)
    print("\n")

if __name__ == "__main__":
    print("Demostraciones de la Biblioteca Acapulco Dynamic\n")
    
    # Demostrar operaciones de SetTensor
    demonstrate_set_tensor_operations()
    
    # Resolver ecuaciones simbólicas
    solve_symbolic_equations()
    
    # Demostrar transformación de Lorentz
    demonstrate_lorentz_transformation()
    
    # Simular el modelo SIR
    simulate_sir_model()
