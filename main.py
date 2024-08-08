import sympy as sp
import numpy as np
from acadyne.tensores.dynamic_tensor import DynamicTensor
from acadyne.tensores.set_tensor import SetTensor
from acadyne.tensores.lorentz_tensor import LorentzTransformation

def show_menu():
    print("Menu:")
    print("1. Operaciones con DynamicTensor")
    print("2. Operaciones con SetTensor")
    print("3. Transformación de Lorentz")
    print("4. Salir")

def pause():
    input("Presione Enter para continuar...")

def dynamic_tensor_operations():
    print("\nOperaciones con DynamicTensor:\n")

    # Definir variables simbólicas
    x, y, z = sp.symbols('x y z')

    # Crear un tensor dinámico con expresiones simbólicas
    tensor = DynamicTensor([
        [x**2 + y, sp.sin(x) * sp.cos(y), z**2],
        [sp.exp(y), sp.log(x + 1), sp.sin(z)],
        [x * y, y * z, z * x]
    ])

    print("Tensor original:")
    print(tensor)

    # Evaluar el tensor con valores específicos para las variables
    subs = {x: 1, y: 2, z: 3}
    tensor_evaluated = tensor.evaluate(subs)
    print("\nTensor evaluado con x=1, y=2, z=3:")
    print(tensor_evaluated)

    # Integrar el tensor respecto a x, y y z
    tensor_integrated_xyz = tensor.integrate([x, y, z])
    print("\nTensor integrado con respecto a x, y y z:")
    for t in tensor_integrated_xyz:
        print(t)

    # Diferenciar el tensor cruzadamente respecto a x y y
    tensor_diff_xy = tensor.differentiate(x).differentiate(y)
    print("\nTensor diferenciado cruzadamente con respecto a x y y:")
    print(tensor_diff_xy)

    # Descomposición SVD
    u, s, v = tensor.singular_value_decomposition(subs)
    print("\nDescomposición SVD:")
    print("Matriz U:")
    print(u)
    print("Valores singulares:")
    print(s)
    print("Matriz V:")
    print(v)

    # Descomposición LU
    L, U, perm = tensor.lu_decomposition()
    print("\nDescomposición LU:")
    print("Matriz L:")
    print(L)
    print("Matriz U:")
    print(U)
    print("Matriz de Permutación:")
    print(perm)

    # Calculo de autovalores
    eigenvalues = tensor.eigenvalues()
    print("\nAutovalores del tensor:")
    print(eigenvalues)

    pause()

def set_tensor_operations():
    print("\nOperaciones con SetTensor:\n")

    # Crear conjuntos A y B
    A = SetTensor([1, 2, 3])
    B = SetTensor([3, 4, 5])
    empty_set = SetTensor([])

    print(f"Set A: {A.elements}")
    print(f"Set B: {B.elements}")
    print(f"Set Empty: {empty_set.elements}")

    # Realizar la unión de los conjuntos A y B
    union_set = A.union(B)
    print(f"\nUnion de Set A y B: {union_set.elements}")

    # Realizar la intersección de los conjuntos A y B
    intersection_set = A.intersect(B)
    print(f"Intersección de Set A y B: {intersection_set.elements}")

    # Calcular la diferencia entre los conjuntos A y B
    difference_set = A.difference(B)
    print(f"Diferencia de Set A y B: {difference_set.elements}")

    # Verificar si el conjunto vacío es un subconjunto de A
    is_empty_subset = empty_set.is_subset(A)
    print(f"¿Es el Set Empty un subconjunto de Set A? {is_empty_subset}")

    # Calcular el conjunto potencia de A
    power_set_A = A.power_set()
    print("\nConjunto potencia de A:")
    for subset in power_set_A:
        print(subset.elements)

    # Aplicar una función a cada elemento del conjunto A
    def square(x):
        return x**2

    applied_set_A = A.apply_function(square)
    print(f"\nSet A con función aplicada (x^2): {applied_set_A.elements}")

    # Simplificar elementos de A
    simplified_set_A = A.simplify_elements()
    print(f"Simplified Set A (x^2): {simplified_set_A.elements}")

    # Integrar elementos de A con respecto a x
    integrated_set_A = A.integrate_elements(sp.Symbol('x'))
    print(f"Integrated Set A with respect to x: {integrated_set_A.elements}")

    # Diferenciar elementos de A con respecto a x
    differentiated_set_A = A.differentiate_elements(sp.Symbol('x'))
    print(f"Differentiated Set A with respect to x: {differentiated_set_A.elements}")

    # Expandir elementos de A
    expanded_set_A = A.expand_elements()
    print(f"Expanded Set A (x^2): {expanded_set_A.elements}")

    # Factorizar elementos de A
    factored_set_A = A.factor_elements()
    print(f"Factored Set A (x^2): {factored_set_A.elements}")

    # Producto cartesiano de A y B
    cartesian_product_set = A.cartesian_product(B)
    print(f"\nProducto cartesiano de A y B: {cartesian_product_set.elements}")

    # Resolver un sistema de ecuaciones
    x, y = sp.symbols('x y')
    equations = [sp.Eq(x + y, 2), sp.Eq(x - y, 0)]
    solutions = A.solve_system(equations, [x, y])
    print("\nSoluciones del sistema de ecuaciones:")
    for solution in solutions:
        print(solution)

    pause()

def lorentz_transformation_example():
    print("\nEjemplo de Transformación de Lorentz:\n")

    # Definir la velocidad para la transformación de Lorentz
    velocity = 0.5
    lorentz = LorentzTransformation(velocity)

    # Definir un evento en el espacio-tiempo
    event = [1, 1, 0, 0]
    transformed_event = lorentz.transform(event)
    inverse_transformed_event = lorentz.inverse_transform(transformed_event)

    print(f"Evento original: {event}")
    print(f"Evento transformado: {transformed_event}")
    print(f"Evento transformado inversamente: {inverse_transformed_event}")

    pause()

def main():
    while True:
        show_menu()
        choice = input("Seleccione una opción: ")

        if choice == "1":
            dynamic_tensor_operations()
        elif choice == "2":
            set_tensor_operations()
        elif choice == "3":
            lorentz_transformation_example()
        elif choice == "4":
            print("Saliendo...")
            break
        else:
            print("Opción inválida. Por favor, intente de nuevo.")

if __name__ == "__main__":
    main()
