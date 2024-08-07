from acadyne.tensor.set_tensor import SetTensor
import sympy as sp

def main():
    # Inicializar un conjunto de ecuaciones
    x, y = sp.symbols('x y')
    equations = [
        sp.Eq(x**2 + y, 10),
        sp.Eq(x + y**2, 10)
    ]

    # Resolver ecuaciones simbólicas
    tensor = SetTensor([])
    solutions = tensor.solve_system(equations, [x, y])

    print("Soluciones del sistema de ecuaciones:")
    for sol in solutions:
        print(sol)

if __name__ == "__main__":
    main()
