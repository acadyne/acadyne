import sympy as sp

from tensores.set_tensor import SetTensor

def main():
    # Inicializar un SetTensor para usar la transformación de Lorentz
    tensor = SetTensor([])

    # Calcular la matriz de transformación de Lorentz para beta=0.5 en la dirección x
    beta_value = 0.5
    lorentz_matrix = tensor.lorentz_transformation(beta_value, direction='x')

    print("Matriz de Lorentz para dirección x con beta=0.5:")
    sp.pprint(lorentz_matrix)

if __name__ == "__main__":
    main()
