

from acadyne.tensores.set_tensor import SetTensor
from acadyne.tensores.set_tensor import SetTensor


def main():
    # Crear un SetTensor para usar la generación de Fibonacci
    tensor = SetTensor([])

    # Generar la serie de Fibonacci usando la función generadora
    n_terms = 10
    fibonacci_series = tensor.generate_fibonacci_series(n_terms)
    print(f"Serie de Fibonacci usando la función generadora (primeros {n_terms} términos):")
    print(fibonacci_series)

    # Calcular el número de Fibonacci usando exponenciación de matrices
    n = 20
    fib_n = tensor.matrix_fib(n)
    print(f"Fibonacci número {n} es {fib_n}")

    # Aproximación de Fibonacci usando la fórmula de Binet
    approx_fib_n = tensor.binet_formula(n)
    print(f"Aproximación rápida de Fibonacci número {n} usando la fórmula de Binet es {approx_fib_n}")

if __name__ == "__main__":
    main()
