
from tensores.set_tensor import SetTensor


def main():
    # Inicializar un SetTensor para simulación de SIR
    tensor = SetTensor([])

    # Simular el modelo SIR
    beta = 0.3  # Tasa de transmisión
    gamma = 0.1  # Tasa de recuperación
    initial_conditions = {'S': 0.99, 'I': 0.01, 'R': 0.0}
    duration = 160
    step_size = 1

    sir_results = tensor.simulate_sir(beta, gamma, initial_conditions, duration, step_size)
    
    # Mostrar los primeros 5 resultados de la simulación
    for result in sir_results[:5]:
        print(result)

if __name__ == "__main__":
    main()
