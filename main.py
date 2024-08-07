# main.py

import sys

# Importar las funciones principales de los archivos de demostración
from acadyne.demos.demo_set_tensor import main as demo_set_tensor
from acadyne.demos.demo_solve_equations import main as demo_solve_equations
from acadyne.demos.demo_lorentz_transformation import main as demo_lorentz_transformation
from acadyne.demos.demo_fibonacci import main as demo_fibonacci
from acadyne.demos.demo_sir_model import main as demo_sir_model

def show_menu():
    print("\nSeleccione una demostración para ejecutar:")
    print("1. Operaciones Básicas con SetTensor")
    print("2. Solución de Ecuaciones Simbólicas")
    print("3. Transformaciones de Lorentz")
    print("4. Generación de la Serie de Fibonacci")
    print("5. Simulación de un Modelo Epidemiológico SIR")
    print("6. Salir")

def main():
    while True:
        show_menu()
        choice = input("Ingrese su elección (1-6): ")

        if choice == '1':
            demo_set_tensor()
        elif choice == '2':
            demo_solve_equations()
        elif choice == '3':
            demo_lorentz_transformation()
        elif choice == '4':
            demo_fibonacci()
        elif choice == '5':
            demo_sir_model()
        elif choice == '6':
            print("Saliendo...")
            sys.exit(0)
        else:
            print("Opción inválida, por favor intente de nuevo.")

if __name__ == "__main__":
    main()
