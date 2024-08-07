# bionic_tensor.py

import numpy as np
import sympy as sp
from typing import List

class BionicTensor:
    def __init__(self, dimensions=(4, 4)):
        self.dimensions = dimensions
        # Convertir el resultado de np.prod a un entero
        self.tensor = sp.MutableDenseNDimArray([0] * int(np.prod(self.dimensions)), self.dimensions)

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