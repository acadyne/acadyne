import sympy as sp

class LorentzTransformation:
    def __init__(self, velocity):
        """
        Inicializa una transformación de Lorentz con una velocidad dada.

        :param velocity: Velocidad relativa entre los sistemas de referencia (como una fracción de la velocidad de la luz).
        """
        self.velocity = velocity
        self.gamma = 1 / sp.sqrt(1 - velocity**2)
        self.transformation_matrix = self._create_lorentz_matrix()

    def _create_lorentz_matrix(self):
        """
        Crea la matriz de transformación de Lorentz para la velocidad dada.

        :return: Matriz de transformación de Lorentz.
        """
        v = self.velocity
        gamma = self.gamma

        return sp.Matrix([
            [gamma, -gamma * v, 0, 0],
            [-gamma * v, gamma, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def transform(self, event):
        """
        Aplica la transformación de Lorentz a un evento espacio-temporal.

        :param event: Evento espacio-temporal representado como una lista o matriz [t, x, y, z].
        :return: Evento transformado.
        """
        event_matrix = sp.Matrix(event)
        return self.transformation_matrix * event_matrix

    def inverse_transform(self, event):
        """
        Aplica la transformación inversa de Lorentz a un evento espacio-temporal.

        :param event: Evento espacio-temporal representado como una lista o matriz [t, x, y, z].
        :return: Evento transformado inversamente.
        """
        event_matrix = sp.Matrix(event)
        inverse_matrix = self.transformation_matrix.inv()
        return inverse_matrix * event_matrix



# def test_lorentz_transformation():
#     # Velocidad relativa
#     velocity = 0.5
#     lorentz = LorentzTransformation(velocity)
    
#     # Matriz de transformación esperada
#     gamma = 1 / sp.sqrt(1 - velocity**2)
#     expected_matrix = sp.Matrix([
#         [gamma, -gamma * velocity, 0, 0],
#         [-gamma * velocity, gamma, 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1]
#     ])
#     print("Matriz de transformación esperada:")
#     print(expected_matrix)

#     # Verificar la matriz de transformación
#     print("Matriz de transformación obtenida:")
#     print(lorentz.transformation_matrix)
#     assert lorentz.transformation_matrix == expected_matrix, "Matriz de transformación incorrecta"

#     # Evento original
#     event = [1, 1, 0, 0]
#     print("Evento original:")
#     print(event)

#     # Evento transformado esperado
#     expected_transformed_event = expected_matrix * sp.Matrix(event)
#     print("Evento transformado esperado:")
#     print(expected_transformed_event)

#     # Verificar la transformación de un evento
#     transformed_event = lorentz.transform(event)
#     print("Evento transformado obtenido:")
#     print(transformed_event)
#     assert transformed_event == expected_transformed_event, f"Evento transformado incorrecto: {transformed_event} != {expected_transformed_event}"

#     # Verificar la transformación inversa
#     inverse_transformed_event = lorentz.inverse_transform(transformed_event)
#     print("Evento transformado inversamente obtenido:")
#     print(inverse_transformed_event)
#     assert sp.simplify(inverse_transformed_event - sp.Matrix(event)) == sp.zeros(4, 1), f"Evento inversamente transformado incorrecto: {inverse_transformed_event} != {event}"

#     print("Todas las pruebas pasaron exitosamente.")

# if __name__ == "__main__":
#     test_lorentz_transformation()
