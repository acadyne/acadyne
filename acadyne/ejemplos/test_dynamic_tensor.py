import sympy as sp
import numpy as np

from tensores.dynamic_tensor import BaseTensor

# Definir símbolos
x, y, z = sp.symbols('x y z')

# Tensor simbólico original
tensor = BaseTensor([
    [x**2 + y, sp.sin(x) * sp.cos(y), z**2],
    [sp.exp(y), sp.log(x + 1), sp.sin(z)],
    [x * y, y * z, z * x]
])

print("Tensor original:")
print(tensor)

# Evaluar el tensor con valores específicos
subs = {x: 1, y: 2, z: 3}
tensor_evaluated = tensor.evaluate(subs)
print("\nTensor evaluado con x=1, y=2, z=3:")
print(tensor_evaluated)

# Transformación de coordenadas (rotación en 3D)
def rotation_matrix_3d(theta_x, theta_y, theta_z):
    Rx = sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(theta_x), -sp.sin(theta_x)],
        [0, sp.sin(theta_x), sp.cos(theta_x)]
    ])
    Ry = sp.Matrix([
        [sp.cos(theta_y), 0, sp.sin(theta_y)],
        [0, 1, 0],
        [-sp.sin(theta_y), 0, sp.cos(theta_y)]
    ])
    Rz = sp.Matrix([
        [sp.cos(theta_z), -sp.sin(theta_z), 0],
        [sp.sin(theta_z), sp.cos(theta_z), 0],
        [0, 0, 1]
    ])
    return Rz * Ry * Rx

theta_x, theta_y, theta_z = sp.symbols('theta_x theta_y theta_z')
rotation_matrix = rotation_matrix_3d(theta_x, theta_y, theta_z)

# Convertir matrices de rotación a np.ndarray
rotation_matrix_np = np.array(rotation_matrix.tolist(), dtype=object)
rotation_matrix_np_T = np.array(rotation_matrix.T.tolist(), dtype=object)

# Crear tensores de rotación
rotation_tensor = BaseTensor(rotation_matrix_np)
rotation_tensor_T = BaseTensor(rotation_matrix_np_T)

# Aplicar rotación
rotated_tensor = rotation_tensor.multiply(tensor).multiply(rotation_tensor_T)
rotated_tensor_evaluated = rotated_tensor.evaluate({theta_x: sp.pi / 4, theta_y: sp.pi / 4, theta_z: sp.pi / 4})
print("\nTensor transformado por rotación 3D:")
print(rotated_tensor_evaluated)

# Integrar el tensor respecto a x, y, y z
tensor_integrated_xyz = tensor.integrate([x, y, z])
print("\nTensor integrado con respecto a x, y y z:")
print(tensor_integrated_xyz)

# Diferenciación cruzada con respecto a x y y
tensor_diff_xy = tensor.differentiate(x).differentiate(y)
print("\nTensor diferenciado cruzadamente con respecto a x y y:")
print(tensor_diff_xy)

# Descomposición SVD
u, s, v = tensor.singular_value_decomposition(subs)
print("\nDescomposición SVD:")
print("U matrix:")
print(u)
print("Singular values:")
print(s)
print("V matrix:")
print(v)

# Descomposición LU
L, U, perm = tensor.lu_decomposition()
print("\nDescomposición LU:")
print("L matrix:")
print(L)
print("U matrix:")
print(U)
print("Permutation matrix:")
print(perm)

# Cálculo de autovalores y autovectores
try:
    eigenvalues = tensor.eigenvalues()
    print("\nAutovalores del tensor:")
    print(eigenvalues)

    eigenvectors = tensor.eigenvectors()
    print("\nAutovectores del tensor:")
    for value, multiplicity, vectors in eigenvectors:
        print(f"Autovalor: {value}, Multiplicidad: {multiplicity}, Autovectores: {vectors}")
except ValueError as e:
    print(f"Error en el cálculo de autovalores/autovectores: {e}")

print("\nCálculo completado.")
