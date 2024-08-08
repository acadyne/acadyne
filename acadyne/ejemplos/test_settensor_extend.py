import sympy as sp
from set_tensor import SetTensor

# Prueba de inicialización y operaciones básicas
set_a = SetTensor([1, 2, 3])
set_b = SetTensor([3, 4, 5])
set_empty = SetTensor([])

print("Set A:", set_a.elements)
print("Set B:", set_b.elements)
print("Set Empty:", set_empty.elements)

# Prueba de unión
union_set = set_a.union(set_b)
print("Union Set A and B:", union_set.elements)

# Prueba de intersección
intersection_set = set_a.intersect(set_b)
print("Intersection Set A and B:", intersection_set.elements)

# Prueba de diferencia
difference_set = set_a.difference(set_b)
print("Difference Set A and B:", difference_set.elements)

# Prueba de is_subset
is_subset = set_empty.is_subset(set_a)
print("Is Empty Set a subset of Set A?", is_subset)

# Prueba de conjunto potencia
power_set = set_a.power_set()
print("Power Set of A:")
for subset in power_set:
    print(subset.elements)

# Prueba de aplicar función
func = lambda x: x**2
transformed_set = set_a.apply_function(func)
print("Set A with function applied (x^2):", transformed_set.elements)

# Prueba de simplificación
simplified_set = transformed_set.simplify_elements()
print("Simplified Set A (x^2):", simplified_set.elements)

# Prueba de integración
x = sp.Symbol('x')
integrated_set = set_a.integrate_elements(x)
print("Integrated Set A with respect to x:", integrated_set.elements)

# Prueba de diferenciación
differentiated_set = set_a.differentiate_elements(x)
print("Differentiated Set A with respect to x:", differentiated_set.elements)

# Prueba de expansión
expanded_set = transformed_set.expand_elements()
print("Expanded Set A (x^2):", expanded_set.elements)

# Prueba de factorización
factored_set = transformed_set.factor_elements()
print("Factored Set A (x^2):", factored_set.elements)
