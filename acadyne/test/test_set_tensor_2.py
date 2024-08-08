# demo_set_tensor.py

from acadyne.tensores.set_tensor import SetTensor

def main():
    # Inicializar dos conjuntos con SetTensor
    set1 = SetTensor([1, 2, 3])
    set2 = SetTensor([3, 4, 5])

    # Realizar operaciones básicas de conjuntos
    print("Set 1:", set1.elements)
    print("Set 2:", set2.elements)

    union_set = set1.union(set2)
    print("Unión:", union_set.elements)

    intersection_set = set1.intersect(set2)
    print("Intersección:", intersection_set.elements)

    difference_set = set1.difference(set2)
    print("Diferencia (Set1 - Set2):", difference_set.elements)

    sym_diff_set = set1.symmetric_difference(set2)
    print("Diferencia Simétrica:", sym_diff_set.elements)

if __name__ == "__main__":
    main()
