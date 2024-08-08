# comprueba las Leyes de De Morgan:
from acadyne.tensores.set_tensor import SetTensor

def main():
    # Definimos los conjuntos A y B
    A = SetTensor([1, 2, 3, 4])
    B = SetTensor([3, 4, 5, 6])

    # Definimos el conjunto universal
    universal_set = SetTensor([1, 2, 3, 4, 5, 6, 7, 8])

    # Ley de De Morgan 1: (A ∪ B)^c = A^c ∩ B^c
    union_complement = A.union(B).complement(universal_set)
    complement_intersection = A.complement(universal_set).intersect(B.complement(universal_set))

    # Comprobamos si son iguales
    print("Ley de De Morgan 1:")
    print(f"(A ∪ B)^c: {union_complement.elements}")
    print(f"A^c ∩ B^c: {complement_intersection.elements}")
    print(f"¿Son iguales? {union_complement.elements == complement_intersection.elements}\n")

    # Ley de De Morgan 2: (A ∩ B)^c = A^c ∪ B^c
    intersection_complement = A.intersect(B).complement(universal_set)
    complement_union = A.complement(universal_set).union(B.complement(universal_set))

    # Comprobamos si son iguales
    print("Ley de De Morgan 2:")
    print(f"(A ∩ B)^c: {intersection_complement.elements}")
    print(f"A^c ∪ B^c: {complement_union.elements}")
    print(f"¿Son iguales? {intersection_complement.elements == complement_union.elements}")

if __name__ == "__main__":
    main()
