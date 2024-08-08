import sympy as sp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable, Union

class InvalidElementStructureError(Exception):
    """Excepción para manejar errores en la estructura de elementos del conjunto."""
    def __init__(self, message: str):
        super().__init__(message)

class SetTensor:
    def __init__(self, elements: Union[List[Union[sp.Basic, str, float, int]], sp.FiniteSet, sp.Interval, sp.Union]):
        self._elements = self._initialize_elements(elements)

    @property
    def elements(self):
        return self._elements

    def _initialize_elements(self, elements: Union[List[Union[sp.Basic, str, float, int]], sp.FiniteSet, sp.Interval, sp.Union]) -> Union[sp.FiniteSet, sp.Interval, sp.Union]:
        if isinstance(elements, list):
            return sp.FiniteSet(*elements)
        elif isinstance(elements, (sp.FiniteSet, sp.Interval, sp.Union)):
            return elements
        else:
            raise InvalidElementStructureError("Elements must be a list, a FiniteSet, Interval, or Union.")

    def union(self, other: 'SetTensor') -> 'SetTensor':
        return SetTensor(self._elements.union(other.elements))

    def intersect(self, other: 'SetTensor') -> 'SetTensor':
        intersection = self._elements.intersect(other.elements)
        return SetTensor(intersection)

    def difference(self, other: 'SetTensor') -> 'SetTensor':
        difference = self._elements - other.elements
        if difference.is_empty:
            return SetTensor([])  # Devuelve un conjunto vacío
        elif isinstance(difference, (sp.FiniteSet, sp.Interval, sp.Union)):
            return SetTensor(difference)
        else:
            raise InvalidElementStructureError("Difference operation returned an unsupported type.")

    def is_subset(self, other: 'SetTensor') -> bool:
        return self._elements.is_subset(other.elements)

    def symmetric_difference(self, other: 'SetTensor') -> 'SetTensor':
        symmetric_diff = self._elements.symmetric_difference(other.elements)
        return SetTensor(symmetric_diff)

    def complement(self, universal_set: 'SetTensor') -> 'SetTensor':
        complement_set = universal_set.elements - self._elements
        return SetTensor(complement_set)

    def power_set(self) -> List['SetTensor']:
        elements_list = list(self._elements)
        power_set_list = []
        for i in range(2**len(elements_list)):
            subset = [elements_list[j] for j in range(len(elements_list)) if (i & (1 << j))]
            power_set_list.append(SetTensor(subset))
        return power_set_list

    def _apply_function_parallel(self, func: Callable[[sp.Basic], sp.Basic], elements: List[sp.Basic]) -> List[sp.Basic]:
        with ThreadPoolExecutor() as executor:
            result = list(executor.map(func, elements))
        return result

    def apply_function(self, func: Callable[[sp.Basic], sp.Basic]) -> 'SetTensor':
        if isinstance(self._elements, sp.FiniteSet):
            transformed_elements = sp.FiniteSet(*self._apply_function_parallel(func, list(self._elements)))
            return SetTensor(transformed_elements)
        raise NotImplementedError("Function application is only supported for FiniteSet.")

    def simplify_elements(self) -> 'SetTensor':
        if isinstance(self._elements, sp.FiniteSet):
            simplified_elements = sp.FiniteSet(*self._apply_function_parallel(sp.simplify, list(self._elements)))
            return SetTensor(simplified_elements)
        raise NotImplementedError("Simplification is only supported for FiniteSet.")

    def integrate_elements(self, symbol: sp.Symbol) -> 'SetTensor':
        if isinstance(self._elements, sp.FiniteSet):
            integrated_elements = sp.FiniteSet(*self._apply_function_parallel(lambda e: sp.integrate(e, symbol), list(self._elements)))
            return SetTensor(integrated_elements)
        raise NotImplementedError("Integration is only supported for FiniteSet.")

    def differentiate_elements(self, symbol: sp.Symbol) -> 'SetTensor':
        if isinstance(self._elements, sp.FiniteSet):
            differentiated_elements = sp.FiniteSet(*self._apply_function_parallel(lambda e: sp.diff(e, symbol), list(self._elements)))
            return SetTensor(differentiated_elements)
        raise NotImplementedError("Differentiation is only supported for FiniteSet.")

    def expand_elements(self) -> 'SetTensor':
        if isinstance(self._elements, sp.FiniteSet):
            expanded_elements = sp.FiniteSet(*self._apply_function_parallel(sp.expand, list(self._elements)))
            return SetTensor(expanded_elements)
        raise NotImplementedError("Expansion is only supported for FiniteSet.")

    def factor_elements(self) -> 'SetTensor':
        if isinstance(self._elements, sp.FiniteSet):
            factored_elements = sp.FiniteSet(*self._apply_function_parallel(sp.factor, list(self._elements)))
            return SetTensor(factored_elements)
        raise NotImplementedError("Factorization is only supported for FiniteSet.")

    def solve_system(self, equations: List[sp.Eq], symbols: List[sp.Symbol]) -> List[sp.FiniteSet]:
        solutions = sp.solve(equations, symbols, dict=True)
        return [sp.FiniteSet(*(sol[sym] for sym in symbols)) for sol in solutions]
    
    def solve_ode(self, equation: sp.Eq, function: sp.Function) -> sp.Expr:
        solution = sp.dsolve(equation, function)
        return solution
    
    def matrix_operations(self, matrix: sp.Matrix) -> dict:
        det = matrix.det()
        trace = matrix.trace()
        inverse = matrix.inv() if matrix.det() != 0 else "No invertible"
        return {'det': det, 'trace': trace, 'inverse': inverse}

    def cartesian_product(self, other: 'SetTensor') -> 'SetTensor':
        if isinstance(self._elements, sp.FiniteSet) and isinstance(other.elements, sp.FiniteSet):
            product_elements = sp.FiniteSet(*[(e1, e2) for e1 in self._elements for e2 in other.elements])
            return SetTensor(product_elements)
        raise NotImplementedError("Cartesian product is only supported for FiniteSet.")
