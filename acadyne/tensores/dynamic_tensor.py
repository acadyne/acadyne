import numpy as np
import sympy as sp
import pandas as pd
from typing import List, Union, Callable, Tuple
import matplotlib.pyplot as plt


class InvalidComponentStructureError(Exception):
    """Exception raised for errors in the tensor component structure."""
    pass


class BaseTensor:
    def __init__(self, components: Union[List[List[Union[sp.Basic, str, float, int]]], np.ndarray, pd.DataFrame]):
        """
        Initializes a symbolic tensor with given components.

        Parameters:
        - components (Union[List[List[Union[sp.Basic, str, float, int]]], np.ndarray, pd.DataFrame]): 
          The components of the tensor, can be a list of lists, a numpy array, or a pandas DataFrame.
        """
        if isinstance(components, pd.DataFrame):
            components = components.to_numpy()

        if isinstance(components, (list, np.ndarray)):
            if not self.validate_components(components):
                raise InvalidComponentStructureError("Invalid component structure.")
            components = np.array(components, dtype=object)
        else:
            raise InvalidComponentStructureError(
                "Components must be either a list of lists, an np.ndarray, or a pd.DataFrame."
            )

        # Convert components to sympy expressions
        try:
            self._components = np.array([
                [item if isinstance(item, sp.Basic) else sp.sympify(item) for item in row]
                for row in components
            ], dtype=object)
        except Exception as e:
            raise InvalidComponentStructureError(f"Failed to sympify components: {e}")

    @staticmethod
    def validate_components(components) -> bool:
        """
        Validates the components structure.

        Parameters:
        - components: The components to validate.

        Returns:
        - bool: True if valid, False otherwise.
        """
        try:
            components = np.array(components, dtype=object)
            for row in components:
                for item in row:
                    if not isinstance(item, (sp.Basic, str, float, int)):
                        return False
            return True
        except Exception:
            return False

    def add(self, other: 'BaseTensor') -> 'BaseTensor':
        """
        Adds this tensor to another symbolic tensor.

        Parameters:
        - other (BaseTensor): The tensor to add.

        Returns:
        - BaseTensor: A new tensor representing the sum of this tensor and the other.

        Raises:
        - ValueError: If the dimensions of the tensors do not match.
        """
        if self._components.shape != other.components.shape:
            raise ValueError("Tensors must have the same dimensions for addition.")
        result = self._components + other.components
        return BaseTensor(result)

    def subtract(self, other: 'BaseTensor') -> 'BaseTensor':
        """
        Subtracts another symbolic tensor from this tensor.

        Parameters:
        - other (BaseTensor): The tensor to subtract.

        Returns:
        - BaseTensor: A new tensor representing the difference between this tensor and the other.

        Raises:
        - ValueError: If the dimensions of the tensors do not match.
        """
        if self._components.shape != other.components.shape:
            raise ValueError("Tensors must have the same dimensions for subtraction.")
        result = self._components - other.components
        return BaseTensor(result)

    def multiply(self, other: Union['BaseTensor', sp.Basic, float, int]) -> 'BaseTensor':
        """
        Multiplies this tensor with another tensor or a scalar.

        Parameters:
        - other (Union[BaseTensor, sp.Basic, float, int]): The tensor or scalar to multiply.

        Returns:
        - BaseTensor: A new tensor representing the product.

        Raises:
        - ValueError: If the dimensions are incompatible for matrix multiplication.
        """
        if isinstance(other, BaseTensor):
            if self._components.shape[1] != other.components.shape[0]:
                raise ValueError("Incompatible dimensions for matrix multiplication.")
            try:
                result = np.dot(self._components, other.components)
                return BaseTensor(result)
            except TypeError:
                result = sp.Matrix(self._components) * sp.Matrix(other.components)
                return BaseTensor(np.array(result).astype(object))
        else:  # Assuming scalar multiplication
            result = self._components * other
            return BaseTensor(result)

    def determinant(self) -> sp.Basic:
        """
        Calculates the determinant of the tensor if it is square.

        Returns:
        - sp.Basic: The determinant of the tensor.

        Raises:
        - ValueError: If the tensor is not square.
        """
        if self._components.shape[0] != self._components.shape[1]:
            raise ValueError("Determinant is only defined for square matrices.")
        return sp.Matrix(self._components).det()

    def inverse(self) -> 'BaseTensor':
        """
        Calculates the inverse of the tensor if it is square and non-singular.

        Returns:
        - BaseTensor: The inverse of the tensor.

        Raises:
        - ValueError: If the tensor is not square or is singular.
        """
        if self._components.shape[0] != self._components.shape[1]:
            raise ValueError("Inverse is only defined for square matrices.")
        if sp.Matrix(self._components).det() == 0:
            raise ValueError("The matrix is singular and cannot be inverted.")
        inverse_matrix = sp.Matrix(self._components).inv()
        return BaseTensor(np.array(inverse_matrix).astype(object))

    def evaluate(self, subs: dict) -> 'BaseTensor':
        """
        Evaluates the tensor using a dictionary of substitutions.

        Parameters:
        - subs (dict): A dictionary of substitutions to apply to the tensor's components.

        Returns:
        - BaseTensor: A new tensor with evaluated components.
        """
        evaluated = np.array([
            [elem.subs(subs) if isinstance(elem, sp.Basic) else elem for elem in row]
            for row in self._components
        ])
        return BaseTensor(evaluated)

    def differentiate(self, symbol: sp.Symbol) -> 'BaseTensor':
        """
        Differentiates each component of the tensor with respect to a given symbol.

        Parameters:
        - symbol (sp.Symbol): The symbol to differentiate with respect to.

        Returns:
        - BaseTensor: A new tensor with differentiated components.
        """
        differentiated = np.array([
            [elem.diff(symbol) if isinstance(elem, sp.Basic) else 0 for elem in row]
            for row in self._components
        ])
        return BaseTensor(differentiated)

    def integrate(self, vars: List[sp.Symbol]) -> List['BaseTensor']:
        """
        Integrates each component of the tensor with respect to the given variables.

        Parameters:
        - vars (List[sp.Symbol]): The variables to integrate with respect to.

        Returns:
        - List[BaseTensor]: A list of new tensors, each integrated with respect to one of the variables.
        """
        integrated = [np.array([[elem.integrate(var) if isinstance(elem, sp.Basic) else elem for elem in row] for row in self._components]) for var in vars]
        return [BaseTensor(integral) for integral in integrated]

    def gradient(self, vars: List[sp.Symbol]) -> sp.Matrix:
        """
        Calculates the gradient of each component of the tensor with respect to the given variables.

        Parameters:
        - vars (List[sp.Symbol]): The variables to calculate the gradient with respect to.

        Returns:
        - sp.Matrix: The gradient matrix of the tensor.
        """
        grad_matrix = []
        for row in self._components.tolist():
            grad_row = []
            for item in row:
                grad_item = [sp.diff(item, var) for var in vars]
                grad_row.append(sp.Matrix(grad_item))
            grad_matrix.append(grad_row)
        return sp.Matrix(grad_matrix)

    def advanced_simplify(self) -> 'BaseTensor':
        """
        Applies advanced simplification to each component of the tensor.

        Returns:
        - BaseTensor: A new tensor with simplified components.
        """
        simplified = np.vectorize(sp.simplify)(self._components)
        return BaseTensor(simplified)

    def factorize(self) -> 'BaseTensor':
        """
        Factorizes each component of the tensor.

        Returns:
        - BaseTensor: A new tensor with factorized components.
        """
        factored = np.vectorize(sp.factor)(self._components)
        return BaseTensor(factored)

    def eigenvalues(self) -> List[sp.Basic]:
        """
        Calculates the eigenvalues of the tensor if it is square.

        Returns:
        - List[sp.Basic]: The eigenvalues of the tensor.

        Raises:
        - ValueError: If the tensor is not square.
        """
        if self._components.shape[0] != self._components.shape[1]:
            raise ValueError("Eigenvalues are only defined for square matrices.")
        return list(sp.Matrix(self._components).eigenvals().keys())

    def eigenvectors(self) -> List[Tuple[sp.Basic, int, List[sp.Matrix]]]:
        """
        Calculates the eigenvectors of the tensor if it is square.

        Returns:
        - List[Tuple[sp.Basic, int, List[sp.Matrix]]]: A list of tuples containing eigenvalues, 
          their algebraic multiplicities, and their corresponding eigenvectors.

        Raises:
        - ValueError: If the tensor is not square.
        """
        if self._components.shape[0] != self._components.shape[1]:
            raise ValueError("Eigenvectors are only defined for square matrices.")
        eigenvectors = sp.Matrix(self._components).eigenvects()
        return [(vec[0], vec[1], [v.tolist() for v in vec[2]]) for vec in eigenvectors]

    def singular_value_decomposition(self, subs: dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs Singular Value Decomposition (SVD) on the tensor.

        Parameters:
        - subs (dict): A dictionary of substitutions to apply before performing SVD.

        Returns:
        - Tuple[np.ndarray, np.ndarray, np.ndarray]: The U, S, and V matrices from the SVD.

        Raises:
        - ValueError: If the tensor contains non-numeric elements after substitution.
        """
        if subs:
            evaluated = self.evaluate(subs).components
        else:
            evaluated = self._components

        try:
            numeric_matrix = np.array(evaluated, dtype=float)
        except TypeError:
            raise ValueError("Cannot perform SVD on a tensor with non-numeric elements. Provide substitutions.")

        u, s, v = np.linalg.svd(numeric_matrix)
        return u, s, v

    def lu_decomposition(self) -> Tuple[List[List[sp.Basic]], List[List[sp.Basic]], List[int]]:
        """
        Performs LU Decomposition on the tensor.

        Returns:
        - Tuple[List[List[sp.Basic]], List[List[sp.Basic]], List[int]]: The L and U matrices as lists 
          and the permutation matrix as a list.
        """
        L, U, perm = sp.Matrix(self._components).LUdecomposition()
        L_list = np.array(L).astype(object).tolist()
        U_list = np.array(U).astype(object).tolist()
        return L_list, U_list, perm

    def qr_decomposition(self, subs: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs QR Decomposition on the tensor.

        Parameters:
        - subs (dict): A dictionary of substitutions to apply before performing QR decomposition.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: The Q and R matrices from the QR decomposition.

        Raises:
        - ValueError: If the tensor contains non-numeric elements after substitution.
        """
        if subs:
            evaluated = self.evaluate(subs).components
        else:
            evaluated = self._components

        try:
            numeric_matrix = np.array(evaluated, dtype=float)
        except TypeError:
            raise ValueError("Cannot perform QR decomposition on a tensor with non-numeric elements. Provide substitutions.")

        q, r = np.linalg.qr(numeric_matrix)
        return q, r

    @property
    def components(self) -> np.ndarray:
        """
        Returns the symbolic components of the tensor.

        Returns:
        - np.ndarray: The components of the tensor.
        """
        return self._components

    def to_dataframe(self, row_labels=None, column_labels=None) -> pd.DataFrame:
        """
        Converts the tensor components to a pandas DataFrame.

        Parameters:
        - row_labels: Labels for the rows of the DataFrame.
        - column_labels: Labels for the columns of the DataFrame.

        Returns:
        - pd.DataFrame: The tensor components as a DataFrame.
        """
        return pd.DataFrame(self._components, index=row_labels, columns=column_labels)

    def to_csv(self, filepath: str, index=False, header=True):
        """
        Exports the tensor components to a CSV file.

        Parameters:
        - filepath (str): The file path to save the CSV file.
        - index (bool): Whether to include row indices in the CSV file.
        - header (bool): Whether to include column headers in the CSV file.
        """
        df = self.to_dataframe()
        df.to_csv(filepath, index=index, header=header)

    def to_excel(self, filepath: str, index=False, header=True):
        """
        Exports the tensor components to an Excel file.

        Parameters:
        - filepath (str): The file path to save the Excel file.
        - index (bool): Whether to include row indices in the Excel file.
        - header (bool): Whether to include column headers in the Excel file.
        """
        df = self.to_dataframe()
        df.to_excel(filepath, index=index, header=header)

    def plot(self, subs: dict = None, title="Matrix Plot"):
        """
        Plots the tensor as a heatmap.

        Parameters:
        - subs (dict): A dictionary of substitutions to apply before plotting.
        - title (str): The title of the plot.
        """
        if subs:
            evaluated = self.evaluate(subs).components
        else:
            evaluated = self._components

        try:
            numeric_matrix = np.array(evaluated, dtype=float)
        except TypeError:
            raise ValueError("Cannot plot a tensor with non-numeric elements. Provide substitutions.")

        plt.imshow(numeric_matrix, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(title)
        plt.show()

    def __str__(self) -> str:
        """
        String representation of the tensor.

        Returns:
        - str: A string representation of the tensor.
        """
        return str(sp.Matrix(self._components))


class DynamicTensor(BaseTensor):
    def __init__(self, components: Union[List[List[Union[sp.Basic, str, float, int]]], np.ndarray, pd.DataFrame]):
        """
        Initializes a dynamic tensor, allowing for transformations and expansions.

        Parameters:
        - components (Union[List[List[Union[sp.Basic, str, float, int]]], np.ndarray, pd.DataFrame]): 
          The components of the dynamic tensor.
        """
        super().__init__(components)

    def apply_transformation(self, transform: Callable[[np.ndarray], np.ndarray]) -> 'DynamicTensor':
        """
        Applies a given transformation to the tensor components.

        Parameters:
        - transform (Callable[[np.ndarray], np.ndarray]): A function that takes a numpy array and returns a transformed array.

        Returns:
        - DynamicTensor: A new tensor with transformed components.
        """
        try:
            transformed = transform(self.components)
            return DynamicTensor(transformed)
        except Exception as e:
            raise InvalidComponentStructureError(f"Transformation failed: {e}")

    def expand_tensor(self, expand_func: Callable[[np.ndarray], np.ndarray]) -> 'DynamicTensor':
        """
        Expands the tensor using a specified function.

        Parameters:
        - expand_func (Callable[[np.ndarray], np.ndarray]): A function to expand the tensor.

        Returns:
        - DynamicTensor: A new tensor with expanded components.
        """
        try:
            expanded = expand_func(self.components)
            return DynamicTensor(expanded)
        except Exception as e:
            raise InvalidComponentStructureError(f"Expansion failed: {e}")

    def nonlinear_transformation(self, func: Callable[[sp.Basic], sp.Basic]) -> 'DynamicTensor':
        """
        Applies a nonlinear transformation to each element of the tensor.

        Parameters:
        - func (Callable[[sp.Basic], sp.Basic]): A function to apply to each element of the tensor.

        Returns:
        - DynamicTensor: A new tensor with transformed components.
        """
        try:
            transformed = np.array([
                [func(elem) for elem in row]
                for row in self._components
            ])
            return DynamicTensor(transformed)
        except Exception as e:
            raise InvalidComponentStructureError(f"Nonlinear transformation failed: {e}")
