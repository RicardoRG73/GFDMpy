# API Reference - GFDFlow

The `GFDFlow` package provides classes and utilities for solving 2D interface problems using the Generalized Finite Difference Method (GFDM).

## Core Classes

### `GFDMI_2D_problem`

This is the main class used to define and solve 2D PDEs with material interfaces.

#### `__init__(coords, triangles, L, source)`
Initializes the problem.
- **`coords`**: `npt.NDArray[np.float64]` (n, 2) node coordinates.
- **`triangles`**: `npt.NDArray[np.int_]` (m, 3) connectivity.
- **`L`**: `npt.NDArray[np.float64]` (6,) coefficients `[A, B, C, D, E, F]`.
- **`source`**: `Callable[[npt.NDArray[np.float64]], float]` source function `f(x, y)`.

#### `material(label, permeability, interior_nodes)`
Defines material properties for a set of nodes.
- **`label`**: `str` identifier.
- **`permeability`**: `Callable` returning the material's permeability at a point.
- **`interior_nodes`**: `npt.NDArray[np.int_]` indices of nodes with this material.

#### `neumann_boundary(label, permeability, boundary_nodes, condition)`
Defines Neumann boundary conditions (`du/dn = g`).
- **`label`**: `str` identifier.
- **`permeability`**: `Callable` for the material at the boundary.
- **`boundary_nodes`**: `npt.NDArray[np.int_]` indices of boundary nodes.
- **`condition`**: `Callable` returning the flux value at a point.

#### `dirichlet_boundary(label, boundary_nodes, condition)`
Defines Dirichlet boundary conditions (`u = h`).
- **`label`**: `str` identifier.
- **`boundary_nodes`**: `npt.NDArray[np.int_]` indices of boundary nodes.
- **`condition`**: `Callable` returning the fixed value at a point.

#### `interface(label, k_left, k_right, nodes_left, nodes_right, beta, alpha, interior_left, interior_right)`
Defines interface conditions between two materials.
- **`label`**: `str` identifier.
- **`k_left`, `k_right`**: `Callable` permeabilities for both sides.
- **`nodes_left`, `nodes_right`**: `npt.NDArray[np.int_]` nodes along the interface from both sides.
- **`beta`**: `Callable` flux jump condition.
- **`alpha`**: `Callable` potential jump condition.
- **`interior_left`, `interior_right`**: `npt.NDArray[np.int_]` nodes in the respective materials.

#### `discretization_K_F(continuous=False)`
Assembles the global stiffness matrix `K` and force vector `F`.
- **`continuous`**: `bool` (default `False`). If `True`, assumes a continuous interface.
- **Returns**: `Tuple[sp.csr_matrix, npt.NDArray[np.float64]]` (K, F).

---

## Utility Functions

### `get_support_nodes(node_idx, triangles, min_support_nodes=5, max_iter=2)`
Finds the support nodes for a given central node using mesh connectivity.

### `compute_normal_vectors(boundary_nodes, coords)`
Computes outward-pointing normal vectors for a set of boundary nodes.
