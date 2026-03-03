# GFDMpy: Generalized Finite Difference Method library for transport phenomena problems in multilayer materials

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/) [![NumPy](https://img.shields.io/badge/NumPy-1.20+-blue.svg)](https://numpy.org/) [![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Advanced library for modeling and solving 2D PDEs using the Generalized Finite Difference Method (GFDM).**

*Tailored for complex transport phenomena in systems with material interfaces.*

</div>

---
## :clipboard: Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Scientific Background](#scientific-background)
- [Contributing](#contributing)
- [Research Team](#research-team)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

The **GFDMI** library is a specialized computational tool for the numerical solution of partial differential equations (PDEs) in two-dimensional domains. Leveraging the **Generalized Finite Difference Method (GFDM)**, it excels in handling irregular geometries and material interfaces, making it ideal for simulating transport phenomena in multilayer materials.

### Key Capabilities
- **Unstructured Domain Solving**: Works on unstructured meshes.
- **Multilayer Material Support**: Native handling of jumps in physical properties (e.g., thermal conductivity, permeability).
- **Interface Physics**: Precise implementation of flux and solution continuity conditions across material boundaries.
- **Efficient Numerical Operations**: Built on top of the SciPy sparse matrix ecosystem.

### Applications

| Field | Application | Use Case |
|-------|-------------|----------|
| **Heat Transfer** | Thermal Analysis | Multilayer insulation, composite heat dissipation |
| **Fluid Dynamics** | Porous Media Flow | Groundwater transport, oil reservoir simulation |
| **Material Science** | Diffusion Studies | Interfacial diffusion in alloys, polymer membranes |
| **Environmental Engineering** | Pollutant Transport | Leaching in soil layers, atmospheric dispersion |

---

## Features

### Core Solver
- **2D GFDM Core**: High-order discretization on unstructured node sets.
- **Flexible Operators**: Support for generic linear differential operators $L(u) = Au + Bu_x + Cu_y + Du_{xx} + Eu_{xy} + Fu_{yy}$.
- **Node Support Logic**: Intelligent selection of support stars for optimal accuracy and stability.

### Material & Boundaries
- **Interface Modeling**: Dedicated logic for nodes on the boundary between two different materials.
- **Boundary Conditions**: Implementation of **Dirichlet** (prescribed values) and **Neumann** (prescribed flux) conditions.
- **Normal Computation**: Automated geometric analysis to determine normal vectors at irregular boundaries.

---

## Installation & Setup

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.9+ | 3.10+ |
| **RAM** | 4 GB | 8 GB+ |
| **OS** | Windows/Linux/macOS | Linux |

### Dependencies

```bash
# Core requirements
numpy >= 1.20
scipy >= 1.7
matplotlib >= 3.4
```

### Installation Steps

```bash
git clone https://github.com/RicardoRG73/GFDM-Interfaces.git
cd GFDM-Interfaces
pip install -e .
```

---

## Quick Start

### 1. Configure a Problem
Use the `GFDMI_2D_problem` class to define your domain and physics.

```python
from gfdm_interfaces.GFDMI import GFDMI_2D_problem as gfdmi
import numpy as np

# Operator for Poisson equation: Laplacian (D=1, F=1)
L = np.array([0, 0, 0, 1, 0, 1])
source = lambda p: -2

# Initialize (assuming 'coords' and 'triangles' are loaded)
problem = gfdmi(coords, triangles, L, source)
```

### 2. Define Boundaries and Materials
```python
problem.material('region1', permeability_func, interior_nodes)
problem.dirichlet_boundary('wall', edge_nodes, lambda p: 0)
```

### 3. Solve
```python
K, F = problem.continuous_discretization()
# solve KU = F using scipy.sparse.linalg.spsolve
```

For a complete working example, see `examples/basic_example.py`.

---

## Project Structure

```text
GFDMpy/
├── examples/
│   ├── basic_example.py
│   └── legacy/
│       ├── Meshes/           # JSON geometry files
│       ├── figures/          # Generated results
│       ├── ex0.py            # Legacy examples...
│       └── ex8.py
├── src/
│   └── gfdmpy/               # Core package
│       ├── GFDM.py           # Main class implementation
│       ├── utils.py          # Utility functions
│       └── __init__.py
├── tests/
│   └── test_gfdmi.py         # Unit tests
├── pyproject.toml            # Build system configuration
├── README.md                 # This file
├── requirements.txt          # Dependencies
└── verify_refactor.py        # Verification script
```

### Detailed Breakdown

- **`src/gfdmpy/`**: The core package containing the library logic.
  - `GFDM.py`: Main implementation of the `GFDMI_2D_problem` class.
  - `utils.py`: Utility functions for support node selection and normal vector calculations.
  - `__init__.py`: Package initialization.

- **`examples/`**: Ready-to-run demonstration scripts.
  - `basic_example.py`: A complete example solving a 2D Laplacian problem.
  - **`legacy/`**: Adapted examples from the original project (`ex0` to `ex8`).
    - `Meshes/`: Contains JSON files for the geometries.
    - `figures/`: Resulting plots generated by the scripts.

- **`tests/`**: Unit tests to ensure core functionality.
  - `test_gfdmi.py`: Tests for initialization, support nodes, and assembly.

- **Other Files**:
  - `verify_refactor.py`: A quick-start script to verify the installation and refactoring.
  - `requirements.txt`: List of required Python packages (`numpy`, `scipy`, `matplotlib`, `calfem-python`).
  - `pyproject.toml`: Configuration for modern Python build tools and installation.

---

## Scientific Background

The **Generalized Finite Difference Method (GFDM)** belongs to the family of meshless methods. Unlike traditional FDM which requires a structured grid, GFDM approximates derivatives using a Taylor series expansion over a local "star" of nodes. 

This library specifically implements interface conditions using the **Ghost Node** method and **Discontinuous Discretization** to maintain accuracy across material boundaries where physical properties might be discontinuous.

---

## Contributing

We welcome contributions from the scientific and developer community. If you find a bug, have a feature request, or want to contribute to the core algorithms:

1. **Bug Reports**: Please use GitHub Issues to report bugs.
2. **Pull Requests**: For code contributions, please fork the repository and create a feature branch.
3. **Research Collaboration**: For academic partnerships, contact the research team via email.

---

## Research Team

This project is led by researchers dedicated to advancing meshless computational methods at **Universidad Michoacana de San Nicolás de Hidalgo (UMSNH)**.

- **Dr. Ricardo Román-Gutiérrez**: Principal Researcher and Lead Developer.
- **Dr. Francisco Domínguez-Mota**: Contributor and scientific advisor.
- **Dr. Carlos Chávez-Negrete**: Contributor and scientific advisor.
- **Dr. Gerardo Tinoco-Guerrero**: Contributor and scientific advisor.
- **Dr. José Alberto Guzmán-Torres**: Contributor and scientific advisor.
- **Dr. Gerardo Tinoco-Ruiz**: Contributor and scientific advisor.

---

## Acknowledgments
We express our sincere gratitude to:
- **AULA CIMNE-Morelia**, for their generous hospitality in providing a collaborative working environment, and for their steadfast commitment to fostering cutting-edge research.
- **FIC-UMSNH** (Faculty of Civil Engineering, Universidad Michoacana de San Nicolás de Hidalgo), for kindly offering their facilities and for their enduring dedication to the promotion of scientific and academic development.
- **FCFM-UMSNH** (Faculty of Physical Mathematical Sciences, Universidad Michoacana de San Nicolás de Hidalgo), for graciously granting access to their computational infrastructure, which was instrumental in carrying out this work.
- **SECIHITI** (Secretaría de Ciencia, Humanidades, Tecnología e Innovación), for their valuable financial support, without which the successful completion of this research would not have been possible.

---

## License

This project is licensed under the **MIT License** - see the LICENSE file for details.
