"""
Tokamak-Py: Physics Interface Module (Backward Compatibility Layer).

Re-exports core physics routines and the unified PhysicsEngine from tokamak_py.
Maintained for backward compatibility with existing scripts.
"""

import numpy as np
from tokamak_py.engine import PhysicsEngine
from tokamak_py.tree import (
    get_bounding_box_2d as get_bounding_box,
    get_quadrant,
    build_quadtree,
    calculate_forces_2d as calculate_forces
)
from tokamak_py.integrators import apply_boundary_circular_2d as apply_boundary_conditions

__all__ = [
    "PhysicsEngine",
    "get_bounding_box",
    "get_quadrant",
    "build_quadtree",
    "calculate_forces",
    "apply_boundary_conditions"
]
