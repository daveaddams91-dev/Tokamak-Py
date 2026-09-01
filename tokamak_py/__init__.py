"""
Tokamak-Py: High-Performance Electromagnetic Plasma Confinement Engine.

Simulates N-body electrostatic and magnetic plasma confinement using
Barnes-Hut quadtrees/octrees, exact direct solvers, and symplectic Boris
integration vectorized via Numba JIT.
"""

__version__ = "1.0.0"
__author__ = "Raj"

from .engine import PhysicsEngine
from .presets import SimulationPreset, PRESETS, get_preset
from .fields import (
    FieldConfig, Tokamak2DConfig, Tokamak3DConfig,
    MagneticMirrorConfig, IECFusorConfig, ExBDriftConfig, ZeroFieldConfig
)
from .constants import (
    Species, SPECIES_ELECTRON, SPECIES_PROTON, SPECIES_DEUTERON,
    SPECIES_TRITON, SPECIES_ALPHA, SPECIES_NORMALIZED_ION, SPECIES_NORMALIZED_ELECTRON
)

__all__ = [
    "PhysicsEngine",
    "SimulationPreset",
    "PRESETS",
    "get_preset",
    "FieldConfig",
    "Tokamak2DConfig",
    "Tokamak3DConfig",
    "MagneticMirrorConfig",
    "IECFusorConfig",
    "ExBDriftConfig",
    "ZeroFieldConfig",
    "Species",
    "SPECIES_ELECTRON",
    "SPECIES_PROTON",
    "SPECIES_DEUTERON",
    "SPECIES_TRITON",
    "SPECIES_ALPHA",
]
