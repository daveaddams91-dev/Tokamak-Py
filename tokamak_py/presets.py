"""
Tokamak-Py: Simulation Presets.

Provides pre-configured physical scenarios:
- 'tokamak_2d': Tokamak poloidal cross-section with banana orbits and helical flux surfaces
- 'tokamak_3d': Full 3D toroidal magnetic confinement
- 'magnetic_mirror': Magnetic bottle trap with mirror reflection and loss cones
- 'fusor': Inertial Electrostatic Confinement (IEC Fusor) with ion recirculation
- 'exb_drift': Fundamental E x B cross-field drift validation
- 'classic_2d': Original 2D electrostatic N-body sandbox
"""

from dataclasses import dataclass
from typing import Optional
from .fields import (
    FieldConfig, Tokamak2DConfig, Tokamak3DConfig,
    MagneticMirrorConfig, IECFusorConfig, ExBDriftConfig, ZeroFieldConfig
)

@dataclass
class SimulationPreset:
    name: str
    label: str
    dimension: int                     # 2 or 3
    n_particles: int
    dt: float
    field_config: FieldConfig
    boundary_type: str                 # 'circular', 'tokamak', 'spherical', 'none'
    boundary_param: float              # radius or minor radius a
    R0: float = 100.0                  # Major radius (for tokamak)
    theta: float = 0.25                # Barnes-Hut opening angle
    softening: float = 1.0             # Coulomb softening length
    k_e: float = 1000.0                # Coulomb force constant
    initial_thermal_speed: float = 20.0# Initial velocity dispersion
    species_mode: str = "balanced"     # 'balanced' (50% +, 50% -), 'ions_only', 'deuterium_tritium'
    description: str = ""

PRESETS = {
    "tokamak_2d": SimulationPreset(
        name="tokamak_2d",
        label="Tokamak Poloidal Slice (2D)",
        dimension=2,
        n_particles=600,
        dt=0.001,
        field_config=Tokamak2DConfig(B0=2.0, R0=100.0, a=50.0, B_theta0=0.4),
        boundary_type="circular",
        boundary_param=50.0,
        R0=100.0,
        theta=0.25,
        softening=1.5,
        k_e=300.0,
        initial_thermal_speed=30.0,
        species_mode="balanced",
        description="Tokamak poloidal cross-section showing magnetic flux surfaces, gyro-orbits, and trapped-particle banana orbits."
    ),
    
    "tokamak_3d": SimulationPreset(
        name="tokamak_3d",
        label="Tokamak Torus (3D)",
        dimension=3,
        n_particles=500,
        dt=0.001,
        field_config=Tokamak3DConfig(B0=1.5, R0=100.0, a=35.0, B_theta0=0.3),
        boundary_type="tokamak",
        boundary_param=35.0,
        R0=100.0,
        theta=0.3,
        softening=2.0,
        k_e=200.0,
        initial_thermal_speed=25.0,
        species_mode="balanced",
        description="Full 3D toroidal magnetic confinement with helical field lines and toroidal plasma current."
    ),

    "magnetic_mirror": SimulationPreset(
        name="magnetic_mirror",
        label="Magnetic Mirror (Bottle Trap)",
        dimension=3,
        n_particles=400,
        dt=0.001,
        field_config=MagneticMirrorConfig(B0=1.5, L=80.0, Rm=3.5),
        boundary_type="spherical",
        boundary_param=90.0,
        theta=0.3,
        softening=2.0,
        k_e=100.0,
        initial_thermal_speed=20.0,
        species_mode="ions_only",
        description="Magnetic bottle confinement demonstrating magnetic reflection and loss cone escape."
    ),

    "fusor": SimulationPreset(
        name="fusor",
        label="Inertial Electrostatic Confinement (Fusor)",
        dimension=2,
        n_particles=600,
        dt=0.001,
        field_config=IECFusorConfig(V0=3000.0, r_grid=25.0),
        boundary_type="circular",
        boundary_param=120.0,
        theta=0.25,
        softening=1.5,
        k_e=500.0,
        initial_thermal_speed=5.0,
        species_mode="ions_only",
        description="Farnsworth-Hirsch Fusor potential well accelerating ions into a dense central collision core."
    ),

    "exb_drift": SimulationPreset(
        name="exb_drift",
        label="E x B Drift Dynamics",
        dimension=2,
        n_particles=400,
        dt=0.001,
        field_config=ExBDriftConfig(Ey=80.0, Bz=2.0),
        boundary_type="circular",
        boundary_param=150.0,
        theta=0.25,
        softening=2.0,
        k_e=50.0,
        initial_thermal_speed=15.0,
        species_mode="balanced",
        description="Perpendicular E and B fields inducing universal cross-field drift independent of mass or charge."
    ),

    "classic_2d": SimulationPreset(
        name="classic_2d",
        label="Classic Electrostatic Sandbox (2D)",
        dimension=2,
        n_particles=1000,
        dt=0.001,
        field_config=ZeroFieldConfig(),
        boundary_type="circular",
        boundary_param=200.0,
        theta=0.25,
        softening=1.0,
        k_e=1000.0,
        initial_thermal_speed=0.0,
        species_mode="balanced",
        description="Original 2D Barnes-Hut electrostatic N-body plasma simulation with reflective boundary."
    )
}

def get_preset(name: str) -> SimulationPreset:
    """Retrieves preset by name, defaulting to classic_2d if not found."""
    return PRESETS.get(name.lower(), PRESETS["classic_2d"])
