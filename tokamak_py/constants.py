"""
Tokamak-Py: Physical Constants and Plasma Species Definitions.

Contains standard SI constants, normalized plasma units, and species properties
for electrons, protons, deuterons, tritons, and alpha particles.
"""

from dataclasses import dataclass
import numpy as np

# =============================================================================
# SI Physical Constants
# =============================================================================
ELEMENTARY_CHARGE = 1.602176634e-19       # Coulombs [C]
ELECTRON_MASS = 9.1093837015e-31          # Kilograms [kg]
PROTON_MASS = 1.67262192369e-27           # Kilograms [kg]
DEUTERON_MASS = 3.3435837724e-27          # Kilograms [kg] (D+)
TRITON_MASS = 5.0073567446e-27            # Kilograms [kg] (T+)
ALPHA_MASS = 6.6446573357e-27             # Kilograms [kg] (He2+)

VACUUM_PERMITTIVITY = 8.8541878128e-12    # Farads per meter [F/m]
VACUUM_PERMEABILITY = 1.25663706212e-6    # Henry per meter [H/m]
SPEED_OF_LIGHT = 299792458.0              # Meters per second [m/s]
BOLTZMANN_CONSTANT = 1.380649e-23         # Joules per Kelvin [J/K]
COULOMB_CONSTANT = 1.0 / (4.0 * np.pi * VACUUM_PERMITTIVITY) # ~ 8.98755e9 N m^2/C^2

# =============================================================================
# Species Properties
# =============================================================================
@dataclass(frozen=True)
class Species:
    name: str
    symbol: str
    charge_q: float       # In units of elementary charge e
    mass_m: float         # In units of proton mass m_p (or normalized unit)
    color: str            # Hex or RGBA string for visualization

SPECIES_ELECTRON = Species(name="Electron", symbol="e-", charge_q=-1.0, mass_m=1.0 / 1836.15, color="#3498db")
SPECIES_PROTON = Species(name="Proton", symbol="p+", charge_q=1.0, mass_m=1.0, color="#e74c3c")
SPECIES_DEUTERON = Species(name="Deuteron", symbol="D+", charge_q=1.0, mass_m=2.014, color="#e67e22")
SPECIES_TRITON = Species(name="Triton", symbol="T+", charge_q=1.0, mass_m=2.980, color="#f39c12")
SPECIES_ALPHA = Species(name="Alpha Particle", symbol="He2+", charge_q=2.0, mass_m=3.972, color="#9b59b6")

# Normalized species for fast real-time interactive simulation
# (using reduced mass ratio to prevent extreme electron time-step constraints)
SPECIES_NORMALIZED_ION = Species(name="Ion", symbol="i+", charge_q=1.0, mass_m=1.0, color="#e74c3c")
SPECIES_NORMALIZED_ELECTRON = Species(name="Electron", symbol="e-", charge_q=-1.0, mass_m=1.0, color="#3498db")

# =============================================================================
# Normalized Simulation Defaults
# =============================================================================
DEFAULT_KE = 1000.0                       # Coulomb constant multiplier
DEFAULT_SOFTENING = 1.0                   # Plumber / softening length to prevent singular 1/r^2
DEFAULT_THETA = 0.25                      # Barnes-Hut opening angle threshold
DEFAULT_DT = 0.001                        # Default integration time step
DEFAULT_BOUNDARY_RADIUS = 200.0           # Default spatial confinement radius
