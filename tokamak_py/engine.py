"""
Tokamak-Py: Unified Plasma Physics Engine.

Seamlessly integrates Barnes-Hut quadtrees/octrees, exact direct N-body solvers,
symplectic Boris electromagnetic integration, and customizable confinement
geometries for both 2D and 3D simulations.
"""

from typing import Optional, Union, Dict, Any
import numpy as np

from .constants import (
    DEFAULT_KE, DEFAULT_SOFTENING, DEFAULT_THETA, DEFAULT_DT,
    DEFAULT_BOUNDARY_RADIUS, SPECIES_NORMALIZED_ION, SPECIES_NORMALIZED_ELECTRON
)
from .fields import FieldConfig, ZeroFieldConfig, Tokamak2DConfig
from .tree import (
    get_bounding_box_2d, build_quadtree, calculate_forces_2d,
    get_bounding_box_3d, build_octree, calculate_forces_3d
)
from .direct import calculate_direct_forces_2d, calculate_direct_forces_3d
from .integrators import (
    boris_step_2d, boris_step_3d,
    apply_boundary_circular_2d, apply_boundary_tokamak_3d, apply_boundary_spherical_3d
)
from .presets import get_preset, SimulationPreset


class PhysicsEngine:
    """
    Unified Plasma Physics Engine for N-body electrostatic and electromagnetic confinement.
    
    Supports:
    - 2D and 3D particle positions with 3D velocity vectors (2D-3V and 3D-3V).
    - Symplectic Boris Lorentz-force integration (F = q(E + v x B)).
    - Barnes-Hut Quadtree/Octree (O(N log N)) and Direct Pairwise (O(N^2)) solvers.
    - Multiple plasma confinement presets (Tokamak, Mirror, Fusor, ExB, Classic).
    """

    def __init__(
        self,
        n_particles: Optional[int] = None,
        dt: Optional[float] = None,
        theta: Optional[float] = None,
        softening: Optional[float] = None,
        boundary_radius: Optional[float] = None,
        k_e: Optional[float] = None,
        dimension: Optional[int] = None,
        field_config: Optional[FieldConfig] = None,
        solver: str = "barnes_hut",
        preset: Optional[Union[str, SimulationPreset]] = None,
        initial_thermal_speed: Optional[float] = None,
        species_mode: Optional[str] = None
    ):
        # Handle preset configuration if passed
        self.preset_obj: Optional[SimulationPreset] = None
        if preset is not None:
            if isinstance(preset, str):
                self.preset_obj = get_preset(preset)
            else:
                self.preset_obj = preset
                
            self.dimension = dimension if dimension is not None else self.preset_obj.dimension
            self.n_particles = n_particles if n_particles is not None else self.preset_obj.n_particles
            self.dt = dt if dt is not None else self.preset_obj.dt
            self.theta = theta if theta is not None else self.preset_obj.theta
            self.softening = softening if softening is not None else self.preset_obj.softening
            self.k_e = k_e if k_e is not None else self.preset_obj.k_e
            self.boundary_radius = boundary_radius if boundary_radius is not None else self.preset_obj.boundary_param
            self.boundary_type = self.preset_obj.boundary_type
            self.R0 = getattr(self.preset_obj, 'R0', 100.0)
            self.field_config = field_config if field_config is not None else self.preset_obj.field_config
            self.initial_thermal_speed = initial_thermal_speed if initial_thermal_speed is not None else self.preset_obj.initial_thermal_speed
            self.species_mode = species_mode if species_mode is not None else self.preset_obj.species_mode
        else:
            self.dimension = dimension if dimension is not None else 2
            self.n_particles = n_particles if n_particles is not None else 1000
            self.dt = dt if dt is not None else DEFAULT_DT
            self.theta = theta if theta is not None else DEFAULT_THETA
            self.softening = softening if softening is not None else DEFAULT_SOFTENING
            self.boundary_radius = boundary_radius if boundary_radius is not None else DEFAULT_BOUNDARY_RADIUS
            self.boundary_type = "circular" if self.dimension == 2 else "spherical"
            self.R0 = 100.0
            self.k_e = k_e if k_e is not None else DEFAULT_KE
            self.field_config = field_config if field_config is not None else ZeroFieldConfig()
            self.initial_thermal_speed = initial_thermal_speed if initial_thermal_speed is not None else 0.0
            self.species_mode = species_mode if species_mode is not None else "balanced"

        self.solver = solver.lower() # 'barnes_hut' or 'direct'
        self.step_count = 0
        self.b_field_scale = 1.0 # Interactive multiplier for B field

        # Allocate arrays
        self._initialize_particles()
        
        # Telemetry metrics
        self.kinetic_energy = 0.0
        self.potential_energy = 0.0
        self._update_kinetic_energy()
        
        # Initial force calculation
        self._update_forces()
        self.initial_total_energy = self.get_total_energy()

    def _initialize_particles(self):
        """Initializes particle positions, velocities, masses, and charges."""
        n = self.n_particles
        dim = self.dimension
        
        self.positions = np.zeros((n, dim), dtype=np.float64)
        self.velocities = np.zeros((n, 3), dtype=np.float64) # Always 3V
        self.masses = np.ones(n, dtype=np.float64)
        self.charges = np.ones(n, dtype=np.float64)
        
        # Set charges & species
        if self.species_mode == "ions_only":
            self.charges[:] = 1.0
        elif self.species_mode == "electrons_only":
            self.charges[:] = -1.0
        else: # "balanced"
            half = n // 2
            self.charges[:half] = 1.0
            self.charges[half:] = -1.0
            
        # Spatial placement based on confinement geometry
        if self.dimension == 2:
            radii = np.random.uniform(0.05 * self.boundary_radius, self.boundary_radius * 0.8, n)
            angles = np.random.uniform(0.0, 2.0 * np.pi, n)
            self.positions[:, 0] = radii * np.cos(angles)
            self.positions[:, 1] = radii * np.sin(angles)
        else: # 3D
            if self.boundary_type == "tokamak":
                # Torus placement: R around R0, r around 0 <= a
                a = self.boundary_radius
                toroidal_angles = np.random.uniform(0.0, 2.0 * np.pi, n)
                poloidal_angles = np.random.uniform(0.0, 2.0 * np.pi, n)
                minor_r = np.random.uniform(0.1 * a, 0.75 * a, n)
                
                R = self.R0 + minor_r * np.cos(poloidal_angles)
                self.positions[:, 0] = R * np.cos(toroidal_angles)
                self.positions[:, 1] = R * np.sin(toroidal_angles)
                self.positions[:, 2] = minor_r * np.sin(poloidal_angles)
            else:
                # Spherical distribution
                r = np.random.uniform(0.1 * self.boundary_radius, 0.8 * self.boundary_radius, n)
                phi = np.random.uniform(0.0, 2.0 * np.pi, n)
                theta = np.arccos(np.random.uniform(-1.0, 1.0, n))
                self.positions[:, 0] = r * np.sin(theta) * np.cos(phi)
                self.positions[:, 1] = r * np.sin(theta) * np.sin(phi)
                self.positions[:, 2] = r * np.cos(theta)
                
        # Thermal velocities (Maxwellian distribution)
        if self.initial_thermal_speed > 0.0:
            sigma = self.initial_thermal_speed / np.sqrt(3.0)
            self.velocities = np.random.normal(0.0, sigma, size=(n, 3))
            
        self.coulomb_forces = np.zeros_like(self.positions)

    def _update_forces(self):
        """Computes internal Coulomb forces and evaluates external fields."""
        if self.dimension == 2:
            if self.solver == "direct" or self.n_particles < 60:
                self.coulomb_forces, self.potential_energy = calculate_direct_forces_2d(
                    self.positions, self.charges, self.k_e, self.softening
                )
            else:
                cx, cy, size = get_bounding_box_2d(self.positions)
                node_f, node_i, _ = build_quadtree(self.positions, self.charges, cx, cy, size)
                self.coulomb_forces, self.potential_energy = calculate_forces_2d(
                    self.positions, self.charges, node_f, node_i, self.k_e, self.softening, self.theta
                )
        else: # 3D
            if self.solver == "direct" or self.n_particles < 60:
                self.coulomb_forces, self.potential_energy = calculate_direct_forces_3d(
                    self.positions, self.charges, self.k_e, self.softening
                )
            else:
                cx, cy, cz, size = get_bounding_box_3d(self.positions)
                node_f, node_i, _ = build_octree(self.positions, self.charges, cx, cy, cz, size)
                self.coulomb_forces, self.potential_energy = calculate_forces_3d(
                    self.positions, self.charges, node_f, node_i, self.k_e, self.softening, self.theta
                )

    def step(self):
        """
        Advances the simulation by one time step dt using the symplectic Boris algorithm.
        Conserves phase-space volume and maintains exact kinetic energy under magnetic rotation.
        """
        self.step_count += 1
        
        # 1. Evaluate external fields at current positions
        E_ext, B_ext = self.field_config.get_fields(self.positions)
        B_ext = B_ext * self.b_field_scale
        
        # Total electric force = Coulomb forces + q * E_external
        total_forces = self.coulomb_forces + self.charges[:, None] * E_ext
        
        # 2. Advance velocities and positions using Boris scheme
        if self.dimension == 2:
            boris_step_2d(
                self.positions, self.velocities, self.masses, self.charges,
                total_forces, B_ext, self.dt
            )
            # Apply 2D boundary limiter
            if self.boundary_type == "circular":
                apply_boundary_circular_2d(self.positions, self.velocities, self.boundary_radius)
        else:
            boris_step_3d(
                self.positions, self.velocities, self.masses, self.charges,
                total_forces, B_ext, self.dt
            )
            # Apply 3D boundary limiters
            if self.boundary_type == "tokamak":
                apply_boundary_tokamak_3d(self.positions, self.velocities, self.R0, self.boundary_radius)
            elif self.boundary_type == "spherical":
                apply_boundary_spherical_3d(self.positions, self.velocities, self.boundary_radius)
                
        # 3. Calculate new internal forces for the next step
        self._update_forces()
        
        # 4. Update kinetic energy metric
        self._update_kinetic_energy()

    def _update_kinetic_energy(self):
        """Calculates total kinetic energy sum(0.5 * m * v^2)."""
        speed_sq = np.sum(self.velocities ** 2, axis=1)
        self.kinetic_energy = 0.5 * np.sum(self.masses * speed_sq)

    def get_total_energy(self) -> float:
        """Returns total Hamiltonian energy (Kinetic + Potential)."""
        return self.kinetic_energy + self.potential_energy

    def get_relative_energy_drift(self) -> float:
        """Returns relative deviation |E(t) - E(0)| / |E(0)|."""
        if abs(self.initial_total_energy) < 1e-9:
            return 0.0
        return abs(self.get_total_energy() - self.initial_total_energy) / abs(self.initial_total_energy)

    def get_speed_distribution(self) -> np.ndarray:
        """Returns array of particle speeds ||v||."""
        return np.sqrt(np.sum(self.velocities ** 2, axis=1))

    def get_temperature(self) -> float:
        """Estimates plasma temperature via 2/3 * <KE> / k_B."""
        return float(np.mean(0.5 * self.masses * np.sum(self.velocities ** 2, axis=1)))

    def reset(self):
        """Resets the simulation to initial state."""
        self.step_count = 0
        self._initialize_particles()
        self._update_forces()
        self.initial_total_energy = self.get_total_energy()
