"""
Tests for Symplectic Boris Integrator and Boundary limiters.
"""

import pytest
import numpy as np
from tokamak_py.integrators import (
    boris_step_3d, boris_step_2d,
    apply_boundary_circular_2d, apply_boundary_tokamak_3d
)


def test_cyclotron_frequency_and_energy_conservation():
    """
    Validates that Boris algorithm reproduces analytical cyclotron gyromotion:
    omega_c = q * B / m, Period T = 2 * pi / omega_c, Gyroradius r_L = v_perp / omega_c.
    Kinetic energy must remain strictly constant under magnetic rotation.
    """
    m = 1.0
    q = 1.0
    Bz = 2.0
    v0 = 10.0

    omega_c = q * Bz / m # = 2.0 rad/s
    T_period = 2.0 * np.pi / omega_c # ~ 3.14159 s
    r_L = v0 / omega_c # = 5.0

    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    velocities = np.array([[v0, 0.0, 0.0]], dtype=np.float64)
    masses = np.array([m], dtype=np.float64)
    charges = np.array([q], dtype=np.float64)
    forces = np.zeros((1, 3), dtype=np.float64)
    B_fields = np.array([[0.0, 0.0, Bz]], dtype=np.float64)

    dt = 0.001
    n_steps = int(round(T_period / dt))

    initial_ke = 0.5 * m * v0**2

    for _ in range(n_steps):
        boris_step_3d(positions, velocities, masses, charges, forces, B_fields, dt)
        current_ke = 0.5 * m * np.sum(velocities[0]**2)
        # Magnetic force does zero work: KE strictly conserved
        assert np.isclose(current_ke, initial_ke, atol=1e-10)

    # Particle should complete a closed loop and return close to origin
    assert np.isclose(positions[0, 0], 0.0, atol=0.05)
    assert np.isclose(positions[0, 1], 0.0, atol=0.05)
    assert np.isclose(velocities[0, 0], v0, atol=0.05)


def test_exb_drift_velocity():
    """
    Validates universal E x B drift velocity: v_d = (E x B) / B^2.
    For E = (0, Ey, 0) and B = (0, 0, Bz), the guiding-center drift is along +X with speed Ey / Bz.
    """
    m = 1.0
    q = 1.0
    Ey = 50.0
    Bz = 5.0
    expected_vd = Ey / Bz # = 10.0

    # 1. Steady drift initialized at guiding-center velocity
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    velocities = np.array([[expected_vd, 0.0, 0.0]], dtype=np.float64)
    masses = np.array([m], dtype=np.float64)
    charges = np.array([q], dtype=np.float64)
    forces = np.array([[0.0, q * Ey, 0.0]], dtype=np.float64)
    B_fields = np.array([[0.0, 0.0, Bz]], dtype=np.float64)

    dt = 0.001
    total_time = 1.0
    n_steps = int(total_time / dt)

    for _ in range(n_steps):
        boris_step_3d(positions, velocities, masses, charges, forces, B_fields, dt)

    measured_vd = positions[0, 0] / total_time
    assert np.isclose(measured_vd, expected_vd, rtol=0.001)

    # 2. Starting from rest: average velocity over exact integer cyclotron periods
    omega_c = q * Bz / m # = 5.0 rad/s
    T_period = 2.0 * np.pi / omega_c # ~ 1.2566 s
    total_time_rest = 3 * T_period
    n_steps_rest = int(round(total_time_rest / dt))

    pos_rest = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    vel_rest = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

    for _ in range(n_steps_rest):
        boris_step_3d(pos_rest, vel_rest, masses, charges, forces, B_fields, dt)

    avg_vd_rest = pos_rest[0, 0] / (n_steps_rest * dt)
    assert np.isclose(avg_vd_rest, expected_vd, rtol=0.005)


def test_boundary_circular_reflection():
    """Elastic reflection should reverse normal velocity component."""
    r_bound = 100.0
    positions = np.array([[105.0, 0.0]], dtype=np.float64) # Outside boundary
    velocities = np.array([[10.0, 5.0]], dtype=np.float64) # Moving outwards

    apply_boundary_circular_2d(positions, velocities, r_bound)

    assert np.isclose(positions[0, 0], 100.0)
    assert np.isclose(positions[0, 1], 0.0)
    assert velocities[0, 0] < 0.0 # Normal velocity reversed
    assert np.isclose(velocities[0, 1], 5.0) # Tangential velocity preserved
