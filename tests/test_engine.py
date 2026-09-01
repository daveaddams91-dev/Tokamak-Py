"""
Tests for unified PhysicsEngine integration, presets, and energy stability.
"""

import pytest
import numpy as np
from tokamak_py.engine import PhysicsEngine
from tokamak_py.presets import PRESETS


@pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
def test_engine_all_presets_initialize_and_step(preset_name):
    """Verifies that every preset initializes cleanly and steps without error."""
    engine = PhysicsEngine(preset=preset_name, n_particles=80)
    assert engine.positions.shape[0] == 80
    assert engine.velocities.shape[0] == 80

    initial_energy = engine.get_total_energy()
    assert not np.isnan(initial_energy)
    assert not np.isinf(initial_energy)

    # Step forward 5 times
    for _ in range(5):
        engine.step()

    final_energy = engine.get_total_energy()
    assert not np.isnan(final_energy)
    assert not np.isinf(final_energy)
    assert engine.step_count == 5


def test_engine_reset():
    engine = PhysicsEngine(preset="tokamak_2d", n_particles=50)
    for _ in range(10):
        engine.step()
    assert engine.step_count == 10

    engine.reset()
    assert engine.step_count == 0
    assert engine.positions.shape[0] == 50


def test_engine_species_mode():
    engine_ions = PhysicsEngine(preset="fusor", species_mode="ions_only", n_particles=60)
    assert np.all(engine_ions.charges == 1.0)

    engine_balanced = PhysicsEngine(preset="classic_2d", species_mode="balanced", n_particles=60)
    assert np.sum(engine_balanced.charges == 1.0) == 30
    assert np.sum(engine_balanced.charges == -1.0) == 30
