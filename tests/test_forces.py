"""
Tests for N-body force algorithms: Direct O(N^2) vs Barnes-Hut O(N log N).
"""

import pytest
import numpy as np
from tokamak_py.direct import calculate_direct_forces_2d, calculate_direct_forces_3d
from tokamak_py.tree import (
    get_bounding_box_2d, build_quadtree, calculate_forces_2d,
    get_bounding_box_3d, build_octree, calculate_forces_3d
)


def test_forces_2d_accuracy():
    np.random.seed(123)
    n = 200
    positions = np.random.uniform(-100, 100, (n, 2))
    charges = np.ones(n)
    charges[n // 2:] = -1.0

    k_e = 500.0
    softening = 2.0

    # 1. Exact Direct
    f_direct, pe_direct = calculate_direct_forces_2d(positions, charges, k_e, softening)

    # 2. Barnes-Hut with high precision theta=0.1
    cx, cy, s = get_bounding_box_2d(positions)
    nf, ni, _ = build_quadtree(positions, charges, cx, cy, s)
    f_tree_fine, pe_tree_fine = calculate_forces_2d(positions, charges, nf, ni, k_e, softening, theta=0.1)

    # Relative error should be < 0.5% with theta=0.1
    rel_err_f = np.linalg.norm(f_tree_fine - f_direct) / np.linalg.norm(f_direct)
    assert rel_err_f < 0.005
    assert np.isclose(pe_tree_fine, pe_direct, rtol=0.01)


def test_forces_3d_accuracy():
    np.random.seed(456)
    n = 150
    positions = np.random.uniform(-80, 80, (n, 3))
    charges = np.ones(n)
    charges[n // 2:] = -1.0

    k_e = 500.0
    softening = 2.0

    f_direct, pe_direct = calculate_direct_forces_3d(positions, charges, k_e, softening)

    cx, cy, cz, s = get_bounding_box_3d(positions)
    nf, ni, _ = build_octree(positions, charges, cx, cy, cz, s)
    f_tree, pe_tree = calculate_forces_3d(positions, charges, nf, ni, k_e, softening, theta=0.15)

    rel_err_f = np.linalg.norm(f_tree - f_direct) / np.linalg.norm(f_direct)
    assert rel_err_f < 0.01
    assert np.isclose(pe_tree, pe_direct, rtol=0.02)


def test_newton_third_law():
    """Total internal Coulomb force on closed system must sum to zero."""
    np.random.seed(789)
    n = 100
    positions = np.random.uniform(-50, 50, (n, 2))
    charges = np.random.choice([-1.0, 1.0], size=n)

    f_direct, _ = calculate_direct_forces_2d(positions, charges, 1000.0, 1.0)
    net_force = np.sum(f_direct, axis=0)
    assert np.allclose(net_force, [0.0, 0.0], atol=1e-8)
