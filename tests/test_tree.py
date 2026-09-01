"""
Tests for Barnes-Hut 2D Quadtree and 3D Octree modules.
"""

import pytest
import numpy as np
from tokamak_py.tree import (
    get_bounding_box_2d, get_quadrant, build_quadtree,
    get_bounding_box_3d, get_octant, build_octree
)


def test_bounding_box_2d():
    positions = np.array([
        [-10.0, -20.0],
        [30.0, 40.0],
        [0.0, 0.0]
    ], dtype=np.float64)
    cx, cy, size = get_bounding_box_2d(positions)
    assert np.isclose(cx, 10.0)
    assert np.isclose(cy, 10.0)
    assert size >= 60.0


def test_quadrant_assignment():
    cx, cy = 0.0, 0.0
    assert get_quadrant(-1.0, -1.0, cx, cy) == 0 # SW
    assert get_quadrant(-1.0,  1.0, cx, cy) == 1 # NW
    assert get_quadrant( 1.0, -1.0, cx, cy) == 2 # SE
    assert get_quadrant( 1.0,  1.0, cx, cy) == 3 # NE


def test_build_quadtree():
    np.random.seed(42)
    positions = np.random.uniform(-50, 50, (100, 2))
    charges = np.ones(100)
    charges[50:] = -1.0

    cx, cy, size = get_bounding_box_2d(positions)
    node_f, node_i, count = build_quadtree(positions, charges, cx, cy, size)

    assert count > 1
    assert node_f.shape[0] >= count
    # Root node charges
    assert np.isclose(node_f[0, 2], np.sum(charges))
    assert np.isclose(node_f[0, 3], np.sum(np.abs(charges)))


def test_bounding_box_3d():
    positions = np.array([
        [-10.0, -20.0, -5.0],
        [30.0, 40.0, 15.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float64)
    cx, cy, cz, size = get_bounding_box_3d(positions)
    assert np.isclose(cx, 10.0)
    assert np.isclose(cy, 10.0)
    assert np.isclose(cz, 5.0)
    assert size >= 60.0


def test_octant_assignment():
    cx, cy, cz = 0.0, 0.0, 0.0
    assert get_octant(-1.0, -1.0, -1.0, cx, cy, cz) == 0
    assert get_octant( 1.0, -1.0, -1.0, cx, cy, cz) == 1
    assert get_octant(-1.0,  1.0, -1.0, cx, cy, cz) == 2
    assert get_octant( 1.0,  1.0, -1.0, cx, cy, cz) == 3
    assert get_octant(-1.0, -1.0,  1.0, cx, cy, cz) == 4
    assert get_octant( 1.0, -1.0,  1.0, cx, cy, cz) == 5
    assert get_octant(-1.0,  1.0,  1.0, cx, cy, cz) == 6
    assert get_octant( 1.0,  1.0,  1.0, cx, cy, cz) == 7


def test_build_octree():
    np.random.seed(42)
    positions = np.random.uniform(-50, 50, (100, 3))
    charges = np.ones(100)
    charges[50:] = -1.0

    cx, cy, cz, size = get_bounding_box_3d(positions)
    node_f, node_i, count = build_octree(positions, charges, cx, cy, cz, size)

    assert count > 1
    assert np.isclose(node_f[0, 3], np.sum(charges))
    assert np.isclose(node_f[0, 4], np.sum(np.abs(charges)))
