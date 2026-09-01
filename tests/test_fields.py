"""
Tests for electromagnetic field models and Maxwell constraints.
"""

import pytest
import numpy as np
from tokamak_py.fields import (
    eval_tokamak_field_3d, eval_tokamak_field_2d,
    eval_magnetic_mirror_3d, eval_iec_fusor_field, eval_exb_drift_field
)


def test_tokamak_field_finite_and_divergence_free():
    """
    Tests that the 3D Tokamak field is smooth, finite, and approximately
    divergence-free (div B = dBx/dx + dBy/dy + dBz/dz = 0).
    """
    # Sample point inside the plasma core
    x0, y0, z0 = 100.0, 0.0, 10.0
    pos = np.array([[x0, y0, z0]], dtype=np.float64)
    _, B0 = eval_tokamak_field_3d(pos, B0=2.0, R0=100.0, a=40.0, B_theta0=0.3)

    assert not np.any(np.isnan(B0))
    assert not np.any(np.isinf(B0))

    # Numerical divergence via central differences
    h = 1e-4
    pts = np.array([
        [x0 + h, y0, z0], [x0 - h, y0, z0],
        [x0, y0 + h, z0], [x0, y0 - h, z0],
        [x0, y0, z0 + h], [x0, y0, z0 - h]
    ], dtype=np.float64)
    _, B_pts = eval_tokamak_field_3d(pts, B0=2.0, R0=100.0, a=40.0, B_theta0=0.3)

    dBx_dx = (B_pts[0, 0] - B_pts[1, 0]) / (2.0 * h)
    dBy_dy = (B_pts[2, 1] - B_pts[3, 1]) / (2.0 * h)
    dBz_dz = (B_pts[4, 2] - B_pts[5, 2]) / (2.0 * h)

    div_B = dBx_dx + dBy_dy + dBz_dz
    # Divergence should be close to zero relative to field gradients
    assert abs(div_B) < 0.05


def test_magnetic_mirror_field():
    """Tests that the magnetic mirror field has stronger field at the ends than at the center."""
    center = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    throat = np.array([[0.0, 0.0, 80.0]], dtype=np.float64)

    _, B_center = eval_magnetic_mirror_3d(center, B0=1.0, L=80.0, Rm=3.0)
    _, B_throat = eval_magnetic_mirror_3d(throat, B0=1.0, L=80.0, Rm=3.0)

    # Throat field magnitude should be Rm times center field
    mag_center = np.linalg.norm(B_center[0])
    mag_throat = np.linalg.norm(B_throat[0])
    assert np.isclose(mag_throat / mag_center, 3.0, rtol=0.01)


def test_fusor_attractive_field():
    """Fusor electric field should point radially inward toward origin."""
    pos = np.array([[50.0, 50.0]], dtype=np.float64)
    E_field, _ = eval_iec_fusor_field(pos, V0=2000.0, r_grid=30.0)

    # Radial vector r = (50, 50)
    # E-field must point in opposite direction (negative dot product)
    assert E_field[0, 0] < 0.0
    assert E_field[0, 1] < 0.0
