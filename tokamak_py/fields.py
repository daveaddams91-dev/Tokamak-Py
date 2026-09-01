"""
Tokamak-Py: Electromagnetic Field Models.

Implements Numba-accelerated magnetic and electric field models:
- Tokamak Torus (Toroidal + Poloidal fields with flux surfaces and safety factor q)
- 2D Poloidal Tokamak Slice
- Magnetic Mirror / Bottle Trap (satisfies div B = 0)
- Inertial Electrostatic Confinement (IEC Fusor)
- Orthogonal E x B Drift Configuration
"""

import numpy as np
from numba import njit

# =============================================================================
# Vectorized Numba Field Functions
# =============================================================================

@njit(fastmath=True)
def eval_tokamak_field_3d(positions, B0=1.0, R0=100.0, a=40.0, B_theta0=0.2, Bz_eq=0.0):
    """
    Computes magnetic field for a Tokamak torus in 3D Cartesian coordinates (x, y, z).
    Torus center at origin (0, 0, 0), torus axis along Z.
    
    B = B_toroidal * phi_hat + B_poloidal * theta_hat + Bz_eq * z_hat
    
    Parameters:
    -----------
    positions : ndarray of shape (N, 3)
    B0 : float, on-axis toroidal magnetic field strength
    R0 : float, major radius
    a  : float, minor radius (plasma boundary)
    B_theta0 : float, poloidal magnetic field at minor radius a (from plasma current)
    Bz_eq : float, vertical equilibrium field (counteracts outward hoop expansion)
    
    Returns:
    --------
    B_field : ndarray of shape (N, 3)
    E_field : ndarray of shape (N, 3) (zero for ideal static field)
    """
    n = positions.shape[0]
    B_field = np.zeros((n, 3), dtype=np.float64)
    E_field = np.zeros((n, 3), dtype=np.float64)
    
    for i in range(n):
        x = positions[i, 0]
        y = positions[i, 1]
        z = positions[i, 2]
        
        R_sq = x * x + y * y
        R = np.sqrt(R_sq)
        if R < 1e-6:
            R = 1e-6
            
        # Toroidal unit vector: phi_hat = (-y/R, x/R, 0)
        phi_x = -y / R
        phi_y =  x / R
        
        # Toroidal field: B_phi = B0 * (R0 / R)
        B_phi = B0 * (R0 / R)
        
        # Distance from magnetic axis (R0 in the x-y plane)
        delta_R = R - R0
        r_sq = delta_R * delta_R + z * z
        r = np.sqrt(r_sq)
        if r < 1e-6:
            r = 1e-6
            
        # Poloidal field from plasma current: B_theta(r) = B_theta0 * (r / a) for r <= a
        # (for r > a, B_theta(r) = B_theta0 * (a / r) by Ampere's law)
        if r <= a:
            B_theta = B_theta0 * (r / a)
        else:
            B_theta = B_theta0 * (a / r)
            
        # Cylindrical radial unit vector: R_hat = (x/R, y/R, 0)
        R_hat_x = x / R
        R_hat_y = y / R
        
        # Poloidal unit vector: theta_hat = phi_hat x r_hat
        # r_hat = (delta_R/r) * R_hat + (z/r) * z_hat
        # theta_hat = -(z/r) * R_hat + (delta_R/r) * z_hat
        cos_theta = delta_R / r
        sin_theta = z / r
        
        theta_x = -sin_theta * R_hat_x
        theta_y = -sin_theta * R_hat_y
        theta_z =  cos_theta
        
        # Total B = B_phi * phi_hat + B_theta * theta_hat + Bz_eq * z_hat
        B_field[i, 0] = B_phi * phi_x + B_theta * theta_x
        B_field[i, 1] = B_phi * phi_y + B_theta * theta_y
        B_field[i, 2] = B_theta * theta_z + Bz_eq
        
    return E_field, B_field


@njit(fastmath=True)
def eval_tokamak_field_2d(positions, B0=1.0, R0=100.0, a=40.0, B_theta0=0.2):
    """
    Computes 2D poloidal plane slice representation of Tokamak fields.
    Coordinates (x, y) represent (R - R0, Z).
    
    Out-of-plane toroidal field: Bz = B0 / (1 + x / R0)
    In-plane poloidal field: (Bx, By) circling around magnetic axis (0, 0).
    
    Returns:
    --------
    E_field : (N, 2)
    B_field : (N, 3) -> [Bx, By, Bz]
    """
    n = positions.shape[0]
    B_field = np.zeros((n, 3), dtype=np.float64)
    E_field = np.zeros((n, 2), dtype=np.float64)
    
    for i in range(n):
        x = positions[i, 0]
        y = positions[i, 1]
        
        r_sq = x * x + y * y
        r = np.sqrt(r_sq)
        if r < 1e-6:
            r = 1e-6
            
        if r <= a:
            B_theta = B_theta0 * (r / a)
        else:
            B_theta = B_theta0 * (a / r)
            
        # Poloidal field in (x, y) plane circling origin: (-y/r, x/r)
        B_field[i, 0] = -B_theta * (y / r)
        B_field[i, 1] =  B_theta * (x / r)
        
        # Toroidal field perpendicular to poloidal plane
        # Stronger on inboard (x < 0), weaker on outboard (x > 0) -> causes banana orbits!
        denom = 1.0 + x / R0
        if denom < 0.1:
            denom = 0.1
        B_field[i, 2] = B0 / denom
        
    return E_field, B_field


@njit(fastmath=True)
def eval_magnetic_mirror_3d(positions, B0=1.0, L=80.0, Rm=3.0):
    """
    Computes magnetic field for a Magnetic Mirror (Bottle Trap) in 3D.
    Mirror coils located at z = +/- L.
    
    B_z(r, z) = B0 * [1 + (Rm - 1) * (z / L)^2]
    B_r(r, z) = -0.5 * r * (dBz/dz) = - (Rm - 1) * B0 * (z * r) / L^2
    This field satisfies div B = 0 to first order in r.
    """
    n = positions.shape[0]
    B_field = np.zeros((n, 3), dtype=np.float64)
    E_field = np.zeros((n, 3), dtype=np.float64)
    
    k = (Rm - 1.0) / (L * L)
    
    for i in range(n):
        x = positions[i, 0]
        y = positions[i, 1]
        z = positions[i, 2]
        
        r_sq = x * x + y * y
        r = np.sqrt(r_sq)
        
        Bz = B0 * (1.0 + k * z * z)
        
        # Radial field components: Bx = Br * (x/r), By = Br * (y/r)
        # Br = -B0 * k * z * r
        Bx = -B0 * k * z * x
        By = -B0 * k * z * y
        
        B_field[i, 0] = Bx
        B_field[i, 1] = By
        B_field[i, 2] = Bz
        
    return E_field, B_field


@njit(fastmath=True)
def eval_iec_fusor_field(positions, V0=2000.0, r_grid=30.0, softening=2.0):
    """
    Inertial Electrostatic Confinement (IEC Fusor) potential well.
    Central transparent spherical cathode grid creates deep negative potential well.
    """
    n = positions.shape[0]
    dim = positions.shape[1]
    E_field = np.zeros((n, dim), dtype=np.float64)
    B_field = np.zeros((n, 3), dtype=np.float64)
    
    for i in range(n):
        r_sq = softening * softening
        for d in range(dim):
            r_sq += positions[i, d] ** 2
        r = np.sqrt(r_sq)
        
        # Potential well: attractive inward radial electric field toward center
        if r < r_grid:
            E_mag = -V0 * (r / (r_grid * r_grid))
        else:
            E_mag = -V0 / (r * r)
            
        for d in range(dim):
            E_field[i, d] = E_mag * (positions[i, d] / r)
            
    return E_field, B_field


@njit(fastmath=True)
def eval_exb_drift_field(positions, Ey=50.0, Bz=2.0):
    """
    Uniform orthogonal E and B field configuration.
    Demonstrates pure E x B drift: v_d = (Ey / Bz) along x-axis.
    """
    n = positions.shape[0]
    dim = positions.shape[1]
    E_field = np.zeros((n, dim), dtype=np.float64)
    B_field = np.zeros((n, 3), dtype=np.float64)
    
    E_field[:, 1] = Ey
    B_field[:, 2] = Bz
    
    return E_field, B_field


# =============================================================================
# Object-Oriented Configuration Wrappers
# =============================================================================

class FieldConfig:
    """Base class for electromagnetic field configurations."""
    def get_fields(self, positions):
        raise NotImplementedError

class Tokamak3DConfig(FieldConfig):
    def __init__(self, B0=1.0, R0=100.0, a=40.0, B_theta0=0.2, Bz_eq=0.0):
        self.B0 = B0
        self.R0 = R0
        self.a = a
        self.B_theta0 = B_theta0
        self.Bz_eq = Bz_eq
        self.name = "Tokamak Torus (3D)"
        
    def get_fields(self, positions):
        return eval_tokamak_field_3d(positions, self.B0, self.R0, self.a, self.B_theta0, self.Bz_eq)

class Tokamak2DConfig(FieldConfig):
    def __init__(self, B0=1.0, R0=100.0, a=50.0, B_theta0=0.2):
        self.B0 = B0
        self.R0 = R0
        self.a = a
        self.B_theta0 = B_theta0
        self.name = "Tokamak Poloidal Slice (2D)"
        
    def get_fields(self, positions):
        return eval_tokamak_field_2d(positions, self.B0, self.R0, self.a, self.B_theta0)

class MagneticMirrorConfig(FieldConfig):
    def __init__(self, B0=1.0, L=80.0, Rm=3.0):
        self.B0 = B0
        self.L = L
        self.Rm = Rm
        self.name = "Magnetic Mirror (Bottle Trap)"
        
    def get_fields(self, positions):
        return eval_magnetic_mirror_3d(positions, self.B0, self.L, self.Rm)

class IECFusorConfig(FieldConfig):
    def __init__(self, V0=2000.0, r_grid=30.0):
        self.V0 = V0
        self.r_grid = r_grid
        self.name = "Inertial Electrostatic Confinement (Fusor)"
        
    def get_fields(self, positions):
        return eval_iec_fusor_field(positions, self.V0, self.r_grid)

class ExBDriftConfig(FieldConfig):
    def __init__(self, Ey=50.0, Bz=2.0):
        self.Ey = Ey
        self.Bz = Bz
        self.name = "E x B Drift Configuration"
        
    def get_fields(self, positions):
        return eval_exb_drift_field(positions, self.Ey, self.Bz)

class ZeroFieldConfig(FieldConfig):
    def __init__(self):
        self.name = "Self-Consistent Electrostatic (No External Field)"
        
    def get_fields(self, positions):
        n = positions.shape[0]
        dim = positions.shape[1]
        return np.zeros((n, dim), dtype=np.float64), np.zeros((n, 3), dtype=np.float64)
