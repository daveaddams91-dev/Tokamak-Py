"""
Tokamak-Py: Symplectic Integrators and Boundary Conditions.

Implements:
1. Boris Algorithm: The gold-standard symplectic/phase-space-conserving integrator
   for charged particles in electromagnetic fields (F = q(E + v x B)).
2. Velocity Verlet Integrator: Symplectic integrator for electrostatic systems (B = 0).
3. Kinematic Boundaries: 2D circular boundary and 3D toroidal/spherical/cylindrical limiters.
"""

import numpy as np
from numba import njit, prange

# =============================================================================
# Symplectic Boris Integrator
# =============================================================================

@njit(parallel=True, fastmath=True)
def boris_step_3d(positions, velocities, masses, charges, total_forces, B_fields, dt):
    """
    Executes one step of the Boris algorithm in 3D space with 3D velocities (3D-3V).
    
    total_forces = F_coulomb + q * E_ext  (electric forces)
    B_fields = external magnetic field (N, 3)
    """
    n = positions.shape[0]
    dt_half = 0.5 * dt
    
    for i in prange(n):
        m = masses[i]
        q = charges[i]
        inv_m = 1.0 / m
        
        # 1. First half-step electric push
        # v_minus = v(t) + (F_electric / m) * (dt / 2)
        vx_minus = velocities[i, 0] + total_forces[i, 0] * inv_m * dt_half
        vy_minus = velocities[i, 1] + total_forces[i, 1] * inv_m * dt_half
        vz_minus = velocities[i, 2] + total_forces[i, 2] * inv_m * dt_half
        
        # 2. Magnetic rotation vector: t = (q * B / m) * (dt / 2)
        tx = (q * B_fields[i, 0] * inv_m) * dt_half
        ty = (q * B_fields[i, 1] * inv_m) * dt_half
        tz = (q * B_fields[i, 2] * inv_m) * dt_half
        
        t_mag_sq = tx * tx + ty * ty + tz * tz
        inv_denom = 2.0 / (1.0 + t_mag_sq)
        sx = tx * inv_denom
        sy = ty * inv_denom
        sz = tz * inv_denom
        
        # v_prime = v_minus + v_minus x t
        v_prime_x = vx_minus + (vy_minus * tz - vz_minus * ty)
        v_prime_y = vy_minus + (vz_minus * tx - vx_minus * tz)
        v_prime_z = vz_minus + (vx_minus * ty - vy_minus * tx)
        
        # v_plus = v_minus + v_prime x s
        vx_plus = vx_minus + (v_prime_y * sz - v_prime_z * sy)
        vy_plus = vy_minus + (v_prime_z * sx - v_prime_x * sz)
        vz_plus = vz_minus + (v_prime_x * sy - v_prime_y * sx)
        
        # 3. Second half-step electric push
        # v(t + dt) = v_plus + (F_electric / m) * (dt / 2)
        vx_new = vx_plus + total_forces[i, 0] * inv_m * dt_half
        vy_new = vy_plus + total_forces[i, 1] * inv_m * dt_half
        vz_new = vz_plus + total_forces[i, 2] * inv_m * dt_half
        
        velocities[i, 0] = vx_new
        velocities[i, 1] = vy_new
        velocities[i, 2] = vz_new
        
        # 4. Position update: r(t + dt) = r(t) + v(t + dt) * dt
        positions[i, 0] += vx_new * dt
        positions[i, 1] += vy_new * dt
        positions[i, 2] += vz_new * dt


@njit(parallel=True, fastmath=True)
def boris_step_2d(positions, velocities_3d, masses, charges, total_forces_2d, B_fields_3d, dt):
    """
    Executes Boris algorithm in 2D space with 3D velocities (2D-3V).
    Positions update in (x, y). Out-of-plane velocity vz undergoes full Lorentz rotation.
    """
    n = positions.shape[0]
    dt_half = 0.5 * dt
    
    for i in prange(n):
        m = masses[i]
        q = charges[i]
        inv_m = 1.0 / m
        
        # 1. Electric push (F_z = 0 in 2D plane)
        vx_minus = velocities_3d[i, 0] + total_forces_2d[i, 0] * inv_m * dt_half
        vy_minus = velocities_3d[i, 1] + total_forces_2d[i, 1] * inv_m * dt_half
        vz_minus = velocities_3d[i, 2]
        
        # 2. Magnetic rotation
        tx = (q * B_fields_3d[i, 0] * inv_m) * dt_half
        ty = (q * B_fields_3d[i, 1] * inv_m) * dt_half
        tz = (q * B_fields_3d[i, 2] * inv_m) * dt_half
        
        t_mag_sq = tx * tx + ty * ty + tz * tz
        inv_denom = 2.0 / (1.0 + t_mag_sq)
        sx = tx * inv_denom
        sy = ty * inv_denom
        sz = tz * inv_denom
        
        v_prime_x = vx_minus + (vy_minus * tz - vz_minus * ty)
        v_prime_y = vy_minus + (vz_minus * tx - vx_minus * tz)
        v_prime_z = vz_minus + (vx_minus * ty - vy_minus * tx)
        
        vx_plus = vx_minus + (v_prime_y * sz - v_prime_z * sy)
        vy_plus = vy_minus + (v_prime_z * sx - v_prime_x * sz)
        vz_plus = vz_minus + (v_prime_x * sy - v_prime_y * sx)
        
        # 3. Second electric push
        vx_new = vx_plus + total_forces_2d[i, 0] * inv_m * dt_half
        vy_new = vy_plus + total_forces_2d[i, 1] * inv_m * dt_half
        vz_new = vz_plus
        
        velocities_3d[i, 0] = vx_new
        velocities_3d[i, 1] = vy_new
        velocities_3d[i, 2] = vz_new
        
        # 4. Position update
        positions[i, 0] += vx_new * dt
        positions[i, 1] += vy_new * dt


# =============================================================================
# Kinematic Boundary Limiters
# =============================================================================

@njit(fastmath=True)
def apply_boundary_circular_2d(positions, velocities, boundary_radius):
    """Elastic reflective circular boundary in 2D."""
    n = positions.shape[0]
    r_max_sq = boundary_radius * boundary_radius
    
    for i in range(n):
        r_sq = positions[i, 0] * positions[i, 0] + positions[i, 1] * positions[i, 1]
        if r_sq > r_max_sq:
            r = np.sqrt(r_sq)
            nx = positions[i, 0] / r
            ny = positions[i, 1] / r
            
            positions[i, 0] = nx * boundary_radius
            positions[i, 1] = ny * boundary_radius
            
            dot_prod = velocities[i, 0] * nx + velocities[i, 1] * ny
            if dot_prod > 0.0:
                velocities[i, 0] -= 2.0 * dot_prod * nx
                velocities[i, 1] -= 2.0 * dot_prod * ny


@njit(fastmath=True)
def apply_boundary_tokamak_3d(positions, velocities, R0=100.0, a=40.0):
    """
    Toroidal vessel boundary / limiter for Tokamak in 3D.
    Reflects particles that strike the first wall r = sqrt((R - R0)^2 + z^2) = a.
    """
    n = positions.shape[0]
    a_sq = a * a
    
    for i in range(n):
        x = positions[i, 0]
        y = positions[i, 1]
        z = positions[i, 2]
        
        R = np.sqrt(x * x + y * y)
        if R < 1e-6:
            R = 1e-6
        delta_R = R - R0
        r_sq = delta_R * delta_R + z * z
        
        if r_sq > a_sq:
            r = np.sqrt(r_sq)
            # Surface normal outward vector in (x, y, z):
            # n = (delta_R/r) * (x/R, y/R, 0) + (z/r) * (0, 0, 1)
            nr = delta_R / r
            nz = z / r
            nx = nr * (x / R)
            ny = nr * (y / R)
            
            # Reposition to boundary
            target_R = R0 + nr * a
            scale_R = target_R / R
            positions[i, 0] = x * scale_R
            positions[i, 1] = y * scale_R
            positions[i, 2] = nz * a
            
            dot_prod = velocities[i, 0] * nx + velocities[i, 1] * ny + velocities[i, 2] * nz
            if dot_prod > 0.0:
                velocities[i, 0] -= 2.0 * dot_prod * nx
                velocities[i, 1] -= 2.0 * dot_prod * ny
                velocities[i, 2] -= 2.0 * dot_prod * nz


@njit(fastmath=True)
def apply_boundary_spherical_3d(positions, velocities, radius):
    """Elastic reflective spherical boundary in 3D."""
    n = positions.shape[0]
    r_sq_max = radius * radius
    
    for i in range(n):
        r_sq = positions[i, 0]**2 + positions[i, 1]**2 + positions[i, 2]**2
        if r_sq > r_sq_max:
            r = np.sqrt(r_sq)
            nx = positions[i, 0] / r
            ny = positions[i, 1] / r
            nz = positions[i, 2] / r
            
            positions[i, 0] = nx * radius
            positions[i, 1] = ny * radius
            positions[i, 2] = nz * radius
            
            dot_prod = velocities[i, 0] * nx + velocities[i, 1] * ny + velocities[i, 2] * nz
            if dot_prod > 0.0:
                velocities[i, 0] -= 2.0 * dot_prod * nx
                velocities[i, 1] -= 2.0 * dot_prod * ny
                velocities[i, 2] -= 2.0 * dot_prod * nz
