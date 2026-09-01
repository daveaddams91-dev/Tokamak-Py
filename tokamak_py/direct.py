"""
Tokamak-Py: Exact Direct Pairwise N-Body Force Calculation.

Implements vectorized, parallel O(N^2) Coulomb force and potential energy
summation using Numba JIT. Serves as ground-truth benchmark and high-precision
solver for moderate particle numbers.
"""

import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def calculate_direct_forces_2d(positions, charges, k_e=1000.0, softening=1.0):
    """
    Computes exact pairwise Coulomb forces and potential energy in 2D.
    
    F_i = sum_{j != i} k_e * q_i * q_j * (r_i - r_j) / (||r_i - r_j||^2 + eps^2)^(3/2)
    PE = 0.5 * sum_{i} sum_{j != i} k_e * q_i * q_j / sqrt(||r_i - r_j||^2 + eps^2)
    """
    n = positions.shape[0]
    forces = np.zeros_like(positions)
    pe_arr = np.zeros(n, dtype=np.float64)
    eps_sq = softening * softening
    
    for i in prange(n):
        xi = positions[i, 0]
        yi = positions[i, 1]
        qi = charges[i]
        
        fx = 0.0
        fy = 0.0
        pe = 0.0
        
        for j in range(n):
            if i == j:
                continue
            dx = xi - positions[j, 0]
            dy = yi - positions[j, 1]
            qj = charges[j]
            
            dist_sq = dx * dx + dy * dy + eps_sq
            dist = np.sqrt(dist_sq)
            inv_dist3 = 1.0 / (dist_sq * dist)
            
            f_factor = k_e * qi * qj * inv_dist3
            fx += f_factor * dx
            fy += f_factor * dy
            pe += (k_e * qi * qj) / dist
            
        forces[i, 0] = fx
        forces[i, 1] = fy
        pe_arr[i] = pe * 0.5
        
    return forces, np.sum(pe_arr)


@njit(parallel=True, fastmath=True)
def calculate_direct_forces_3d(positions, charges, k_e=1000.0, softening=1.0):
    """
    Computes exact pairwise Coulomb forces and potential energy in 3D.
    """
    n = positions.shape[0]
    forces = np.zeros_like(positions)
    pe_arr = np.zeros(n, dtype=np.float64)
    eps_sq = softening * softening
    
    for i in prange(n):
        xi = positions[i, 0]
        yi = positions[i, 1]
        zi = positions[i, 2]
        qi = charges[i]
        
        fx = 0.0
        fy = 0.0
        fz = 0.0
        pe = 0.0
        
        for j in range(n):
            if i == j:
                continue
            dx = xi - positions[j, 0]
            dy = yi - positions[j, 1]
            dz = zi - positions[j, 2]
            qj = charges[j]
            
            dist_sq = dx * dx + dy * dy + dz * dz + eps_sq
            dist = np.sqrt(dist_sq)
            inv_dist3 = 1.0 / (dist_sq * dist)
            
            f_factor = k_e * qi * qj * inv_dist3
            fx += f_factor * dx
            fy += f_factor * dy
            fz += f_factor * dz
            pe += (k_e * qi * qj) / dist
            
        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz
        pe_arr[i] = pe * 0.5
        
    return forces, np.sum(pe_arr)
