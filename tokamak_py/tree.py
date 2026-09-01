"""
Tokamak-Py: Barnes-Hut Spatial Tree Engines (2D Quadtree & 3D Octree).

Reduces N-body Coulomb interaction complexity from O(N^2) to O(N log N)
using dynamically generated spatial decomposition trees vectorized and compiled
via Numba JIT. Includes robust stack bounds and safety checks.
"""

import numpy as np
from numba import njit, prange

# =============================================================================
# 2D Quadtree Engine
# =============================================================================

@njit(fastmath=True)
def get_bounding_box_2d(positions):
    """Calculates the bounding square for 2D particles."""
    min_x = np.min(positions[:, 0])
    max_x = np.max(positions[:, 0])
    min_y = np.min(positions[:, 1])
    max_y = np.max(positions[:, 1])
    
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    
    width = max_x - min_x
    height = max_y - min_y
    size = max(width, height) + 1e-4
    
    return cx, cy, size

@njit(fastmath=True)
def get_quadrant(x, y, cx, cy):
    """Returns quadrant index (0: SW, 1: NW, 2: SE, 3: NE)."""
    if x < cx:
        return 0 if y < cy else 1
    else:
        return 2 if y < cy else 3

@njit
def build_quadtree(positions, charges, root_cx, root_cy, root_size):
    """
    Builds array-based Barnes-Hut Quadtree for 2D charges.
    
    node_f: [center_charge_x, center_charge_y, total_charge, total_abs_charge, box_cx, box_cy, box_size]
    node_i: [child_SW, child_NW, child_SE, child_NE, particle_id]
    """
    n_particles = len(positions)
    max_nodes = max(n_particles * 12, 1024)
    
    node_f = np.zeros((max_nodes, 7), dtype=np.float64)
    node_i = np.full((max_nodes, 5), -1, dtype=np.int32)
    
    node_f[0, 4] = root_cx
    node_f[0, 5] = root_cy
    node_f[0, 6] = root_size
    node_count = 1
    
    for i in range(n_particles):
        px = positions[i, 0]
        py = positions[i, 1]
        q = charges[i]
        abs_q = abs(q)
        
        curr_node = 0
        while True:
            old_abs_q = node_f[curr_node, 3]
            new_abs_q = old_abs_q + abs_q
            
            if new_abs_q > 0.0:
                node_f[curr_node, 0] = (node_f[curr_node, 0] * old_abs_q + px * abs_q) / new_abs_q
                node_f[curr_node, 1] = (node_f[curr_node, 1] * old_abs_q + py * abs_q) / new_abs_q
                
            node_f[curr_node, 2] += q
            node_f[curr_node, 3] = new_abs_q
            
            # Check if leaf
            p_idx = node_i[curr_node, 4]
            is_leaf = (node_i[curr_node, 0] == -1 and node_i[curr_node, 1] == -1 and
                       node_i[curr_node, 2] == -1 and node_i[curr_node, 3] == -1)
                       
            if is_leaf:
                if p_idx == -1:
                    # Empty leaf -> insert particle
                    node_i[curr_node, 4] = i
                    break
                else:
                    # Node already has a particle -> subdivide
                    if node_f[curr_node, 6] < 1e-6:
                        break
                    node_i[curr_node, 4] = -1
                    
                    old_px = positions[p_idx, 0]
                    old_py = positions[p_idx, 1]
                    old_quad = get_quadrant(old_px, old_py, node_f[curr_node, 4], node_f[curr_node, 5])
                    
                    child_node = node_count
                    if child_node >= max_nodes:
                        break
                    node_count += 1
                    node_i[curr_node, old_quad] = child_node
                    
                    child_size = node_f[curr_node, 6] * 0.5
                    offset = child_size * 0.5
                    cx = node_f[curr_node, 4]
                    cy = node_f[curr_node, 5]
                    
                    if old_quad == 0:
                        node_f[child_node, 4] = cx - offset; node_f[child_node, 5] = cy - offset
                    elif old_quad == 1:
                        node_f[child_node, 4] = cx - offset; node_f[child_node, 5] = cy + offset
                    elif old_quad == 2:
                        node_f[child_node, 4] = cx + offset; node_f[child_node, 5] = cy - offset
                    else:
                        node_f[child_node, 4] = cx + offset; node_f[child_node, 5] = cy + offset
                        
                    node_f[child_node, 6] = child_size
                    node_f[child_node, 0] = old_px
                    node_f[child_node, 1] = old_py
                    node_f[child_node, 2] = charges[p_idx]
                    node_f[child_node, 3] = abs(charges[p_idx])
                    node_i[child_node, 4] = p_idx
                    is_leaf = False
                    
            if not is_leaf:
                quad = get_quadrant(px, py, node_f[curr_node, 4], node_f[curr_node, 5])
                next_node = node_i[curr_node, quad]
                
                if next_node == -1:
                    child_node = node_count
                    if child_node >= max_nodes:
                        break
                    node_count += 1
                    node_i[curr_node, quad] = child_node
                    
                    child_size = node_f[curr_node, 6] * 0.5
                    offset = child_size * 0.5
                    cx = node_f[curr_node, 4]
                    cy = node_f[curr_node, 5]
                    
                    if quad == 0:
                        node_f[child_node, 4] = cx - offset; node_f[child_node, 5] = cy - offset
                    elif quad == 1:
                        node_f[child_node, 4] = cx - offset; node_f[child_node, 5] = cy + offset
                    elif quad == 2:
                        node_f[child_node, 4] = cx + offset; node_f[child_node, 5] = cy - offset
                    else:
                        node_f[child_node, 4] = cx + offset; node_f[child_node, 5] = cy + offset
                        
                    node_f[child_node, 6] = child_size
                    node_f[child_node, 0] = px
                    node_f[child_node, 1] = py
                    node_f[child_node, 2] = q
                    node_f[child_node, 3] = abs_q
                    node_i[child_node, 4] = i
                    break
                else:
                    curr_node = next_node
                    
    return node_f, node_i, node_count

@njit(parallel=True, fastmath=True)
def calculate_forces_2d(positions, charges, node_f, node_i, k_e=1000.0, softening=1.0, theta=0.25):
    """Calculates Coulomb forces and Potential Energy in 2D using Barnes-Hut quadtree."""
    n_particles = len(positions)
    forces = np.zeros_like(positions)
    potential_energy = np.zeros(n_particles, dtype=np.float64)
    eps_sq = softening * softening
    
    for i in prange(n_particles):
        px = positions[i, 0]
        py = positions[i, 1]
        q = charges[i]
        
        stack = np.zeros(256, dtype=np.int32)
        stack[0] = 0
        stack_ptr = 1
        
        fx = 0.0
        fy = 0.0
        pe = 0.0
        
        while stack_ptr > 0:
            stack_ptr -= 1
            node_idx = stack[stack_ptr]
            
            p_idx = node_i[node_idx, 4]
            if p_idx == i:
                continue
                
            node_cx = node_f[node_idx, 0]
            node_cy = node_f[node_idx, 1]
            node_q = node_f[node_idx, 2]
            node_size = node_f[node_idx, 6]
            
            dx = px - node_cx
            dy = py - node_cy
            dist_sq = dx * dx + dy * dy + eps_sq
            dist = np.sqrt(dist_sq)
            
            is_leaf = (node_i[node_idx, 0] == -1 and node_i[node_idx, 1] == -1 and 
                       node_i[node_idx, 2] == -1 and node_i[node_idx, 3] == -1)
                       
            if is_leaf or (node_size / dist < theta):
                if node_q != 0.0:
                    inv_dist3 = 1.0 / (dist_sq * dist)
                    f_factor = k_e * q * node_q * inv_dist3
                    fx += f_factor * dx
                    fy += f_factor * dy
                    pe += (k_e * q * node_q) / dist
            else:
                for c in range(4):
                    child_idx = node_i[node_idx, c]
                    if child_idx != -1 and stack_ptr < 250:
                        stack[stack_ptr] = child_idx
                        stack_ptr += 1
                        
        forces[i, 0] = fx
        forces[i, 1] = fy
        potential_energy[i] = pe * 0.5
        
    return forces, np.sum(potential_energy)


# =============================================================================
# 3D Octree Engine
# =============================================================================

@njit(fastmath=True)
def get_bounding_box_3d(positions):
    """Calculates the bounding cube for 3D particles."""
    min_x = np.min(positions[:, 0]); max_x = np.max(positions[:, 0])
    min_y = np.min(positions[:, 1]); max_y = np.max(positions[:, 1])
    min_z = np.min(positions[:, 2]); max_z = np.max(positions[:, 2])
    
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    cz = (min_z + max_z) * 0.5
    
    size = max(max_x - min_x, max(max_y - min_y, max_z - min_z)) + 1e-4
    return cx, cy, cz, size

@njit(fastmath=True)
def get_octant(x, y, z, cx, cy, cz):
    """Determines octant (0 to 7) for 3D Barnes-Hut octree."""
    idx = 0
    if x >= cx: idx |= 1
    if y >= cy: idx |= 2
    if z >= cz: idx |= 4
    return idx

@njit
def build_octree(positions, charges, root_cx, root_cy, root_cz, root_size):
    """
    Builds array-based Barnes-Hut Octree for 3D charges.
    
    node_f: [center_charge_x, center_charge_y, center_charge_z, total_charge, total_abs_charge, box_cx, box_cy, box_cz, box_size] (shape: M x 9)
    node_i: [child_0..7, particle_id] (shape: M x 9)
    """
    n_particles = len(positions)
    max_nodes = max(n_particles * 16, 2048)
    
    node_f = np.zeros((max_nodes, 9), dtype=np.float64)
    node_i = np.full((max_nodes, 9), -1, dtype=np.int32)
    
    node_f[0, 5] = root_cx
    node_f[0, 6] = root_cy
    node_f[0, 7] = root_cz
    node_f[0, 8] = root_size
    node_count = 1
    
    for i in range(n_particles):
        px = positions[i, 0]
        py = positions[i, 1]
        pz = positions[i, 2]
        q = charges[i]
        abs_q = abs(q)
        
        curr_node = 0
        while True:
            old_abs_q = node_f[curr_node, 4]
            new_abs_q = old_abs_q + abs_q
            
            if new_abs_q > 0.0:
                node_f[curr_node, 0] = (node_f[curr_node, 0] * old_abs_q + px * abs_q) / new_abs_q
                node_f[curr_node, 1] = (node_f[curr_node, 1] * old_abs_q + py * abs_q) / new_abs_q
                node_f[curr_node, 2] = (node_f[curr_node, 2] * old_abs_q + pz * abs_q) / new_abs_q
                
            node_f[curr_node, 3] += q
            node_f[curr_node, 4] = new_abs_q
            
            p_idx = node_i[curr_node, 8]
            is_leaf = True
            for c in range(8):
                if node_i[curr_node, c] != -1:
                    is_leaf = False
                    break
                    
            if is_leaf:
                if p_idx == -1:
                    node_i[curr_node, 8] = i
                    break
                else:
                    if node_f[curr_node, 8] < 1e-6:
                        break
                    node_i[curr_node, 8] = -1
                    
                    old_px = positions[p_idx, 0]
                    old_py = positions[p_idx, 1]
                    old_pz = positions[p_idx, 2]
                    old_oct = get_octant(old_px, old_py, old_pz, node_f[curr_node, 5], node_f[curr_node, 6], node_f[curr_node, 7])
                    
                    child_node = node_count
                    if child_node >= max_nodes:
                        break
                    node_count += 1
                    node_i[curr_node, old_oct] = child_node
                    
                    child_size = node_f[curr_node, 8] * 0.5
                    offset = child_size * 0.5
                    cx = node_f[curr_node, 5]
                    cy = node_f[curr_node, 6]
                    cz = node_f[curr_node, 7]
                    
                    node_f[child_node, 5] = cx + (offset if (old_oct & 1) else -offset)
                    node_f[child_node, 6] = cy + (offset if (old_oct & 2) else -offset)
                    node_f[child_node, 7] = cz + (offset if (old_oct & 4) else -offset)
                    node_f[child_node, 8] = child_size
                    node_f[child_node, 0] = old_px
                    node_f[child_node, 1] = old_py
                    node_f[child_node, 2] = old_pz
                    node_f[child_node, 3] = charges[p_idx]
                    node_f[child_node, 4] = abs(charges[p_idx])
                    node_i[child_node, 8] = p_idx
                    is_leaf = False
                    
            if not is_leaf:
                oct_idx = get_octant(px, py, pz, node_f[curr_node, 5], node_f[curr_node, 6], node_f[curr_node, 7])
                next_node = node_i[curr_node, oct_idx]
                
                if next_node == -1:
                    child_node = node_count
                    if child_node >= max_nodes:
                        break
                    node_count += 1
                    node_i[curr_node, oct_idx] = child_node
                    
                    child_size = node_f[curr_node, 8] * 0.5
                    offset = child_size * 0.5
                    cx = node_f[curr_node, 5]
                    cy = node_f[curr_node, 6]
                    cz = node_f[curr_node, 7]
                    
                    node_f[child_node, 5] = cx + (offset if (oct_idx & 1) else -offset)
                    node_f[child_node, 6] = cy + (offset if (oct_idx & 2) else -offset)
                    node_f[child_node, 7] = cz + (offset if (oct_idx & 4) else -offset)
                    node_f[child_node, 8] = child_size
                    node_f[child_node, 0] = px
                    node_f[child_node, 1] = py
                    node_f[child_node, 2] = pz
                    node_f[child_node, 3] = q
                    node_f[child_node, 4] = abs_q
                    node_i[child_node, 8] = i
                    break
                else:
                    curr_node = next_node
                    
    return node_f, node_i, node_count

@njit(parallel=True, fastmath=True)
def calculate_forces_3d(positions, charges, node_f, node_i, k_e=1000.0, softening=1.0, theta=0.25):
    """Calculates Coulomb forces and Potential Energy in 3D using Barnes-Hut octree."""
    n_particles = len(positions)
    forces = np.zeros_like(positions)
    potential_energy = np.zeros(n_particles, dtype=np.float64)
    eps_sq = softening * softening
    
    for i in prange(n_particles):
        px = positions[i, 0]
        py = positions[i, 1]
        pz = positions[i, 2]
        q = charges[i]
        
        stack = np.zeros(512, dtype=np.int32)
        stack[0] = 0
        stack_ptr = 1
        
        fx = 0.0
        fy = 0.0
        fz = 0.0
        pe = 0.0
        
        while stack_ptr > 0:
            stack_ptr -= 1
            node_idx = stack[stack_ptr]
            
            p_idx = node_i[node_idx, 8]
            if p_idx == i:
                continue
                
            node_cx = node_f[node_idx, 0]
            node_cy = node_f[node_idx, 1]
            node_cz = node_f[node_idx, 2]
            node_q = node_f[node_idx, 3]
            node_size = node_f[node_idx, 8]
            
            dx = px - node_cx
            dy = py - node_cy
            dz = pz - node_cz
            dist_sq = dx * dx + dy * dy + dz * dz + eps_sq
            dist = np.sqrt(dist_sq)
            
            is_leaf = True
            for c in range(8):
                if node_i[node_idx, c] != -1:
                    is_leaf = False
                    break
                    
            if is_leaf or (node_size / dist < theta):
                if node_q != 0.0:
                    inv_dist3 = 1.0 / (dist_sq * dist)
                    f_factor = k_e * q * node_q * inv_dist3
                    fx += f_factor * dx
                    fy += f_factor * dy
                    fz += f_factor * dz
                    pe += (k_e * q * node_q) / dist
            else:
                for c in range(8):
                    child_idx = node_i[node_idx, c]
                    if child_idx != -1 and stack_ptr < 500:
                        stack[stack_ptr] = child_idx
                        stack_ptr += 1
                        
        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz
        potential_energy[i] = pe * 0.5
        
    return forces, np.sum(potential_energy)
