import numpy as np
from numba import njit, prange

@njit
def get_bounding_box(positions):
    """Dynamically calculates the bounding square for the current particles."""
    min_x = np.min(positions[:, 0])
    max_x = np.max(positions[:, 0])
    min_y = np.min(positions[:, 1])
    max_y = np.max(positions[:, 1])
    
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    
    width = max_x - min_x
    height = max_y - min_y
    size = max(width, height) + 1e-5 # Add small padding
    
    return cx, cy, size

@njit
def get_quadrant(x, y, cx, cy):
    """Determines which quadrant a position falls into relative to a center."""
    if x < cx:
        if y < cy: return 0  # SW
        else:      return 1  # NW
    else:
        if y < cy: return 2  # SE
        else:      return 3  # NE

@njit
def build_quadtree(positions, charges, root_cx, root_cy, root_size):
    """
    Builds the Barnes-Hut quadtree.
    Returns array-based node representations for Numba compatibility.
    """
    n_particles = len(positions)
    max_nodes = n_particles * 10 # Safe heuristic for max tree size
    
    # node_f: [center_of_charge_x, center_of_charge_y, total_charge, total_abs_charge, box_cx, box_cy, box_size]
    node_f = np.zeros((max_nodes, 7), dtype=np.float64) 
    # node_i: [child_SW, child_NW, child_SE, child_NE, particle_id]
    node_i = np.full((max_nodes, 5), -1, dtype=np.int32)
    
    # Root node
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
            # Update center of absolute charge and total charge for the current node
            old_abs_q = node_f[curr_node, 3]
            new_abs_q = old_abs_q + abs_q
            
            if new_abs_q > 0:
                node_f[curr_node, 0] = (node_f[curr_node, 0] * old_abs_q + px * abs_q) / new_abs_q
                node_f[curr_node, 1] = (node_f[curr_node, 1] * old_abs_q + py * abs_q) / new_abs_q
                
            node_f[curr_node, 2] += q
            node_f[curr_node, 3] = new_abs_q
            
            # Check if current node is a leaf
            p_idx = node_i[curr_node, 4]
            is_leaf = True
            for c in range(4):
                if node_i[curr_node, c] != -1:
                    is_leaf = False
                    break
                    
            if is_leaf:
                if p_idx == -1:
                    # Empty leaf -> insert particle
                    node_i[curr_node, 4] = i
                    break
                else:
                    # Node has a particle -> Subdivide
                    # Prevent infinite recursion for overlapping particles
                    if node_f[curr_node, 6] < 1e-6:
                        break
                        
                    node_i[curr_node, 4] = -1 # Turn into internal node
                    
                    # Re-insert the old particle into the new children
                    old_px = positions[p_idx, 0]
                    old_py = positions[p_idx, 1]
                    old_quad = get_quadrant(old_px, old_py, node_f[curr_node, 4], node_f[curr_node, 5])
                    
                    child_node = node_count
                    if child_node >= max_nodes:
                        break # Prevent out-of-bounds array access safety guard
                    node_count += 1
                    node_i[curr_node, old_quad] = child_node
                    
                    # Setup child bounding box
                    child_size = node_f[curr_node, 6] / 2.0
                    offset = child_size / 2.0
                    cx = node_f[curr_node, 4]
                    cy = node_f[curr_node, 5]
                    
                    if old_quad == 0: # SW
                        node_f[child_node, 4] = cx - offset; node_f[child_node, 5] = cy - offset
                    elif old_quad == 1: # NW
                        node_f[child_node, 4] = cx - offset; node_f[child_node, 5] = cy + offset
                    elif old_quad == 2: # SE
                        node_f[child_node, 4] = cx + offset; node_f[child_node, 5] = cy - offset
                    elif old_quad == 3: # NE
                        node_f[child_node, 4] = cx + offset; node_f[child_node, 5] = cy + offset
                        
                    node_f[child_node, 6] = child_size
                    node_f[child_node, 0] = old_px
                    node_f[child_node, 1] = old_py
                    node_f[child_node, 2] = charges[p_idx]
                    node_f[child_node, 3] = abs(charges[p_idx])
                    node_i[child_node, 4] = p_idx
                    
                    is_leaf = False # Continue inserting the NEW particle into this now-internal node

            if not is_leaf:
                # Move down to the correct child
                quad = get_quadrant(px, py, node_f[curr_node, 4], node_f[curr_node, 5])
                next_node = node_i[curr_node, quad]
                
                if next_node == -1:
                    # Create new child and place particle there
                    child_node = node_count
                    if child_node >= max_nodes:
                        break # Prevent out-of-bounds array access safety guard
                    node_count += 1
                    node_i[curr_node, quad] = child_node
                    
                    child_size = node_f[curr_node, 6] / 2.0
                    offset = child_size / 2.0
                    cx = node_f[curr_node, 4]
                    cy = node_f[curr_node, 5]
                    
                    if quad == 0: # SW
                        node_f[child_node, 4] = cx - offset; node_f[child_node, 5] = cy - offset
                    elif quad == 1: # NW
                        node_f[child_node, 4] = cx - offset; node_f[child_node, 5] = cy + offset
                    elif quad == 2: # SE
                        node_f[child_node, 4] = cx + offset; node_f[child_node, 5] = cy - offset
                    elif quad == 3: # NE
                        node_f[child_node, 4] = cx + offset; node_f[child_node, 5] = cy + offset
                        
                    node_f[child_node, 6] = child_size
                    node_f[child_node, 0] = px
                    node_f[child_node, 1] = py
                    node_f[child_node, 2] = q
                    node_f[child_node, 3] = abs_q
                    node_i[child_node, 4] = i
                    break # Successfully inserted
                else:
                    curr_node = next_node
                    
    return node_f, node_i, node_count

@njit(parallel=True)
def calculate_forces(positions, charges, node_f, node_i, k_e, softening, theta):
    """Calculates Coulomb forces and Potential Energy using the Barnes-Hut quadtree."""
    n_particles = len(positions)
    forces = np.zeros_like(positions)
    potential_energy = np.zeros(n_particles)
    
    for i in prange(n_particles):
        px = positions[i, 0]
        py = positions[i, 1]
        q = charges[i]
        
        stack = np.zeros(100, dtype=np.int32)
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
                continue # Skip self interaction
                
            node_cx = node_f[node_idx, 0]
            node_cy = node_f[node_idx, 1]
            node_q = node_f[node_idx, 2]
            node_size = node_f[node_idx, 6]
            
            dx = px - node_cx
            dy = py - node_cy
            dist_sq = dx**2 + dy**2 + softening**2
            dist = np.sqrt(dist_sq)
            
            # Check if leaf
            is_leaf = (node_i[node_idx, 0] == -1 and node_i[node_idx, 1] == -1 and 
                       node_i[node_idx, 2] == -1 and node_i[node_idx, 3] == -1)
                       
            if is_leaf or (node_size / dist < theta):
                if node_q != 0.0:
                    f_mag = k_e * q * node_q / dist_sq
                    fx += f_mag * (dx / dist)
                    fy += f_mag * (dy / dist)
                    pe += k_e * q * node_q / dist
            else:
                for c in range(4):
                    child_idx = node_i[node_idx, c]
                    if child_idx != -1:
                        stack[stack_ptr] = child_idx
                        stack_ptr += 1
                        
        forces[i, 0] = fx
        forces[i, 1] = fy
        potential_energy[i] = pe * 0.5 # Halved to avoid double counting across all pairs
        
    return forces, np.sum(potential_energy)

@njit
def apply_boundary_conditions(positions, velocities, boundary_radius):
    """Applies a circular reflective kinematic boundary."""
    n_particles = len(positions)
    for i in range(n_particles):
        r_sq = positions[i, 0]**2 + positions[i, 1]**2
        if r_sq > boundary_radius**2:
            r = np.sqrt(r_sq)
            # Normal vector
            nx = positions[i, 0] / r
            ny = positions[i, 1] / r
            
            # Clamp position to the boundary
            positions[i, 0] = nx * boundary_radius
            positions[i, 1] = ny * boundary_radius
            
            # Reflect velocity (Elastic collision: v = v - 2(v.n)n )
            dot_product = velocities[i, 0] * nx + velocities[i, 1] * ny
            if dot_product > 0: # Only reflect if moving outwards
                velocities[i, 0] -= 2.0 * dot_product * nx
                velocities[i, 1] -= 2.0 * dot_product * ny

class PhysicsEngine:
    """Object-Oriented interface for the Physics engine."""
    
    def __init__(self, n_particles=1000, dt=0.001, theta=0.25, softening=1.0, boundary_radius=200.0, k_e=1000.0):
        self.n_particles = n_particles
        self.dt = dt
        self.theta = theta
        self.softening = softening
        self.boundary_radius = boundary_radius
        self.k_e = k_e
        
        self.masses = np.ones(n_particles, dtype=np.float64)
        
        # Initialize positions randomly inside the boundary
        radii = np.random.uniform(0, boundary_radius * 0.8, n_particles)
        angles = np.random.uniform(0, 2 * np.pi, n_particles)
        self.positions = np.zeros((n_particles, 2), dtype=np.float64)
        self.positions[:, 0] = radii * np.cos(angles)
        self.positions[:, 1] = radii * np.sin(angles)
        
        self.velocities = np.zeros((n_particles, 2), dtype=np.float64)
        
        # 50% positive, 50% negative charges
        self.charges = np.ones(n_particles, dtype=np.float64)
        self.charges[n_particles // 2:] = -1.0
        
        self.forces = np.zeros_like(self.positions)
        self.kinetic_energy = 0.0
        self.potential_energy = 0.0
        
        # Compute initial forces and potentials
        self._update_forces()
        
    def _update_forces(self):
        cx, cy, size = get_bounding_box(self.positions)
        node_f, node_i, _ = build_quadtree(self.positions, self.charges, cx, cy, size)
        self.forces, self.potential_energy = calculate_forces(
            self.positions, self.charges, node_f, node_i, self.k_e, self.softening, self.theta
        )
        
    def step(self):
        """Advances the simulation by one dt using Symplectic Velocity Verlet integration."""
        # 1. v(t + dt/2) = v(t) + F(t)/m * dt/2
        self.velocities += (self.forces / self.masses[:, None]) * (self.dt / 2.0)
        
        # 2. r(t + dt) = r(t) + v(t + dt/2) * dt
        self.positions += self.velocities * self.dt
        
        # Ensure particles stay within bounds
        apply_boundary_conditions(self.positions, self.velocities, self.boundary_radius)
        
        # 3. Calculate new forces F(t + dt)
        self._update_forces()
        
        # 4. v(t + dt) = v(t + dt/2) + F(t + dt)/m * dt/2
        self.velocities += (self.forces / self.masses[:, None]) * (self.dt / 2.0)
        
        # Update metrics
        speed_sq = self.velocities[:, 0]**2 + self.velocities[:, 1]**2
        self.kinetic_energy = np.sum(0.5 * self.masses * speed_sq)
        
    def get_total_energy(self):
        return self.kinetic_energy + self.potential_energy
