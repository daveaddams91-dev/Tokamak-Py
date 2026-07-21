# Tokamak-Py: Electrostatic Plasma Confinement Engine

Tokamak-Py is a high-performance N-body computational physics engine for simulating electrostatic plasma confinement. Instead of relying on brute-force O(n^2) calculations or standard Euler integration, it is built for mathematical rigor and real-time visualization, modeling the chaotic kinematics of charged particles confined within a kinematic boundary using JIT-compiled, C-level operations in Python.

## Core Architecture

**1. Algorithmic Optimization**
Uses a dynamically generated Barnes-Hut Quadtree to partition space and group distant charges, reducing computational complexity from O(n^2) to O(n log n).

**2. Symplectic Integration**
Employs Velocity Verlet integration for time stepping. Unlike standard integrators, this conserves the total energy of the closed system over time, preventing orbital decay or artificial energy injection.

**3. JIT Compilation**
The entire mathematical engine (tree construction, force calculation, and boundary logic) is vectorized with NumPy and compiled to machine code via Numba, enabling real-time simulation of thousands of particles without Python's GIL bottleneck.

**4. Real-Time Analytics**
Built on PyQt6 and PyQtGraph, the dashboard bypasses standard Matplotlib rendering limits to deliver 60-FPS telemetry, including:

- A spatial sandbox tracking positive and negative charges
- Energy conservation monitoring (kinetic, potential, and total energy)
- Phase-space plotting to observe chaotic attractors of tracer particles

## Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/tokamak-py.git
cd tokamak-py
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install the required dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the simulation**

```bash
python main.py
```

## Usage

On launch, the application immediately begins simulating 1,000 charged particles.

- **The Sandbox** — Watch emergent behavior as positively charged (red) and negatively charged (blue) particles cluster, repel, and interact with the reflective boundary.
- **The Analytics** — Monitor the "Total Energy" line in the upper right; a flat line indicates stable, accurate integration. The bottom-right panel shows the phase space of a single tracer particle, revealing the underlying chaotic structure of the simulation.

## Tuning the Engine

Core physics parameters can be modified in `main.py` where `PhysicsEngine` is initialized:

```python
self.engine = PhysicsEngine(
    n_particles=1000,      # Total number of particles
    dt=0.001,              # Integration time step
    theta=0.25,            # Barnes-Hut threshold (lower = more accurate, slower)
    softening=1.0,         # Prevents infinite forces on collision
    boundary_radius=200.0, # Size of the confinement field
    k_e=1000.0             # Coulomb's constant multiplier
)
```

## Future Roadmap

- Migration from 2D quadtrees to 3D octrees for volumetric simulation
- Implementation of magnetic fields (Lorentz force interactions)
- GPU acceleration via CUDA (Numba `@cuda.jit`)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
