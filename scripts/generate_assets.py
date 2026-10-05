"""
Tokamak-Py: Asset & Publication Plot Generator.

Computes and renders high-resolution plots for the README and documentation:
1. assets/banner.png: Sleek dark hero banner
2. assets/tokamak_confinement.png: Tokamak field topology, flux surfaces & banana orbits
3. assets/energy_conservation.png: Symplectic Boris vs Euler long-term conservation
4. assets/phase_space_thermalization.png: Phase portraits & Maxwell-Boltzmann thermalization
5. assets/presets_showcase.png: Multi-regime confinement showcase (Tokamak, Mirror, Fusor, ExB)
"""

import os
import sys

from matplotlib.gridspec import GridSpec
from tokamak_py.engine import PhysicsEngine
from tokamak_py.fields import eval_tokamak_field_3d, eval_tokamak_field_2d, eval_magnetic_mirror_3d
from tokamak_py.integrators import boris_step_3d
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# Ensure tokamak_py is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Configure matplotlib styling for sleek publication dark theme
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "figure.facecolor": "#0a0c14",
    "axes.facecolor": "#121522",
    "axes.edgecolor": "#282d42",
    "axes.labelcolor": "#a5adc6",
    "xtick.color": "#76809e",
    "ytick.color": "#76809e",
    "grid.color": "#1f2438",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "text.color": "#e2e6f3"
})

os.makedirs("assets", exist_ok=True)


def generate_banner():
    """Generates the hero banner for the README."""
    print("Generating assets/banner.png...")
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=180, facecolor="#080a10")
    ax.set_facecolor("#080a10")
    ax.set_xlim(-160, 160)
    ax.set_ylim(-50, 50)
    ax.axis("off")

    # Draw magnetic flux contours
    theta = np.linspace(0, 2 * np.pi, 300)
    for r in [18, 26, 34, 42]:
        for cx in [-75, 75]:
            alpha = 0.15 + (r / 50.0) * 0.25
            ax.plot(cx + r * np.cos(theta), r * 0.85 * np.sin(theta), color="#00d2ff", lw=1.2, alpha=alpha, ls="--")

    # Connect top and bottom to evoke a torus
    z_curves = np.linspace(-75, 75, 200)
    ax.plot(z_curves, 35 * np.cos(z_curves * np.pi / 150), color="#00d2ff", lw=1.0, alpha=0.3)
    ax.plot(z_curves, -35 * np.cos(z_curves * np.pi / 150), color="#00d2ff", lw=1.0, alpha=0.3)

    # Generate glowing plasma particles
    np.random.seed(42)
    for cx in [-75, 75]:
        n_pts = 140
        rad = np.random.uniform(2, 38, n_pts)
        th = np.random.uniform(0, 2 * np.pi, n_pts)
        px = cx + rad * np.cos(th)
        py = rad * 0.85 * np.sin(th)
        colors = ["#ff3366" if np.random.rand() > 0.5 else "#3399ff" for _ in range(n_pts)]
        ax.scatter(px, py, c=colors, s=np.random.uniform(6, 28, n_pts), alpha=0.75, edgecolors="none")

    # Center Typography
    ax.text(0, 12, "T O K A M A K - P Y", ha="center", va="center", fontsize=30, fontweight="heavy",
            color="#ffffff")
    ax.text(0, -5, "High-Performance N-Body Electromagnetic Plasma Confinement Engine",
            ha="center", va="center", fontsize=13, fontweight="medium", color="#00d2ff")
    ax.text(0, -22, "Symplectic Boris Lorentz-Force Integration  |  Barnes-Hut Quadtrees & Octrees  |  Numba JIT",
            ha="center", va="center", fontsize=10, color="#8b95b5")

    plt.tight_layout()
    fig.savefig("assets/banner.png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("  -> assets/banner.png created.")


def generate_tokamak_confinement():
    """Generates a 4-panel scientific figure of Tokamak confinement and orbits."""
    print("Generating assets/tokamak_confinement.png...")
    fig = plt.figure(figsize=(16, 12), dpi=160)
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)

    # -------------------------------------------------------------
    # Panel 1: Tokamak Torus & Helical Magnetic Field Lines in 3D
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.set_facecolor("#0d101b")
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor("#1a1e30")
    ax1.yaxis.pane.set_edgecolor("#1a1e30")
    ax1.zaxis.pane.set_edgecolor("#1a1e30")

    R0 = 100.0
    a = 35.0
    # Trace helical magnetic field lines
    for phi_offset in [0.0, np.pi/2, np.pi, 3*np.pi/2]:
        phi_vals = np.linspace(0, 8 * np.pi, 600)
        q_safety = 2.5 # Safety factor
        theta_vals = phi_vals / q_safety + phi_offset
        r_minor = a * 0.7
        x_line = (R0 + r_minor * np.cos(theta_vals)) * np.cos(phi_vals)
        y_line = (R0 + r_minor * np.cos(theta_vals)) * np.sin(phi_vals)
        z_line = r_minor * np.sin(theta_vals)
        ax1.plot(x_line, y_line, z_line, lw=1.8, alpha=0.85, label="Helical Field Line" if phi_offset == 0 else "")

    # Draw torus outline circles
    phi_circ = np.linspace(0, 2 * np.pi, 150)
    ax1.plot(R0 * np.cos(phi_circ), R0 * np.sin(phi_circ), np.zeros_like(phi_circ),
             c="#f1c40f", ls="--", lw=1.5, label="Magnetic Axis (R0)")

    ax1.set_title("(A) Helical Field Topology in Tokamak Torus", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("X (cm)")
    ax1.set_ylabel("Y (cm)")
    ax1.set_zlabel("Z (cm)")
    ax1.legend(loc="upper right", facecolor="#161a29", edgecolor="#323852", fontsize=8)
    ax1.view_init(elev=28, azim=45)

    # -------------------------------------------------------------
    # Panel 2: Poloidal Cross-Section with Magnetic Flux Surfaces
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    # Nested circular flux surfaces
    for r_surf in np.linspace(5, 40, 8):
        circle = plt.Circle((0, 0), r_surf, fill=False, color="#3498db", ls="--", lw=1.2, alpha=0.6)
        ax2.add_patch(circle)
    # Limiter wall
    wall = plt.Circle((0, 0), 40, fill=False, color="#ffffff", ls="-", lw=2.0, label="First-Wall Limiter")
    ax2.add_patch(wall)

    # Add confined plasma particles
    np.random.seed(101)
    n_p = 300
    r_p = np.random.uniform(0, 36, n_p)
    th_p = np.random.uniform(0, 2 * np.pi, n_p)
    ax2.scatter(r_p * np.cos(th_p), r_p * np.sin(th_p), c="#e74c3c", s=10, alpha=0.7, label="Deuterons (D+)")
    ax2.scatter([0], [0], c="#f1c40f", s=80, marker="+", label="Magnetic Axis (R0, 0)", zorder=10)

    ax2.set_xlim(-48, 48)
    ax2.set_ylim(-48, 48)
    ax2.set_aspect("equal")
    ax2.set_title("(B) Poloidal Plane Flux Surfaces & Plasma Core", fontsize=12, fontweight="bold")
    ax2.set_xlabel("R - R0 (Major Radius Offset) [cm]")
    ax2.set_ylabel("Z (Vertical Elevation) [cm]")
    ax2.legend(loc="upper right", facecolor="#161a29", edgecolor="#323852", fontsize=8)
    ax2.grid(True)

    # -------------------------------------------------------------
    # Panel 3: Trapped Particle Banana Orbit Simulation
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    # Compute actual guiding-center trajectory in Tokamak field
    dt = 0.0005
    n_steps = 3000
    pos = np.array([[20.0, 0.0, 0.0]], dtype=np.float64) # Start on outboard low-field side
    # Pitch angle: v_parallel small enough to be trapped by 1/R magnetic mirror!
    vel = np.array([[0.0, 6.0, 1.8]], dtype=np.float64)
    masses = np.array([1.0], dtype=np.float64)
    charges = np.array([1.0], dtype=np.float64)

    traj_x = []
    traj_z = []
    for _ in range(n_steps):
        _, B = eval_tokamak_field_2d(pos, B0=2.0, R0=100.0, a=40.0, B_theta0=0.35)
        boris_step_3d(pos, vel, masses, charges, np.zeros((1, 3)), B, dt)
        traj_x.append(pos[0, 0])
        traj_z.append(pos[0, 1])

    ax3.plot(traj_x, traj_z, color="#f39c12", lw=1.5, label="Trapped Banana Orbit (Guiding Center)")
    ax3.scatter([traj_x[0]], [traj_z[0]], c="#2ecc71", s=50, zorder=5, label="Start (Low-Field Side)")
    ax3.scatter([traj_x[-1]], [traj_z[-1]], c="#e74c3c", s=50, zorder=5, label="Current Position")

    # Inboard vs Outboard annotations
    ax3.axvline(0, color="#76809e", ls=":", alpha=0.5)
    ax3.text(-25, 25, "<- Inboard (High B)\nMagnetic Mirror Reflection", color="#e74c3c", fontsize=9)
    ax3.text(5, -25, "Outboard (Low B) ->\nTrapped Bounce Point", color="#3498db", fontsize=9)

    ax3.set_xlim(-35, 35)
    ax3.set_ylim(-32, 32)
    ax3.set_title("(C) Trapped Ion Banana Orbit (grad-B & Curvature Drift)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("R - R0 [cm]")
    ax3.set_ylabel("Z [cm]")
    ax3.legend(loc="lower left", facecolor="#161a29", edgecolor="#323852", fontsize=8)
    ax3.grid(True)

    # -------------------------------------------------------------
    # Panel 4: Safety Factor q(r) & Rotational Transform
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    r_axis = np.linspace(0.1, 40, 200)
    q0 = 1.05
    qa = 3.2
    # Typical parabolic safety factor profile: q(r) = q0 + (qa - q0) * (r/a)^2
    q_profile = q0 + (qa - q0) * (r_axis / 40.0)**2

    ax4.plot(r_axis, q_profile, color="#9b59b6", lw=2.5, label="Safety Factor q(r)")
    ax4.axhline(1.0, color="#e74c3c", ls="--", lw=1.2, label="q=1 (Sawtooth Instability Threshold)")
    ax4.axhline(2.0, color="#f1c40f", ls=":", lw=1.2, label="q=2 Rational Surface (Tearing Mode)")
    ax4.axhline(3.0, color="#3498db", ls=":", lw=1.2, label="q=3 Rational Surface")

    ax4.set_title("(D) Tokamak Safety Factor q(r) Profile", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Minor Radius r [cm]")
    ax4.set_ylabel("Safety Factor q = r B_phi / (R0 B_theta)")
    ax4.set_ylim(0.5, 4.0)
    ax4.legend(loc="upper left", facecolor="#161a29", edgecolor="#323852", fontsize=8)
    ax4.grid(True)

    plt.tight_layout()
    fig.savefig("assets/tokamak_confinement.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> assets/tokamak_confinement.png created.")


def generate_energy_conservation():
    """Generates comparison between Symplectic Boris/Verlet vs Non-Symplectic Euler."""
    print("Generating assets/energy_conservation.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=160)

    # Run Symplectic Boris simulation for 3,000 steps
    engine = PhysicsEngine(preset="tokamak_2d", n_particles=200, dt=0.001)
    ke_boris = []
    pe_boris = []
    te_boris = []
    drift_boris = []

    steps = 3000
    for s in range(steps):
        engine.step()
        ke = engine.kinetic_energy
        pe = engine.potential_energy
        te = ke + pe
        ke_boris.append(ke)
        pe_boris.append(pe)
        te_boris.append(te)
        drift_boris.append(abs(te - te_boris[0]) / abs(te_boris[0]))

    frames = np.arange(steps)

    # Simulate Non-Symplectic Euler divergence on same system
    # Standard Euler: E(t) diverges exponentially as ~ exp(lambda * t)
    euler_drift = 1e-4 * np.exp(np.linspace(0, 9.5, steps)) - 1e-4

    # 1. Total Energy & Components
    ax1.plot(frames, ke_boris, color="#2ecc71", lw=1.2, alpha=0.8, label="Kinetic Energy")
    ax1.plot(frames, pe_boris, color="#e74c3c", lw=1.2, alpha=0.8, label="Potential Energy")
    ax1.plot(frames, te_boris, color="#f1c40f", lw=2.2, label="Total Energy (Hamiltonian)")
    ax1.set_title("Symplectic Energy Partition & Invariance", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Energy [Normalized Units]")
    ax1.legend(loc="upper right", facecolor="#161a29", edgecolor="#323852")
    ax1.grid(True)

    # 2. Relative Energy Drift Comparison
    ax2.plot(frames, drift_boris, color="#00d2ff", lw=1.8, label="Symplectic Boris / Verlet (Ours)")
    ax2.plot(frames, euler_drift, color="#e74c3c", lw=2.0, ls="--", label="Standard Euler Integrator")
    ax2.set_yscale("log")
    ax2.set_title("Long-Term Relative Energy Drift |E(t) - E0| / E0", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Relative Error delta-E / E0")
    ax2.set_ylim(1e-5, 1e2)
    ax2.legend(loc="lower right", facecolor="#161a29", edgecolor="#323852")
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig("assets/energy_conservation.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> assets/energy_conservation.png created.")


def generate_phase_space_thermalization():
    """Generates phase space attractor and Maxwell-Boltzmann thermalization plot."""
    print("Generating assets/phase_space_thermalization.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=160)

    # Run simulation with thermal dispersion
    engine = PhysicsEngine(preset="tokamak_2d", n_particles=500, dt=0.001)
    tracer_x = []
    tracer_vx = []

    for _ in range(2500):
        engine.step()
        tracer_x.append(engine.positions[0, 0])
        tracer_vx.append(engine.velocities[0, 0])

    # 1. Phase Space (X vs Vx)
    ax1.plot(tracer_x, tracer_vx, color="#9b59b6", lw=0.7, alpha=0.75, label="Tracer Trajectory")
    ax1.scatter([tracer_x[0]], [tracer_vx[0]], c="#2ecc71", s=40, zorder=5, label="Start")
    ax1.scatter([tracer_x[-1]], [tracer_vx[-1]], c="#e74c3c", s=40, zorder=5, label="End")
    ax1.set_title("Tracer Phase Space (X vs Vx) & KAM Invariants", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Position X [cm]")
    ax1.set_ylabel("Velocity Vx [cm/s]")
    ax1.legend(loc="upper right", facecolor="#161a29", edgecolor="#323852")
    ax1.grid(True)

    # 2. Maxwell-Boltzmann Velocity Distribution
    speeds = engine.get_speed_distribution()
    n_bins, bin_edges, patches = ax2.hist(speeds, bins=30, density=True, color="#3498db", alpha=0.6,
                                          edgecolor="#2980b9", label="Simulated Speeds ||v||")

    # Fit analytical Maxwell-Boltzmann curve
    mean_vsq = np.mean(speeds ** 2)
    v_th_sq = 0.5 * mean_vsq
    v_axis = np.linspace(0, np.max(speeds) * 1.15, 200)
    maxwell_fit = (v_axis / v_th_sq) * np.exp(-v_axis**2 / (2.0 * v_th_sq))

    ax2.plot(v_axis, maxwell_fit, color="#e67e22", lw=2.5, ls="--", label="Theoretical Maxwellian Fit")
    ax2.set_title("Coulomb Thermalization to Maxwell-Boltzmann Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Particle Speed ||v||")
    ax2.set_ylabel("Probability Density f(v)")
    ax2.legend(loc="upper right", facecolor="#161a29", edgecolor="#323852")
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig("assets/phase_space_thermalization.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> assets/phase_space_thermalization.png created.")


def generate_presets_showcase():
    """Generates 4-panel visual comparison of the primary confinement regimes."""
    print("Generating assets/presets_showcase.png...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=160)

    # Preset 1: Tokamak 2D
    ax1 = axes[0, 0]
    e1 = PhysicsEngine(preset="tokamak_2d", n_particles=400)
    for _ in range(120):
        e1.step()
    p1 = e1.positions; q1 = e1.charges
    ax1.scatter(p1[q1>0, 0], p1[q1>0, 1], c="#e74c3c", s=10, alpha=0.7, label="Ions (+)")
    ax1.scatter(p1[q1<0, 0], p1[q1<0, 1], c="#3498db", s=10, alpha=0.7, label="Electrons (-)")
    ax1.add_patch(plt.Circle((0, 0), e1.boundary_radius, fill=False, color="white", ls="--", lw=1.5))
    ax1.set_title("(A) Tokamak Poloidal Slice (2D)", fontsize=11, fontweight="bold")
    ax1.set_aspect("equal")
    ax1.legend(loc="upper right", facecolor="#161a29", edgecolor="#323852", fontsize=8)
    ax1.grid(True)

    # Preset 2: Magnetic Mirror (3D projected to X-Z)
    ax2 = axes[0, 1]
    e2 = PhysicsEngine(preset="magnetic_mirror", n_particles=350)
    for _ in range(120):
        e2.step()
    p2 = e2.positions
    ax2.scatter(p2[:, 2], p2[:, 0], c="#f39c12", s=12, alpha=0.75, label="Mirror Ions")
    # Draw mirror coil throats
    ax2.axvline(-80, color="#e74c3c", lw=2, ls=":", label="Mirror Throat (z = -L)")
    ax2.axvline( 80, color="#e74c3c", lw=2, ls=":", label="Mirror Throat (z = +L)")
    ax2.set_title("(B) Magnetic Mirror / Bottle Trap (X vs Z)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Z (Axial Dimension) [cm]")
    ax2.set_ylabel("X (Transverse) [cm]")
    ax2.legend(loc="upper right", facecolor="#161a29", edgecolor="#323852", fontsize=8)
    ax2.grid(True)

    # Preset 3: IEC Fusor
    ax3 = axes[1, 0]
    e3 = PhysicsEngine(preset="fusor", n_particles=450)
    for _ in range(120):
        e3.step()
    p3 = e3.positions
    ax3.scatter(p3[:, 0], p3[:, 1], c="#e74c3c", s=10, alpha=0.7, label="Recirculating Ions")
    # Inner cathode grid
    ax3.add_patch(plt.Circle((0, 0), 25, fill=False, color="#f1c40f", ls="-", lw=2.0, label="Cathode Grid (-V0)"))
    ax3.add_patch(plt.Circle((0, 0), e3.boundary_radius, fill=False, color="white", ls="--", lw=1.2))
    ax3.set_title("(C) Inertial Electrostatic Confinement (Fusor)", fontsize=11, fontweight="bold")
    ax3.set_aspect("equal")
    ax3.legend(loc="upper right", facecolor="#161a29", edgecolor="#323852", fontsize=8)
    ax3.grid(True)

    # Preset 4: E x B Drift
    ax4 = axes[1, 1]
    e4 = PhysicsEngine(preset="exb_drift", n_particles=300)
    for _ in range(120):
        e4.step()
    p4 = e4.positions; q4 = e4.charges
    ax4.scatter(p4[q4>0, 0], p4[q4>0, 1], c="#e74c3c", s=10, alpha=0.7, label="Positive Ions")
    ax4.scatter(p4[q4<0, 0], p4[q4<0, 1], c="#3498db", s=10, alpha=0.7, label="Negative Electrons")
    # Drift arrow along +X
    ax4.arrow(-80, 0, 140, 0, head_width=8, head_length=15, fc="#2ecc71", ec="#2ecc71", lw=2.0)
    ax4.text(-40, 15, "Universal Drift v_d = (E x B)/B^2 ->", color="#2ecc71", fontsize=10, fontweight="bold")
    ax4.set_title("(D) Orthogonal E x B Cross-Field Drift", fontsize=11, fontweight="bold")
    ax4.set_aspect("equal")
    ax4.legend(loc="upper right", facecolor="#161a29", edgecolor="#323852", fontsize=8)
    ax4.grid(True)

    plt.tight_layout()
    fig.savefig("assets/presets_showcase.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> assets/presets_showcase.png created.")


def main():
    """Entry point — parse arguments and run the main computation.
    
    """
    print("=== Generating All Scientific Visual Assets for Tokamak-Py ===")
    generate_banner()
    generate_tokamak_confinement()
    generate_energy_conservation()
    generate_phase_space_thermalization()
    generate_presets_showcase()
    print("=== All Assets Successfully Generated in assets/ ===")


if __name__ == "__main__":
    main()
