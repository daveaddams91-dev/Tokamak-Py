"""
Tokamak-Py: Command-Line Interface (CLI).

Provides headless execution, automated benchmarking, diagnostic plotting,
and asset generation without requiring a GUI display server.
"""

import sys
import os
import time
import argparse
import numpy as np

from .engine import PhysicsEngine
from .presets import PRESETS, get_preset
from .direct import calculate_direct_forces_2d
from .tree import get_bounding_box_2d, build_quadtree, calculate_forces_2d


def run_simulation(args):
    """Executes a simulation run in headless mode and optionally saves a summary figure."""
    import matplotlib.pyplot as plt

    preset = get_preset(args.preset)
    n_particles = args.particles if args.particles is not None else preset.n_particles
    dt = args.dt if args.dt is not None else preset.dt
    
    print(f"=== Tokamak-Py Simulation Runner ===")
    print(f"Preset      : {preset.label} ({preset.dimension}D)")
    print(f"Particles   : {n_particles}")
    print(f"Time Step dt: {dt}")
    print(f"Steps       : {args.steps}")
    print(f"Solver      : {args.solver}")
    print(f"------------------------------------")

    engine = PhysicsEngine(
        preset=preset,
        n_particles=n_particles,
        dt=dt,
        solver=args.solver
    )
    if args.b_field is not None:
        engine.b_field_scale = args.b_field

    times = []
    ke_history = []
    pe_history = []
    te_history = []
    tracer_x = []
    tracer_y = []
    tracer_vx = []

    t_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        engine.step()
        t1 = time.perf_counter()
        times.append(t1 - t0)

        ke_history.append(engine.kinetic_energy)
        pe_history.append(engine.potential_energy)
        te_history.append(engine.get_total_energy())

        tracer_x.append(engine.positions[0, 0])
        tracer_y.append(engine.positions[0, 1])
        tracer_vx.append(engine.velocities[0, 0])

        if not args.quiet and (step % max(1, args.steps // 10) == 0 or step == args.steps):
            avg_step_ms = np.mean(times[-50:]) * 1000.0
            print(f"Step {step:5d}/{args.steps} | Total Energy: {engine.get_total_energy():12.2f} | Step Time: {avg_step_ms:6.2f} ms")

    total_time = time.perf_counter() - t_start
    avg_fps = args.steps / total_time
    print(f"------------------------------------")
    print(f"Completed {args.steps} steps in {total_time:.3f} s ({avg_fps:.1f} FPS, {1000.0 / avg_fps:.2f} ms/step)")
    print(f"Initial Energy: {te_history[0]:.4f} | Final Energy: {te_history[-1]:.4f} | Relative Drift: {engine.get_relative_energy_drift():.2e}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0f111a")
        for ax in axes:
            ax.set_facecolor("#171926")
            ax.tick_params(colors="#a0a0b0")
            for spine in ax.spines.values():
                spine.set_color("#30334e")

        # 1. Spatial distribution
        ax1 = axes[0]
        pos = engine.positions
        charges = engine.charges
        ax1.scatter(pos[charges > 0, 0], pos[charges > 0, 1], s=8, c="#e74c3c", label="Positive", alpha=0.7)
        ax1.scatter(pos[charges < 0, 0], pos[charges < 0, 1], s=8, c="#3498db", label="Negative", alpha=0.7)
        ax1.plot(tracer_x, tracer_y, c="#f1c40f", lw=1.2, label="Tracer Orbit")
        ax1.set_title(f"Spatial Sandbox ({preset.label})", color="#e0e0e8", fontsize=12, fontweight="bold")
        ax1.set_aspect("equal", "datalim")
        ax1.legend(loc="upper right", facecolor="#1e2235", edgecolor="#404566", labelcolor="#e0e0e8", fontsize=9)
        ax1.grid(True, color="#25293d", ls="--", alpha=0.6)

        # 2. Energy Conservation
        ax2 = axes[1]
        frames = np.arange(len(te_history))
        ax2.plot(frames, ke_history, label="Kinetic Energy", color="#2ecc71", lw=1.5)
        ax2.plot(frames, pe_history, label="Potential Energy", color="#e74c3c", lw=1.5)
        ax2.plot(frames, te_history, label="Total Energy", color="#f1c40f", lw=2.0)
        ax2.set_title("Energy Telemetry (Symplectic Boris)", color="#e0e0e8", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Time Steps", color="#a0a0b0")
        ax2.set_ylabel("Energy", color="#a0a0b0")
        ax2.legend(loc="upper right", facecolor="#1e2235", edgecolor="#404566", labelcolor="#e0e0e8", fontsize=9)
        ax2.grid(True, color="#25293d", ls="--", alpha=0.6)

        # 3. Phase Space (x, v_x)
        ax3 = axes[2]
        ax3.plot(tracer_x, tracer_vx, color="#9b59b6", lw=0.8, alpha=0.8)
        ax3.scatter([tracer_x[-1]], [tracer_vx[-1]], c="#e74c3c", s=25, zorder=5)
        ax3.set_title("Tracer Phase Space (X vs Vx)", color="#e0e0e8", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Position X", color="#a0a0b0")
        ax3.set_ylabel("Velocity X", color="#a0a0b0")
        ax3.grid(True, color="#25293d", ls="--", alpha=0.6)

        plt.tight_layout()
        fig.savefig(args.output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Diagnostic plot saved to: {args.output}")


def run_benchmark(args):
    """Benchmarks Barnes-Hut tree vs Direct O(N^2) summation across particle scales."""
    print("=== Tokamak-Py Computational Benchmark ===")
    print("Comparing Barnes-Hut O(N log N) vs Direct O(N^2) Summation")
    print(f"Theta: {args.theta} | Softening: {args.softening}")
    print("-------------------------------------------------------------------------")
    print(f"{'N Particles':>12} | {'Direct (ms)':>12} | {'Barnes-Hut (ms)':>16} | {'Speedup':>9} | {'Rel Error':>11}")
    print("-------------------------------------------------------------------------")

    particle_counts = [100, 250, 500, 1000, 2000, 4000]
    direct_times = []
    tree_times = []
    rel_errors = []

    # Warm up JIT compilers
    dummy_pos = np.random.uniform(-100, 100, (64, 2))
    dummy_q = np.ones(64)
    calculate_direct_forces_2d(dummy_pos, dummy_q, 1000.0, 1.0)
    cx, cy, s = get_bounding_box_2d(dummy_pos)
    nf, ni, _ = build_quadtree(dummy_pos, dummy_q, cx, cy, s)
    calculate_forces_2d(dummy_pos, dummy_q, nf, ni, 1000.0, 1.0, 0.25)

    for n in particle_counts:
        # Generate random test distribution
        pos = np.random.uniform(-150, 150, (n, 2))
        q = np.ones(n)
        q[n // 2:] = -1.0

        # Benchmark Direct O(N^2)
        n_repeats = 5 if n <= 1000 else 2
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            f_direct, pe_direct = calculate_direct_forces_2d(pos, q, 1000.0, args.softening)
        t_direct = (time.perf_counter() - t0) / n_repeats * 1000.0 # ms

        # Benchmark Barnes-Hut O(N log N)
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            cx, cy, size = get_bounding_box_2d(pos)
            nf, ni, _ = build_quadtree(pos, q, cx, cy, size)
            f_tree, pe_tree = calculate_forces_2d(pos, q, nf, ni, 1000.0, args.softening, args.theta)
        t_tree = (time.perf_counter() - t0) / n_repeats * 1000.0 # ms

        # Force relative error: ||F_tree - F_direct|| / ||F_direct||
        f_err = np.linalg.norm(f_tree - f_direct) / np.linalg.norm(f_direct)
        speedup = t_direct / t_tree if t_tree > 0 else 0.0

        direct_times.append(t_direct)
        tree_times.append(t_tree)
        rel_errors.append(f_err)

        print(f"{n:12d} | {t_direct:12.3f} | {t_tree:16.3f} | {speedup:8.2f}x | {f_err:11.4e}")

    print("-------------------------------------------------------------------------")
    
    if args.save_chart:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0f111a")
        for ax in (ax1, ax2):
            ax.set_facecolor("#171926")
            ax.tick_params(colors="#a0a0b0")
            for spine in ax.spines.values():
                spine.set_color("#30334e")
            ax.grid(True, color="#25293d", ls="--", alpha=0.6)

        # 1. Execution Time Scaling
        ax1.plot(particle_counts, direct_times, "o--", color="#e74c3c", lw=2, label="Direct O(N^2)")
        ax1.plot(particle_counts, tree_times, "s-", color="#3498db", lw=2, label=f"Barnes-Hut (theta={args.theta})")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Particle Count N", color="#a0a0b0")
        ax1.set_ylabel("Execution Time per Step (ms)", color="#a0a0b0")
        ax1.set_title("N-Body Algorithmic Scaling", color="#e0e0e8", fontsize=12, fontweight="bold")
        ax1.legend(facecolor="#1e2235", edgecolor="#404566", labelcolor="#e0e0e8")

        # 2. Relative Error
        ax2.plot(particle_counts, rel_errors, "o-", color="#2ecc71", lw=2)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Particle Count N", color="#a0a0b0")
        ax2.set_ylabel("Relative Force Error", color="#a0a0b0")
        ax2.set_title(f"Force Accuracy vs Direct Summation (theta={args.theta})", color="#e0e0e8", fontsize=12, fontweight="bold")

        plt.tight_layout()
        fig.savefig(args.save_chart, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Benchmark chart saved to: {args.save_chart}")


def list_presets():
    """Prints all configured presets and descriptions."""
    print("=== Tokamak-Py Available Presets ===")
    for key, p in PRESETS.items():
        print(f"\n* [{key}] - {p.label}")
        print(f"  Dimension  : {p.dimension}D | Particles: {p.n_particles} | Boundary: {p.boundary_type}")
        print(f"  Description: {p.description}")


def main():
    parser = argparse.ArgumentParser(description="Tokamak-Py: Plasma Physics & Confinement Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Command: run
    parser_run = subparsers.add_parser("run", help="Run simulation headlessly")
    parser_run.add_argument("--preset", "-p", type=str, default="tokamak_2d", choices=list(PRESETS.keys()))
    parser_run.add_argument("--particles", "-n", type=int, default=None, help="Override particle count")
    parser_run.add_argument("--steps", "-s", type=int, default=500, help="Number of integration steps")
    parser_run.add_argument("--dt", type=float, default=None, help="Override integration time step")
    parser_run.add_argument("--b-field", "-b", type=float, default=None, help="Magnetic field multiplier")
    parser_run.add_argument("--solver", type=str, default="barnes_hut", choices=["barnes_hut", "direct"])
    parser_run.add_argument("--output", "-o", type=str, default=None, help="File path to save diagnostic plot")
    parser_run.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    # Command: benchmark
    parser_bench = subparsers.add_parser("benchmark", help="Benchmark Barnes-Hut vs Direct O(N^2)")
    parser_bench.add_argument("--theta", type=float, default=0.25, help="Barnes-Hut opening angle")
    parser_bench.add_argument("--softening", type=float, default=1.0, help="Softening length")
    parser_bench.add_argument("--save-chart", type=str, default=None, help="Path to save benchmark chart")

    # Command: presets
    subparsers.add_parser("presets", help="List all available physical presets")

    args = parser.parse_args()

    if args.command == "run":
        run_simulation(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "presets":
        list_presets()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
