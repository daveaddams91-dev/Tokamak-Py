"""
Tokamak-Py: Universal Entrypoint.

Launches the real-time interactive PyQt6/PyQtGraph dashboard by default,
or routes to the headless CLI runner when flags (--headless, --benchmark, run)
are supplied or when running in a displayless environment.
"""

import sys
import argparse

from tokamak_py.engine import PhysicsEngine
from tokamak_py.presets import PRESETS

def run_gui(default_preset="tokamak_2d"):
    """Launches the interactive PyQt6 dashboard."""
    try:
        from PyQt6.QtWidgets import QApplication
        from tokamak_py.gui import PlasmaSimulationApp
        
        app = QApplication(sys.argv)
        window = PlasmaSimulationApp(default_preset=default_preset)
        window.show()
        return app.exec()
    except Exception as e:
        print(f"[Tokamak-Py Notice] Could not initialize GUI ({e}).")
        print("Falling back to headless CLI runner. Use 'python main.py --help' for CLI options.\n")
        from tokamak_py.cli import run_simulation
        # Run default headless simulation as fallback
        fallback_args = argparse.Namespace(
            preset=default_preset,
            particles=600,
            steps=300,
            dt=0.001,
            b_field=1.0,
            solver="barnes_hut",
            output="assets/quickstart_sim.png",
            quiet=False
        )
        run_simulation(fallback_args)
        return 0


def main():
    # If explicit CLI subcommands or headless flags are passed, invoke CLI
    cli_commands = {"run", "benchmark", "presets"}
    if len(sys.argv) > 1:
        first_arg = sys.argv[1].lower()
        if first_arg in cli_commands or "--headless" in sys.argv or "-h" in sys.argv or "--help" in sys.argv:
            from tokamak_py.cli import main as cli_main
            # Strip --headless if present
            if "--headless" in sys.argv:
                sys.argv.remove("--headless")
            if len(sys.argv) == 1:
                sys.argv.append("run")
            return cli_main()

    # Default: launch interactive GUI
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
