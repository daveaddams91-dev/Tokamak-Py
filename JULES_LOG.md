
## Refactored tree.py into quadtree.py and octree.py
Broke down the massive 440-line `tree.py` file which had high cyclomatic complexity by separating the 2D Barnes-Hut implementation (`quadtree.py`) and the 3D Barnes-Hut implementation (`octree.py`). This separation of concerns improves readability and maintainability. Updated all cross-references across `tests/`, `tokamak_py/`, and `physics.py`.

Refactored `tokamak_py/gui.py` to break down the monolithic UI class into smaller, single-responsibility modules: `tokamak_py/gui_controls.py` for control inputs and `tokamak_py/gui_telemetry.py` for the pyqtgraph plots. This separation significantly improves code maintainability and testability of the GUI components without altering the underlying physics engine logic.

## Enabled Numba JIT caching
Updated the Numba `@njit` decorators across the core physics modules (`direct.py`, `fields.py`, `integrators.py`, `octree.py`, `quadtree.py`) to include `cache=True`. This resolves a major developer experience and performance issue where compilation overhead (often taking over 15 seconds) was incurred on every script execution. Subsequent runs now start almost instantly by leveraging pre-compiled bytecode caches.
