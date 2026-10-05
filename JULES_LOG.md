
## Refactored tree.py into quadtree.py and octree.py
Broke down the massive 440-line `tree.py` file which had high cyclomatic complexity by separating the 2D Barnes-Hut implementation (`quadtree.py`) and the 3D Barnes-Hut implementation (`octree.py`). This separation of concerns improves readability and maintainability. Updated all cross-references across `tests/`, `tokamak_py/`, and `physics.py`.
