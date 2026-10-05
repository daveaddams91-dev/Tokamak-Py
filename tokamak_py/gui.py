"""
Tokamak-Py: Modern Interactive PyQt6 / PyQtGraph Simulation Dashboard.

Provides real-time interactive telemetry:
- Live particle confinement sandbox with field contours and tracer trails
- Symplectic energy conservation monitoring (KE, PE, Total Energy)
- Real-time Maxwell-Boltzmann velocity distribution histogram
- Phase-space attractor portrait (X vs Vx)
- Full interactive control bar: presets, play/pause/step, B-field slider, dt tuner
"""

import sys
from collections import deque
import numpy as np

import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout
)
from PyQt6.QtCore import Qt, QTimer

from .engine import PhysicsEngine
from .presets import PRESETS
from .gui_controls import ControlToolbar
from .gui_telemetry import TelemetryGraphics


class PlasmaSimulationApp(QMainWindow):
    """Interactive GUI application for Tokamak-Py."""

    def __init__(self, default_preset: str = "tokamak_2d"):
        """Init.
        
        Args:
            default_preset (str):
        
        """
        super().__init__()
        self.setWindowTitle("Tokamak-Py: Advanced Plasma Confinement Engine")
        self.resize(1500, 920)

        # Apply modern dark theme styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0d0f17;
            }
            QWidget {
                background-color: #0d0f17;
                color: #e0e0ea;
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #232738;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #9ba1b8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #1a1d2e;
                border: 1px solid #323850;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: 600;
                color: #e0e0ea;
            }
            QPushButton:hover {
                background-color: #262c45;
                border-color: #4a5478;
            }
            QPushButton:pressed {
                background-color: #353d61;
            }
            QPushButton#btn_play {
                background-color: #1b4d3e;
                border-color: #2ecc71;
                color: #ffffff;
            }
            QPushButton#btn_play:hover {
                background-color: #27ae60;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #161824;
                border: 1px solid #2e334a;
                border-radius: 4px;
                padding: 4px 8px;
                color: #e0e0ea;
            }
            QComboBox::drop-down {
                border: none;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #232738;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #3498db;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #ecf0f1;
                border: 1px solid #bdc3c7;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
            QCheckBox {
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #323850;
                background-color: #161824;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #3498db;
                border-color: #3498db;
            }
        """)

        # Simulation state
        self.is_running = True
        self.preset_name = default_preset
        self.engine = PhysicsEngine(preset=self.preset_name)

        # Telemetry history buffers
        self.history_len = 600
        self.ke_history = deque(maxlen=self.history_len)
        self.pe_history = deque(maxlen=self.history_len)
        self.te_history = deque(maxlen=self.history_len)
        self.frame_history = deque(maxlen=self.history_len)
        self.tracer_x_history = deque(maxlen=self.history_len)
        self.tracer_y_history = deque(maxlen=self.history_len)
        self.tracer_vx_history = deque(maxlen=self.history_len)
        self.frame_count = 0
        self.substeps_per_frame = 5

        # Initialize GUI Layout
        self._init_ui()

        # Main animation timer (~60 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(16)

    def _init_ui(self):
        """Constructs the application layout."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # TOP: Interactive Control Toolbar
        self.controls = ControlToolbar()
        self.controls.connect_signals(self)
        self.controls.initialize_values(self.engine, self.preset_name)
        main_layout.addWidget(self.controls)

        # CENTER: Telemetry Graphics Grid
        pg.setConfigOptions(antialias=True)
        self.telemetry = TelemetryGraphics()
        main_layout.addWidget(self.telemetry, stretch=1)

        # Re-center boundary for initial preset
        self.telemetry.update_boundary_display(self.engine)

    def toggle_play_pause(self):
        """Toggle play pause."""
        self.is_running = not self.is_running
        if self.is_running:
            self.controls.btn_play.setText("Pause")
            self.controls.btn_play.setStyleSheet("background-color: #1b4d3e; border-color: #2ecc71; color: white;")
        else:
            self.controls.btn_play.setText("Play")
            self.controls.btn_play.setStyleSheet("background-color: #7d4e1a; border-color: #e67e22; color: white;")

    def single_step(self):
        """Single step."""
        for _ in range(self.substeps_per_frame):
            self.engine.step()
        self.refresh_display()

    def reset_simulation(self):
        """Reset simulation."""
        self.engine.reset()
        self.ke_history.clear()
        self.pe_history.clear()
        self.te_history.clear()
        self.frame_history.clear()
        self.tracer_x_history.clear()
        self.tracer_y_history.clear()
        self.tracer_vx_history.clear()
        self.frame_count = 0
        self.telemetry.update_boundary_display(self.engine)

    def on_preset_changed(self, index):
        """On preset changed."""
        key = self.controls.combo_preset.itemData(index)
        self.preset_name = key
        self.engine = PhysicsEngine(preset=key)
        self.controls.spin_particles.setValue(self.engine.n_particles)
        self.controls.slider_b_field.setValue(100)
        self.controls.slider_dt.setValue(int(self.engine.dt * 10000))
        self.reset_simulation()

    def on_particle_count_changed(self, val):
        """On particle count changed."""
        self.engine.n_particles = val
        self.reset_simulation()

    def on_b_field_changed(self, val):
        """On b field changed."""
        scale = val / 100.0
        self.engine.b_field_scale = scale
        self.controls.lbl_b_val.setText(f"{scale:.1f}x")

    def on_dt_changed(self, val):
        """On dt changed."""
        dt = val / 10000.0
        self.engine.dt = dt
        self.controls.lbl_dt_val.setText(f"{dt:.4f}")

    def update_simulation(self):
        """Update simulation."""
        if not self.is_running:
            return

        # Sub-stepping for numerical precision
        for _ in range(self.substeps_per_frame):
            self.engine.step()

        self.refresh_display()

    def refresh_display(self):
        """Refresh display."""
        pos = self.engine.positions
        charges = self.engine.charges
        
        # 1. Update Sandbox Scatter Plot
        brushes = [pg.mkBrush('#e74c3c') if q > 0 else pg.mkBrush('#3498db') for q in charges]
        self.telemetry.scatter.setData(x=pos[:, 0], y=pos[:, 1], brush=brushes)

        # 2. Update Tracer Trail
        tracer_x = pos[0, 0]
        tracer_y = pos[0, 1]
        self.tracer_x_history.append(tracer_x)
        self.tracer_y_history.append(tracer_y)
        self.tracer_vx_history.append(self.engine.velocities[0, 0])

        if self.controls.chk_trail.isChecked():
            self.telemetry.trail_item.setData(list(self.tracer_x_history), list(self.tracer_y_history))
        else:
            self.telemetry.trail_item.clear()

        # 3. Update Energy Conservation Curves
        self.frame_count += 1
        self.frame_history.append(self.frame_count)
        self.ke_history.append(self.engine.kinetic_energy)
        self.pe_history.append(self.engine.potential_energy)
        self.te_history.append(self.engine.get_total_energy())

        frames = list(self.frame_history)
        self.telemetry.ke_curve.setData(frames, list(self.ke_history))
        self.telemetry.pe_curve.setData(frames, list(self.pe_history))
        self.telemetry.te_curve.setData(frames, list(self.te_history))

        # 4. Update Maxwell-Boltzmann Speed Distribution
        speeds = self.engine.get_speed_distribution()
        if bool(speeds) and self.frame_count % 3 == 0:
            hist, bin_edges = np.histogram(speeds, bins=25, density=True)
            self.telemetry.hist_curve.setData(bin_edges, hist)

            # Theoretical 2D/3D Maxwellian fit
            mean_vsq = np.mean(speeds ** 2)
            if mean_vsq > 1e-4:
                v_axis = np.linspace(0, np.max(speeds) * 1.2, 100)
                # 2D Maxwell-Boltzmann: f(v) = (v / v_th^2) * exp(-v^2 / (2 v_th^2))
                v_th_sq = 0.5 * mean_vsq
                fit = (v_axis / v_th_sq) * np.exp(-v_axis**2 / (2.0 * v_th_sq))
                self.telemetry.maxwell_curve.setData(v_axis, fit)

        # 5. Status readout
        drift = self.engine.get_relative_energy_drift() * 100.0
        self.controls.lbl_status.setText(f"Steps: {self.engine.step_count} | ΔE/E₀: {drift:.2e}%")


def main():
    """Entry point — parse arguments and run the main computation."""
    app = pg.mkQApp("Tokamak-Py Dashboard")
    window = PlasmaSimulationApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
