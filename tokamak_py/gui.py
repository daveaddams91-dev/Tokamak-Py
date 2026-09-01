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
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QSlider, QLabel, QSpinBox,
    QDoubleSpinBox, QCheckBox, QFrame, QGroupBox, QSplitter
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor

from .engine import PhysicsEngine
from .presets import PRESETS, get_preset, SimulationPreset


class PlasmaSimulationApp(QMainWindow):
    """Interactive GUI application for Tokamak-Py."""

    def __init__(self, default_preset: str = "tokamak_2d"):
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

        # -------------------------------------------------------------
        # TOP: Interactive Control Toolbar
        # -------------------------------------------------------------
        controls_frame = QFrame()
        controls_frame.setStyleSheet("background-color: #131622; border-radius: 6px; padding: 4px;")
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(10, 6, 10, 6)
        controls_layout.setSpacing(12)

        # Play / Pause button
        self.btn_play = QPushButton("Pause", self)
        self.btn_play.setObjectName("btn_play")
        self.btn_play.clicked.connect(self.toggle_play_pause)
        controls_layout.addWidget(self.btn_play)

        # Step button
        self.btn_step = QPushButton("Step", self)
        self.btn_step.clicked.connect(self.single_step)
        controls_layout.addWidget(self.btn_step)

        # Reset button
        self.btn_reset = QPushButton("Reset", self)
        self.btn_reset.clicked.connect(self.reset_simulation)
        controls_layout.addWidget(self.btn_reset)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet("color: #2e334a;")
        controls_layout.addWidget(sep1)

        # Preset Switcher
        controls_layout.addWidget(QLabel("Preset:"))
        self.combo_preset = QComboBox(self)
        for key, p in PRESETS.items():
            self.combo_preset.addItem(p.label, key)
        idx = self.combo_preset.findData(self.preset_name)
        if idx >= 0:
            self.combo_preset.setCurrentIndex(idx)
        self.combo_preset.currentIndexChanged.connect(self.on_preset_changed)
        controls_layout.addWidget(self.combo_preset)

        # Particle Count Selector
        controls_layout.addWidget(QLabel("Particles:"))
        self.spin_particles = QSpinBox(self)
        self.spin_particles.setRange(50, 5000)
        self.spin_particles.setSingleStep(100)
        self.spin_particles.setValue(self.engine.n_particles)
        self.spin_particles.valueChanged.connect(self.on_particle_count_changed)
        controls_layout.addWidget(self.spin_particles)

        # Magnetic Field Multiplier Slider
        controls_layout.addWidget(QLabel("B-Field:"))
        self.slider_b_field = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_b_field.setRange(0, 400) # 0.0x to 4.0x
        self.slider_b_field.setValue(100)
        self.slider_b_field.setFixedWidth(100)
        self.slider_b_field.valueChanged.connect(self.on_b_field_changed)
        controls_layout.addWidget(self.slider_b_field)
        self.lbl_b_val = QLabel("1.0x")
        self.lbl_b_val.setFixedWidth(35)
        controls_layout.addWidget(self.lbl_b_val)

        # Time Step dt Slider
        controls_layout.addWidget(QLabel("dt:"))
        self.slider_dt = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_dt.setRange(1, 30) # 0.0002 to 0.0030
        self.slider_dt.setValue(int(self.engine.dt * 10000))
        self.slider_dt.setFixedWidth(80)
        self.slider_dt.valueChanged.connect(self.on_dt_changed)
        controls_layout.addWidget(self.slider_dt)
        self.lbl_dt_val = QLabel(f"{self.engine.dt:.4f}")
        self.lbl_dt_val.setFixedWidth(45)
        controls_layout.addWidget(self.lbl_dt_val)

        # Tracer Trail Toggle
        self.chk_trail = QCheckBox("Tracer Trail", self)
        self.chk_trail.setChecked(True)
        controls_layout.addWidget(self.chk_trail)

        controls_layout.addStretch()

        # Telemetry Status Label
        self.lbl_status = QLabel("FPS: -- | Energy Drift: 0.00%")
        self.lbl_status.setStyleSheet("color: #2ecc71; font-weight: bold;")
        controls_layout.addWidget(self.lbl_status)

        main_layout.addWidget(controls_frame)

        # -------------------------------------------------------------
        # CENTER: Telemetry Graphics Grid
        # -------------------------------------------------------------
        pg.setConfigOptions(antialias=True)
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground("#0d0f17")
        main_layout.addWidget(self.graphics_widget, stretch=1)

        # Set 2x2 Grid proportions
        self.graphics_widget.ci.layout.setColumnStretchFactor(0, 3)
        self.graphics_widget.ci.layout.setColumnStretchFactor(1, 2)

        # Panel 1: Particle Confinement Sandbox (Row 0, Col 0, Rowspan 2)
        self.sandbox_plot = self.graphics_widget.addPlot(title="Plasma Confinement Sandbox", row=0, col=0, rowspan=2)
        self.sandbox_plot.showGrid(x=True, y=True, alpha=0.25)
        self.sandbox_plot.setAspectLocked(True)

        # Add boundary outline
        self.boundary_item = pg.PlotDataItem(pen=pg.mkPen('#ffffff', width=1.5, style=Qt.PenStyle.DashLine))
        self.sandbox_plot.addItem(self.boundary_item)

        # Add tracer trail line
        self.trail_item = pg.PlotDataItem(pen=pg.mkPen('#f1c40f', width=1.5))
        self.sandbox_plot.addItem(self.trail_item)

        # Scatter plot for particles
        self.scatter = pg.ScatterPlotItem(size=5, pen=None)
        self.sandbox_plot.addItem(self.scatter)

        # Panel 2: Symplectic Energy Conservation (Row 0, Col 1)
        self.energy_plot = self.graphics_widget.addPlot(title="Symplectic Energy Telemetry", row=0, col=1)
        self.energy_plot.addLegend(offset=(10, 10))
        self.energy_plot.setLabel('left', 'Energy')
        self.energy_plot.setLabel('bottom', 'Frames')
        self.energy_plot.showGrid(x=True, y=True, alpha=0.25)

        self.ke_curve = self.energy_plot.plot(pen=pg.mkPen('#2ecc71', width=1.8), name="Kinetic Energy")
        self.pe_curve = self.energy_plot.plot(pen=pg.mkPen('#e74c3c', width=1.8), name="Potential Energy")
        self.te_curve = self.energy_plot.plot(pen=pg.mkPen('#f1c40f', width=2.2), name="Total Energy")

        # Panel 3: Velocity Distribution / Maxwell-Boltzmann (Row 1, Col 1)
        self.dist_plot = self.graphics_widget.addPlot(title="Maxwell-Boltzmann Speed Distribution", row=1, col=1)
        self.dist_plot.setLabel('left', 'Probability Density')
        self.dist_plot.setLabel('bottom', 'Speed ||v||')
        self.dist_plot.showGrid(x=True, y=True, alpha=0.25)

        self.hist_curve = self.dist_plot.plot(stepMode="center", fillLevel=0, brush=pg.mkBrush(52, 152, 219, 100), pen=pg.mkPen('#3498db', width=1.5))
        self.maxwell_curve = self.dist_plot.plot(pen=pg.mkPen('#e67e22', width=2.0, style=Qt.PenStyle.DashLine), name="Maxwell Fit")

        # Re-center boundary for initial preset
        self._update_boundary_display()

    def _update_boundary_display(self):
        """Updates boundary visual curve based on current engine geometry."""
        r = self.engine.boundary_radius
        self.sandbox_plot.setXRange(-r * 1.05, r * 1.05)
        self.sandbox_plot.setYRange(-r * 1.05, r * 1.05)

        theta = np.linspace(0, 2 * np.pi, 250)
        bx = r * np.cos(theta)
        by = r * np.sin(theta)
        self.boundary_item.setData(bx, by)

    def toggle_play_pause(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.btn_play.setText("Pause")
            self.btn_play.setStyleSheet("background-color: #1b4d3e; border-color: #2ecc71; color: white;")
        else:
            self.btn_play.setText("Play")
            self.btn_play.setStyleSheet("background-color: #7d4e1a; border-color: #e67e22; color: white;")

    def single_step(self):
        for _ in range(self.substeps_per_frame):
            self.engine.step()
        self.refresh_display()

    def reset_simulation(self):
        self.engine.reset()
        self.ke_history.clear()
        self.pe_history.clear()
        self.te_history.clear()
        self.frame_history.clear()
        self.tracer_x_history.clear()
        self.tracer_y_history.clear()
        self.tracer_vx_history.clear()
        self.frame_count = 0
        self._update_boundary_display()

    def on_preset_changed(self, index):
        key = self.combo_preset.itemData(index)
        self.preset_name = key
        self.engine = PhysicsEngine(preset=key)
        self.spin_particles.setValue(self.engine.n_particles)
        self.slider_b_field.setValue(100)
        self.slider_dt.setValue(int(self.engine.dt * 10000))
        self.reset_simulation()

    def on_particle_count_changed(self, val):
        self.engine.n_particles = val
        self.reset_simulation()

    def on_b_field_changed(self, val):
        scale = val / 100.0
        self.engine.b_field_scale = scale
        self.lbl_b_val.setText(f"{scale:.1f}x")

    def on_dt_changed(self, val):
        dt = val / 10000.0
        self.engine.dt = dt
        self.lbl_dt_val.setText(f"{dt:.4f}")

    def update_simulation(self):
        if not self.is_running:
            return

        # Sub-stepping for numerical precision
        for _ in range(self.substeps_per_frame):
            self.engine.step()

        self.refresh_display()

    def refresh_display(self):
        # 1. Update Sandbox Scatter Plot
        pos = self.engine.positions
        charges = self.engine.charges
        
        brushes = [pg.mkBrush('#e74c3c') if q > 0 else pg.mkBrush('#3498db') for q in charges]
        self.scatter.setData(x=pos[:, 0], y=pos[:, 1], brush=brushes)

        # 2. Update Tracer Trail
        tracer_x = pos[0, 0]
        tracer_y = pos[0, 1]
        self.tracer_x_history.append(tracer_x)
        self.tracer_y_history.append(tracer_y)
        self.tracer_vx_history.append(self.engine.velocities[0, 0])

        if self.chk_trail.isChecked():
            self.trail_item.setData(list(self.tracer_x_history), list(self.tracer_y_history))
        else:
            self.trail_item.clear()

        # 3. Update Energy Conservation Curves
        self.frame_count += 1
        self.frame_history.append(self.frame_count)
        self.ke_history.append(self.engine.kinetic_energy)
        self.pe_history.append(self.engine.potential_energy)
        self.te_history.append(self.engine.get_total_energy())

        frames = list(self.frame_history)
        self.ke_curve.setData(frames, list(self.ke_history))
        self.pe_curve.setData(frames, list(self.pe_history))
        self.te_curve.setData(frames, list(self.te_history))

        # 4. Update Maxwell-Boltzmann Speed Distribution
        speeds = self.engine.get_speed_distribution()
        if len(speeds) > 0 and self.frame_count % 3 == 0:
            hist, bin_edges = np.histogram(speeds, bins=25, density=True)
            self.hist_curve.setData(bin_edges, hist)

            # Theoretical 2D/3D Maxwellian fit
            mean_vsq = np.mean(speeds ** 2)
            if mean_vsq > 1e-4:
                v_axis = np.linspace(0, np.max(speeds) * 1.2, 100)
                # 2D Maxwell-Boltzmann: f(v) = (v / v_th^2) * exp(-v^2 / (2 v_th^2))
                v_th_sq = 0.5 * mean_vsq
                fit = (v_axis / v_th_sq) * np.exp(-v_axis**2 / (2.0 * v_th_sq))
                self.maxwell_curve.setData(v_axis, fit)

        # 5. Status readout
        drift = self.engine.get_relative_energy_drift() * 100.0
        self.lbl_status.setText(f"Steps: {self.engine.step_count} | ΔE/E₀: {drift:.2e}%")


def main():
    app = pg.mkQApp("Tokamak-Py Dashboard")
    window = PlasmaSimulationApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
