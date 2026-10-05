import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QComboBox, QSlider, QLabel, QSpinBox,
    QCheckBox, QFrame
)
from PyQt6.QtCore import Qt

from .presets import PRESETS

class ControlToolbar(QFrame):
    """Interactive control toolbar for the simulation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #131622; border-radius: 6px; padding: 4px;")

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 6, 10, 6)
        self.layout.setSpacing(12)

        # Play / Pause button
        self.btn_play = QPushButton("Pause", self)
        self.btn_play.setObjectName("btn_play")
        self.layout.addWidget(self.btn_play)

        # Step button
        self.btn_step = QPushButton("Step", self)
        self.layout.addWidget(self.btn_step)

        # Reset button
        self.btn_reset = QPushButton("Reset", self)
        self.layout.addWidget(self.btn_reset)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet("color: #2e334a;")
        self.layout.addWidget(sep1)

        # Preset Switcher
        self.layout.addWidget(QLabel("Preset:"))
        self.combo_preset = QComboBox(self)
        for key, p in PRESETS.items():
            self.combo_preset.addItem(p.label, key)
        self.layout.addWidget(self.combo_preset)

        # Particle Count Selector
        self.layout.addWidget(QLabel("Particles:"))
        self.spin_particles = QSpinBox(self)
        self.spin_particles.setRange(50, 5000)
        self.spin_particles.setSingleStep(100)
        self.layout.addWidget(self.spin_particles)

        # Magnetic Field Multiplier Slider
        self.layout.addWidget(QLabel("B-Field:"))
        self.slider_b_field = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_b_field.setRange(0, 400) # 0.0x to 4.0x
        self.slider_b_field.setValue(100)
        self.slider_b_field.setFixedWidth(100)
        self.layout.addWidget(self.slider_b_field)

        self.lbl_b_val = QLabel("1.0x")
        self.lbl_b_val.setFixedWidth(35)
        self.layout.addWidget(self.lbl_b_val)

        # Time Step dt Slider
        self.layout.addWidget(QLabel("dt:"))
        self.slider_dt = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_dt.setRange(1, 30) # 0.0002 to 0.0030
        self.slider_dt.setFixedWidth(80)
        self.layout.addWidget(self.slider_dt)

        self.lbl_dt_val = QLabel("0.0000")
        self.lbl_dt_val.setFixedWidth(45)
        self.layout.addWidget(self.lbl_dt_val)

        # Tracer Trail Toggle
        self.chk_trail = QCheckBox("Tracer Trail", self)
        self.chk_trail.setChecked(True)
        self.layout.addWidget(self.chk_trail)

        self.layout.addStretch()

        # Telemetry Status Label
        self.lbl_status = QLabel("FPS: -- | Energy Drift: 0.00%")
        self.lbl_status.setStyleSheet("color: #2ecc71; font-weight: bold;")
        self.layout.addWidget(self.lbl_status)

    def connect_signals(self, app):
        """Connect UI signals to the main app."""
        self.btn_play.clicked.connect(app.toggle_play_pause)
        self.btn_step.clicked.connect(app.single_step)
        self.btn_reset.clicked.connect(app.reset_simulation)
        self.combo_preset.currentIndexChanged.connect(app.on_preset_changed)
        self.spin_particles.valueChanged.connect(app.on_particle_count_changed)
        self.slider_b_field.valueChanged.connect(app.on_b_field_changed)
        self.slider_dt.valueChanged.connect(app.on_dt_changed)

    def initialize_values(self, engine, preset_name):
        """Initialize UI values based on the engine state."""
        idx = self.combo_preset.findData(preset_name)
        if idx >= 0:
            self.combo_preset.setCurrentIndex(idx)
        self.spin_particles.setValue(engine.n_particles)
        self.slider_dt.setValue(int(engine.dt * 10000))
        self.lbl_dt_val.setText(f"{engine.dt:.4f}")
