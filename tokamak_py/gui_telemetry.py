import pyqtgraph as pg
from PyQt6.QtCore import Qt
import numpy as np

class TelemetryGraphics(pg.GraphicsLayoutWidget):
    """Telemetry graphics widget including sandbox, energy, and speed distribution."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground("#0d0f17")

        # Set 2x2 Grid proportions
        self.ci.layout.setColumnStretchFactor(0, 3)
        self.ci.layout.setColumnStretchFactor(1, 2)

        # Panel 1: Particle Confinement Sandbox (Row 0, Col 0, Rowspan 2)
        self.sandbox_plot = self.addPlot(title="Plasma Confinement Sandbox", row=0, col=0, rowspan=2)
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
        self.energy_plot = self.addPlot(title="Symplectic Energy Telemetry", row=0, col=1)
        self.energy_plot.addLegend(offset=(10, 10))
        self.energy_plot.setLabel('left', 'Energy')
        self.energy_plot.setLabel('bottom', 'Frames')
        self.energy_plot.showGrid(x=True, y=True, alpha=0.25)

        self.ke_curve = self.energy_plot.plot(pen=pg.mkPen('#2ecc71', width=1.8), name="Kinetic Energy")
        self.pe_curve = self.energy_plot.plot(pen=pg.mkPen('#e74c3c', width=1.8), name="Potential Energy")
        self.te_curve = self.energy_plot.plot(pen=pg.mkPen('#f1c40f', width=2.2), name="Total Energy")

        # Panel 3: Velocity Distribution / Maxwell-Boltzmann (Row 1, Col 1)
        self.dist_plot = self.addPlot(title="Maxwell-Boltzmann Speed Distribution", row=1, col=1)
        self.dist_plot.setLabel('left', 'Probability Density')
        self.dist_plot.setLabel('bottom', 'Speed ||v||')
        self.dist_plot.showGrid(x=True, y=True, alpha=0.25)

        self.hist_curve = self.dist_plot.plot(stepMode="center", fillLevel=0, brush=pg.mkBrush(52, 152, 219, 100), pen=pg.mkPen('#3498db', width=1.5))
        self.maxwell_curve = self.dist_plot.plot(pen=pg.mkPen('#e67e22', width=2.0, style=Qt.PenStyle.DashLine), name="Maxwell Fit")

    def update_boundary_display(self, engine):
        """Updates boundary visual curve based on current engine geometry."""
        r = engine.boundary_radius
        self.sandbox_plot.setXRange(-r * 1.05, r * 1.05)
        self.sandbox_plot.setYRange(-r * 1.05, r * 1.05)

        theta = np.linspace(0, 2 * np.pi, 250)
        bx = r * np.cos(theta)
        by = r * np.sin(theta)
        self.boundary_item.setData(bx, by)
