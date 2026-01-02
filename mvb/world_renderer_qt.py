# ------------------------------------------------------------
# world_renderer_qt.py
# Pure visualization renderer for Byte simulation using Qt
#
# This renderer is a "dumb display" that:
# - Shows the current world state when draw() is called
# - Reads world + worm state
#
# It does NOT:
# - Control simulation flow
# - Change any simulation variables
# - Advance rng
# ------------------------------------------------------------

import sys
import time
import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QFont


class QtRenderer(QMainWindow):
    """
    Pure visualization renderer for the Byte simulation.
    """

    def __init__(self, world, worm, fps: int):
        """
        Args:
            world: World object (read-only)
            worm: Worm object (read-only)
            fps: Target frames per second for wait_frame()
        """
        self.world = world
        self.worm = worm
        self.fps = max(1, int(fps))

        # Initialize Qt application if needed
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        super().__init__()
        self.setWindowTitle("Byte Simulation")

        self._setup_ui()

        self.show()
        self.raise_()
        self.activateWindow()

        # Ensure window appears immediately
        self.app.processEvents()

    # ------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)

        # World image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "background-color: white; border: 1px solid black;"
        )
        layout.addWidget(self.image_label)

        # Sensory info
        self.sense_label = QLabel("SENSE | initializing...")
        self.sense_label.setFont(QFont("Monospace", 10))
        self.sense_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        layout.addWidget(self.sense_label)

        # Status info
        self.status_label = QLabel("tick=0 energy=0 eats=0 dist=0")
        self.status_label.setFont(QFont("Monospace", 10))
        self.status_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        layout.addWidget(self.status_label)

        self.resize(650, 750)

    # ------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------

    def energy_to_color(self, energy: float, capacity: float) -> tuple[int, int, int]:
        """
        Convert energy level to RGB color.
        Empty = red, full = green.
        """
        if capacity <= 0:
            return (255, 0, 0)

        frac = np.clip(energy / capacity, 0.0, 1.0)
        r = int(255 * (1.0 - frac))
        g = int(255 * frac)
        b = 0
        return (r, g, b)

    # ------------------------------------------------------------
    # Main draw call
    # ------------------------------------------------------------

    def draw(self):
        """
        Render the current state.
        Pure function with side effects only in the UI.
        """
        # Process pending Qt events
        self.app.processEvents()

        h, w = self.world.height, self.world.width

        # White background
        img = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Food (black)
        food_mask = self.world.food > 0
        img[food_mask] = (0, 0, 0)

        # Worm
        wy, wx = self.worm.y, self.worm.x
        color = self.energy_to_color(
            self.worm.energy,
            self.worm.cfg.energy_capacity,
        )
        img[wy, wx] = color

        # Convert to QImage
        qimg = QImage(
            img.data,
            w,
            h,
            3 * w,
            QImage.Format_RGB888,
        )

        scale = 8
        pixmap = QPixmap.fromImage(qimg).scaled(
            w * scale,
            h * scale,
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )

        self.image_label.setPixmap(pixmap)

        # Sensory info
        sense = getattr(self.worm, "sensory_information", {})
        if sense:
            sense_str = " ".join(f"{k}:{v}" for k, v in sense.items())
        else:
            sense_str = "no sensory data"
        self.sense_label.setText(f"SENSE | {sense_str}")

        # Status info (purely factual)
        state = "DEAD" if not self.worm.alive else "ALIVE"
        self.status_label.setText(
            f"tick={self.worm.ticks} "
            f"energy={self.worm.energy} "
            f"eats={self.worm.eats} "
            f"dist={self.worm.distance} "
            f"{state}"
        )

        # Final event processing for smooth UI
        self.app.processEvents()

    # ------------------------------------------------------------
    # Frame pacing
    # ------------------------------------------------------------

    def wait_frame(self):
        """Sleep to maintain target FPS."""
        time.sleep(1.0 / self.fps)

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------

    def close(self):
        super().close()
        self.app.quit()
