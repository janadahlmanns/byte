# ------------------------------------------------------------
# render_qt.py
# Pure visualization renderer for Byte simulation using Qt
# 
# This renderer is a "dumb display" that:
# - Shows the current world state when draw() is called
# - Listens for keyboard shortcuts and sets flags
# - Does NOT control simulation logic or timing
# ------------------------------------------------------------

import time
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont
import sys


class QtRenderer(QMainWindow):
    """
    A pure visualization renderer for the Byte simulation.
    
    Responsibilities:
    - Display world grid with food (black) and worm (colored by energy)
    - Show two status lines: sensory info and simulation metrics
    - Listen for keyboard shortcuts and set flags for simulation to read
    
    Does NOT:
    - Control simulation loop or timing
    - Execute any simulation logic
    - Modify world or worm state
    """
    
    def __init__(self, world, worm, fps: int):
        """
        Initialize the Qt renderer.
        
        Args:
            world: World object with .food grid, .height, .width
            worm: Worm object with .x, .y, .energy, .cfg.energy_capacity, etc.
            fps: Target frames per second (only used for wait_frame timing)
        """
        # Store references to simulation objects (read-only!)
        self.world = world
        self.worm = worm
        self.fps = max(1, int(fps))
        
        # Control flags - simulation reads these
        self.paused = False      # True when user presses 'p' to pause
        self.single_step = False # True when user presses 'n' to step once
        self.stop_flag = False   # True when user presses 'r' (reset) or 'c' (cancel)
        self.running = True      # Simulation can set this to False when ending
        
        # Initialize Qt application if not already created
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
        
        # Setup the main window
        super().__init__()
        self.setWindowTitle("Byte Simulation — p:pause | n:step | r:reset | c:cancel")
        self._setup_ui()
        
        # Show the window and bring it to front
        self.show()
        self.raise_()
        self.activateWindow()
        
        # Force initial event processing to ensure window appears
        self.app.processEvents()
        
        
    def _setup_ui(self):
        """Create and layout all UI elements."""
        # Central widget and layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Image display area (the world grid)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: white; border: 1px solid black;")
        layout.addWidget(self.image_label)
        
        # Status text line 1: Sensory information
        self.sense_label = QLabel("SENSE | initializing...")
        self.sense_label.setFont(QFont("Monospace", 10))
        self.sense_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        layout.addWidget(self.sense_label)
        
        # Status text line 2: Simulation metrics
        self.status_label = QLabel("tick=0 energy=0 eats=0 dist=0 INITIALIZING")
        self.status_label.setFont(QFont("Monospace", 10))
        self.status_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        layout.addWidget(self.status_label)
        
        # Resize window to fit content nicely
        self.resize(650, 750)
        
    def keyPressEvent(self, event):
        """
        Handle keyboard shortcuts.
        Sets flags that the simulation reads - does NOT execute simulation logic.
        """
        key = event.key()
        
        if key == Qt.Key_P:
            # Toggle pause
            self.paused = not self.paused
            
        elif key == Qt.Key_N:
            # Single step: pause and request one step
            self.single_step = True
            self.paused = True
            
        elif key == Qt.Key_R:
            # Reset: pause and set stop flag
            # Simulation checks: if stop_flag and paused -> reset
            self.paused = True
            self.stop_flag = True
            
        elif key == Qt.Key_C:
            # Cancel: set stop flag without pausing
            # Simulation checks: if stop_flag and not paused -> exit
            self.stop_flag = True
            
        else:
            # Pass other keys to parent
            super().keyPressEvent(event)
    
    def energy_to_color(self, energy: float, capacity: float) -> tuple:
        """
        Convert energy level to RGB color.
        Full energy = green, empty energy = red.
        
        Args:
            energy: Current energy level
            capacity: Maximum energy capacity
            
        Returns:
            (r, g, b) tuple with values 0-255
        """
        if capacity <= 0:
            return (255, 0, 0)  # Red for invalid capacity
        
        # Calculate energy fraction (0.0 = empty, 1.0 = full)
        frac = np.clip(energy / capacity, 0.0, 1.0)
        
        # Interpolate: red (empty) -> green (full)
        r = int(255 * (1.0 - frac))
        g = int(255 * frac)
        b = 0
        
        return (r, g, b)
    
    def draw(self):
        """
        Update the display with current world and worm state.
        This is a pure rendering function - reads state but doesn't modify it.
        """
        # Process events FIRST to catch keyboard input
        self.app.processEvents()
        
        # Get world dimensions
        h, w = self.world.height, self.world.width
        
        # Create RGB image (height x width x 3 channels)
        # Start with white background
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Draw food as black pixels
        food_mask = self.world.food > 0
        img[food_mask] = [0, 0, 0]  # Black
        
        # Draw worm colored by energy level
        wy, wx = self.worm.y, self.worm.x
        worm_color = self.energy_to_color(
            self.worm.energy,
            self.worm.cfg.energy_capacity
        )
        img[wy, wx] = worm_color
        
        # Convert numpy array to Qt image
        # Note: QImage expects data in specific format
        qimg = QImage(
            img.data,
            w, h,
            3 * w,  # bytes per line
            QImage.Format_RGB888
        )
        
        # Scale up for better visibility (each grid cell -> 8x8 pixels)
        scale_factor = 8
        pixmap = QPixmap.fromImage(qimg).scaled(
            w * scale_factor,
            h * scale_factor,
            Qt.KeepAspectRatio,
            Qt.FastTransformation  # Use FastTransformation for speed (nearest neighbor)
        )
        
        # Update the image display
        self.image_label.setPixmap(pixmap)
        
        # Update sensory information line
        sense = getattr(self.worm, "sensory_information", {})
        if sense:
            sense_str = " ".join(f"{k}:{v}" for k, v in sense.items())
        else:
            sense_str = "no sensory data"
        self.sense_label.setText(f"SENSE | {sense_str}")
        
        # Update status line
        if not self.worm.alive:
            state = "DEAD"
        elif not self.running:
            state = "STOPPED"
        elif self.paused:
            state = "PAUSED"
        else:
            state = "RUNNING"
            
        self.status_label.setText(
            f"tick={self.worm.ticks} "
            f"energy={self.worm.energy} "
            f"eats={self.worm.eats} "
            f"dist={self.worm.distance} "
            f"{state}"
        )
        
        # Process Qt events (allows window to respond, update, handle keys)
        # We already did this at the start, but do it again to ensure smooth updates
        self.app.processEvents()
    
    def wait_frame(self):
        """
        Sleep to maintain target FPS.
        Call this after draw() to pace the simulation.
        """
        time.sleep(1.0 / self.fps)
    
    def close(self):
        """Clean up and close the window."""
        super().close()
        self.app.quit()