# ------------------------------------------------------------
# brain_renderer_qt.py
# Pure visualization renderer for Byte brain using Qt (PySide6)
#
# This renderer is a "dumb display" that:
# - Shows the current brain state when draw() is called
# - Listens for keyboard shortcuts and sets flags
# - Does NOT mutate brain/world/worm state (ever)
# ------------------------------------------------------------

import sys
import time
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# Brand colors (from Jana)
ACTIVE_BG = "#0B3D2E"    # dark green
INACTIVE_BG = "#D4AF37"  # gold
ACTIVE_FG = "#FFFFFF"
INACTIVE_FG = "#000000"
BORDER = "#000000"


def _source_label(src: Any) -> str:
    """
    Compact identifier for a connection source.
    - InputSource: has .key
    - Neuron: has .id
    - Fallback: class name
    """
    if hasattr(src, "key"):
        return str(getattr(src, "key"))
    if hasattr(src, "id"):
        return f"n{int(getattr(src, 'id'))}"
    return src.__class__.__name__


class NeuronBox(QFrame):
    """
    One neuron "box" widget. Dumb UI: it only formats what it's given.
    """
    def __init__(self, neuron_id: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.neuron_id = int(neuron_id)

        self.setFrameShape(QFrame.Box)
        self.setLineWidth(1)
        self.setStyleSheet(f"border: 1px solid {BORDER};")

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Monospace-ish for compact dense info
        self.font_title = QFont("Monospace", 10)
        self.font_body = QFont("Monospace", 9)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self.lbl_title = QLabel(f"Neuron {self.neuron_id}")
        self.lbl_title.setFont(self.font_title)
        self.lbl_title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.lbl_title)

        self.lbl_params = QLabel("thr=?  noise=?  tonic=?")
        self.lbl_params.setFont(self.font_body)
        self.lbl_params.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.lbl_params)

        self.lbl_act = QLabel("act=?  next=?")
        self.lbl_act.setFont(self.font_body)
        self.lbl_act.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.lbl_act)

        # Compact single-line incoming connections
        self.lbl_in = QLabel("in: (none)")
        self.lbl_in.setFont(self.font_body)
        self.lbl_in.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_in.setWordWrap(True)  # allows wrap if it gets too long
        layout.addWidget(self.lbl_in)

        # Start inactive styling
        self._set_active(False)

    def _set_active(self, active: bool):
        if active:
            bg, fg = ACTIVE_BG, ACTIVE_FG
        else:
            bg, fg = INACTIVE_BG, INACTIVE_FG

        # Border stays black; fill changes.
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: {bg};
                color: {fg};
                border: 1px solid {BORDER};
                border-radius: 2px;
            }}
            """
        )

    @staticmethod
    def _fmt_float(x: Any, nd: int = 3) -> str:
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return str(x)

    def update_from_neuron(self, neuron: Any, max_conn_display: int = 8):
        """
        Update visuals from a neuron-like object.
        Expected fields: id, threshold, noise_level, tonic_level, activity, next_activity, incoming (list of connections)
        """
        # Title
        nid = int(getattr(neuron, "id", self.neuron_id))
        self.lbl_title.setText(f"Neuron {nid}")

        thr = getattr(neuron, "threshold", "?")
        noi = getattr(neuron, "noise_level", "?")
        ton = getattr(neuron, "tonic_level", "?")
        self.lbl_params.setText(
            f"thr={self._fmt_float(thr)}  noise={self._fmt_float(noi)}  tonic={self._fmt_float(ton)}"
        )

        act = getattr(neuron, "activity", "?")
        nxt = getattr(neuron, "next_activity", "?")
        self.lbl_act.setText(
            f"act={self._fmt_float(act, nd=3)}  next={self._fmt_float(nxt, nd=3)}"
        )

        # Active if "old activation" (activity) is on
        is_active = False
        try:
            is_active = float(act) > 0.0
        except Exception:
            is_active = bool(act)

        self._set_active(is_active)

        incoming = getattr(neuron, "incoming", []) or []
        if not incoming:
            self.lbl_in.setText("in: (none)")
            return

        parts = []
        shown = 0
        for conn in incoming:
            if shown >= max_conn_display:
                break
            src = getattr(conn, "source", None)
            w = getattr(conn, "weight", "?")
            r = getattr(conn, "reliability", "?")
            parts.append(f"{_source_label(src)}:w={self._fmt_float(w)} r={self._fmt_float(r)}")
            shown += 1

        more = len(incoming) - shown
        if more > 0:
            parts.append(f"...(+{more})")

        self.lbl_in.setText("in: " + " | ".join(parts))


class BrainQtRenderer(QMainWindow):
    """
    Dumb brain visualizer. Reads state when draw() is called.

    Keyboard shortcuts (flags only):
    - p: toggle pause
    - n: single step (forces pause + one step request)
    - r: reset request (pause + stop_flag)
    - c: cancel request (stop_flag without pausing)
    """

    def __init__(self, fps: int = 30):
        self.fps = max(1, int(fps))

        # Control flags (simulation/brain code reads these)
        self.paused = False
        self.single_step = False
        self.stop_flag = False
        self.running = True

        # Qt app singleton
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        super().__init__()
        self.setWindowTitle("Byte Brain — p:pause | n:step | r:reset | c:cancel")

        self._boxes: Dict[int, NeuronBox] = {}

        self._setup_ui()

        self.show()
        self.raise_()
        self.activateWindow()
        self.app.processEvents()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(10, 10, 10, 10)

        # Top shortcuts bar
        self.shortcuts = QLabel("Byte Brain — p:pause | n:step | r:reset | c:cancel")
        self.shortcuts.setFont(QFont("Monospace", 10))
        self.shortcuts.setStyleSheet("background-color: #f0f0f0; padding: 6px;")
        root.addWidget(self.shortcuts)

        # Grid area
        grid_wrap = QWidget()
        grid = QGridLayout(grid_wrap)
        grid.setSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)

        # Create 11 boxes and place them in the requested layout:
        # Row 0: 0-4
        for i, col in zip(range(0, 5), range(0, 5)):
            box = NeuronBox(i)
            self._boxes[i] = box
            grid.addWidget(box, 0, col)

        # Row 1: neuron 10 centered under col 2
        box10 = NeuronBox(10)
        self._boxes[10] = box10
        grid.addWidget(box10, 1, 2)

        # Row 2: 5-9
        for i, col in zip(range(5, 10), range(0, 5)):
            box = NeuronBox(i)
            self._boxes[i] = box
            grid.addWidget(box, 2, col)

        root.addWidget(grid_wrap)

        # Bottom status rows
        self.sense_label = QLabel("SENSE | initializing...")
        self.sense_label.setFont(QFont("Monospace", 10))
        self.sense_label.setStyleSheet("background-color: #f0f0f0; padding: 6px;")
        root.addWidget(self.sense_label)

        self.decision_label = QLabel("DECISION | initializing...")
        self.decision_label.setFont(QFont("Monospace", 10))
        self.decision_label.setStyleSheet("background-color: #f0f0f0; padding: 6px;")
        root.addWidget(self.decision_label)

        self.resize(1050, 850)

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_P:
            self.paused = not self.paused

        elif key == Qt.Key_N:
            self.single_step = True
            self.paused = True

        elif key == Qt.Key_R:
            self.paused = True
            self.stop_flag = True

        elif key == Qt.Key_C:
            self.stop_flag = True

        else:
            super().keyPressEvent(event)

    def draw(
        self,
        state: Any,
        brain_tick: int,
        decision_status: str,
        sense: Optional[Dict[str, Any]] = None,
    ):
        """
        Update UI from current brain state.
        Pure render: reads state, never mutates.
        """
        self.app.processEvents()

        # Update all neuron boxes (if ids match)
        neurons = getattr(state, "neurons", []) or []
        by_id = {}
        for n in neurons:
            try:
                by_id[int(getattr(n, "id"))] = n
            except Exception:
                continue

        for nid, box in self._boxes.items():
            neuron = by_id.get(nid, None)
            if neuron is None:
                # If missing, leave it as is but keep label accurate
                box.lbl_title.setText(f"Neuron {nid} (missing)")
                continue
            box.update_from_neuron(neuron)

        # Sense row
        if sense is None:
            sense = {}

        if sense:
            sense_str = " ".join(f"{k}:{v}" for k, v in sense.items())
        else:
            sense_str = "no sensory data"

        self.sense_label.setText(f"SENSE | {sense_str}")

        # Decision row
        self.decision_label.setText(f"DECISION | beat={brain_tick} | {decision_status}")

        self.app.processEvents()

    def wait_frame(self):
        time.sleep(1.0 / self.fps)

    def close(self):
        super().close()
        self.app.quit()
