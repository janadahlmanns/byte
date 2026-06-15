# ============================================================
# Pause Manager: Handles pause/resume/step/exit functionality
# ============================================================
#
# Pure state holder — no threads, no pynput.
# Qt renderers update the flags via keyPressEvent;
# simulate_run() polls check_pause() each tick.
# ============================================================

import threading
import time
from typing import Optional, Callable


class PauseManagerExit(Exception):
    """Raised when user requests exit via pause manager."""
    pass


class PauseManager:
    """
    Manages pause/resume/step/exit of simulation via shared flags.

    Key bindings (handled by Qt renderers, not here):
    - 'p': Toggle pause/resume
    - 'n': Step forward one tick (only when paused)
    - 'c': Cancel and exit simulation
    """

    def __init__(self):
        self._paused = False
        self._step_requested = False
        self._exit_requested = False
        self._lock = threading.Lock()

    def check_pause(self, process_events: Optional[Callable] = None):
        """
        Call this at each pause checkpoint.

        If paused, blocks execution until user resumes ('p'), steps ('n'),
        or exits ('c').  While blocked the optional *process_events*
        callable is invoked each iteration so that the Qt event loop keeps
        delivering key-press events to the renderers.

        Args:
            process_events: Optional callable (e.g. ``app.processEvents``)
                            invoked during the pause spin-loop to keep the
                            UI responsive.

        Raises:
            PauseManagerExit: When the user requests exit.
        """
        with self._lock:
            if self._exit_requested:
                raise PauseManagerExit("Simulation exited via pause manager.")

            if not self._paused:
                return

            if self._step_requested:
                self._step_requested = False
                return

        # Paused and no step — block here until something changes.
        while True:
            if process_events is not None:
                process_events()

            with self._lock:
                if self._exit_requested:
                    raise PauseManagerExit("Simulation exited via pause manager.")
                if self._step_requested:
                    self._step_requested = False
                    return
                if not self._paused:
                    return

            # Brief sleep to avoid busy-waiting.
            time.sleep(0.01)

    def is_paused(self) -> bool:
        """Check if simulation is currently paused."""
        with self._lock:
            return self._paused

    def should_exit(self) -> bool:
        """Check if exit was requested."""
        with self._lock:
            return self._exit_requested

    def cleanup(self):
        """Reset state."""
        with self._lock:
            self._paused = False
            self._step_requested = False
            self._exit_requested = False


# Global instance
_instance: Optional[PauseManager] = None


def init_pause_manager() -> PauseManager:
    """Initialize and return the global pause manager instance."""
    global _instance
    if _instance is None:
        _instance = PauseManager()
    return _instance


def get_pause_manager() -> PauseManager:
    """Get the global pause manager instance."""
    global _instance
    if _instance is None:
        raise RuntimeError("Pause manager not initialized. Call init_pause_manager() first.")
    return _instance


def cleanup_pause_manager():
    """Clean up the pause manager."""
    global _instance
    if _instance:
        _instance.cleanup()
        _instance = None
