# ============================================================
# Pause Manager: Handles pause/resume/step/exit functionality
# ============================================================

import threading
from typing import Optional

try:
    from pynput import keyboard
except ImportError:
    raise ImportError(
        "pynput is required for pause functionality. "
        "Install it with: pip install pynput"
    )


class PauseManagerExit(Exception):
    """Raised when user requests exit via pause manager."""
    pass


class PauseManager:
    """
    Manages pause/resume/step/exit of simulation via keyboard input.
    
    Key bindings:
    - 'p': Toggle pause/resume
    - 'n': Step forward one checkpoint (only when paused)
    - 'c': Cancel and exit simulation
    """
    
    def __init__(self):
        self._paused = False
        self._step_requested = False
        self._exit_requested = False
        self._lock = threading.Lock()
        self._listener: Optional[keyboard.Listener] = None
        self._start_listener()
    
    def _start_listener(self):
        """Start the background keyboard listener thread."""
        def on_press(key):
            try:
                char = key.char
            except AttributeError:
                # Handle special keys (they don't have .char attribute)
                return
            
            if char == 'p':
                with self._lock:
                    self._paused = not self._paused
                    if self._paused:
                        print("\n[PAUSED] Press 'n' to step, 'p' to resume, or 'c' to cancel.")
                    else:
                        print("\n[RESUMED]")
                        self._step_requested = False  # Clear any pending steps when resuming
            
            elif char == 'n':
                with self._lock:
                    if self._paused:
                        self._step_requested = True
                        print("[STEP] Advancing one checkpoint...")
            
            elif char == 'c':
                with self._lock:
                    self._exit_requested = True
                    print("\n[EXIT] Stopping simulation...")
        
        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()
    
    def check_pause(self):
        """
        Call this at each pause checkpoint.
        
        If paused, blocks execution until user resumes (presses 'p') or exits (presses 'c').
        If exit is requested, raises PauseManagerExit exception.
        Otherwise returns normally (simulation continues).
        """
        with self._lock:
            if self._exit_requested:
                raise PauseManagerExit("Simulation exited via pause manager.")
            
            if self._paused:
                if self._step_requested:
                    # Execute one step
                    self._step_requested = False
                    return
                else:
                    # Stay paused, don't return yet
                    pass
        
        # If paused and no step requested, block here
        while True:
            with self._lock:
                if self._exit_requested:
                    raise PauseManagerExit("Simulation exited via pause manager.")
                if self._step_requested:
                    self._step_requested = False
                    return
                if not self._paused:
                    return
            
            # Sleep briefly to avoid busy-waiting and let keyboard listener work
            threading.Event().wait(0.01)
    
    def is_paused(self) -> bool:
        """Check if simulation is currently paused."""
        with self._lock:
            return self._paused
    
    def should_exit(self) -> bool:
        """Check if exit was requested."""
        with self._lock:
            return self._exit_requested
    
    def cleanup(self):
        """Stop the keyboard listener."""
        if self._listener:
            self._listener.stop()
            self._listener.join(timeout=1.0)


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
