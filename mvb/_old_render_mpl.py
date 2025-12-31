import time
import numpy as np
import matplotlib.pyplot as plt


class MPLRenderer:
    def __init__(self, world, worm, fps: int):
        self.world = world
        self.worm = worm
        self.fps = max(1, int(fps))
        self.paused = False
        self.single_step = False
        self.stop_flag = False
        self.running = True  # <-- NEW: sim tells us when it stops

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title("Byte — p: pause/resume | n: step | r: reset | c: cancel")
        self.ax.set_axis_off()

        # Prepare RGB image buffer
        h, w = self.world.height, self.world.width
        self.img = np.ones((h, w, 3), dtype=np.float32)  # start white
        self.im = self.ax.imshow(
            self.img, interpolation="nearest", vmin=0.0, vmax=1.0
        )

        # --- HUD text (two lines) ---
        self.text_sense = self.ax.text(
            0.01, -0.02, "", transform=self.ax.transAxes, ha="left", va="top"
        )

        self.text_status = self.ax.text(
            0.01, -0.07, "", transform=self.ax.transAxes, ha="left", va="top"
        )

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def on_key(self, event):
        k = (event.key or "").lower()
        if k == "p":
            self.paused = not self.paused
        elif k == "n":
            self.single_step = True
            self.paused = True
        elif k == "r":
            self.paused = True
            self.stop_flag = True
        elif k == "c":
            self.stop_flag = True

    def energy_to_color(self, energy, cap):
        # green (full) -> red (empty)
        if cap <= 0:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        frac = np.clip(energy / cap, 0.0, 1.0)
        return np.array([1.0 - frac, frac, 0.0], dtype=np.float32)

    def draw(self):
        # rebuild the RGB frame
        self.img[:] = 1.0  # white

        # food = black
        food_mask = self.world.food > 0
        self.img[food_mask] = [0.0, 0.0, 0.0]

        # worm overwrites cell color
        wy, wx = self.worm.y, self.worm.x
        color = self.energy_to_color(
            self.worm.energy, self.worm.cfg.energy_capacity
        )
        self.img[wy, wx] = color

        self.im.set_data(self.img)

        # --- sensory HUD ---
        sense = getattr(self.worm, "sensory_information", {})
        if sense:
            sense_str = "  ".join(f"{k}:{v}" for k, v in sense.items())
        else:
            sense_str = "no sensory data"
        self.text_sense.set_text(f"SENSE | {sense_str}")

        # --- status HUD ---
        if not self.worm.alive:
            state = "DEAD"
        elif not self.running:
            state = "STOPPED"
        elif self.paused:
            state = "PAUSED"
        else:
            state = "RUNNING"

        self.text_status.set_text(
            f"tick={self.worm.ticks}  "
            f"energy={self.worm.energy}  "
            f"eats={self.worm.eats}  "
            f"dist={self.worm.distance}  "
            f"{state}"
        )

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def wait_frame(self):
        time.sleep(1.0 / self.fps)
