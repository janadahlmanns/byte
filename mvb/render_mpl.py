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

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title("MVB Worm — p: pause/resume | n: step | r: reset | s: stop")
        self.ax.set_axis_off()

        # Prepare RGB image buffer
        h, w = self.world.height, self.world.width
        self.img = np.ones((h, w, 3), dtype=np.float32)  # start white
        self.im = self.ax.imshow(self.img, interpolation="nearest", vmin=0.0, vmax=1.0)

        self.text = self.ax.text(
            0.01, -0.02, "", transform=self.ax.transAxes, ha="left", va="top"
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
            # Signal to reset handled externally by sim; here just pause
            self.paused = True
            self.stop_flag = True  # sim will interpret as reset if it wants
        elif k == "s":
            self.stop_flag = True

    def energy_to_color(self, energy, cap):
        # green (full) -> red (empty)
        if cap <= 0:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        frac = np.clip(energy / cap, 0.0, 1.0)
        # simple linear: red = 1-frac, green = frac
        return np.array([1.0 - frac, frac, 0.0], dtype=np.float32)

    def draw(self):
        # rebuild the RGB frame
        self.img[:] = 1.0  # white
        # food = black
        food_mask = self.world.food > 0
        self.img[food_mask] = [0.0, 0.0, 0.0]

        # worm overwrites cell color
        wy, wx = self.worm.y, self.worm.x
        color = self.energy_to_color(self.worm.energy, self.worm.cfg.energy_capacity)
        if 0 <= wy < self.world.height and 0 <= wx < self.world.width:
            self.img[wy, wx] = color

        self.im.set_data(self.img)

        # HUD
        self.text.set_text(
            f"tick={self.worm.ticks}  energy={self.worm.energy}  eats={self.worm.eats}  dist={self.worm.distance}  "
            f"{'PAUSED' if self.paused else 'RUNNING'}"
        )
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def wait_frame(self):
        time.sleep(1.0 / self.fps)
