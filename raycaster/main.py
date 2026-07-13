# =============================================================================
# main.py — window setup and the tkinter render loop. Everything else
# (math, rendering, entities/gameplay, HUD, controls) is imported in as a
# module; this file just wires them together and owns the event loop.
# =============================================================================

import time
from pathlib import Path

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from . import config
from . import controls
from . import hud
from . import level as level_mod
from . import render as render_mod
from . import textures as textures_mod
from .entities import GameState, update_bob, update_doors, update_pickups, update_win, try_shoot

ASSET_DIR = Path(__file__).parent / 'assets_cache'


def run_engine(level_path=None):
    """
    Initialise the tkinter window and run the main render loop. The loop
    is driven by tkinter's after() scheduler (1 ms delay) rather than a
    blocking while-loop so the GUI event queue stays responsive.
    """
    lvl = level_mod.load_level(level_path)
    tex = textures_mod.build_textures(ASSET_DIR, lvl.wall_colors)
    state = GameState.new_game(lvl, tex)
    state.minimap = hud.build_minimap(lvl)

    render_buf = np.zeros((config.RENDER_H, config.RENDER_W, 3), dtype=np.uint8)
    frame      = np.zeros((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)

    root = tk.Tk()
    root.title(f'Raycaster — {lvl.name}  |  WASD / Arrows  |  Q = quit')
    root.resizable(False, False)

    canvas = tk.Canvas(root, width=config.WIDTH, height=config.HEIGHT,
                        bg='black', highlightthickness=0)
    canvas.pack()

    root.bind('<KeyPress>',   controls.on_key_press)
    root.bind('<KeyRelease>', controls.on_key_release)

    # ── Mouse look (emulated pointer-lock: hide + re-centre on motion) ─────
    center_x, center_y = config.WIDTH // 2, config.HEIGHT // 2

    def _capture_mouse(_event=None):
        if controls.MOUSE_CAPTURED[0]:
            try_shoot(state, time.perf_counter())
            return
        controls.MOUSE_CAPTURED[0] = True
        canvas.config(cursor='none')
        canvas.event_generate('<Motion>', warp=True, x=center_x, y=center_y)

    def _release_mouse():
        controls.MOUSE_CAPTURED[0] = False
        canvas.config(cursor='')

    def _on_motion(event):
        if not controls.MOUSE_CAPTURED[0]:
            return
        dx = event.x - center_x
        if dx != 0:
            controls.MOUSE_DX[0] += dx
            canvas.event_generate('<Motion>', warp=True, x=center_x, y=center_y)

    canvas.bind('<Button-1>', _capture_mouse)
    canvas.bind('<Motion>', _on_motion)

    # One PhotoImage, updated in place via .paste() every frame — see
    # README "Performance" section for why this matters.
    photo = ImageTk.PhotoImage(Image.fromarray(frame, 'RGB'))
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    fps_val = [0]
    fps_cnt = [0]
    fps_t   = [time.perf_counter()]
    last_t  = [time.perf_counter()]

    def loop():
        now = time.perf_counter()
        dt  = now - last_t[0]
        last_t[0] = now

        if not controls.apply_input(state, dt):
            _release_mouse()
            root.destroy()
            return

        update_bob(state, dt)
        update_doors(state, dt)
        update_pickups(state)
        update_win(state)
        if state.hud_message_t > 0:
            state.hud_message_t -= dt
        if state.muzzle_flash_t > 0:
            state.muzzle_flash_t -= dt
        if state.recoil_t > 0:
            state.recoil_t -= dt

        render_buf[:] = 0
        z = render_mod.render(state, render_buf)
        render_mod.draw_sprites(state, render_buf, z)

        upscaled = Image.fromarray(render_buf, 'RGB').resize(
            (config.WIDTH, config.HEIGHT), Image.NEAREST)
        frame[:, :, :] = np.asarray(upscaled)
        hud.draw_hud(state, frame, z, fps_val[0])

        photo.paste(Image.fromarray(frame, 'RGB'))

        fps_cnt[0] += 1
        elapsed = now - fps_t[0]
        if elapsed >= 0.5:
            fps_val[0] = int(fps_cnt[0] / elapsed)
            fps_cnt[0] = 0
            fps_t[0]   = now

        if canvas.winfo_exists():
            canvas.after(1, loop)

    canvas.after(0, loop)
    print('Click the window to focus, then use WASD / arrow keys. Click again to shoot.')
    root.mainloop()
    print('Engine stopped.')


if __name__ == '__main__':
    run_engine()
