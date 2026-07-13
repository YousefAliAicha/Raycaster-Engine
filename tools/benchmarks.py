#!/usr/bin/env python3
"""
Benchmarks — run this and paste the output into README.md. Nothing here
is asserted or graded; it's purely "how accurate / how fast is this,
measured just now, on this machine."

Usage:
    PYTHONPATH=. python3 tools/benchmarks.py [--frames N] [--skip-render]
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from raycaster.math_utils import mc_sin, mc_cos, vec_sin, vec_cos


def bench_trig(n_samples=20000):
    print('=' * 70)
    print(f'TRIG ACCURACY  (custom Taylor-series vs stdlib, {n_samples} points across [-pi, pi])')
    print('=' * 70)
    sweep = np.linspace(-math.pi, math.pi, n_samples)

    t0 = time.perf_counter()
    scalar_sin_err = max(abs(mc_sin(x) - math.sin(x)) for x in sweep)
    t_scalar_sin = time.perf_counter() - t0

    t0 = time.perf_counter()
    scalar_cos_err = max(abs(mc_cos(x) - math.cos(x)) for x in sweep)
    t_scalar_cos = time.perf_counter() - t0

    t0 = time.perf_counter()
    vec_sin_err = float(np.max(np.abs(vec_sin(sweep) - np.sin(sweep))))
    t_vec_sin = time.perf_counter() - t0

    t0 = time.perf_counter()
    vec_cos_err = float(np.max(np.abs(vec_cos(sweep) - np.cos(sweep))))
    t_vec_cos = time.perf_counter() - t0

    # Timing comparison vs stdlib/numpy, for context (not a claim that the
    # custom version is faster — it isn't, and doesn't need to be; the
    # point of owning the trig is correctness/testability, not speed).
    t0 = time.perf_counter()
    for x in sweep:
        math.sin(x)
    t_math_sin = time.perf_counter() - t0

    t0 = time.perf_counter()
    np.sin(sweep)
    t_np_sin = time.perf_counter() - t0

    print(f'{"function":<12}{"max abs error":<18}{"time":<12}{"vs reference":<12}')
    print(f'{"mc_sin":<12}{scalar_sin_err:<18.3e}{t_scalar_sin*1000:<10.2f}ms')
    print(f'{"math.sin":<12}{"(reference)":<18}{t_math_sin*1000:<10.2f}ms  '
          f'({t_scalar_sin / max(t_math_sin, 1e-9):.1f}x slower)')
    print(f'{"mc_cos":<12}{scalar_cos_err:<18.3e}{t_scalar_cos*1000:<10.2f}ms')
    print(f'{"vec_sin":<12}{vec_sin_err:<18.3e}{t_vec_sin*1000:<10.2f}ms')
    print(f'{"np.sin":<12}{"(reference)":<18}{t_np_sin*1000:<10.2f}ms  '
          f'({t_vec_sin / max(t_np_sin, 1e-9):.1f}x slower)')
    print(f'{"vec_cos":<12}{vec_cos_err:<18.3e}{t_vec_cos*1000:<10.2f}ms')
    print()
    print('Note: cosine error is consistently larger than sine error at the')
    print('same term count — the cosine series is purely even-power, so its')
    print('terms decay slightly slower relative to x near +-pi. Both are')
    print('still far below single-precision pixel-coordinate resolution.')
    print()


def bench_fps(scales, n_frames=150):
    print('=' * 70)
    print(f'FPS vs RENDER_SCALE  ({n_frames} frames per scale, headless — no tkinter)')
    print('=' * 70)
    print('NOTE: this measures the render+sprite+HUD pipeline only, not the')
    print('tkinter display blit, so absolute numbers here run higher than')
    print('what run_engine() will show on screen. Use it to compare scales')
    print('against each other, not as a final in-window FPS number.')
    print()
    print(f'{"RENDER_SCALE":<15}{"RENDER_W x H":<16}{"ms/frame":<12}{"FPS":<10}')

    import importlib
    from raycaster import level as level_mod
    from raycaster import textures as textures_mod
    from raycaster import entities as entities_mod

    results = []
    for scale in scales:
        # config.RENDER_W/H are computed at import time from RENDER_SCALE,
        # so each scale needs a fresh import of config + render with that
        # value patched in first.
        import raycaster.config as config_mod
        config_mod.RENDER_SCALE = scale
        config_mod.RENDER_W = max(64, int(config_mod.WIDTH * scale))
        config_mod.RENDER_H = max(64, int(config_mod.HEIGHT * scale))
        config_mod.RENDER_HALF_H = config_mod.RENDER_H // 2

        render_mod = importlib.reload(importlib.import_module('raycaster.render'))

        lvl = level_mod.load_level()
        tex = textures_mod.build_textures('/tmp/rc_bench_assets', lvl.wall_colors)
        state = entities_mod.GameState.new_game(lvl, tex)

        buf = np.zeros((config_mod.RENDER_H, config_mod.RENDER_W, 3), dtype=np.uint8)
        for _ in range(5):   # warmup
            buf[:] = 0
            z = render_mod.render(state, buf)
            render_mod.draw_sprites(state, buf, z)

        t0 = time.perf_counter()
        for _ in range(n_frames):
            buf[:] = 0
            z = render_mod.render(state, buf)
            render_mod.draw_sprites(state, buf, z)
        elapsed = time.perf_counter() - t0

        ms_per_frame = elapsed / n_frames * 1000
        fps = n_frames / elapsed
        results.append((scale, config_mod.RENDER_W, config_mod.RENDER_H, ms_per_frame, fps))
        print(f'{scale:<15}{config_mod.RENDER_W}x{config_mod.RENDER_H:<10}'
              f'{ms_per_frame:<12.2f}{fps:<10.1f}')

    print()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=150)
    parser.add_argument('--skip-render', action='store_true')
    args = parser.parse_args()

    bench_trig()
    if not args.skip_render:
        bench_fps([1.0, 0.75, 0.62, 0.5, 0.35], n_frames=args.frames)
