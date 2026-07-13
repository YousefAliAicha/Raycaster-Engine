# Raycaster — A Wolfenstein-Style Engine Built From Scratch

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-blue?style=flat)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![2.5D Renderer](https://img.shields.io/badge/2.5D-Renderer-lightgrey?style=flat)
![Raycaster](https://img.shields.io/badge/Raycaster-Engine-orange?style=flat)
![Doom](https://img.shields.io/badge/Inspired%20by-Doom-8B0000?style=flat)

![Gameplay demo](assets/demo.gif)

A 3D raycaster in Python — NumPy for the math, tkinter for the window, no game engine, no OpenGL, and no `math.sin`/`math.cos` in the hot path. The render path is fully vectorised: zero per-column Python loops anywhere, from wall rasterisation through sprite compositing.

---

## Contents

- [Features](#features)
- [Benchmarks](#benchmarks)
- [How it works](#how-it-works)
- [Engineering decisions](#engineering-decisions)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Level data format](#level-data-format)

---

## Features

**Rendering**

- Vectorised DDA wall casting, floor/ceiling projection, and sprite billboarding, all done as single NumPy gather/broadcast operations over the whole frame
- Distance-based lighting falloff and fog blending, visible over the full length of the showcase level's corridor
- Doors rendered with their own texture and slide animation, distinct from static walls

**Gameplay**

- Collision with axis-separated sliding, so the player doesn't stop dead against a wall at an angle
- Shootable sprites — barrels and persons, hit-scanned against the camera plane
- A key pickup, a locked door, and an exit flag, tied together into one guided level

**HUD**

![HUD close-up](assets/hud.png)

- Mini-map with live door-state coloring (locked / closed / open)
- Debug panel: position, direction, forward wall distance, FPS
- Weapon sprite with head-bob and firing recoil, plus a muzzle flash sprite (not a flat color blend)

---

## Benchmarks

Run these yourself with `PYTHONPATH=. python3 tools/benchmarks.py`. Numbers below are a real run, not projections — machine was an AMD Ryzen 7 6800H, Python 3.11.15, NumPy 2.4.6, Windows. Expect different absolute numbers on different hardware; the shape of the results should hold everywhere.

**Trig accuracy** (20,000-point sweep across `[-pi, pi]`, vs `math`/`numpy`):

| Function | Max abs error | Time (20k calls) | vs stdlib |
|---|---|---|---|
| `mc_sin` (scalar) | 5.289e-10 | 63.11 ms | 37.7x slower than `math.sin` |
| `mc_cos` (scalar) | 3.529e-9 | 62.97 ms | — |
| `vec_sin` (vectorised) | 5.289e-10 | 0.98 ms | 7.9x slower than `np.sin` |
| `vec_cos` (vectorised) | 3.529e-9 | 0.79 ms | — |

Cosine's error is consistently about 7x larger than sine's at the same term count — its series is purely even-power, so convergence is marginally slower near the interval edges. Both are still five to nine orders of magnitude below anything visible at pixel resolution. The vectorised forms are what actually run in the rotation hot path; the slowdown against stdlib is the accepted cost of owning and testing the numerics instead of trusting an opaque libm call.

**FPS vs `RENDER_SCALE`** (render + sprite pass only, headless, 150 frames per scale — this measures the pipeline in isolation, not the tkinter display blit, so in-window FPS runs somewhat lower):

| RENDER_SCALE | Internal resolution | ms/frame | FPS |
|---|---|---|---|
| 1.00 | 640x408 | 24.95 | 40.1 |
| 0.75 | 480x306 | 13.95 | 71.7 |
| 0.62 (default) | 396x252 | 10.28 | 97.3 |
| 0.50 | 320x204 | 7.03 | 142.3 |
| 0.35 | 224x142 | 3.68 | 271.7 |

FPS scales close to linearly with pixel count (`RENDER_W * RENDER_H`), which tracks with the dominant cost being the full-frame wall-color gather in `render.py` — it does the same amount of work per pixel regardless of scale. `RENDER_SCALE = 0.62` is the default because it's roughly where softening from the nearest-neighbor upscale starts to show, and the chunky-pixel look at lower scales is period-appropriate for the genre anyway — this setting is as much an aesthetic call as a performance one.

---

## How it works

### Data flow

`controls.apply_input` mutates `Player` inside `GameState` → `entities.update_*` advances doors/pickups/win/bob → `render.render` and `render.draw_sprites` fill the internal-resolution buffer → `main` upscales it (nearest-neighbor) to the window size → `hud.draw_hud` composites the UI on top → one `PhotoImage.paste()` call pushes it to the tkinter canvas.

### Why a hand-rolled Taylor series instead of `math.sin`

`math_utils.py` reimplements `sin`/`cos` from their Taylor series rather than calling `math.sin`/`np.sin`. This isn't a speed decision — libm's sin/cos will always be at least as fast, per the benchmark above — it's a correctness-ownership one. Two real bugs shipped in earlier versions specifically because the trig was hand-rolled:

1. `mc_cos` originally contained a Newton-Raphson square-root iteration copy-pasted from elsewhere in the file, not a cosine approximation at all. It ran without raising and returned plausible-looking floats — the kind of bug that survives casual testing and only shows up as "the camera rotation feels subtly wrong."
2. `vec_sin` had `t = X.copy` (missing the call parentheses, so `t` bound to the method object, not an array) alongside a normalized array `X` that was computed but never read in the loop. Results only happened to be correct for inputs already inside `[-pi, pi]`.

Both are permanently guarded by `tests/test_trig.py`, which checks all four functions against `math`/`numpy` reference values across a 2,000-point sweep, plus explicit regression cases matching each bug's exact failure mode.

### The showcase level

`data/level1.json` is laid out as a short guided tour rather than a maze, so each engine feature gets its own space:

| Room | Showcases |
|---|---|
| Spawn Hall | Basic textured walls, starting orientation |
| Lighting & Fog Hall | A long straight corridor — brightness falloff and fog blending are visible over its full length |
| Sprite Gallery | Persons and barrels, all shootable |
| Key Alcove | The key pickup, off the main path |
| Vault (behind the locked door) | The door mechanic end to end — find key, unlock, slide-open animation, gating the exit flag |

Vault connectivity is checked with a BFS over the level grid (`tools/verify_connectivity.py`), not assumed: with every door treated as closed, 261 cells are reachable from spawn; with every door treated as open, 310 are. The 49-cell gap is exactly the vault, confirming it's genuinely gated and not reachable some other way through the grid.

The door retracts upward when opened, bottom edge receding first, animated by shrinking the drawn vertical range as `open_amount` increases (`render.render`). The door texture's rib pattern runs vertically rather than matching that direction of travel — a known cosmetic mismatch, not a gameplay bug, left as is for now.

---

## Engineering decisions

A short log of the calls that shaped this project, kept as it happened rather than cleaned up after the fact.

- **The trig was hand-rolled on purpose, and it caught real bugs.** See [Why a hand-rolled Taylor series](#why-a-hand-rolled-taylor-series-instead-of-mathsin). Both bugs ran silently, no exception, no crash, which is exactly the failure mode a regression test suite exists to catch.
- **Vault connectivity is verified, not asserted.** The BFS in `tools/verify_connectivity.py` gives an exact reachable-cell count with doors open vs closed, rather than trusting the level layout does what it looks like it does.
- **Shooting persons was a real fix, not a design choice from the start.** An earlier version only made barrels shootable; `config.SHOOTABLE_TYPES` now covers both, and `tests/test_gameplay.py::test_shooting_hits_barrels_and_persons` checks both explicitly rather than just one.
- **The door's texture and animation direction don't currently agree**, and that's stated plainly above rather than left for someone to notice on their own. Fixing it means either changing the rib orientation back or reworking the slide direction — neither has been done yet.
- **Benchmark numbers are tied to specific hardware and software versions**, not presented as universal. Anyone re-running `tools/benchmarks.py` on different hardware should expect different absolute numbers, though the relative shape — vectorised beating scalar, FPS scaling with pixel count — should hold.

---

## Project structure

```
raycaster/
  math_utils.py   Custom Taylor-series sin/cos (scalar + vectorised) and
                   the 2D rotation matrix built from them.
  config.py        Every tunable constant in one place (resolution,
                   speeds, lighting falloff, fog, shoot/door timings).
  level.py         Loads a level JSON into runtime structures: the wall
                   grid, DoorState objects, sprite/pickup placement, spawn
                   pose. Also owns collision testing and the per-frame
                   door-aware hit-map used by the ray caster.
  textures.py      Procedural pixel-art generation (walls, floor/ceiling,
                   sprites, weapon, muzzle flash) plus load-or-generate-
                   and-cache helpers.
  entities.py      Player (movement/collision/rotation), GameState (the
                   one mutable object gameplay functions take instead of
                   scattered globals), doors, pickups, win condition,
                   shooting.
  render.py        The renderer: floor/ceiling projection, vectorised DDA
                   wall casting, lighting/fog, and sprite billboarding —
                   all at the internal render resolution.
  hud.py           Everything drawn at full window resolution on top of
                   the upscaled scene: mini-map, debug panel, weapon with
                   recoil, muzzle flash, prompts, win screen.
  controls.py      Keyboard/mouse state and per-frame input application.
  main.py          tkinter window setup and the render loop. Wires every
                   other module together; owns no game logic itself.
  data/level*.json Level layout and entity placement.
  assets_cache/    Procedurally generated textures/sprites, cached on
                   first run. Gitignored — delete this folder any time
                   to force a clean regeneration.
tests/
  test_trig.py      Regression tests for the custom trig.
  test_gameplay.py  Headless tests for collision, doors, shooting, win
                    condition — no tkinter window required.
tools/
  benchmarks.py           Trig accuracy and FPS vs RENDER_SCALE. Not
                          graded or asserted, just measured.
  verify_connectivity.py  BFS over the level grid, doors open vs closed,
                          confirming the vault is genuinely gated.
assets/
  demo.gif   Gameplay clip — movement, shooting, a door opening.
  hud.png    Close-up of the mini-map/debug panel/weapon HUD.
  wincon.png The win screen.
Raycaster.ipynb  Notebook walkthrough — loads a level, renders a frame
                 headlessly and displays it inline, and runs the trig
                 and connectivity checks without a tkinter window.
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Textures and sprites are procedurally generated on first run and cached to `raycaster/assets_cache/` — no assets to download.

---

## Usage

### Run the game

```bash
python -m raycaster
```

Click the window to capture the mouse and look around. Click again to shoot.

To try the second, much smaller level stub, demonstrating that level layout is pure data, not code:

```python
from raycaster.main import run_engine
run_engine('raycaster/data/level2.json')
```

![Win screen](assets/wincon.png)

### Run the tests

```bash
pip install pytest
PYTHONPATH=. python3 -m pytest tests/ -v
```

32 tests, all headless — `Player`/`Level`/`render` operate on plain NumPy buffers, no window created. Covers trig accuracy and both documented bug regressions, level loading, the full render pipeline, strafe direction (checked against the actual on-screen projection math, not just self-consistency), collision, door lock/unlock/toggle, shooting both sprite types, and the win condition. Verified passing 32/32 on Python 3.11.15, NumPy 2.4.6, pytest 9.1.1.

### Run the notebook

```bash
jupyter notebook Raycaster.ipynb
```

Run from the repo root, so the `raycaster` package resolves without setting `PYTHONPATH` manually. Loads a level, renders one frame headlessly, and displays it inline with matplotlib — a way to look at what the renderer produces without launching the full game window.

---

## Level data format

Map layout, wall colors, door placement, sprites, key pickups, and the exit all live in `data/level*.json` — none of it is hardcoded in Python. To build a new level, write a new JSON file in the same shape (see `level.py::Level.__init__` for the exact schema) and pass its path to `run_engine(level_path=...)`. Grid cell values: `0` is floor, `1`/`2`/`3`/`5` are wall variants (see `wall_colors`), `4` is a door — listed explicitly in `doors` for lock state, or left unlocked by default if just placed in the grid.

---

## License

MIT License — see [LICENSE](LICENSE) for full terms.
