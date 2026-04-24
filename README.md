# 🎮 Python Raycaster

A Wolfenstein-style 3D raycaster built entirely in Python using NumPy and tkinter — no game engine, no OpenGL. Rays are cast per screen column using a vectorised DDA algorithm, with textured walls, floor/ceiling projection, distance fog, sprite rendering, and a first-person weapon bob.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-required-orange)
![Pillow](https://img.shields.io/badge/Pillow-required-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Features

- Vectorised DDA ray marching across all screen columns simultaneously
- Textured walls (procedurally generated brick patterns, colour-coded by type)
- Textured floor and ceiling projection
- Distance-based lighting with exponential fall-off
- Side-face darkening for a cheap directional-light effect
- Distance fog blending toward a horizon colour
- Billboarded sprite rendering with z-buffer occlusion
- Head-bob weapon animation driven by a custom sine implementation
- Mini-map overlay with player dot and direction arrow
- Debug HUD (position, direction, wall distance, FPS)
- Zero external game-engine dependencies

---

## Project Structure

```
raycaster/
├── Raycaster.py          # Cleaned, commented release version
├── Raycaster_raw.py      # Original source (unmodified)
├── Raycaster.ipynb       # Jupyter notebook — annotated walkthrough
├── requirements.txt      # Python dependencies
├── README.md
├── LICENSE
├── textures/             # Auto-generated on first run
├── floor/                # Auto-generated on first run
├── sprites/              # Auto-generated on first run
└── assets/               # Auto-generated on first run
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/raycaster.git
cd raycaster
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> tkinter ships with most Python installers. If it's missing on Linux:
> `sudo apt install python3-tk`

### 3. Run

```bash
python Raycaster.py
```

Texture files are generated automatically on the first run and saved to `textures/`, `floor/`, `sprites/`, and `assets/`.

---

## Controls

| Key | Action |
|-----|--------|
| `W` / `↑` | Move forward |
| `S` / `↓` | Move backward |
| `A` | Strafe left |
| `D` | Strafe right |
| `←` | Turn left |
| `→` | Turn right |
| `Q` / `Esc` | Quit |

---

## Configuration

All tunable constants live at the top of `Raycaster.py`:

| Constant | Default | Effect |
|----------|---------|--------|
| `WIDTH` / `HEIGHT` | 640 × 408 | Window resolution |
| `MOVE_SPEED` | 0.08 | Movement speed (world units/frame) |
| `ROT_SPEED` | 0.05 | Rotation speed (radians/frame) |
| `LIGHT_DECAY` | 0.09 | How quickly walls dim with distance |
| `LIGHT_MIN` | 0.15 | Minimum brightness (prevents pitch black) |
| `FOG_ONSET` | 6.0 | Distance (units) at which fog begins |
| `FOG_FULL` | 18.0 | Distance (units) at which fog is total |
| `BOB_FREQ` | 2.8 | Head-bob oscillation frequency (Hz) |
| `BOB_AMP` | 12 | Head-bob pixel amplitude |
| `TEX_SIZE` | 64 | Texture resolution (must be power of 2) |

---

## Map Format

`RAW_MAP` in `Raycaster.py` is a 2D list of integers:

- `0` — open floor  
- `1` — grey wall  
- `2` — red wall  
- `3` — green wall  
- `4` — blue wall (also used for doorway gaps)  
- `5` — yellow wall  

Edit the list directly to design your own levels. The outer border must be fully walled to prevent rays escaping.

---

## How It Works

The engine uses the **DDA (Digital Differential Analysis)** algorithm:

1. **Floor & ceiling** — for each screen row below/above the horizon, back-project a world position and sample the texture.
2. **Walls** — cast one ray per screen column. The ray steps through the grid until it hits a wall cell, then the perpendicular distance is used to compute the projected wall height (avoiding fish-eye distortion).
3. **Lighting** — brightness = `exp(-LIGHT_DECAY × distance)`, clamped to `LIGHT_MIN`. Side faces (Y-axis walls) receive a ×0.65 darkening multiplier.
4. **Fog** — linearly blends pixel colour toward `FOG_COLOR` between `FOG_ONSET` and `FOG_FULL` units.
5. **Sprites** — sorted back-to-front, projected into camera space, and drawn column by column with alpha masking and z-buffer clipping.

For a detailed annotated walkthrough, open `Raycaster.ipynb`.

---

## Requirements

- Python 3.8+
- numpy
- Pillow
- tkinter (usually bundled with Python)

---

## License

MIT — see `LICENSE`.
"# Raycaster-Engine" 
