# =============================================================================
# Raycaster.py
# A Wolfenstein-style raycaster built with NumPy + tkinter.
#
# Controls:
#   W / Up    — move forward       S / Down  — move backward
#   A         — strafe left        D         — strafe right
#   Left      — turn left          Right     — turn right
#   Q / Esc   — quit
# =============================================================================

import numpy as np
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageTk
import time


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Manual PI constants — used by the custom trig functions below.
PI     = 3.14159265358979
TWO_PI = 6.28318530717959


# ---------------------------------------------------------------------------
# Custom scalar math helpers
# (These avoid importing math.sin/cos so the trig stays self-contained.)
# ---------------------------------------------------------------------------

def norm(x):
    """Normalise an angle (radians) into the range [-π, π]."""
    # First bring it into [0, 2π)
    x = x - TWO_PI * int(x / TWO_PI)
    if x < 0:
        x += TWO_PI
    # Then fold the upper half down to [-π, 0)
    if x > PI:
        x -= TWO_PI
    return x


def mc_sin(x):
    """
    Scalar sine via a Taylor series: sin(x) = x - x³/3! + x⁵/5! - …
    The series is evaluated up to the 19th-degree term (n = 1..9),
    which is accurate to machine precision for |x| ≤ π.
    """
    x = norm(x)   # bring into [-π, π] for best convergence
    t = x         # first term of the series is just x
    s = 0.0

    for n in range(1, 10):
        s += t
        # Each step multiplies by -x² / ((2n)(2n+1)) to get the next term
        t *= -(x * x) / ((2 * n) * (2 * n + 1))

    return s + t


def mc_cos(x):
    """
    Scalar cosine via a Taylor series: cos(x) = 1 - x²/2! + x⁴/4! - …
    Mirrors mc_sin but starts from the constant term 1 and uses even
    factorials in the denominator.

    BUG FIX: The original function contained Newton-Raphson square-root
    iteration, not a cosine approximation. Replaced with the correct
    Taylor series.
    """
    x = norm(x)   # bring into [-π, π] for best convergence
    t = 1.0       # first term of the cosine series is 1
    s = 0.0

    for n in range(1, 10):
        s += t
        # Each step multiplies by -x² / ((2n-1)(2n)) to reach the next term
        t *= -(x * x) / ((2 * n - 1) * (2 * n))

    return s + t


def scratch_floor(x):
    """Integer floor without importing math — equivalent to math.floor(x)."""
    ix = int(x)
    return ix - 1 if x < ix else ix


def absolute(x):
    """Absolute value without importing math — equivalent to abs(x)."""
    return x if x >= 0 else -x


# ---------------------------------------------------------------------------
# Vectorised trig helpers (operate on NumPy arrays)
# ---------------------------------------------------------------------------

def vec_sin(arr):
    """
    Element-wise sine for a NumPy array using the same Taylor series as
    mc_sin.  Normalises the input to [-π, π] before evaluating.

    BUG FIXES:
      1. `t = x.copy` → `t = x.copy()` (was assigning the method object).
      2. The normalised array `X` was computed but then ignored; the loop
         now uses `X` throughout.
    """
    # Step 1: reduce to [0, 2π)
    x = arr - TWO_PI * np.floor(arr / TWO_PI)
    # Step 2: fold upper half to [-π, 0) so the series converges well
    X = np.where(x > PI, x - TWO_PI, x)

    t = X.copy()             # first term of the series is X  (was X.copy — missing ())
    s = np.zeros_like(X)

    for n in range(1, 10):
        s += t
        t = t * (-(X * X) / ((2 * n) * (2 * n + 1)))

    return s + t


def vec_cos(arr):
    """
    Element-wise cosine for a NumPy array using the Taylor series.
    Normalises the input to [-π, π] before evaluating.
    """
    x = arr - TWO_PI * np.floor(arr / TWO_PI)
    x = np.where(x > PI, x - TWO_PI, x)

    t = np.ones_like(x)    # first term of the cosine series is 1
    s = np.zeros_like(x)

    for n in range(1, 10):
        s += t
        t = t * (-(x * x) / ((2 * n - 1) * (2 * n)))

    return s + t


# ---------------------------------------------------------------------------
# Renderer / window settings
# ---------------------------------------------------------------------------

WIDTH  = 640
HEIGHT = 408
HALF_H = HEIGHT // 2

MOVE_SPEED = 0.08   # world units per frame
ROT_SPEED  = 0.05   # radians per frame

# Textures are square power-of-two; TEX_MASK = 63 allows fast modulo via &
TEX_SIZE = 64
TEX_MASK = TEX_SIZE - 1

# Distance-based lighting: brightness = exp(-LIGHT_DECAY * dist), floored at
# LIGHT_MIN so the furthest walls are still barely visible.
LIGHT_DECAY = 0.09
LIGHT_MIN   = 0.15

# Fog blends wall/floor colour toward FOG_COLOR as distance increases.
# Fog starts at FOG_ONSET units and is fully opaque at FOG_FULL units.
FOG_ONSET = 6.0
FOG_FULL  = 18.0
FOG_COLOR = np.array([30, 30, 35], dtype=np.float32)

# Head-bob parameters for the weapon sprite
BOB_FREQ = 2.8   # oscillations per second
BOB_AMP  = 12    # pixel amplitude of the bob


# ---------------------------------------------------------------------------
# World map
# ---------------------------------------------------------------------------
# Each cell is either 0 (open floor) or a positive wall-type ID (1–5).
# Wall IDs correspond to colours defined in WALL_SPECS below.
# Value 4 is used on doorway cells — they share the same blue texture.

RAW_MAP = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,2,2,0,2,0,1,0,3,3,3,3,3,0,0,0,0,0,0,0,0,0,1],
    [1,0,2,0,0,2,0,4,0,3,0,0,0,3,0,0,0,0,0,0,0,0,0,1],
    [1,0,2,0,0,2,0,1,0,3,0,0,0,3,0,0,0,0,0,0,0,0,0,1],
    [1,0,2,2,0,2,0,1,0,3,3,0,3,3,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,4,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1,0,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,5,5,5,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,5,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,5,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,5,5,0,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

WORLD_MAP = np.array(RAW_MAP, dtype=np.int32)
MAP_ROWS, MAP_COLS = WORLD_MAP.shape

# Sprite list: (world_x, world_y, sprite_type)
SPRITES = [
    ( 3.5, 11.5, 'person'),
    ( 4.5, 11.5, 'person'),
    ( 9.0,  4.5, 'person'),
    ( 9.0, 20.0, 'person'),
    (14.5, 10.5, 'person'),
]


# ---------------------------------------------------------------------------
# Asset directories
# BUG FIX: 'textrues' → 'textures' and 'spirites' → 'sprites'
# ---------------------------------------------------------------------------

for d in ['textures', 'floor', 'sprites', 'assets']:
    Path(d).mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Procedural texture generators
# (Run once on first launch; results are saved to disk for future runs.)
# ---------------------------------------------------------------------------

def make_brick(rgb, s=TEX_SIZE):
    """
    Generate a brick-wall texture of size s×s pixels.
    Each row of 10 pixels is treated as one brick course; every other
    course is offset by s/4 pixels to give the typical running-bond
    pattern.  A small deterministic noise term keeps it from looking
    too flat.
    """
    br, bg, bb = rgb
    y, x = np.mgrid[0:s, 0:s]

    row    = y // 10
    offset = np.where(row % 2 == 0, s // 4, 0)
    mortar = (y % 10 == 0) | ((x + offset) % (s // 4) == 0)
    noise  = ((x * 3 + y * 7) % 20) - 10   # cheap hash-like noise in [-10, +10]

    tex = np.zeros((s, s, 3), dtype=np.uint8)
    for ch, base in enumerate([br, bg, bb]):
        v = np.where(mortar, base - 40, base + noise)
        tex[:, :, ch] = np.clip(v, 0, 255)

    return tex


def make_checker(c1, c2, s=TEX_SIZE, n=8):
    """
    Generate a checkerboard texture of size s×s split into an n×n grid.
    Used for both the floor and the ceiling.
    """
    y, x = np.mgrid[0:s, 0:s]
    sq   = s // n
    tile = ((x // sq) + (y // sq)) % 2   # 0 or 1, alternating per square

    return np.where(tile[:, :, None] == 0, c1, c2).astype(np.uint8)


def make_person(s=TEX_SIZE):
    """
    Generate a simple stick-figure person sprite (RGBA, 64×64).
    The figure has a skin-coloured head, a red torso and blue legs.
    """
    t  = np.zeros((s, s, 4), dtype=np.uint8)
    cx = s // 2
    t[ 8:18, cx-5:cx+5] = [220, 180, 140, 255]   # head  (skin tone)
    t[18:40, cx-7:cx+7] = [180,  50,  50, 255]   # torso (red shirt)
    t[40:58, cx-7:cx-2] = [ 50,  50, 180, 255]   # left leg  (blue trousers)
    t[40:58, cx+2:cx+7] = [ 50,  50, 180, 255]   # right leg
    return t


def make_hand(h=150, w=200):
    """
    Generate a first-person weapon/hand sprite (RGBA, 200×150).
    The image shows a hand holding a gun, drawn from simple rectangles.
    """
    t = np.zeros((h, w, 4), dtype=np.uint8)
    t[70:h,  40:160] = [200, 160, 120, 255]   # forearm / wrist
    t[40:80, 50:150] = [200, 160, 120, 255]   # palm
    t[20:55, 90:115] = [ 60,  60,  60, 255]   # gun barrel (dark metal)
    t[30:50, 50: 65] = [200, 160, 120, 255]   # index finger (left side)
    t[30:50,150:165] = [200, 160, 120, 255]   # index finger (right side)
    return t


# ---------------------------------------------------------------------------
# Texture loaders
# Each function tries to load from disk; if the file doesn't exist it runs
# the provided generator, saves the result, and returns the array.
# ---------------------------------------------------------------------------

def load_rgb(path, make_fn):
    """Load (or generate) an RGB texture and return it as a uint8 array."""
    if Path(path).exists():
        img = Image.open(path).convert('RGB').resize(
            (TEX_SIZE, TEX_SIZE), Image.NEAREST)
        return np.array(img, dtype=np.uint8)
    arr = make_fn()
    Image.fromarray(arr).save(path)
    print(f'  generated {path}')
    return arr


def load_rgba(path, make_fn, size=None):
    """Load (or generate) an RGBA sprite and return it as a uint8 array."""
    if Path(path).exists():
        img = Image.open(path).convert('RGBA')
        if size:
            img = img.resize(size, Image.NEAREST)
        return np.array(img, dtype=np.uint8)
    arr = make_fn()
    Image.fromarray(arr).save(path)
    print(f'  generated {path}')
    return arr


# ---------------------------------------------------------------------------
# Load / generate all textures at startup
# ---------------------------------------------------------------------------

# Wall textures: one brick texture per wall-ID, tinted with a different colour.
WALL_SPECS = {
    1: ('textures/wall_1.png', ( 90,  90,  90)),   # grey
    2: ('textures/wall_2.png', (150,  40,  40)),   # red
    3: ('textures/wall_3.png', ( 40, 150,  40)),   # green
    4: ('textures/wall_4.png', ( 40,  40, 150)),   # blue
    5: ('textures/wall_5.png', (150, 150,  40)),   # yellow
}

# Stack all wall textures into a single (6, 64, 64, 3) array so the renderer
# can index by wall-ID without a Python loop per column.
WALL_TEX_STACK = np.zeros((6, TEX_SIZE, TEX_SIZE, 3), dtype=np.uint8)
for wid, (path, col) in WALL_SPECS.items():
    WALL_TEX_STACK[wid] = load_rgb(path, lambda c=col: make_brick(c))

FLOOR_TEX = load_rgb('floor/floor.png',
                     lambda: make_checker((50, 50, 50), (35, 35, 35)))
CEIL_TEX  = load_rgb('floor/ceiling.png',
                     lambda: make_checker((30, 30, 40), (20, 20, 30)))

PERSON_SPR = load_rgba('sprites/person.png', make_person,
                       size=(TEX_SIZE, TEX_SIZE))
HAND_SPR   = load_rgba('assets/hand.png',   make_hand,
                       size=(200, 150))   # PIL resize expects (width, height)

# Map sprite-type strings to their loaded RGBA arrays
SPR_TEX = {'person': PERSON_SPR}


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class Player:
    """
    Holds the player's position, view direction and camera plane.

    Coordinate system
    -----------------
    px, py   — world position (floats, measured in map cells)
    dx, dy   — unit view direction vector
    plx, ply — camera plane vector (perpendicular to direction, length ≈ 0.66
               which gives a ~66° horizontal field of view)

    The camera plane length controls FOV: 0.66 → ~66°, 1.0 → ~90°.
    """

    __slots__ = ('px', 'py', 'dx', 'dy', 'plx', 'ply', 'moving', 'bob_phase')

    def __init__(self, px=12.0, py=12.0,
                 dx=-1.0, dy=0.0,
                 plx=0.0, ply=0.66):
        self.px,  self.py  = px,  py
        self.dx,  self.dy  = dx,  dy
        self.plx, self.ply = plx, ply
        self.moving    = False   # True for any frame in which the player moved
        self.bob_phase = 0.0    # Accumulated time (radians) driving the walk bob

    def move(self, speed):
        """Move forward (+) or backward (-) along the view direction."""
        nx = self.px + self.dx * speed
        ny = self.py + self.dy * speed
        # Axis-separated collision: test each axis independently so the player
        # can slide along walls rather than stopping dead.
        if WORLD_MAP[int(nx), int(self.py)] == 0: self.px = nx
        if WORLD_MAP[int(self.px), int(ny)] == 0: self.py = ny
        self.moving = True

    def strafe(self, speed):
        """Move sideways relative to the view direction."""
        nx = self.px - self.dy * speed
        ny = self.py + self.dx * speed
        if WORLD_MAP[int(nx), int(self.py)] == 0: self.px = nx
        if WORLD_MAP[int(self.px), int(ny)] == 0: self.py = ny
        self.moving = True

    def rotate(self, angle):
        """Rotate the view and camera plane by `angle` radians."""
        R = rotation_2d(angle)
        d = R @ np.array([self.dx,  self.dy ])
        p = R @ np.array([self.plx, self.ply])
        self.dx,  self.dy  = float(d[0]), float(d[1])
        self.plx, self.ply = float(p[0]), float(p[1])


# ---------------------------------------------------------------------------
# 2-D rotation matrix (depends on mc_sin / mc_cos defined above)
# ---------------------------------------------------------------------------

def rotation_2d(angle):
    """
    Return a 2×2 rotation matrix for the given angle (radians).

    BUG FIX: Second row was `s, c` (missing brackets) which caused
    NumPy to construct an invalid ragged sequence instead of a 2×2 matrix.
    Corrected to `[s, c]`.
    """
    c = mc_cos(angle)
    s = mc_sin(angle)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Pre-computed per-column arrays (built once, reused every frame)
# ---------------------------------------------------------------------------

_COL       = np.arange(WIDTH, dtype=np.float64)              # screen column indices [0 .. WIDTH-1]
_CAMX      = 2.0 * _COL / WIDTH - 1.0                        # camera-space X in [-1, +1]
_FLOOR_ROW = np.arange(HALF_H + 1, HEIGHT, dtype=np.float64) # screen rows that show the floor


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render(player, frame):
    """
    Fill `frame` (HEIGHT × WIDTH × 3 uint8) with the current view and
    return the z-buffer (perpendicular wall distances per column).

    Pipeline
    --------
    1. Floor & ceiling — vectorised texture mapping using row-distance.
    2. Walls — DDA ray-march for every column, then vertical texture strips.
    3. Lighting — exponential fall-off, side-face darkening, distance fog.
    """
    px, py   = player.px, player.py
    dx, dy   = player.dx, player.dy
    plx, ply = player.plx, player.ply

    # ── 1. Floor & ceiling ──────────────────────────────────────────────────

    # Distance from the camera to the floor at each screen row below the horizon.
    row_dist = HALF_H / (_FLOOR_ROW - HALF_H)

    # World-space step per screen pixel in each direction for the current row.
    step_x = row_dist * (2.0 * plx / WIDTH)
    step_y = row_dist * (2.0 * ply / WIDTH)

    # World position of the leftmost floor pixel in each row.
    fx0 = px + row_dist * (dx - plx)
    fy0 = py + row_dist * (dy - ply)

    # Broadcast across all columns to get world positions for every floor pixel.
    col_range = np.arange(WIDTH, dtype=np.float64)
    fx = fx0[:, None] + step_x[:, None] * col_range[None, :]
    fy = fy0[:, None] + step_y[:, None] * col_range[None, :]

    # Convert world positions to texture coordinates with a fast bitmask.
    tx = (fx * TEX_SIZE).astype(np.int32) & TEX_MASK
    ty = (fy * TEX_SIZE).astype(np.int32) & TEX_MASK
    tx = np.clip(tx, 0, TEX_MASK)
    ty = np.clip(ty, 0, TEX_MASK)

    # Write floor pixels; mirror the same texture coordinates for the ceiling.
    frame[HALF_H + 1:HEIGHT, :] = FLOOR_TEX[ty, tx]
    ceil_y = (HEIGHT - 1 - _FLOOR_ROW).astype(int)
    frame[ceil_y, :] = CEIL_TEX[ty, tx]

    # ── 2. Walls — vectorised DDA ────────────────────────────────────────────

    # Ray direction for every screen column (direction + camera-plane offset).
    rdx = dx + plx * _CAMX
    rdy = dy + ply * _CAMX

    # Current map cell for each ray (all start at the player's cell).
    mx = np.full(WIDTH, int(px), dtype=np.int32)
    my = np.full(WIDTH, int(py), dtype=np.int32)

    # Distance a ray must travel to cross one full grid line in each axis.
    with np.errstate(divide='ignore', invalid='ignore'):
        dlx = np.where(rdx == 0, 1e30, np.abs(1.0 / rdx))
        dly = np.where(rdy == 0, 1e30, np.abs(1.0 / rdy))

    # Step direction (+1 or -1) per axis per ray.
    stx = np.where(rdx < 0, -1, 1).astype(np.int32)
    sty = np.where(rdy < 0, -1, 1).astype(np.int32)

    # Initial side-distance: distance to the first grid crossing in each axis.
    sdx = np.where(rdx < 0, (px - mx) * dlx, (mx + 1.0 - px) * dlx)
    sdy = np.where(rdy < 0, (py - my) * dly, (my + 1.0 - py) * dly)

    hit  = np.zeros(WIDTH, dtype=bool)
    side = np.zeros(WIDTH, dtype=np.int32)

    # Advance each ray one step at a time until all have found a wall.
    for _ in range(MAP_ROWS + MAP_COLS):
        if hit.all():
            break   # every ray has hit something; no point iterating further

        alive = ~hit
        go_x  = alive & (sdx < sdy)   # rays that cross an X boundary next
        go_y  = alive & ~go_x          # rays that cross a Y boundary next

        sdx  = np.where(go_x, sdx + dlx, sdx)
        mx   = np.where(go_x, mx  + stx, mx)
        side = np.where(go_x, 0, side)

        sdy  = np.where(go_y, sdy + dly, sdy)
        my   = np.where(go_y, my  + sty, my)
        side = np.where(go_y, 1, side)

        # Clamp before indexing to avoid out-of-bounds on map edges.
        cmx = np.clip(mx, 0, MAP_ROWS - 1)
        cmy = np.clip(my, 0, MAP_COLS - 1)
        hit |= (WORLD_MAP[cmx, cmy] > 0)

    # ── 3. Wall geometry ─────────────────────────────────────────────────────

    # Perpendicular distance (avoids the fish-eye effect from Euclidean dist).
    perp = np.where(side == 0, sdx - dlx, sdy - dly)
    perp = np.maximum(perp, 0.0001)   # guard against division by zero
    z_buffer = perp.copy()             # saved for the sprite pass

    wall_h     = np.maximum(1, (HEIGHT / perp).astype(np.int32))
    draw_start = np.maximum(0,      -wall_h // 2 + HALF_H)
    draw_end   = np.minimum(HEIGHT,  wall_h // 2 + HALF_H)

    # Exact position along the wall face (used to pick the texture column).
    wall_hit = np.where(side == 0, py + perp * rdy, px + perp * rdx)
    wall_hit -= np.floor(wall_hit)
    tex_x = (wall_hit * TEX_SIZE).astype(np.int32)

    # Flip the texture column on faces that would otherwise appear mirrored.
    flip  = ((side == 0) & (rdx > 0)) | ((side == 1) & (rdy < 0))
    tex_x = np.where(flip, TEX_MASK - tex_x, tex_x)
    tex_x = np.clip(tex_x, 0, TEX_MASK)

    cmx      = np.clip(mx, 0, MAP_ROWS - 1)
    cmy      = np.clip(my, 0, MAP_COLS - 1)
    wall_ids = np.clip(WORLD_MAP[cmx, cmy], 1, 5)   # clamp to valid texture IDs

    # ── 4. Lighting & fog ────────────────────────────────────────────────────

    # Exponential brightness fall-off: close walls are fully lit, far ones dim.
    brightness = np.exp(-LIGHT_DECAY * perp)
    brightness = np.clip(brightness, LIGHT_MIN, 1.0)

    # Side faces (Y-axis walls) are darkened to give a cheap directional feel.
    brightness = np.where(side == 1, brightness * 0.65, brightness)

    # Fog factor: 0.0 at FOG_ONSET, 1.0 at FOG_FULL, clamped outside that range.
    fog = np.clip((perp - FOG_ONSET) / (FOG_FULL - FOG_ONSET), 0.0, 1.0)

    # ── 5. Rasterise each wall column ────────────────────────────────────────
    for x in range(WIDTH):
        ds, de = int(draw_start[x]), int(draw_end[x])
        if de <= ds:
            continue   # degenerate slice (shouldn't happen, but be safe)

        n   = de - ds
        wh  = max(1, int(wall_h[x]))
        tx  = int(tex_x[x])
        tex = WALL_TEX_STACK[int(wall_ids[x])]   # (TEX_SIZE, TEX_SIZE, 3)

        # Map screen rows to texture rows, centering the slice on HALF_H.
        ty_f = (np.arange(n, dtype=np.float32) + (ds - HALF_H + wh * 0.5)) \
               * TEX_SIZE / wh
        ty   = np.clip(ty_f.astype(np.int32), 0, TEX_MASK)

        colour = tex[ty, tx].astype(np.float32)   # (n, 3) texel colours
        colour *= brightness[x]                    # apply distance lighting

        # Blend toward the fog colour for distant walls.
        f = fog[x]
        if f > 0.0:
            colour = colour * (1.0 - f) + FOG_COLOR * f

        frame[ds:de, x] = np.clip(colour, 0, 255).astype(np.uint8)

    return z_buffer


# ---------------------------------------------------------------------------
# Sprite renderer
# ---------------------------------------------------------------------------

def draw_sprites(player, frame, z_buffer):
    """
    Draw all world sprites (NPCs, items) into `frame`, sorted back-to-front
    so nearer sprites correctly occlude farther ones.

    Uses the z-buffer produced by `render()` to clip sprite columns that are
    hidden behind a wall.
    """
    px, py   = player.px, player.py
    dx, dy   = player.dx, player.dy
    plx, ply = player.plx, player.ply

    # Inverse determinant of the view matrix — used to transform world
    # positions into camera space without a full matrix inverse.
    inv_det = 1.0 / (plx * dy - dx * ply + 1e-30)

    # Painter's algorithm: draw furthest sprites first.
    ordered = sorted(
        SPRITES,
        key=lambda s: (s[0] - px) ** 2 + (s[1] - py) ** 2,
        reverse=True
    )

    for (sx, sy, stype) in ordered:
        tex = SPR_TEX.get(stype)
        if tex is None:
            continue
        th, tw = tex.shape[:2]

        # Transform sprite position relative to the camera.
        rx = sx - px
        ry = sy - py
        cam_xs =  inv_det * ( dy * rx - dx * ry)   # camera-space X (left/right)
        cam_z  =  inv_det * (-ply * rx + plx * ry)  # camera-space Z (depth)

        # Discard sprites that are behind or almost at the camera plane.
        if cam_z <= 0.1:
            continue

        # Project onto the screen.
        scr_x  = int((WIDTH / 2) * (1 + cam_xs / cam_z))
        proj_h = abs(int(HEIGHT / cam_z))
        proj_w = proj_h   # assume square sprite aspect ratio
        if proj_h == 0:
            continue

        # Screen-space bounding box of the sprite.
        ds_y = max(0,      -proj_h // 2 + HALF_H)
        de_y = min(HEIGHT,  proj_h // 2 + HALF_H)
        ds_x = max(0,       scr_x - proj_w // 2)
        de_x = min(WIDTH,   scr_x + proj_w // 2)
        if de_y <= ds_y or de_x <= ds_x:
            continue

        # Pre-compute texture Y indices for all visible rows in this sprite.
        screen_ys = np.arange(ds_y, de_y, dtype=np.float32)
        ty = ((screen_ys - (HALF_H - proj_h // 2)) * th / proj_h).astype(np.int32)
        ty = np.clip(ty, 0, th - 1)

        for x in range(ds_x, de_x):
            # Skip columns where a wall is closer than this sprite.
            if cam_z >= z_buffer[x]:
                continue

            tx = int((x - (scr_x - proj_w // 2)) * tw / proj_w)
            tx = max(0, min(tw - 1, tx))

            # Only paint pixels where the sprite's alpha channel is opaque.
            alpha   = tex[ty, tx, 3]
            visible = alpha >= 128
            if not visible.any():
                continue

            rows = screen_ys[visible].astype(np.int32)
            frame[rows, x] = tex[ty[visible], tx, :3]


# ---------------------------------------------------------------------------
# HUD (mini-map, debug overlay, weapon, crosshair)
# ---------------------------------------------------------------------------

# Try to load a proportional font; fall back to the built-in bitmap font if
# arial.ttf isn't available on this system.
try:
    _FONT = ImageFont.truetype('arial.ttf', 14)
except Exception:
    _FONT = ImageFont.load_default()

# Mini-map layout constants (computed once from map dimensions).
_MM_CELL = 5                                    # pixels per map cell
_MM_PAD  = 10                                   # margin from window edge
_MM_W    = MAP_COLS * _MM_CELL                  # total mini-map width  (pixels)
_MM_H    = MAP_ROWS * _MM_CELL                  # total mini-map height (pixels)
_MM_OX   = WIDTH  - _MM_W - _MM_PAD            # top-left X of the mini-map
_MM_OY   = _MM_PAD                              # top-left Y of the mini-map

# Colour for each wall type on the mini-map.
_WCOLORS = {
    1: (100, 100, 100),
    2: (150,  50,  50),
    3: ( 50, 150,  50),
    4: ( 50,  50, 150),
    5: (150, 150,  50),
}

# Pre-render the static wall tiles once; the player dot is overlaid each frame.
_mm_base = np.zeros((_MM_H, _MM_W, 3), dtype=np.uint8)
for _r in range(MAP_ROWS):
    for _c in range(MAP_COLS):
        _v = WORLD_MAP[_r, _c]
        if _v > 0:
            _y0, _x0 = _r * _MM_CELL, _c * _MM_CELL
            _mm_base[_y0:_y0 + _MM_CELL, _x0:_x0 + _MM_CELL] = \
                _WCOLORS.get(_v, (80, 80, 80))

# Boolean mask: True wherever a wall colour was painted, False on open floor.
_mm_mask = (_mm_base > 0).any(axis=2, keepdims=True)


def _fwd_dist(player):
    """
    Return the perpendicular distance to the wall directly in front of the
    player.  Uses a scalar DDA (same algorithm as the vectorised wall renderer)
    so the result matches what appears on screen.
    """
    px, py   = player.px, player.py
    rdx, rdy = player.dx, player.dy
    mx, my   = int(px), int(py)

    dlx = 1e30 if rdx == 0 else absolute(1.0 / rdx)
    dly = 1e30 if rdy == 0 else absolute(1.0 / rdy)

    stx = -1 if rdx < 0 else 1
    sty = -1 if rdy < 0 else 1

    sdx = (px - mx) * dlx      if rdx < 0 else (mx + 1.0 - px) * dlx
    sdy = (py - my) * dly      if rdy < 0 else (my + 1.0 - py) * dly

    side = 0
    for _ in range(MAP_ROWS + MAP_COLS):
        if sdx < sdy: sdx += dlx; mx += stx; side = 0
        else:         sdy += dly; my += sty; side = 1
        if 0 <= mx < MAP_ROWS and 0 <= my < MAP_COLS and WORLD_MAP[mx, my] > 0:
            break

    return (sdx - dlx) if side == 0 else (sdy - dly)


def update_bob(player, dt):
    """
    Advance the head-bob phase when the player is moving, and let it
    decay smoothly to zero when they stop.  Called once per frame.
    """
    if player.moving:
        player.bob_phase += TWO_PI * BOB_FREQ * dt
        if player.bob_phase > TWO_PI:
            player.bob_phase -= TWO_PI
    else:
        # Exponential decay: the bob fades out 3× faster than it oscillates.
        player.bob_phase *= max(0.0, 1.0 - dt * BOB_FREQ * 3.0)

    player.moving = False   # reset; move/strafe set it True again next frame


def draw_hud(player, frame, z_buffer, fps):
    """
    Composite all HUD elements onto `frame` in-place:
      1. Mini-map background — dims the scene behind the map.
      2. Mini-map tiles      — wall colours from _mm_base.
      3. Player dot & arrow  — position + facing direction.
      4. Debug text panel    — position, direction, wall distance, FPS.
      5. Weapon hand         — right-aligned, with head-bob offset.
      6. Crosshair           — centred on screen.
    """

    # ── 1 & 2: Mini-map ──────────────────────────────────────────────────────
    oy, ox = _MM_OY, _MM_OX
    region = frame[oy:oy + _MM_H, ox:ox + _MM_W]

    # Darken the scene behind the map, then paint wall tiles on top.
    bg = (region * 0.25).astype(np.uint8)
    frame[oy:oy + _MM_H, ox:ox + _MM_W] = np.where(_mm_mask, _mm_base, bg)

    # ── 3a: Player dot ───────────────────────────────────────────────────────
    pdx = np.clip(int(ox + player.py * _MM_CELL), ox, ox + _MM_W - 1)
    pdy = np.clip(int(oy + player.px * _MM_CELL), oy, oy + _MM_H - 1)
    frame[pdy - 2:pdy + 3, pdx - 2:pdx + 3] = (255, 50, 50)

    # ── 3b: Direction arrow (10-point line) ──────────────────────────────────
    ax  = int(pdx + player.dy * 9)
    ay  = int(pdy + player.dx * 9)
    ts  = np.linspace(0, 1, 10)
    lxs = np.clip((pdx + ts * (ax - pdx)).astype(int), ox, ox + _MM_W - 1)
    lys = np.clip((pdy + ts * (ay - pdy)).astype(int), oy, oy + _MM_H - 1)
    frame[lys, lxs] = (60, 230, 230)

    # ── 4: Debug text panel ──────────────────────────────────────────────────
    fwd    = _fwd_dist(player)
    TW, TH = 270, 88
    surf = Image.new('RGB', (TW, TH), (0, 0, 0))
    d = ImageDraw.Draw(surf)
    d.text(( 8,  4), f'POS  X:{player.px:.2f}  Y:{player.py:.2f}',
           fill=(220, 220, 220), font=_FONT)
    d.text(( 8, 24), f'DIR  X:{player.dx:.2f}  Y:{player.dy:.2f}',
           fill=(220, 220, 220), font=_FONT)
    d.text(( 8, 44), f'WALL DIST: {fwd:.2f} units',
           fill=(255, 200,  50), font=_FONT)
    d.text(( 8, 64), f'FPS: {fps}',
           fill=(100, 220, 100), font=_FONT)
    txt = np.array(surf, dtype=np.uint8)

    # Alpha-blend the text surface (70%) over the darkened scene (30%).
    frame[6:6 + TH, 6:6 + TW] = (
        frame[6:6 + TH, 6:6 + TW] * 0.30 + txt * 0.70
    ).astype(np.uint8)

    # ── 5: Weapon hand with head-bob ─────────────────────────────────────────
    hh, hw = HAND_SPR.shape[:2]
    hx = WIDTH - hw - 8   # right-aligned with a small margin

    # BUG FIX: 'scratch_sin' was called here but was never defined anywhere.
    # Changed to mc_sin, which is the correct custom sine function.
    bob_offset = int(BOB_AMP * mc_sin(player.bob_phase))
    hy = HEIGHT - hh - 4 + bob_offset
    hy = max(0, min(HEIGHT - hh, hy))   # clamp so the sprite stays on screen

    if hx >= 0:
        a   = HAND_SPR[:, :, 3:4].astype(np.float32) / 255.0   # alpha  (H,W,1)
        rgb = HAND_SPR[:, :, :3 ].astype(np.float32)            # colour (H,W,3)
        roi = frame[hy:hy + hh, hx:hx + hw].astype(np.float32)
        frame[hy:hy + hh, hx:hx + hw] = (rgb * a + roi * (1.0 - a)).astype(np.uint8)

    # ── 6: Crosshair ─────────────────────────────────────────────────────────
    cx, cy = WIDTH // 2, HALF_H
    frame[cy,        cx - 8:cx + 9] = (200, 200, 200)   # horizontal bar
    frame[cy - 8:cy + 9, cx       ] = (200, 200, 200)   # vertical bar


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

KEYS_HELD: set = set()   # set of currently pressed key names (lowercase)

def _on_key_press(event):   KEYS_HELD.add(event.keysym.lower())
def _on_key_release(event): KEYS_HELD.discard(event.keysym.lower())


def apply_input(player):
    """
    Read the current key state and update the player accordingly.
    Returns False if the user has requested to quit, True otherwise.
    """
    if 'escape' in KEYS_HELD or 'q' in KEYS_HELD:
        return False

    if 'w'     in KEYS_HELD or 'up'    in KEYS_HELD: player.move( MOVE_SPEED)
    if 's'     in KEYS_HELD or 'down'  in KEYS_HELD: player.move(-MOVE_SPEED)
    if 'a'     in KEYS_HELD:                          player.strafe(-MOVE_SPEED)
    if 'd'     in KEYS_HELD:                          player.strafe( MOVE_SPEED)
    if 'left'  in KEYS_HELD:                          player.rotate( ROT_SPEED)
    if 'right' in KEYS_HELD:                          player.rotate(-ROT_SPEED)

    return True


# ---------------------------------------------------------------------------
# Engine entry point
# ---------------------------------------------------------------------------

def run_engine():
    """
    Initialise the tkinter window and run the main render loop.

    The loop is driven by tkinter's `after()` scheduler (1 ms delay) rather
    than a blocking while-loop so the GUI event queue stays responsive.
    """

    # ── Initialise state ──────────────────────────────────────────────────────
    player = Player()   # spawns at (12, 12), facing left (dx=-1, dy=0)
    frame  = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)   # reused every frame

    # ── Create window ─────────────────────────────────────────────────────────
    root = tk.Tk()
    root.title('DOOM Raycaster  |  WASD / Arrows  |  Q = quit')
    root.resizable(False, False)

    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT,
                       bg='black', highlightthickness=0)
    canvas.pack()

    root.bind('<KeyPress>',   _on_key_press)
    root.bind('<KeyRelease>', _on_key_release)

    # photo_ref prevents tkinter's GC from deleting the PhotoImage between
    # the assignment and the canvas draw call.
    photo_ref = [None]

    # FPS tracking (updated every 0.5 s to avoid flickering numbers).
    fps_val = [0]
    fps_cnt = [0]
    fps_t   = [time.perf_counter()]
    last_t  = [time.perf_counter()]

    # ── Render loop ───────────────────────────────────────────────────────────
    def loop():
        now = time.perf_counter()
        dt  = now - last_t[0]   # seconds since last frame (used for smooth bob)
        last_t[0] = now

        # Input must be processed before rendering so this frame reflects the
        # latest key state.
        if not apply_input(player):
            root.destroy()
            return

        update_bob(player, dt)

        # Clear to black first (floor/ceiling rendering will overwrite most of
        # the frame, but this ensures any gaps are clean).
        frame[:] = 0
        z = render(player, frame)             # walls, floor, ceiling, lighting
        draw_sprites(player, frame, z)        # world sprites
        draw_hud(player, frame, z, fps_val[0])  # UI overlay

        # Convert the NumPy frame to a tkinter-compatible image and display it.
        img = ImageTk.PhotoImage(Image.fromarray(frame, 'RGB'))
        photo_ref[0] = img
        canvas.create_image(0, 0, anchor=tk.NW, image=img)

        # Update the FPS counter once every half-second.
        fps_cnt[0] += 1
        elapsed = now - fps_t[0]
        if elapsed >= 0.5:
            fps_val[0] = int(fps_cnt[0] / elapsed)
            fps_cnt[0] = 0
            fps_t[0]   = now

        if canvas.winfo_exists():
            canvas.after(1, loop)   # schedule the next frame

    canvas.after(0, loop)
    print('Click the window to focus, then use WASD / arrow keys.')
    root.mainloop()
    print('Engine stopped.')


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run_engine()
