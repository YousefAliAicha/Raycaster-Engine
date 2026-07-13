# =============================================================================
# textures.py — procedural pixel-art texture/sprite generation, and the
# load-or-generate-and-cache helpers used to build a TextureSet at startup.
# =============================================================================

from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import config

TEX_SIZE = config.TEX_SIZE


# ---------------------------------------------------------------------------
# Wall / floor / ceiling generators
# ---------------------------------------------------------------------------

def make_brick(rgb, s=TEX_SIZE):
    """
    Generate a brick-wall texture of size sxs pixels. Each row of 10
    pixels is one brick course; every other course is offset by s/4
    pixels for the typical running-bond pattern, plus cheap hash noise
    so it doesn't look flat.
    """
    br, bg, bb = rgb
    y, x = np.mgrid[0:s, 0:s]

    row    = y // 10
    offset = np.where(row % 2 == 0, s // 4, 0)
    mortar = (y % 10 == 0) | ((x + offset) % (s // 4) == 0)
    noise  = ((x * 3 + y * 7) % 20) - 10

    tex = np.zeros((s, s, 3), dtype=np.uint8)
    for ch, base in enumerate([br, bg, bb]):
        v = np.where(mortar, base - 40, base + noise)
        tex[:, :, ch] = np.clip(v, 0, 255)
    return tex


def make_checker(c1, c2, s=TEX_SIZE, n=8):
    """Checkerboard texture split into an n x n grid (floor/ceiling)."""
    y, x = np.mgrid[0:s, 0:s]
    sq   = s // n
    tile = ((x // sq) + (y // sq)) % 2
    return np.where(tile[:, :, None] == 0, c1, c2).astype(np.uint8)


def make_door(s=TEX_SIZE):
    """
    Door texture: vertical ribbed panels, with downward-pointing chevrons
    (purely a visual choice — the ribs and chevrons don't need to agree
    on a "direction of travel", they're just decoration on the panel).
    """
    y, x = np.mgrid[0:s, 0:s]
    panel   = np.array([110,  90,  60], dtype=np.float32)
    trim    = np.array([ 65,  55,  38], dtype=np.float32)
    groove  = np.array([ 55,  45,  30], dtype=np.float32)
    handle_c = np.array([220, 190, 90], dtype=np.float32)

    tex = np.tile(panel, (s, s, 1))
    border = (x < 4) | (x >= s - 4) | (y < 4) | (y >= s - 4)
    tex[border] = trim

    rib = (x % 10 < 2) & ~border
    tex[rib] = groove

    cx = s // 2
    for base_y in (14, 30, 46):
        chevron = ((np.abs(x - cx) <= (base_y - y)) &
                   (y >= base_y - 5) & (y <= base_y) & (y >= 4))
        tex[chevron] = trim * 1.3

    hy0, hy1 = s - 14, s - 8
    hx0, hx1 = int(s * 0.46), int(s * 0.58)
    tex[hy0:hy1, hx0:hx1] = handle_c
    return np.clip(tex, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sprite generators
# ---------------------------------------------------------------------------

def make_person(s=TEX_SIZE):
    """Simple stick-figure sprite: skin head, red torso, blue legs."""
    t  = np.zeros((s, s, 4), dtype=np.uint8)
    cx = s // 2
    t[ 8:18, cx-5:cx+5] = [220, 180, 140, 255]
    t[18:40, cx-7:cx+7] = [180,  50,  50, 255]
    t[40:58, cx-7:cx-2] = [ 50,  50, 180, 255]
    t[40:58, cx+2:cx+7] = [ 50,  50, 180, 255]
    return t


def make_barrel(s=TEX_SIZE):
    """Wooden barrel: cylindrical silhouette with metal hoop bands."""
    t  = np.zeros((s, s, 4), dtype=np.uint8)
    y, x = np.mgrid[0:s, 0:s]
    cx = s / 2.0
    body_top, body_bot = int(s * 0.12), int(s * 0.95)
    row_frac = np.clip((y - body_top) / max(1, (body_bot - body_top)), 0, 1)
    bulge = 1.0 - 0.35 * np.abs(row_frac - 0.5) * 2.0
    half_w = (s * 0.32) * bulge
    inside = (np.abs(x - cx) <= half_w) & (y >= body_top) & (y <= body_bot)

    edge_shade = 1.0 - 0.5 * (np.abs(x - cx) / np.maximum(half_w, 1))
    base = np.array([120, 80, 40], dtype=np.float32)
    band = (y % 14 < 3)
    colour = np.where(band[..., None],
                       np.array([60, 60, 65], dtype=np.float32),
                       base) * edge_shade[..., None]

    t[..., :3] = np.clip(colour, 0, 255).astype(np.uint8)
    t[..., 3]  = np.where(inside, 255, 0).astype(np.uint8)
    return t


def make_flag(s=TEX_SIZE):
    """Chequered goal-flag sprite marking the level exit."""
    t = np.zeros((s, s, 4), dtype=np.uint8)
    y, x = np.mgrid[0:s, 0:s]
    pole = (np.abs(x - int(s * 0.28)) <= 2) & (y >= int(s * 0.15)) & (y <= int(s * 0.95))
    flag_area = (x > int(s * 0.28)) & (x < int(s * 0.85)) & \
                (y >= int(s * 0.15)) & (y <= int(s * 0.55))
    checker = (((x // 6) + (y // 6)) % 2 == 0)
    t[pole] = (90, 70, 40, 255)
    t[flag_area & checker]  = (240, 240, 240, 255)
    t[flag_area & ~checker] = (30, 30, 30, 255)
    return t


def make_key(s=TEX_SIZE):
    """Small golden key pickup sprite."""
    t = np.zeros((s, s, 4), dtype=np.uint8)
    gold = np.array([230, 190, 60, 255], dtype=np.uint8)
    dark = np.array([160, 120, 20, 255], dtype=np.uint8)
    cx, cy = s // 2, int(s * 0.35)
    y, x = np.mgrid[0:s, 0:s]

    ring = ((x - cx) ** 2 + (y - cy) ** 2 <= (s * 0.16) ** 2) & \
           ((x - cx) ** 2 + (y - cy) ** 2 >= (s * 0.09) ** 2)
    shaft = (np.abs(x - cx) <= s * 0.045) & (y >= cy) & (y <= int(s * 0.82))
    tooth1 = (x >= cx) & (x <= cx + int(s * 0.14)) & \
             (y >= int(s * 0.68)) & (y <= int(s * 0.76))
    tooth2 = (x >= cx) & (x <= cx + int(s * 0.10)) & \
             (y >= int(s * 0.78)) & (y <= int(s * 0.82))

    mask = ring | shaft | tooth1 | tooth2
    t[mask] = gold
    t[ring & (((x - cx) ** 2 + (y - cy) ** 2) >= (s * 0.13) ** 2)] = dark
    t[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    return t


def make_hand(h=170, w=220):
    """
    First-person weapon sprite: a pistol held in a shaded hand, replacing
    the earlier "just colored rectangles" placeholder with a silhouette
    that reads as a gun (barrel, slide, trigger guard, grip) rather than
    an abstract block shape.
    """
    t = np.zeros((h, w, 4), dtype=np.uint8)
    skin      = np.array([200, 160, 120, 255])
    skin_dark = np.array([170, 130,  95, 255])
    metal     = np.array([ 55,  55,  60, 255])
    metal_lt  = np.array([ 90,  90,  98, 255])
    grip_c    = np.array([ 45,  35,  30, 255])

    cx = w // 2

    # Forearm/wrist entering from the bottom.
    t[110:h, cx-70:cx+70] = skin

    # Palm wrapping the grip.
    t[78:118, cx-55:cx+45] = skin
    t[95:125, cx-40:cx+10] = skin_dark   # shading where fingers wrap around

    # Grip (angled block under the slide).
    t[80:118, cx-15:cx+20] = grip_c

    # Trigger guard (a thin dark oval-ish loop).
    t[86:96, cx-28:cx-16] = metal
    t[92:102, cx-30:cx-14] = metal
    t[100:104, cx-28:cx-16] = metal

    # Slide (the main body of the pistol on top of the grip).
    t[55:82, cx-45:cx+35] = metal_lt
    t[58:76, cx-42:cx+32] = metal

    # Barrel, extending forward (toward the top of the sprite = away from
    # the player, matching the on-screen weapon-in-hand convention).
    t[38:58, cx-14:cx+14] = metal
    t[34:40, cx-10:cx+10] = metal_lt   # front sight nub

    # Thumb resting along the side of the slide.
    t[62:80, cx+30:cx+48] = skin

    return t


def make_muzzle_flash(s=48):
    """
    Small radial starburst used for the muzzle flash, rendered at the
    gun's barrel tip and faded out over MUZZLE_FLASH_DURATION. Replaces
    the earlier plain colour-blend rectangle with something that actually
    reads as a gunshot.
    """
    t = np.zeros((s, s, 4), dtype=np.uint8)
    y, x = np.mgrid[0:s, 0:s]
    cx = cy = s / 2.0
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    # An 8-point star: distance threshold modulated by angle so it isn't
    # a plain circle.
    spikes = 8
    star_r = (s * 0.46) * (0.55 + 0.45 * np.abs(np.cos(spikes * theta / 2)))
    core   = r <= (s * 0.16)
    star   = r <= star_r

    t[star] = [255, 200, 100, 200]
    t[core] = [255, 250, 220, 255]
    t[~star] = [0, 0, 0, 0]
    return t


# ---------------------------------------------------------------------------
# Load-or-generate-and-cache helpers
# ---------------------------------------------------------------------------

def load_rgb(path, make_fn):
    """Load (or generate) an RGB texture and return it as a uint8 array."""
    path = Path(path)
    if path.exists():
        img = Image.open(path).convert('RGB').resize((TEX_SIZE, TEX_SIZE), Image.NEAREST)
        return np.array(img, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = make_fn()
    Image.fromarray(arr).save(path)
    return arr


def load_rgba(path, make_fn, size=None):
    """Load (or generate) an RGBA sprite and return it as a uint8 array."""
    path = Path(path)
    if path.exists():
        img = Image.open(path).convert('RGBA')
        if size:
            img = img.resize(size, Image.NEAREST)
        return np.array(img, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = make_fn()
    Image.fromarray(arr).save(path)
    return arr


DEFAULT_WALL_COLORS = {
    1: (90, 90, 90),
    2: (150, 40, 40),
    3: (40, 150, 40),
    5: (150, 150, 40),
}


@dataclass
class TextureSet:
    wall_tex_stack: np.ndarray
    floor_tex: np.ndarray
    ceil_tex: np.ndarray
    sprite_tex: dict = field(default_factory=dict)
    hand_spr: np.ndarray = None
    muzzle_flash: np.ndarray = None


def build_textures(asset_dir, wall_colors=None) -> TextureSet:
    """
    Load (or procedurally generate + cache to asset_dir) every texture and
    sprite the renderer needs, keyed by wall-ID for walls and by type
    string for sprites.
    """
    asset_dir = Path(asset_dir)
    wall_colors = wall_colors or DEFAULT_WALL_COLORS

    wall_tex_stack = np.zeros((6, TEX_SIZE, TEX_SIZE, 3), dtype=np.uint8)
    for wid, col in wall_colors.items():
        wall_tex_stack[wid] = load_rgb(asset_dir / 'textures' / f'wall_{wid}.png',
                                        lambda c=col: make_brick(c))
    wall_tex_stack[config.DOOR_ID] = load_rgb(asset_dir / 'textures' / 'door.png', make_door)

    floor_tex = load_rgb(asset_dir / 'floor' / 'floor.png',
                          lambda: make_checker((50, 50, 50), (35, 35, 35)))
    ceil_tex = load_rgb(asset_dir / 'floor' / 'ceiling.png',
                         lambda: make_checker((30, 30, 40), (20, 20, 30)))

    sprite_tex = {
        'person': load_rgba(asset_dir / 'sprites' / 'person.png', make_person, (TEX_SIZE, TEX_SIZE)),
        'barrel': load_rgba(asset_dir / 'sprites' / 'barrel.png', make_barrel, (TEX_SIZE, TEX_SIZE)),
        'key':    load_rgba(asset_dir / 'sprites' / 'key.png',    make_key,    (TEX_SIZE, TEX_SIZE)),
        'flag':   load_rgba(asset_dir / 'sprites' / 'flag.png',   make_flag,   (TEX_SIZE, TEX_SIZE)),
    }

    hand_spr = load_rgba(asset_dir / 'assets' / 'hand.png', make_hand, (220, 170))
    muzzle_flash = load_rgba(asset_dir / 'assets' / 'muzzle_flash.png', make_muzzle_flash, (48, 48))

    return TextureSet(wall_tex_stack, floor_tex, ceil_tex, sprite_tex, hand_spr, muzzle_flash)
