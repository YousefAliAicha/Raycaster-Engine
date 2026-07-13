# =============================================================================
# hud.py — everything drawn at full window resolution on top of the
# upscaled 3D scene: mini-map, debug panel, weapon (with recoil), muzzle
# flash, interact prompts, transient messages, and the win overlay.
# =============================================================================

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config
from .math_utils import mc_sin, absolute
from .entities import _find_facing_door, BOB_AMP

WIDTH, HEIGHT, HALF_H = config.WIDTH, config.HEIGHT, config.HALF_H

try:
    _FONT = ImageFont.truetype('arial.ttf', 14)
    _BIG_FONT = ImageFont.truetype('arial.ttf', 32)
except Exception:
    _FONT = ImageFont.load_default()
    _BIG_FONT = _FONT

DOOR_MM_LOCKED = (220,  60,  60)
DOOR_MM_CLOSED = (255, 165,  40)
DOOR_MM_OPEN   = ( 90, 220, 120)

_WCOLORS = {
    1: (100, 100, 100),
    2: (150,  50,  50),
    3: ( 50, 150,  50),
    5: (150, 150,  50),
}


@dataclass
class Minimap:
    base: np.ndarray
    mask: np.ndarray
    oy: int
    ox: int
    w: int
    h: int
    cell: int


def build_minimap(level, cell=6, margin=10):
    """Precompute the static (per-level) mini-map tile colours/mask once,
    so draw_hud only needs to composite it each frame."""
    rows, cols = level.map_rows, level.map_cols
    w, h = cols * cell, rows * cell
    base = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)

    for r in range(rows):
        for c in range(cols):
            v = int(level.world_map[r, c])
            if v > 0 and v != config.DOOR_ID:
                colour = _WCOLORS.get(v, (80, 80, 80))
                base[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = colour
                mask[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = True

    return Minimap(base=base, mask=mask[:, :, None], oy=margin, ox=WIDTH - w - margin,
                    w=w, h=h, cell=cell)


def _fwd_dist(state):
    """Perpendicular distance to the wall directly ahead, via a scalar DDA
    matching the vectorised renderer (used only for the HUD readout)."""
    level = state.level
    player = state.player
    px, py   = player.px, player.py
    rdx, rdy = player.dx, player.dy
    mx, my   = int(px), int(py)

    dlx = 1e30 if rdx == 0 else absolute(1.0 / rdx)
    dly = 1e30 if rdy == 0 else absolute(1.0 / rdy)
    stx = -1 if rdx < 0 else 1
    sty = -1 if rdy < 0 else 1
    sdx = (px - mx) * dlx if rdx < 0 else (mx + 1.0 - px) * dlx
    sdy = (py - my) * dly if rdy < 0 else (my + 1.0 - py) * dly

    side = 0
    for _ in range(level.map_rows + level.map_cols):
        if sdx < sdy: sdx += dlx; mx += stx; side = 0
        else:         sdy += dly; my += sty; side = 1
        if 0 <= mx < level.map_rows and 0 <= my < level.map_cols and level.world_map[mx, my] > 0:
            break
    return (sdx - dlx) if side == 0 else (sdy - dly)


def draw_hud(state, frame, z_buffer, fps):
    """
    Composite all HUD elements onto `frame` in-place:
      1/2. Mini-map (background dim + wall tiles + door colours by state)
      3.   Player dot & direction arrow
      4.   Debug text panel
      5.   Weapon hand, with head-bob AND firing recoil
      5b.  Door interact prompt
      6.   Crosshair
      6b.  Muzzle flash (a real starburst sprite, not a flat colour blend)
      7.   Transient message
      8.   Win overlay
    """
    mm = state.minimap
    level = state.level
    player = state.player
    tex = state.textures

    # ── 1 & 2: Mini-map ──────────────────────────────────────────────────
    oy, ox = mm.oy, mm.ox
    region = frame[oy:oy + mm.h, ox:ox + mm.w]
    bg = (region * 0.25).astype(np.uint8)
    frame[oy:oy + mm.h, ox:ox + mm.w] = np.where(mm.mask, mm.base, bg)

    for (dr, dc), door in level.door_cells.items():
        colour = (DOOR_MM_LOCKED if door.locked else
                  DOOR_MM_OPEN if door.open_amount > 0.5 else
                  DOOR_MM_CLOSED)
        y0, x0 = oy + dr * mm.cell, ox + dc * mm.cell
        frame[y0:y0 + mm.cell, x0:x0 + mm.cell] = colour

    # ── 3a: Player dot ──────────────────────────────────────────────────
    pdx = np.clip(int(ox + player.py * mm.cell), ox, ox + mm.w - 1)
    pdy = np.clip(int(oy + player.px * mm.cell), oy, oy + mm.h - 1)
    frame[pdy - 2:pdy + 3, pdx - 2:pdx + 3] = (255, 50, 50)

    # ── 3b: Direction arrow ─────────────────────────────────────────────
    ax = int(pdx + player.dy * 9)
    ay = int(pdy + player.dx * 9)
    ts = np.linspace(0, 1, 10)
    lxs = np.clip((pdx + ts * (ax - pdx)).astype(int), ox, ox + mm.w - 1)
    lys = np.clip((pdy + ts * (ay - pdy)).astype(int), oy, oy + mm.h - 1)
    frame[lys, lxs] = (60, 230, 230)

    # ── 4: Debug text panel ─────────────────────────────────────────────
    fwd = _fwd_dist(state)
    TW, TH = 270, 88
    surf = Image.new('RGB', (TW, TH), (0, 0, 0))
    d = ImageDraw.Draw(surf)
    d.text((8,  4), f'POS  X:{player.px:.2f}  Y:{player.py:.2f}', fill=(220, 220, 220), font=_FONT)
    d.text((8, 24), f'DIR  X:{player.dx:.2f}  Y:{player.dy:.2f}', fill=(220, 220, 220), font=_FONT)
    d.text((8, 44), f'WALL DIST: {fwd:.2f} units', fill=(255, 200, 50), font=_FONT)
    d.text((8, 64), f'FPS: {fps}   KEY: {"YES" if state.has_key else "no"}', fill=(100, 220, 100), font=_FONT)
    txt = np.array(surf, dtype=np.uint8)
    frame[6:6 + TH, 6:6 + TW] = (frame[6:6 + TH, 6:6 + TW] * 0.30 + txt * 0.70).astype(np.uint8)

    # ── 5: Weapon hand — head-bob AND firing recoil ─────────────────────
    hh, hw = tex.hand_spr.shape[:2]
    hx = WIDTH - hw - 8

    bob_offset = int(BOB_AMP * mc_sin(player.bob_phase))

    # Recoil: quick kick down-and-right then ease back to rest across
    # config.RECOIL_DURATION. This is what actually makes firing feel
    # like it did something, versus the flash-only version before.
    recoil_frac = max(0.0, state.recoil_t / config.RECOIL_DURATION)
    kick = recoil_frac * recoil_frac   # eased: sharp kick, slower return
    recoil_y = int(22 * kick)
    recoil_x = int(8 * kick)

    hy = HEIGHT - hh - 4 + bob_offset + recoil_y
    hy = max(0, min(HEIGHT - hh, hy))
    hxr = min(WIDTH - hw, hx + recoil_x)

    if hxr >= 0:
        a   = tex.hand_spr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = tex.hand_spr[:, :, :3].astype(np.float32)
        roi = frame[hy:hy + hh, hxr:hxr + hw].astype(np.float32)
        frame[hy:hy + hh, hxr:hxr + hw] = (rgb * a + roi * (1.0 - a)).astype(np.uint8)

    # ── 5b: Door interact prompt ────────────────────────────────────────
    facing_cell = _find_facing_door(state)
    if facing_cell is not None:
        door = level.door_cells[facing_cell]
        if door.locked:
            prompt = '[E] Locked - need a key' if not state.has_key else '[E] Unlock door'
        else:
            prompt = '[E] Close door' if door.target > 0.0 else '[E] Open door'

        psurf = Image.new('RGBA', (220, 22), (0, 0, 0, 0))
        pd = ImageDraw.Draw(psurf)
        pd.text((0, 2), prompt, fill=(255, 230, 150, 255), font=_FONT)
        parr = np.array(psurf)
        ph, pw = parr.shape[:2]
        px0 = WIDTH // 2 - pw // 2
        py0 = HALF_H - 34
        palpha = parr[:, :, 3:4].astype(np.float32) / 255.0
        proi = frame[py0:py0 + ph, px0:px0 + pw].astype(np.float32)
        frame[py0:py0 + ph, px0:px0 + pw] = (
            parr[:, :, :3].astype(np.float32) * palpha + proi * (1 - palpha)
        ).astype(np.uint8)

    # ── 6: Crosshair ─────────────────────────────────────────────────────
    cx, cy = WIDTH // 2, HALF_H
    frame[cy, cx - 8:cx + 9] = (200, 200, 200)
    frame[cy - 8:cy + 9, cx] = (200, 200, 200)

    # ── 6b: Muzzle flash — real starburst sprite at the barrel tip ─────
    if state.muzzle_flash_t > 0 and tex.muzzle_flash is not None:
        strength = min(1.0, state.muzzle_flash_t / config.MUZZLE_FLASH_DURATION)
        fh, fw = tex.muzzle_flash.shape[:2]
        # Anchor near the gun's barrel: just above/left of the hand sprite.
        fx0 = hxr + hw // 2 - fw // 2 - 6
        fy0 = hy - fh // 2 + 14
        fx0 = max(0, min(WIDTH - fw, fx0))
        fy0 = max(0, min(HEIGHT - fh, fy0))
        a = (tex.muzzle_flash[:, :, 3:4].astype(np.float32) / 255.0) * strength
        rgb = tex.muzzle_flash[:, :, :3].astype(np.float32)
        roi = frame[fy0:fy0 + fh, fx0:fx0 + fw].astype(np.float32)
        frame[fy0:fy0 + fh, fx0:fx0 + fw] = (rgb * a + roi * (1 - a)).astype(np.uint8)

    # ── 7: Transient message ────────────────────────────────────────────
    if state.hud_message_t > 0 and state.hud_message:
        msurf = Image.new('RGBA', (300, 26), (0, 0, 0, 0))
        md = ImageDraw.Draw(msurf)
        md.text((0, 4), state.hud_message, fill=(255, 255, 255, 255), font=_FONT)
        marr = np.array(msurf)
        mh, mw = marr.shape[:2]
        mx0 = WIDTH // 2 - mw // 2
        my0 = HALF_H + 40
        alpha = marr[:, :, 3:4].astype(np.float32) / 255.0
        roi = frame[my0:my0 + mh, mx0:mx0 + mw].astype(np.float32)
        frame[my0:my0 + mh, mx0:mx0 + mw] = (
            marr[:, :, :3].astype(np.float32) * alpha + roi * (1 - alpha)
        ).astype(np.uint8)

    # ── 8: Win overlay ───────────────────────────────────────────────────
    if state.game_won:
        frame[:, :, :] = (frame.astype(np.float32) * 0.35).astype(np.uint8)
        wsurf = Image.new('RGBA', (WIDTH, 120), (0, 0, 0, 0))
        wd = ImageDraw.Draw(wsurf)
        wd.text((WIDTH // 2 - 140, 20), 'LEVEL COMPLETE', fill=(255, 220, 120, 255), font=_BIG_FONT)
        wd.text((WIDTH // 2 - 110, 70), 'Press Q or Esc to quit', fill=(220, 220, 220, 255), font=_FONT)
        warr = np.array(wsurf)
        wy0 = HALF_H - 60
        walpha = warr[:, :, 3:4].astype(np.float32) / 255.0
        wroi = frame[wy0:wy0 + 120, 0:WIDTH].astype(np.float32)
        frame[wy0:wy0 + 120, 0:WIDTH] = (
            warr[:, :, :3].astype(np.float32) * walpha + wroi * (1 - walpha)
        ).astype(np.uint8)
