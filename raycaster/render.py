# =============================================================================
# render.py — the 3D scene renderer: floor/ceiling projection, vectorised
# DDA wall raycasting, lighting/fog, and sprite compositing.
#
# Everything here operates at the INTERNAL render resolution (config.RENDER_W
# x config.RENDER_H), not the window size — see config.RENDER_SCALE. The
# caller (main.run_engine) upscales the result and draws the HUD separately
# at full resolution.
#
# Zero per-column Python loops: wall rasterisation, floor/ceiling
# projection, and sprite column compositing are all done as single NumPy
# gather/broadcast operations over the whole frame.
# =============================================================================

import numpy as np

from . import config

RENDER_W, RENDER_H, RENDER_HALF_H = config.RENDER_W, config.RENDER_H, config.RENDER_HALF_H
TEX_SIZE, TEX_MASK = config.TEX_SIZE, config.TEX_MASK
FOG_COLOR = np.array(config.FOG_COLOR, dtype=np.float32)

_COL       = np.arange(RENDER_W, dtype=np.float64)
_CAMX      = 2.0 * _COL / RENDER_W - 1.0
_FLOOR_ROW = np.arange(RENDER_HALF_H + 1, RENDER_H, dtype=np.float64)


def render(state, frame):
    """
    Fill `frame` (RENDER_H x RENDER_W x 3 uint8) with the current view and
    return the z-buffer (perpendicular wall distance per render-space
    column), used both for sprite occlusion and by the HUD's forward-
    distance readout.
    """
    level = state.level
    tex   = state.textures
    player = state.player

    px, py   = player.px, player.py
    dx, dy   = player.dx, player.dy
    plx, ply = player.plx, player.ply

    MAP_ROWS, MAP_COLS = level.map_rows, level.map_cols

    # ── 1. Floor & ceiling ────────────────────────────────────────────────
    row_dist = RENDER_HALF_H / (_FLOOR_ROW - RENDER_HALF_H)
    step_x = row_dist * (2.0 * plx / RENDER_W)
    step_y = row_dist * (2.0 * ply / RENDER_W)
    fx0 = px + row_dist * (dx - plx)
    fy0 = py + row_dist * (dy - ply)

    col_range = np.arange(RENDER_W, dtype=np.float64)
    fx = fx0[:, None] + step_x[:, None] * col_range[None, :]
    fy = fy0[:, None] + step_y[:, None] * col_range[None, :]

    tx = (fx * TEX_SIZE).astype(np.int32) & TEX_MASK
    ty = (fy * TEX_SIZE).astype(np.int32) & TEX_MASK
    tx = np.clip(tx, 0, TEX_MASK)
    ty = np.clip(ty, 0, TEX_MASK)

    frame[RENDER_HALF_H + 1:RENDER_H, :] = tex.floor_tex[ty, tx]
    ceil_y = (RENDER_H - 1 - _FLOOR_ROW).astype(int)
    frame[ceil_y, :] = tex.ceil_tex[ty, tx]

    # ── 2. Walls — vectorised DDA ────────────────────────────────────────
    rdx = dx + plx * _CAMX
    rdy = dy + ply * _CAMX

    mx = np.full(RENDER_W, int(px), dtype=np.int32)
    my = np.full(RENDER_W, int(py), dtype=np.int32)

    with np.errstate(divide='ignore', invalid='ignore'):
        dlx = np.where(rdx == 0, 1e30, np.abs(1.0 / rdx))
        dly = np.where(rdy == 0, 1e30, np.abs(1.0 / rdy))

    stx = np.where(rdx < 0, -1, 1).astype(np.int32)
    sty = np.where(rdy < 0, -1, 1).astype(np.int32)

    sdx = np.where(rdx < 0, (px - mx) * dlx, (mx + 1.0 - px) * dlx)
    sdy = np.where(rdy < 0, (py - my) * dly, (my + 1.0 - py) * dly)

    hit  = np.zeros(RENDER_W, dtype=bool)
    side = np.zeros(RENDER_W, dtype=np.int32)
    hit_map = level.current_hit_map()

    for _ in range(MAP_ROWS + MAP_COLS):
        if hit.all():
            break
        alive = ~hit
        go_x  = alive & (sdx < sdy)
        go_y  = alive & ~go_x

        sdx  = np.where(go_x, sdx + dlx, sdx)
        mx   = np.where(go_x, mx  + stx, mx)
        side = np.where(go_x, 0, side)

        sdy  = np.where(go_y, sdy + dly, sdy)
        my   = np.where(go_y, my  + sty, my)
        side = np.where(go_y, 1, side)

        cmx = np.clip(mx, 0, MAP_ROWS - 1)
        cmy = np.clip(my, 0, MAP_COLS - 1)
        hit |= (hit_map[cmx, cmy] > 0)

    # ── 3. Wall geometry ─────────────────────────────────────────────────
    perp = np.where(side == 0, sdx - dlx, sdy - dly)
    perp = np.maximum(perp, 0.0001)
    z_buffer = perp.copy()

    wall_h     = np.maximum(1, (RENDER_H / perp).astype(np.int32))
    draw_start = np.maximum(0,       -wall_h // 2 + RENDER_HALF_H)
    draw_end   = np.minimum(RENDER_H, wall_h // 2 + RENDER_HALF_H)

    wall_hit = np.where(side == 0, py + perp * rdy, px + perp * rdx)
    wall_hit -= np.floor(wall_hit)
    tex_x = (wall_hit * TEX_SIZE).astype(np.int32)

    flip  = ((side == 0) & (rdx > 0)) | ((side == 1) & (rdy < 0))
    tex_x = np.where(flip, TEX_MASK - tex_x, tex_x)
    tex_x = np.clip(tex_x, 0, TEX_MASK)

    cmx      = np.clip(mx, 0, MAP_ROWS - 1)
    cmy      = np.clip(my, 0, MAP_COLS - 1)
    wall_ids = np.clip(level.world_map[cmx, cmy], 1, 5)

    # ── 4. Lighting & fog ────────────────────────────────────────────────
    brightness = np.exp(-config.LIGHT_DECAY * perp)
    brightness = np.clip(brightness, config.LIGHT_MIN, 1.0)
    brightness = np.where(side == 1, brightness * 0.65, brightness)

    fog = np.clip((perp - config.FOG_ONSET) / (config.FOG_FULL - config.FOG_ONSET), 0.0, 1.0)

    # ── 5. Door slide-up visual ──────────────────────────────────────────
    # Bottom recedes first: as open_amount increases, draw_end shrinks
    # upward, so the door appears to retract up and out of frame rather
    # than the top shrinking down toward the floor.
    is_door_col = (wall_ids == config.DOOR_ID)
    door_open_amt = np.zeros(RENDER_W, dtype=np.float32)
    if is_door_col.any() and level.door_cells:
        amounts_map = np.zeros((MAP_ROWS, MAP_COLS), dtype=np.float32)
        for (dr, dc), st in level.door_cells.items():
            amounts_map[dr, dc] = st.open_amount
        door_open_amt = amounts_map[cmx, cmy]

    slide_px = (door_open_amt * (draw_end - draw_start).astype(np.float32)) * 0.98
    draw_end_v = np.where(is_door_col,
                           np.maximum(draw_start, draw_end - slide_px.astype(np.int32)),
                           draw_end)
    draw_start_v = draw_start

    # ── 6. Rasterise every wall column at once (fully vectorised) ───────
    rows2d  = np.arange(RENDER_H, dtype=np.float32)[:, None]
    wh_f    = np.maximum(1, wall_h).astype(np.float32)[None, :]
    ty_f    = (rows2d - RENDER_HALF_H + wh_f * 0.5) * (TEX_SIZE / wh_f)
    ty_grid = np.clip(ty_f.astype(np.int32), 0, TEX_MASK)

    row_mask = (rows2d >= draw_start_v[None, :]) & (rows2d < draw_end_v[None, :])

    colour = tex.wall_tex_stack[wall_ids[None, :], ty_grid, tex_x[None, :]].astype(np.float32)
    colour *= brightness[None, :, None]

    fog_b = fog[None, :, None]
    colour = colour * (1.0 - fog_b) + FOG_COLOR[None, None, :] * fog_b

    colour = np.clip(colour, 0, 255).astype(np.uint8)
    frame[:, :, :] = np.where(row_mask[:, :, None], colour, frame)

    return z_buffer


def draw_sprites(state, frame, z_buffer):
    """
    Billboard every sprite (persons, barrels, key pickups, the exit flag)
    into `frame`, occluded per-column against z_buffer. Floor-anchored by
    type (config.SPRITE_SCALE) so small items like the key sit on the
    ground instead of being stretched to person-height.
    """
    level = state.level
    tex   = state.textures
    player = state.player
    px, py   = player.px, player.py
    dx, dy   = player.dx, player.dy
    plx, ply = player.plx, player.ply

    inv_det = 1.0 / (plx * dy - dx * ply + 1e-30)

    dynamic_sprites = list(level.sprites) + [
        (kx, ky, 'key') for i, (kx, ky) in enumerate(level.key_pickups)
        if i not in state.collected_keys
    ] + [(level.exit_cell[0], level.exit_cell[1], 'flag')]

    ordered = sorted(
        dynamic_sprites,
        key=lambda s: (s[0] - px) ** 2 + (s[1] - py) ** 2,
        reverse=True
    )

    for (sx, sy, stype) in ordered:
        spr = tex.sprite_tex.get(stype)
        if spr is None:
            continue
        th, tw = spr.shape[:2]
        scale = config.SPRITE_SCALE.get(stype, 1.0)

        rx = sx - px
        ry = sy - py
        cam_xs = inv_det * ( dy * rx - dx * ry)
        cam_z  = inv_det * (-ply * rx + plx * ry)
        if cam_z <= 0.1:
            continue

        scr_x       = int((RENDER_W / 2) * (1 + cam_xs / cam_z))
        proj_h_full = abs(int(RENDER_H / cam_z))
        if proj_h_full == 0:
            continue
        proj_h = max(1, int(proj_h_full * scale))
        proj_w = proj_h

        # Floor-anchor: bottom of the sprite sits on the floor line at this
        # distance (same line a wall's base would occupy here).
        floor_y = RENDER_HALF_H + proj_h_full // 2
        de_y = min(RENDER_H, floor_y)
        ds_y = max(0,         floor_y - proj_h)
        ds_x = max(0,         scr_x - proj_w // 2)
        de_x = min(RENDER_W,  scr_x + proj_w // 2)
        if de_y <= ds_y or de_x <= ds_x:
            continue

        screen_ys = np.arange(ds_y, de_y, dtype=np.float32)
        ty = ((screen_ys - ds_y) * th / proj_h).astype(np.int32)
        ty = np.clip(ty, 0, th - 1)

        screen_xs = np.arange(ds_x, de_x, dtype=np.int32)
        txi = ((screen_xs - (scr_x - proj_w // 2)) * tw / proj_w).astype(np.int32)
        txi = np.clip(txi, 0, tw - 1)

        occluded = cam_z >= z_buffer[ds_x:de_x]
        rgba = spr[ty[:, None], txi[None, :]]
        visible = (rgba[:, :, 3] >= 128) & ~occluded[None, :]

        if visible.any():
            dest = frame[ds_y:de_y, ds_x:de_x]
            frame[ds_y:de_y, ds_x:de_x] = np.where(visible[:, :, None], rgba[:, :, :3], dest)
