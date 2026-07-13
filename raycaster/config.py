# =============================================================================
# config.py — all tunable constants in one place
# =============================================================================

WIDTH  = 640
HEIGHT = 408
HALF_H = HEIGHT // 2

# ---------------------------------------------------------------------------
# Internal render resolution
# ---------------------------------------------------------------------------
# The 3D scene (floor/ceiling/walls/sprites) is rendered at a reduced
# internal resolution and upscaled with nearest-neighbour to the window
# size. Pixel work for floor/ceiling projection and the wall-column gather
# scales with RENDER_W * RENDER_H, so this is the single biggest lever on
# FPS. See tools/benchmarks.py for measured FPS at several scales, and
# README.md for the results table. The HUD is drawn separately afterward
# at full WIDTH x HEIGHT so it always stays crisp.
RENDER_SCALE = 0.62
RENDER_W = max(64, int(WIDTH  * RENDER_SCALE))
RENDER_H = max(64, int(HEIGHT * RENDER_SCALE))
RENDER_HALF_H = RENDER_H // 2

# NOTE: these are PER-SECOND rates (dt-scaled in controls.apply_input), not
# per-frame deltas, so movement speed doesn't depend on FPS.
MOVE_SPEED = 4.8     # world units per second
ROT_SPEED  = 3.0     # radians per second
MOUSE_SENS = 0.0028  # radians of turn per pixel of mouse delta

# Escape hatch: A=left / D=right has been verified mathematically and
# empirically (see tests/test_gameplay.py::test_strafe_direction — a
# landmark placed at a known world offset is confirmed to render on the
# matching screen side, and D is confirmed to move toward it). If it still
# feels backwards on a given machine, the most likely cause is a
# non-US keyboard layout remapping which physical key sends keysym 'a'/'d'
# to tkinter. Flip this to -1 as an immediate fix without touching the
# strafe logic itself.
STRAFE_DIR = 1

PLAYER_RADIUS = 0.20   # collision box half-width, world units

TEX_SIZE = 64
TEX_MASK = TEX_SIZE - 1

# Distance-based lighting: brightness = exp(-LIGHT_DECAY * dist), floored
# at LIGHT_MIN so the furthest walls are still barely visible.
LIGHT_DECAY = 0.09
LIGHT_MIN   = 0.15

# Fog blends wall/floor colour toward FOG_COLOR as distance increases.
FOG_ONSET = 6.0
FOG_FULL  = 18.0
FOG_COLOR = (30, 30, 35)

DOOR_ID = 4
DOOR_OPEN_SPEED  = 2.2    # open/close fraction per second
DOOR_AUTOCLOSE_T = 3.5    # seconds a door stays open before auto-closing

PICKUP_RADIUS = 0.5

# Relative on-screen scale for each sprite type (1.0 = "occupies floor to
# ceiling like a wall", used as the reference for floor-anchored sizing in
# render.draw_sprites). Small floor items need a much smaller scale or
# they'd be stretched to person-height and clip into nearby walls.
SPRITE_SCALE = {
    'person': 0.85,
    'barrel': 0.55,
    'key':    0.28,
    'flag':   0.9,
}

# Sprite types that can be destroyed by shooting (previously only barrels
# were shootable — persons are now included too).
SHOOTABLE_TYPES = {'barrel', 'person'}

SHOOT_RANGE     = 14.0
SHOOT_COOLDOWN  = 0.35   # seconds between shots
SHOOT_HIT_WIDTH = 0.08   # camera-space half-width of the hit-scan "beam"
MUZZLE_FLASH_DURATION = 0.09
RECOIL_DURATION       = 0.16   # weapon-kick animation length, seconds
