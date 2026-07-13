# =============================================================================
# entities.py — Player, mutable game state, and all gameplay logic that
# isn't rendering: doors, pickups, the win condition, and shooting.
# =============================================================================

from dataclasses import dataclass, field

import numpy as np

from . import config
from .math_utils import rotation_2d


class Player:
    """
    Holds the player's position, view direction and camera plane.

    Coordinate system
    -----------------
    px, py   - world position (floats, measured in map cells)
    dx, dy   - unit view direction vector
    plx, ply - camera plane vector (perpendicular to direction, length ~0.66
               which gives a ~66 degree horizontal field of view)
    """

    __slots__ = ('px', 'py', 'dx', 'dy', 'plx', 'ply', 'moving', 'bob_phase')

    def __init__(self, px=1.5, py=1.5, dx=-1.0, dy=0.0, plx=0.0, ply=0.66):
        self.px,  self.py  = px,  py
        self.dx,  self.dy  = dx,  dy
        self.plx, self.ply = plx, ply
        self.moving    = False
        self.bob_phase = 0.0

    def move(self, speed, level):
        """Move forward (+) or backward (-) along the view direction."""
        nx = self.px + self.dx * speed
        ny = self.py + self.dy * speed
        # Axis-separated collision: test each axis independently so the
        # player can slide along walls rather than stopping dead.
        if level.cell_passable(nx, self.py): self.px = nx
        if level.cell_passable(self.px, ny): self.py = ny
        self.moving = True

    def strafe(self, speed, level):
        """
        Move sideways relative to the view direction. Positive speed =
        right (matches the camera plane / screen-right, i.e. the 'D' key).
        Verified in tests/test_gameplay.py::test_strafe_direction.
        """
        speed *= config.STRAFE_DIR
        nx = self.px + self.dy * speed
        ny = self.py - self.dx * speed
        if level.cell_passable(nx, self.py): self.px = nx
        if level.cell_passable(self.px, ny): self.py = ny
        self.moving = True

    def rotate(self, angle):
        """Rotate the view and camera plane by `angle` radians."""
        R = rotation_2d(angle)
        d = (R[0][0] * self.dx  + R[0][1] * self.dy,
             R[1][0] * self.dx  + R[1][1] * self.dy)
        p = (R[0][0] * self.plx + R[0][1] * self.ply,
             R[1][0] * self.plx + R[1][1] * self.ply)
        self.dx,  self.dy  = d
        self.plx, self.ply = p


@dataclass
class GameState:
    """
    Everything that changes over the course of a run: the player, per-run
    flags (has_key, game_won), collected pickups, and short-lived UI
    timers (HUD message, muzzle flash, weapon recoil). Bundled into one
    object instead of scattered module-level globals so render/gameplay
    functions take explicit state rather than reading hidden module state.
    """
    level: object
    textures: object
    player: Player = field(default_factory=Player)

    has_key: bool = False
    collected_keys: set = field(default_factory=set)
    game_won: bool = False

    hud_message: str = ''
    hud_message_t: float = 0.0

    muzzle_flash_t: float = 0.0
    recoil_t: float = 0.0
    last_shot_t: float = -999.0

    minimap: object = None   # populated by hud.build_minimap() in main.py

    def show_message(self, text, duration=1.6):
        self.hud_message = text
        self.hud_message_t = duration

    @classmethod
    def new_game(cls, level, textures):
        sp = level.spawn
        player = Player(**sp)
        return cls(level=level, textures=textures, player=player)


# ---------------------------------------------------------------------------
# Doors
# ---------------------------------------------------------------------------

_door_timers: dict = {}   # cell -> seconds remaining before auto-close, per level


def update_doors(state, dt):
    """Animate every door's open_amount toward its target, and auto-close
    doors that have been open (unobstructed) past DOOR_AUTOCLOSE_T."""
    for cell, door in state.level.door_cells.items():
        if cell in _door_timers:
            _door_timers[cell] -= dt
            if _door_timers[cell] <= 0:
                door.target = 0.0
                del _door_timers[cell]

        step = config.DOOR_OPEN_SPEED * dt
        if door.open_amount < door.target:
            door.open_amount = min(door.target, door.open_amount + step)
        elif door.open_amount > door.target:
            door.open_amount = max(door.target, door.open_amount - step)


def _find_facing_door(state, max_range=1.5):
    """Return the cell of the nearest door the player is roughly facing
    within max_range, or None."""
    player = state.player
    best_cell, best_dist = None, max_range
    for (cx, cy) in state.level.door_cells:
        dxp, dyp = (cx + 0.5) - player.px, (cy + 0.5) - player.py
        dist = (dxp ** 2 + dyp ** 2) ** 0.5
        if dist > best_dist:
            continue
        facing = (dxp * player.dx + dyp * player.dy) / max(dist, 1e-6)
        if facing < 0.3:
            continue
        best_cell, best_dist = (cx, cy), dist
    return best_cell


def try_interact(state):
    """Handle the 'E' key: toggle the nearest facing door open/closed."""
    cell = _find_facing_door(state)
    if cell is None:
        return
    door = state.level.door_cells[cell]

    if door.locked:
        if state.has_key:
            door.locked = False
            state.show_message('Unlocked the door.')
        else:
            state.show_message('The door is locked. Find a key.')
            return

    if door.target > 0.0:
        door.target = 0.0
        _door_timers.pop(cell, None)
    else:
        door.target = 1.0
        _door_timers[cell] = config.DOOR_AUTOCLOSE_T


# ---------------------------------------------------------------------------
# Pickups & win condition
# ---------------------------------------------------------------------------

def update_pickups(state):
    """Check for key pickups within range of the player and collect them."""
    for i, (kx, ky) in enumerate(state.level.key_pickups):
        if i in state.collected_keys:
            continue
        p = state.player
        if (kx - p.px) ** 2 + (ky - p.py) ** 2 <= config.PICKUP_RADIUS ** 2:
            state.collected_keys.add(i)
            state.has_key = True
            state.show_message('Picked up a key!')


def update_win(state):
    """Check whether the player has reached the level's exit cell."""
    if state.game_won:
        return
    ex, ey = state.level.exit_cell
    p = state.player
    dx, dy = ex - p.px, ey - p.py
    if dx * dx + dy * dy <= state.level.exit_radius ** 2:
        state.game_won = True
        state.show_message('Level complete!', duration=999)


# ---------------------------------------------------------------------------
# Shooting
# ---------------------------------------------------------------------------

def try_shoot(state, now):
    """
    Hitscan straight down the crosshair: find the nearest shootable
    sprite (see config.SHOOTABLE_TYPES — previously only barrels, now
    also persons) close enough to dead-centre, closer than any other hit
    candidate, and within SHOOT_RANGE. Always triggers muzzle flash +
    weapon recoil regardless of whether anything was hit.
    """
    if now - state.last_shot_t < config.SHOOT_COOLDOWN:
        return False
    state.last_shot_t = now
    state.muzzle_flash_t = config.MUZZLE_FLASH_DURATION
    state.recoil_t = config.RECOIL_DURATION

    player = state.player
    px, py   = player.px, player.py
    dx, dy   = player.dx, player.dy
    plx, ply = player.plx, player.ply
    inv_det = 1.0 / (plx * dy - dx * ply + 1e-30)

    sprites = state.level.sprites
    best_i, best_z = None, config.SHOOT_RANGE
    for i, (sx, sy, stype) in enumerate(sprites):
        if stype not in config.SHOOTABLE_TYPES:
            continue
        rx, ry = sx - px, sy - py
        cam_xs = inv_det * (dy * rx - dx * ry)
        cam_z  = inv_det * (-ply * rx + plx * ry)
        if cam_z <= 0.15 or cam_z >= best_z:
            continue
        if abs(cam_xs / cam_z) > config.SHOOT_HIT_WIDTH:
            continue
        best_i, best_z = i, cam_z

    if best_i is not None:
        _, _, stype = sprites[best_i]
        del sprites[best_i]
        state.show_message(f'{stype.capitalize()} destroyed!')
        return True
    return False


# ---------------------------------------------------------------------------
# Weapon bob
# ---------------------------------------------------------------------------

BOB_FREQ = 2.8
BOB_AMP  = 12


def update_bob(state, dt):
    player = state.player
    if player.moving:
        player.bob_phase += dt * BOB_FREQ * 2 * np.pi
    else:
        player.bob_phase = 0.0
    player.moving = False
