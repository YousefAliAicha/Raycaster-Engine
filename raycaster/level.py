# =============================================================================
# level.py — loads level layout/entity placement from a JSON file.
#
# Map data, sprite placement, door/key configuration and the exit location
# used to be hardcoded directly in the render/gameplay code. That's fine
# for a single throwaway map, but it means every new layout requires
# editing Python. Pulling it out to data/level1.json separates "what the
# level looks like" from "how the engine renders/simulates it" — you can
# now build a second level (see data/level2.json for a stub) without
# touching engine code at all.
# =============================================================================

import json
from pathlib import Path

import numpy as np

from . import config


class DoorState:
    """Runtime state for one door cell. Not persisted — reset on load."""
    __slots__ = ('open_amount', 'target', 'locked')

    def __init__(self, locked=False):
        self.open_amount = 0.0   # 0 = fully closed, 1 = fully open
        self.target      = 0.0   # animates toward this value
        self.locked      = locked


class Level:
    """
    Everything derived from a level JSON file: the wall grid, door
    states, sprite/pickup placement, spawn pose, and per-wall-ID colours
    (used as a fallback tint if no texture is supplied).
    """

    def __init__(self, data):
        self.name = data.get('name', 'Untitled')

        grid = data['grid']
        self.world_map = np.array(grid, dtype=np.int32)
        self.map_rows, self.map_cols = self.world_map.shape

        self.wall_colors = {int(k): tuple(v) for k, v in data.get('wall_colors', {}).items()}

        self.door_cells = {}
        for entry in data.get('doors', []):
            cell = tuple(entry['cell'])
            self.door_cells[cell] = DoorState(locked=entry.get('locked', False))
        # Any grid cell tagged DOOR_ID that wasn't explicitly listed still
        # needs a (unlocked) DoorState so rendering/collision works.
        door_rows, door_cols = np.where(self.world_map == config.DOOR_ID)
        for r, c in zip(door_rows.tolist(), door_cols.tolist()):
            self.door_cells.setdefault((r, c), DoorState(locked=False))

        self.sprites = [
            [s['x'], s['y'], s['type']] for s in data.get('sprites', [])
        ]

        self.key_pickups = [(k['x'], k['y']) for k in data.get('key_pickups', [])]

        ex = data.get('exit_cell', {'x': 0, 'y': 0, 'radius': 0.6})
        self.exit_cell = (ex['x'], ex['y'])
        self.exit_radius = ex.get('radius', 0.6)

        sp = data.get('spawn', {})
        self.spawn = dict(
            px=sp.get('px', 1.5), py=sp.get('py', 1.5),
            dx=sp.get('dx', -1.0), dy=sp.get('dy', 0.0),
            plx=sp.get('plx', 0.0), ply=sp.get('ply', 0.66),
        )

        # Door cells as parallel index arrays, for fast per-frame masking
        # in render.py without a Python-level dict lookup per ray.
        keys = list(self.door_cells.keys())
        self._door_keys = keys
        self.door_r = np.array([k[0] for k in keys], dtype=np.int32) if keys else np.zeros(0, dtype=np.int32)
        self.door_c = np.array([k[1] for k in keys], dtype=np.int32) if keys else np.zeros(0, dtype=np.int32)

    def current_hit_map(self):
        """
        WORLD_MAP is static (used for wall-ID/texture lookups), but ray
        hit-testing needs doors to stop blocking once they're open enough
        to walk through. Returns a fresh copy each frame with sufficiently
        open door cells zeroed.
        """
        hit_map = self.world_map
        if self._door_keys:
            amounts = np.array([self.door_cells[k].open_amount for k in self._door_keys])
            open_enough = amounts > 0.85
            if open_enough.any():
                hit_map = self.world_map.copy()
                hit_map[self.door_r[open_enough], self.door_c[open_enough]] = 0
        return hit_map

    def cell_passable(self, nx, ny):
        """Collision test: is the PLAYER_RADIUS box around (nx, ny) clear?"""
        r = config.PLAYER_RADIUS
        for ox, oy in ((-r, -r), (-r, r), (r, -r), (r, r)):
            cx, cy = int(nx + ox), int(ny + oy)
            if not (0 <= cx < self.map_rows and 0 <= cy < self.map_cols):
                return False
            cell = self.world_map[cx, cy]
            if cell == 0:
                continue
            if cell == config.DOOR_ID:
                door = self.door_cells.get((cx, cy))
                if door is not None and door.open_amount > 0.85:
                    continue
                return False
            return False
        return True


def load_level(path=None) -> Level:
    if path is None:
        path = Path(__file__).parent / 'data' / 'level1.json'
    with open(path) as f:
        data = json.load(f)
    return Level(data)
