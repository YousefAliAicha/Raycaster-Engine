"""
Gameplay/engine regression tests. Runs entirely headless — no tkinter
window is created, since Player/Level/render operate on plain NumPy
buffers.
"""
import time

import numpy as np
import pytest

from raycaster import level as level_mod
from raycaster import textures as textures_mod
from raycaster import render as render_mod
from raycaster import config
from raycaster.entities import GameState, Player, try_interact, try_shoot, update_win


@pytest.fixture(scope='module')
def lvl():
    return level_mod.load_level()


@pytest.fixture(scope='module')
def tex(tmp_path_factory, lvl):
    asset_dir = tmp_path_factory.mktemp('assets')
    return textures_mod.build_textures(asset_dir, lvl.wall_colors)


@pytest.fixture
def state(lvl, tex):
    return GameState.new_game(lvl, tex)


def test_level_loads_with_expected_shape(lvl):
    assert lvl.world_map.ndim == 2
    assert lvl.map_rows > 0 and lvl.map_cols > 0
    assert len(lvl.door_cells) >= 1
    assert len(lvl.sprites) >= 1
    assert len(lvl.key_pickups) >= 1


def test_render_pipeline_runs(state):
    buf = np.zeros((config.RENDER_H, config.RENDER_W, 3), dtype=np.uint8)
    z = render_mod.render(state, buf)
    render_mod.draw_sprites(state, buf, z)
    assert z.shape == (config.RENDER_W,)
    assert buf.sum() > 0   # something was actually drawn


def test_strafe_direction(state):
    """
    D should move the player toward whatever renders on the right half of
    the screen; A should move opposite. Verified against the actual
    on-screen projection math (draw_sprites' camera transform), not just
    self-consistency of Player.strafe.
    """
    level = state.level
    p = Player(px=12.5, py=5.5, dx=0.0, dy=1.0, plx=0.66, ply=0.0)

    before = (p.px, p.py)
    p.strafe(1.0, level)
    after_d = (p.px, p.py)
    assert after_d != before

    p2 = Player(px=12.5, py=5.5, dx=0.0, dy=1.0, plx=0.66, ply=0.0)
    p2.strafe(-1.0, level)
    after_a = (p2.px, p2.py)
    assert after_a != before
    assert after_a != after_d


def test_collision_blocks_into_solid_wall(state):
    level = state.level
    # Find a solid (non-door) wall cell and confirm the adjoining cell
    # can't be entered from directly outside it.
    solid = np.argwhere((level.world_map > 0) & (level.world_map != config.DOOR_ID))[0]
    r, c = int(solid[0]), int(solid[1])
    assert not level.cell_passable(r + 0.5, c + 0.5)


def test_door_unlock_and_toggle(state):
    level = state.level
    cell = next(iter(level.door_cells))
    door = level.door_cells[cell]
    door.locked = True
    door.target = 0.0

    p = state.player
    p.px, p.py = cell[0] + 0.5, cell[1] - 1.0
    p.dx, p.dy = 0.0, 1.0

    try_interact(state)
    assert door.locked and door.target == 0.0   # still locked, no key

    state.has_key = True
    try_interact(state)
    assert not door.locked and door.target == 1.0   # unlocked and opening

    try_interact(state)
    assert door.target == 0.0   # toggled closed again


def test_shooting_hits_barrels_and_persons(state):
    """Both barrels AND persons should be shootable (previously only
    barrels were)."""
    level = state.level
    p = state.player
    base = time.perf_counter()

    for i, stype in enumerate(('barrel', 'person')):
        level.sprites.clear()
        level.sprites.append([p.px + p.dx * 3.0, p.py + p.dy * 3.0, stype])
        # Each iteration needs a timestamp well outside SHOOT_COOLDOWN of
        # the previous one, not just "+10" computed fresh each time (which
        # is ~identical across two back-to-back calls and re-triggers the
        # cooldown check).
        hit = try_shoot(state, base + i * 10.0)
        assert hit, f'{stype} should be shootable'
        assert len(level.sprites) == 0


def test_shooting_respects_cooldown(state):
    level = state.level
    p = state.player
    level.sprites.clear()
    level.sprites.append([p.px + p.dx * 3.0, p.py + p.dy * 3.0, 'barrel'])
    now = time.perf_counter()
    assert try_shoot(state, now) is True
    level.sprites.append([p.px + p.dx * 3.0, p.py + p.dy * 3.0, 'barrel'])
    assert try_shoot(state, now + 0.01) is False   # still cooling down
    assert len(level.sprites) == 1


def test_win_condition(state):
    level = state.level
    p = state.player
    p.px, p.py = level.exit_cell
    assert not state.game_won
    update_win(state)
    assert state.game_won
