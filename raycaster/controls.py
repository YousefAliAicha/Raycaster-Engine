# =============================================================================
# controls.py — keyboard/mouse state tracking and per-frame input handling.
# =============================================================================

from . import config
from .entities import try_interact

KEYS_HELD: set = set()
KEYS_PRESSED_EDGE: set = set()   # keys pressed since the last frame (edge-triggered)

MOUSE_DX = [0.0]   # accumulated mouse-look delta since last frame (pixels)
MOUSE_CAPTURED = [False]


def on_key_press(event):
    key = event.keysym.lower()
    if key not in KEYS_HELD:
        KEYS_PRESSED_EDGE.add(key)
    KEYS_HELD.add(key)


def on_key_release(event):
    KEYS_HELD.discard(event.keysym.lower())


def apply_input(state, dt):
    """
    Read the current key/mouse state and update the player accordingly.
    Movement/rotation are scaled by dt so behaviour is frame-rate
    independent. Returns False if the user has requested to quit.
    """
    if 'escape' in KEYS_HELD or 'q' in KEYS_HELD:
        return False

    if state.game_won:
        KEYS_PRESSED_EDGE.clear()
        return True   # level complete: ignore movement/interact, Esc/Q still quit

    player = state.player
    level  = state.level
    move_amt = config.MOVE_SPEED * dt
    rot_amt  = config.ROT_SPEED * dt

    if 'w'    in KEYS_HELD or 'up'   in KEYS_HELD: player.move( move_amt, level)
    if 's'    in KEYS_HELD or 'down' in KEYS_HELD: player.move(-move_amt, level)
    if 'a'    in KEYS_HELD: player.strafe(-move_amt, level)
    if 'd'    in KEYS_HELD: player.strafe( move_amt, level)
    if 'left'  in KEYS_HELD: player.rotate( rot_amt)
    if 'right' in KEYS_HELD: player.rotate(-rot_amt)

    if MOUSE_DX[0] != 0.0:
        player.rotate(-MOUSE_DX[0] * config.MOUSE_SENS)
        MOUSE_DX[0] = 0.0

    if 'e' in KEYS_PRESSED_EDGE:
        try_interact(state)
    KEYS_PRESSED_EDGE.clear()

    return True
