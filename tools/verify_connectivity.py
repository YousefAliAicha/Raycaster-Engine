"""
Verifies vault-room reachability: confirms the vault is unreachable with
the door closed, and reachable once it's open. Run from repo root:
    PYTHONPATH=. python3 tools/verify_connectivity.py
"""
from collections import deque
from raycaster import level as level_mod
from raycaster import config


def bfs_reachable_cells(lvl, door_open):
    """Return the set of (r, c) floor cells reachable from spawn."""
    grid = lvl.world_map
    rows, cols = lvl.map_rows, lvl.map_cols
    start = (int(lvl.spawn['px']), int(lvl.spawn['py']))

    def passable(r, c):
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        v = grid[r, c]
        if v == 0:
            return True
        if v == config.DOOR_ID:
            return door_open  # treat ALL doors as open/closed uniformly
        return False

    visited = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if (nr, nc) not in visited and passable(nr, nc):
                visited.add((nr, nc))
                q.append((nr, nc))
    return visited


if __name__ == '__main__':
    lvl = level_mod.load_level()

    closed = bfs_reachable_cells(lvl, door_open=False)
    opened = bfs_reachable_cells(lvl, door_open=True)

    # Cells only reachable once doors are open = the gated room(s).
    gated = opened - closed
    print(f'Reachable with doors CLOSED: {len(closed)} cells')
    print(f'Reachable with doors OPEN:   {len(opened)} cells')
    print(f'Cells gated behind a door:   {len(gated)} cells')
    print(f'Gated cells: {sorted(gated)}')