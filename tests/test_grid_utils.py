from policies.grid_utils import (
    astar_path,
    bfs_distance_field,
    bfs_path,
    flood_fill_count,
    simulate_follow_path,
    tail_reachable_after_path,
)


def test_bfs_path_simple():
    path = bfs_path((0, 0), (3, 0), set(), 5, 5)
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (3, 0)
    assert len(path) == 4


def test_bfs_path_blocked():
    blocked = {(1, y) for y in range(5)}
    assert bfs_path((0, 0), (3, 0), blocked, 5, 5) is None


def test_astar_matches_bfs_length():
    blocked = {(2, 0), (2, 1), (2, 2), (2, 3)}
    bfs = bfs_path((0, 0), (4, 0), blocked, 5, 5)
    astar = astar_path((0, 0), (4, 0), blocked, 5, 5)
    assert bfs is not None and astar is not None
    assert len(bfs) == len(astar)


def test_flood_fill_count():
    assert flood_fill_count((0, 0), set(), 4, 4) == 16
    blocked = {(1, 0), (1, 1), (1, 2), (1, 3)}
    assert flood_fill_count((0, 0), blocked, 4, 4) == 4


def test_distance_field():
    field = bfs_distance_field([(0, 0)], set(), 3, 3)
    assert field[(0, 0)] == 0
    assert field[(2, 2)] == 4


def test_simulate_follow_path_no_food():
    snake = [(2, 0), (1, 0), (0, 0)]
    path = [(2, 0), (3, 0), (4, 0)]
    future = simulate_follow_path(snake, path, food=(9, 9))
    assert future == [(4, 0), (3, 0), (2, 0)]


def test_simulate_follow_path_with_growth():
    snake = [(2, 0), (1, 0), (0, 0)]
    path = [(2, 0), (3, 0)]
    future = simulate_follow_path(snake, path, food=(3, 0))
    assert future == [(3, 0), (2, 0), (1, 0), (0, 0)]


def test_tail_reachable_after_path():
    snake = [(2, 0), (1, 0), (0, 0)]
    path = [(2, 0), (3, 0)]
    assert tail_reachable_after_path(snake, path, (3, 0), 6, 6)
