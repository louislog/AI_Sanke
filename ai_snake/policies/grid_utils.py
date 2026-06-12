"""网格搜索基础工具：BFS / A* / flood fill / 路径安全模拟。

所有函数都基于纯数据（坐标元组），与游戏渲染解耦，便于单测与复用。
坐标约定：(x, y)，x 向右增长，y 向下增长。

高频路径（训练 step / obs 构造）优先使用 numpy 数组实现，避免每步分配 dict/set。
"""

from collections import deque
from heapq import heappop, heappush
from typing import Iterable

import numpy as np

Coord = tuple[int, int]

# 四邻方向：(dy, dx) 便于直接索引 array[y, x]
_NEIGHBOR_OFFSETS = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)], dtype=np.int32)

CARDINAL_DELTAS: tuple[Coord, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))


def in_bounds(pos: Coord, width: int, height: int) -> bool:
    return 0 <= pos[0] < width and 0 <= pos[1] < height


def neighbors(pos: Coord, width: int, height: int) -> Iterable[Coord]:
    x, y = pos
    for dx, dy in CARDINAL_DELTAS:
        nxt = (x + dx, y + dy)
        if in_bounds(nxt, width, height):
            yield nxt


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs_path(
    start: Coord,
    goal: Coord,
    blocked: set[Coord],
    width: int,
    height: int,
) -> list[Coord] | None:
    """返回 start -> goal 的最短路径（含两端），不存在返回 None。"""
    if start == goal:
        return [start]
    if goal in blocked or not in_bounds(goal, width, height):
        return None

    parents: dict[Coord, Coord] = {start: start}
    queue: deque[Coord] = deque([start])
    while queue:
        current = queue.popleft()
        for nxt in neighbors(current, width, height):
            if nxt in parents or nxt in blocked:
                continue
            parents[nxt] = current
            if nxt == goal:
                return _reconstruct(parents, start, goal)
            queue.append(nxt)
    return None


def astar_path(
    start: Coord,
    goal: Coord,
    blocked: set[Coord],
    width: int,
    height: int,
) -> list[Coord] | None:
    """A* 最短路径（曼哈顿启发），返回含两端的路径，不存在返回 None。"""
    if start == goal:
        return [start]
    if goal in blocked or not in_bounds(goal, width, height):
        return None

    open_heap: list[tuple[int, int, Coord]] = [(manhattan(start, goal), 0, start)]
    g_score: dict[Coord, int] = {start: 0}
    parents: dict[Coord, Coord] = {start: start}
    counter = 0

    while open_heap:
        _, _, current = heappop(open_heap)
        if current == goal:
            return _reconstruct(parents, start, goal)
        for nxt in neighbors(current, width, height):
            if nxt in blocked:
                continue
            tentative = g_score[current] + 1
            if tentative < g_score.get(nxt, 1 << 30):
                g_score[nxt] = tentative
                parents[nxt] = current
                counter += 1
                heappush(open_heap, (tentative + manhattan(nxt, goal), counter, nxt))
    return None


def bfs_distance_field(
    sources: Iterable[Coord],
    blocked: set[Coord],
    width: int,
    height: int,
) -> dict[Coord, int]:
    """多源 BFS 距离场，仅包含可达格子。"""
    dist: dict[Coord, int] = {}
    queue: deque[Coord] = deque()
    for src in sources:
        if in_bounds(src, width, height):
            dist[src] = 0
            queue.append(src)
    while queue:
        current = queue.popleft()
        for nxt in neighbors(current, width, height):
            if nxt in dist or nxt in blocked:
                continue
            dist[nxt] = dist[current] + 1
            queue.append(nxt)
    return dist


def blocked_mask_from_coords(
    blocked: Iterable[Coord],
    width: int,
    height: int,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """将障碍坐标集合转为 (height, width) bool 掩码。"""
    if out is not None:
        mask = out
        mask.fill(False)
    else:
        mask = np.zeros((height, width), dtype=np.bool_)
    for x, y in blocked:
        if 0 <= x < width and 0 <= y < height:
            mask[y, x] = True
    return mask


def bfs_distance_field_array(
    sources: Iterable[Coord],
    blocked: np.ndarray,
    width: int,
    height: int,
    *,
    out_dist: np.ndarray | None = None,
) -> np.ndarray:
    """多源 BFS 距离场，返回 (height, width) int32；不可达格为 -1。"""
    dist = (
        out_dist
        if out_dist is not None
        else np.zeros((height, width), dtype=np.int32)
    )
    dist.fill(-1)

    queue: deque[tuple[int, int]] = deque()
    for sx, sy in sources:
        if 0 <= sx < width and 0 <= sy < height and not blocked[sy, sx]:
            if dist[sy, sx] < 0:
                dist[sy, sx] = 0
                queue.append((sx, sy))

    while queue:
        x, y = queue.popleft()
        d = dist[y, x]
        for dy, dx in _NEIGHBOR_OFFSETS:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if blocked[ny, nx] or dist[ny, nx] >= 0:
                continue
            dist[ny, nx] = d + 1
            queue.append((nx, ny))
    return dist


def fill_normalized_distance_channel(
    channel: np.ndarray,
    sources: Iterable[Coord],
    blocked: np.ndarray,
    width: int,
    height: int,
    max_dist: int,
    *,
    dist_buf: np.ndarray | None = None,
) -> None:
    """将 1 - d/max_dist 写入 channel[:height, :width]；不可达为 0。"""
    region = channel[:height, :width]
    region.fill(0.0)
    dist = bfs_distance_field_array(
        sources, blocked, width, height, out_dist=dist_buf
    )
    reachable = dist >= 0
    if not np.any(reachable):
        return
    norm = 1.0 - np.minimum(dist[reachable], max_dist).astype(np.float32) / float(
        max_dist
    )
    region[reachable] = norm


def reachable_ratio_numpy(
    start: Coord,
    blocked: set[Coord],
    width: int,
    height: int,
    *,
    blocked_mask: np.ndarray | None = None,
    visited_buf: np.ndarray | None = None,
) -> float:
    """从 start 出发 flood fill，返回可达空格 / 总空格比例。"""
    sx, sy = start
    if not in_bounds(start, width, height):
        return 0.0

    mask = blocked_mask
    if mask is None:
        mask = blocked_mask_from_coords(blocked, width, height)
    else:
        mask.fill(False)
        for x, y in blocked:
            if 0 <= x < width and 0 <= y < height:
                mask[y, x] = True

    free_cells = int((~mask).sum())
    if free_cells == 0:
        return 0.0
    if mask[sy, sx]:
        return 0.0

    visited = (
        visited_buf
        if visited_buf is not None
        else np.zeros((height, width), dtype=np.bool_)
    )
    visited.fill(False)
    queue: deque[tuple[int, int]] = deque([(sx, sy)])
    visited[sy, sx] = True
    count = 1

    while queue:
        x, y = queue.popleft()
        for dy, dx in _NEIGHBOR_OFFSETS:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if mask[ny, nx] or visited[ny, nx]:
                continue
            visited[ny, nx] = True
            count += 1
            queue.append((nx, ny))

    return count / free_cells


def flood_fill_count(
    start: Coord,
    blocked: set[Coord],
    width: int,
    height: int,
) -> int:
    """从 start 出发可达的空格数量（含 start 本身，若 start 被堵返回 0）。"""
    if start in blocked or not in_bounds(start, width, height):
        return 0
    visited = {start}
    queue: deque[Coord] = deque([start])
    while queue:
        current = queue.popleft()
        for nxt in neighbors(current, width, height):
            if nxt in visited or nxt in blocked:
                continue
            visited.add(nxt)
            queue.append(nxt)
    return len(visited)


def snake_blocked_cells(snake: list[Coord], *, will_grow: bool) -> set[Coord]:
    """蛇身障碍格。不吃食物时尾巴本步移走，不算障碍。"""
    body = snake if will_grow else snake[:-1]
    return set(body)


def simulate_follow_path(
    snake: list[Coord],
    path: list[Coord],
    food: Coord,
) -> list[Coord]:
    """模拟蛇沿 path（path[0] 为当前头）逐步移动后的身体。

    路径中每一步若头到达 food 则增长一次。返回模拟后的蛇身列表。
    不检查碰撞——调用方应保证 path 是按当前规则可行的最短路径。
    """
    body = deque(snake)
    for step_pos in path[1:]:
        body.appendleft(step_pos)
        if step_pos != food:
            body.pop()
    return list(body)


def tail_reachable_after_path(
    snake: list[Coord],
    path: list[Coord],
    food: Coord,
    width: int,
    height: int,
) -> bool:
    """模拟吃完食物后，检查蛇头是否仍能到达蛇尾（防止短视自困）。"""
    future = simulate_follow_path(snake, path, food)
    if len(future) >= width * height:
        return True  # 已满图
    head, tail = future[0], future[-1]
    blocked = set(future[1:-1])
    return bfs_path(head, tail, blocked, width, height) is not None


def _reconstruct(
    parents: dict[Coord, Coord], start: Coord, goal: Coord
) -> list[Coord]:
    path = [goal]
    current = goal
    while current != start:
        current = parents[current]
        path.append(current)
    path.reverse()
    return path
