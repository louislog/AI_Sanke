"""Hamiltonian Cycle 策略：满图覆盖的强基线。

核心思想：
1. 预先构造一条经过棋盘每个格子恰好一次并回到起点的回路；
2. 蛇头永远沿回路前进，则永远不会撞到自己，最终必然吃满全图；
3. 为避免前期绕全图太慢，允许 "抄近路"（shortcut）：
   只要目标格在回路顺序上位于 蛇头 与 蛇尾 之间（不越过尾巴，留安全余量），
   就可以沿回路顺序向前跳，蛇身的回路序关系保持单调，依然不会自撞。

数学事实：w x h 网格图是二分图，存在 Hamiltonian 回路当且仅当格子总数为偶数
（即 w、h 至少一个为偶数）。奇 x 奇地图请使用 search / hybrid 策略。
"""

from typing import TYPE_CHECKING

import numpy as np

from ai_snake.snake_game import SnakeGame

from .base import BasePolicy
from .grid_utils import Coord, flood_fill_count, manhattan

if TYPE_CHECKING:
    from ai_snake.snake_env import SnakeEnv


def has_hamiltonian_cycle(width: int, height: int) -> bool:
    return (width * height) % 2 == 0 and width >= 2 and height >= 2


def build_hamiltonian_cycle(width: int, height: int) -> list[Coord]:
    """构造 w x h 棋盘的 Hamiltonian 回路（按访问顺序返回格子列表）。

    构造方式（要求 h 为偶数，否则转置构造再换轴）：
    - 第 0 行从 (0,0) 走到 (w-1,0)；
    - 第 1..h-1 行在列 1..w-1 之间蛇形往返；
    - 最后沿第 0 列从 (0,h-1) 回到 (0,1)，与起点闭合。
    """
    if not has_hamiltonian_cycle(width, height):
        raise ValueError(
            f"{width}x{height} 棋盘不存在 Hamiltonian 回路（奇x奇），"
            "请改用 search/hybrid 策略或偶数边长地图。"
        )

    if height % 2 == 0:
        return _build_cycle_even_height(width, height)
    # h 为奇数则 w 必为偶数：转置构造后交换坐标轴
    transposed = _build_cycle_even_height(height, width)
    return [(y, x) for (x, y) in transposed]


def _build_cycle_even_height(width: int, height: int) -> list[Coord]:
    order: list[Coord] = [(x, 0) for x in range(width)]
    for row in range(1, height):
        if row % 2 == 1:
            xs = range(width - 1, 0, -1)
        else:
            xs = range(1, width)
        order.extend((x, row) for x in xs)
    order.extend((0, y) for y in range(height - 1, 0, -1))

    assert len(order) == width * height
    return order


class HamiltonianPolicy(BasePolicy):
    """沿 Hamiltonian 回路行进，可选启用安全 shortcut 加速吃食物。

    shortcut 规则（保证安全的充分条件）：
    - 蛇身在回路序上保持单调（头在最前，尾在最后）；
    - 候选邻格的回路前向距离 < 头到尾的回路前向距离 - 安全余量；
    - 蛇身长度超过 shortcut_disable_ratio * 容量后禁用 shortcut，
      完全退化为纯回路行进，保证终局稳定满图。
    """

    name = "hamiltonian"

    def __init__(
        self,
        shortcuts: bool = True,
        shortcut_disable_ratio: float = 0.5,
        safety_margin: int = 3,
    ):
        self.shortcuts = shortcuts
        self.shortcut_disable_ratio = shortcut_disable_ratio
        self.safety_margin = safety_margin
        self._cycle: list[Coord] | None = None
        self._index: dict[Coord, int] = {}
        self._size: tuple[int, int] | None = None

    def reset(self, env: "SnakeEnv") -> None:
        self._ensure_cycle(env.game)

    def _ensure_cycle(self, game: SnakeGame) -> None:
        size = (game.width, game.height)
        if self._size != size:
            cycle = build_hamiltonian_cycle(*size)
            self._cycle = cycle
            self._index = {cell: i for i, cell in enumerate(cycle)}
            self._size = size

    def cycle_index_map(self, game: SnakeGame) -> dict[Coord, int]:
        self._ensure_cycle(game)
        return self._index

    def _forward_distance(self, src_idx: int, dst_idx: int) -> int:
        n = len(self._cycle or [])
        return (dst_idx - src_idx) % n

    def _order_monotonic(self, game: SnakeGame) -> bool:
        """蛇身回路序是否单调（shortcut 安全的前提）。"""
        head_idx = self._index[self.head(game)]
        prev = 0
        for seg in game.snake[1:]:
            dist = self._forward_distance(self._index[(seg.x, seg.y)], head_idx)
            if dist <= prev:
                return False
            prev = dist
        return True

    def safe_cycle_action(self, game: SnakeGame) -> int | None:
        """严格回路动作（或安全 shortcut）。

        蛇身与回路未对齐时，回路动作不再有安全保证，此时额外做
        flood fill 防自困检查；不安全或不可行返回 None，由调用方回退。
        """
        self._ensure_cycle(game)
        assert self._cycle is not None

        head = self.head(game)
        head_idx = self._index[head]
        n = len(self._cycle)
        target = self._cycle[(head_idx + 1) % n]

        aligned = self._order_monotonic(game)
        if aligned and self._shortcut_allowed(game):
            shortcut = self._best_shortcut(game, head_idx)
            if shortcut is not None:
                target = shortcut

        action = self.action_towards(game, target)
        if action is None or not game.valid_action_mask()[action]:
            return None
        if not aligned and not self._move_keeps_space(game, target):
            return None
        return action

    def _move_keeps_space(self, game: SnakeGame, cell: Coord) -> bool:
        snake = self.snake_cells(game)
        grows = cell == self.food(game)
        new_snake = [cell] + (snake if grows else snake[:-1])
        space = flood_fill_count(cell, set(new_snake[1:]), game.width, game.height)
        return space >= len(new_snake)

    def select_action(self, env: "SnakeEnv", obs: np.ndarray | None = None) -> int:
        game = env.game
        action = self.safe_cycle_action(game)
        if action is not None:
            return action

        # 蛇身与回路不对齐（开局或混合策略切换期）：
        # 在不自困的合法动作中选择回路前向距离最小的格子，尽快回到回路
        self._ensure_cycle(game)
        head_idx = self._index[self.head(game)]
        snake = self.snake_cells(game)
        food = self.food(game)
        candidates: list[tuple[int, int, int]] = []  # (forward, action, space)
        for candidate in self.valid_actions(game):
            cell = self.next_cell_for_action(game, candidate)
            forward = self._forward_distance(head_idx, self._index[cell])
            grows = cell == food
            new_snake = [cell] + (snake if grows else snake[:-1])
            space = flood_fill_count(
                cell, set(new_snake[1:]), game.width, game.height
            )
            if forward > 0:
                candidates.append((forward, candidate, space))

        if not candidates:
            valid = self.valid_actions(game)
            return valid[0] if valid else 0

        need = len(snake) + 1
        safe = [c for c in candidates if c[2] >= need]
        if safe:
            return min(safe)[1]
        # 没有完全安全的动作时，保空间优先于贴回路
        return max(candidates, key=lambda c: c[2])[1]

    def _shortcut_allowed(self, game: SnakeGame) -> bool:
        if not self.shortcuts:
            return False
        if len(game.snake) >= self.shortcut_disable_ratio * game.board_capacity():
            return False
        return self._order_monotonic(game)

    def _best_shortcut(self, game: SnakeGame, head_idx: int) -> Coord | None:
        """在安全窗口内选择朝食物推进最多的邻格。"""
        tail = (game.snake[-1].x, game.snake[-1].y)
        tail_dist = self._forward_distance(head_idx, self._index[tail])
        if tail_dist == 0:
            tail_dist = len(self._cycle or [])

        food = self.food(game)
        food_dist = self._forward_distance(head_idx, self._index[food])

        best: Coord | None = None
        best_forward = 0
        for action in self.valid_actions(game):
            cell = self.next_cell_for_action(game, action)
            forward = self._forward_distance(head_idx, self._index[cell])
            if forward <= 0:
                continue
            # 不越过食物（食物在窗口内时），不逼近尾巴
            if forward >= tail_dist - self.safety_margin:
                continue
            if food_dist > 0 and forward > food_dist:
                continue
            if forward > best_forward or (
                forward == best_forward
                and best is not None
                and manhattan(cell, food) < manhattan(best, food)
            ):
                best = cell
                best_forward = forward

        if best_forward <= 1:
            return None  # 与直接沿回路走相同，无需 shortcut
        return best
