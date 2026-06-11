"""搜索策略：A*/BFS 安全寻路 + 追尾 + flood fill 空间兜底。

决策优先级（每步重新计算）：
1. A* 最短路吃食物，且模拟吃完后蛇头仍能到达蛇尾（防自困）才执行；
2. 最短路不安全 / 不存在时，追尾（朝蛇尾方向走，保持回旋空间）；
3. 追尾也不可行时，在合法动作中选 flood fill 可达空间最大的方向；
4. 实在无路可走则任选合法动作（或等死）。

该策略不依赖地图奇偶性，是奇x奇地图上 hybrid 的回退方案。
"""

from typing import TYPE_CHECKING

import numpy as np

from snake_game import SnakeGame

from .base import BasePolicy
from .grid_utils import (
    Coord,
    astar_path,
    bfs_path,
    flood_fill_count,
    snake_blocked_cells,
    tail_reachable_after_path,
)

if TYPE_CHECKING:
    from snake_env import SnakeEnv


class SearchPolicy(BasePolicy):
    name = "search"

    def __init__(self, min_space_ratio: float = 1.0):
        # 动作后可达空间需 >= 蛇长 * min_space_ratio，否则视为危险动作
        self.min_space_ratio = min_space_ratio

    def select_action(self, env: "SnakeEnv", obs: np.ndarray | None = None) -> int:
        game = env.game
        action = self.safe_food_action(game)
        if action is not None:
            return action

        action = self.chase_tail_action(game)
        if action is not None:
            return action

        return self.max_space_action(game)

    # ---- 第 1 级：安全吃食物 ----

    def safe_food_action(self, game: SnakeGame) -> int | None:
        snake = self.snake_cells(game)
        food = self.food(game)
        blocked = snake_blocked_cells(snake, will_grow=False)
        blocked.discard(snake[0])

        path = astar_path(snake[0], food, blocked, game.width, game.height)
        if path is None or len(path) < 2:
            return None
        if not tail_reachable_after_path(snake, path, food, game.width, game.height):
            return None

        action = self.action_towards(game, path[1])
        if action is None or not game.valid_action_mask()[action]:
            return None
        return action

    # ---- 第 2 级：追尾保命 ----

    def chase_tail_action(self, game: SnakeGame) -> int | None:
        snake = self.snake_cells(game)
        if len(snake) < 2:
            return None
        head, tail = snake[0], snake[-1]
        blocked = set(snake[1:-1])

        path = bfs_path(head, tail, blocked, game.width, game.height)
        if path is None or len(path) < 2:
            return None

        # 追尾时绕远一点更安全：在合法且不进死路的前提下，
        # 优先选择 flood fill 空间最大的一步，而不是贪最短路
        action = self._safest_step_towards(game, path[1])
        return action

    def _safest_step_towards(self, game: SnakeGame, fallback_cell: Coord) -> int | None:
        candidates = self._scored_actions(game)
        if not candidates:
            return None
        min_space = max(1, int(len(game.snake) * self.min_space_ratio))
        safe = [(space, a) for space, a in candidates if space >= min_space]
        if safe:
            return max(safe)[1]
        # 没有足够空间的动作时，退回朝目标格走
        action = self.action_towards(game, fallback_cell)
        if action is not None and game.valid_action_mask()[action]:
            return action
        return max(candidates)[1]

    # ---- 第 3 级：最大可达空间 ----

    def max_space_action(self, game: SnakeGame) -> int:
        candidates = self._scored_actions(game)
        if candidates:
            return max(candidates)[1]
        valid = self.valid_actions(game)
        return valid[0] if valid else 0

    def _scored_actions(self, game: SnakeGame) -> list[tuple[int, int]]:
        """返回 (动作后 flood fill 空间, 动作) 列表，仅含合法动作。"""
        snake = self.snake_cells(game)
        food = self.food(game)
        results: list[tuple[int, int]] = []
        for action in self.valid_actions(game):
            cell = self.next_cell_for_action(game, action)
            grows = cell == food
            new_snake = [cell] + (snake if grows else snake[:-1])
            blocked = set(new_snake[1:])
            space = flood_fill_count(cell, blocked - {cell}, game.width, game.height)
            results.append((space, action))
        return results

    @staticmethod
    def action_space_after(game: SnakeGame, action: int) -> int:
        """评估某个动作后的 flood fill 可达空间（供奖励塑形等复用）。"""
        policy = SearchPolicy()
        for space, a in policy._scored_actions(game):
            if a == action:
                return space
        return 0

    def select_action_array(self, env: "SnakeEnv") -> np.ndarray:
        """调试用：返回各动作的 flood fill 空间评分。"""
        scores = np.zeros(3)
        for space, action in self._scored_actions(env.game):
            scores[action] = space
        return scores
