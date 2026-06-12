"""策略抽象接口。

所有策略（规则 / 搜索 / 混合 / RL）实现统一的 select_action 接口，
评估、数据采集、可视化代码可以无差别地使用任意策略。
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ai_snake.snake_game import Direction, Position, SnakeGame

from .grid_utils import Coord

if TYPE_CHECKING:
    from ai_snake.snake_env import SnakeEnv


class BasePolicy(ABC):
    """策略基类：输入环境，输出 SnakeEnv 的相对动作（0 直行 / 1 右转 / 2 左转）。"""

    name = "base"

    def reset(self, env: "SnakeEnv") -> None:
        """每局开始时调用，用于重置策略内部状态。"""

    @abstractmethod
    def select_action(self, env: "SnakeEnv", obs: np.ndarray | None = None) -> int:
        """返回当前局面下的动作。obs 为环境观测（RL 策略使用）。"""

    # ---- 规则策略常用的工具方法 ----

    @staticmethod
    def snake_cells(game: SnakeGame) -> list[Coord]:
        return [(seg.x, seg.y) for seg in game.snake]

    @staticmethod
    def head(game: SnakeGame) -> Coord:
        return (game.snake[0].x, game.snake[0].y)

    @staticmethod
    def food(game: SnakeGame) -> Coord:
        return (game.food.x, game.food.y)

    @staticmethod
    def direction_to(src: Coord, dst: Coord) -> Direction:
        """相邻格 src -> dst 的绝对方向。"""
        dx, dy = dst[0] - src[0], dst[1] - src[1]
        mapping = {(0, -1): Direction.UP, (1, 0): Direction.RIGHT,
                   (0, 1): Direction.DOWN, (-1, 0): Direction.LEFT}
        return mapping[(dx, dy)]

    @staticmethod
    def action_towards(game: SnakeGame, target: Coord) -> int | None:
        """朝相邻格 target 移动所需的相对动作；若需要掉头则返回 None。"""
        direction = BasePolicy.direction_to(BasePolicy.head(game), target)
        return game.action_for_direction(direction)

    @staticmethod
    def valid_actions(game: SnakeGame) -> list[int]:
        return [a for a, ok in enumerate(game.valid_action_mask()) if ok]

    @staticmethod
    def next_cell_for_action(game: SnakeGame, action: int) -> Coord:
        direction = game._get_new_direction(game.direction, action)
        head = game.snake[0]
        nxt = game._next_position(Position(head.x, head.y), direction)
        return (nxt.x, nxt.y)


class RandomPolicy(BasePolicy):
    """在合法动作中随机选择，作为最弱基线。"""

    name = "random"

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def select_action(self, env: "SnakeEnv", obs: np.ndarray | None = None) -> int:
        valid = self.valid_actions(env.game)
        if not valid:
            return 0
        return int(self._rng.choice(valid))
