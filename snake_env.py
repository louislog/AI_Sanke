from typing import Literal

import numpy as np
from gymnasium import Env, spaces

from snake_game import SnakeGame


class SnakeEnv(Env):
    """Gymnasium 兼容的贪吃蛇环境。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}
    FEATURE_DIM = 10
    DISTANCE_REWARD_SCALE = 0.1

    def __init__(
        self,
        width: int = SnakeGame.DEFAULT_WIDTH,
        height: int = SnakeGame.DEFAULT_HEIGHT,
        max_steps: int | None = None,
        render_mode: Literal["human", "rgb_array"] | None = None,
        distance_reward_scale: float = DISTANCE_REWARD_SCALE,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps or width * height * 4
        self.render_mode = render_mode
        self.distance_reward_scale = distance_reward_scale
        self.game = SnakeGame(width=width, height=height, render_mode=render_mode)
        self._episode_steps = 0
        self._prev_food_dist = 0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.FEATURE_DIM,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        self._episode_steps = 0
        self._prev_food_dist = self.game.food_manhattan_distance()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        assert self.action_space.contains(action)
        prev_dist = self._prev_food_dist
        reward, terminated = self.game.step(int(action))
        curr_dist = self.game.food_manhattan_distance()
        if not terminated:
            reward += (prev_dist - curr_dist) * self.distance_reward_scale
        self._prev_food_dist = curr_dist
        self._episode_steps += 1
        truncated = self._episode_steps >= self.max_steps

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        self.game.draw()

        if self.render_mode == "rgb_array":
            from snake_game import _get_pygame

            pygame = _get_pygame()
            array = pygame.surfarray.array3d(self.game.surface).transpose(1, 0, 2)
            return array.copy()

        return None

    def close(self):
        self.game.close()

    def _get_obs(self) -> np.ndarray:
        return np.asarray(self.game.extract_features(), dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "score": self.game.score,
            "steps": self.game.steps,
            "game_over": self.game.game_over,
            "ate_food": getattr(self.game, "_ate_food", False),
        }
