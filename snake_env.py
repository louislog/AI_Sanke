"""Gymnasium 贪吃蛇环境：可配置奖励 + 多种观测模式。

观测模式：
- vector    : 24 维手工特征（兼容旧版）
- grid      : 3 通道栅格（蛇头 / 蛇身 / 食物，兼容旧版）
- grid_full : 8 通道栅格，面向满图覆盖任务：
              蛇头、蛇身、食物、蛇身顺序场、棋盘掩码（区分 padding 墙）、
              到食物 BFS 距离场、到蛇尾 BFS 距离场、Hamiltonian index 场

奖励：通过 RewardConfig 完全可配置，内置 default / sparse / coverage / greedy
四种预设，方便消融实验。
"""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from gymnasium import Env, spaces

from policies.grid_utils import bfs_distance_field, bfs_path
from snake_game import SnakeGame


@dataclass
class RewardConfig:
    """奖励配置。所有惩罚字段填正数，计算时取负。"""

    food_reward: float = 10.0
    step_penalty: float = 0.0
    # 死亡惩罚 = death_penalty + death_penalty_coverage_scale * 当前覆盖率
    # 覆盖率越高死亡越可惜，后期死亡惩罚更大
    death_penalty: float = 50.0
    death_penalty_coverage_scale: float = 0.0
    win_reward: float = 100.0
    # 吃到食物时按覆盖率给的额外奖励：coverage_bonus_scale * coverage
    coverage_bonus_scale: float = 0.0
    # 可达空间骤降惩罚（flood fill 比例下降超过阈值时按降幅惩罚）
    trap_penalty_scale: float = 2.0
    trap_drop_threshold: float = 0.05
    # 吃完食物后蛇头无法到达蛇尾的惩罚（短视自困信号）
    tail_unreachable_penalty: float = 0.0
    # 靠近食物 BFS 距离塑形（谨慎使用，蛇长超过阈值后自动关闭）
    distance_reward_scale: float = 0.05
    distance_shaping_length_threshold: int | None = None
    # 里程碑奖励：每吃 interval 个食物给一次
    milestone_interval: int = 10
    milestone_reward: float = 5.0


REWARD_PRESETS: dict[str, RewardConfig] = {
    # 旧版行为，向后兼容
    "default": RewardConfig(),
    # 几乎无塑形，检验算法本身的信用分配能力
    "sparse": RewardConfig(
        food_reward=1.0,
        death_penalty=1.0,
        win_reward=10.0,
        trap_penalty_scale=0.0,
        distance_reward_scale=0.0,
        milestone_interval=0,
        milestone_reward=0.0,
    ),
    # 面向满图覆盖：后期死亡惩罚大、奖励高覆盖、惩罚自困，弱化靠近食物奖励
    "coverage": RewardConfig(
        food_reward=10.0,
        death_penalty=30.0,
        death_penalty_coverage_scale=120.0,
        win_reward=200.0,
        coverage_bonus_scale=20.0,
        trap_penalty_scale=4.0,
        trap_drop_threshold=0.04,
        tail_unreachable_penalty=5.0,
        distance_reward_scale=0.02,
        distance_shaping_length_threshold=12,
        milestone_interval=10,
        milestone_reward=5.0,
    ),
    # 短视贪吃对照组（用于消融，预期后期表现差）
    "greedy": RewardConfig(
        food_reward=10.0,
        death_penalty=10.0,
        trap_penalty_scale=0.0,
        distance_reward_scale=0.2,
    ),
}


def resolve_reward_config(reward: "RewardConfig | str | None") -> RewardConfig:
    if reward is None:
        return replace(REWARD_PRESETS["default"])
    if isinstance(reward, RewardConfig):
        return reward
    if reward in REWARD_PRESETS:
        return replace(REWARD_PRESETS[reward])
    raise ValueError(f"未知奖励预设: {reward}，可选 {sorted(REWARD_PRESETS)}")


ObsMode = Literal["vector", "grid", "grid_full"]

GRID_CHANNELS = {"grid": 3, "grid_full": 8}


class SnakeEnv(Env):
    """Gymnasium 兼容的贪吃蛇环境。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}
    FEATURE_DIM = SnakeGame.FEATURE_DIM
    GRID_CHANNELS = 3  # 旧版兼容
    MAX_STEPS_FACTOR = 20

    def __init__(
        self,
        width: int = SnakeGame.DEFAULT_WIDTH,
        height: int = SnakeGame.DEFAULT_HEIGHT,
        max_steps: int | None = None,
        max_steps_factor: int = MAX_STEPS_FACTOR,
        render_mode: Literal["human", "rgb_array"] | None = None,
        obs_mode: ObsMode = "grid",
        grid_pad_size: int | None = None,
        reward: RewardConfig | str | None = "default",
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps or width * height * max_steps_factor
        self.render_mode = render_mode
        self.obs_mode = obs_mode
        self.grid_pad_size = grid_pad_size or max(width, height)
        self.reward_config = resolve_reward_config(reward)
        self.game = SnakeGame(width=width, height=height, render_mode=render_mode)

        self._episode_steps = 0
        self._prev_food_bfs: int | None = None
        self._prev_reachable_ratio = 1.0
        self._last_milestone = 0
        self._ham_index_map = self._build_ham_index_map()

        self.action_space = spaces.Discrete(3)
        self.observation_space = self._make_observation_space()

    # ---- 空间定义 ----

    def _make_observation_space(self) -> spaces.Box:
        if self.obs_mode in GRID_CHANNELS:
            size = self.grid_pad_size
            return spaces.Box(
                low=0.0,
                high=1.0,
                shape=(GRID_CHANNELS[self.obs_mode], size, size),
                dtype=np.float32,
            )
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.FEATURE_DIM,),
            dtype=np.float32,
        )

    def _build_ham_index_map(self) -> np.ndarray | None:
        from policies.hamiltonian import build_hamiltonian_cycle, has_hamiltonian_cycle

        if not has_hamiltonian_cycle(self.width, self.height):
            return None
        cycle = build_hamiltonian_cycle(self.width, self.height)
        index_map = np.zeros((self.height, self.width), dtype=np.float32)
        denom = max(len(cycle) - 1, 1)
        for i, (x, y) in enumerate(cycle):
            index_map[y, x] = i / denom
        return index_map

    # ---- Gym API ----

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        self._episode_steps = 0
        self._prev_food_bfs = self.game.food_bfs_distance()
        self._prev_reachable_ratio = self.game.reachable_ratio()
        self._last_milestone = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        assert self.action_space.contains(action)
        cfg = self.reward_config
        prev_bfs = self._prev_food_bfs
        prev_coverage = self.game.coverage_ratio()

        _, terminated = self.game.step(int(action))
        ate_food = getattr(self.game, "_ate_food", False)

        reward = -cfg.step_penalty
        if terminated:
            if self.game.won:
                reward += cfg.win_reward + self.game.board_capacity()
            else:
                reward -= (
                    cfg.death_penalty
                    + cfg.death_penalty_coverage_scale * prev_coverage
                )
        else:
            if ate_food:
                reward += cfg.food_reward
                reward += cfg.coverage_bonus_scale * self.game.coverage_ratio()
                reward += self._milestone_reward()
                if cfg.tail_unreachable_penalty > 0 and not self._tail_reachable():
                    reward -= cfg.tail_unreachable_penalty

            curr_bfs = self.game.food_bfs_distance()
            reward += self._bfs_distance_shaping(prev_bfs, curr_bfs)
            self._prev_food_bfs = curr_bfs

            curr_reachable = self.game.reachable_ratio()
            drop = self._prev_reachable_ratio - curr_reachable
            if cfg.trap_penalty_scale > 0 and drop > cfg.trap_drop_threshold:
                reward -= cfg.trap_penalty_scale * drop
            self._prev_reachable_ratio = curr_reachable

        self._episode_steps += 1
        truncated = not terminated and self._episode_steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def action_masks(self) -> np.ndarray:
        return np.asarray(self.game.valid_action_mask(), dtype=bool)

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

    # ---- 奖励组件 ----

    def _tail_reachable(self) -> bool:
        snake = [(seg.x, seg.y) for seg in self.game.snake]
        if len(snake) < 3:
            return True
        blocked = set(snake[1:-1])
        return (
            bfs_path(snake[0], snake[-1], blocked, self.width, self.height)
            is not None
        )

    def _bfs_distance_shaping(
        self, prev_bfs: int | None, curr_bfs: int | None
    ) -> float:
        cfg = self.reward_config
        if cfg.distance_reward_scale <= 0 or prev_bfs is None or curr_bfs is None:
            return 0.0
        if (
            cfg.distance_shaping_length_threshold is not None
            and len(self.game.snake) > cfg.distance_shaping_length_threshold
        ):
            return 0.0
        return (prev_bfs - curr_bfs) * cfg.distance_reward_scale

    def _milestone_reward(self) -> float:
        cfg = self.reward_config
        if cfg.milestone_interval <= 0 or cfg.milestone_reward <= 0:
            return 0.0
        milestone = self.game.score // cfg.milestone_interval
        if milestone <= self._last_milestone:
            return 0.0
        delta = milestone - self._last_milestone
        self._last_milestone = milestone
        return delta * cfg.milestone_reward

    # ---- 观测 ----

    def _get_obs(self) -> np.ndarray:
        if self.obs_mode == "grid":
            grid = self.game.rasterize_board(
                pad_width=self.grid_pad_size,
                pad_height=self.grid_pad_size,
            )
            return np.asarray(grid, dtype=np.float32)
        if self.obs_mode == "grid_full":
            return self._full_grid_obs()
        return np.asarray(self.game.extract_features(), dtype=np.float32)

    def _full_grid_obs(self) -> np.ndarray:
        size = self.grid_pad_size
        obs = np.zeros((8, size, size), dtype=np.float32)
        game = self.game
        w, h = self.width, self.height
        snake = [(seg.x, seg.y) for seg in game.snake]
        head, tail = snake[0], snake[-1]
        food = (game.food.x, game.food.y)

        # 0 蛇头 / 1 蛇身 / 2 食物
        obs[0, head[1], head[0]] = 1.0
        for x, y in snake[1:]:
            obs[1, y, x] = 1.0
        obs[2, food[1], food[0]] = 1.0

        # 3 蛇身顺序场：头 1.0 -> 尾递减
        n = len(snake)
        for i, (x, y) in enumerate(snake):
            obs[3, y, x] = (n - i) / n

        # 4 棋盘掩码：实际棋盘为 1，padding（墙外）为 0
        obs[4, :h, :w] = 1.0

        # 5/6 BFS 距离场（被身体阻挡，不可达为 0）
        blocked = set(snake[1:-1])
        max_dist = w * h
        food_field = bfs_distance_field([food], blocked, w, h)
        for (x, y), d in food_field.items():
            obs[5, y, x] = 1.0 - d / max_dist
        tail_field = bfs_distance_field([tail], blocked, w, h)
        for (x, y), d in tail_field.items():
            obs[6, y, x] = 1.0 - d / max_dist

        # 7 Hamiltonian index 场（奇x奇地图为全 0）
        if self._ham_index_map is not None:
            obs[7, :h, :w] = self._ham_index_map

        return obs

    def _get_info(self) -> dict:
        return {
            "score": self.game.score,
            "steps": self.game.steps,
            "game_over": self.game.game_over,
            "won": self.game.won,
            "ate_food": getattr(self.game, "_ate_food", False),
            "snake_length": len(self.game.snake),
            "coverage": self.game.coverage_ratio(),
            "death_reason": getattr(self.game, "death_reason", None),
        }
