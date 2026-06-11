"""RL 策略包装：把 SB3 模型适配到统一的 BasePolicy 接口。"""

from typing import TYPE_CHECKING

import numpy as np

from .base import BasePolicy

if TYPE_CHECKING:
    from snake_env import SnakeEnv


class RLPolicy(BasePolicy):
    name = "rl"

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic
        self._is_maskable = type(model).__name__ == "MaskablePPO"

    @classmethod
    def load(cls, model_path: str, algo: str = "auto", deterministic: bool = True):
        """从 checkpoint 加载模型。algo='auto' 时按 maskable_ppo -> ppo -> dqn -> qrdqn 依次尝试。"""
        from algos import load_model

        return cls(load_model(model_path, algo), deterministic=deterministic)

    def select_action(self, env: "SnakeEnv", obs: np.ndarray | None = None) -> int:
        if obs is None:
            obs = env._get_obs()
        if self._is_maskable:
            action, _ = self.model.predict(
                obs,
                deterministic=self.deterministic,
                action_masks=env.action_masks(),
            )
        else:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)
