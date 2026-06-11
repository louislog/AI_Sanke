"""混合策略：前期安全寻路吃食物，中后期切换 Hamiltonian 回路保满图。

阶段划分（按蛇长 / 棋盘容量比例）：
- coverage < switch_ratio：用 SearchPolicy 安全最短路吃食物（快）；
- coverage >= switch_ratio：切换 Hamiltonian（带 shortcut，稳），
  切换初期蛇身与回路未对齐时，由 HamiltonianPolicy 在 flood fill
  防自困校验下逐步贴回回路；
- 奇x奇地图不存在回路，全程使用 SearchPolicy。

switch_ratio 实验结论（20 局 / 尺寸）：<=0.25 时 6x6~10x10 满图率 100%，
更高阈值会因长蛇回贴回路风险增大而偶发自困，默认取 0.25。

目标是稳定生存和最终覆盖，而不是最快吃食物。
"""

from typing import TYPE_CHECKING

import numpy as np

from .base import BasePolicy
from .hamiltonian import HamiltonianPolicy, has_hamiltonian_cycle
from .search import SearchPolicy

if TYPE_CHECKING:
    from snake_env import SnakeEnv


class HybridPolicy(BasePolicy):
    name = "hybrid"

    def __init__(
        self,
        switch_ratio: float = 0.25,
        shortcut_disable_ratio: float = 0.5,
        safety_margin: int = 3,
        min_space_ratio: float = 1.0,
    ):
        self.switch_ratio = switch_ratio
        self.search = SearchPolicy(min_space_ratio=min_space_ratio)
        self.hamiltonian = HamiltonianPolicy(
            shortcuts=True,
            shortcut_disable_ratio=shortcut_disable_ratio,
            safety_margin=safety_margin,
        )

    def reset(self, env: "SnakeEnv") -> None:
        game = env.game
        self._has_cycle = has_hamiltonian_cycle(game.width, game.height)
        if self._has_cycle:
            self.hamiltonian.reset(env)
        self.search.reset(env)

    def select_action(self, env: "SnakeEnv", obs: np.ndarray | None = None) -> int:
        game = env.game
        if not getattr(self, "_has_cycle", None):
            self._has_cycle = has_hamiltonian_cycle(game.width, game.height)

        if not self._has_cycle:
            return self.search.select_action(env, obs)

        coverage = game.coverage_ratio()
        if coverage < self.switch_ratio:
            # 前期：安全最短路吃食物，不安全时退化为追尾 / 最大空间
            return self.search.select_action(env, obs)

        # 中后期：沿 Hamiltonian 回路（带安全 shortcut）；
        # 蛇身与回路未对齐时，HamiltonianPolicy 内部会在
        # 不自困（flood fill 校验）的前提下选择回路前向距离最小的动作回到回路
        return self.hamiltonian.select_action(env, obs)
