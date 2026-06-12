"""贪吃蛇策略库：规则 / 搜索 / 混合 / RL 策略的统一接口。"""

from .base import BasePolicy, RandomPolicy
from .hamiltonian import (
    HamiltonianPolicy,
    build_hamiltonian_cycle,
    has_hamiltonian_cycle,
)
from .hybrid import HybridPolicy
from .search import SearchPolicy

RULE_POLICIES = ("random", "hamiltonian", "search", "hybrid")
ALL_POLICIES = RULE_POLICIES + ("rl",)


def make_policy(name: str, **kwargs) -> BasePolicy:
    """策略工厂。name='rl' 时需要 model_path（可选 algo、deterministic）。"""
    name = name.lower()
    if name == "random":
        return RandomPolicy(**kwargs)
    if name == "hamiltonian":
        return HamiltonianPolicy(**kwargs)
    if name == "search":
        return SearchPolicy(**kwargs)
    if name == "hybrid":
        return HybridPolicy(**kwargs)
    if name == "rl":
        from .rl import RLPolicy

        return RLPolicy.load(**kwargs)
    raise ValueError(f"未知策略: {name}，可选 {ALL_POLICIES}")


__all__ = [
    "BasePolicy",
    "RandomPolicy",
    "HamiltonianPolicy",
    "SearchPolicy",
    "HybridPolicy",
    "make_policy",
    "build_hamiltonian_cycle",
    "has_hamiltonian_cycle",
    "RULE_POLICIES",
    "ALL_POLICIES",
]
