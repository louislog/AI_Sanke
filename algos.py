"""RL 算法工厂：统一创建 / 加载 PPO、MaskablePPO、DQN、QR-DQN 模型。

新增算法只需在 ALGO_REGISTRY 注册，train.py / eval.py / bc_train.py 自动支持。
"""

import os
from typing import Any

from stable_baselines3 import DQN, PPO

from snake_cnn import SnakeCNN

try:
    from sb3_contrib import MaskablePPO, QRDQN

    _HAS_CONTRIB = True
except ImportError:
    MaskablePPO = None  # type: ignore[misc, assignment]
    QRDQN = None  # type: ignore[misc, assignment]
    _HAS_CONTRIB = False

try:
    from sb3_contrib import RecurrentPPO

    _HAS_RECURRENT = True
except ImportError:
    RecurrentPPO = None  # type: ignore[misc, assignment]
    _HAS_RECURRENT = False


def _registry() -> dict[str, Any]:
    registry: dict[str, Any] = {"ppo": PPO, "dqn": DQN}
    if _HAS_CONTRIB:
        registry["maskable_ppo"] = MaskablePPO
        registry["qrdqn"] = QRDQN
    if _HAS_RECURRENT:
        registry["recurrent_ppo"] = RecurrentPPO
    return registry


ALGO_REGISTRY = _registry()
ON_POLICY_ALGOS = {"ppo", "maskable_ppo", "recurrent_ppo"}
OFF_POLICY_ALGOS = {"dqn", "qrdqn"}


def available_algos() -> list[str]:
    return sorted(ALGO_REGISTRY)


def supports_action_masking(algo: str) -> bool:
    return algo == "maskable_ppo"


def _policy_kwargs(obs_mode: str, features_dim: int) -> tuple[str, dict]:
    if obs_mode in ("grid", "grid_full"):
        return "MlpPolicy", dict(
            features_extractor_class=SnakeCNN,
            features_extractor_kwargs=dict(features_dim=features_dim),
            net_arch=dict(pi=[256], vf=[256]),
        )
    return "MlpPolicy", dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))


def build_model(
    algo: str,
    vec_env,
    *,
    obs_mode: str = "grid_full",
    features_dim: int = 512,
    tensorboard_log: str | None = None,
    n_steps: int = 4096,
    batch_size: int = 512,
    n_epochs: int = 10,
    learning_rate: float = 2.5e-4,
    gamma: float = 0.995,
    ent_coef: float = 0.01,
    buffer_size: int = 200_000,
    verbose: int = 1,
):
    algo = algo.lower()
    if algo not in ALGO_REGISTRY:
        raise ValueError(f"未知算法: {algo}，可用 {available_algos()}（缺失项请安装 sb3-contrib）")

    policy, policy_kwargs = _policy_kwargs(obs_mode, features_dim)
    cls = ALGO_REGISTRY[algo]

    if algo in ON_POLICY_ALGOS:
        if algo == "recurrent_ppo":
            policy = "MlpLstmPolicy"
        return cls(
            policy=policy,
            env=vec_env,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            device="auto",
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=0.95,
            ent_coef=ent_coef,
            vf_coef=1.0,
            max_grad_norm=0.5,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
        )

    # off-policy（DQN / QR-DQN）：net_arch 不支持 pi/vf 字典
    policy_kwargs = dict(policy_kwargs)
    policy_kwargs["net_arch"] = [256]
    return cls(
        policy=policy,
        env=vec_env,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        device="auto",
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=10_000,
        train_freq=4,
        target_update_interval=5_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        policy_kwargs=policy_kwargs,
    )


def load_model(model_path: str, algo: str = "auto", env=None):
    """加载 checkpoint。algo='auto' 时按注册表依次尝试。"""
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    if algo != "auto":
        algo = algo.lower()
        if algo not in ALGO_REGISTRY:
            raise ValueError(f"未知算法: {algo}，可用 {available_algos()}")
        return ALGO_REGISTRY[algo].load(model_path, env=env)

    order = ["maskable_ppo", "ppo", "qrdqn", "dqn", "recurrent_ppo"]
    errors: list[str] = []
    for name in order:
        cls = ALGO_REGISTRY.get(name)
        if cls is None:
            continue
        try:
            return cls.load(model_path, env=env)
        except Exception as exc:  # noqa: BLE001 - 尝试下一个算法
            errors.append(f"{name}: {exc}")
    raise ValueError(f"无法加载模型 {model_path}，尝试记录:\n" + "\n".join(errors))
