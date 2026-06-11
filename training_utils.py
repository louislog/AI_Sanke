"""训练辅助：设备解析、向量化环境工厂、默认超参与安全奖励频率。"""

from __future__ import annotations

import os
from typing import Any, Callable, Type

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def resolve_training_device(device: str, cuda_device: int = 0) -> str:
    """解析训练设备字符串，CUDA 不可用时给出明确提示。"""
    import torch as th

    device = device.lower().strip()
    if device == "auto":
        if th.cuda.is_available():
            return f"cuda:{cuda_device}"
        if getattr(th.backends, "mps", None) and th.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device == "cuda":
        if not th.cuda.is_available():
            reason = _cuda_unavailable_reason()
            print(
                f"[device] 请求 CUDA 但不可用（{reason}），已回退到 CPU。"
                " 环境采样与奖励计算仍在 CPU 上，GPU 主要加速策略网络。"
            )
            return "cpu"
        return f"cuda:{cuda_device}"

    if device == "cpu":
        return "cpu"

    if device.startswith("cuda:"):
        if not th.cuda.is_available():
            reason = _cuda_unavailable_reason()
            print(f"[device] 请求 {device} 但 CUDA 不可用（{reason}），已回退到 CPU。")
            return "cpu"
        return device

    if device == "mps":
        if getattr(th.backends, "mps", None) and th.backends.mps.is_available():
            return "mps"
        print("[device] 请求 MPS 但不可用，已回退到 CPU。")
        return "cpu"

    raise ValueError(f"未知 device: {device}，可选 auto/cpu/cuda/mps 或 cuda:N")


def _cuda_unavailable_reason() -> str:
    import torch as th

    if not th.cuda.is_available():
        if not hasattr(th.version, "cuda") or th.version.cuda is None:
            return "当前 PyTorch 为 CPU 构建，未包含 CUDA"
        return "驱动或 GPU 不可用"
    return "unknown"


def print_device_info(device: str) -> None:
    """启动训练时打印设备与 PyTorch 版本信息。"""
    import torch as th

    print("=== Training device ===")
    print(f"  torch.__version__     : {th.__version__}")
    print(f"  torch.cuda.is_available(): {th.cuda.is_available()}")
    if th.cuda.is_available():
        idx = 0
        if device.startswith("cuda:"):
            idx = int(device.split(":")[1])
        print(f"  torch.version.cuda    : {th.version.cuda}")
        print(f"  GPU name              : {th.cuda.get_device_name(idx)}")
    elif getattr(th.backends, "mps", None) and th.backends.mps.is_available():
        print("  Apple MPS             : available")
    else:
        print(f"  CUDA unavailable      : {_cuda_unavailable_reason()}")
    print(f"  training device       : {device}")
    print("=======================")


def make_snake_vec_env(
    env_class: Type | None = None,
    n_envs: int = 1,
    env_kwargs: dict[str, Any] | None = None,
    monitor_dir: str | None = None,
    vec_env_type: str = "dummy",
    seed: int | None = None,
    monitor_kwargs: dict[str, Any] | None = None,
):
    """创建向量化环境；n_envs>1 且 subproc 时使用 SubprocVecEnv。"""
    if env_class is None:
        from snake_env import SnakeEnv

        env_class = SnakeEnv

    vec_env_type = vec_env_type.lower()
    if vec_env_type not in ("dummy", "subproc"):
        raise ValueError(f"vec_env_type 须为 dummy/subproc，收到 {vec_env_type}")

    if n_envs > 1 and vec_env_type == "subproc":
        vec_env_cls = SubprocVecEnv
    else:
        vec_env_cls = DummyVecEnv

    kwargs: dict[str, Any] = {
        "n_envs": n_envs,
        "env_kwargs": env_kwargs or {},
        "vec_env_cls": vec_env_cls,
    }
    if seed is not None:
        kwargs["seed"] = seed
    if monitor_dir:
        os.makedirs(monitor_dir, exist_ok=True)
        kwargs["monitor_dir"] = monitor_dir
        kwargs["monitor_kwargs"] = monitor_kwargs or {
            "info_keywords": ("coverage", "score"),
        }

    env = make_vec_env(env_class, **kwargs)
    # 便于验证 SubprocVecEnv 是否生效
    env.vec_env_type = vec_env_type  # type: ignore[attr-defined]
    env.vec_env_cls_name = vec_env_cls.__name__  # type: ignore[attr-defined]
    return env


def algo_training_defaults(algo: str, grid_size: int) -> dict[str, Any]:
    """按算法与地图尺寸给出偏吞吐的默认超参（仍保持训练稳定）。"""
    algo = algo.lower()
    side = grid_size
    # 大图环境 step 更重，减少 rollout 长度与 epoch 数以提高更新频率
    if side <= 8:
        n_steps = 2048
        n_epochs = 6
    elif side <= 12:
        n_steps = 1536
        n_epochs = 5
    else:
        n_steps = 1024
        n_epochs = 4

    common = dict(
        n_steps=n_steps,
        batch_size=256,
        n_epochs=n_epochs,
        learning_rate=2.5e-4,
        gamma=0.995,
        ent_coef=0.01,
        eval_freq=200_000,
        save_freq=200_000,
        n_eval_episodes=5,
        log_interval=10,
    )

    if algo in ("dqn", "qrdqn"):
        return dict(
            common,
            buffer_size=100_000 if side <= 10 else 200_000,
            batch_size=128,
            learning_starts=5_000 if side <= 10 else 10_000,
        )
    return common


def apply_torch_compile(model, enabled: bool) -> None:
    """可选 torch.compile 策略网络（PyTorch 2+）。"""
    if not enabled:
        return
    import torch as th

    if not hasattr(th, "compile"):
        print("[torch-compile] 当前 PyTorch 不支持 torch.compile，已跳过。")
        return
    try:
        model.policy = th.compile(model.policy)  # type: ignore[assignment]
        print("[torch-compile] policy 已启用 torch.compile")
    except Exception as exc:  # noqa: BLE001
        print(f"[torch-compile] 启用失败，继续未编译训练: {exc}")
