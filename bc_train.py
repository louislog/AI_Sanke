import runtime  # noqa: F401

"""行为克隆（Behavior Cloning）：用专家数据监督训练策略网络。

产出的模型与 SB3 checkpoint 完全兼容，可直接：
1. python eval.py --policy rl --model tmp/bc_model.zip 评估；
2. python train.py --init-model tmp/bc_model.zip 继续 RL fine-tuning。

BC + RL 相比纯 RL 的优势：专家（Hamiltonian/hybrid）演示直接提供了
长期规划行为，网络无需从随机探索中发现“绕路保命”这类稀疏回报策略，
fine-tuning 阶段从一个已会玩的策略出发，样本效率和最终覆盖率都更高。

示例：
    python bc_train.py --data data/expert_hybrid_8.npz --epochs 20 \
        --grid-size 8 --obs-mode grid_full --out tmp/bc_model
"""

import argparse

import numpy as np
import torch as th
from stable_baselines3.common.env_util import make_vec_env

from algos import build_model
from snake_env import SnakeEnv


def bc_train(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 3e-4,
    val_ratio: float = 0.05,
) -> None:
    policy = model.policy
    device = policy.device
    optimizer = th.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = th.nn.CrossEntropyLoss()

    n = len(actions)
    indices = np.random.permutation(n)
    n_val = max(1, int(n * val_ratio))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    # 数据保留在 CPU，按 batch 搬运到设备，避免大数据集占满显存
    obs_t = th.as_tensor(observations)
    act_t = th.as_tensor(actions)

    def _logits(idx: np.ndarray) -> th.Tensor:
        return policy.get_distribution(obs_t[idx].to(device)).distribution.logits

    for epoch in range(1, epochs + 1):
        policy.train()
        perm = np.random.permutation(train_idx)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(perm), batch_size):
            batch = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(_logits(batch), act_t[batch].to(device))
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            total_loss += float(loss)
            n_batches += 1

        policy.eval()
        with th.no_grad():
            val_logits = _logits(val_idx)
            val_targets = act_t[val_idx].to(device)
            val_loss = float(loss_fn(val_logits, val_targets))
            val_acc = float((val_logits.argmax(dim=1) == val_targets).float().mean())
        print(
            f"Epoch {epoch:3d}/{epochs}: train_loss={total_loss / max(n_batches, 1):.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2%}"
        )


def _main():
    parser = argparse.ArgumentParser(description="Behavior cloning from expert data.")
    parser.add_argument("--data", type=str, required=True, help="collect_expert.py 输出的 npz")
    parser.add_argument("--algo", type=str, default="maskable_ppo")
    parser.add_argument("--grid-size", type=int, default=8, help="须与采集数据一致")
    parser.add_argument("--grid-pad-size", type=int, default=None)
    parser.add_argument(
        "--obs-mode", choices=("grid_full", "grid", "vector"), default="grid_full"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--features-dim", type=int, default=512)
    parser.add_argument("--out", type=str, default="tmp/bc_model")
    args = parser.parse_args()

    data = np.load(args.data)
    observations, actions = data["observations"], data["actions"]
    print(f"Loaded {len(actions):,} transitions from {args.data}")

    vec_env = make_vec_env(
        SnakeEnv,
        n_envs=1,
        env_kwargs=dict(
            width=args.grid_size,
            height=args.grid_size,
            obs_mode=args.obs_mode,
            grid_pad_size=args.grid_pad_size or args.grid_size,
        ),
    )
    model = build_model(
        args.algo,
        vec_env,
        obs_mode=args.obs_mode,
        features_dim=args.features_dim,
        verbose=0,
    )

    if observations.shape[1:] != model.observation_space.shape:
        raise ValueError(
            f"数据观测形状 {observations.shape[1:]} 与模型 "
            f"{model.observation_space.shape} 不一致，请检查 --grid-size/--obs-mode"
        )

    bc_train(
        model,
        observations,
        actions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    model.save(args.out)
    print(f"BC model saved to {args.out}.zip")
    print(f"评估: python eval.py --policy rl --model {args.out}.zip --grid-size {args.grid_size}")
    print(f"微调: python train.py --init-model {args.out}.zip --grid-size {args.grid_size}")


if __name__ == "__main__":
    _main()
