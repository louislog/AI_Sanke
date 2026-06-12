"""模仿学习：专家数据采集 + 行为克隆。"""

import ai_snake.runtime  # noqa: F401

import argparse
import os

import numpy as np
import torch as th
from stable_baselines3.common.env_util import make_vec_env

from ai_snake.algos import build_model
from ai_snake.policies import RULE_POLICIES, make_policy
from ai_snake.snake_env import SnakeEnv


def collect_expert(
    policy_name: str,
    grid_size: int,
    n_episodes: int,
    obs_mode: str,
    grid_pad_size: int | None,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    env = SnakeEnv(
        width=grid_size,
        height=grid_size,
        obs_mode=obs_mode,
        grid_pad_size=grid_pad_size or grid_size,
        reward="default",
    )
    policy = make_policy(policy_name)

    observations: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    episode_scores: list[int] = []
    episode_coverages: list[float] = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=None if seed is None else seed + episode)
        policy.reset(env)
        while True:
            action = policy.select_action(env, obs)
            observations.append(obs)
            actions.append(action)
            masks.append(env.action_masks())
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                episode_scores.append(info["score"])
                episode_coverages.append(info["coverage"])
                break
        print(
            f"Episode {episode + 1}/{n_episodes}: score={info['score']} "
            f"coverage={info['coverage']:.2%} won={info['won']}"
        )

    env.close()
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "action_masks": np.asarray(masks, dtype=bool),
        "episode_scores": np.asarray(episode_scores, dtype=np.int64),
        "episode_coverages": np.asarray(episode_coverages, dtype=np.float32),
    }


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


def _run_collect(argv: list[str] | None) -> None:
    parser = argparse.ArgumentParser(description="Collect expert demonstrations.")
    parser.add_argument("--policy", choices=RULE_POLICIES, default="hybrid")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--grid-pad-size", type=int, default=None)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument(
        "--obs-mode", choices=("grid_full", "grid", "vector"), default="grid_full"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="输出 npz 路径")
    args = parser.parse_args(argv)

    out = args.out or f"data/expert_{args.policy}_{args.grid_size}.npz"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    data = collect_expert(
        args.policy,
        args.grid_size,
        args.n_episodes,
        args.obs_mode,
        args.grid_pad_size,
        seed=args.seed,
    )
    np.savez_compressed(out, **data)

    scores = data["episode_scores"]
    coverages = data["episode_coverages"]
    print(
        f"\nSaved {len(data['actions']):,} transitions from {len(scores)} episodes to {out}\n"
        f"expert avg_score={scores.mean():.1f} max_score={scores.max()} "
        f"avg_coverage={coverages.mean():.2%}"
    )


def _run_train(argv: list[str] | None) -> None:
    parser = argparse.ArgumentParser(description="Behavior cloning from expert data.")
    parser.add_argument("--data", type=str, required=True, help="imitate collect 输出的 npz")
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
    args = parser.parse_args(argv)

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
    print(f"评估: snake-ai eval --policy rl --model {args.out}.zip --grid-size {args.grid_size}")
    print(f"微调: snake-ai train --init-model {args.out}.zip --grid-size {args.grid_size}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Imitation learning utilities.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("collect", help="采集专家演示数据")
    sub.add_parser("train", help="行为克隆预训练")
    args, rest = parser.parse_known_args(argv)

    if args.command == "collect":
        _run_collect(rest)
    else:
        _run_train(rest)


if __name__ == "__main__":
    main()
