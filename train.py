import runtime  # noqa: F401

import argparse
import os

from snake_env import SnakeEnv
from snake_game import SnakeGame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env


def _main():
    parser = argparse.ArgumentParser(description="Train a PPO agent to play Snake.")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=16,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2_000_000,
        help="Number of timesteps to train for",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50_000,
        help="Checkpoint frequency in environment steps",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50_000,
        help="Evaluation frequency in environment steps",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=SnakeGame.DEFAULT_WIDTH,
        help="Grid width and height",
    )
    args = parser.parse_args()

    log_dir = "tmp/"
    best_dir = os.path.join(log_dir, "best")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    vec_env = make_vec_env(
        SnakeEnv,
        n_envs=args.n_envs,
        env_kwargs={
            "width": args.grid_size,
            "height": args.grid_size,
            "render_mode": None,
        },
        monitor_dir=log_dir,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        device="cpu",
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    )

    tb_log_name = "ppo"
    if args.n_envs > 0:
        tb_log_name += f"_nenv{args.n_envs}"

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=log_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
    )

    eval_env = SnakeEnv(
        width=args.grid_size,
        height=args.grid_size,
        render_mode=None,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=tb_log_name,
    )

    final_path = os.path.join(log_dir, "rl_model_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}.zip")
    print(f"Best eval model saved to {os.path.join(best_dir, 'best_model.zip')}")


if __name__ == "__main__":
    _main()
