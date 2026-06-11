import runtime  # noqa: F401

import argparse
import subprocess
import time
from pathlib import Path

from snake_env import SnakeEnv
from stable_baselines3 import PPO

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    _HAS_MASKABLE = True
except ImportError:
    MaskablePPO = None  # type: ignore[misc, assignment]
    _HAS_MASKABLE = False


class VideoWriter:
    def __init__(self, out_path: str | None = None, fps: int = 15):
        self.fps = fps
        self.out_path = out_path
        self.video_proc = None

    def _start(self, frame):
        if self.out_path and frame is not None and self.video_proc is None:
            height, width = frame.shape[:2]
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(self.fps),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                self.out_path,
            ]
            self.video_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        if self.out_path and frame is not None:
            self._start(frame)
            if self.video_proc and self.video_proc.stdin:
                self.video_proc.stdin.write(frame.tobytes())

    def close(self):
        if self.video_proc and self.video_proc.stdin:
            self.video_proc.stdin.close()
        if self.video_proc:
            self.video_proc.wait()
            self.video_proc = None


def _load_model(model_path: str, env: SnakeEnv):
    if _HAS_MASKABLE:
        try:
            return MaskablePPO.load(model_path, env=env)
        except Exception:
            pass
    return PPO.load(model_path, env=env)


def _predict_action(model, obs, env: SnakeEnv):
    if _HAS_MASKABLE and isinstance(model, MaskablePPO):
        masks = get_action_masks(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        return action
    action, _ = model.predict(obs, deterministic=True)
    return action


def eval_agent(
    out_video: str | None = None,
    model_path: str | None = None,
    n_episodes: int = 5,
    obs_mode: str = "grid",
    grid_size: int = 20,
    grid_pad_size: int | None = None,
):
    render_mode = "rgb_array" if out_video else "human"
    env = SnakeEnv(
        width=grid_size,
        height=grid_size,
        render_mode=render_mode,
        obs_mode=obs_mode,
        grid_pad_size=grid_pad_size or grid_size,
    )
    model = _load_model(model_path, env)
    video_writer = VideoWriter(out_video) if out_video else None

    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        step = 0

        while True:
            action = _predict_action(model, obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if video_writer:
                frame = env.render()
                if terminated or truncated:
                    for _ in range(10):
                        video_writer.write(frame)
                else:
                    video_writer.write(frame)
            else:
                env.render()
                time.sleep(0.05)

            if terminated or truncated:
                won = " WIN" if info.get("won") else ""
                print(
                    f"Episode {episode + 1}: score={info['score']}, "
                    f"steps={step}, reward={total_reward:.2f}{won}"
                )
                break

    if video_writer:
        video_writer.close()

    env.close()


def _main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Snake PPO agent.")
    parser.add_argument("--out-video", type=str, help="Optional MP4 output path")
    parser.add_argument("--model", type=str, help="Path to model checkpoint (.zip)")
    parser.add_argument("--n-episodes", type=int, default=5, help="Evaluation episodes")
    parser.add_argument(
        "--obs-mode",
        choices=("grid", "vector"),
        default="grid",
        help="Must match the observation mode used during training",
    )
    parser.add_argument("--grid-size", type=int, default=20, help="Board width and height")
    parser.add_argument(
        "--grid-pad-size",
        type=int,
        default=None,
        help="Padded observation size (defaults to --grid-size)",
    )
    args = parser.parse_args()

    model = args.model
    if model is None:
        log_dir = Path("tmp")
        model_files = sorted(log_dir.glob("*.zip")) if log_dir.exists() else []
        if not model_files:
            raise FileNotFoundError("No model checkpoint found in tmp/; specify --model.")
        model = str(model_files[-1])
        print(f"Using latest checkpoint: {model}")

    eval_agent(
        out_video=args.out_video,
        model_path=model,
        n_episodes=args.n_episodes,
        obs_mode=args.obs_mode,
        grid_size=args.grid_size,
        grid_pad_size=args.grid_pad_size,
    )


if __name__ == "__main__":
    _main()
