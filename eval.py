import runtime  # noqa: F401

import argparse
import subprocess
import time
from pathlib import Path

from snake_env import SnakeEnv
from stable_baselines3 import PPO


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


def eval_agent(out_video: str | None = None, model_path: str | None = None, n_episodes: int = 5):
    render_mode = "rgb_array" if out_video else "human"
    env = SnakeEnv(render_mode=render_mode)
    model = PPO.load(model_path, env=env)
    video_writer = VideoWriter(out_video) if out_video else None

    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        step = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
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
                print(
                    f"Episode {episode + 1}: score={info['score']}, "
                    f"steps={step}, reward={total_reward:.2f}"
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
    )


if __name__ == "__main__":
    _main()
