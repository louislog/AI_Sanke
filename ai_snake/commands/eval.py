import ai_snake.runtime  # noqa: F401

"""统一评估入口：支持规则 / 搜索 / 混合 / RL 策略，多地图尺寸对比。

示例：
    # 单策略评估
    snake-ai eval --policy hybrid --grid-size 8 --n-episodes 20
    snake-ai eval --policy rl --model tmp/best/best_model.zip --grid-size 8

    # 多策略 x 多尺寸对比表
    snake-ai eval --compare random,search,hamiltonian,hybrid --grid-sizes 6,8,10 --n-episodes 10

    # 视频 / GIF 导出与死亡回放
    snake-ai eval --policy hybrid --grid-size 10 --out-video demo.mp4
    snake-ai eval --policy rl --model tmp/best/best_model.zip --replay-dir replays/
"""

import argparse
import subprocess
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

from ai_snake.policies import ALL_POLICIES, make_policy
from ai_snake.policies.base import BasePolicy
from ai_snake.snake_env import SnakeEnv


def _has_ffmpeg() -> bool:
    import shutil

    return shutil.which("ffmpeg") is not None


class VideoWriter:
    """视频导出：mp4 走 ffmpeg；gif 优先 ffmpeg，缺失时回退 Pillow。"""

    def __init__(self, out_path: str, fps: int = 15):
        self.fps = fps
        self.out_path = out_path
        self.video_proc = None
        self._is_gif = out_path.endswith(".gif")
        self._use_pil = self._is_gif and not _has_ffmpeg()
        self._pil_frames: list = []
        if not self._is_gif and not _has_ffmpeg():
            raise RuntimeError(
                "未找到 ffmpeg，无法导出 mp4。请安装 ffmpeg 或改用 .gif 输出。"
            )

    def _start(self, frame):
        if self.video_proc is not None or frame is None:
            return
        height, width = frame.shape[:2]
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(self.fps), "-i", "-", "-an",
        ]
        if self._is_gif:
            cmd += ["-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"]
        else:
            cmd += ["-vcodec", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
        cmd.append(self.out_path)
        self.video_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        if frame is None:
            return
        if self._use_pil:
            from PIL import Image

            self._pil_frames.append(Image.fromarray(frame))
            return
        self._start(frame)
        if self.video_proc and self.video_proc.stdin:
            self.video_proc.stdin.write(frame.tobytes())

    def close(self):
        if self._use_pil:
            if self._pil_frames:
                self._pil_frames[0].save(
                    self.out_path,
                    save_all=True,
                    append_images=self._pil_frames[1:],
                    duration=int(1000 / self.fps),
                    loop=0,
                )
            self._pil_frames = []
            return
        if self.video_proc and self.video_proc.stdin:
            self.video_proc.stdin.close()
        if self.video_proc:
            self.video_proc.wait()
            self.video_proc = None


@dataclass
class EpisodeResult:
    score: int
    length: int
    steps: int
    coverage: float
    won: bool
    death_reason: str | None
    truncated: bool


def run_episode(
    env: SnakeEnv,
    policy: BasePolicy,
    *,
    video_writer: VideoWriter | None = None,
    replay_frames: deque | None = None,
    render_human: bool = False,
    seed: int | None = None,
) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    policy.reset(env)
    if replay_frames is not None:
        replay_frames.clear()

    while True:
        action = policy.select_action(env, obs)
        obs, _, terminated, truncated, info = env.step(action)

        if video_writer or replay_frames is not None or render_human:
            frame = env.render()
            if video_writer:
                video_writer.write(frame)
                if terminated or truncated:
                    for _ in range(10):
                        video_writer.write(frame)
            if replay_frames is not None and frame is not None:
                replay_frames.append(frame)
            if render_human:
                time.sleep(0.04)

        if terminated or truncated:
            reason = info["death_reason"]
            if truncated and not terminated:
                reason = "timeout"
            return EpisodeResult(
                score=info["score"],
                length=info["snake_length"],
                steps=info["steps"],
                coverage=info["coverage"],
                won=info["won"],
                death_reason=None if info["won"] else reason,
                truncated=truncated,
            )


def summarize(results: list[EpisodeResult]) -> dict:
    n = len(results)
    deaths = Counter(r.death_reason for r in results if r.death_reason)
    return {
        "episodes": n,
        "average_score": sum(r.score for r in results) / n,
        "max_score": max(r.score for r in results),
        "average_length": sum(r.length for r in results) / n,
        "max_length": max(r.length for r in results),
        "average_steps": sum(r.steps for r in results) / n,
        "coverage_ratio": sum(r.coverage for r in results) / n,
        "max_coverage": max(r.coverage for r in results),
        "full_map_success_rate": sum(r.won for r in results) / n,
        "death_reasons": dict(deaths),
    }


def evaluate_policy(
    policy_name: str,
    grid_size: int,
    n_episodes: int,
    *,
    model_path: str | None = None,
    algo: str = "auto",
    obs_mode: str = "grid_full",
    grid_pad_size: int | None = None,
    max_steps_factor: int = 60,
    out_video: str | None = None,
    replay_dir: str | None = None,
    replay_steps: int = 90,
    render_human: bool = False,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    if render_human:
        render_mode = "human"
    elif out_video or replay_dir:
        render_mode = "rgb_array"
    else:
        render_mode = None

    env = SnakeEnv(
        width=grid_size,
        height=grid_size,
        render_mode=render_mode,
        obs_mode=obs_mode,
        grid_pad_size=grid_pad_size or grid_size,
        max_steps_factor=max_steps_factor,
        reward="default",
    )
    if policy_name == "rl":
        policy = make_policy("rl", model_path=model_path, algo=algo)
    else:
        policy = make_policy(policy_name)

    video_writer = VideoWriter(out_video) if out_video else None
    replay_frames: deque | None = deque(maxlen=replay_steps) if replay_dir else None
    if replay_dir:
        Path(replay_dir).mkdir(parents=True, exist_ok=True)

    results: list[EpisodeResult] = []
    for episode in range(n_episodes):
        result = run_episode(
            env,
            policy,
            video_writer=video_writer,
            replay_frames=replay_frames,
            render_human=render_human,
            seed=None if seed is None else seed + episode,
        )
        results.append(result)

        if verbose:
            status = "WIN" if result.won else (result.death_reason or "?")
            print(
                f"  Episode {episode + 1}: score={result.score} "
                f"length={result.length} steps={result.steps} "
                f"coverage={result.coverage:.2%} [{status}]"
            )

        # 死亡前若干步状态回放
        if replay_dir and not result.won and replay_frames:
            ext = "mp4" if _has_ffmpeg() else "gif"
            replay_path = str(
                Path(replay_dir)
                / f"{policy_name}_g{grid_size}_ep{episode + 1}_{result.death_reason}.{ext}"
            )
            writer = VideoWriter(replay_path, fps=8)
            for frame in replay_frames:
                writer.write(frame)
            writer.close()
            if verbose:
                print(f"    death replay saved: {replay_path}")

    if video_writer:
        video_writer.close()
    env.close()

    summary = summarize(results)
    summary["policy"] = policy_name
    summary["grid_size"] = grid_size
    return summary


def print_summary(summary: dict) -> None:
    print(f"\n=== {summary['policy']} @ {summary['grid_size']}x{summary['grid_size']} ===")
    print(f"  episodes              : {summary['episodes']}")
    print(f"  average_score         : {summary['average_score']:.2f}")
    print(f"  max_score             : {summary['max_score']}")
    print(f"  average_length        : {summary['average_length']:.2f}")
    print(f"  max_length            : {summary['max_length']}")
    print(f"  average_steps         : {summary['average_steps']:.1f}")
    print(f"  coverage_ratio        : {summary['coverage_ratio']:.2%}")
    print(f"  max_coverage          : {summary['max_coverage']:.2%}")
    print(f"  full_map_success_rate : {summary['full_map_success_rate']:.2%}")
    print(f"  death_reasons         : {summary['death_reasons'] or '{}'}")


def print_comparison_table(summaries: list[dict]) -> None:
    headers = (
        "policy", "grid", "avg_score", "max_score", "avg_len",
        "coverage", "full_map", "deaths",
    )
    rows = []
    for s in summaries:
        deaths = ",".join(f"{k}:{v}" for k, v in s["death_reasons"].items()) or "-"
        rows.append((
            s["policy"],
            f"{s['grid_size']}x{s['grid_size']}",
            f"{s['average_score']:.1f}",
            str(s["max_score"]),
            f"{s['average_length']:.1f}",
            f"{s['coverage_ratio']:.1%}",
            f"{s['full_map_success_rate']:.0%}",
            deaths,
        ))

    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print("\n" + line)
    print("-" * len(line))
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Evaluate Snake policies.")
    parser.add_argument("--policy", choices=ALL_POLICIES, default=None, help="单策略评估")
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="逗号分隔的策略列表，输出对比表，如 'search,hamiltonian,hybrid,rl'",
    )
    parser.add_argument("--model", type=str, default=None, help="RL 模型 checkpoint (.zip)")
    parser.add_argument("--algo", type=str, default="auto", help="RL 算法（auto 自动尝试）")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument(
        "--grid-sizes", type=str, default=None, help="逗号分隔的多尺寸，如 '6,8,10'"
    )
    parser.add_argument("--grid-pad-size", type=int, default=None, help="RL 观测 padding 尺寸")
    parser.add_argument(
        "--obs-mode", choices=("grid_full", "grid", "vector"), default="grid_full"
    )
    parser.add_argument("--max-steps-factor", type=int, default=60)
    parser.add_argument("--out-video", type=str, default=None, help="mp4 或 gif 输出路径")
    parser.add_argument("--replay-dir", type=str, default=None, help="死亡回放保存目录")
    parser.add_argument("--replay-steps", type=int, default=90, help="回放保留的死亡前帧数")
    parser.add_argument("--render", action="store_true", help="实时窗口渲染")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    if args.policy is None and args.compare is None:
        # 兼容旧用法：默认评估 RL 模型
        args.policy = "rl"

    if args.policy == "rl" or (args.compare and "rl" in args.compare.split(",")):
        if args.model is None:
            log_dir = Path("tmp")
            candidates = sorted(log_dir.glob("**/*.zip")) if log_dir.exists() else []
            if not candidates:
                raise FileNotFoundError("未找到模型 checkpoint，请用 --model 指定。")
            args.model = str(candidates[-1])
            print(f"Using latest checkpoint: {args.model}")

    grid_sizes = (
        [int(s) for s in args.grid_sizes.split(",")]
        if args.grid_sizes
        else [args.grid_size]
    )

    common = dict(
        n_episodes=args.n_episodes,
        model_path=args.model,
        algo=args.algo,
        obs_mode=args.obs_mode,
        grid_pad_size=args.grid_pad_size,
        max_steps_factor=args.max_steps_factor,
        replay_dir=args.replay_dir,
        replay_steps=args.replay_steps,
        seed=args.seed,
    )

    summaries: list[dict] = []
    if args.compare:
        policies = [p.strip() for p in args.compare.split(",") if p.strip()]
        for grid_size in grid_sizes:
            for policy_name in policies:
                print(f"\n>>> {policy_name} @ {grid_size}x{grid_size}")
                summaries.append(
                    evaluate_policy(policy_name, grid_size, **common)
                )
        print_comparison_table(summaries)
    else:
        for grid_size in grid_sizes:
            print(f"\n>>> {args.policy} @ {grid_size}x{grid_size}")
            summary = evaluate_policy(
                args.policy,
                grid_size,
                out_video=args.out_video if grid_size == grid_sizes[0] else None,
                render_human=args.render,
                **common,
            )
            summaries.append(summary)
            print_summary(summary)
        if len(summaries) > 1:
            print_comparison_table(summaries)


if __name__ == "__main__":
    main()
