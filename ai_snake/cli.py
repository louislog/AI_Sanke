"""统一 CLI：snake-ai <command> ..."""

from __future__ import annotations

import argparse
import sys

import ai_snake.runtime  # noqa: F401

from ai_snake.commands import bench, eval as eval_cmd, imitate, play, train


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="snake-ai",
        description="Snake AI：训练、评估、模仿学习与基准测试",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="强化学习训练")
    sub.add_parser("eval", help="策略评估与对比")
    imitate_parser = sub.add_parser("imitate", help="模仿学习（collect / train）")
    imitate_parser.add_argument("imitate_command", nargs="?", choices=("collect", "train"))
    bench_parser = sub.add_parser("bench", help="性能基准（env / obs）")
    bench_parser.add_argument("bench_command", nargs="?", choices=("env", "obs"))
    sub.add_parser("play", help="手动游玩")

    args, rest = parser.parse_known_args(argv)

    if args.command == "train":
        train.main(rest)
    elif args.command == "eval":
        eval_cmd.main(rest)
    elif args.command == "imitate":
        imitate_argv = ([args.imitate_command] if args.imitate_command else []) + rest
        imitate.main(imitate_argv)
    elif args.command == "bench":
        bench_argv = ([args.bench_command] if args.bench_command else []) + rest
        bench.main(bench_argv)
    elif args.command == "play":
        play.main(rest)
    else:
        parser.error(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
