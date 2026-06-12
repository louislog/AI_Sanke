"""手动游玩贪吃蛇。"""

import ai_snake.runtime  # noqa: F401

from ai_snake.snake_game import play_manual


def main(argv: list[str] | None = None) -> None:
    del argv
    play_manual()


if __name__ == "__main__":
    main()
