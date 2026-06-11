import math
import random
import sys
import warnings
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pygame as pygame_types

_pygame: "pygame_types | None" = None


def _get_pygame():
    global _pygame
    if _pygame is None:
        import pygame

        _pygame = pygame
    return _pygame


def _glyph(*rows: int) -> tuple[str, ...]:
    return tuple(
        "".join("#" if (row >> (4 - col)) & 1 else "." for col in range(5))
        for row in rows
    )


_BITMAP_FONT: dict[str, tuple[str, ...]] = {
    " ": _glyph(0, 0, 0, 0, 0, 0, 0),
    "0": _glyph(0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "1": _glyph(0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": _glyph(0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F),
    "3": _glyph(0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E),
    "4": _glyph(0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": _glyph(0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E),
    "6": _glyph(0x0E, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": _glyph(0x1F, 0x01, 0x02, 0x04, 0x04, 0x04, 0x04),
    "8": _glyph(0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": _glyph(0x0E, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x0E),
    ".": _glyph(0, 0, 0, 0, 0, 0x0C, 0x0C),
    ":": _glyph(0, 0x0C, 0x0C, 0, 0x0C, 0x0C, 0),
    "-": _glyph(0, 0, 0, 0x1F, 0, 0, 0),
    "/": _glyph(0x01, 0x02, 0x04, 0x08, 0x10, 0, 0),
    "A": _glyph(0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "B": _glyph(0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E),
    "C": _glyph(0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E),
    "E": _glyph(0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F),
    "G": _glyph(0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E),
    "H": _glyph(0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "I": _glyph(0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "L": _glyph(0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F),
    "M": _glyph(0x11, 0x1B, 0x15, 0x11, 0x11, 0x11, 0x11),
    "N": _glyph(0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11),
    "O": _glyph(0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "P": _glyph(0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10),
    "R": _glyph(0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11),
    "S": _glyph(0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E),
    "T": _glyph(0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "V": _glyph(0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "X": _glyph(0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11),
    "Y": _glyph(0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04),
    "?": _glyph(0x0E, 0x11, 0x01, 0x02, 0x04, 0, 0x04),
}

_BITMAP_STYLE_SCALE = {
    "title": 2,
    "label": 2,
    "value": 2,
    "overlay": 2,
}


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


@dataclass
class Position:
    x: int
    y: int


class SnakeGame:
    """贪吃蛇游戏核心逻辑与 Pygame 渲染。"""

    DEFAULT_WIDTH = 20
    DEFAULT_HEIGHT = 20
    CELL_SIZE = 20
    HUD_HEIGHT = 52
    FPS = 15
    HIGH_SCORE_FILE = Path(__file__).with_name(".snake_highscore")

    REWARD_FOOD = 10.0
    REWARD_DEATH = -50.0
    REWARD_STEP = 0.0
    REWARD_WIN_BASE = 100.0

    CARDINAL_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))

    # 8 方向射线：N, NE, E, SE, S, SW, W, NW
    RAY_DIRECTIONS = (
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
    )
    BASE_FEATURE_DIM = 10
    RAY_FEATURE_DIM = len(RAY_DIRECTIONS)
    SPATIAL_FEATURE_DIM = 6
    FEATURE_DIM = BASE_FEATURE_DIM + RAY_FEATURE_DIM + SPATIAL_FEATURE_DIM

    COLOR_BG = (18, 22, 30)
    COLOR_BG_ALT = (22, 27, 36)
    COLOR_GRID = (34, 40, 52)
    COLOR_SNAKE_HEAD = (72, 210, 150)
    COLOR_SNAKE_BODY = (48, 168, 115)
    COLOR_SNAKE_OUTLINE = (28, 100, 72)
    COLOR_FOOD = (255, 88, 88)
    COLOR_FOOD_GLOW = (255, 130, 100)
    COLOR_HUD_BG = (12, 16, 24)
    COLOR_HUD_BORDER = (42, 50, 64)
    COLOR_HUD_TEXT = (225, 230, 240)
    COLOR_HUD_LABEL = (130, 142, 160)
    COLOR_HUD_ACCENT = (72, 210, 150)

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        render_mode: Literal["human", "rgb_array"] | None = None,
        seed: int | None = None,
    ):
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self._rng = random.Random(seed)

        self.screen = None
        self.clock = None
        self.surface = None
        self._font_title = None
        self._font_label = None
        self._font_value = None
        self._font_overlay = None
        self._fonts_available = False
        self.high_score = self._load_high_score()
        self.session_scores: list[int] = []

        self.board_width_px = self.width * self.CELL_SIZE
        self.board_height_px = self.height * self.CELL_SIZE
        self.surface_width_px = self.board_width_px
        self.surface_height_px = self.board_height_px + self.HUD_HEIGHT

        if render_mode is not None:
            pygame = _get_pygame()
            if not pygame.get_init():
                pygame.init()

            if render_mode == "human":
                self.screen = pygame.display.set_mode(
                    (self.surface_width_px, self.surface_height_px)
                )
                pygame.display.set_caption("贪吃蛇 Snake")
                self.clock = pygame.time.Clock()

            self.surface = pygame.Surface(
                (self.surface_width_px, self.surface_height_px)
            )
            self._init_fonts()

        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)

        center_x = self.width // 2
        center_y = self.height // 2
        self.snake = [
            Position(center_x, center_y),
            Position(center_x - 1, center_y),
            Position(center_x - 2, center_y),
        ]
        self.direction = Direction.RIGHT
        self.food = self._generate_food()
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.won = False

    def _generate_food(self) -> Position:
        occupied = {(segment.x, segment.y) for segment in self.snake}
        for _ in range(1000):
            food = Position(
                self._rng.randrange(self.width),
                self._rng.randrange(self.height),
            )
            if (food.x, food.y) not in occupied:
                return food
        return Position(0, 0)

    def _get_new_direction(self, current: Direction, action: int) -> Direction:
        if action == 0:
            return current
        if action == 1:
            return Direction((current + 1) % 4)
        if action == 2:
            return Direction((current + 3) % 4)
        return current

    def _next_position(self, pos: Position, direction: Direction) -> Position:
        deltas = [
            Position(0, -1),
            Position(1, 0),
            Position(0, 1),
            Position(-1, 0),
        ]
        delta = deltas[direction]
        return Position(pos.x + delta.x, pos.y + delta.y)

    def _will_grow_at(self, pos: Position) -> bool:
        return pos.x == self.food.x and pos.y == self.food.y

    def _is_collision(self, pos: Position, *, will_grow: bool = False) -> bool:
        if pos.x < 0 or pos.x >= self.width or pos.y < 0 or pos.y >= self.height:
            return True
        # 不吃食物时尾巴本步会移走，不应算作障碍
        body = self.snake if will_grow else self.snake[:-1]
        return any(segment.x == pos.x and segment.y == pos.y for segment in body)

    def food_manhattan_distance(self) -> int:
        head = self.snake[0]
        return abs(self.food.x - head.x) + abs(self.food.y - head.y)

    def food_bfs_distance(self) -> int | None:
        passable = self._passable_cells()
        head = self.snake[0]
        return self._bfs_distance(
            (head.x, head.y),
            (self.food.x, self.food.y),
            passable,
        )

    def board_capacity(self) -> int:
        return self.width * self.height

    def is_board_full(self) -> bool:
        return len(self.snake) >= self.board_capacity()

    def win_reward(self) -> float:
        return self.REWARD_WIN_BASE + self.board_capacity()

    def valid_action_mask(self) -> list[bool]:
        head = self.snake[0]
        masks: list[bool] = []
        for action in range(3):
            direction = self._get_new_direction(self.direction, action)
            new_head = self._next_position(head, direction)
            will_grow = self._will_grow_at(new_head)
            masks.append(not self._is_collision(new_head, will_grow=will_grow))
        return masks

    def _max_ray_distance_to_wall(self, head: Position, dx: int, dy: int) -> int:
        limits: list[int] = []
        if dx < 0:
            limits.append(head.x)
        elif dx > 0:
            limits.append(self.width - 1 - head.x)
        if dy < 0:
            limits.append(head.y)
        elif dy > 0:
            limits.append(self.height - 1 - head.y)
        return min(limits) if limits else 0

    def _ray_clear_distance(self, head: Position, dx: int, dy: int) -> int:
        occupied = {(segment.x, segment.y) for segment in self.snake[1:]}
        x, y = head.x, head.y
        steps = 0
        while True:
            x += dx
            y += dy
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                break
            if (x, y) in occupied:
                break
            steps += 1
        return steps

    def _ray_features(self, head: Position) -> list[float]:
        features: list[float] = []
        for dx, dy in self.RAY_DIRECTIONS:
            max_dist = self._max_ray_distance_to_wall(head, dx, dy)
            clear_dist = self._ray_clear_distance(head, dx, dy)
            if max_dist <= 0:
                features.append(0.0)
            else:
                features.append(clear_dist / max_dist)
        return features

    def _passable_cells(self) -> set[tuple[int, int]]:
        # 蛇身中间段为障碍；头所在格可站立，尾巴下步会移走视为可通过
        occupied = {(segment.x, segment.y) for segment in self.snake[1:-1]}
        return {
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied
        }

    def _bfs_distance(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        passable: set[tuple[int, int]],
    ) -> int | None:
        if start == goal:
            return 0
        if goal not in passable:
            return None

        queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
        visited = {start}
        while queue:
            (x, y), dist = queue.popleft()
            for dx, dy in self.CARDINAL_DELTAS:
                nxt = (x + dx, y + dy)
                if nxt not in passable or nxt in visited:
                    continue
                if nxt == goal:
                    return dist + 1
                visited.add(nxt)
                queue.append((nxt, dist + 1))
        return None

    def reachable_ratio(self) -> float:
        passable = self._passable_cells()
        if not passable:
            return 0.0

        head = self.snake[0]
        start = (head.x, head.y)
        if start not in passable:
            return 0.0

        queue: deque[tuple[int, int]] = deque([start])
        visited = {start}
        while queue:
            x, y = queue.popleft()
            for dx, dy in self.CARDINAL_DELTAS:
                nxt = (x + dx, y + dy)
                if nxt in passable and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return len(visited) / len(passable)

    def _relative_direction_flags(
        self, dx: int, dy: int, direction: Direction
    ) -> tuple[float, float, float]:
        if direction == Direction.UP:
            return float(dy < 0), float(dx < 0), float(dx > 0)
        if direction == Direction.RIGHT:
            return float(dx > 0), float(dy < 0), float(dy > 0)
        if direction == Direction.DOWN:
            return float(dy > 0), float(dx > 0), float(dx < 0)
        return float(dx < 0), float(dy > 0), float(dy < 0)

    def _spatial_features(self, head: Position, direction: Direction) -> list[float]:
        passable = self._passable_cells()
        reachable_ratio = self.reachable_ratio()

        head_pos = (head.x, head.y)
        food_pos = (self.food.x, self.food.y)
        bfs_dist = self._bfs_distance(head_pos, food_pos, passable)
        max_dist = self.width * self.height
        bfs_food_norm = 1.0 if bfs_dist is None else bfs_dist / max_dist

        tail = self.snake[-1]
        tail_ahead, tail_left, tail_right = self._relative_direction_flags(
            tail.x - head.x, tail.y - head.y, direction
        )

        free_neighbors = sum(
            1
            for dx, dy in self.CARDINAL_DELTAS
            if not self._is_collision(
                Position(head.x + dx, head.y + dy),
                will_grow=self._will_grow_at(Position(head.x + dx, head.y + dy)),
            )
        )

        return [
            reachable_ratio,
            bfs_food_norm,
            tail_ahead,
            tail_left,
            tail_right,
            free_neighbors / 4.0,
        ]

    def step(self, action: int) -> tuple[float, bool]:
        if self.game_over:
            return 0.0, True

        self.direction = self._get_new_direction(self.direction, action)
        head = self.snake[0]
        new_head = self._next_position(head, self.direction)
        will_grow = self._will_grow_at(new_head)

        if self._is_collision(new_head, will_grow=will_grow):
            self.game_over = True
            if self.score > self.high_score:
                self.high_score = self.score
                self._save_high_score()
            return self.REWARD_DEATH, True

        self.snake.insert(0, new_head)
        reward = self.REWARD_STEP
        ate_food = False

        if new_head.x == self.food.x and new_head.y == self.food.y:
            self.score += 1
            if self.score > self.high_score:
                self.high_score = self.score
                self._save_high_score()
            reward = self.REWARD_FOOD
            ate_food = True
            if self.is_board_full():
                self.game_over = True
                self.won = True
                reward = self.win_reward()
                self._ate_food = ate_food
                self.steps += 1
                return reward, True
            self.food = self._generate_food()
        else:
            self.snake.pop()

        self.steps += 1
        self._ate_food = ate_food
        return reward, False

    def rasterize_board(self, pad_width: int | None = None, pad_height: int | None = None) -> list[list[list[float]]]:
        """返回 (C, H, W) 栅格观测：通道依次为蛇头、蛇身、食物。"""
        pad_w = pad_width or self.width
        pad_h = pad_height or self.height
        channels = 3
        grid = [[[0.0 for _ in range(pad_w)] for _ in range(pad_h)] for _ in range(channels)]

        for segment in self.snake[1:]:
            if segment.x < pad_w and segment.y < pad_h:
                grid[1][segment.y][segment.x] = 1.0

        head = self.snake[0]
        if head.x < pad_w and head.y < pad_h:
            grid[0][head.y][head.x] = 1.0

        if self.food.x < pad_w and self.food.y < pad_h:
            grid[2][self.food.y][self.food.x] = 1.0

        return grid

    def extract_features(self) -> list[float]:
        head = self.snake[0]
        direction = self.direction

        ahead_pos = self._next_position(head, direction)
        right_dir = Direction((direction + 1) % 4)
        left_dir = Direction((direction + 3) % 4)
        right_pos = self._next_position(head, right_dir)
        left_pos = self._next_position(head, left_dir)

        danger_ahead = float(
            self._is_collision(ahead_pos, will_grow=self._will_grow_at(ahead_pos))
        )
        danger_right = float(
            self._is_collision(right_pos, will_grow=self._will_grow_at(right_pos))
        )
        danger_left = float(
            self._is_collision(left_pos, will_grow=self._will_grow_at(left_pos))
        )

        food_ahead, food_left, food_right = self._relative_direction_flags(
            self.food.x - head.x, self.food.y - head.y, direction
        )

        return [
            danger_ahead,
            danger_left,
            danger_right,
            food_ahead,
            food_left,
            food_right,
            float(direction == Direction.UP),
            float(direction == Direction.RIGHT),
            float(direction == Direction.DOWN),
            float(direction == Direction.LEFT),
            *self._ray_features(head),
            *self._spatial_features(head, direction),
        ]

    def _load_high_score(self) -> int:
        try:
            return int(self.HIGH_SCORE_FILE.read_text().strip())
        except (OSError, ValueError):
            return 0

    def _save_high_score(self) -> None:
        try:
            self.HIGH_SCORE_FILE.write_text(str(self.high_score))
        except OSError:
            pass

    def _init_fonts(self) -> None:
        pygame = _get_pygame()
        self._fonts_available = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if not pygame.font.get_init():
                    pygame.font.init()
                self._font_title = pygame.font.SysFont(
                    "PingFang SC,Microsoft YaHei,Arial", 18, bold=True
                )
                self._font_label = pygame.font.SysFont(
                    "PingFang SC,Microsoft YaHei,Arial", 13
                )
                self._font_value = pygame.font.SysFont("Menlo,Consolas,Arial", 20, bold=True)
                self._font_overlay = pygame.font.SysFont(
                    "PingFang SC,Microsoft YaHei,Arial", 22, bold=True
                )
                self._font_title.render("A", True, (255, 255, 255))
            self._fonts_available = True
        except (NotImplementedError, ImportError, OSError, AttributeError):
            self._font_title = None
            self._font_label = None
            self._font_value = None
            self._font_overlay = None

    def _font_for_style(self, style: str):
        return {
            "title": self._font_title,
            "label": self._font_label,
            "value": self._font_value,
            "overlay": self._font_overlay,
        }[style]

    def _ui(self, key: str) -> str:
        catalog = {
            "title": ("贪吃蛇", "Snake"),
            "score": ("得分", "Score"),
            "best": ("最高", "Best"),
            "length": ("长度", "Len"),
            "steps": ("步数", "Steps"),
            "session_best": ("会话最佳", "Session"),
            "game_over": ("游戏结束", "Game Over"),
            "final_score": ("本局得分", "Score"),
            "high_score": ("历史最高", "Best"),
            "restart": ("按 R 重新开始", "Press R to restart"),
        }
        zh, en = catalog[key]
        return zh if self._fonts_available else en

    def _bitmap_char(self, char: str) -> str:
        if char in _BITMAP_FONT:
            return char
        upper = char.upper()
        if upper in _BITMAP_FONT:
            return upper
        return "?"

    def _text_height(self, style: str) -> int:
        if self._fonts_available:
            return self._font_for_style(style).get_height()
        return 7 * _BITMAP_STYLE_SCALE[style]

    def _text_width(self, text: str, style: str) -> int:
        if self._fonts_available:
            return self._font_for_style(style).size(text)[0]
        scale = _BITMAP_STYLE_SCALE[style]
        return len(text) * (6 * scale)

    def _draw_text(
        self,
        text: str,
        *,
        x: int | None = None,
        y: int,
        color: tuple[int, int, int],
        style: str = "label",
        center_x: int | None = None,
    ) -> None:
        if center_x is not None:
            x = center_x - self._text_width(text, style) // 2

        if x is None:
            x = 0

        if self._fonts_available:
            surface = self._font_for_style(style).render(text, True, color)
            self.surface.blit(surface, (x, y))
            return

        pygame = _get_pygame()
        scale = _BITMAP_STYLE_SCALE[style]
        cursor_x = x
        for char in text:
            glyph_key = self._bitmap_char(char)
            glyph = _BITMAP_FONT[glyph_key]
            for row_index, row in enumerate(glyph):
                for col_index, pixel in enumerate(row):
                    if pixel == "#":
                        pygame.draw.rect(
                            self.surface,
                            color,
                            pygame.Rect(
                                cursor_x + col_index * scale,
                                y + row_index * scale,
                                scale,
                                scale,
                            ),
                        )
            cursor_x += 6 * scale

    def _board_origin_y(self) -> int:
        return self.HUD_HEIGHT

    def _cell_center(self, pos: Position, origin_y: int) -> tuple[int, int]:
        cell = self.CELL_SIZE
        return (
            int((pos.x + 0.5) * cell),
            int(origin_y + (pos.y + 0.5) * cell),
        )

    def _draw_hud(self) -> None:
        pygame = _get_pygame()
        hud_rect = pygame.Rect(0, 0, self.surface_width_px, self.HUD_HEIGHT)
        pygame.draw.rect(self.surface, self.COLOR_HUD_BG, hud_rect)
        pygame.draw.line(
            self.surface,
            self.COLOR_HUD_BORDER,
            (0, self.HUD_HEIGHT - 1),
            (self.surface_width_px, self.HUD_HEIGHT - 1),
        )

        self._draw_text(
            self._ui("title"),
            x=12,
            y=8,
            color=self.COLOR_HUD_ACCENT,
            style="title",
        )

        stats = [
            (self._ui("score"), str(self.score)),
            (self._ui("best"), str(self.high_score)),
            (self._ui("length"), str(len(self.snake))),
            (self._ui("steps"), str(self.steps)),
        ]
        if self.session_scores:
            session_best = max([*self.session_scores, self.score])
            stats.append((self._ui("session_best"), str(session_best)))

        stat_width = max(88, (self.surface_width_px - 120) // len(stats))
        start_x = max(96, self.surface_width_px - stat_width * len(stats) - 8)

        for index, (label, value) in enumerate(stats):
            x = start_x + index * stat_width
            self._draw_text(label, x=x, y=10, color=self.COLOR_HUD_LABEL, style="label")
            self._draw_text(value, x=x, y=26, color=self.COLOR_HUD_TEXT, style="value")

    def _draw_board_background(self, origin_y: int) -> None:
        pygame = _get_pygame()
        cell = self.CELL_SIZE

        for y in range(self.height):
            for x in range(self.width):
                color = self.COLOR_BG if (x + y) % 2 == 0 else self.COLOR_BG_ALT
                pygame.draw.rect(
                    self.surface,
                    color,
                    pygame.Rect(x * cell, origin_y + y * cell, cell, cell),
                )

        for i in range(self.width + 1):
            pygame.draw.line(
                self.surface,
                self.COLOR_GRID,
                (i * cell, origin_y),
                (i * cell, origin_y + self.board_height_px),
                1,
            )
        for i in range(self.height + 1):
            pygame.draw.line(
                self.surface,
                self.COLOR_GRID,
                (0, origin_y + i * cell),
                (self.board_width_px, origin_y + i * cell),
                1,
            )

    def _draw_food(self, origin_y: int) -> None:
        pygame = _get_pygame()
        cell = self.CELL_SIZE
        center = self._cell_center(self.food, origin_y)
        pulse = 0.82 + 0.18 * math.sin(self.steps * 0.35)
        glow_radius = int(cell * 0.48 * pulse)
        food_radius = int(cell * 0.32 * pulse)

        glow = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*self.COLOR_FOOD_GLOW, 70), (glow_radius, glow_radius), glow_radius)
        self.surface.blit(glow, (center[0] - glow_radius, center[1] - glow_radius))

        pygame.draw.circle(self.surface, self.COLOR_FOOD, center, food_radius)
        highlight = (center[0] - food_radius // 3, center[1] - food_radius // 3)
        pygame.draw.circle(
            self.surface,
            (255, 180, 170),
            highlight,
            max(2, food_radius // 4),
        )

    def _draw_snake_segment(
        self,
        segment: Position,
        *,
        index: int,
        total: int,
        origin_y: int,
    ) -> None:
        pygame = _get_pygame()
        cell = self.CELL_SIZE
        padding = 2
        rect = pygame.Rect(
            segment.x * cell + padding,
            origin_y + segment.y * cell + padding,
            cell - padding * 2,
            cell - padding * 2,
        )
        radius = max(3, cell // 4)

        if index == 0:
            color = self.COLOR_SNAKE_HEAD
        else:
            blend = index / max(total - 1, 1)
            color = tuple(
                int(self.COLOR_SNAKE_HEAD[i] * (1 - blend * 0.35) + self.COLOR_SNAKE_BODY[i] * (blend * 0.35))
                for i in range(3)
            )

        pygame.draw.rect(self.surface, self.COLOR_SNAKE_OUTLINE, rect, border_radius=radius)
        inner = rect.inflate(-2, -2)
        pygame.draw.rect(self.surface, color, inner, border_radius=max(2, radius - 1))

        if index == 0:
            self._draw_snake_eyes(inner)

    def _draw_snake_eyes(self, head_rect: "pygame_types.Rect") -> None:
        pygame = _get_pygame()
        cx = head_rect.centerx
        cy = head_rect.centery
        eye_r = max(2, self.CELL_SIZE // 7)
        offset = self.CELL_SIZE // 5

        if self.direction == Direction.UP:
            positions = [(cx - offset, cy + offset // 2), (cx + offset, cy + offset // 2)]
        elif self.direction == Direction.DOWN:
            positions = [(cx - offset, cy - offset // 2), (cx + offset, cy - offset // 2)]
        elif self.direction == Direction.LEFT:
            positions = [(cx + offset // 2, cy - offset), (cx + offset // 2, cy + offset)]
        else:
            positions = [(cx - offset // 2, cy - offset), (cx - offset // 2, cy + offset)]

        for eye_pos in positions:
            pygame.draw.circle(self.surface, (240, 250, 245), eye_pos, eye_r)
            pygame.draw.circle(self.surface, (20, 30, 25), eye_pos, max(1, eye_r // 2))

    def _draw_game_over_overlay(self, origin_y: int) -> None:
        pygame = _get_pygame()
        overlay = pygame.Surface((self.board_width_px, self.board_height_px), pygame.SRCALPHA)
        overlay.fill((8, 12, 20, 190))
        self.surface.blit(overlay, (0, origin_y))

        lines: list[tuple[str, tuple[int, int, int], str]] = [
            (self._ui("game_over"), self.COLOR_HUD_ACCENT, "overlay"),
            (
                f"{self._ui('final_score')}  {self.score}",
                self.COLOR_HUD_TEXT,
                "overlay",
            ),
            (
                f"{self._ui('high_score')}  {self.high_score}",
                self.COLOR_HUD_TEXT,
                "label",
            ),
        ]

        if self.session_scores:
            session_best = max(self.session_scores)
            session_avg = sum(self.session_scores) / len(self.session_scores)
            if self._fonts_available:
                session_line = (
                    f"本次会话  {len(self.session_scores)} 局 · 最佳 {session_best} · 均分 {session_avg:.1f}"
                )
            else:
                session_line = (
                    f"Session {len(self.session_scores)} games - Best {session_best} - Avg {session_avg:.1f}"
                )
            lines.append((session_line, self.COLOR_HUD_LABEL, "label"))

        lines.append((self._ui("restart"), self.COLOR_HUD_LABEL, "label"))

        total_height = sum(self._text_height(style) + 8 for _, _, style in lines) - 8
        current_y = origin_y + (self.board_height_px - total_height) // 2

        for text, color, style in lines:
            line_height = self._text_height(style)
            self._draw_text(
                text,
                y=current_y,
                color=color,
                style=style,
                center_x=self.board_width_px // 2,
            )
            current_y += line_height + 8

    def draw(self) -> None:
        if self.surface is None:
            return

        pygame = _get_pygame()
        origin_y = self._board_origin_y()

        self.surface.fill(self.COLOR_HUD_BG)
        self._draw_hud()
        self._draw_board_background(origin_y)
        self._draw_food(origin_y)

        total_segments = len(self.snake)
        for index, segment in enumerate(self.snake):
            self._draw_snake_segment(
                segment,
                index=index,
                total=total_segments,
                origin_y=origin_y,
            )

        if self.game_over:
            self._draw_game_over_overlay(origin_y)

        if self.render_mode == "human" and self.screen is not None:
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(self.FPS)

    def close(self) -> None:
        if _pygame is not None and _pygame.get_init():
            _pygame.quit()


def play_manual() -> None:
    import runtime  # noqa: F401

    pygame = _get_pygame()
    game = SnakeGame(render_mode="human")
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r and game.game_over:
                    game.session_scores.append(game.score)
                    game.reset()
                elif not game.game_over:
                    key_to_direction = {
                        pygame.K_UP: Direction.UP,
                        pygame.K_w: Direction.UP,
                        pygame.K_RIGHT: Direction.RIGHT,
                        pygame.K_d: Direction.RIGHT,
                        pygame.K_DOWN: Direction.DOWN,
                        pygame.K_s: Direction.DOWN,
                        pygame.K_LEFT: Direction.LEFT,
                        pygame.K_a: Direction.LEFT,
                    }
                    if event.key in key_to_direction:
                        new_direction = key_to_direction[event.key]
                        opposite = (game.direction + 2) % 4
                        if new_direction != opposite:
                            game.direction = new_direction

        if not game.game_over:
            game.step(0)

        game.draw()

    game.close()
    sys.exit(0)


if __name__ == "__main__":
    play_manual()
