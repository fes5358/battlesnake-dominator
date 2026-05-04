from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class Point(BaseModel):
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False


class Food(BaseModel):
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, Food):
            return self.x == other.x and self.y == other.y
        return False


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

    @property
    def dx(self) -> int:
        return {Direction.UP: 0, Direction.DOWN: 0, Direction.LEFT: -1, Direction.RIGHT: 1}[self]

    @property
    def dy(self) -> int:
        return {Direction.UP: 1, Direction.DOWN: -1, Direction.LEFT: 0, Direction.RIGHT: 0}[self]

    @classmethod
    def from_board_delta(cls, delta: tuple[int, int]) -> Direction:
        dx, dy = delta
        if dx == 0 and dy == 1:
            return cls.UP
        if dx == 0 and dy == -1:
            return cls.DOWN
        if dx == -1 and dy == 0:
            return cls.LEFT
        if dx == 1 and dy == 0:
            return cls.RIGHT
        raise ValueError(f"Invalid delta: {delta}")


class MoveAction(BaseModel):
    move: Direction

    def model_dump(self, **kwargs):
        return {"move": self.move.value}


class Snake(BaseModel):
    id: str
    name: str
    health: int
    body: List[Optional[Point]]
    head: Optional[Point] = None
    length: Optional[int] = None

    model_config = {"extra": "allow"}


class RulesetSettings(BaseModel):
    viewRadius: Optional[int] = None

    model_config = {"extra": "allow"}


class Ruleset(BaseModel):
    name: str
    version: Optional[str] = None
    settings: RulesetSettings = RulesetSettings()

    model_config = {"extra": "allow"}


class Game(BaseModel):
    id: str
    ruleset: Ruleset

    model_config = {"extra": "allow"}


class Board(BaseModel):
    height: int
    width: int
    food: List[Food] = []
    snakes: List[Snake] = []

    model_config = {"extra": "allow"}


class GameState(BaseModel):
    game: Game
    turn: int
    board: Board
    you: Snake

    model_config = {"extra": "allow"}


class BaseAgent(ABC):
    @abstractmethod
    def get_name(self) -> str: ...

    @abstractmethod
    def get_color(self) -> str: ...

    def get_author(self) -> Optional[str]:
        return None

    def get_head(self) -> Optional[str]:
        return None

    def get_tail(self) -> Optional[str]:
        return None

    @abstractmethod
    def start(self, game_state: GameState) -> None: ...

    @abstractmethod
    def move(self, game_state: GameState) -> MoveAction: ...

    @abstractmethod
    def end(self, game_state: GameState) -> None: ...
