from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Coord:
    x: int
    y: int

    def __post_init__(self):
        if not isinstance(self.x, int) or not isinstance(self.y, int):
            x = int(self.x)
            y = int(self.y)
            object.__setattr__(self, "x", x)
            object.__setattr__(self, "y", y)

    def __str__(self):
        return f"Coord(x={self.x}, y={self.y})"

    def __repr__(self):
        return self.__str__()
