# Daylight Factor Simulation Server
# Copyright (C) 2024 BIMTech Innovations AB (developed by the Upskiller group)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU GPL v3.0 along with this program.
# If not, see <https://www.gnu.org/licenses/>.

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
