# Daylight Factor Estimation Server
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
from dataclasses import dataclass, field
import numpy as np

from .image_transformer import ImageTransformer
from .external import EXTERNAL_KEYS
from .extended_enum import ExtendedEnum


@dataclass(frozen=True)
class Stat:
    """
    Stat over building performance estimation, result in values.
    """

    value: float = 0
    name: str = "Base"
    round: float = 5

    @property
    def out(self) -> dict[str, float]:
        return {self.name: self.value}

    @classmethod
    def _get_area(cls, matrix: np.array) -> np.array:
        return matrix[ImageTransformer.get_mask(matrix)]

    @classmethod
    def build(cls, matrix: np.ndarray) -> Stat:
        value = cls.calculate(matrix)
        return cls(round(value, cls.round))

    @classmethod
    def calculate(cls, matrix: np.ndarray) -> float:
        return 0


@dataclass(frozen=True)
class AvgStat(Stat):
    name: str = "average_value"

    @classmethod
    def calculate(cls, matrix) -> float:
        """
        Calculates the average of building performance values in a given area.
        """
        return np.mean(cls._get_area(matrix))


@dataclass(frozen=True)
class Gt1Stat(Stat):
    name: str = "ratio_gt1"

    @classmethod
    def _get_area(cls, matrix: np.array) -> np.array:
        return ImageTransformer.get_mask(matrix)

    @classmethod
    def calculate(cls, matrix) -> float:
        """
        Calculates the GT1 ratio of building performance values in a given area.
        """
        _area = cls._get_area(matrix)

        return np.sum(_area > 1.0) / np.sum(_area)


class STATS(ExtendedEnum):
    AVG = AvgStat
    GT1 = Gt1Stat


@dataclass(frozen=True)
class StatsPack:
    content: dict[STATS, Stat] = field(default_factory=dict)

    @property
    def out(self) -> dict[str, dict[str, float]]:
        return {
            EXTERNAL_KEYS.METRICS.value: {
                stat.name: stat.value for stat in self.content.values()
            }
        }

    @classmethod
    def build(cls, matrix: np.array) -> StatsPack:
        return StatsPack({s: s.value.build(matrix) for s in STATS.get_members()})
