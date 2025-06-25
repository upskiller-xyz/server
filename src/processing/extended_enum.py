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
from enum import Enum


class ExtendedEnum(Enum):
    """
    Extension of the basic enumerator.
    """

    @classmethod
    def by_value(cls, val):
        res = [x.name for x in cls.__members__.values() if x.value == val]
        if len(res) > 0:
            return cls[res[0]]

    @classmethod
    def from_name(cls, val):
        res = [
            x.name for x in cls.__members__.values() if x.name.lower() == val.lower()
        ]
        if len(res) > 0:
            return cls[res[0]]

    @classmethod
    def get_members(cls):
        return [cls[x] for x in cls.__members__]

    @classmethod
    def get_values(cls):
        return [x.value for x in cls.get_members()]
