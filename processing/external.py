from __future__ import annotations
from dataclasses import dataclass

from .extended_enum import ExtendedEnum


class EXTERNAL_KEYS(ExtendedEnum):
    METRICS = "metrics"
    CONTENT = "content"
    SUCCESS = "success"
