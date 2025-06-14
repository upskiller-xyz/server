from __future__ import annotations
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)    



@dataclass
class PipelineConfig:


    value:int

    @classmethod
    def default(cls)->PipelineConfig:
        return PipelineConfig(0)


