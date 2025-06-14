from __future__ import annotations
from dataclasses import dataclass
import json
import logging



logger = logging.getLogger(__name__)    

@dataclass
class PipelineInput:
    value: int

    @classmethod
    def default(cls)->PipelineInput:
        return PipelineInput(0)