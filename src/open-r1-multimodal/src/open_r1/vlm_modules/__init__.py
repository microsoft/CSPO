from .vlm_module import VLMBaseModule
from .qwen_module import Qwen2VLModule
from .internvl_module import InvernVLModule
from .teds_utils import compute_teds_score

# Compatibility aliases for entrypoints that still import historical names.

__all__ = [
    "VLMBaseModule",
    "Qwen2VLModule",
    "InvernVLModule",
    "compute_teds_score",
]
