"""
ComfyUI VolcEngine Video/Image Generation Node

Supports Seedance series models (2.0 / 1.5 / 1.0)
Features: Text-to-video, Image-to-video (multi-image + reference audio)
"""

__version__ = "1.1.0"

from .nodes import (
    VolcEngineTextToVideo,
    VolcEngineImageToVideo,
    VolcEngineImageGeneration,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = [
    "__version__",
    "VolcEngineTextToVideo",
    "VolcEngineImageToVideo",
    "VolcEngineImageGeneration",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
