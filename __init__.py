"""
ComfyUI 火山引擎视频生成节点

支持 Seedance 系列模型（2.0 / 1.5 / 1.0）
功能：文生视频、图生视频（多图+参考音频）
"""

from .nodes import (
    火山引擎文生视频,
    火山引擎图生视频,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = [
    "火山引擎文生视频",
    "火山引擎图生视频",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
