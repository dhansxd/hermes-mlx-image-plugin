"""Hermes directory-plugin entry point; package implementation stays in src."""
from .src.hermes_mlx_image import MLXImageGenProvider, register

__all__ = ["MLXImageGenProvider", "register"]
