"""Local MLX image-generation backend for Hermes Agent.

Generic by design: provider name is ``mlx``, models are rows in ``MODEL_TABLE``
resolved against the single local cupboard (``~/.local/share/dyra-ai``).
Adding a model = one table row + alias lines. No plugin rename needed.

Config (``image_gen.mlx`` in config.yaml, non-secret):
    model       — model key (default: first table entry)
    models_dir  — local models root (default: ``~/.local/share/dyra-ai/models/image``)
    wrapper     — cupboard entrypoint (default: ``~/.local/bin/dyra-image``)

Env overrides (config wins when set): ``MLX_IMAGE_MODEL``, ``MLX_MODELS_DIR``,
``MLX_WRAPPER``. No secrets required.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import ImageGenProvider, error_response, success_response
from hermes_cli.plugins import PluginContext
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# --- Generic model table (1 entry for now; append rows to add models) ---
MODEL_TABLE: Dict[str, Dict[str, Any]] = {
    "flux2-klein-4b-4bit": {
        "hf_id": "mlx-community/FLUX.2-Klein-4B-4bit",
        "local_subdir": "flux2-klein-4b-4bit",
        "base_model": "flux2-klein-4b",
        "quantize": "4",
        "display": "FLUX.2 Klein 4B (local MLX)",
        "strengths": "local Apple Silicon via mflux",
    },
}

_ALIASES: Dict[str, str] = {
    "flux2-klein-4b-4bit": "flux2-klein-4b-4bit",
    "flux2-klein-4b": "flux2-klein-4b-4bit",
    "mlx-community/flux.2-klein-4b-4bit": "flux2-klein-4b-4bit",
}

DEFAULT_MODELS_DIR = os.path.expanduser("~/.local/share/dyra-ai/models/image")
DEFAULT_WRAPPER = os.path.expanduser("~/.local/bin/dyra-image")


def _load_settings() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
    except ImportError:  # tests / non-Hermes context: env + defaults only
        return {}
    cfg = load_config() or {}
    scoped = cfg.get("image_gen") if isinstance(cfg, dict) else None
    scoped = scoped.get("mlx") if isinstance(scoped, dict) else None
    return scoped if isinstance(scoped, dict) else {}


def _models_dir(settings: Dict[str, Any]) -> str:
    raw = settings.get("models_dir") or os.environ.get("MLX_MODELS_DIR") or DEFAULT_MODELS_DIR
    return os.path.expanduser(str(raw))


def _wrapper_path(settings: Dict[str, Any]) -> str:
    raw = settings.get("wrapper") or os.environ.get("MLX_WRAPPER") or DEFAULT_WRAPPER
    return os.path.expanduser(str(raw))


def _canonical_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return _ALIASES.get(str(value).strip().lower())


def _default_key(settings: Dict[str, Any]) -> str:
    for candidate in (
        _canonical_key(str(settings.get("model") or "").strip() or None),
        _canonical_key((os.environ.get("MLX_IMAGE_MODEL") or "").strip() or None),
    ):
        if candidate and candidate in MODEL_TABLE:
            return candidate
    # Prefer a model whose local dir already exists in the cupboard.
    root = _models_dir(settings)
    for key, entry in MODEL_TABLE.items():
        if Path(root, entry["local_subdir"]).is_dir():
            return key
    return next(iter(MODEL_TABLE))


def _resolve_entry(kwarg_model: Optional[str], settings: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    key = _canonical_key((kwarg_model or "").strip() or None) or _default_key(settings)
    return key, MODEL_TABLE[key]


def _resolution_request(prompt: str, kwargs: dict) -> tuple[str, int]:
    """Resolve per-request square side from tool kwargs or prompt marker.

    Supported range is 256..1536. Prompt marker exists for current Hermes
    schema, which exposes aspect ratio but not width/height yet.
    """
    raw = kwargs.get("resolution") or kwargs.get("size")
    clean_prompt = prompt
    marker = re.search(r"(?:^|\s)\[(?:resolution|size)\s*=\s*(\d{3,4})\](?:\s|$)", prompt, re.I)
    natural = re.search(r"(?:resolution|size|resolusi)\s*[:=]?\s*(\d{3,4})\s*(?:px|p)?\b|\b(\d{3,4})\s*[x×]\s*\d{3,4}\b", prompt, re.I)
    if marker:
        raw = marker.group(1)
        clean_prompt = (prompt[:marker.start()] + " " + prompt[marker.end():]).strip()
    elif raw is None and natural:
        raw = natural.group(1) or natural.group(2)
    try:
        side = int(raw) if raw is not None else 256
    except (TypeError, ValueError):
        side = 256
    side = max(256, min(1536, side))
    side = (side // 16) * 16
    return clean_prompt, max(256, side)


def _dimensions(aspect_ratio: str, side: int) -> tuple[int, int]:
    if aspect_ratio == "landscape":
        return side, max(144, round(side * 9 / 16 / 16) * 16)
    if aspect_ratio == "portrait":
        return max(144, round(side * 9 / 16 / 16) * 16), side
    return side, side


def _steps_for(side: int) -> int:
    # Quality schedule requested by Dani; 1536 uses 12 for high-resolution quality.
    return 4 if side <= 256 else 6 if side <= 512 else 8 if side <= 1024 else 12


class MLXImageGenProvider(ImageGenProvider):
    """Local image generation on Apple Silicon using MLX/mflux."""

    @property
    def name(self) -> str:
        return "mlx"

    @property
    def display_name(self) -> str:
        return "MLX (local)"

    @property
    def description(self) -> str:
        return "Local image generation on Apple Silicon using MLX/mflux."

    def is_available(self) -> bool:
        """Check if shared user-global image wrapper is installed."""
        wrapper = _wrapper_path(_load_settings())
        return os.path.isfile(wrapper) and os.access(wrapper, os.X_OK)

    def capabilities(self) -> dict:
        return {"modalities": ["text", "image"], "max_reference_images": 0}

    def list_models(self) -> List[Dict[str, Any]]:
        settings = _load_settings()
        root = _models_dir(settings)
        out = []
        for key, entry in MODEL_TABLE.items():
            out.append(
                {
                    "id": key,
                    "display": entry.get("display", key),
                    "strengths": entry.get("strengths", "local MLX via mflux"),
                    "local_available": Path(root, entry["local_subdir"]).is_dir(),
                }
            )
        return out

    def default_model(self) -> Optional[str]:
        return _default_key(_load_settings())

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "MLX (local)",
            "tag": "Local image generation on Apple Silicon via mflux",
            "env_vars": [],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = "landscape",
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        """Generate or edit an image using the local mflux wrapper."""
        settings = _load_settings()
        wrapper = _wrapper_path(settings)
        key, entry = _resolve_entry(str(kwargs.get("model") or "").strip() or None, settings)
        model_label = entry["hf_id"]

        if not (os.path.isfile(wrapper) and os.access(wrapper, os.X_OK)):
            return error_response(
                error="Shared mflux wrapper is unavailable.",
                error_type="mflux_not_installed",
                provider=self.name,
                model=model_label,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        prompt, side = _resolution_request(prompt, kwargs)
        width, height = _dimensions(aspect_ratio, side)
        steps = _steps_for(side)

        # --- Prepare directories and paths ---
        image_cache_dir = os.path.join(get_hermes_home(), "cache", "images", "mlx")
        os.makedirs(image_cache_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"mlx_{timestamp}.png"
        output_path = os.path.join(image_cache_dir, output_filename)

        # Base command from the generic table (NOT hardcoded per model).
        command = [
            wrapper,
            "--model", entry["hf_id"],
            "--base-model", entry["base_model"],
            "--quantize", entry["quantize"],
            "--steps", str(steps),
            "--prompt", prompt,
            "--output", output_path,
        ]

        # Per-request dimensions, clamped to 256..1536 and aligned to 16.
        command.extend(["--width", str(width), "--height", str(height)])

        # Handle Image-to-Image
        if image_url:
            if os.path.exists(image_url):
                command.extend(["--image-path", image_url])
                command.extend(["--image-strength", "0.65"])
            else:
                return error_response(
                    error=f"Input image does not exist: {image_url}",
                    error_type="input_image_not_found",
                    provider=self.name,
                    model=model_label,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                )

        logger.info(f"Running mflux command: {' '.join(shlex.quote(c) for c in command)}")

        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=None,
            )

            if os.path.exists(output_path):
                return success_response(
                    image=output_path,
                    model=model_label,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                    provider=self.name,
                    modality="image" if image_url else "text",
                )
            else:
                return error_response(
                    error=(
                        "mflux command ran but output image missing. "
                        f"STDOUT: {process.stdout} | STDERR: {process.stderr}"
                    ),
                    error_type="file_not_found",
                    provider=self.name,
                    model=model_label,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                )

        except subprocess.CalledProcessError as e:
            error_msg = f"mflux command failed with exit code {e.returncode}."
            logger.error(f"{error_msg} STDERR: {e.stderr} STDOUT: {e.stdout}")
            return error_response(
                error=f"{error_msg}\n---\nSTDERR:\n{e.stderr}\n---\nSTDOUT:\n{e.stdout}",
                error_type="mflux_execution_failed",
                provider=self.name,
                model=model_label,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )
        except subprocess.TimeoutExpired as e:
            error_msg = f"mflux command timed out after {e.timeout} seconds."
            logger.error(error_msg)
            return error_response(
                error=error_msg,
                error_type="mflux_timeout",
                provider=self.name,
                model=model_label,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )
        except Exception as e:
            logger.exception("An unexpected error occurred while running mflux.")
            return error_response(
                error=f"An unexpected error occurred: {str(e)}",
                error_type="unexpected_error",
                provider=self.name,
                model=model_label,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )


def register(ctx: PluginContext):
    """Plugin registration function called by Hermes at startup."""
    ctx.register_image_gen_provider(MLXImageGenProvider())
