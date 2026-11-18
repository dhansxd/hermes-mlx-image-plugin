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

import base64
import logging
import os
import re
import stat
import subprocess
from datetime import datetime
from urllib.parse import urlsplit
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agent.file_safety import raise_if_read_blocked
from agent.image_gen_provider import (
    ImageGenProvider, error_response, resolve_aspect_ratio, save_b64_image,
    save_url_image, success_response,
)
from agent.secret_scope import get_secret
from hermes_cli.plugins import PluginContext
from hermes_constants import get_hermes_home, get_real_home

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

DEFAULT_MODELS_DIR = "~/.local/share/dyra-ai/models/image"
DEFAULT_WRAPPER = "~/.local/bin/dyra-image"
GENERATION_TIMEOUT_SECONDS = 1800
MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class _RequestError(ValueError):
    """A safe diagnostic, independent of paths, prompts and runtime output."""

    def __init__(self, message: str, error_type: str):
        super().__init__(message)
        self.error_type = error_type


def _load_settings() -> Dict[str, Any]:
    from hermes_cli.config import load_config

    cfg = load_config()
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise _RequestError("Hermes config must be a mapping.", "invalid_config")
    section = cfg.get("image_gen", {})
    if not isinstance(section, dict):
        raise _RequestError("image_gen config must be a mapping.", "invalid_config")
    settings = section.get("mlx", {})
    if not isinstance(settings, dict):
        raise _RequestError("image_gen.mlx config must be a mapping.", "invalid_config")
    for name in ("model", "models_dir", "wrapper"):
        if name in settings and (not isinstance(settings[name], str) or not settings[name].strip()):
            raise _RequestError("MLX model and path settings must be nonempty strings.", "invalid_config")
    return settings


def _setting(settings: Dict[str, Any], name: str, env_name: str, default: str) -> str:
    raw = settings.get(name)
    if raw is None:
        raw = get_secret(env_name)
    if raw is None:
        raw = default
    if not isinstance(raw, str) or not raw.strip():
        raise _RequestError("MLX model and path settings must be nonempty strings.", "invalid_config")
    return raw.strip()


def _configured_path(settings: Dict[str, Any], name: str, env_name: str, default: str) -> str:
    raw = _setting(settings, name, env_name, default)
    if raw == "~" or raw.startswith("~/"):
        raw = str(Path(get_real_home()) / raw[2:]) if raw != "~" else str(get_real_home())
    path = Path(raw)
    if not path.is_absolute():
        raise _RequestError("MLX paths must be absolute or start with ~/.", "invalid_config")
    return str(path)


def _models_dir(settings: Dict[str, Any]) -> str:
    return _configured_path(settings, "models_dir", "MLX_MODELS_DIR", DEFAULT_MODELS_DIR)


def _wrapper_path(settings: Dict[str, Any]) -> str:
    return _configured_path(settings, "wrapper", "MLX_WRAPPER", DEFAULT_WRAPPER)


def _canonical_key(value: Optional[str]) -> Optional[str]:
    return _ALIASES.get(value.strip().lower()) if isinstance(value, str) and value.strip() else None


def _default_key(settings: Dict[str, Any]) -> str:
    raw = settings.get("model")
    if raw is None:
        raw = get_secret("MLX_IMAGE_MODEL")
    if raw is not None:
        key = _canonical_key(raw)
        if key is None:
            raise _RequestError("The configured MLX model is not supported.", "unsupported_model")
        return key
    root = _models_dir(settings)
    for key, entry in MODEL_TABLE.items():
        if Path(root, entry["local_subdir"]).is_dir():
            return key
    return next(iter(MODEL_TABLE))


def _resolve_entry(kwarg_model: Any, settings: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    if kwarg_model is not None:
        if not isinstance(kwarg_model, str):
            raise _RequestError("The requested MLX model must be a string.", "invalid_request")
        if kwarg_model.strip():
            key = _canonical_key(kwarg_model)
            if key is None:
                raise _RequestError("The requested MLX model is not supported.", "unsupported_model")
            return key, MODEL_TABLE[key]
    key = _default_key(settings)
    return key, MODEL_TABLE[key]


def _resolution_request(prompt: str, kwargs: dict) -> tuple[str, int]:
    """Legacy pixel controls; intentionally not Hermes's native 1K/2K control."""
    raw = kwargs.get("resolution")
    if raw is None:
        raw = kwargs.get("size")
    clean_prompt = prompt
    marker = re.search(r"(?:^|\s)\[(?:resolution|size)\s*=\s*([^\]]*)\](?:\s|$)", prompt, re.I)
    natural = re.search(r"(?:resolution|size|resolusi)\s*[:=]?\s*(\d{3,4})\s*(?:px|p)?\b|\b(\d{3,4})\s*[x×]\s*\d{3,4}\b", prompt, re.I)
    if marker:
        raw = marker.group(1).strip()
        clean_prompt = (prompt[:marker.start()] + " " + prompt[marker.end():]).strip()
    elif raw is None and natural:
        raw = natural.group(1) or natural.group(2)
    if raw is None:
        side = 256
    elif isinstance(raw, int) and not isinstance(raw, bool):
        side = raw
    elif isinstance(raw, str) and re.fullmatch(r"[0-9]{1,6}", raw.strip()):
        side = int(raw.strip())
    else:
        raise _RequestError("MLX resolution must be an integer pixel size; native 1K/2K controls are unsupported.", "invalid_resolution")
    side = max(256, min(1536, side))
    return clean_prompt, (side // 16) * 16


def _dimensions(aspect_ratio: str, side: int) -> tuple[int, int]:
    if aspect_ratio == "landscape":
        return side, max(144, round(side * 9 / 16 / 16) * 16)
    if aspect_ratio == "portrait":
        return max(144, round(side * 9 / 16 / 16) * 16), side
    return side, side


def _steps_for(side: int) -> int:
    return 4 if side <= 256 else 6 if side <= 512 else 8 if side <= 1024 else 12


def _image_header(path: Path, *, output: bool = False) -> bool:
    """Check a bounded header, not full image decoding or image quality."""
    info = path.lstat() if output else path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size <= len(PNG_SIGNATURE):
        return False
    if not output and info.st_size > MAX_INPUT_IMAGE_BYTES:
        return False
    with path.open("rb") as stream:
        header = stream.read(12)
    if output:
        return header.startswith(PNG_SIGNATURE)
    return (
        header.startswith(PNG_SIGNATURE) or header.startswith(b"\xff\xd8\xff")
        or header.startswith((b"GIF87a", b"GIF89a"))
        or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
    )


def _remove_attempt_file(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)  # unlinks a symlink itself, never its target
    except OSError:
        logger.warning("Could not remove an MLX attempt file.")


def _remove_input_attempt(prefix: str) -> None:
    """Native media helpers can fail after opening a file but before returning it."""
    try:
        cache = Path(get_hermes_home()) / "cache" / "generated" / "images"
        for path in cache.glob(prefix + "_*"):
            _remove_attempt_file(path)
    except OSError:
        logger.warning("Could not remove an MLX input attempt file.")


def _source_image(value: Any) -> tuple[Path, bool]:
    """Materialize native URLs/data URLs through Hermes's public media helpers."""
    if not isinstance(value, str) or not value.strip():
        raise _RequestError("The source image must be a URL, data URL or absolute file path.", "invalid_input_image")
    value = value.strip()
    scheme = urlsplit(value).scheme.lower()
    prefix = "mlx_input_" + uuid4().hex
    if scheme in ("http", "https"):
        try:
            path = Path(save_url_image(value, prefix=prefix, max_bytes=MAX_INPUT_IMAGE_BYTES))
        except Exception:
            _remove_input_attempt(prefix)
            raise _RequestError("Could not load the source image through Hermes URL safety.", "input_image_unavailable") from None
        owned = True
    elif scheme == "data":
        match = re.fullmatch(r"data:(image/(?:png|jpeg|jpg|webp|gif));base64,(.*)", value, re.I | re.S)
        if not match or len(match.group(2)) > 4 * ((MAX_INPUT_IMAGE_BYTES + 2) // 3):
            raise _RequestError("The image data URL is unsupported or too large.", "invalid_input_image")
        try:
            raw = base64.b64decode(match.group(2), validate=True)
            if not raw or len(raw) > MAX_INPUT_IMAGE_BYTES:
                raise ValueError
            extension = {"image/jpeg": "jpg", "image/jpg": "jpg"}.get(match.group(1).lower(), match.group(1).lower().split("/")[1])
            path = Path(save_b64_image(match.group(2), prefix=prefix, extension=extension))
        except Exception:
            _remove_input_attempt(prefix)
            raise _RequestError("The image data URL is invalid or could not be cached.", "invalid_input_image") from None
        owned = True
    else:
        path = Path(value).expanduser()
        if scheme or not path.is_absolute():
            raise _RequestError("The source image must be an HTTP(S) URL, image data URL or absolute path.", "invalid_input_image")
        try:
            raise_if_read_blocked(str(path))
            path = path.resolve()
        except (OSError, ValueError, RuntimeError):
            raise _RequestError("Hermes denied access to the source image.", "input_image_unavailable") from None
        owned = False
    try:
        if not _image_header(path):
            raise ValueError
    except (OSError, ValueError):
        if owned:
            _remove_attempt_file(path)
        raise _RequestError("The source image is missing, too large or not a supported image file.", "input_image_not_found") from None
    return path, owned


def _child_environment() -> Dict[str, str]:
    """Pass runtime essentials, not another profile's credentials or Python ABI."""
    names = {"PATH", "USER", "LOGNAME", "TMPDIR", "TMP", "TEMP", "LANG", "TZ", "SYSTEMROOT", "WINDIR"}
    prefixes = ("LC_", "OMP_", "OPENBLAS_", "MKL_", "VECLIB_", "METAL_")
    env = {name: value for name, value in os.environ.items() if name in names or name.startswith(prefixes)}
    env["HOME"] = str(get_real_home())
    timeout = get_secret("DYRA_LOCAL_AI_LOCK_TIMEOUT")
    if timeout is not None:
        env["DYRA_LOCAL_AI_LOCK_TIMEOUT"] = timeout
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


class MLXImageGenProvider(ImageGenProvider):
    """Local image generation on Apple Silicon using the external MLX runtime."""

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
        try:
            settings = _load_settings()
            wrapper = _wrapper_path(settings)
            root = _models_dir(settings)
            # The native picker may override a stale provider-scoped default.
            return (
                os.path.isfile(wrapper) and os.access(wrapper, os.X_OK)
                and any(Path(root, entry["local_subdir"]).is_dir() for entry in MODEL_TABLE.values())
            )
        except Exception:
            return False

    def capabilities(self) -> dict:
        return {"modalities": ["text", "image"], "max_reference_images": 0, "supports_upscale": False}

    def list_models(self) -> List[Dict[str, Any]]:
        root = _models_dir(_load_settings())
        return [{
            "id": key, "display": entry.get("display", key),
            "strengths": entry.get("strengths", "local MLX via mflux"),
            "local_available": Path(root, entry["local_subdir"]).is_dir(),
        } for key, entry in MODEL_TABLE.items()]

    def default_model(self) -> Optional[str]:
        return _default_key(_load_settings())

    def get_setup_schema(self) -> Dict[str, Any]:
        return {"name": "MLX (local)", "tag": "Local image generation on Apple Silicon via mflux", "env_vars": []}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = "landscape",
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        """Generate/edit through the native provider contract; unknown kwargs are ignored."""
        model_label = ""
        output_path = None
        input_path = None
        keep_output = False
        aspect_ratio = resolve_aspect_ratio(aspect_ratio)
        try:
            if not isinstance(prompt, str) or not prompt.strip():
                raise _RequestError("An image prompt is required.", "invalid_request")
            if reference_image_urls:
                raise _RequestError("This MLX provider accepts one primary image, not additional references.", "modality_unsupported")
            settings = _load_settings()
            key, entry = _resolve_entry(kwargs.get("model"), settings)
            model_label = entry["hf_id"]
            wrapper = _wrapper_path(settings)
            if not (os.path.isfile(wrapper) and os.access(wrapper, os.X_OK)):
                raise _RequestError("Shared mflux wrapper is unavailable.", "mflux_not_installed")
            # The table is trusted source, never a user-supplied HF id or path.
            local_model = Path(_models_dir(settings), entry["local_subdir"]).resolve()
            if not local_model.is_dir():
                raise _RequestError("The configured local model directory does not exist.", "model_not_found")
            prompt, side = _resolution_request(prompt, kwargs)
            if not prompt.strip():
                raise _RequestError("An image prompt is required after removing the size marker.", "invalid_request")
            width, height = _dimensions(aspect_ratio, side)
            steps = _steps_for(side)
            sources = []
            if image_url is not None and image_url != "":
                source, owned = _source_image(image_url)
                sources = ["--image-paths", str(source)]
                input_path = source if owned else None
            image_cache_dir = Path(get_hermes_home()).resolve() / "cache" / "generated" / "images"
            image_cache_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = image_cache_dir / f"mlx_{timestamp}_{uuid4().hex}.png"
            command = [
                wrapper, "--model", str(local_model), "--base-model", entry["base_model"],
                "--quantize", entry["quantize"], "--steps", str(steps), "--prompt", prompt,
                "--output", str(output_path), "--width", str(width), "--height", str(height),
                *sources,
            ]
            logger.info("Starting local MLX generation for %s (%dx%d, %d steps).", key, width, height, steps)
            subprocess.run(
                command, env=_child_environment(), stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, check=True, timeout=GENERATION_TIMEOUT_SECONDS,
            )
            try:
                usable = _image_header(output_path, output=True)
            except OSError:
                usable = False
            if not usable:
                raise _RequestError("mflux finished without a usable PNG output image.", "file_not_found")
            result = success_response(
                image=str(output_path), model=model_label, prompt=prompt,
                aspect_ratio=aspect_ratio, provider=self.name,
                modality="image" if sources else "text",
            )
            keep_output = True
            return result
        except _RequestError as exc:
            error, error_type = str(exc), exc.error_type
        except subprocess.CalledProcessError as exc:
            error, error_type = f"mflux command failed with exit code {exc.returncode}.", "mflux_execution_failed"
        except subprocess.TimeoutExpired:
            error, error_type = "mflux command timed out.", "mflux_timeout"
        except Exception as exc:
            logger.error("MLX generation failed (%s).", type(exc).__name__)
            error, error_type = "MLX generation could not complete. Check the configured local runtime and paths.", "unexpected_error"
        finally:
            _remove_attempt_file(input_path)
            if not keep_output:
                _remove_attempt_file(output_path)
        logger.error("MLX generation failed (%s).", error_type)
        return error_response(
            error=error, error_type=error_type, provider=self.name, model=model_label,
            prompt=prompt if isinstance(prompt, str) else "", aspect_ratio=aspect_ratio,
        )


def register(ctx: PluginContext):
    """Hermes owns provider registration and release via the official ledger."""
    ctx.register_image_gen_provider(MLXImageGenProvider())
