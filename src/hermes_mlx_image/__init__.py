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
import json
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
DEFAULT_PRESETS_FILE = "~/.local/share/dyra-ai/production/image/presets.json"
GENERATION_TIMEOUT_SECONDS = 1800
MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
ASPECT_RATIOS = {
    "landscape": "landscape", "16:9": "landscape",
    "portrait": "portrait", "9:16": "portrait",
    "square": "square", "1:1": "square",
}
GENERATION_FIELDS = {"aspect_ratio", "resolution", "steps"}


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


def _resolution_request(prompt: str, kwargs: dict, default: Optional[int] = 256) -> tuple[str, Optional[int]]:
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
        return clean_prompt, default
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


def _validated_options(values: Any, source: str, error_type: str = "invalid_config") -> dict[str, Any]:
    if not isinstance(values, dict) or values.keys() - GENERATION_FIELDS:
        raise _RequestError(f"{source} must contain only aspect_ratio, resolution, and steps.", error_type)
    options = dict(values)
    if "aspect_ratio" in options:
        raw = options["aspect_ratio"]
        ratio = ASPECT_RATIOS.get(raw.strip().lower()) if isinstance(raw, str) else None
        if ratio is None:
            raise _RequestError(f"{source} aspect_ratio must be landscape, portrait, or square.", error_type)
        options["aspect_ratio"] = ratio
    if "resolution" in options:
        side = options["resolution"]
        if type(side) is not int or not 256 <= side <= 1536 or side % 16:
            raise _RequestError(f"{source} resolution must be 256–1536 pixels in multiples of 16.", error_type)
    if "steps" in options:
        steps = options["steps"]
        if type(steps) is not int or not 1 <= steps <= 12:
            raise _RequestError(f"{source} steps must be an integer from 1 to 12.", error_type)
    return options


def _presets(settings: dict) -> dict:
    path = Path(_configured_path(settings, "presets_file", "MLX_IMAGE_PRESETS_FILE", DEFAULT_PRESETS_FILE))
    try:
        with path.open(encoding="utf-8") as stream:
            data = json.loads(stream.read(65537))
    except (OSError, ValueError) as exc:
        raise _RequestError("Cannot read the shared MLX presets file.", "invalid_config") from exc
    if not isinstance(data, dict):
        raise _RequestError("The shared MLX presets file must contain a mapping.", "invalid_config")
    return data


def _preset_options(presets: dict, name: Any, error_type: str) -> dict:
    if not isinstance(name, str) or name not in presets:
        available = ", ".join(sorted(presets))
        raise _RequestError(f"MLX preset is unavailable. Available presets: {available or 'none'}.", error_type)
    return _validated_options(presets[name], "MLX preset")


def _generation_options(settings: dict, preset: Any, requested: dict) -> dict:
    default_preset = settings.get("preset")
    presets = _presets(settings) if default_preset is not None or preset is not None else {}
    defaults = _preset_options(presets, default_preset, "invalid_config") if default_preset is not None else {}
    defaults.update(_validated_options(settings.get("defaults", {}), "MLX defaults"))
    overrides = _preset_options(presets, preset, "invalid_request") if preset is not None else {}
    overrides.update(requested)
    locked = _validated_options(settings.get("locked", {}), "MLX locks")
    for field, value in locked.items():
        if field in overrides and overrides[field] != value:
            raise _RequestError(f"MLX {field} is locked to {value}. Change the profile lock to use another value.",
                                "setting_locked")
    return {**defaults, **overrides, **locked}


def _mlx_selected() -> bool:
    from hermes_cli.config import load_config

    section = (load_config() or {}).get("image_gen")
    return isinstance(section, dict) and str(section.get("provider") or "").strip().lower() == "mlx"


def _apply_native_aspect_default(tool_name: str, args: dict, **_: Any) -> Optional[dict]:
    """Apply the profile's aspect before the native tool fills in its own default."""
    if tool_name != "image_generate" or args.get("aspect_ratio") is not None or not _mlx_selected():
        return None
    try:
        ratio = _generation_options(_load_settings(), None, {}).get("aspect_ratio")
    except _RequestError as exc:
        return {"action": "block", "message": str(exc)}
    if ratio is not None:
        return {"action": "modify", "args": {"aspect_ratio": ratio}}
    return None


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
        aspect_ratio: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        *,
        legacy_prompt_size: bool = True,
        **kwargs: Any,
    ) -> dict:
        """Generate/edit through the native provider contract; unknown kwargs are ignored."""
        model_label = ""
        output_path = None
        input_path = None
        keep_output = False
        requested_ratio = resolve_aspect_ratio(aspect_ratio) if aspect_ratio is not None else None
        aspect_ratio = requested_ratio or "landscape"
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
            requested_side = kwargs.get("resolution")
            if legacy_prompt_size:
                prompt, requested_side = _resolution_request(prompt, kwargs, default=None)
            if not prompt.strip():
                raise _RequestError("An image prompt is required after removing the size marker.", "invalid_request")
            requested = {"aspect_ratio": requested_ratio, "resolution": requested_side, "steps": kwargs.get("steps")}
            options = _generation_options(settings, kwargs.get("preset"), _validated_options(
                {name: value for name, value in requested.items() if value is not None},
                "MLX request", "invalid_request",
            ))
            aspect_ratio = options.get("aspect_ratio", "landscape")
            side = options.get("resolution", 256)
            width, height = _dimensions(aspect_ratio, side)
            steps = options.get("steps", _steps_for(side))
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
            result.update(width=width, height=height, steps=steps)
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


_TOOL_SCHEMA = {
    "name": "mlx_image_generate",
    "description": (
        "Generate a new image using local FLUX.2 with user-requested resolution, steps, or a named "
        "preset. For a standard image or editing an existing image use image_generate. Return the "
        "result's image path as MEDIA: to deliver it. Report the actual width, height, and steps."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Describe the image to generate."},
            "aspect_ratio": {"type": "string", "enum": list(ASPECT_RATIOS),
                             "description": "Aspect ratio, or omit to use the configured default."},
            "resolution": {"type": "integer", "minimum": 256, "maximum": 1536, "multipleOf": 16,
                           "description": "Length of the longest side in pixels."},
            "steps": {"type": "integer", "minimum": 1, "maximum": 12,
                      "description": "Number of inference steps for this image."},
            "preset": {"type": "string", "description": "Named preset in the shared local AI catalogue."},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
}


def _generate_explicit(provider: MLXImageGenProvider, args: dict, **_: Any) -> str:
    try:
        options = _validated_options({field: args[field] for field in GENERATION_FIELDS if field in args},
                                     "MLX request", "invalid_request")
    except _RequestError as exc:
        return json.dumps(error_response(error=str(exc), error_type=exc.error_type, provider=provider.name))
    result = provider.generate(args.get("prompt", ""), **options, preset=args.get("preset"), legacy_prompt_size=False)
    return json.dumps(result, ensure_ascii=False)


def register(ctx: PluginContext):
    """Register the provider and opt-in request controls through Hermes's plugin API."""
    provider = MLXImageGenProvider()
    ctx.register_image_gen_provider(provider)
    ctx.register_hook("pre_tool_call", _apply_native_aspect_default)
    ctx.register_tool(
        name=_TOOL_SCHEMA["name"], toolset="image_gen", schema=_TOOL_SCHEMA,
        handler=lambda args, **kwargs: _generate_explicit(provider, args, **kwargs),
        check_fn=lambda: _mlx_selected() and provider.is_available(), emoji="🖼️",
    )
