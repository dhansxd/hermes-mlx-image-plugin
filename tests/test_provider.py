"""Tests for the generic local MLX image provider. Offline: monkeypatched subprocess.

Run: python3 tests/test_provider.py  (or: python3 -m pytest tests/ -q)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Minimal stubs so the provider module imports outside the Hermes tree.
import types


def _ensure_stubs() -> None:
    if "agent" not in sys.modules:
        sys.modules["agent"] = types.ModuleType("agent")
    if "agent.image_gen_provider" not in sys.modules:
        m = types.ModuleType("agent.image_gen_provider")

        class _Provider:
            pass

        m.ImageGenProvider = _Provider
        m.error_response = lambda **kw: {
            "success": False,
            "image": None,
            **{k: kw.get(k, "") for k in ("error", "error_type", "provider", "model", "prompt", "aspect_ratio")},
        }
        m.success_response = lambda **kw: {"success": True, **kw}
        sys.modules["agent.image_gen_provider"] = m
    if "hermes_cli" not in sys.modules:
        h = types.ModuleType("hermes_cli")
        sys.modules["hermes_cli"] = h
    if "hermes_cli.plugins" not in sys.modules:
        p = types.ModuleType("hermes_cli.plugins")

        class PluginContext:
            pass

        p.PluginContext = PluginContext
        sys.modules["hermes_cli.plugins"] = p
    if "hermes_constants" not in sys.modules:
        c = types.ModuleType("hermes_constants")
        import tempfile

        c.get_hermes_home = lambda: tempfile.gettempdir() + "/hermes-mlx-test-home"
        sys.modules["hermes_constants"] = c


_ensure_stubs()


def _load():
    for name in list(sys.modules):
        if name.startswith("hermes_mlx_image"):
            del sys.modules[name]
    import hermes_mlx_image as mod

    return mod, mod.MLXImageGenProvider()


def main() -> int:
    import tempfile

    mod, provider = _load()

    # 1. Generic identity (NOT a model name).
    assert provider.name == "mlx", provider.name
    assert "MLX" in provider.description.upper(), provider.description

    # 2. Single-row table for now, keyed generically.
    assert list(mod.MODEL_TABLE.keys()) == ["flux2-klein-4b-4bit"], list(mod.MODEL_TABLE.keys())
    entry = mod.MODEL_TABLE["flux2-klein-4b-4bit"]
    assert entry["hf_id"] == "mlx-community/FLUX.2-Klein-4B-4bit"
    assert entry["base_model"] == "flux2-klein-4b"
    assert entry["quantize"] == "4"

    # 3. Alias resolution (model name never pins the plugin name).
    assert mod._canonical_key("flux2-klein-4b-4bit") == "flux2-klein-4b-4bit"
    assert mod._canonical_key("flux2-klein-4b") == "flux2-klein-4b-4bit"
    assert mod._canonical_key("mlx-community/FLUX.2-Klein-4B-4bit") == "flux2-klein-4b-4bit"
    assert mod._canonical_key("") is None

    # 4. list_models / default_model work offline.
    models = provider.list_models()
    assert len(models) == 1 and models[0]["id"] == "flux2-klein-4b-4bit", models
    assert provider.default_model() == "flux2-klein-4b-4bit"

    # 5. Resolution helpers preserved from the original plugin.
    p, side = mod._resolution_request("a cat [size=512]", {})
    assert (p, side) == ("a cat", 512), (p, side)
    assert mod._dimensions("landscape", 512) == (512, 288)
    assert mod._dimensions("portrait", 512) == (288, 512)
    assert mod._dimensions("square", 512) == (512, 512)
    assert mod._steps_for(256) == 4
    assert mod._steps_for(1536) == 12

    # 6. generate() builds the wrapper command from the TABLE (stub subprocess).
    captured: dict = {}
    fake_out = Path(tempfile.mkdtemp()) / "mlx_fake.png"

    class _Completed:
        stdout = "ok"
        stderr = ""

    def fake_run(cmd, **kw):
        captured["cmd"] = list(cmd)
        captured["env"] = kw.get("env", os.environ.copy())
        # Simulate wrapper writing the output file (last arg after --output).
        out = Path(cmd[cmd.index("--output") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        return _Completed()

    import subprocess as real_subprocess

    orig_run = real_subprocess.run
    orig_isfile = os.path.isfile
    orig_access = os.access
    real_subprocess.run = fake_run
    os.path.isfile = lambda p: True if str(p).endswith("dyra-image") else orig_isfile(p)
    os.access = lambda p, m: True if str(p).endswith("dyra-image") else orig_access(p, m)
    from unittest.mock import patch
    with patch.dict(os.environ, {"PYTHONPATH": "/hermes/python3.14", "PYTHONHOME": "/hermes", "VIRTUAL_ENV": "/hermes/venv", "MLX_TEST_KEEP": "yes"}):
        try:
            mod2, provider2 = _load()
            result = provider2.generate("a red apple", "square", resolution=256)
            assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} & captured["env"].keys(), "Python runtime environment leaked"
            assert captured["env"]["MLX_TEST_KEEP"] == "yes"
            assert os.environ["PYTHONPATH"] == "/hermes/python3.14"
        finally:
            real_subprocess.run = orig_run
            os.path.isfile = orig_isfile
            os.access = orig_access

    assert result["success"] is True, result
    assert result["provider"] == "mlx", result
    assert result["model"] == "mlx-community/FLUX.2-Klein-4B-4bit", result
    cmd = captured["cmd"]
    assert "--base-model" in cmd and cmd[cmd.index("--base-model") + 1] == "flux2-klein-4b", cmd
    assert "--quantize" in cmd and cmd[cmd.index("--quantize") + 1] == "4", cmd
    assert "--width" in cmd and "--height" in cmd

    # 7. Missing input image → clean error (no subprocess).
    os.path.isfile = lambda p: True if str(p).endswith("dyra-image") else orig_isfile(p)
    os.access = lambda p, m: True if str(p).endswith("dyra-image") else orig_access(p, m)
    try:
        mod3, provider3 = _load()
        bad = provider3.generate("x", "square", image_url="/definitely/missing.png")
    finally:
        os.path.isfile = orig_isfile
        os.access = orig_access
    assert bad["success"] is False and bad["error_type"] == "input_image_not_found", bad

    # Git installs must load register() from the repository root.
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    assert (root / "__init__.py").is_file(), "Git plugin root entry point missing"
    spec = importlib.util.spec_from_file_location(
        "mlx_plugin_install", root / "__init__.py", submodule_search_locations=[str(root)]
    )
    assert spec is not None and spec.loader is not None
    installed = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = installed
    spec.loader.exec_module(installed)
    registered = []
    installed.register(types.SimpleNamespace(register_image_gen_provider=registered.append))
    assert len(registered) == 1 and registered[0].name == "mlx"

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
