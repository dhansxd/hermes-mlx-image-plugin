"""Offline provider regressions: python3 -m unittest discover -s tests -p test_provider.py.

Hermes imports are scoped stubs; all files and executable fixtures live in a
TemporaryDirectory. The lifecycle integration uses real Hermes separately.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hermes-mlx-provider-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root / "models"
        self.local_model = self.models / "flux2-klein-4b-4bit"
        self.local_model.mkdir(parents=True)
        self.wrapper = self.root / "dyra-image"
        self.wrapper.write_text("#!/bin/sh\nexit 0\n")
        self.wrapper.chmod(0o700)
        self.settings = {"wrapper": str(self.wrapper), "models_dir": str(self.models)}
        self.config_loader = Mock(return_value={"image_gen": {"mlx": self.settings}})
        modules = {name: types.ModuleType(name) for name in (
            "agent", "agent.image_gen_provider", "hermes_cli", "hermes_cli.plugins",
            "hermes_cli.config", "hermes_constants",
        )}
        modules["agent.image_gen_provider"].ImageGenProvider = type("ImageGenProvider", (), {})
        modules["agent.image_gen_provider"].success_response = lambda **kw: {"success": True, **kw}
        modules["agent.image_gen_provider"].error_response = lambda **kw: {"success": False, **kw}
        modules["hermes_cli.plugins"].PluginContext = type("PluginContext", (), {})
        modules["hermes_cli.config"].load_config = self.config_loader
        modules["hermes_constants"].get_hermes_home = lambda: str(self.root / "hermes")
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        # Prevent ambient MLX overrides affecting defaults and discovery assertions.
        self.env_patch = patch.dict(os.environ, {}, clear=False)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        for key in ("MLX_IMAGE_MODEL", "MLX_MODELS_DIR", "MLX_WRAPPER"):
            os.environ.pop(key, None)
        self.mod = self.load_module(ROOT / "src/hermes_mlx_image/__init__.py")
        self.provider = self.mod.MLXImageGenProvider()
        self.calls = []

    def load_module(self, path, package=False):
        name = "mlx_test_" + uuid4().hex
        kwargs = {"submodule_search_locations": [str(path.parent)]} if package else {}
        spec = importlib.util.spec_from_file_location(name, path, **kwargs)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        def remove_modules():
            for key in list(sys.modules):
                if key == name or key.startswith(name + "."):
                    sys.modules.pop(key, None)
        self.addCleanup(remove_modules)
        spec.loader.exec_module(module)
        return module

    def fake_run(self, command, **kwargs):
        self.calls.append((command, kwargs))
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        return subprocess.CompletedProcess(command, 0)

    def test_import_is_passive_and_registration_is_explicit(self):
        self.config_loader.reset_mock()
        with patch.object(subprocess, "run", side_effect=AssertionError("import launched child")):
            installed = self.load_module(ROOT / "__init__.py", package=True)
        self.config_loader.assert_not_called()
        registrations = []
        installed.register(types.SimpleNamespace(register_image_gen_provider=registrations.append))
        self.assertEqual([provider.name for provider in registrations], ["mlx"])
        self.config_loader.assert_not_called()

    def test_identity_models_aliases_and_dimensions(self):
        self.assertEqual(self.provider.name, "mlx")
        self.assertIn("MLX", self.provider.description.upper())
        self.assertEqual(list(self.mod.MODEL_TABLE), ["flux2-klein-4b-4bit"])
        for alias in ("flux2-klein-4b-4bit", "flux2-klein-4b", "mlx-community/FLUX.2-Klein-4B-4bit"):
            self.assertEqual(self.mod._canonical_key(alias), "flux2-klein-4b-4bit")
        self.assertIsNone(self.mod._canonical_key(""))
        self.assertTrue(self.provider.is_available())
        self.assertEqual(self.provider.default_model(), "flux2-klein-4b-4bit")
        self.assertTrue(self.provider.list_models()[0]["local_available"])
        for prompt, kwargs, expected in (
            ("a cat [size=512]", {}, ("a cat", 512)),
            ("a cat", {"size": 1}, ("a cat", 256)),
            ("a cat", {"resolution": 9999}, ("a cat", 1536)),
            ("a cat", {"resolution": 513}, ("a cat", 512)),
            ("a cat", {"resolution": "bad"}, ("a cat", 256)),
        ):
            self.assertEqual(self.mod._resolution_request(prompt, kwargs), expected)
        self.assertEqual(self.mod._dimensions("landscape", 512), (512, 288))
        self.assertEqual(self.mod._dimensions("portrait", 512), (288, 512))
        self.assertEqual(self.mod._dimensions("square", 512), (512, 512))
        self.assertEqual([self.mod._steps_for(size) for size in (256, 512, 1024, 1536)], [4, 6, 8, 12])

    def test_local_command_environment_and_unique_outputs(self):
        sentinel = {"PYTHONPATH": "/private/hermes/python", "PYTHONHOME": "/private/hermes",
                    "VIRTUAL_ENV": "/private/hermes/venv", "MLX_TEST_KEEP": "yes",
                    "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}
        self.settings["models_dir"] = str(self.models / ".." / "models")
        with patch.dict(os.environ, sentinel), patch.object(subprocess, "run", side_effect=self.fake_run), patch.object(self.mod, "datetime") as clock:
            clock.now.return_value.strftime.return_value = "20261118_120000"
            first = self.provider.generate("a red apple", "square", resolution=512)
            second = self.provider.generate("a red apple", "square", resolution=512)
            for key, value in sentinel.items():
                self.assertEqual(os.environ[key], value)
        self.assertTrue(first["success"], first)
        self.assertNotEqual(first["image"], second["image"])
        self.assertEqual(first["provider"], "mlx")
        self.assertEqual(first["model"], "mlx-community/FLUX.2-Klein-4B-4bit")
        command, kwargs = self.calls[0]
        arguments = {key: command[command.index(key) + 1] for key in (
            "--model", "--base-model", "--quantize", "--steps", "--width", "--height", "--prompt")}
        self.assertEqual(arguments, {"--model": str(self.local_model.resolve()),
            "--base-model": "flux2-klein-4b", "--quantize": "4", "--steps": "6",
            "--width": "512", "--height": "512", "--prompt": "a red apple"})
        self.assertEqual(kwargs["timeout"], 1800)
        self.assertTrue(kwargs["check"])
        self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
        self.assertFalse({"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} & kwargs["env"].keys())
        self.assertEqual(kwargs["env"]["MLX_TEST_KEEP"], "yes")
        self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(kwargs["env"]["TRANSFORMERS_OFFLINE"], "1")

    def test_missing_runtime_and_model_stop_before_subprocess(self):
        with patch.object(subprocess, "run") as run:
            self.wrapper.chmod(0o600)
            self.assertEqual(self.provider.generate("x")["error_type"], "mflux_not_installed")
            self.wrapper.chmod(0o700)
            self.local_model.rmdir()
            self.assertFalse(self.provider.list_models()[0]["local_available"])
            self.assertEqual(self.provider.generate("x")["error_type"], "model_not_found")
            run.assert_not_called()

    def test_edit_uses_mflux_image_paths_and_rejects_missing_or_directory(self):
        source = self.root / "reference.png"
        source.write_bytes(b"fixture")
        with patch.object(subprocess, "run", side_effect=self.fake_run):
            edited = self.provider.generate("make the apple blue", "square", image_url=str(source))
        self.assertTrue(edited["success"], edited)
        self.assertEqual(edited["modality"], "image")
        command = self.calls[-1][0]
        self.assertEqual(command[command.index("--image-paths") + 1], str(source))
        self.assertNotIn("--image-strength", command)
        self.assertNotIn("--image-path", command)
        with patch.object(subprocess, "run") as run:
            for invalid in (self.root / "missing.png", self.root):
                result = self.provider.generate("x", image_url=str(invalid))
                self.assertEqual(result["error_type"], "input_image_not_found")
                self.assertNotIn(str(invalid), result["error"])
            run.assert_not_called()

    def test_missing_empty_or_directory_output_is_not_success(self):
        for mode in ("missing", "empty", "directory"):
            def output_run(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                if output.is_file():
                    output.unlink()
                if mode == "empty":
                    output.touch()
                elif mode == "directory":
                    output.mkdir()
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            with self.subTest(mode=mode), patch.object(subprocess, "run", side_effect=output_run):
                result = self.provider.generate("x")
                self.assertFalse(result["success"])
                self.assertEqual(result["error_type"], "file_not_found")

    def test_logs_and_errors_exclude_private_values(self):
        private_prompt = "SECRET_PROMPT_FOR_TEST"
        private_path = "/private/SECRET_USER/image.png"
        private_output = "SECRET_CHILD_OUTPUT"
        errors = (
            (subprocess.CalledProcessError(7, [private_path], output=private_output, stderr=private_output), "mflux_execution_failed"),
            (subprocess.TimeoutExpired([private_path], 1800, output=private_output, stderr=private_output), "mflux_timeout"),
            (OSError(private_path + private_output), "unexpected_error"),
        )
        for exception, error_type in errors:
            with self.subTest(error_type=error_type), self.assertLogs(self.mod.logger, logging.INFO) as logs:
                with patch.object(subprocess, "run", side_effect=exception):
                    result = self.provider.generate(private_prompt)
            self.assertEqual(result["error_type"], error_type)
            # Hermes responses preserve their standard prompt field; diagnostics must not.
            diagnostics = result["error"] + "\n" + "\n".join(logs.output)
            for private_value in (private_prompt, private_path, private_output, str(self.root)):
                self.assertNotIn(private_value, diagnostics)

    @unittest.skipUnless(sys.platform != "win32", "requires POSIX flock and child signals")
    def test_timeout_kills_child_and_releases_shared_lock(self):
        import fcntl
        lock_path = self.root / "shared.lock"
        marker_path = self.root / "child.pid"
        # The real wrapper runs MFlux in its own Python process. This fixture has the same
        # single-process shape and never imports MLX or accesses the network.
        self.wrapper.write_text("#!" + sys.executable + "\n" + textwrap.dedent(f"""
            import fcntl
            import os
            from pathlib import Path
            import time
            with open({str(lock_path)!r}, 'w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                Path({str(marker_path)!r}).write_text(str(os.getpid()))
                time.sleep(60)
        """))
        self.wrapper.chmod(0o700)
        with patch.object(self.mod, "GENERATION_TIMEOUT_SECONDS", 0.5), self.assertLogs(self.mod.logger, logging.ERROR):
            result = self.provider.generate("timeout fixture")
        self.assertEqual(result["error_type"], "mflux_timeout", result)
        self.assertTrue(marker_path.is_file(), "child did not acquire the lock before timeout")
        pid = int(marker_path.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        with open(lock_path, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    unittest.main(verbosity=2)
