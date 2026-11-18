"""Deferred offline provider regressions; source reviewed, not executed in this audit.

Later authorized run: python3 -m unittest discover -s tests -p test_provider.py.

Hermes imports are scoped stubs; all files and executable fixtures live in a
TemporaryDirectory. The lifecycle integration uses real Hermes separately.
"""
from __future__ import annotations

import base64
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
PNG_DATA = b"\x89PNG\r\n\x1a\n" + b"fixture-png-payload"


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
        self.secret_values = {}
        self.secret_loader = Mock(side_effect=lambda name, default=None: self.secret_values.get(name, default))
        self.real_home = Mock(return_value=str(self.root / "account-home"))
        self.file_guard = Mock()
        self.url_source = self.root / "downloaded-url.png"
        self.url_source.write_bytes(PNG_DATA)
        self.url_loader = Mock(return_value=self.url_source)
        self.data_source = self.root / "downloaded-data.png"

        def save_data(payload, **kwargs):
            self.data_source.write_bytes(base64.b64decode(payload, validate=True))
            return self.data_source

        self.data_loader = Mock(side_effect=save_data)
        self.config_loader = Mock(return_value={"image_gen": {"mlx": self.settings}})
        modules = {name: types.ModuleType(name) for name in (
            "agent", "agent.image_gen_provider", "agent.secret_scope", "agent.file_safety",
            "hermes_cli", "hermes_cli.plugins",
            "hermes_cli.config", "hermes_constants",
        )}
        modules["agent.image_gen_provider"].ImageGenProvider = type("ImageGenProvider", (), {})
        modules["agent.image_gen_provider"].success_response = lambda **kw: {"success": True, **kw}
        modules["agent.image_gen_provider"].error_response = lambda **kw: {"success": False, **kw}
        def resolve_aspect(value):
            value = value.strip().lower() if isinstance(value, str) else ""
            return value if value in ("landscape", "square", "portrait") else "landscape"
        modules["agent.image_gen_provider"].resolve_aspect_ratio = resolve_aspect
        modules["agent.image_gen_provider"].save_url_image = self.url_loader
        modules["agent.image_gen_provider"].save_b64_image = self.data_loader
        modules["agent.secret_scope"].get_secret = self.secret_loader
        modules["agent.file_safety"].raise_if_read_blocked = self.file_guard
        modules["hermes_cli.plugins"].PluginContext = type("PluginContext", (), {})
        modules["hermes_cli.config"].load_config = self.config_loader
        modules["hermes_constants"].get_hermes_home = lambda: str(self.root / "hermes")
        modules["hermes_constants"].get_real_home = self.real_home
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
        self.real_home.reset_mock()
        self.secret_loader.reset_mock()
        with patch.object(subprocess, "run", side_effect=AssertionError("import launched child")):
            installed = self.load_module(ROOT / "__init__.py", package=True)
        self.config_loader.assert_not_called()
        self.real_home.assert_not_called()
        self.secret_loader.assert_not_called()
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
        self.assertEqual(self.provider.capabilities()["max_reference_images"], 0)
        self.assertFalse(self.provider.capabilities().get("supports_upscale", False))
        self.assertNotIn("resolution", self.provider.capabilities().get("creative_controls", []))
        for prompt, kwargs, expected in (
            ("a cat [size=512]", {}, ("a cat", 512)),
            ("a cat", {"size": 1}, ("a cat", 256)),
            ("a cat", {"resolution": 9999}, ("a cat", 1536)),
            ("a cat", {"resolution": 513}, ("a cat", 512)),
        ):
            self.assertEqual(self.mod._resolution_request(prompt, kwargs), expected)
        self.assertEqual(self.mod._dimensions("landscape", 512), (512, 288))
        self.assertEqual(self.mod._dimensions("portrait", 512), (288, 512))
        self.assertEqual(self.mod._dimensions("square", 512), (512, 512))
        self.assertEqual([self.mod._steps_for(size) for size in (256, 512, 1024, 1536)], [4, 6, 8, 12])

    def test_local_command_environment_and_unique_outputs(self):
        sentinel = {"PYTHONPATH": "/private/hermes/python", "PYTHONHOME": "/private/hermes",
                    "VIRTUAL_ENV": "/private/hermes/venv", "OPENAI_API_KEY": "PRIVATE_TEST_KEY",
                    "DYRA_LOCAL_AI_LOCK_TIMEOUT": "999",
                    "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}
        self.settings["models_dir"] = str(self.models / ".." / "models")
        self.secret_values["DYRA_LOCAL_AI_LOCK_TIMEOUT"] = "13"
        with patch.dict(os.environ, sentinel), patch.object(subprocess, "run", side_effect=self.fake_run), patch.object(self.mod, "datetime") as clock:
            clock.now.return_value.strftime.return_value = "20261118_120000"
            first = self.provider.generate("a red apple", "square", resolution=512)
            second = self.provider.generate("a red apple", "square", resolution=512)
            for key, value in sentinel.items():
                self.assertEqual(os.environ[key], value)
        self.assertTrue(first["success"], first)
        self.assertNotEqual(first["image"], second["image"])
        self.assertEqual(first["provider"], "mlx")
        self.assertEqual(Path(first["image"]).parent, self.root / "hermes/cache/generated/images")
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
        self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
        self.assertEqual(kwargs["env"]["DYRA_LOCAL_AI_LOCK_TIMEOUT"], "13")
        self.assertEqual(kwargs["env"]["HOME"], str(self.root / "account-home"))
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
        source.write_bytes(PNG_DATA)
        with patch.object(subprocess, "run", side_effect=self.fake_run):
            edited = self.provider.generate("make the apple blue", "square", image_url=str(source))
        self.assertTrue(edited["success"], edited)
        self.assertEqual(edited["modality"], "image")
        command = self.calls[-1][0]
        self.assertEqual(command[command.index("--image-paths") + 1], str(source))
        self.assertNotIn("--image-strength", command)
        self.assertNotIn("--image-path", command)
        self.assertEqual(source.read_bytes(), PNG_DATA)
        self.file_guard.assert_called_with(str(source))
        with patch.object(subprocess, "run") as run:
            for invalid in (self.root / "missing.png", self.root):
                result = self.provider.generate("x", image_url=str(invalid))
                self.assertEqual(result["error_type"], "input_image_not_found")
                self.assertNotIn(str(invalid), result["error"])
            run.assert_not_called()

    def test_missing_empty_or_directory_output_is_not_success(self):
        for mode in ("missing", "empty", "directory", "garbage"):
            def output_run(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                if output.is_file():
                    output.unlink()
                if mode == "empty":
                    output.touch()
                elif mode == "directory":
                    output.mkdir()
                elif mode == "garbage":
                    output.write_bytes(b"not-an-image" * 3)
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            with self.subTest(mode=mode), patch.object(subprocess, "run", side_effect=output_run):
                result = self.provider.generate("x")
                self.assertFalse(result["success"])
                self.assertEqual(result["error_type"], "file_not_found")


    def test_invalid_config_and_model_do_not_launch_or_fetch(self):
        cases = [
            ([], "invalid_config"),
            ({"image_gen": None}, "invalid_config"),
            ({"image_gen": {"mlx": []}}, "invalid_config"),
            ({"image_gen": {"mlx": {**self.settings, "wrapper": []}}}, "invalid_config"),
            ({"image_gen": {"mlx": {**self.settings, "models_dir": "relative/models"}}}, "invalid_config"),
            ({"image_gen": {"mlx": {**self.settings, "model": "unknown-model"}}}, "unsupported_model"),
        ]
        with patch.object(subprocess, "run") as run:
            for config, error_type in cases:
                with self.subTest(config=config):
                    self.config_loader.return_value = config
                    result = self.provider.generate("x", image_url="https://example.invalid/source.png")
                    self.assertFalse(result["success"], result)
                    self.assertEqual(result["error_type"], error_type, result)
            run.assert_not_called()
        self.url_loader.assert_not_called()
        self.data_loader.assert_not_called()


    def test_native_model_override_survives_stale_scoped_default(self):
        self.settings["model"] = "old-unsupported-model"
        self.assertTrue(self.provider.is_available(), "native model selection must remain reachable")
        with patch.object(subprocess, "run", side_effect=self.fake_run):
            explicit = self.provider.generate("x", model="flux2-klein-4b-4bit")
        self.assertTrue(explicit["success"], explicit)
        with patch.object(subprocess, "run") as run:
            defaulted = self.provider.generate("x")
            self.assertEqual(defaulted["error_type"], "unsupported_model", defaulted)
            run.assert_not_called()

    def test_unknown_model_references_and_native_resolution_are_explicit_errors(self):
        cases = [
            ({"model": "unknown-model"}, "unsupported_model"),
            ({"model": 42}, "invalid_request"),
            ({"reference_image_urls": ["https://example.invalid/ref.png"]}, "modality_unsupported"),
            ({"resolution": "1K"}, "invalid_resolution"),
            ({"resolution": "2K"}, "invalid_resolution"),
            ({"resolution": "bad"}, "invalid_resolution"),
        ]
        with patch.object(subprocess, "run") as run:
            for kwargs, error_type in cases:
                with self.subTest(kwargs=kwargs):
                    result = self.provider.generate("x", **kwargs)
                    self.assertEqual(result["error_type"], error_type, result)
            run.assert_not_called()
        self.url_loader.assert_not_called()
        self.data_loader.assert_not_called()
        with patch.object(subprocess, "run", side_effect=self.fake_run):
            result = self.provider.generate("x", "not-an-aspect", future_control={"ignored": True})
        self.assertTrue(result["success"], result)
        self.assertEqual(result["aspect_ratio"], "landscape")

    def test_scoped_overrides_and_real_home_defaults(self):
        self.settings.clear()
        self.secret_values.update(
            MLX_WRAPPER=str(self.wrapper), MLX_MODELS_DIR=str(self.models),
            MLX_IMAGE_MODEL="flux2-klein-4b",
        )
        with patch.dict(os.environ, {"MLX_WRAPPER": "/ambient/wrong-wrapper", "MLX_MODELS_DIR": "/ambient/models",
                                      "MLX_IMAGE_MODEL": "wrong-model"}), patch.object(subprocess, "run", side_effect=self.fake_run):
            scoped = self.provider.generate("x")
        self.assertTrue(scoped["success"], scoped)
        self.assertEqual(self.calls[-1][0][0], str(self.wrapper))
        self.secret_values.clear()
        for name in ("first-home", "second-home"):
            account = self.root / name
            wrapper = account / ".local/bin/dyra-image"
            wrapper.parent.mkdir(parents=True)
            wrapper.write_text("#!/bin/sh\nexit 0\n")
            wrapper.chmod(0o700)
            model = account / ".local/share/dyra-ai/models/image/flux2-klein-4b-4bit"
            model.mkdir(parents=True)
            self.real_home.return_value = str(account)
            with patch.object(subprocess, "run", side_effect=self.fake_run):
                result = self.provider.generate("x")
            self.assertTrue(result["success"], result)
            command, kwargs = self.calls[-1]
            self.assertEqual(command[0], str(wrapper))
            self.assertEqual(command[command.index("--model") + 1], str(model))
            self.assertEqual(kwargs["env"]["HOME"], str(account))

    def test_edit_url_and_data_use_public_helpers_and_clean_owned_inputs(self):
        url = "https://example.invalid/source.png"
        with patch.object(subprocess, "run", side_effect=self.fake_run):
            result = self.provider.generate("edit", image_url=url)
        self.assertTrue(result["success"], result)
        self.url_loader.assert_called_once()
        url_args, url_kwargs = self.url_loader.call_args
        self.assertEqual(url_args, (url,))
        self.assertEqual(url_kwargs["max_bytes"], 25 * 1024 * 1024)
        self.assertRegex(url_kwargs["prefix"], r"^mlx_input_[0-9a-f]+$")
        self.assertEqual(self.calls[-1][0][self.calls[-1][0].index("--image-paths") + 1], str(self.url_source))
        self.assertFalse(self.url_source.exists(), "downloaded input must be cleaned after the request")
        payload = base64.b64encode(PNG_DATA).decode("ascii")
        with patch.object(subprocess, "run", side_effect=self.fake_run):
            result = self.provider.generate("edit", image_url="data:image/png;base64," + payload)
        self.assertTrue(result["success"], result)
        self.data_loader.assert_called_once()
        data_args, data_kwargs = self.data_loader.call_args
        self.assertEqual(data_args, (payload,))
        self.assertEqual(data_kwargs["extension"], "png")
        self.assertRegex(data_kwargs["prefix"], r"^mlx_input_[0-9a-f]+$")
        self.assertNotEqual(url_kwargs["prefix"], data_kwargs["prefix"])
        self.assertFalse(self.data_source.exists())
        self.assertTrue(Path(result["image"]).is_file(), "successful output must be retained")

    def test_bad_inputs_and_file_safety_stop_before_runtime(self):
        source = self.root / "protected.png"
        source.write_bytes(PNG_DATA)
        self.file_guard.side_effect = ValueError("/private/DO_NOT_EXPOSE")
        self.url_loader.side_effect = ValueError("https://private.invalid/DO_NOT_EXPOSE")
        with patch.object(subprocess, "run") as run:
            for value in (
                str(source), "https://example.invalid/blocked.png", "data:image/svg+xml;base64,AAAA",
                "data:image/png;base64,***", "file:///private/image.png",
            ):
                with self.subTest(value=value):
                    result = self.provider.generate("edit", image_url=value)
                    self.assertFalse(result["success"], result)
                    self.assertNotIn("DO_NOT_EXPOSE", result["error"])
            run.assert_not_called()
        self.data_loader.assert_not_called()
        self.assertEqual(source.read_bytes(), PNG_DATA, "caller-owned inputs must survive rejection")
        self.file_guard.side_effect = None
        with patch.object(self.mod, "MAX_INPUT_IMAGE_BYTES", 16), patch.object(subprocess, "run") as run:
            oversized = "data:image/png;base64," + base64.b64encode(PNG_DATA * 3).decode("ascii")
            result = self.provider.generate("edit", image_url=oversized)
            self.assertEqual(result["error_type"], "invalid_input_image", result)
            run.assert_not_called()

    def test_failed_output_cleanup_does_not_follow_symlinks(self):
        target = self.root / "caller-owned.png"
        target.write_bytes(PNG_DATA)
        outputs = []
        def symlink_run(command, **kwargs):
            output = Path(command[command.index("--output") + 1])
            outputs.append(output)
            output.symlink_to(target)
        with patch.object(subprocess, "run", side_effect=symlink_run):
            result = self.provider.generate("x")
        self.assertEqual(result["error_type"], "file_not_found", result)
        self.assertFalse(outputs[-1].is_symlink())
        self.assertEqual(target.read_bytes(), PNG_DATA)
        for exception in (subprocess.CalledProcessError(9, []), subprocess.TimeoutExpired([], 1800)):
            def partial_run(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                outputs.append(output)
                output.write_bytes(b"partial")
                raise exception
            with self.subTest(exception=type(exception).__name__), patch.object(subprocess, "run", side_effect=partial_run):
                result = self.provider.generate("x")
            self.assertFalse(result["success"])
            self.assertFalse(outputs[-1].exists(), "failed attempt must not leave partial output")

    def test_setup_io_errors_keep_safe_response_envelope(self):
        with patch.object(Path, "mkdir", side_effect=OSError("/private/DO_NOT_EXPOSE")), patch.object(subprocess, "run") as run:
            result = self.provider.generate("x")
        self.assertEqual(result["error_type"], "unexpected_error", result)
        self.assertNotIn("DO_NOT_EXPOSE", result["error"])
        run.assert_not_called()


    def test_marker_only_prompt_does_not_launch_runtime(self):
        with patch.object(subprocess, "run") as run:
            for prompt in ("[size=512]", "[resolution=512] "):
                result = self.provider.generate(prompt)
                self.assertEqual(result["error_type"], "invalid_request", result)
            run.assert_not_called()

    def test_failed_media_helpers_clean_only_their_attempt_files(self):
        cache = self.root / "hermes/cache/generated/images"
        cache.mkdir(parents=True)
        unrelated = cache / "mlx_input_other_attempt_partial.png"
        unrelated.write_bytes(PNG_DATA)
        local = self.root / "caller-input.png"
        local.write_bytes(PNG_DATA)
        attempts = []
        def fail_after_partial(value, **kwargs):
            partial = cache / (kwargs["prefix"] + "_partial.png")
            attempts.append(partial)
            partial.write_bytes(b"partial")
            raise OSError("/private/DO_NOT_EXPOSE")
        payload = base64.b64encode(PNG_DATA).decode("ascii")
        cases = (
            (self.url_loader, "https://example.invalid/interrupted.png"),
            (self.data_loader, "data:image/png;base64," + payload),
        )
        with patch.object(subprocess, "run") as run:
            for helper, value in cases:
                helper.side_effect = fail_after_partial
                result = self.provider.generate("edit", image_url=value)
                self.assertFalse(result["success"], result)
                self.assertNotIn("DO_NOT_EXPOSE", result["error"])
                self.assertFalse(attempts[-1].exists())
                self.assertEqual(unrelated.read_bytes(), PNG_DATA)
                self.assertEqual(local.read_bytes(), PNG_DATA)
            run.assert_not_called()

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
