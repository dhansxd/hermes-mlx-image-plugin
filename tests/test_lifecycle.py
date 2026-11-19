"""Deferred opt-in integration check against Hermes stable 0.23.2.

Source reviewed for the pinned target; not executed during the 1.0.2 audit.

Run this separately from the isolated provider unit tests, using a Python
environment with Hermes's dependencies installed. No Hermes modules are stubbed.
The CLI mode uses Hermes's actual ``python -m hermes_cli.main`` entry point;
``--manager-only`` checks directory discovery/registration without claiming that
the package-manager-backed CLI installation passed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile


HERMES_REVISION = "ae2a81ccc00e9b7c06301523e5bb09608397149c"
PLUGIN_ID = "mlx-image-gen"


def command(argv, *, cwd, env, timeout=90):
    process = subprocess.Popen(
        [str(arg) for arg in argv], cwd=cwd, env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace", start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, stderr = process.communicate()
        raise AssertionError(f"Command exceeded {timeout}s: {argv}\n{stdout[-3000:]}\n{stderr[-3000:]}") from None
    if process.returncode:
        raise AssertionError(
            f"Command failed ({process.returncode}): {argv}\n"
            f"{stdout[-6000:]}\n{stderr[-6000:]}"
        )
    return stdout.strip()


def run(args):
    upstream = args.hermes_root.resolve()
    source = Path(__file__).resolve().parents[1]
    revision = command(["git", "rev-parse", "HEAD"], cwd=upstream, env=os.environ)
    assert revision == HERMES_REVISION, ("Unexpected upstream revision", revision)
    with tempfile.TemporaryDirectory(prefix="mlx-lifecycle-", dir=args.work_dir) as tmp:
        root = Path(tmp).resolve()
        home = root / "hermes-home"
        home.mkdir()
        external = root / "external-runtime"
        model = external / "models" / "flux2-klein-4b-4bit"
        model.mkdir(parents=True)
        sentinel = model / "weights-sentinel"
        sentinel.write_text("external assets survive plugin removal", encoding="utf-8")
        wrapper = external / "dyra-image"
        wrapper.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        wrapper.chmod(0o700)
        bundled = root / "empty-bundled-plugins"
        bundled.mkdir()
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("HERMES_", "MLX_", "PYTEST_"))
               and not k.endswith(("_API_KEY", "_TOKEN", "_SECRET"))
               and k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "GITHUB_TOKEN", "GH_TOKEN"}}
        env.update(
            HOME=str(root / "user-home"), USERPROFILE=str(root / "user-home"),
            HERMES_HOME=str(home), HERMES_RUNTIME_DIR=str(root / "runtime"),
            HERMES_BUNDLED_PLUGINS=str(bundled),
            HERMES_DISABLE_LAZY_INSTALLS="1", PYTHONPATH=str(upstream),
            PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1", NO_COLOR="1",
            UV_CACHE_DIR=str(root / "uv-cache"),
            XDG_CONFIG_HOME=str(root / "config"),
            XDG_CACHE_HOME=str(root / "cache"),
        )
        (root / "user-home").mkdir()
        os.environ.clear()
        os.environ.update(env)
        sys.path.insert(0, str(upstream))
        from hermes_cli.plugins import PluginManager
        from agent.image_gen_registry import get_provider
        from tools.registry import registry
        import hermes_yaml as yaml

        config = {
            "plugins": {"enabled": [], "disabled": []},
            "image_gen": {"provider": "mlx", "mlx": {
                "models_dir": str(model.parent), "wrapper": str(wrapper),
            }},
        }

        def write_config(enabled):
            config["plugins"]["enabled"] = [PLUGIN_ID] if enabled else []
            config["plugins"]["disabled"] = [] if enabled else [PLUGIN_ID]
            (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

        write_config(False)
        installed = home / "plugins" / PLUGIN_ID
        record_path = home / "plugins" / ".install-metadata.json"
        report = {"hermes_revision": revision, "mode": "manager" if args.manager_only else "cli"}

        def cli(*argv):
            output = command(
                [sys.executable, "-m", "hermes_cli.main", "plugins", *argv],
                cwd=root, env=env, timeout=args.timeout,
            )
            print(f"CLI {' '.join(argv[:2])}: passed", flush=True)
            return output

        if args.manager_only:
            shutil.copytree(source, installed, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
        else:
            # A local Git origin contains the exact worktree under review, including
            # uncommitted fixes, without ever touching the user's production snapshot.
            origin = root / "origin"
            shutil.copytree(source, origin, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
            git_env = dict(env, GIT_AUTHOR_NAME="Lifecycle test", GIT_COMMITTER_NAME="Lifecycle test",
                           GIT_AUTHOR_EMAIL="test@example.invalid", GIT_COMMITTER_EMAIL="test@example.invalid")

            def git(*argv):
                return command(["git", *argv], cwd=origin, env=git_env)

            git("init", "-q", "-b", "main")
            git("add", ".")
            git("commit", "-qm", "Lifecycle fixture from plugin worktree")
            initial_sha = git("rev-parse", "HEAD")
            identifier = args.remote_source or origin.as_uri()
            cli("install", identifier, "--no-enable", "--yes-deps", "--force")
            assert installed.is_dir(), "Hermes must install using manifest.name"
            record = json.loads(record_path.read_text(encoding="utf-8"))[PLUGIN_ID]
            expected_source = (f"https://github.com/{args.remote_source}.git"
                               if args.remote_source else origin.as_uri())
            expected_revision = args.expected_plugin_revision if args.remote_source else initial_sha
            assert record["source"] == expected_source, record
            assert record["revision"] == expected_revision and record["pinned"] is False, record
            report["manifest_id_and_provenance"] = "passed"
            if args.install_only:
                cli("doctor", PLUGIN_ID, "--ci")
                report["doctor_ci"] = "passed"
                report["installed_plugin_revision"] = record["revision"]
                report["source"] = record["source"]
                print(json.dumps(report, indent=2), flush=True)
                return

        manager = PluginManager()
        manager.discover_and_load()
        assert get_provider("mlx") is None, "Disabled discovery must not register mlx"
        assert registry.get_entry("mlx_image_generate") is None
        assert not any(str(getattr(m, "__file__", "") or "").startswith(str(installed))
                       for m in sys.modules.values()), "Disabled plugin must not be imported"
        report["passive_disabled_discovery"] = "passed"

        if args.manager_only:
            write_config(True)
        else:
            cli("enable", PLUGIN_ID)
        manager.discover_and_load(force=True)
        provider = get_provider("mlx")
        assert provider is not None, manager.list_plugins()
        assert provider.name == "mlx" and provider.is_available(), manager.list_plugins()
        assert registry.get_entry("mlx_image_generate") is not None
        assert provider.list_models()[0]["local_available"] is True
        assert manager.unload(PLUGIN_ID), "Public unload must find the plugin"
        assert get_provider("mlx") is None, "Hermes must release ctx-owned registration"
        assert registry.get_entry("mlx_image_generate") is None
        manager.discover_and_load(force=True)
        assert get_provider("mlx") is not None, manager.list_plugins()
        report["enable_and_public_unload"] = "passed"

        if args.manager_only:
            write_config(False)
        else:
            cli("disable", PLUGIN_ID)
        manager.discover_and_load(force=True)
        assert get_provider("mlx") is None, "Disable + host reload must remove provider"
        report["disable_and_reload"] = "passed"

        if args.manager_only:
            write_config(True)
        else:
            cli("enable", PLUGIN_ID)
        manager.discover_and_load(force=True)
        assert get_provider("mlx") is not None, manager.list_plugins()
        report["re_enable"] = "passed"

        if not args.manager_only:
            with (origin / "README.md").open("a", encoding="utf-8") as handle:
                handle.write("\n<!-- temporary lifecycle update fixture -->\n")
            git("add", "README.md")
            git("commit", "-qm", "Temporary lifecycle update fixture")
            updated_sha = git("rev-parse", "HEAD")
            cli("update", PLUGIN_ID, "--force")
            record = json.loads(record_path.read_text(encoding="utf-8"))[PLUGIN_ID]
            assert record["revision"] == updated_sha and record["source"] == origin.as_uri(), record
            manager.discover_and_load(force=True)
            assert get_provider("mlx") is not None, manager.list_plugins()
            report["update_provenance_and_reload"] = "passed"

        if args.manager_only:
            shutil.rmtree(installed)
            write_config(False)
        else:
            cli("remove", PLUGIN_ID)
            assert not installed.exists()
            assert PLUGIN_ID not in json.loads(record_path.read_text(encoding="utf-8"))
            saved = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
            assert PLUGIN_ID not in saved["plugins"].get("enabled", [])
        # A separate CLI cannot unload this process's registry. The official
        # manager's reload is the host lifecycle boundary, also used by gateways.
        report["provider_present_before_remove_reload"] = get_provider("mlx") is not None
        manager.discover_and_load(force=True)
        assert get_provider("mlx") is None, "Removed plugin must be absent after host reload"
        assert registry.get_entry("mlx_image_generate") is None
        assert sentinel.read_text(encoding="utf-8") == "external assets survive plugin removal"
        assert wrapper.is_file() and model.is_dir()
        report["remove_and_external_assets"] = "passed"
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--manager-only", action="store_true")
    parser.add_argument("--install-only", action="store_true", help="Install disabled, verify provenance, and run Doctor; no core dependency sync")
    parser.add_argument("--remote-source", help="owner/repo to install instead of a temporary local Git origin; requires --install-only")
    parser.add_argument("--expected-plugin-revision", help="Expected exact remote Git SHA")
    args = parser.parse_args()
    if args.manager_only and args.install_only:
        parser.error("--manager-only and --install-only are mutually exclusive")
    if args.remote_source and (not args.install_only or not args.expected_plugin_revision):
        parser.error("--remote-source requires --install-only and --expected-plugin-revision")
    run(args)
