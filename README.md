# hermes-mlx-image-plugin

Local MLX image-generation backend plugin for [Hermes Agent](https://hermes-agent.nousresearch.com) that routes `image_generate` through a local `mflux` install on Apple Silicon.

Generic by design: the provider name is `mlx` (not a model name). Models live in one local cupboard and are selected via config — adding a new model later is one row in `MODEL_TABLE`, not a new plugin.

## What it does

- Text-to-image + local image-to-image edit (`image_url` → `--image-paths`)
- Per-request resolution (`resolution`/`size` kwarg or `[size=NNN]` prompt marker, 256..1536, 16-aligned)
- Aspect-ratio aware dimensions (`landscape` | `square` | `portrait`)
- Quality step schedule by size (4 / 6 / 8 / 12)
- Runs `~/.local/bin/dyra-image` with an existing, resolved local model directory and Hugging Face offline mode; missing weights fail without downloading
- Unique output filenames, a 30-minute subprocess deadline, and errors/logs that omit prompts, private paths and raw runtime output

## Current model table (1 entry for now)

| key | HF id | local subdir | base-model | quantize |
|---|---|---|---|---|
| `flux2-klein-4b-4bit` | `mlx-community/FLUX.2-Klein-4B-4bit` | `flux2-klein-4b-4bit` | `flux2-klein-4b` | `4` |

The local subdirectory is relative to `models_dir` (default `~/.local/share/dyra-ai/models/image`).

Aliases accepted for `model`: `flux2-klein-4b-4bit`, `flux2-klein-4b`, `mlx-community/FLUX.2-Klein-4B-4bit` (case-insensitive).

To add a model later: append one dict row + alias lines. No other code changes needed.

## Install

First install the external Lemari/Dyra image runtime, executable wrapper and model
weights on Apple Silicon. Hermes does not install MFlux, MLX, models or the wrapper:
`python_runtime: external` keeps this plugin's packaging out of Hermes' Python
dependency environment.

Install through Hermes, then enable when ready:

```bash
hermes plugins install dhansxd/hermes-mlx-image-plugin --no-enable --yes-deps
hermes plugins enable mlx-image-gen
hermes plugins doctor mlx-image-gen --ci
```

Git installs include a root `__init__.py` that imports the implementation from
`src/hermes_mlx_image`. No manual file copies or symlinks are required.
The manifest ID is **`mlx-image-gen`**; the provider selected in `image_gen.provider`
is **`mlx`**. Hermes publishes the Git install under
`$HERMES_HOME/plugins/mlx-image-gen/` and records its source/revision in
`plugins/.install-metadata.json`. Use the manifest ID for every lifecycle command,
even though the GitHub repository has a different name.

```bash
hermes plugins disable mlx-image-gen
hermes plugins enable mlx-image-gen
hermes plugins update mlx-image-gen
hermes plugins disable mlx-image-gen
hermes plugins remove mlx-image-gen
```

Hermes discovers metadata without activating this backend. Only Hermes calls
`register(ctx)` after enablement; `ctx.register_image_gen_provider` gives Hermes
ownership of the registration and cleanup on unload/rediscovery. There are no
plugin install/uninstall hooks or registry mutations at import time. Disable
before removing from a running gateway: the tested host's remove command deletes
the install/config record but does not itself reload an already-running manager.
Use the host's reload/restart if its disable output requests it.
After removal, select another `image_gen.provider` if it still points to `mlx`;
Hermes does not choose a replacement image backend for you.

An unpinned Git install updates through its saved source. To pin a reproducible
version, add `--ref <full-40-character-commit-SHA>` to install; Hermes requires an
explicit forced install at a new SHA to move that pin. Local edits and update
rollback are managed by Hermes, not by this plugin. A plain copied folder lacks
Git install provenance: use the manager install flow for tracked updates.

The optional wheel uses the same `mlx-image-gen` entry-point ID. Choose one
installation method; remove any old `hermes_mlx_image` pip alias with
`pip uninstall hermes-mlx-image` in the environment where it was installed before
using the Git route. Hermes' directory `remove` does not uninstall pip packages.

The external wrapper keeps its own Python runtime. Before launching it, the
plugin removes inherited `PYTHONPATH`, `PYTHONHOME`, and `VIRTUAL_ENV` from the
child environment, preventing Hermes's Python packages from shadowing the MLX
runtime. The child also receives `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`; the parent environment stays unchanged. A custom wrapper
must run inference in its own process (or replace itself with it), so the timeout
can terminate inference and release its shared lock.

## Configure

`~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - mlx-image-gen

image_gen:
  provider: mlx
  mlx:
    model: flux2-klein-4b-4bit
    models_dir: ~/.local/share/dyra-ai/models/image   # optional override
    wrapper: ~/.local/bin/dyra-image                  # optional override
```

Env overrides (optional, config wins):

```
MLX_IMAGE_MODEL=flux2-klein-4b-4bit
MLX_MODELS_DIR=~/.local/share/dyra-ai/models/image
MLX_WRAPPER=~/.local/bin/dyra-image
```

No secrets needed. Model weights stay in the local cupboard (`~/.local/share/dyra-ai/`), never in this repo and never in `~/.hermes/`.

## Local cupboard contract (the "1 lemari" rule)

- Canonical entrypoints: `dyra-image`, `dyra-tts`, `dyra-stt`, `dyra-f5-tts`
- This plugin only shells out to the wrapper. Runtime/venv/model ownership stays outside Hermes.
- Wiping `~/.hermes/` total must not touch `~/.local/`.
- Disable/remove only changes the Hermes plugin and its activation record. Lemari's image, speech and voice assets remain usable independently.

## Test

```
python3 tests/test_provider.py
```

Offline provider tests use temporary fixtures and stubbed Hermes imports. A real
fake executable also verifies timeout termination and lock release; no model or
GPU is needed.

The lifecycle integration check uses real Hermes source and an isolated home:

```bash
python3 tests/test_lifecycle.py --hermes-root /path/to/hermes-agent --work-dir /path/to/scratch --timeout 300
```

It exercises the official CLI's Git install, enable, disable, re-enable, update,
and remove, plus actual provider ownership/cleanup and external asset survival.
Use a Python environment with the Hermes CLI dependencies; enabling can provision
the host's core runtime in the temporary home. `--manager-only` checks real
discovery/registry cleanup without claiming a CLI or dependency-admission test.
See `tests/test_lifecycle.py --help` for remote install verification. The contract
was checked against upstream
[`065f5abbb3c55d849ed7177ad5c1f231074b849a`](https://github.com/NousResearch/hermes-agent/commit/065f5abbb3c55d849ed7177ad5c1f231074b849a),
not an unverified minimum Hermes release. Official references:
[Plugins](https://hermes-agent.nousresearch.com/docs/user-guide/features/plugins),
[plugin developer guide](https://hermes-agent.nousresearch.com/docs/developer-guide/plugins),
and [image provider guide](https://hermes-agent.nousresearch.com/docs/developer-guide/image-gen-provider-plugin).

## License

MIT
