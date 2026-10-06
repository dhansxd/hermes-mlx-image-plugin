# hermes-mlx-image-plugin

Local MLX image-generation backend plugin for [Hermes Agent](https://hermes-agent.nousresearch.com) that routes `image_generate` through a local `mflux` install on Apple Silicon.

Generic by design: the provider name is `mlx` (not a model name). Models live in one local cupboard and are selected via config — adding a new model later is one row in `MODEL_TABLE`, not a new plugin.

## What it does

- Text-to-image + image-to-image edit (`image_url` → `--image-path`)
- Per-request resolution (`resolution`/`size` kwarg or `[size=NNN]` prompt marker, 256..1536, 16-aligned)
- Aspect-ratio aware dimensions (`landscape` | `square` | `portrait`)
- Quality step schedule by size (4 / 6 / 8 / 12)
- Zero cloud calls — runs `~/.local/bin/dyra-image` (the single cupboard entrypoint), which rewrites `--model` to the cached local path when available

## Current model table (1 entry for now)

| key | HF id | local subdir | base-model | quantize |
|---|---|---|---|---|
| `flux2-klein-4b-4bit` | `mlx-community/FLUX.2-Klein-4B-4bit` | `models/image/flux2-klein-4b-4bit` | `flux2-klein-4b` | `4` |

Aliases accepted for `model`: `flux2-klein-4b-4bit`, `flux2-klein-4b`, `mlx-community/FLUX.2-Klein-4B-4bit` (case-insensitive).

To add a model later: append one dict row + alias lines. No other code changes needed.

## Install

Install and enable through Hermes:

```bash
hermes plugins install dhansxd/hermes-mlx-image-plugin --enable --yes-deps
hermes plugins doctor mlx-image-gen --ci
```

Git installs include a root `__init__.py` that imports the implementation from
`src/hermes_mlx_image`. No manual file copies or symlinks are required.
Alternatively, copy the complete repository into `$HERMES_HOME/plugins/mlx-image-gen/`
and enable `mlx-image-gen`. Do not enable the pip entry-point alias as well.

The external wrapper keeps its own Python runtime. Before launching it, the
plugin removes inherited `PYTHONPATH`, `PYTHONHOME`, and `VIRTUAL_ENV` from the
child environment, preventing Hermes's Python packages from shadowing the MLX
runtime. Other environment variables and the parent process remain unchanged.

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

## Test

```
python3 tests/test_provider.py
```

Offline unit tests with monkeypatched `subprocess` + stubbed Hermes imports — no GPU/gateway needed.

## License

MIT
