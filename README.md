# hermes-mlx-image-plugin

A selectable local image provider for [Hermes Agent](https://hermes-agent.nousresearch.com) on Apple Silicon. Hermes calls an external MFlux wrapper; it does not own or install the MLX runtime or weights.

The manager ID is **mlx-image-gen**, the native image provider is **mlx**, and the optional pip entry-point ID remains **mlx-image-gen**. Models are entries in one table, not separate providers.

## Install, enable and select

Prepare the external runtime, wrapper and local model separately. This plugin downloads no models, installs no MLX dependencies and changes no Hermes core files. The manifest uses `python_runtime: external`.

Use the official manager in the intended Hermes profile:

```bash
hermes plugins install dhansxd/hermes-mlx-image-plugin --no-enable --yes-deps
hermes plugins enable mlx-image-gen
hermes tools
```

In **Image Generation**, select **MLX (local)** and its model. Enabling the plugin registers an option; it does not replace the selected image backend automatically. The equivalent config in the active profile's `config.yaml` is:

```yaml
image_gen:
  provider: mlx
  model: flux2-klein-4b-4bit
  mlx:
    # Optional provider default when no explicit/native model is passed:
    model: flux2-klein-4b-4bit
    models_dir: ~/.local/share/dyra-ai/models/image
    wrapper: ~/.local/bin/dyra-image
```

Use absolute paths or `~/` paths on other machines. Tilde defaults resolve against the OS account home at call time, so Hermes profile homes do not create a second model cupboard. The wrapper and models may be placed elsewhere via config; existing local model symlinks are resolved rather than copied.

Optional compatibility overrides are `MLX_IMAGE_MODEL`, `MLX_MODELS_DIR` and `MLX_WRAPPER`. Config wins over these overrides, which are read through Hermes's active profile scope. Prefer YAML for non-secret settings. Explicit models passed by the native picker/API win over the provider default; unknown models and malformed config fail clearly instead of generating with another model.

Git installs use the root `__init__.py` shim. Hermes publishes the tree under `$HERMES_HOME/plugins/mlx-image-gen/` and owns its source/revision metadata. Do not install both a Git directory and a pip package. Older pip installations with the `hermes_mlx_image` alias need to be removed using `pip uninstall hermes-mlx-image` in their owning environment; directory removal does not uninstall pip distributions.

## Update, disable and remove

```bash
hermes plugins update mlx-image-gen
hermes plugins disable mlx-image-gen
hermes plugins enable mlx-image-gen
hermes plugins disable mlx-image-gen
hermes plugins remove mlx-image-gen
```

All commands use the manifest ID, not the repository name. An unpinned Git install follows its recorded source. For a reproducible install, use `--ref <full-40-character-commit-SHA>`; move an existing pin with an explicit forced install at the new SHA. The official manager owns update admission, backups and rollback.

Importing the plugin starts no inference. Hermes owns registration and release of the provider, its optional tool and the native image-tool hook. Disable before remove in a running gateway and follow Hermes's reload/restart instructions. A separately running process must reload to observe removal; the remove command itself does not cancel a call already running. Select another image backend after removal if `image_gen.provider` still points to `mlx`.

Disable/remove does not delete, move or modify the central Lemari runtime, model weights, voices or its standalone entrypoints. Other applications can continue using that same external runtime.

## Supported requests

- Text-to-image and one primary edit image via native `image_url`.
- Local absolute image paths checked through Hermes's public file-safety API; HTTP(S) images loaded through Hermes's SSRF-safe media helper; bounded PNG/JPEG/WebP/GIF base64 image data URLs, including sandbox inputs materialized by Hermes. Input files are capped at 25 MiB and checked for a supported image header. Temporary URL/data inputs are removed after the call; caller-owned local inputs are kept.
- Additional `reference_image_urls` are unsupported and return an error rather than being silently discarded. Upscaling and creative controls are not advertised.
- The native tool still accepts a prompt marker such as `[size=512]` or `[resolution=1024]`. Direct Python callers may also pass integer `resolution` / `size` kwargs. Legacy sizes clamp to 256–1536 and align down to 16 pixels. With no preset or step setting, the legacy defaults are a 256 px long side and 4 / 6 / 8 / 12 steps by size.
- Stable Hermes's native creative resolution control has `1K` / `2K` semantics. This provider does not advertise that control or silently translate it into a different size. Explicit unsupported resolution values fail; unrelated future kwargs are ignored as required by the ABC.

## Presets, one-time choices, and locks

The optional `mlx_image_generate` tool accepts structured `aspect_ratio`, `resolution` (longest side in pixels), `steps`, and `preset` for a new image. It is available with the image toolset when MLX is selected. Native `image_generate` remains available for standard generation and editing. Both routes report the actual `width`, `height`, and `steps` and use the same runner.

Store named presets in the shared Lemari. The default path is `~/.local/share/dyra-ai/production/image/presets.json`. For example:

```json
{
  "draft": {"resolution": 512, "steps": 4},
  "daily": {"resolution": 768, "steps": 4},
  "detail": {"resolution": 1024, "steps": 4}
}
```

Select the defaults and locked values for the active Hermes profile under `image_gen.mlx`:

```yaml
image_gen:
  provider: mlx
  mlx:
    preset: daily
    # presets_file: ~/.local/share/dyra-ai/production/image/presets.json
    defaults:
      aspect_ratio: portrait
    locked:
      resolution: 768
```

A profile default overrides the named default preset for its field. A one-time preset or explicit argument then overrides those defaults for this image, with explicit fields taking precedence over a one-time preset. Locked values apply to omitted fields. An explicit value that conflicts with a lock returns a clear error. Remove the field from `locked` to unlock it. The image tool never changes these settings.

Structured resolutions are 256–1536 px in multiples of 16. Steps are 1–12. Ratios are `landscape` (16:9), `portrait` (9:16), and `square` (1:1). The short side is aligned for the runtime. Existing installations can keep using the plugin without a preset file by leaving `preset` unset.

## Central runtime contract

| Model key | Accepted aliases | Local subdirectory | MFlux base model | Quantize |
|---|---|---|---|---|
| flux2-klein-4b-4bit | flux2-klein-4b; mlx-community/FLUX.2-Klein-4B-4bit | flux2-klein-4b-4bit | flux2-klein-4b | 4 |

Aliases are case-insensitive. The local subdirectory is relative to `models_dir`; generation passes its resolved existing local path to `--model`, never the HF ID. Missing directories fail without a download. Model contents/tokenizer completeness must be prepared by the runtime owner.

The wrapper must accept MFlux arguments `--model --base-model --quantize --steps --prompt --output --width --height`; editing adds `--image-paths`. It must write a PNG at exactly the requested output path and run inference in its own PID or replace itself via exec. A custom wrapper that detaches children is outside the timeout/lock-release guarantee.

The child receives runtime essentials, the real account home, scoped `DYRA_LOCAL_AI_LOCK_TIMEOUT` when set, and forced `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`. Hermes Python variables and arbitrary inherited credentials are not passed to it. The parent environment stays unchanged. The 1800-second subprocess timeout kills/reaps the direct wrapper process; disable/unload does not promise immediate cancellation of an existing call. Queue/lock implementation remains the external runtime's responsibility.

Generated deliverables go to the active profile's official `cache/generated/images/` area, separate from weights and inbound media caches. Names include UUIDs. Failed/timeout attempts remove their own partial file; a successful output must be a regular non-symlink file with a PNG signature. This is a bounded header check, not a full image decode or a quality guarantee. Diagnostics omit private paths, prompts and raw subprocess output.

## Verification

The provider regression suite covers the wrapper arguments, defaults, per-request options, lock conflicts, and media cleanup using temporary fixtures. Run it with `python3 -m unittest discover -s tests -p test_provider.py`.

The opt-in lifecycle harness in `tests/test_lifecycle.py` uses isolated homes and the official install, enable, disable, update, and remove commands. It is pinned to Hermes stable **0.23.2**, commit [ae2a81ccc00e9b7c06301523e5bb09608397149c](https://github.com/NousResearch/hermes-agent/commit/ae2a81ccc00e9b7c06301523e5bb09608397149c).

Official references: [Plugins](https://hermes-agent.nousresearch.com/docs/user-guide/features/plugins), [developer guide](https://hermes-agent.nousresearch.com/docs/developer-guide/plugins), [image provider guide](https://hermes-agent.nousresearch.com/docs/developer-guide/image-gen-provider-plugin).

## License

MIT
