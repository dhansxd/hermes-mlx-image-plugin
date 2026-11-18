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

Import/discovery does not register this provider or start inference. Only `register(ctx)` calls `ctx.register_image_gen_provider`; Hermes owns the registration lease and its release on unload/rediscovery. Disable before remove in a running gateway and follow Hermes's reload/restart instructions. A separately running process must reload to observe removal; the remove command itself does not cancel a call already running. Select another image backend after removal if `image_gen.provider` still points to `mlx`.

Disable/remove does not delete, move or modify the central Lemari runtime, model weights, voices or its standalone entrypoints. Other applications can continue using that same external runtime.

## Supported requests

- Text-to-image and one primary edit image via native `image_url`.
- Local absolute image paths checked through Hermes's public file-safety API; HTTP(S) images loaded through Hermes's SSRF-safe media helper; bounded PNG/JPEG/WebP/GIF base64 image data URLs, including sandbox inputs materialized by Hermes. Input files are capped at 25 MiB and checked for a supported image header. Temporary URL/data inputs are removed after the call; caller-owned local inputs are kept.
- Additional `reference_image_urls` are unsupported and return an error rather than being silently discarded. Upscaling and creative controls are not advertised.
- Legacy pixel sizing via a prompt marker such as `[size=512]` or `[resolution=1024]`. Direct Python callers may also pass integer `resolution` / `size` kwargs. Sizes clamp to 256..1536 and align down to 16 pixels; aspect ratios are landscape, square and portrait. Steps remain 4 / 6 / 8 / 12 by size.
- Stable Hermes's native creative resolution control has `1K` / `2K` semantics. This provider does not advertise that control or silently translate it into a different size. Explicit unsupported resolution values fail; unrelated future kwargs are ignored as required by the ABC.

## Central runtime contract

| Model key | Accepted aliases | Local subdirectory | MFlux base model | Quantize |
|---|---|---|---|---|
| flux2-klein-4b-4bit | flux2-klein-4b; mlx-community/FLUX.2-Klein-4B-4bit | flux2-klein-4b-4bit | flux2-klein-4b | 4 |

Aliases are case-insensitive. The local subdirectory is relative to `models_dir`; generation passes its resolved existing local path to `--model`, never the HF ID. Missing directories fail without a download. Model contents/tokenizer completeness must be prepared by the runtime owner.

The wrapper must accept MFlux arguments `--model --base-model --quantize --steps --prompt --output --width --height`; editing adds `--image-paths`. It must write a PNG at exactly the requested output path and run inference in its own PID or replace itself via exec. A custom wrapper that detaches children is outside the timeout/lock-release guarantee.

The child receives runtime essentials, the real account home, scoped `DYRA_LOCAL_AI_LOCK_TIMEOUT` when set, and forced `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`. Hermes Python variables and arbitrary inherited credentials are not passed to it. The parent environment stays unchanged. The 1800-second subprocess timeout kills/reaps the direct wrapper process; disable/unload does not promise immediate cancellation of an existing call. Queue/lock implementation remains the external runtime's responsibility.

Generated deliverables go to the active profile's official `cache/generated/images/` area, separate from weights and inbound media caches. Names include UUIDs. Failed/timeout attempts remove their own partial file; a successful output must be a regular non-symlink file with a PNG signature. This is a bounded header check, not a full image decode or a quality guarantee. Diagnostics omit private paths, prompts and raw subprocess output.

## Review and deferred testing

Version 1.0.2 was reviewed statically against original Hermes stable **0.23.2**, commit [ae2a81ccc00e9b7c06301523e5bb09608397149c](https://github.com/NousResearch/hermes-agent/commit/ae2a81ccc00e9b7c06301523e5bb09608397149c). Source syntax, metadata, public API usage and independent source review were checked in memory. No current runtime, CLI lifecycle or generation test has been executed for this release; no CI workflow was added.

Regression source in `tests/test_provider.py` and the opt-in real-host harness in `tests/test_lifecycle.py` are prepared for later execution. The harness is pinned to the reviewed stable commit and uses isolated homes when explicitly run. Historical v1.0.1 tests against a different upstream commit are not validation of v1.0.2 or this stable target.

Official references: [Plugins](https://hermes-agent.nousresearch.com/docs/user-guide/features/plugins), [developer guide](https://hermes-agent.nousresearch.com/docs/developer-guide/plugins), [image provider guide](https://hermes-agent.nousresearch.com/docs/developer-guide/image-gen-provider-plugin).

## License

MIT
