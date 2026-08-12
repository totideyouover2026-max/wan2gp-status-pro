# Status Pro v1.0.0

Status Pro gives WanGP a responsive generation timeline, trustworthy live timing, and a privacy-controlled local history ledger. It observes WanGP's existing generation process and does not change model output or generation behaviour.

## Highlights

- Selectable Prepare, Inputs, Encode, Generate, Decode, Enhance, and Save stages, with optional stages shown only when used.
- Live elapsed time, stabilized denoising and enhancement ETA, and smoothed time per step where WanGP exposes measurable progress.
- Model loading, preloaded-model, model-unloading, and model-download detail inside the same Status Pro interface.
- Concise transformer, text-encoder, input-VAE, and output-VAE filenames without full local paths or download URLs in the UI.
- Responsive stage cards, a compact collapsed header, and a narrow-screen layout that keeps the selected stage readable.
- Queue-aware History with completed, sliding-window, incomplete, aborted, and failed outcomes.
- Collapsed task groups for multi-window and repeated runs, with chronological child records and tri-state selection.
- Per-stage timings, per-step performance, cache-skip observations, and sampled RAM/VRAM summaries.
- Direct navigation to matching WanGP gallery items without scrolling or autoplay.
- A large synchronized History workspace for reviewing busy generation ledgers.

## Export and import

- JSON, CSV, and Markdown exports.
- Standard, Performance, Reproducibility, and Share-safe presets, plus multiple named custom presets.
- Field-by-field export controls with concise hover help.
- JSON history import into an empty ledger, preserving available provenance and recorded metadata.
- Imported outputs that still exist inside WanGP's configured output folders can be restored to the native video/image or audio gallery.

## Privacy and retention

History is stored in the browser and is never sent to an external service by Status Pro. Users can choose to retain it:

- until manually cleared;
- until the browser tab or app webview closes; or
- until WanGP restarts (the default).

Prompt memory is optional and page-session scoped. Prompt fields are unchecked in exports by default, and prompts are not placed in the longer-lived WanGP-runtime or manually-cleared stores. Share-safe exports exclude prompts, local output paths, exact checkpoints, complete settings objects, and browser/session identifiers.

## Compatibility

- Requires WanGP 12.452 or later.
- No additional required Python dependencies.
- Process-memory telemetry uses `psutil` when it is already available through WanGP and otherwise degrades gracefully.
- Detailed model-download telemetry is plugin-local and fails open: generation continues normally if WanGP or Hugging Face changes an observed internal interface.

## Known limitations

- Decode commonly runs as one blocking VAE operation. When WanGP provides no intermediate units, Status Pro shows elapsed activity without inventing a percentage or ETA.
- ETA is shown only for work with measurable incremental progress and may adjust during early steps or changing GPU load.
- RAM and VRAM figures are periodic observations rather than profiler traces; device-used VRAM can include other applications.
- Status Pro retains at most 100 history records and the latest 300 callback step observations per observed generation phase session.
- Browser polling can differ slightly from WanGP's internal terminal timings.
- Detailed download reporting depends on plugin-observed WanGP and Hugging Face activity; its absence does not affect downloading or generation.

## Installation

Install the public repository through WanGP's Plugin Manager, enable `wan2gp-status-pro`, save the plugin setting, and restart WanGP. Manual installation is also supported by placing the repository in `plugins/wan2gp-status-pro/`, enabling it, and restarting WanGP.

See [README.md](README.md) for installation and compatibility details and [USER_GUIDE.md](USER_GUIDE.md) for a visual walkthrough of the complete feature set.
