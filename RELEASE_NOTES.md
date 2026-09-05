# Status Pro v1.0.6

Status Pro 1.0.6 improves queue completion, background timing, history reliability, and coexistence with Status Lite. It observes WanGP's existing generation process and does not change model output or generation behaviour.

## What's new since 1.0.5

- A stopped queue no longer revives an endless Prepare stage from lingering abort text or completed asset activity. Active downloads and model unloading remain visible.
- Temporary or incomplete telemetry snapshots no longer close an active generation's history record.
- Shared history reconciles changes and deletions across browser tabs, with serialized writes when the browser supports Web Locks. Tab-local history remains independent.
- Queue completion banners distinguish completed, failed, aborted, and incomplete runs.
- Per-step memory observations no longer cause periodic samples to be counted twice in memory averages.
- Missing or invalid export timestamps remain empty instead of becoming January 1970 dates.
- A run completed while the browser is minimized now ends at WanGP's reported queue completion or output creation time, excluding any later idle gap before the window is restored.
- Affected retained records are repaired from their output metadata when possible, and impossible background-inflated stage timings are labelled unreported.
- Gallery-applied LTX `Distilled refinement` now remains in Enhance and carries forward the preceding upsampling-start time after Inputs and Encode.
- When both Status editions are enabled, Status Pro takes precedence and Status Lite remains dormant. This prevents duplicate callback/download observers and competing status panels.
- Both editions resolve WanGP's native status component independently of plugin insertion order.

## Privacy and retention

History is stored in the browser and is never sent to an external service by Status Pro. Users can disable new recording or retain recorded runs:

- without recording new runs;
- until manually cleared;
- until the browser tab or app webview closes; or
- until WanGP restarts (the default).

Prompt memory is optional and page-session scoped. It is paused while automatic History recording is off. Prompt fields remain unchecked in exports by default, and prompts are not placed in the longer-lived WanGP-runtime or manually-cleared stores.

## Compatibility

- Tested with WanGP 12.452 and later. No hard minimum is declared because earlier versions may also be compatible, although they are not guaranteed.
- No additional required Python dependencies.
- Process-memory telemetry uses `psutil` when it is already available through WanGP and otherwise degrades gracefully.
- Detailed model-download telemetry is plugin-local and fails open: generation continues normally if WanGP or Hugging Face changes an observed internal interface.

## Known limitations

- Decode commonly runs as one blocking VAE operation. When WanGP provides no intermediate units, Status Pro shows elapsed activity without inventing a percentage or ETA.
- ETA is shown only for work with measurable incremental progress and may adjust during early steps or changing GPU load.
- RAM and VRAM figures are periodic observations rather than profiler traces; device-used VRAM can include other applications.
- Status Pro retains at most 100 history records and the latest 300 callback step observations per observed generation phase session when automatic recording is enabled.
- Browser polling can differ slightly from WanGP's internal terminal timings.
- Detailed download reporting depends on plugin-observed WanGP and Hugging Face activity; its absence does not affect downloading or generation.
- Import Media from an imported History JSON will add in the same manner as adding media without extended JSON details.  

## Installation

Install the public repository through WanGP's Plugin Manager, enable `wan2gp-status-pro`, save the plugin setting, and restart WanGP. Manual installation is also supported by placing the repository in `plugins/wan2gp-status-pro/`, enabling it, and restarting WanGP.

See [README.md](README.md) for installation and compatibility details and [USER_GUIDE.md](USER_GUIDE.md) for a visual walkthrough of the complete feature set.
