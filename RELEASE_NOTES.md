# Status Pro v1.0.6

Status Pro 1.0.6 makes Status Pro and Status Lite safe to install together while retaining the post-processing accuracy improvements from 1.0.5. It observes WanGP's existing generation process and does not change model output or generation behaviour.

## What's new

- When both Status editions are enabled, Status Pro takes precedence and Status Lite remains dormant. This prevents duplicate callback/download observers and competing status panels.
- Both editions resolve WanGP's native status component independently of plugin insertion order.
- Standalone gallery post-processing records the LTX 2.3 or 2.5 upscaler as the effective model instead of inheriting the generation model selected in WanGP.
- Inline post-processing preserves the original generation model and adds a separate `Post:` summary describing temporal upsampling, spatial upscaling, and film grain.
- Live LTX stages identify the resolved LTX transformer, Gemma text encoder, and LTX video/audio VAEs. Accurate role labels replace stale H3/Qwen details if WanGP does not expose an exact filename.
- LTX's own progress callbacks provide all eight refinement observations per internal subwindow. Subwindows remain labelled phases within one task, such as `Window 1 / 2` and `Window 2 / 2`.
- Step observers are bound to their WanGP queue task, preventing the previous task's final callback from leaking into the next record.
- Impossible inherited generation-time values are discarded, while plausible WanGP timing remains available.
- Generate and LTX refinement details missed while the WanGP window is minimized are recovered from retained server-side callbacks when the page resumes or the run completes.
- Existing retained and imported gallery-LTX records are normalized when loaded, including model identity, component metadata, step totals, and structured post-processing details.

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
