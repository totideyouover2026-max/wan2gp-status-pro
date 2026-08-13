# Status Pro v1.0.1

Status Pro 1.0.1 is a focused usability update for users who want the live stage-based dashboard without a generation ledger, and for faster review of detailed History records. It observes WanGP's existing generation process and does not change model output or generation behaviour.

## What's new

- **Do not record new runs** can be selected from History settings. Live stages, elapsed time, ETA, downloads, and current performance continue normally, but newly completed, aborted, and failed runs are not added to History.
- Existing records remain available for review, export, import, gallery actions, or clearing while automatic recording is off.
- The top control reads **History off**, and prompt memory is paused until automatic recording is enabled again.
- Expanded History uses clearer, top-aligned field labels and a less crowded responsive grid.
- A measured **Observed timing composition** bar gives a quick stage-duration overview. Wall-clock time not covered by observed stages uses a theme-aware diagonal pattern instead of a fixed grey or theme accent.
- Step observations highlight the fastest and slowest valid Time value within each pass. Skipped steps are excluded.
- Visible LoRA names are reduced to filename-only labels without `.safetensors`, one per line; complete captured values remain available in tooltips and structured exports.

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
