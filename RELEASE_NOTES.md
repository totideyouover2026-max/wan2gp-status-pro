# Status Pro v1.1.2

Status Pro 1.1.2 stabilises task-owned live stage timing across queued and separate generations. Task identity and execution epoch are bound before progress callbacks, so fast Encode activity is retained and timing from a previous task cannot create or extend stages in the next task.

Authoritative WanGP V13 state now takes precedence over stale native DOM status. Save is shown while saving, then yields cleanly to the completed presentation without flashing between the two views.

## Status Pro v1.1.1

Status Pro 1.1.1 is a focused download compatibility hotfix. WanGP V13 model, module, and LoRA downloads now pass every native argument through the Status observer unchanged, including progress generators and filename display settings. Older WanGP download calls remain supported.

YuE2 score and semantic-audio token updates now remain single, advancing activities instead of creating duplicate live and History entries. Token, tile, and layer counters remain phase-local rather than entering denoising performance, while genuine acoustic-synthesis steps remain recorded and YuE2 audio decoding appears under Decode.

## Status Pro v1.1.0

Status Pro 1.1.0 adds full support for WanGP V13's richer progress reporting while retaining compatibility with older WanGP releases.

## Live progress

- Native WanGP phase state now takes priority, with V13 WangpProgress and legacy progress markup retained as fallbacks.
- Text encoding can show layer progress, VAE encoding and decoding can show tile progress, and denoising continues to show step progress.
- These activities remain inside Prepare, Inputs, Encode, Generate, Decode, Enhance, and Save. Repeated prompts, passes, and windows remain individually visible.
- Genuine V13 Decode progress is displayed when available. Older blocking Decode operations still show elapsed activity without an invented percentage or ETA.
- Layer and tile callbacks remain separate from denoising step counts, average step time, cache-skip observations, and phase numbering.
- Cancellation, minimized-window recovery, Qwen Encode fallback, and Pro-over-Lite precedence remain supported.

## History enhancement

Expanded History now presents timing in three levels:

1. **Pipeline timing** shows all seven stages in order, using `—` for stages that did not occur.
2. **Observed timing composition** retains individual activities and their final layer, tile, or step counters where available.
3. **Step Observations** remains specific to genuine denoising and performance callbacks.

New runs can store the reported WanGP version, repeated phase activities, partial counters from failed or aborted runs, and concise Save activity labels while retaining the original output filename in output metadata. This richer telemetry survives Status Pro export and import. Older History entries and exports remain supported and may naturally contain less detail.

## Compatibility and privacy

- Older WanGP releases remain supported through the legacy progress fallback.
- No additional Python dependencies are required.
- History remains browser-local and retains the existing recording, retention, prompt-memory, and export privacy controls.
- Status Pro observes WanGP's generation process and does not alter generated output.

## Known limitations

- ETA appears only where the complete stage has measurable incremental progress; a layer or tile counter does not automatically imply a stage-level ETA.
- RAM and VRAM figures remain periodic observations rather than profiler traces.
- Browser polling can differ slightly from WanGP's internal terminal timings.

Install or update through WanGP's Plugin Manager, enable `wan2gp-status-pro`, save the plugin setting, and restart WanGP.
