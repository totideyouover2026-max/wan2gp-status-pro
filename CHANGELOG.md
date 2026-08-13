# Changelog

All notable Status Pro changes will be recorded here. Versions follow Semantic Versioning.

## [1.0.1] - 2026-08-13

### Added

- **Do not record new runs** in History settings. Live stages, elapsed timing, ETA, and current performance remain active; completed runs are not added to the ledger while recording is off, and existing records remain unchanged.
- A measured timing-composition bar in expanded History, with consistent stage colours and a theme-aware striped segment for unaccounted wall time.
- Fastest and slowest valid step highlighting within each observed pass, excluding skipped observations.

### Changed

- Removed the hard WanGP version requirement from plugin metadata. A neutral compatibility baseline prevents WanGP from restoring an older cached requirement; WanGP 12.452 remains the tested baseline while earlier releases may still work.
- Expanded-record labels are clearer and aligned to the top for faster scanning across responsive layouts.
- Visible LoRA values use filename-only names without `.safetensors`, one per line, while complete captured values remain available in tooltips and exports.
- Added regression coverage for History recording preferences, timing composition, per-pass step outliers, and LoRA display formatting.

## [1.0.0] - 2026-08-12

### Added

- Persistent, responsive Prepare, Encode, Generate, and Decode timeline with conditional Inputs, Enhance, and Save stages.
- Queue-aware browser-local history with image, video, audio, sliding-window, aborted, and failed run handling.
- JSON, CSV, and Markdown exports with privacy-conscious defaults and named custom presets.
- Per-step timing, cache-skip observations, RAM/VRAM samples, gallery navigation, model component labels, and download telemetry.
- Model-agnostic Preloaded and Encode lifecycle handling, with exact Qwen Encode recovery where WanGP exposes trustworthy callback boundaries.
- A task-oriented user guide covering stages, timing, downloads, history, privacy, exports, workflows, and common questions.
- Model-agnostic phase routing for control/source preprocessing, input VAE work, prompt enhancement, generative audio, output decoding, and post-processing.

### Changed

- Decode remains deliberately indeterminate when WanGP provides no intermediate decoder progress.
- ETA is shown only for stages with measurable incremental progress.
- Browser-storage quota trimming now updates the visible ledger and presents an explicit warning.
- Generation history now offers manually-cleared, browser-tab, and WanGP-runtime lifetimes, with backend launch-ID detection for reliable clearing after a WanGP restart.
- Every history-lifetime change now displays a confirmation explaining the selected behaviour before any records are moved between stores.
- Half-width layouts now keep the selected stage expanded while collapsing other stage cards to accessible tick/number buttons, centre the second-row timing summary, and use deliberate header and history-toolbar layouts instead of incidental wrapping.
- Stage cards now keep their icon, label, and timing as a centred content cluster at both regular widths and for the expanded selection in compact mode.
- Prompt availability and privacy messaging now appears inside the Prompts field group instead of occupying a collision-prone row above the sticky export footer.
- Conditional Inputs cards now remain in timeline order and retain individual activity labels plus cumulative time when pipelines such as LTX alternate between input preparation and text encoding.
- Removed the Capture prompts toolbar option; optional page-session prompt memory now lives with the other privacy settings and prompts remain unchecked in exports by default.
- Multi-window and multi-run queue tasks now collapse into aggregate history rows with chronological children, tri-state selection, outcome priority, and wall-clock timing.
- Performance history now distinguishes configured steps from callback observations and summarizes multi-pass work such as MiniMax H3 Spectrum capture and replay.
- History now separates saved settings from execution: the modal uses Save/Cancel semantics, while the toolbar owns the direct Export action using the saved format and fields.
- Page-session prompt memory is now an explicit privacy setting in the Prompts group; disabling it removes currently held prompt text and prevents capture for future runs until re-enabled.
- The settings window uses a viewport-bounded height and a dedicated footer row so its bottom controls remain aligned and visible.
- The History toolbar now uses a fixed Export, Select all, clearing actions, and cog-settings order with visual separators, one normalized vertical control axis, and compact responsive behavior.
- Stage cards now remain attached during live refreshes and no longer move on hover, preventing pointer interruption, hover oscillation, and missed selections.
- History now labels Prepare-only timing as Model loading, omits empty list fields such as unused LoRAs, and uses actual per-pass observation counts when passes do not complete an equal configured-step total.
- Per-step telemetry now retains each phase's announced step total, so short secondary phases such as LTX's three-step distilled pass display `1/3` through `3/3` instead of inheriting the first pass's denominator. Existing completed LTX-style history is repaired when loaded.
- History settings now reserve extra scroll clearance above the footer and provide short native hover tooltips for every field group and export option.
- Generation history can now expand into a viewport-sized modal workspace. It reuses the live drawer so scope, selections, open task/run rows, exports, clearing, and gallery actions remain synchronized when returning to the embedded view.
- Status Pro JSON exports can now be imported into an empty history for review. Import refuses to merge or overwrite existing records, preserves available run/media/performance metadata and provenance, and applies existing prompt-retention rules.
- Imported history rows with retained output paths now offer an Import media action that safely restores still-existing output files to Wan2GP's native video/image or audio gallery, confirms success, and clearly reports moved or deleted files.

### Reliability

- WanGP and Hugging Face download observers now fail open: an incompatible or unavailable download API disables only download detail instead of preventing Status Pro from loading.

## [0.7.7] - 2026-08-10

- Final pre-v1 development build.
