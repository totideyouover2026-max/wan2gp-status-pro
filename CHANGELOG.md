# Changelog

All notable Status Pro changes will be recorded here. Versions follow Semantic Versioning.

## [1.0.6] - 2026-09-05

### Fixed

- Temporary or incomplete telemetry snapshots no longer close an active generation's history record.
- Shared history reconciles changes and deletions across browser tabs, with serialized writes when the browser supports Web Locks. Tab-local history remains independent.
- Queue completion banners distinguish completed, failed, aborted, and incomplete runs.
- Per-step memory observations no longer cause periodic samples to be counted twice in memory averages.
- Missing or invalid export timestamps remain empty instead of becoming January 1970 dates.
- A stopped queue no longer revives a Prepare stage from lingering abort text or completed asset activity.
- Runs that finish while the WanGP window is minimized now use WanGP's reported queue duration or the generated output's creation timestamp, rather than treating the later browser-resume time as completion. The idle minimized gap no longer inflates History's Completed and Total time fields or the final observed stage.
- Previously retained affected records are repaired when loaded when their output metadata provides an authoritative completion time. Any stage duration made impossible by the old resume-time stamp is shown as unreported instead of retained as false work.
- Gallery-applied LTX 2/2.5 `Distilled refinement` progress is now classified as Enhance instead of falling through to Prepare. Enhance timing resumes from the earlier upsampling-start phase after the workflow's Inputs and Encode stages.
- Status Pro and Status Lite can now be installed together safely. When both are enabled, Status Pro deterministically takes precedence and Lite does not install duplicate backend observers or insert a competing panel.
- Status panels now locate WanGP's native `gen_status` component directly, with a sibling fallback that skips either plugin container, so plugin insertion order cannot make one panel observe the other.

## [1.0.5] - 2026-08-31

### Fixed

- Standalone LTX 2.3/2.5 pixel-spatial upscaling records now identify the model-backed upsampler actually loaded instead of inheriting the generation model selected in WanGP. Existing retained and imported records are repaired when loaded, including their UI summary and exported model fields.
- Live LTX upscaling stages now use the LTX backing component list, so Inputs/Decode identify the resolved LTX VAEs and Encode identifies the resolved Gemma text encoder. This applies to standalone gallery tasks and the upscaling portion of inline runs; stale generation-model components are suppressed when exact LTX files are unavailable.
- Standalone gallery processing now treats every internal LTX subwindow as part of the same processor task and keeps LTX component identity across Prepare, Inputs, Encode, and Decode. Exact WanGP-resolved filenames are preferred; accurate LTX/Gemma/VAE role labels are used when a model definition omits a filename, never the selected generation model's H3/Qwen components.
- LTX gallery and inline upscaling now capture the registered processor's own progress callbacks, retain the final eighth refinement observation across WanGP task teardown, and record eight configured steps per window instead of reusing stale generation-step telemetry.
- Performance observers are now bound to their WanGP queue task, preventing a preceding task's final callback from appearing as the next run's only step observation; retained legacy rows whose observer clearly predates the run are repaired when loaded.
- History now rejects an inherited WanGP `generation_time` when it is impossible for the observed run's wall-clock duration, while retaining plausible reported timings.
- Live Generate and LTX refinement stages now recover missed timing, step, and phase details from retained server-side callbacks after a minimized or backgrounded WanGP window resumes. Visibility/focus changes trigger an immediate resynchronization, and phases not exposed while backgrounded are labelled as unreported instead of appearing not to have run.
- Retained gallery-LTX records discard legacy step rows that cannot be attributed to an upscaler phase; Status Pro does not relabel or synthesize observations that the older telemetry did not capture.
- Post-processing is now a distinct structured history field. Inline temporal/spatial upsampling and film grain preserve the generation model and add a `Post:` summary; gallery late-processing tasks use a processor-led summary, and expanded details identify whether processing ran with generation or as a separate task.

## [1.0.4] - 2026-08-24

### Added

- Step observation tables now offer a compact **Hide skipped** switch for reviewing only observations where work was performed; filtering is display-only and does not alter retained or exported telemetry.

## [1.0.3] - 2026-08-23

### Added

- History now records the effective WanGP attention mode, including global, model-specific, and per-generation choices. Sol Attention runs also retain their tau (`attention_sparsity`) level, and both values are available in History details and exports.
- The History settings modal now uses a larger viewport-bounded workspace and independent checkbox-panel columns, reducing empty grid space and keeping more settings visible above the footer.

## [1.0.1] - 2026-08-13

### Added

- **Do not record new runs** in History settings. Live stages, elapsed timing, ETA, and current performance remain active; completed runs are not added to the ledger while recording is off, and existing records remain unchanged.
- A measured timing-composition bar in expanded History, with consistent stage colours and a theme-aware striped segment for unaccounted wall time.
- Fastest and slowest valid step highlighting within each observed pass, excluding skipped observations.

### Changed

- Removed the hard WanGP version requirement from plugin metadata. A neutral compatibility baseline prevents WanGP from restoring an older cached requirement; WanGP 12.452 remains the tested baseline while earlier releases may still work.
- Expanded-record labels are clearer and aligned to the top for faster scanning across responsive layouts.
- Visible LoRA values use filename-only names without `.safetensors`, one per line, while complete captured values remain available in tooltips and exports.
- The completed top bar now reports only the latest generation duration. Session counts, cumulative duration, and the latest finished-at time remain in the expanded summary.
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
