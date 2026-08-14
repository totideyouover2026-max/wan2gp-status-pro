# Status Pro User Guide

Status Pro gives WanGP a clearer view of what a generation is doing, how long it has taken, and what happened during earlier runs. It also provides a browser-local history ledger with configurable exports and optional prompt memory.

You do not need to configure anything before using it. Start a generation normally and Status Pro will follow its progress automatically.

Jump to: [stages](#what-the-stages-mean) · [timing](#understanding-the-timing-information) · [downloads](#model-loading-and-downloads) · [History](#history) · [storage and privacy](#history-storage-and-privacy) · [export and import](#exporting-generation-data) · [common questions](#common-questions)

## Quick tour

![Status Pro during generation](<Screenshots/1- Generate (ETA available after 1st window - If using step skipping wait until the percentage selected for a more accurate eta).png>)

The main display has four parts:

1. **Top bar** — shows the current activity, completed steps, total elapsed time, ETA, and History button.
2. **Stage cards** — show where the generation is in the overall process.
3. **Detail panel** — gives more information about the selected stage.
4. **Progress line** — provides a quick visual indication of measurable progress.

Select any stage card to inspect it. Selecting a card does not interrupt or change the generation.

Use the **▼** button to collapse Status Pro into a compact top bar. Select it again to restore the full display. Status Pro remembers this choice in the browser.

At narrower window widths, the current stage remains expanded while inactive stages reduce to their tick or number icons. This keeps the active information readable without shrinking all text. Select a compact stage icon to inspect that stage; the newly selected card expands in place.

![Status Pro at a narrower window width](<Screenshots/12 - Narrow View (half monitor or small screens).png>)

## Before and after a generation

Status Pro remains useful when nothing is currently running:

- **Before the first run**, it displays **Ready to generate** and explains that timing and settings will appear after generation.
- **After a queue completes**, the top bar shows the most recent generation task's full duration, including all of its sliding windows. The expanded summary shows the current session's generation count, combined recorded time, and the clock time when the latest generation finished.
- **When retained history comes from an earlier session**, the ready view points you to the saved records in History without treating them as new runs.

The History drawer can remain open before, during, and after generation.

![Status Pro ready before the first generation](<Screenshots/10 - Ready to generate.png>)

## What the stages mean

The stages shown depend on the model and options being used. Optional stages appear only when WanGP reaches them.

| Stage | What is happening |
| --- | --- |
| **Prepare** | WanGP is loading, downloading, changing, or preparing the main model. **Preloaded** means the required model was already available. Model unloading is also shown here when switching models. |
| **Inputs** | Optional source or control media is being prepared—for example control-video VAE conversion, pose/depth/face extraction, background removal, resizing, or similar preprocessing. This card appears only when WanGP reports such work. |
| **Encode** | Prompts and semantic conditioning are being converted into information the model can use. **Not reported** means WanGP did not expose this as a separately measurable stage; it does not mean encoding failed. |
| **Generate** | The main denoising or sampling work is running. This usually provides steps, progress, average step time, and an ETA. |
| **Decode** | The generated data is being converted into an image, video, or audio output. WanGP usually performs this as one blocking operation, so Status Pro shows elapsed activity rather than inventing a percentage or ETA. |
| **Enhance** | Optional work such as upscaling, interpolation, or other post-processing. Incremental tools can show steps, progress, step time, and ETA. |
| **Save** | Final writing, muxing, or export work. It appears only when WanGP reports a separate Save phase. |

Some models perform several Inputs, Generate, or Decode phases. Status Pro records the individual phases while keeping the main row easy to read.

The detail panel can also identify the model components involved in a stage—for example the transformer during Prepare, an input VAE during Inputs, text encoders during Encode, and output VAEs during Decode. Status Pro shows concise filenames rather than full local paths or download URLs. When several components are involved, they are listed one per line. Repeated work, such as a pipeline alternating between Inputs and Encode, is retained as separate activity lines with cumulative stage timing.

![Decode shown as an active indeterminate stage](<Screenshots/2 -Decode (can't give ETA).png>)

## Understanding the timing information

- **Elapsed** is how long the current stage or complete run has been active.
- **ETA** is an estimate based on measurable progress. It becomes more useful after several steps have completed.
- **Avg step time** is a smoothed estimate from completed Generate or Enhance step updates. Sub-second stages retain decimal precision instead of being rounded to zero.
- **Progress** is displayed only when WanGP provides meaningful units such as steps or transferred bytes.

**Steps** is the configured number for each denoising pass. **Step observations** is the amount of work Status Pro actually saw across every pass. These can differ: MiniMax H3's offline Spectrum mode performs an anchor-capture pass and then a smoothing-replay pass, so a 20-step configuration can correctly produce 40 observations shown as **2 × 20 configured steps**. If passes contain different observed counts, history shows the actual split, such as **2 passes · 8 + 3 observations**. The `res_multistep` sampler changes the update method inside those steps; it does not create the second pass.

An ETA may move during the first few steps as Status Pro learns the current speed. Between completed steps it counts down from the last measured rate rather than repeatedly treating an unfinished step as slower work. Model changes, memory pressure, step-skipping systems, and other applications using the GPU can still affect it.

Status Pro deliberately avoids expected times for pending stages. A previous run with similar settings is not assumed to predict a new run accurately.

## Model loading and downloads

![Model download information](<Screenshots/3 - Model Downloads (wait for 6 transfer cycles before judging ETA).png>)

When model files are required, the Prepare section can show:

- files already completed;
- the file currently transferring;
- files still waiting;
- transferred and total size;
- transfer speed and recent activity;
- an ETA once enough real transfer activity has been observed.

Some Hugging Face downloads transfer data in bursts. A temporary **Waiting** state does not necessarily mean the download has stopped. Status Pro waits for several observed transfer cycles before displaying an ETA so that quiet periods are included in the estimate.

When switching model families, Prepare can first show the outgoing model being unloaded from RAM and VRAM, followed by the incoming model loading. It also keeps the brief **Model loaded** handoff inside Status Pro so the native status box does not need to reappear between stages.

![Model loading shown in Prepare](<Screenshots/4a - Model Loading.png>)

![Outgoing model unloading shown in Prepare](<Screenshots/4b - Model Unloading.png>)

If WanGP or Hugging Face changes an internal download interface, generation will continue normally even when detailed download information is unavailable.

## History

Open **History** from the top-right of Status Pro. History remains available while another generation is running.

Each row summarizes the task, model family and variant, media type, resolution, outcome, duration, and completion time. Expand a row for stage timings, settings, step performance, memory observations, outputs, and failure details.

Expanded records use highlighted, top-aligned field labels to keep long values easier to scan. LoRAs are shown one per line using only the filename without `.safetensors`; hover the value for the complete captured source. Exports continue to contain the original LoRA values.

The **Observed timing composition** bar summarizes the measured stage durations across the run. Its colours identify Prepare, Inputs, Encode, Generate, Decode, Enhance, and Save. A diagonally striped **Unaccounted** segment represents wall-clock time that was not covered by an observed stage; it is not an error indicator or a second progress bar.

In **Step observations**, the fastest valid Time value in each pass is highlighted in green and the slowest in red. Skipped observations are excluded, and the highlights indicate relative timing only—a slowest step is not necessarily faulty.

Status Pro retains up to 100 run records. If browser storage cannot retain the full visible ledger, Status Pro reconciles the list with what was actually saved and displays a storage warning.

### History toolbar

The toolbar is arranged in this order:

| Control | What it does |
| --- | --- |
| **Import** | Restores a Status Pro JSON export into an empty history. |
| **Export** | Immediately exports the selected rows—or the active history scope—using the format, preset, and fields last saved in History settings. |
| **Select all** | Selects every visible run in **This session** or **All history**. |
| **Clear selected** | Removes only the selected history records. |
| **Clear history** | Removes the complete Status Pro history ledger. It does not delete generated files from WanGP's Outputs folder. |
| **⚙** | Opens History settings for retention, prompt memory, export defaults, fields, and presets. |
| **⛶** | Moves the same live History drawer into a larger modal workspace. |

### Tasks with several windows or runs

A task that produces several history entries is collapsed into one summary row. For example, a ten-window generation occupies one row instead of ten.

The task row shows its overall outcome, wall-clock duration, completion time, and window/run count. An unfinished task can show a count such as **7/10 windows**.

Expand the task to see its windows in chronological order. Each child can then be expanded for its own settings and performance details, or opened in the gallery with **View**.

The checkbox on the task row selects every child entry. It displays a mixed state when only some of those entries are selected. Exports continue to contain the individual windows/runs so timing data is not lost.

Common history badges include:

| Badge | Meaning |
| --- | --- |
| **completed** | The run finished and produced an output. |
| **Window N** | One completed part of a sliding-window generation. |
| **aborted** | The run was cancelled before completion. |
| **failed** | WanGP reported an error or the run ended without an output. A concise reason is shown when available. |
| **incomplete** | A grouped task ended before every expected window or run was recorded. |

Use **This session** to focus on runs from the active Status Pro working session. Use **All history** to include every entry currently held by Status Pro.

The number beside **History** counts task groups rather than every underlying window. Hover over the History button to see both the task and recorded-run totals.

Select the **⛶ Expand history** button at the end of the History toolbar when the embedded list feels cramped. It opens the same history drawer in a large modal workspace with a fixed toolbar and a taller scrolling record list. Your selected entries and expanded tasks/runs are preserved because this is the same live history view, not a separate copy. Close it with **Esc**, the × button, by selecting ⛶ again, or by clicking outside the modal; the drawer returns to its normal position without scrolling the WanGP page.

### Opening an output in the gallery

Select **View** on a history entry to select its matching item in WanGP's Video / Images or Audio gallery.

Status Pro does not scroll the page or start playback. If an item is no longer in the gallery, check the WanGP Outputs folder—the underlying file may still exist.

When a run produced several media files, expand it to see a separate **View** action for each recorded output.

## History storage and privacy

Open **⚙ History settings** in the History toolbar to choose how long history is retained.

| Setting | Behaviour |
| --- | --- |
| **Do not record new runs** | Keeps the live stage-based Status Pro display active but does not add newly completed, aborted, or failed runs to History. Existing records remain available until cleared under their current retention setting. |
| **Until manually cleared** | Keeps prompt-free history in this browser across WanGP and browser restarts. It remains until you use **Clear selected** or **Clear history**. |
| **Until browser tab closes** | History survives ordinary reloads and WanGP restarts while the same browser tab or app webview remains open. Closing that browsing context normally clears it. |
| **Until WanGP restarts** | The default. History can survive closing and reopening the browser while the same WanGP process remains active, but clears when Status Pro detects a new WanGP launch. |

The **Prompts** section includes **Remember prompts in this page until it closes**. It is enabled by default so prompts from the current browser session can be explicitly included in an export. Turn it off if you do not want Status Pro to hold prompt text; saving that change removes prompts already held by Status Pro. The preference itself can persist, but prompt text is still excluded from longer-lived history and unchecked in exports by default.

When **Do not record new runs** is selected, prompt memory is paused and the top button reads **History off**. You can still open, export, import, or clear records already in the ledger. Turning recording back on uses the retention lifetime you select; runs that finished while recording was off are not recreated.

Choose a different history lifetime and select **Save settings** to see a confirmation explaining exactly what will survive and what event will clear it. **Cancel** or the × discards unsaved changes. The settings window is draggable and remains constrained to the visible browser area.

History and preferences stay in the browser; Status Pro does not send them to an external service. Use **Clear selected** or **Clear history** whenever you want to remove entries manually.

## Exporting generation data

![Expanded Status Pro history](<Screenshots/6 & 7 - Expanded History.png>)

Exports are useful for comparing models, keeping a run log, reproducing settings, or sharing performance results.

1. Open **⚙ History settings** if you want to change the saved preset, fields, or JSON/CSV/Markdown format, then select **Save settings**.
2. Optionally tick individual history rows, or use **Select all**.
3. Select **Export** in the History toolbar.

If no rows are selected, Status Pro exports the active **This session** or **All history** scope.

### Export presets

| Preset | Best for |
| --- | --- |
| **Standard** | A broad personal archive containing all available fields except prompts. |
| **Performance** | Comparing stage times, step performance, skipped steps, and RAM/VRAM observations. |
| **Reproducibility** | Recording model and generation settings needed to recreate a run. Prompts are included only when available and explicitly selected. |
| **Share-safe** | Publishing timing and performance information without prompts, local paths, exact checkpoints, settings objects, or browser/session IDs. |

You can create several named custom presets for field combinations you use regularly. Use the **i** button in the settings window for an explanation of each preset. Hover a field-group heading or individual field for a short explanation of the data it includes.

![History settings and export fields](<Screenshots/8 & 9 - History Settings Modal.png>)

**Reset to default** restores the Standard field selection. **Save as preset…** stores the current field combination under a name of your choice; several custom presets can coexist. Selecting **Save settings** makes the current preset, format, fields, history lifetime, and prompt-memory choice the defaults used by the toolbar.

### Export formats

- **JSON** preserves the most structure and is best for scripts, tools, or a complete machine-readable record.
- **CSV** is convenient for Excel, LibreOffice, Google Sheets, and performance comparisons.
- **Markdown** creates a readable report suitable for notes, documentation, or issue reports.

Standard and Reproducibility exports can contain local output paths and detailed settings. Use Share-safe when posting results publicly.

### Importing an earlier history export

Status Pro can restore one of its own **JSON** exports into History for convenient review. CSV and Markdown files are not importable because they may not preserve the structure needed to rebuild history reliably.

History must be empty before importing:

1. Export any current history you want to keep.
2. Use **Clear history**.
3. Select **Import** and choose the Status Pro `.json` file.

If records are still present, Status Pro stops before opening the file picker and reminds you to export and clear them. Imports never merge with existing records and never replace them automatically.

An import can contain at most 100 records and the JSON file must be no larger than 20 MB.

Imported records appear under **All history**, retain whichever fields were included in the export, and follow the currently selected history-retention policy. Their expanded details show the import time, original export time, and export version.

The top-row **View** button remains inactive for imported history. If the JSON includes an output path, expand the record and use **Import media** beside Output to add that file back to its appropriate Video / Images or Audio gallery. Status Pro confirms a successful import and does not scroll the gallery into view. If the recorded file has since been moved or deleted, it warns that the file is no longer available at that output path; the original Outputs folder is the best place to check. For safety, this action accepts only supported media inside WanGP's configured output folders.

Prompt text in an imported file follows the same privacy rules as new generations: it is available only when page prompt memory is enabled and is not added to longer-lived history storage.

## Useful workflows

### Monitor a long generation

Watch Generate step time and ETA, then select Decode or Enhance when the run moves forward. Collapse the display when you only need the live top-line summary.

### Compare models or settings

Run your tests, select the relevant history rows, and export the **Performance** preset as CSV. Compare total duration, phase timings, average steps, skipped steps, and observed memory use.

### Keep a reproducibility record

Before ending the browser-tab session, export the **Reproducibility** preset. Explicitly enable prompt fields if you want them included and they are still available.

### Share a result safely

Choose **Share-safe** and review the selected fields before exporting. This preset is designed to leave out prompts and machine-specific paths.

## Common questions

### Why does Encode say “Not reported”?

The model may still have encoded its inputs, but WanGP did not provide a separate start and finish signal that Status Pro could time reliably.

### Why does Prepare say “Preloaded”?

The required model was already loaded, so WanGP could move directly to Encode or Generate without another loading phase.

### Why did an optional stage disappear?

Inputs, Enhance, and Save appear only when the current run actually reaches them. Hiding unused optional stages keeps the timeline from implying that work was configured when it was not.

### Why is Decode running without a percentage?

WanGP generally exposes VAE decoding as one operation with no intermediate progress units. Status Pro shows that it is active and counts elapsed time instead of displaying a misleading static 0%.

### Why is the ETA changing?

Early estimates have only a few completed steps to learn from. The value should become steadier as more steps finish, although GPU load and model behaviour can still change the speed.

### Why does the terminal timing differ slightly?

Status Pro observes WanGP from the browser at short intervals. Small differences from internal terminal timings are normal.

### Why are there more step observations than configured steps?

Some pipelines perform more than one real denoising pass. Status Pro keeps the configured step count separate, shows how many passes were observed, and retains every observation for performance analysis. For example, MiniMax H3 Spectrum can run 20 capture steps followed by 20 replay steps.

### What does “Unaccounted” time mean in History?

It is the material difference between the observed wall-clock run time and the individual phase durations Status Pro could measure. It can include short WanGP handoffs, queue work, polling gaps, or model-specific operations that did not emit a separate status phase. It is not automatically an error.

### Why can History not find an output in the gallery?

WanGP's gallery contents can change independently from Status Pro history. Check the Outputs folder because the file may still be present on disk.

### Does clearing History delete generated media?

No. **Clear selected** and **Clear history** remove Status Pro's browser ledger only. They do not delete files from WanGP's Outputs folder or remove media that is already in WanGP's gallery.

### Why is Import media unavailable or rejected?

The imported JSON must include an output path, the file must still exist, and it must be a supported image, video, or audio file inside WanGP's configured output folders. If the file was moved, check the original Outputs folder and add it to WanGP's gallery manually if appropriate.

### Are RAM and VRAM figures exact profiler measurements?

No. They are periodic observations intended for practical comparisons. Other applications can contribute to total GPU memory use.

## If something looks wrong

1. Allow the current generation to finish or abort it safely.
2. Restart WanGP after installing or updating Status Pro.
3. Confirm the plugin is enabled in WanGP's Plugins tab.
4. Compare the native terminal message with the selected Status Pro stage.
5. Include a screenshot and a Status Pro JSON or Markdown export when reporting a repeatable issue. Remove sensitive fields or use Share-safe before posting publicly.

Status Pro never controls generation itself. Its job is to observe WanGP and present the available progress information more clearly.
