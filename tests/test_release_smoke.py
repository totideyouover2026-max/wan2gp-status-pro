import ast
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PLUGIN_PATH = ROOT / "plugin.py"
DOWNLOAD_PATH = ROOT / "download_telemetry.py"


def _source() -> str:
    return PLUGIN_PATH.read_text(encoding="utf-8")


def _returned_string(function_name: str) -> str:
    tree = ast.parse(_source())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != function_name:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and isinstance(child.value, ast.Constant) and isinstance(child.value.value, str):
                return child.value.value
    raise AssertionError(f"No constant string return found for {function_name}")


def _download_module():
    spec = importlib.util.spec_from_file_location("status_pro_download_release_test", DOWNLOAD_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _javascript_with_exports(*names: str) -> str:
    marker = "    boot();\n})();"
    javascript = _returned_string("_javascript")
    if marker not in javascript:
        raise AssertionError("Embedded JavaScript boot marker changed")
    exports = ", ".join(names)
    return javascript.replace(
        marker,
        f"    globalThis.__statusProReleaseTest = {{ {exports} }};\n}})();",
    )


def _python_functions(*names: str):
    tree = ast.parse(_source())
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = [name for name in names if name not in definitions]
    if missing:
        raise AssertionError(f"Missing Python helper(s): {', '.join(missing)}")
    namespace = {"os": os}
    module = ast.Module(body=[definitions[name] for name in names], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(PLUGIN_PATH), "exec"), namespace)
    return namespace


class ReleaseSmokeTests(unittest.TestCase):
    def test_python_and_manifest_versions_agree(self):
        source = _source()
        ast.parse(source)
        ast.parse(DOWNLOAD_PATH.read_text(encoding="utf-8"))
        for document in ("README.md", "USER_GUIDE.md", "CHANGELOG.md", "RELEASE_CHECKLIST.md"):
            self.assertTrue((ROOT / document).is_file(), document)
        manifest = json.loads((ROOT / "plugin_info.json").read_text(encoding="utf-8"))
        version = manifest["version"]
        self.assertEqual(version, "1.0.1")
        self.assertIn(f'self.version = "{version}"', source)
        self.assertIn(f'version: "{version}"', source)
        self.assertEqual(manifest["type"], "extension")
        # WanGP backfills blank metadata from its cached catalogue. A populated
        # zero baseline behaves as no hard minimum without reviving stale values.
        self.assertEqual(manifest["wan2gp_version"], "0")

    def test_embedded_markup_has_release_controls(self):
        markup = _returned_string("_markup")
        for token in (
            "data-status-pro",
            "data-sp-stages",
            "data-sp-history",
            "data-sp-history-home",
            "data-sp-history-expand",
            "data-sp-history-modal",
            "data-sp-history-modal-content",
            "data-sp-history-modal-close",
            "data-sp-import-button",
            "data-sp-import-file",
            "data-sp-history-storage-note",
            "data-sp-history-persistence",
            "data-sp-export-modal",
            "data-sp-settings-button",
            "data-sp-export-button",
            "data-sp-detail-activities",
        ):
            self.assertIn(token, markup)
        self.assertIn('option value="off">Do not record new runs', markup)
        self.assertNotIn("data-sp-capture-prompts", markup)
        toolbar_order = (
            "data-sp-import-button",
            "data-sp-export-button",
            "data-sp-select-all-history",
            "data-sp-clear-selected",
            "data-sp-clear-history",
            "data-sp-settings-button",
            "data-sp-history-expand",
        )
        positions = [markup.index(token) for token in toolbar_order]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("⚙", markup)
        self.assertIn("⛶", markup)
        self.assertIn('accept=".json,application/json"', markup)
        javascript = _returned_string("_javascript")
        self.assertIn("--sp-history-control-height", javascript)
        self.assertIn("function openHistoryModal(namespace)", javascript)
        self.assertIn("function closeHistoryModal(namespace)", javascript)
        self.assertIn("function restoreHistoryDrawer(namespace)", javascript)
        self.assertIn('.status-pro__history-drawer[data-expanded="true"]', javascript)
        self.assertIn("function normalizeImportedExport(payload", javascript)
        self.assertIn("function importStatusProExport(namespace", javascript)
        self.assertIn("function requestGalleryImport(namespace", javascript)
        self.assertIn("addImportedMediaField(", javascript)
        self.assertIn('button.textContent = records.length === 1 ? "Import media"', javascript)
        self.assertNotIn("Gallery View is disabled for imported history", javascript)
        self.assertIn('(Array.isArray(value) && value.length === 0)', javascript)
        self.assertIn('addRunField(fields, "Model loading"', javascript)
        self.assertNotIn('addRunField(fields, "Model / setup"', javascript)

    def test_embedded_javascript_syntax(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for embedded JavaScript syntax validation")
        result = subprocess.run(
            [node, "--check", "-"],
            input=_returned_string("_javascript"),
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_export_field_tooltips_cover_every_group_and_option(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for export-tooltip validation")
        javascript = _javascript_with_exports("EXPORT_FIELD_DEFS", "EXPORT_FIELD_HELP", "EXPORT_GROUP_HELP")
        test_script = r'''
const api = globalThis.__statusProReleaseTest;
for (const field of api.EXPORT_FIELD_DEFS) {
  if (!api.EXPORT_FIELD_HELP[field.id] || api.EXPORT_FIELD_HELP[field.id].trim().length < 12) {
    throw new Error(`missing concise help for export field ${field.id}`);
  }
  if (!api.EXPORT_GROUP_HELP[field.group] || api.EXPORT_GROUP_HELP[field.group].trim().length < 12) {
    throw new Error(`missing concise help for export group ${field.group}`);
  }
}
'''
        result = subprocess.run(
            [node, "-"],
            input=javascript + "\n" + test_script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_expanded_history_moves_and_restores_the_live_drawer(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for expanded-history validation")
        javascript = _javascript_with_exports("openHistoryModal", "closeHistoryModal", "restoreHistoryDrawer")
        test_script = r'''
const count = {textContent: ""};
const toggle = {
  title: "",
  attributes: {},
  setAttribute(name, value) { this.attributes[name] = value; },
  querySelector(selector) { return selector === "[data-sp-history-count]" ? count : null; }
};
const drawer = {
  dataset: {},
  hidden: true,
  removeAttribute(name) { if (name === "data-expanded") delete this.dataset.expanded; }
};
const expand = {
  title: "",
  attributes: {},
  setAttribute(name, value) { this.attributes[name] = value; }
};
const close = {focus() { this.focused = true; }};
const modal = {
  open: false,
  showModal() { this.open = true; },
  removeAttribute(name) { if (name === "open") this.open = false; },
  querySelector(selector) { return selector === "[data-sp-history-modal-close]" ? close : null; }
};
const content = {appendChild(node) { this.child = node; }};
const home = {after(node) { this.restored = node; }};
const history = {};
const empty = {};
const modalSummary = {textContent: ""};
const bySelector = new Map([
  ["[data-sp-history-drawer]", drawer],
  ["[data-sp-history-home]", home],
  ["[data-sp-history-modal]", modal],
  ["[data-sp-history-modal-content]", content],
  ["[data-sp-history-modal-summary]", modalSummary],
  ["[data-sp-history-modal-close]", close],
  ["[data-sp-history-toggle]", toggle],
  ["[data-sp-history-expand]", expand],
  ["[data-sp-history]", history],
  ["[data-sp-history-empty]", empty]
]);
const panel = {
  querySelector(selector) { return bySelector.get(selector) || null; },
  querySelectorAll() { return []; }
};
const namespace = {
  panel,
  runHistory: [],
  sessionRunIds: new Set(),
  selectedRunIds: new Set(),
  historyScope: "all",
  historyOpen: false,
  historyExpanded: false,
  historyRenderKey: "all:"
};
globalThis.__statusProReleaseTest.openHistoryModal(namespace);
if (!namespace.historyOpen || !namespace.historyExpanded || !modal.open) throw new Error("expanded history did not open");
if (content.child !== drawer || drawer.dataset.expanded !== "true" || drawer.hidden) throw new Error("live drawer was not moved into the modal");
if (expand.attributes["aria-expanded"] !== "true" || !close.focused) throw new Error("expanded controls were not updated");
globalThis.__statusProReleaseTest.closeHistoryModal(namespace);
if (namespace.historyExpanded || modal.open || home.restored !== drawer || drawer.dataset.expanded) {
  throw new Error("live drawer was not restored after closing expanded history");
}
if (expand.attributes["aria-expanded"] !== "false") throw new Error("embedded controls were not restored");
'''
        result = subprocess.run(
            [node, "-"],
            input=javascript + "\n" + test_script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_json_import_normalizes_partial_exports_and_rejects_unsafe_merges(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for JSON-import validation")
        javascript = _javascript_with_exports("normalizeImportedExport", "importStatusProExport")
        test_script = r'''
const payload = {
  exported_at: "2026-08-11T23:22:55.989Z",
  version: "0.5.1",
  scope: "selected",
  preset: "Performance",
  runs: [{
    run_id: "old-run-1",
    session_id: "old-session",
    queue_task_id: 7,
    status: "completed",
    started_at: "2026-08-11T23:17:32.000Z",
    completed_at: "2026-08-11T23:22:37.000Z",
    duration_seconds: 305,
    generation_time: 246,
    model_summary: "LTX-2 - 2.3 Distilled - Video - 960x512",
    model_name: "LTX-2 2.3 Distilled",
    checkpoint: "ltx-2.3-distilled-int8.safetensors",
    resolution: "960x512",
    steps: 8,
    media_type: "video",
    frame_count: 241,
    output_count: 1,
    prompt: "A retained test prompt",
    phase_timings: {"denoise:first": {label: "First", duration_seconds: 90, status: "complete", stage: "denoise"}},
    step_skipping: {
      recorded_steps: 11,
      step_observations: 11,
      observed_passes: 2,
      pass_summaries: [
        {label: "Pass 1", observed_steps: 8, configured_steps: 8},
        {label: "Pass 2", observed_steps: 3, configured_steps: 3}
      ]
    }
  }]
};
const importedAt = Date.parse("2026-08-12T01:00:00.000Z");
const runs = globalThis.__statusProReleaseTest.normalizeImportedExport(payload, importedAt);
if (runs.length !== 1) throw new Error("valid Status Pro JSON did not import");
const run = runs[0];
if (!run.imported || run.id !== "old-run-1" || run.session_id !== "old-session") throw new Error("import identity was not preserved");
if (run.status !== "completed" || run.output_count !== 1) throw new Error("partial import was misclassified as failed");
if (run.media_type !== "video" || run.frame_count !== 241) throw new Error("declared media metadata was lost without output paths");
if (run.settings.model_filename !== "ltx-2.3-distilled-int8.safetensors" || run.settings.prompt !== "A retained test prompt") {
  throw new Error("export fields were not rebuilt into run settings");
}
if (run.step_summary.observed_passes !== 2 || run.step_summary.passes[1].configured_steps !== 3) {
  throw new Error("imported pass summaries were lost");
}
if (run.imported_model_summary !== payload.runs[0].model_summary || run.import_source.version !== "0.5.1") {
  throw new Error("import provenance was not recorded");
}

const legacyPayload = {
  exported_at: "2026-08-08T12:45:14.543Z",
  version: "0.5.1",
  runs: [{
    id: "legacy-internal-run",
    session_id: "legacy-session",
    queue_task_id: 1,
    status: "completed",
    started_at: 1786192392989.7996,
    completed_at: 1786193089198.046,
    duration_seconds: 696.2,
    settings: {model_type: "krea2_identity_turbo", resolution: "2688x1152", num_inference_steps: 12, image_mode: 1},
    stages: {prepare: {label: "Prepare", duration_seconds: 20.5, status: "complete"}},
    outputs: ["outputs/Images/result.jpg"],
    repeats: 1
  }]
};
const legacyRun = globalThis.__statusProReleaseTest.normalizeImportedExport(legacyPayload, importedAt)[0];
if (legacyRun.id !== "legacy-internal-run" || !legacyRun.stages.prepare) throw new Error("legacy internal export keys were not restored");
if (legacyRun.media_type !== "image" || legacyRun.frame_count !== 1 || legacyRun.status !== "completed") {
  throw new Error("legacy image export was not normalized correctly");
}

let blocked = false;
try {
  globalThis.__statusProReleaseTest.importStatusProExport({runHistory: [{}]}, payload, importedAt);
} catch (error) {
  blocked = /must be empty/i.test(String(error && error.message));
}
if (!blocked) throw new Error("non-empty history did not block import");

function makeStorage() {
  const values = new Map();
  return {
    getItem(key) { return values.has(key) ? values.get(key) : null; },
    setItem(key, value) { values.set(key, String(value)); },
    removeItem(key) { values.delete(key); }
  };
}
globalThis.window = {localStorage: makeStorage(), sessionStorage: makeStorage()};
const namespace = {
  runHistory: [],
  historyPersistence: "runtime",
  promptMemory: true,
  recoverablePrompts: new Map(),
  sessionRunIds: new Set(),
  selectedRunIds: new Set(),
  openHistoryGroups: new Set(),
  openHistoryRuns: new Set(),
  visibleHistoryGroups: new Map(),
  galleryFeedback: new Map(),
  galleryRequests: new Map(),
  historyScope: "session",
  historyOpen: false,
  historyRenderKey: "stale",
  historyStorageNotice: ""
};
const applied = globalThis.__statusProReleaseTest.importStatusProExport(namespace, payload, importedAt);
if (applied.imported !== 1 || namespace.historyScope !== "all" || !namespace.historyOpen) throw new Error("imported history was not activated for review");
if (namespace.sessionRunIds.size) throw new Error("past imported records were added to This session");
if (namespace.runHistory[0].settings.prompt) throw new Error("prompt leaked into WanGP-runtime history");
if (!namespace.recoverablePrompts.get("old-run-1")?.settings?.prompt) throw new Error("page-memory prompt recovery was not preserved");
const storedImport = window.localStorage.getItem("wangp.status-pro.run-history.runtime.v1");
if (!storedImport || storedImport.includes("A retained test prompt")) throw new Error("persisted imported history retained prompt text");

let duplicateRejected = false;
try {
  globalThis.__statusProReleaseTest.normalizeImportedExport({...payload, runs: [payload.runs[0], payload.runs[0]]}, importedAt);
} catch (error) {
  duplicateRejected = /repeats run ID/i.test(String(error && error.message));
}
if (!duplicateRejected) throw new Error("duplicate imported run IDs were accepted");

let arbitraryRejected = false;
try {
  globalThis.__statusProReleaseTest.normalizeImportedExport({runs: [{}]}, importedAt);
} catch (error) {
  arbitraryRejected = /metadata/i.test(String(error && error.message));
}
if (!arbitraryRejected) throw new Error("arbitrary JSON was accepted as a Status Pro export");
'''
        result = subprocess.run(
            [node, "-"],
            input=javascript + "\n" + test_script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        javascript = _returned_string("_javascript")
        self.assertNotIn("container.replaceChildren(orderedButtons)", javascript)
        self.assertNotIn(".status-pro__stage:hover { opacity: .92; transform:", javascript)

    def test_download_observer_failures_do_not_escape(self):
        module = _download_module()
        observer = module.DownloadObserver(module.DownloadTelemetry())

        def unavailable():
            raise RuntimeError("simulated incompatible API")

        observer._install_shared_download_wrappers = unavailable
        observer._install_huggingface_progress_wrapper = unavailable
        observer.install()
        status = observer.status()
        self.assertTrue(status["installed"])
        self.assertFalse(status["shared_download_available"])
        self.assertFalse(status["huggingface_progress_available"])
        self.assertEqual(len(status["errors"]), 2)

    def test_download_telemetry_lifecycle_and_json_safety(self):
        module = _download_module()
        telemetry = module.DownloadTelemetry()
        telemetry.begin_batch(["model.safetensors"], label="Test model")
        record_id = telemetry.begin_file("model.safetensors", total=1000)
        telemetry.update_file(record_id, 400, 1000)
        running = telemetry.snapshot()
        self.assertTrue(running["active"])
        self.assertEqual(running["totals"]["known_total"], 1000)
        self.assertFalse(any(key.startswith("_") for key in running["files"][0]))
        telemetry.complete_file(record_id)
        telemetry.end_batch(["model.safetensors"])
        completed = json.loads(telemetry.snapshot_json())
        self.assertFalse(completed["active"])
        self.assertEqual(completed["totals"]["completed"], 1)

    def test_browser_storage_quota_returns_the_persisted_subset(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for browser-storage behavior validation")
        javascript = _javascript_with_exports(
            "loadHistoryPersistence",
            "saveRunHistory",
            "persistRunHistory",
            "setHistoryPersistence",
            "prepareRuntimeHistory",
            "historyPersistenceConfirmation",
            "loadPromptMemoryPreference",
            "setPromptMemory",
            "loadExportSettings",
            "saveExportSettings",
        )
        test_script = r'''
function makeStorage(limit = Infinity) {
  return {
    data: {},
    getItem(key) { return Object.prototype.hasOwnProperty.call(this.data, key) ? this.data[key] : null; },
    setItem(key, value) {
      if (key.includes("run-history") && value.length > limit) throw new Error("quota");
      this.data[key] = value;
    },
    removeItem(key) { delete this.data[key]; }
  };
}
globalThis.window = {
  localStorage: makeStorage(),
  sessionStorage: makeStorage(170)
};

window.localStorage.data["wangp.status-pro.run-history.v1"] = JSON.stringify([{id: "legacy", settings: {prompt: "legacy private prompt"}}]);
if (globalThis.__statusProReleaseTest.loadHistoryPersistence("runtime-a") !== "runtime") throw new Error("new default was not WanGP-runtime scoped");
if (window.localStorage.getItem("wangp.status-pro.run-history.v1") !== null) throw new Error("legacy persistent history was not removed");
if (!window.localStorage.getItem("wangp.status-pro.run-history.runtime.v1").includes("legacy")) throw new Error("legacy history was not moved into the WanGP runtime");
if (window.localStorage.getItem("wangp.status-pro.run-history.runtime.v1").includes("legacy private prompt")) throw new Error("legacy prompt leaked into WanGP-runtime history");
if (window.localStorage.getItem("wangp.status-pro.history-runtime-id.v1") !== "runtime-a") throw new Error("runtime ID was not recorded");

const runs = Array.from({length: 8}, (_, index) => ({id: `run-${index}`, detail: "x".repeat(40)}));
const result = globalThis.__statusProReleaseTest.saveRunHistory(runs, "browser");
if (!result.persisted) throw new Error("history should persist a subset");
if (!(result.dropped > 0)) throw new Error("quota did not trim history");
const stored = JSON.parse(window.sessionStorage.getItem("wangp.status-pro.run-history.session.v1"));
if (stored.length !== result.retained.length) throw new Error("visible/persisted subset mismatch");
if (stored[0].id !== "run-0") throw new Error("newest history was not retained");

const namespace = {
  runHistory: runs,
  historyPersistence: "browser",
  sessionRunIds: new Set(runs.map(run => run.id)),
  selectedRunIds: new Set(["run-0", "run-7"]),
  recoverablePrompts: new Map(runs.map(run => [run.id, {prompt: run.id}])),
  historyStorageNotice: ""
};
const persisted = globalThis.__statusProReleaseTest.persistRunHistory(namespace);
if (namespace.runHistory.length !== persisted.retained.length) throw new Error("namespace history was not reconciled");
if (!namespace.historyStorageNotice.includes("session storage is full")) throw new Error("quota warning is missing");
if (namespace.selectedRunIds.has("run-7")) throw new Error("trimmed selection was retained");
if (namespace.recoverablePrompts.has("run-7")) throw new Error("trimmed recoverable prompt was retained");

window.sessionStorage.setItem = () => { throw new Error("blocked"); };
const unavailable = {
  runHistory: [{id: "session-only"}],
  historyPersistence: "browser",
  sessionRunIds: new Set(["session-only"]),
  selectedRunIds: new Set(),
  recoverablePrompts: new Map(),
  historyStorageNotice: ""
};
const unavailableResult = globalThis.__statusProReleaseTest.persistRunHistory(unavailable);
if (unavailableResult.persisted) throw new Error("blocked storage reported success");
if (unavailable.runHistory.length !== 1) throw new Error("page-session history was discarded");
if (!unavailable.historyStorageNotice.includes("page is closed or reloaded")) throw new Error("unavailable-storage warning is missing");

window.sessionStorage = makeStorage();
window.localStorage = makeStorage();
const switchNamespace = {
  runHistory: [{
    id: "private-run",
    settings: {prompt: "private prompt", negative_prompt: "private negative", resolution: "1280x544"},
    output_records: [{path: "result.png", settings: {prompt: "private prompt"}}]
  }],
  historyPersistence: "browser",
  runtimeId: "runtime-a",
  sessionRunIds: new Set(["private-run"]),
  selectedRunIds: new Set(),
  recoverablePrompts: new Map(),
  historyStorageNotice: "",
  historyRenderKey: null
};
if (!globalThis.__statusProReleaseTest.setHistoryPersistence(switchNamespace, "persistent")) throw new Error("persistent switch failed");
const persistentRaw = window.localStorage.getItem("wangp.status-pro.run-history.v1");
if (!persistentRaw || persistentRaw.includes("private prompt") || persistentRaw.includes("private negative")) {
  throw new Error("persistent history retained prompt data");
}
if (window.sessionStorage.getItem("wangp.status-pro.run-history.session.v1") !== null) throw new Error("old session copy was retained");
if (!globalThis.__statusProReleaseTest.setHistoryPersistence(switchNamespace, "browser")) throw new Error("browser-tab switch failed");
const sessionRaw = window.sessionStorage.getItem("wangp.status-pro.run-history.session.v1");
if (!sessionRaw || !sessionRaw.includes("private prompt")) throw new Error("browser-tab prompt was not restored");
if (window.localStorage.getItem("wangp.status-pro.run-history.v1") !== null) throw new Error("persistent copy was not removed");

if (!globalThis.__statusProReleaseTest.setHistoryPersistence(switchNamespace, "runtime")) throw new Error("WanGP-runtime switch failed");
const runtimeRaw = window.localStorage.getItem("wangp.status-pro.run-history.runtime.v1");
if (!runtimeRaw || runtimeRaw.includes("private prompt")) throw new Error("WanGP-runtime history retained prompt data");
if (window.sessionStorage.getItem("wangp.status-pro.run-history.session.v1") !== null) throw new Error("browser-tab copy was retained");
if (!globalThis.__statusProReleaseTest.prepareRuntimeHistory("runtime-b")) throw new Error("WanGP restart was not detected");
if (window.localStorage.getItem("wangp.status-pro.run-history.runtime.v1") !== null) throw new Error("previous WanGP-runtime history was not cleared");

for (const mode of ["persistent", "browser", "runtime"]) {
  const message = globalThis.__statusProReleaseTest.historyPersistenceConfirmation(mode);
  if (!message || !message.includes("history")) throw new Error(`missing confirmation for ${mode}`);
}

window.sessionStorage = makeStorage();
window.localStorage = makeStorage();
if (!globalThis.__statusProReleaseTest.loadPromptMemoryPreference()) throw new Error("prompt memory did not preserve the enabled default");
const promptNamespace = {
  promptMemory: true,
  historyPersistence: "browser",
  runHistory: [{id: "prompt-run", settings: {prompt: "private prompt"}, output_records: []}],
  recoverablePrompts: new Map([["prompt-run", {settings: {prompt: "private prompt"}, outputRecords: []}]]),
  sessionRunIds: new Set(["prompt-run"]),
  selectedRunIds: new Set(),
  historyStorageNotice: "",
  historyRenderKey: null
};
globalThis.__statusProReleaseTest.setPromptMemory(promptNamespace, false);
if (promptNamespace.promptMemory || promptNamespace.recoverablePrompts.size) throw new Error("disabling prompt memory did not clear page-held prompts");
if (JSON.stringify(promptNamespace.runHistory).includes("private prompt")) throw new Error("disabling prompt memory retained prompt text in visible history");
if (window.localStorage.getItem("wangp.status-pro.prompt-memory.v1") !== "0") throw new Error("prompt-memory preference was not saved");
if (window.sessionStorage.getItem("wangp.status-pro.run-history.session.v1").includes("private prompt")) throw new Error("disabled prompt memory persisted prompt text");

const exportNamespace = {exportFields: new Set(["status", "resolution"]), exportPreset: "custom:unsaved", exportFormat: "md"};
globalThis.__statusProReleaseTest.saveExportSettings(exportNamespace);
const savedExport = globalThis.__statusProReleaseTest.loadExportSettings();
if (savedExport.format !== "md" || savedExport.fields.length !== 2 || !savedExport.fields.includes("resolution")) {
  throw new Error("saved export defaults did not reload");
}
'''
        result = subprocess.run(
            [node, "-"],
            input=javascript + "\n" + test_script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_history_recording_and_v101_history_display_helpers(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for V1.0.1 history behavior validation")
        javascript = _javascript_with_exports(
            "loadHistoryRecordingPreference",
            "setHistoryRecording",
            "historyRecordingConfirmation",
            "compactLoraNames",
            "timingOverviewSegments",
            "stepTimingOutliers",
        )
        test_script = r'''
const values = {};
globalThis.window = {
  localStorage: {
    getItem(key) { return Object.prototype.hasOwnProperty.call(values, key) ? values[key] : null; },
    setItem(key, value) { values[key] = String(value); }
  }
};
const api = globalThis.__statusProReleaseTest;
if (!api.loadHistoryRecordingPreference()) throw new Error("history recording did not default on");
const namespace = {historyRecording: true, historyStorageNotice: "", historyRenderKey: "cached"};
api.setHistoryRecording(namespace, false);
if (namespace.historyRecording || values["wangp.status-pro.history-recording.v1"] !== "0") throw new Error("history off was not saved");
if (!namespace.historyStorageNotice.includes("Existing records are unchanged")) throw new Error("history off notice is unclear");
if (!api.historyRecordingConfirmation(false).includes("Live Status Pro tracking will continue")) throw new Error("history off confirmation omits live tracking");
api.setHistoryRecording(namespace, true);
if (!namespace.historyRecording || values["wangp.status-pro.history-recording.v1"] !== "1") throw new Error("history on was not restored");

const loras = api.compactLoraNames([
  "https://huggingface.co/example/resolve/main/loras/first_style.safetensors?download=true",
  "C:\\models\\second_style.safetensors"
]);
if (loras !== "first_style\nsecond_style") throw new Error(`LoRA display was not compacted line by line: ${loras}`);

const segments = api.timingOverviewSegments({
  duration_seconds: 100,
  stages: {
    setup: {label: "Loading model", duration_seconds: 10},
    denoise: {stage: "denoise", label: "Generate", duration_seconds: 70}
  }
});
if (segments.length !== 3 || segments[2].stage !== "unaccounted" || segments[2].seconds !== 20) {
  throw new Error("timing overview did not preserve unaccounted wall time");
}
if (segments[0].stage !== "prepare") throw new Error("legacy stage labels were not mapped to timing colours");

const observations = [
  {pass_no: 1, duration_seconds: 3},
  {pass_no: 1, duration_seconds: 1},
  {pass_no: 1, duration_seconds: 5},
  {pass_no: 1, duration_seconds: 0.1, skipped: true},
  {pass_no: 2, duration_seconds: 10},
  {pass_no: 2, duration_seconds: 20}
];
const outliers = api.stepTimingOutliers(observations);
if (!outliers.fastest.has(1) || !outliers.slowest.has(2) || !outliers.fastest.has(4) || !outliers.slowest.has(5)) {
  throw new Error("per-pass fastest and slowest observations were not identified");
}
if (outliers.fastest.has(3)) throw new Error("skipped observations were included in timing outliers");
'''
        result = subprocess.run(
            [node, "-"],
            input=javascript + "\n" + test_script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        javascript_source = _returned_string("_javascript")
        self.assertIn('if (namespace.historyRecording === false)', javascript_source)
        self.assertIn('repeating-linear-gradient', javascript_source)
        self.assertIn('chip.dataset.stage = timingStageId', javascript_source)

    def test_stage_media_outcome_and_export_regressions(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for embedded behavior validation")
        javascript = _javascript_with_exports(
            "freshState",
            "applySnapshot",
            "stageIdFor",
            "stageActivities",
            "STAGE_DEFS",
            "reinterpretQwenSilentEncode",
            "normalizeRunMedia",
            "observeRunOutcome",
            "updateStepTiming",
            "exportableRuns",
            "exportCsv",
            "exportMarkdown",
            "EXPORT_PRESETS",
        )
        test_script = r'''
globalThis.window = {
  localStorage: { getItem() { return null; }, setItem() {} },
  sessionStorage: { getItem() { return null; }, setItem() {} }
};
const api = globalThis.__statusProReleaseTest;
const stageOrder = api.STAGE_DEFS.map(stage => stage.id);
if (stageOrder.indexOf("input") !== stageOrder.indexOf("prepare") + 1 || stageOrder.indexOf("input") >= stageOrder.indexOf("encode")) {
  throw new Error("Inputs is not fixed between Prepare and Encode");
}
const phaseCases = {
  "Loading model MiniMax H3": "prepare",
  "Please Wait While Loading Prompt Enhancer": "prepare",
  "Enhancing Prompt": "encode",
  "Encoding Prompt": "encode",
  "Encoding H3 prompt and references": "encode",
  "Encoding Speaker 1 Reference": "encode",
  "Encoding H3 control video": "input",
  "Control video VAE decoding": "input",
  "VAE Encoding": "input",
  "Extracting Face Movements": "input",
  "Extracting Depth Maps": "input",
  "Extracting Gray Levels": "input",
  "Animate preprocessing": "input",
  "Removing Images References Background": "input",
  "Resizing source images": "input",
  "Loading control video": "input",
  "Preparing input frames": "input",
  "Denoising": "denoise",
  "Spectrum smoothing replay": "denoise",
  "Generating Audio": "denoise",
  "VAE Decoding": "decode",
  "Decoding H3 stereo audio": "decode",
  "Upsampling - PiD": "post",
  "Applying SeedVC": "post",
  "Saving output": "save",
  "Muxing audio": "save"
};
for (const [phase, expected] of Object.entries(phaseCases)) {
  const actual = api.stageIdFor(phase);
  if (actual !== expected) throw new Error(`${phase} classified as ${actual}, expected ${expected}`);
}

const snapshot = id => ({
  id,
  rawName: id === "input" ? "Encoding control video" : (id === "encode" ? "Encoding prompt" : (id === "decode" ? "VAE decoding" : "Denoising")),
  rawMessage: id,
  metaText: "",
  stageElapsed: null,
  nativeEta: null,
  overallElapsed: 10,
  progress: id === "decode" ? 0 : 25,
  steps: id === "denoise" ? {current: 1, total: 4} : {current: null, total: null},
  aborting: false,
  textOnly: true
});

let namespace = {state: api.freshState(), activeRun: {}};
if (namespace.state.records.input.visible) throw new Error("optional Inputs stage was visible before use");
api.applySnapshot(namespace, snapshot("input"));
if (!namespace.state.records.prepare.preloaded || !namespace.state.records.input.visible || namespace.state.currentId !== "input") {
  throw new Error("Inputs did not appear conditionally after a preloaded Prepare stage");
}

const originalNow = Date.now;
let clock = 1000;
Date.now = () => clock;
namespace = {state: api.freshState(), activeRun: {}};
const firstInput = {...snapshot("input"), rawName: "Preparing control video", rawMessage: "Preparing control video"};
const secondInput = {...snapshot("input"), rawName: "VAE Encoding", rawMessage: "VAE Encoding"};
api.applySnapshot(namespace, firstInput);
clock = 4000;
api.applySnapshot(namespace, snapshot("encode"));
clock = 5000;
api.applySnapshot(namespace, secondInput);
clock = 7000;
api.applySnapshot(namespace, secondInput);
Date.now = originalNow;
if (Math.abs(namespace.state.records.input.elapsed - 5) > 0.001) {
  throw new Error(`re-entered Inputs lost cumulative time: ${namespace.state.records.input.elapsed}`);
}
const inputActivities = api.stageActivities(namespace.state, namespace.state.records.input);
if (inputActivities.length !== 2 || inputActivities[0].label !== "Preparing control video" || inputActivities[1].label !== "VAE Encoding") {
  throw new Error("Inputs activity history was not preserved across Encode");
}

namespace = {state: api.freshState(), activeRun: {}};
api.applySnapshot(namespace, snapshot("encode"));
if (!namespace.state.records.prepare.preloaded || namespace.state.records.prepare.state !== "complete") {
  throw new Error("explicit Encode did not mark Prepare as model-agnostic Preloaded");
}

namespace = {state: api.freshState(), activeRun: {}};
api.applySnapshot(namespace, snapshot("denoise"));
if (!namespace.state.records.prepare.preloaded) throw new Error("direct Generate did not mark Prepare Preloaded");
if (!namespace.state.records.encode.unreported || namespace.state.records.encode.state !== "complete") {
  throw new Error("direct Generate did not mark Encode Not reported");
}

namespace = {state: api.freshState(), activeRun: {}};
api.applySnapshot(namespace, snapshot("decode"));
if (namespace.state.records.decode.progress !== null) throw new Error("Decode exposed an invented percentage");

const qwenNamespace = {runTelemetry: {
  server_time: 100,
  active_task: {settings: {model_type: "qwen_image_edit"}},
  performance: {callback_phase: 1, phase_started_at: 90, steps: []}
}};
if (api.reinterpretQwenSilentEncode(qwenNamespace, snapshot("denoise")).id !== "encode") {
  throw new Error("Qwen silent Encode recovery failed");
}
const otherNamespace = {runTelemetry: {
  active_task: {settings: {model_type: "krea2_identity_turbo"}},
  performance: {callback_phase: 1, steps: []}
}};
if (api.reinterpretQwenSilentEncode(otherNamespace, snapshot("denoise")).id !== "denoise") {
  throw new Error("Qwen recovery was incorrectly applied to another model");
}

const imageRun = api.normalizeRunMedia({
  status: "completed",
  completed_at: 1,
  settings: {video_length: 81, num_frames: 81},
  stages: {},
  outputs: ["outputs/image.jpg"],
  output_records: [{path: "outputs/image.jpg", media_type: "image", settings: {video_length: 81}}]
});
if (imageRun.media_type !== "image" || imageRun.frame_count !== 1) throw new Error("image media normalization failed");
if ("video_length" in imageRun.settings || "video_length" in imageRun.output_records[0].settings) {
  throw new Error("stale image video length was retained");
}
const audioRun = api.normalizeRunMedia({
  status: "completed",
  completed_at: 1,
  settings: {video_length: 81},
  stages: {},
  outputs: ["outputs/audio.wav"],
  output_records: [{path: "outputs/audio.wav", media_type: "audio", settings: {}}]
});
if (audioRun.media_type !== "audio" || audioRun.frame_count !== null || "video_length" in audioRun.settings) {
  throw new Error("audio media normalization failed");
}

const outcomeNamespace = {activeRun: {notice_baseline: ""}};
api.observeRunOutcome(outcomeNamespace, "RuntimeError: CUDA out of memory. Tried to allocate 2 GiB");
if (outcomeNamespace.activeRun.outcome_status !== "failed" ||
    !outcomeNamespace.activeRun.status_reason.includes("GPU memory")) {
  throw new Error("OOM classification failed");
}

const post = api.freshState().records.post;
post.elapsed = 0;
api.updateStepTiming(post, {current: 0, total: 60}, 1000);
post.elapsed = 1;
api.updateStepTiming(post, {current: 10, total: 60}, 2000);
if (Math.abs(post.stepSeconds - 0.1) > 0.0001) throw new Error("Enhance step timing format source failed");

const exportRun = {
  id: "run-1",
  session_id: "session-1",
  queue_task_id: 1,
  status: "completed",
  started_at: 1000,
  completed_at: 2000,
  duration_seconds: 1,
  settings: {
    model_name: "Krea 2 - Turbo",
    model_filename: "C:/private/models/krea.safetensors",
    resolution: "1280x544",
    prompt: "private prompt",
    negative_prompt: "private negative prompt"
  },
  stages: {denoise: {stage: "denoise", duration_seconds: 1}},
  step_performance: [],
  resources: null,
  step_summary: null,
  media_type: "image",
  frame_count: 1,
  output_count: 1,
  outputs: ["C:/private/outputs/result.jpg"],
  output_records: [{path: "C:/private/outputs/result.jpg", media_type: "image", settings: {prompt: "private prompt"}}]
};
const exportNamespace = {
  runHistory: [exportRun],
  historyScope: "all",
  sessionRunIds: new Set(["run-1"]),
  selectedRunIds: new Set(),
  recoverablePrompts: new Map()
};
const shareFields = new Set(api.EXPORT_PRESETS["share-safe"]);
const shareRecords = api.exportableRuns(exportNamespace, shareFields);
const shareJson = JSON.stringify(shareRecords);
for (const privateValue of ["private prompt", "private negative prompt", "C:/private", "krea.safetensors", "session-1"]) {
  if (shareJson.includes(privateValue)) throw new Error(`Share-safe export leaked ${privateValue}`);
}
const reproducibilityFields = new Set(api.EXPORT_PRESETS.reproducibility);
const reproducibility = api.exportableRuns(exportNamespace, reproducibilityFields);
if (reproducibility[0].prompt !== "private prompt") throw new Error("Reproducibility export omitted selected prompt");
const csv = api.exportCsv(shareRecords, shareFields);
if (!csv.includes('"queue_task_id"')) throw new Error("CSV transformation failed");
const markdown = api.exportMarkdown(shareRecords, shareFields, {scope: "all", preset: "Share-safe"});
if (!markdown.includes("# Status Pro generation history")) throw new Error("Markdown transformation failed");
JSON.parse(JSON.stringify({runs: shareRecords}));
'''
        result = subprocess.run(
            [node, "-"],
            input=javascript + "\n" + test_script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_gallery_matching_is_exact_and_ambiguity_safe(self):
        helpers = _python_functions("_gallery_entry_path", "_gallery_path_keys", "_find_gallery_index")
        find_index = helpers["_find_gallery_index"]
        available = [
            r"D:\outputs\one\clip.mp4",
            r"D:\outputs\two\still.png",
        ]
        self.assertEqual(find_index([r"D:/outputs/one/clip.mp4"], available), 0)
        self.assertEqual(find_index([r"elsewhere\still.png"], available), 1)
        ambiguous = available + [r"E:\archive\still.png"]
        self.assertIsNone(find_index([r"elsewhere\still.png"], ambiguous))

    def test_imported_gallery_paths_are_media_and_output_scoped(self):
        helpers = _python_functions(
            "_media_type_from_path",
            "_gallery_entry_path",
            "_resolve_gallery_import_path",
        )
        resolve = helpers["_resolve_gallery_import_path"]
        with tempfile.TemporaryDirectory() as temporary:
            output_root = pathlib.Path(temporary) / "outputs"
            output_root.mkdir()
            clip = output_root / "clip.mp4"
            clip.write_bytes(b"test")
            resolved, media_type = resolve(str(clip), [str(output_root)])
            self.assertEqual(pathlib.Path(resolved), clip.resolve())
            self.assertEqual(media_type, "video")

            with self.assertRaises(FileNotFoundError):
                resolve(str(output_root / "missing.mp4"), [str(output_root)])
            with self.assertRaises(PermissionError):
                resolve(str(pathlib.Path(temporary) / "outside.mp4"), [str(output_root)])
            unsupported = output_root / "notes.txt"
            unsupported.write_text("not media", encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve(str(unsupported), [str(output_root)])

    def test_imported_media_bridge_requests_import_and_reports_result(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for imported-media bridge validation")
        javascript = _javascript_with_exports("requestGalleryImport", "readGalleryNavigationResult")
        test_script = r'''
const alerts = [];
globalThis.Event = class Event { constructor(type) { this.type = type; } };
globalThis.window = {
  crypto: { randomUUID() { return "import-token"; } },
  alert(message) { alerts.push(String(message)); },
  setTimeout() { return 1; }
};
const requestField = {value: "", dispatchEvent() {}};
const resultField = {value: "{}"};
const trigger = {clicks: 0, click() { this.clicks += 1; }};
const namespace = {
  runHistory: [{
    id: "imported-run",
    imported: true,
    outputs: ["outputs/Videos/example.mp4"],
    output_records: [{path: "outputs/Videos/example.mp4", media_type: "video"}]
  }],
  galleryRequests: new Map(),
  galleryFeedback: new Map(),
  galleryResultRaw: "",
  panel: {querySelectorAll() { return []; }},
  container: {
    querySelector(selector) {
      if (selector.includes("gallery-request-bridge")) return requestField;
      if (selector.includes("gallery-request-trigger")) return trigger;
      if (selector.includes("gallery-result-bridge")) return resultField;
      return null;
    }
  }
};
const api = globalThis.__statusProReleaseTest;
api.requestGalleryImport(namespace, "imported-run", 0);
const request = JSON.parse(requestField.value);
if (request.operation !== "import" || request.outputs.length !== 1 || trigger.clicks !== 1) {
  throw new Error("imported media did not send a gallery import request");
}
resultField.value = JSON.stringify({
  token: request.token,
  run_id: "imported-run",
  status: "imported",
  message: "Imported example.mp4 into Video / Images Gallery."
});
api.readGalleryNavigationResult(namespace);
if (alerts.length !== 1 || !alerts[0].includes("Imported example.mp4")) {
  throw new Error("successful gallery import was not confirmed");
}

window.crypto.randomUUID = () => "missing-token";
api.requestGalleryImport(namespace, "imported-run", 0);
const missingRequest = JSON.parse(requestField.value);
resultField.value = JSON.stringify({
  token: missingRequest.token,
  run_id: "imported-run",
  status: "missing",
  message: "The recorded file is no longer available at its output path. Check the Outputs folder."
});
api.readGalleryNavigationResult(namespace);
if (alerts.length !== 2 || !alerts[1].includes("no longer available")) {
  throw new Error("missing imported media did not display its warning");
}
'''
        result = subprocess.run(
            [node, "-"],
            input=javascript + "\n" + test_script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_history_groups_windows_by_session_and_task(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for history grouping validation")
        javascript = _javascript_with_exports("groupHistoryRuns", "historyTaskSummary")
        test_script = r'''
const api = globalThis.__statusProReleaseTest;
const windows = Array.from({length: 10}, (_, index) => {
  const windowNo = 10 - index;
  return {
    id: `session-a-task-13-window-${windowNo}`,
    session_id: "session-a",
    queue_task_id: 13,
    status: windowNo === 10 ? "completed" : "window",
    window_no: windowNo,
    total_windows: 10,
    started_at: 1000 + (windowNo - 1) * 10000,
    completed_at: 1000 + windowNo * 10000,
    duration_seconds: 10
  };
});
const otherSession = {
  id: "session-b-task-13",
  session_id: "session-b",
  queue_task_id: 13,
  status: "completed",
  started_at: 200000,
  completed_at: 210000,
  duration_seconds: 10
};
const standalone = {
  id: "standalone",
  session_id: "session-a",
  queue_task_id: null,
  status: "completed",
  started_at: 300000,
  completed_at: 301000,
  duration_seconds: 1
};
const groups = api.groupHistoryRuns([...windows, otherSession, standalone]);
if (groups.length !== 3) throw new Error(`expected 3 task groups, received ${groups.length}`);
if (groups[0].runs.length !== 10) throw new Error("ten-window task was not collapsed into one group");
if (groups[0].runs[0].window_no !== 1 || groups[0].runs[9].window_no !== 10) throw new Error("windows were not ordered chronologically");
const complete = api.historyTaskSummary(groups[0]);
if (complete.status !== "completed" || complete.unitLabel !== "10 windows") throw new Error("completed task aggregate is incorrect");
if (Math.abs(complete.duration - 100) > 0.001) throw new Error("task wall-clock duration is incorrect");

const incompleteGroup = api.groupHistoryRuns(windows.filter(run => run.window_no <= 7))[0];
const incomplete = api.historyTaskSummary(incompleteGroup);
if (incomplete.status !== "incomplete" || incomplete.unitLabel !== "7/10 windows") throw new Error("incomplete window aggregate is incorrect");
incompleteGroup.runs[3].status = "failed";
if (api.historyTaskSummary(incompleteGroup).status !== "failed") throw new Error("failed status did not take priority");
'''
        result = subprocess.run(
            [node, "-"],
            input=javascript + "\n" + test_script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_performance_summary_distinguishes_configured_steps_and_passes(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for performance-summary validation")
        javascript = _javascript_with_exports("finalizePerformance", "normalizeRunMedia", "passObservationLabel")
        test_script = r'''
const steps = [];
for (let phase = 1; phase <= 2; phase += 1) {
  for (let step = 1; step <= 20; step += 1) {
    steps.push({
      observer_id: "h3-spectrum",
      sequence: steps.length + 1,
      phase,
      pass_no: -1,
      label: phase === 2 ? "Spectrum smoothing replay" : "",
      step,
      total_steps: 20,
      duration_seconds: phase === 2 && step >= 10 && step <= 13 ? 0.1 : 1,
      skipped_delta: phase === 2 && step >= 10 && step <= 13 ? 1 : 0
    });
  }
}
const run = {step_performance: steps, resources: null};
globalThis.__statusProReleaseTest.finalizePerformance(run);
const summary = run.step_summary;
if (summary.recorded_steps !== 40) throw new Error("callback observations were hidden");
if (summary.observed_passes !== 2 || summary.passes.length !== 2) throw new Error("two Spectrum passes were not identified");
if (summary.passes.some(pass => pass.observed_steps !== 20 || pass.unique_steps !== 20 || pass.configured_steps !== 20)) {
  throw new Error("configured and observed steps were not separated per pass");
}
if (summary.passes[1].label !== "Spectrum smoothing replay") throw new Error("replay label was lost");
if (summary.skipped_steps !== 4) throw new Error("skip count changed while grouping passes");
if (globalThis.__statusProReleaseTest.passObservationLabel(summary) !== "2 × 20 configured steps") {
  throw new Error("equal complete passes lost their configured-step shorthand");
}

const partialPasses = {
  observed_passes: 2,
  passes: [
    {observed_steps: 8, configured_steps: 8},
    {observed_steps: 3, configured_steps: 8}
  ]
};
if (globalThis.__statusProReleaseTest.passObservationLabel(partialPasses) !== "2 passes · 8 + 3 observations") {
  throw new Error("partial passes were presented as a fully observed configured-step total");
}

const ltxSteps = [];
for (let step = 1; step <= 8; step += 1) {
  ltxSteps.push({observer_id: "ltx", sequence: step, phase: 1, pass_no: 1, step, total_steps: 8, duration_seconds: 10});
}
for (let step = 1; step <= 3; step += 1) {
  ltxSteps.push({observer_id: "ltx", sequence: 8 + step, phase: 2, pass_no: 2, step, total_steps: 8, duration_seconds: 30});
}
const ltxLegacy = {
  status: "completed",
  completed_at: 1,
  settings: {},
  stages: {},
  resources: null,
  step_performance: ltxSteps,
  step_summary: {recorded_steps: 11, observed_passes: 2, passes: []},
  outputs: ["result.mp4"],
  output_records: [{path: "result.mp4", media_type: "video", settings: {}}]
};
globalThis.__statusProReleaseTest.normalizeRunMedia(ltxLegacy);
if (ltxLegacy.step_performance.slice(8).some(step => step.total_steps !== 3)) {
  throw new Error("legacy LTX second-phase totals were not repaired");
}
if (ltxLegacy.step_summary.passes[1].configured_steps !== 3) {
  throw new Error("repaired LTX pass summary retained the inherited eight-step total");
}
if (globalThis.__statusProReleaseTest.passObservationLabel(ltxLegacy.step_summary) !== "2 passes · 8 + 3 observations") {
  throw new Error("repaired LTX pass summary was not presented using its true phase totals");
}

const legacy = {
  status: "completed",
  completed_at: 1,
  settings: {},
  stages: {},
  resources: null,
  step_performance: steps,
  step_summary: {recorded_steps: 40, skipped_steps: 4},
  outputs: ["result.jpg"],
  output_records: [{path: "result.jpg", media_type: "image", settings: {}}]
};
globalThis.__statusProReleaseTest.normalizeRunMedia(legacy);
if (legacy.step_summary.observed_passes !== 2 || legacy.step_summary.passes.length !== 2) {
  throw new Error("saved pre-pass-summary history was not upgraded on load");
}
'''
        result = subprocess.run(
            [node, "-"],
            input=javascript + "\n" + test_script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
