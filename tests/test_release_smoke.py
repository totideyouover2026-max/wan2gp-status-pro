import ast
import importlib.util
import inspect
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import types
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
        self.assertEqual(version, "1.1.1")
        self.assertIn(f'self.version = "{version}"', source)
        self.assertIn(f'version: "{version}"', source)
        self.assertEqual(manifest["type"], "extension")
        # WanGP backfills blank metadata from its cached catalogue. A populated
        # zero baseline behaves as no hard minimum without reviving stale values.
        self.assertEqual(manifest["wan2gp_version"], "0")

    def test_status_variants_share_native_source_safely(self):
        source = _source()
        javascript = _returned_string("_javascript")
        self.assertIn('import builtins', source)
        self.assertIn('_register_status_variant("pro")', source)
        self.assertIn('function nativeStatusSource(root, container)', javascript)
        self.assertIn('root.querySelector("#gen_status")', javascript)
        self.assertIn('candidate.id === "status-lite-container"', javascript)
        self.assertNotIn('const source = container.previousElementSibling', javascript)

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
        self.assertIn("width: min(1120px, calc(100vw - 16px))", javascript)
        self.assertIn("height: min(900px, calc(100dvh - 16px))", javascript)
        self.assertIn("status-pro__export-field-column", javascript)
        self.assertIn("columns.length - 1 - ((index - columnCount) % columns.length)", javascript)
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

    def test_stopped_queue_ignores_lingering_abort_and_progress(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for queue-status regression validation")
        javascript = _javascript_with_exports("readLiveSnapshot")
        test_script = r'''
const {readLiveSnapshot} = globalThis.__statusProReleaseTest;
const namespace = {
  state: {currentId: null, overallElapsed: null, steps: {}, records: {}},
  source: {querySelector: selector => selector === "textarea, input" ? {value: "Aborting"} : null, querySelectorAll: () => []},
  download: {active: false, visible: false},
  runTelemetry: {in_progress: true, active_task: {id: 1}, status: "Aborting"}
};
function check(condition, message) { if (!condition) throw new Error(message); }
check(readLiveSnapshot(namespace).aborting, "active abort must remain visible");
namespace.runTelemetry = {in_progress: false, active_task: null, status: "Aborting", queue_length: 0};
for (let tick = 0; tick < 5; tick++) {
  check(readLiveSnapshot(namespace) === null, "stopped queue revived Prepare");
}
namespace.runTelemetry.status = "";
check(readLiveSnapshot(namespace) === null, "stale DOM abort revived Prepare");
namespace.download.visible = true;
namespace.state.currentId = "prepare";
check(readLiveSnapshot(namespace) === null, "completed download revived Prepare");
namespace.runTelemetry.model_lifecycle = {state: "unloaded"};
check(readLiveSnapshot(namespace) === null, "completed unload revived Prepare");
namespace.runTelemetry.model_lifecycle = {state: "unloading"};
check(readLiveSnapshot(namespace).activity === "unload", "live unload was hidden");
namespace.runTelemetry.model_lifecycle = null;
namespace.download.active = true;
namespace.source.querySelector = () => null;
check(readLiveSnapshot(namespace).rawName === "Downloading model files", "live download was hidden");
namespace.download.active = false;
namespace.runTelemetry = {in_progress: true, active_task: {id: 2}, status: "Loading model"};
check(readLiveSnapshot(namespace).rawName === "Loading model", "next queued run was hidden");
namespace.runTelemetry = null;
namespace.source.querySelector = selector => selector === "textarea, input" ? {value: "Aborting"} : null;
check(readLiveSnapshot(namespace).aborting, "missing telemetry disabled DOM fallback");
'''
        result = subprocess.run(
            [node, "-"], input=javascript + "\n" + test_script,
            text=True, encoding="utf-8", capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_review_telemetry_resource_and_export_regressions(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for lifecycle regression validation")
        javascript = _javascript_with_exports(
            "syncRunTelemetry", "startRun", "freshState", "observePerformanceTelemetry",
            "exportFieldValue", "normalizeImportedExport", "renderIdle",
        )
        test_script = r'''
const api = globalThis.__statusProReleaseTest;
const assert = (condition, message) => {if (!condition) throw new Error(message);};
globalThis.window = {localStorage: {getItem: () => null, setItem: () => {}}};
const ns = {
  state: api.freshState(), source: {querySelector: () => null},
  container: {querySelector: () => null}, historyRecording: false,
  runHistory: [], sessionRunIds: new Set(), historyOpen: false,
};
const task = {id: 1, settings: {}};
api.startRun(ns, task, {server_time: 1, in_progress: true, active_task: task});
const originalRun = ns.activeRun;
for (const telemetry of [
  {server_time: 2, error: "temporary snapshot failure"},
  {server_time: 2}, {server_time: 2, in_progress: true, active_task: null},
  {server_time: 2, in_progress: false, error: "snapshot failure"},
]) {
  ns.runTelemetry = telemetry;
  api.syncRunTelemetry(ns);
  assert(ns.activeRun === originalRun, "incomplete telemetry closed the active run");
}
ns.runTelemetry = {server_time: 3, in_progress: true, active_task: task};
api.syncRunTelemetry(ns);
assert(ns.activeRun === originalRun, "telemetry recovery split the run");
ns.runTelemetry = {server_time: 4, in_progress: false, active_task: null};
api.syncRunTelemetry(ns);
assert(ns.activeRun === null, "valid completion did not finish the run");
const run = {queue_task_id: 1, started_at: 0};
const telemetry = {resource_sample: {sampled_at: 2, ram_rss_bytes: 200},
  performance: {id: "p", task_id: 1, steps: [{sequence: 1, completed_at: 1,
    memory: {sampled_at: 1, ram_rss_bytes: 100}}]}};
api.observePerformanceTelemetry(run, telemetry);
api.observePerformanceTelemetry(run, telemetry);
telemetry.resource_sample = {sampled_at: 3, ram_rss_bytes: 400};
api.observePerformanceTelemetry(run, telemetry);
assert(run.resources.sample_count === 2, "periodic sample was counted twice");
assert(run.resources.metrics.ram_rss_bytes.average_bytes === 300, "wrong memory average");
assert(run.step_performance.length === 1, "step observation duplicated");
const imported = api.normalizeImportedExport({version: "1", runs: [{status: "completed"}]})[0];
for (const field of ["started_at", "completed_at"]) {
  assert(api.exportFieldValue(imported, field, new Set()) === null, "missing imported date became epoch");
  for (const value of [null, undefined, "", 1e20]) {
    assert(api.exportFieldValue({[field]: value}, field, new Set()) === null, "invalid date exported");
  }
  assert(api.exportFieldValue({[field]: 0}, field, new Set()) === "1970-01-01T00:00:00.000Z", "valid epoch rejected");
}
const elements = new Map();
ns.panel = {querySelector: key => {
  if (["[data-sp-idle]", "[data-sp-running]", "[data-sp-live]", "[data-sp-idle-title]", "[data-sp-idle-message]"]
      .includes(key)) {if (!elements.has(key)) elements.set(key, {}); return elements.get(key);}
  return null;
}};
ns.historyRecording = true;
for (const status of ["failed", "aborted", "completed"]) {
  ns.runHistory = [{id: "r", status, started_at: 1000, completed_at: 5000, duration_seconds: 4}];
  ns.sessionRunIds = new Set(["r"]);
  api.renderIdle(ns);
  assert(elements.get("[data-sp-idle-message]").textContent.includes(`1 ${status}`), "outcome missing from banner");
  assert(elements.get("[data-sp-live]").textContent === (status === "completed" ? "Complete" : "Queue finished"), "misleading completion headline");
}
'''
        result = subprocess.run([node, "-"], input=javascript + "\n" + test_script,
                                text=True, encoding="utf-8", capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_shared_history_reconciles_deletions_and_serializes_writes(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for shared-history validation")
        javascript = _javascript_with_exports("persistRunHistory", "reconcileRunHistory", "setHistoryPersistence")
        test_script = r'''
(async () => {
const api = globalThis.__statusProReleaseTest;
const assert = (condition, message) => {if (!condition) throw new Error(message);};
const values = new Map();
const storage = {getItem: k => values.get(k) ?? null, setItem: (k, v) => values.set(k, v), removeItem: k => values.delete(k)};
let tail = Promise.resolve();
let lockCalls = 0;
globalThis.window = {localStorage: storage, sessionStorage: storage, navigator: {locks: {
  request: (key, callback) => {lockCalls++; const result = tail.then(callback); tail = result.catch(() => {}); return result;}
}}};
const clone = value => JSON.parse(JSON.stringify(value));
const record = (id, time = 1) => ({id, status: "completed", imported: true, completed_at: time, settings: {}, output_records: []});
const tab = (runs, mode) => ({runHistory: clone(runs), historyBaseline: clone(runs), historyBaselineMode: mode,
  historyPersistence: mode, sessionRunIds: new Set(runs.map(r => r.id)), selectedRunIds: new Set(runs.map(r => r.id)),
  recoverablePrompts: new Map(runs.map(r => [r.id, {}])), openHistoryRuns: new Set(runs.map(r => r.id))});
for (const mode of ["persistent", "runtime"]) {
  values.clear();
  const seed = tab([], mode);
  seed.runHistory = [record("old")];
  await api.persistRunHistory(seed);
  const a = tab(seed.runHistory, mode), b = tab(seed.runHistory, mode);
  a.runHistory = [];
  await api.persistRunHistory(a);
  b.runHistory.unshift(record("new", 2));
  await api.persistRunHistory(b);
  assert(b.runHistory.map(r => r.id).join() === "new", "stale tab resurrected deleted history");
  assert(!b.selectedRunIds.has("old") && !b.recoverablePrompts.has("old"), "deleted row retained references");
  api.reconcileRunHistory(a);
  assert(a.runHistory[0].id === "new", "remote insertion did not synchronize");
  const c = tab(a.runHistory, mode), d = tab(a.runHistory, mode);
  c.runHistory.unshift(record("c", 3));
  d.runHistory.unshift(record("d", 4));
  await Promise.all([api.persistRunHistory(c), api.persistRunHistory(d)]);
  api.reconcileRunHistory(c);
  assert(c.runHistory.map(r => r.id).sort().join() === "c,d,new", "simultaneous saves lost a record");
  d.runHistory = d.runHistory.filter(r => r.id !== "new");
  await api.persistRunHistory(d);
  c.runHistory.unshift(record("e", 5));
  await api.persistRunHistory(c);
  assert(!c.runHistory.some(r => r.id === "new"), "selected deletion was undone");
}
assert(lockCalls > 0, "shared writes bypassed browser lock");
values.clear();
const session = tab([], "browser");
session.runHistory = [record("private")];
const callsBefore = lockCalls;
const result = api.persistRunHistory(session);
assert(result.persisted && lockCalls === callsBefore, "tab-local history used shared synchronization");
assert(await api.setHistoryPersistence(session, "persistent"), "asynchronous mode switch failed");
assert(session.historyPersistence === "persistent", "wrong mode after switch");
window.navigator.locks.request = () => Promise.reject(new Error("unavailable"));
session.runHistory.unshift(record("unsaved"));
const failed = await api.persistRunHistory(session);
assert(!failed.persisted && session.runHistory.some(r => r.id === "unsaved"), "lock failure discarded in-memory history");
})().catch(error => {console.error(error); process.exitCode = 1;});
'''
        result = subprocess.run([node, "-"], input=javascript + "\n" + test_script,
                                text=True, encoding="utf-8", capture_output=True, check=False)
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

    def test_attention_mode_is_recorded_displayed_and_exported(self):
        source = _source()
        for key in ('"attention_mode"', '"override_attention"', '"attention_sparsity"'):
            self.assertIn(key, source)
        for requested_global in (
            'self.request_global("attention_mode")',
            'self.request_global("get_overridden_attention")',
            'self.request_global("get_auto_attention")',
        ):
            self.assertIn(requested_global, source)

        tree = ast.parse(source)
        telemetry_constants = [
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in {"RUN_SETTING_KEYS", "LTX_POSTPROCESSING_MODEL_TYPES", "LTX_POSTPROCESSING_STEPS"}
                for target in node.targets
            )
        ]
        helpers = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {
                "_telemetry_value",
                "_ltx_component_fallbacks",
                "_complete_ltx_components",
                "_normalize_late_postprocessing_settings",
                "_fallback_postprocessing_label",
                "_postprocessing_metadata",
                "_task_telemetry",
            }
        ]
        namespace = {}
        module = ast.Module(body=[*telemetry_constants, *helpers], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(PLUGIN_PATH), "exec"), namespace)
        task_telemetry = namespace["_task_telemetry"]

        sol = task_telemetry(
            {"params": {"model_type": "minimax_h3", "override_attention": "sol", "attention_sparsity": 1.25}},
            attention_mode="sage2",
        )["settings"]
        self.assertEqual(sol["attention_mode"], "sol")
        self.assertEqual(sol["attention_sparsity"], 1.25)
        self.assertNotIn("override_attention", sol)

        automatic = task_telemetry(
            {"params": {"model_type": "wan", "attention_sparsity": 1.0}},
            attention_mode="auto",
            get_auto_attention=lambda: "flash",
        )["settings"]
        self.assertEqual(automatic["attention_mode"], "flash")
        self.assertNotIn("attention_sparsity", automatic)

        model_specific = task_telemetry(
            {"params": {"model_type": "ltx2"}},
            attention_mode="sdpa",
            get_overridden_attention=lambda _model_type: "sage3",
        )["settings"]
        self.assertEqual(model_specific["attention_mode"], "sage3")

        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for attention-history validation")
        javascript = _javascript_with_exports(
            "attentionModeLabel",
            "exportFieldValue",
            "EXPORT_PRESETS",
        )
        test_script = r'''
const api = globalThis.__statusProReleaseTest;
const solSettings = {attention_mode: "sol", attention_sparsity: 1.25};
if (api.attentionModeLabel(solSettings) !== "Sol Attention · tau 1.25") {
  throw new Error("Sol attention and tau were not formatted together");
}
if (api.attentionModeLabel({attention_mode: "flash"}) !== "Flash Attention") {
  throw new Error("Flash Attention was not given a readable label");
}
const run = {settings: solSettings};
if (api.exportFieldValue(run, "attention_mode", new Set()) !== "sol") throw new Error("attention mode export failed");
if (api.exportFieldValue(run, "attention_sparsity", new Set()) !== 1.25) throw new Error("Sol tau export failed");
for (const preset of ["performance", "reproducibility", "share-safe"]) {
  if (!api.EXPORT_PRESETS[preset].includes("attention_mode") || !api.EXPORT_PRESETS[preset].includes("attention_sparsity")) {
    throw new Error(`${preset} preset omits attention settings`);
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

    def test_ltx_late_postprocessing_records_the_model_it_loads(self):
        source = _source()
        tree = ast.parse(source)
        definitions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        constants = [
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in {"RUN_SETTING_KEYS", "LTX_POSTPROCESSING_MODEL_TYPES", "LTX_POSTPROCESSING_STEPS"}
                for target in node.targets
            )
        ]
        helpers = [
            definitions[name]
            for name in (
                "_telemetry_value",
                "_ltx_component_fallbacks",
                "_complete_ltx_components",
                "_normalize_late_postprocessing_settings",
                "_fallback_postprocessing_label",
                "_postprocessing_metadata",
                "_task_telemetry",
            )
        ]
        namespace = {}
        module = ast.Module(body=[*constants, *helpers], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(PLUGIN_PATH), "exec"), namespace)
        task_telemetry = namespace["_task_telemetry"]

        settings = task_telemetry(
            {"params": {
                "mode": "edit_postprocessing",
                "model_type": "minimax_h3_ref2va_pruned_pdd",
                "spatial_upsampling": "ltx252",
                "temporal_upsampling": "rife2",
                "film_grain_intensity": 0.2,
                "film_grain_saturation": 0.5,
                "attention_mode": "sage2",
                "activated_loras": ["stale-h3-lora.safetensors"],
            }},
            get_model_name=lambda model_type: self.fail(f"unexpected display-name lookup for {model_type}"),
            get_model_family=lambda model_type, for_ui=False: "ltx2",
            families_infos={"ltx2": (None, "LTX-2")},
            component_resolver=lambda values: {"prepare": [values["model_type"]]},
        )["settings"]

        self.assertEqual(settings["model_type"], "ltx2_25_22B_distilled")
        self.assertEqual(settings["model_name"], "LTX-2 2.5 Pixel Spatial Upscaler")
        self.assertEqual(settings["model_family"], "LTX-2")
        self.assertEqual(settings["num_inference_steps"], 8)
        self.assertEqual(settings["component_models"]["prepare"], ["ltx2_25_22B_distilled"])
        self.assertEqual(settings["component_models"]["encode"], ["Gemma 4 12B LTX v1 text encoder"])
        self.assertNotIn("base_model_type", settings)
        self.assertNotIn("model_filename", settings)
        self.assertNotIn("activated_loras", settings)
        self.assertEqual(settings["postprocessing"]["application"], "late")
        self.assertEqual(
            [operation["kind"] for operation in settings["postprocessing"]["operations"]],
            ["temporal_upsampling", "spatial_upsampling", "film_grain"],
        )
        self.assertIn("RIFE x2", settings["postprocessing"]["summary"])
        self.assertIn("LTX 2.5 Pixel Spatial Upscaler x2", settings["postprocessing"]["summary"])
        self.assertEqual(
            settings["postprocessing"]["operations"][1]["model"]["model_type"],
            "ltx2_25_22B_distilled",
        )
        self.assertEqual(
            settings["postprocessing"]["operations"][1]["model"]["component_models"]["prepare"],
            ["ltx2_25_22B_distilled"],
        )
        self.assertEqual(settings["postprocessing"]["operations"][1]["steps"], 8)

        ltx23 = task_telemetry({"params": {
            "mode": "edit_postprocessing",
            "model_type": "minimax_h3",
            "spatial_upsampling": "ltx232",
        }})["settings"]
        self.assertEqual(ltx23["model_type"], "ltx2_22B")
        self.assertEqual(ltx23["model_name"], "LTX-2 2.3 Pixel Spatial Upscaler")
        self.assertEqual(
            ltx23["postprocessing"]["operations"][0]["model"]["component_models"]["encode"],
            ["Gemma 3 12B LTX text encoder"],
        )

        model_free = task_telemetry(
            {"params": {
                "mode": "edit_postprocessing",
                "model_type": "minimax_h3",
                "spatial_upsampling": "lanczos2",
                "attention_mode": "sage2",
                "attention_sparsity": 1.25,
                "model_name": "Stale MiniMax H3",
                "model_family": "MiniMax H3",
                "component_models": {"decode": ["MiniMax-H3-video_vae_fp16.safetensors"]},
            }},
            attention_mode="flash",
        )["settings"]
        self.assertNotIn("model_type", model_free)
        self.assertNotIn("model_name", model_free)
        self.assertNotIn("attention_mode", model_free)
        self.assertNotIn("attention_sparsity", model_free)
        self.assertNotIn("model_family", model_free)
        self.assertNotIn("component_models", model_free)
        self.assertEqual(model_free["postprocessing"]["summary"], "Lanczos x2")

        inline = task_telemetry({"params": {
            "mode": "text_to_video",
            "model_type": "minimax_h3_ref2va_pruned_pdd",
            "spatial_upsampling": "ltx252",
            "temporal_upsampling": "rife2",
        }})["settings"]
        self.assertEqual(inline["model_type"], "minimax_h3_ref2va_pruned_pdd")
        self.assertEqual(inline["postprocessing"]["application"], "inline")
        self.assertEqual(len(inline["postprocessing"]["operations"]), 2)

        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for saved-history migration validation")
        javascript = _javascript_with_exports(
            "normalizeRunMedia",
            "normalizeRunSettings",
            "stageModelInfo",
            "runDescriptor",
            "exportFieldValue",
            "EXPORT_PRESETS",
        )
        test_script = r'''
const api = globalThis.__statusProReleaseTest;
const run = {
  status: "completed",
  imported_model_summary: "MiniMax H3 - Ref2VA Pruned PDD 8-Step 20B - Video",
  media_type: "video",
  output_count: 1,
  outputs: ["output_post.mp4"],
  output_records: [{path: "output_post.mp4", media_type: "video", settings: {}}],
  settings: {
    mode: "edit_postprocessing",
    model_type: "minimax_h3_ref2va_pruned_pdd",
    model_name: "MiniMax H3 Ref2VA Pruned PDD 8-Step 20B",
    model_family: "MiniMax H3",
    spatial_upsampling: "ltx252",
    temporal_upsampling: "rife2",
    film_grain_intensity: 0.2,
    film_grain_saturation: 0.5,
    activated_loras: ["stale-h3-lora.safetensors"],
    attention_mode: "sage2",
    component_models: {prepare: ["MiniMax-H3.safetensors"]}
  },
  stages: {},
  step_performance: Array.from({length: 7}, (_, index) => ({
    observer_id: "stale-h3",
    sequence: index + 1,
    phase: 0,
    step: index + 1,
    total_steps: 8,
    duration_seconds: 1
  })),
  step_summary: {recorded_steps: 7, observed_passes: 1, passes: []}
};
api.normalizeRunMedia(run);
if (run.settings.model_type !== "ltx2_25_22B_distilled") throw new Error("saved model type was not repaired");
if (run.settings.model_name !== "LTX-2 2.5 Pixel Spatial Upscaler") throw new Error("saved model name was not repaired");
if (run.settings.model_family !== "LTX-2") throw new Error("saved model family was not repaired");
if (run.settings.num_inference_steps !== 8) throw new Error("saved LTX step count was not repaired");
if (!run.settings.component_models || /MiniMax|Qwen/i.test(JSON.stringify(run.settings.component_models))) {
  throw new Error("stale H3 components survived migration");
}
if (run.settings.attention_mode) throw new Error("stale H3 attention survived migration");
if (run.step_performance.length || run.step_summary) throw new Error("stale H3 step observations survived migration");
if (run.settings.activated_loras) throw new Error("stale H3 LoRAs survived migration");
if (run.imported_model_summary) throw new Error("stale imported summary survived migration");
if (api.runDescriptor(run) !== "RIFE x2 · LTX 2.5 Pixel Spatial Upscaler x2 · Film grain (intensity 0.2, saturation 0.5) - Video") {
  throw new Error(`wrong repaired history label: ${api.runDescriptor(run)}`);
}
if (api.exportFieldValue(run, "model_name", new Set()) !== "LTX-2 2.5 Pixel Spatial Upscaler") {
  throw new Error("repaired model name was not exported");
}
if (api.exportFieldValue(run, "checkpoint", new Set()) !== "ltx2_25_22B_distilled") {
  throw new Error("repaired checkpoint was not exported");
}
const postprocessing = api.exportFieldValue(run, "postprocessing", new Set(["checkpoint"]));
if (postprocessing.application !== "late" || postprocessing.operations.length !== 3) {
  throw new Error("structured late post-processing was not exported");
}
if (postprocessing.operations[1].model.model_type !== "ltx2_25_22B_distilled") {
  throw new Error("LTX backing model was not retained in structured post-processing");
}
const shareSafePostprocessing = api.exportFieldValue(run, "postprocessing", new Set(["postprocessing"]));
if (shareSafePostprocessing.operations[1].model.model_type || shareSafePostprocessing.operations[1].model.component_models) {
  throw new Error("share-safe post-processing leaked exact model metadata");
}

const liveSettings = {
  mode: "edit_postprocessing",
  model_type: "minimax_h3_ref2va_pruned_pdd",
  model_name: "MiniMax H3 Ref2VA Pruned PDD 8-Step 20B",
  spatial_upsampling: "ltx252",
  component_models: {
    input: ["MiniMax-H3-video_vae_fp16.safetensors", "MiniMax-H3-audio_vae_fp32.safetensors"],
    encode: ["Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors"],
    decode: ["MiniMax-H3-video_vae_fp16.safetensors", "MiniMax-H3-audio_vae_fp32.safetensors"]
  },
  postprocessing: {
    application: "late",
    operations: [{
      kind: "spatial_upsampling",
      label: "LTX 2.5 Pixel Spatial Upscaler x2",
      model: {
        model_type: "ltx2_25_22B_distilled",
        model_name: "LTX-2 2.5 Pixel Spatial Upscaler",
        component_models: {
          input: ["ltx-2.5-video-vae-bf16.safetensors", "ltx-2.5-audio-vae-bf16.safetensors"],
          encode: ["gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"],
          decode: ["ltx-2.5-video-vae-bf16.safetensors", "ltx-2.5-audio-vae-bf16.safetensors"]
        }
      }
    }]
  }
};
api.normalizeRunSettings(liveSettings);
if (liveSettings.component_models.encode[0] !== "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors") {
  throw new Error("live Encode did not switch from Qwen to the LTX backing text encoder");
}
if (liveSettings.component_models.decode.some(name => /MiniMax/i.test(name))) {
  throw new Error("live Decode retained stale MiniMax VAEs");
}

const inlineLiveSettings = {
  mode: "text_to_video",
  model_type: "minimax_h3_ref2va_pruned_pdd",
  spatial_upsampling: "ltx252",
  component_models: {
    encode: ["Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors"]
  },
  postprocessing: liveSettings.postprocessing
};
const inlineNamespace = {activeRun: {settings: inlineLiveSettings}};
const generationEncoder = api.stageModelInfo(inlineNamespace, {
  id: "encode",
  rawName: "Encoding prompt",
  rawMessage: "Encoding prompt"
});
if (!generationEncoder || !/Qwen3-VL/.test(generationEncoder.names[0])) {
  throw new Error("inline generation Encode lost its generation-model encoder");
}
const upscalerEncoder = api.stageModelInfo(inlineNamespace, {
  id: "encode",
  rawName: "Upsampling - Window 1 / 2 - Encoding prompt",
  rawMessage: "Upsampling - Window 1 / 2 - Encoding prompt"
});
if (!upscalerEncoder || !/gemma4-12b/.test(upscalerEncoder.names[0])) {
  throw new Error("inline upscaler Encode did not use its LTX backing encoder");
}

const unresolvedLiveSettings = {
  mode: "edit_postprocessing",
  model_type: "minimax_h3_ref2va_pruned_pdd",
  spatial_upsampling: "ltx252",
  component_models: {
    encode: ["Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors"],
    decode: ["MiniMax-H3-video_vae_fp16.safetensors"]
  }
};
api.normalizeRunSettings(unresolvedLiveSettings);
if (!unresolvedLiveSettings.component_models || /Qwen|MiniMax/i.test(JSON.stringify(unresolvedLiveSettings.component_models))) {
  throw new Error("unresolved live LTX components fell back to stale generation-model files");
}
if (!/Gemma 4 12B LTX/.test(unresolvedLiveSettings.component_models.encode[0])) {
  throw new Error("unresolved LTX text encoder did not receive an accurate role fallback");
}

const lateNamespace = {activeRun: {settings: {
  mode: "edit_postprocessing",
  model_type: "minimax_h3_ref2va_pruned_pdd",
  spatial_upsampling: "ltx252",
  component_models: {
    prepare: ["MiniMax-H3-Ref2VA.safetensors"],
    input: ["MiniMax-H3-video_vae_fp16.safetensors"],
    encode: ["Qwen3-VL.safetensors"],
    decode: ["MiniMax-H3-video_vae_fp16.safetensors"]
  }
}}};
for (const [id, rawName, expected] of [
  ["prepare", "Loading model", /LTX-2\.5 22B distilled transformer/],
  ["input", "Upsampling - Window 2 \/ 2 - Audio VAE encoding", /LTX-2\.5 (?:video|audio) VAE/],
  ["encode", "Upsampling - Window 2 \/ 2 - Encoding prompt", /Gemma 4 12B LTX/],
  ["decode", "Upsampling - Window 2 \/ 2 - VAE decoding", /LTX-2\.5 (?:video|audio) VAE/]
]) {
  const info = api.stageModelInfo(lateNamespace, {id, rawName, rawMessage: rawName});
  if (!info || !expected.test(info.names.join(" ")) || /MiniMax|Qwen/i.test(info.names.join(" "))) {
    throw new Error(`late LTX ${id} used generation-model details: ${JSON.stringify(info)}`);
  }
}

const inlineRun = {
  media_type: "video",
  settings: {
    mode: "text_to_video",
    model_type: "minimax_h3_ref2va_pruned_pdd",
    model_name: "MiniMax H3 Ref2VA Pruned PDD 8-Step 20B",
    model_family: "MiniMax H3",
    spatial_upsampling: "ltx252"
  }
};
api.normalizeRunMedia(inlineRun);
const inlineDescriptor = api.runDescriptor(inlineRun);
if (!inlineDescriptor.startsWith("MiniMax H3 - Ref2VA Pruned PDD 8-Step 20B - Video")) {
  throw new Error(`inline processing replaced the generation model: ${inlineDescriptor}`);
}
if (!inlineDescriptor.endsWith("Post: LTX 2.5 Pixel Spatial Upscaler x2")) {
  throw new Error(`inline processor summary is missing: ${inlineDescriptor}`);
}
for (const preset of ["performance", "reproducibility", "share-safe"]) {
  if (!api.EXPORT_PRESETS[preset].includes("postprocessing")) throw new Error(`${preset} omits post-processing`);
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
        source = _source()
        self.assertIn("normalizeRunSettings(settings);", source)
        self.assertIn("normalizeRunSettings(settings);\n        namespace.activeRun.settings = settings;", source)

    def test_postprocessing_progress_retains_the_eighth_ltx_step(self):
        source = _source()
        tree = ast.parse(source)
        definitions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        constants = [
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "MAX_STEP_TELEMETRY" for target in node.targets)
        ]
        helpers = [
            definitions[name]
            for name in (
                "_telemetry_value",
                "_memory_snapshot",
                "_performance_snapshot",
                "_latest_performance_snapshot",
                "_new_performance_observer",
                "_observe_postprocessing_progress",
            )
        ]
        namespace = {"time": time, "_PROCESS": None}
        module = ast.Module(body=[*constants, *helpers], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(PLUGIN_PATH), "exec"), namespace)

        forwarded = []
        performance = namespace["_new_performance_observer"](53)
        callback = namespace["_observe_postprocessing_progress"](
            performance,
            lambda phase, current, total: forwarded.append((phase, current, total)),
        )
        for window in (1, 2):
            phase = f"Window {window} / 2 - Distilled refinement"
            for step in range(1, 9):
                callback(phase, step, 8)

        snapshot = namespace["_latest_performance_snapshot"]({}, performance)
        self.assertEqual(snapshot["task_id"], 53)
        self.assertEqual(len(snapshot["steps"]), 16)
        self.assertEqual(snapshot["steps"][7]["step"], 8)
        self.assertEqual(snapshot["steps"][15]["step"], 8)
        self.assertEqual(snapshot["steps"][15]["phase"], 1)
        self.assertEqual(len(forwarded), 16)

        stale = namespace["_new_performance_observer"]()
        stale["started_at"] = performance["started_at"] - 10
        stale["steps"] = [{"sequence": 1, "step": 1, "total_steps": 8}]
        selected = namespace["_latest_performance_snapshot"](
            {"status_pro_performance": stale},
            performance,
        )
        self.assertEqual(len(selected["steps"]), 16)
        self.assertIn('self.request_global("perform_spatial_upsampling")', source)

    def test_performance_observers_are_bound_to_their_queue_task_and_stale_timing_is_removed(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for performance-observer validation")
        javascript = _javascript_with_exports("observePerformanceTelemetry", "normalizeRunMedia")
        test_script = r'''
const api = globalThis.__statusProReleaseTest;
const run = {
  queue_task_id: 2,
  started_at: 100000,
  completed_at: 130000,
  duration_seconds: 30,
  settings: {generation_time: 300},
  step_performance: [],
  outputs: [],
  output_records: [],
  stages: {}
};
api.observePerformanceTelemetry(run, {
  performance: {
    id: "previous-task-observer",
    task_id: 1,
    steps: [{sequence: 8, step: 8, total_steps: 8, completed_at: 110}]
  }
});
if (run.step_performance.length !== 0) throw new Error("a previous task's final step leaked into the current run");
api.observePerformanceTelemetry(run, {
  performance: {
    id: "current-task-observer",
    task_id: "2",
    steps: [{sequence: 1, step: 1, total_steps: 8, completed_at: 111}]
  }
});
if (run.step_performance.length !== 1 || run.step_performance[0].step !== 1) {
  throw new Error("the current task's observer was not retained");
}
if (run.step_performance[0].observer_task_id !== "2") {
  throw new Error("observer task ownership was not persisted with the step");
}
api.normalizeRunMedia(run);
if (Object.prototype.hasOwnProperty.call(run.settings, "generation_time")) {
  throw new Error("impossible inherited generation time was retained");
}

const plausible = {
  duration_seconds: 300,
  settings: {generation_time: 305},
  step_performance: [],
  outputs: [],
  output_records: [],
  stages: {},
  imported: true,
  status: "completed",
  output_count: 1
};
api.normalizeRunMedia(plausible);
if (plausible.settings.generation_time !== 305) throw new Error("plausible WanGP timing was removed");

const legacyLeak = {
  queue_task_id: 2,
  started_at: 1788118113505,
  duration_seconds: 1283,
  settings: {},
  step_performance: [{
    observer_id: "1788116331085856300",
    sequence: 8,
    step: 8,
    total_steps: 8,
    completed_at: 1788118121
  }],
  outputs: [],
  output_records: [],
  stages: {}
};
api.normalizeRunMedia(legacyLeak);
if (legacyLeak.step_performance.length !== 0) {
  throw new Error("a legacy observer that clearly predates the run was not repaired");
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
    attention_mode: "sol",
    attention_sparsity: 1.25,
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
if (run.settings.attention_mode !== "sol" || run.settings.attention_sparsity !== 1.25) {
  throw new Error("attention export fields were not rebuilt into run settings");
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

    def test_download_wrapper_forwards_current_and_future_arguments(self):
        module = _download_module()
        calls = []
        marker = object()

        def download_file(url, filename, gen=None, show_filename=True, future_option=None):
            calls.append((url, filename, gen, show_filename, future_option))
            if future_option == "fail":
                raise RuntimeError("download failed")
            return filename

        download = types.ModuleType("shared.utils.download")
        download.download_file = download_file
        download.process_files_def = lambda *args, **kwargs: None
        download.create_progress_hook = lambda filename: lambda *args: None
        download.download_def_missing_files = lambda definition: []
        utils = types.ModuleType("shared.utils")
        utils.download = download
        shared = types.ModuleType("shared")
        shared.utils = utils
        names = ("shared", "shared.utils", "shared.utils.download")
        previous = {name: sys.modules.get(name) for name in names}
        sys.modules.update(dict(zip(names, (shared, utils, download))))
        try:
            telemetry = module.DownloadTelemetry()
            observer = module.DownloadObserver(telemetry)
            self.assertTrue(observer._install_shared_download_wrappers())
            wrapped = download.download_file
            self.assertEqual(inspect.signature(wrapped, follow_wrapped=False).parameters["kwargs"].kind,
                             inspect.Parameter.VAR_KEYWORD)
            self.assertEqual(wrapped("https://one", "one.bin"), "one.bin")
            self.assertEqual(wrapped("https://two", "two.bin", gen=marker, show_filename=False), "two.bin")
            self.assertEqual(wrapped(url="https://three", filename="three.bin", gen=marker,
                                     show_filename=False, future_option="future"), "three.bin")
            self.assertEqual(calls[1], ("https://two", "two.bin", marker, False, None))
            self.assertEqual(calls[2], ("https://three", "three.bin", marker, False, "future"))
            self.assertEqual(telemetry.snapshot()["files"][-1]["name"], "three.bin")
            second = module.DownloadObserver(module.DownloadTelemetry())
            self.assertTrue(second._install_shared_download_wrappers())
            self.assertIs(download.download_file, wrapped)
            with self.assertRaisesRegex(RuntimeError, "download failed"):
                wrapped("https://four", "four.bin", future_option="fail")
            failed = telemetry.snapshot()
            self.assertEqual(failed["files"][-1]["name"], "four.bin")
            self.assertEqual(failed["files"][-1]["state"], "failed")
        finally:
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

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

    def test_history_recording_and_v105_history_display_helpers(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for V1.0.5 history behavior validation")
        javascript = _javascript_with_exports(
            "loadHistoryRecordingPreference",
            "setHistoryRecording",
            "historyRecordingConfirmation",
            "compactLoraNames",
            "timingOverviewSegments",
            "stepTimingOutliers",
            "stepIsSkipped",
            "sessionCompletionSummary",
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
if (!api.stepIsSkipped({skipped: true}) || !api.stepIsSkipped({skipped_delta: 2}) || api.stepIsSkipped({skipped: false, skipped_delta: 0})) {
  throw new Error("skipped observation detection is incomplete");
}

const completion = api.sessionCompletionSummary([
  {id: "older", session_id: "s", queue_task_id: 1, started_at: 1000, completed_at: 31000, duration_seconds: 30, repeats: 1},
  {id: "window-1", session_id: "s", queue_task_id: 2, started_at: 40000, completed_at: 45000, duration_seconds: 5, repeats: 1, window_no: 1, total_windows: 2, status: "window"},
  {id: "window-2", session_id: "s", queue_task_id: 2, started_at: 45000, completed_at: 52000, duration_seconds: 7, repeats: 1, window_no: 2, total_windows: 2, status: "completed"}
]);
if (completion.generationCount !== 3 || completion.totalDuration !== 42) throw new Error("session completion totals are incorrect");
if (completion.latestFinishedAt !== 52000 || completion.latestDuration !== 12) throw new Error("latest task did not aggregate all sliding windows");
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
        self.assertIn('completed ? `${formatDuration(latestDuration)}` : ""', javascript_source)
        self.assertIn('filterToggle.setAttribute("role", "switch")', javascript_source)
        self.assertIn('Recorded data and exports are unchanged.', javascript_source)
        self.assertIn('row.hidden = hideSkipped && row.dataset.skipped === "true"', javascript_source)

    def test_stage_media_outcome_and_export_regressions(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for embedded behavior validation")
        javascript = _javascript_with_exports(
            "freshState",
            "applySnapshot",
            "recoverMissedPerformanceStages",
            "stageDurations",
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
  "Distilled refinement": "post",
  "Window 1 / 2 - Distilled refinement": "post",
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

clock = 1000;
Date.now = () => clock;
namespace = {state: api.freshState(), activeRun: {}};
const postStarting = {...snapshot("post"), rawName: "Upsampling - Starting", rawMessage: "Upsampling - Starting"};
const postRefining = {...snapshot("post"), rawName: "Distilled refinement", rawMessage: "Distilled refinement"};
api.applySnapshot(namespace, postStarting);
clock = 3000;
api.applySnapshot(namespace, snapshot("input"));
clock = 5000;
api.applySnapshot(namespace, postRefining);
clock = 8000;
api.applySnapshot(namespace, postRefining);
Date.now = originalNow;
if (namespace.state.currentId !== "post" || Math.abs(namespace.state.records.post.elapsed - 5) > 0.001) {
  throw new Error(`LTX refinement did not resume Enhance timing: ${JSON.stringify(namespace.state.records.post)}`);
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

const inactivityNow = Date.now;
let inactivityClock = 1000;
Date.now = () => inactivityClock;
namespace = {state: api.freshState(), activeRun: {step_performance: []}};
api.applySnapshot(namespace, snapshot("denoise"));
namespace.state.inactiveSince = 1500;
inactivityClock = 10000;
api.applySnapshot(namespace, snapshot("decode"));
Date.now = inactivityNow;
if (namespace.state.records.denoise.state !== "complete" || namespace.state.records.decode.state !== "current") {
  throw new Error("an active task was reset after a background-style tracker gap");
}

const backgroundSteps = Array.from({length: 8}, (_, index) => ({
  observer_id: "background-generation",
  sequence: index + 1,
  phase: 0,
  pass_no: -1,
  step: index + 1,
  total_steps: 8,
  duration_seconds: 2.5,
  completed_at: 100 + index * 2.5
}));
namespace = {
  state: api.freshState(),
  activeRun: {step_performance: backgroundSteps}
};
if (!api.recoverMissedPerformanceStages(namespace, snapshot("decode"))) {
  throw new Error("backgrounded Generate telemetry was not recovered");
}
const recoveredGenerate = namespace.state.records.denoise;
if (recoveredGenerate.state !== "complete" || !recoveredGenerate.recovered || Math.abs(recoveredGenerate.elapsed - 20) > 0.001) {
  throw new Error(`recovered Generate stage is incorrect: ${JSON.stringify(recoveredGenerate)}`);
}
if (recoveredGenerate.stepCurrent !== 8 || recoveredGenerate.stepTotal !== 8 || namespace.state.phaseOrder.length !== 1) {
  throw new Error("recovered Generate steps or phases are incomplete");
}
if (!namespace.state.records.prepare.unreported || !namespace.state.records.encode.unreported) {
  throw new Error("unobserved prerequisite stages were not labelled honestly");
}
const recoveredTimings = api.stageDurations(namespace.state);
if (!Object.values(recoveredTimings).some(stage => stage.stage === "denoise" && stage.duration_seconds === 20)) {
  throw new Error("recovered Generate timing was not retained for History");
}
api.applySnapshot(namespace, snapshot("decode"));
if (namespace.state.records.denoise.state !== "complete" || namespace.state.records.decode.state !== "current") {
  throw new Error("Decode activation erased the recovered Generate stage");
}

const ltxWindowSteps = [];
for (let windowNo = 1; windowNo <= 2; windowNo += 1) {
  for (let stepNo = 1; stepNo <= 8; stepNo += 1) {
    ltxWindowSteps.push({
      observer_id: "background-ltx",
      sequence: ltxWindowSteps.length + 1,
      phase: windowNo - 1,
      pass_no: -1,
      label: `Window ${windowNo} / 2 - Distilled refinement`,
      step: stepNo,
      total_steps: 8,
      duration_seconds: 1,
      completed_at: 200 + ltxWindowSteps.length
    });
  }
}
namespace = {state: api.freshState(), activeRun: {step_performance: ltxWindowSteps}};
api.recoverMissedPerformanceStages(namespace, {...snapshot("save"), rawName: "Saving output"});
const recoveredPost = namespace.state.records.post;
const recoveredPostPhases = namespace.state.phaseOrder.map(id => namespace.state.phases[id]).filter(phase => phase.stage === "post");
if (recoveredPost.stepCurrent !== 16 || recoveredPost.stepTotal !== 16 || recoveredPostPhases.length !== 2) {
  throw new Error("LTX subwindows were not recovered as two phases within one stage");
}
if (recoveredPostPhases[0].label !== "Window 1 / 2 - Distilled refinement" || recoveredPostPhases[1].label !== "Window 2 / 2 - Distilled refinement") {
  throw new Error("recovered LTX subwindow labels were lost");
}

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

    def test_backgrounded_completion_uses_wangp_end_time(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for background completion validation")
        javascript = _javascript_with_exports(
            "inferredRunCompletionTime",
            "repairBackgroundCompletionTiming",
            "finishStage",
            "finishPhase",
        )
        test_script = r'''
globalThis.window = {
  localStorage: { getItem() { return null; }, setItem() {} },
  sessionStorage: { getItem() { return null; }, setItem() {} }
};
const api = globalThis.__statusProReleaseTest;
const startedAt = 1700000000000;
const actualDuration = 14 * 60 + 36;
const resumedAt = startedAt + actualDuration * 1000 + 90 * 60 * 1000;
const baseRun = {started_at: startedAt, total_windows: null};

const statusEnded = api.inferredRunCompletionTime(
  baseRun,
  {active_task: null, status: "Total Generation Time: 14m 36s"},
  resumedAt,
  []
);
if (statusEnded !== startedAt + actualDuration * 1000) {
  throw new Error(`WanGP total duration did not remove the minimized gap: ${statusEnded}`);
}

const outputRecord = {
  path: "outputs/result.mp4",
  settings: {creation_timestamp: (startedAt + actualDuration * 1000) / 1000, generation_time: actualDuration}
};
const outputEnded = api.inferredRunCompletionTime(baseRun, {status: ""}, resumedAt, [outputRecord]);
if (outputEnded !== startedAt + actualDuration * 1000) {
  throw new Error("output creation timestamp was not used as the completion fallback");
}
const multiTaskStatusEnded = api.inferredRunCompletionTime(
  baseRun,
  {active_task: null, status: "Total Generation Time: 44m 36s"},
  resumedAt,
  [outputRecord]
);
if (multiTaskStatusEnded !== startedAt + actualDuration * 1000) {
  throw new Error("a queue-wide duration overrode the task's output completion timestamp");
}

const inheritedTimestampRecord = {
  path: "outputs/result_post.mp4",
  settings: {creation_timestamp: startedAt / 1000 - 3600, generation_time: actualDuration}
};
const postEnded = api.inferredRunCompletionTime(baseRun, {status: ""}, resumedAt, [inheritedTimestampRecord]);
if (postEnded !== startedAt + actualDuration * 1000) {
  throw new Error("late post-processing did not fall back to its operation duration");
}

const retained = {
  status: "completed",
  started_at: startedAt,
  completed_at: resumedAt,
  duration_seconds: (resumedAt - startedAt) / 1000,
  settings: {generation_time: actualDuration},
  output_records: [outputRecord],
  stages: {decode: {label: "Decode", stage: "decode", status: "complete", duration_seconds: 5400}}
};
if (!api.repairBackgroundCompletionTiming(retained)) throw new Error("retained minimized record was not repaired");
if (retained.completed_at !== startedAt + actualDuration * 1000 || retained.duration_seconds !== actualDuration) {
  throw new Error(`retained completion repair is incorrect: ${JSON.stringify(retained)}`);
}
if (!retained.stages.decode.unreported || "duration_seconds" in retained.stages.decode) {
  throw new Error("an impossible background-inflated stage duration was retained");
}

const state = {
  history: {},
  records: {decode: {
    id: "decode", state: "current", startedAt: startedAt + 800000,
    elapsedBase: 0, elapsed: null, progress: null, eta: null, stepTotal: null
  }},
  phases: {decodePhase: {state: "current", startedAt: startedAt + 800000}},
  currentPhaseId: "decodePhase"
};
api.finishStage(state, "decode", startedAt + actualDuration * 1000);
api.finishPhase(state, startedAt + actualDuration * 1000);
if (state.records.decode.elapsed !== 76 || state.phases.decodePhase.elapsed !== 76) {
  throw new Error("stage completion still included browser-resume idle time");
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


    def test_v13_execution_boundary_and_completed_stage_retention(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for execution-boundary validation")
        source = _source()
        for token in ('"status_display":', '"execution_task_known":', '"executing_task": executing_task'):
            self.assertIn(token, source)
        javascript = _javascript_with_exports("syncRunTelemetry", "readLiveSnapshot", "applySnapshot", "freshState", "applyServerStageTiming", "stageElapsedNow")
        script = r"""
const api=globalThis.__statusProReleaseTest,ok=(v,m)=>{if(!v)throw new Error(m)};
globalThis.window={localStorage:{getItem:()=>null,setItem:()=>{}}};
let mono=10000;Object.defineProperty(globalThis,"performance",{value:{now:()=>mono},configurable:true});
const ns={state:api.freshState(),source:{querySelector:()=>null,querySelectorAll:()=>[]},
container:{querySelector:()=>null},download:{active:false,visible:false},historyRecording:false,
runHistory:[],sessionRunIds:new Set(),sessionId:"test",lastExecutingTaskKey:"",
lastExecutionProgressSignature:"",progressEpochReady:true};
const A={id:"A",settings:{}},B={id:"B",settings:{}};
const t=(task,phase,extra={})=>({server_time:10,in_progress:true,execution_task_known:true,
executing_task:task,active_task:task,queue_length:task?1:0,status_display:Boolean(task),status:phase,
progress_phase:[phase,null],native_progress:{phase,current:null,total:null,unit:null,progress:null},...extra});
const sync=x=>{ns.runTelemetry=x;api.syncRunTelemetry(ns)};
sync(t(A,"Saving"));api.applySnapshot(ns,api.readLiveSnapshot(ns));
ok(ns.activeRun.queue_task_id==="A"&&ns.state.currentId==="save","A did not reach Save");
sync(t(null,"Saved",{queue_length:1,status_display:true,output_records:[{path:"result.mp4",settings:{}}]}));
ok(ns.activeRun===null&&ns.state.records.save.state==="complete","A completion was not retained");
ok(ns.completedStateUntil>Date.now()&&api.readLiveSnapshot(ns)===null,"stale Saved survived transition");
sync(t(B,"Saved"));ok(ns.activeRun.queue_task_id==="B"&&!ns.progressEpochReady&&api.readLiveSnapshot(ns)===null,"B inherited A progress");
sync(t(B,"Loading model"));ok(ns.progressEpochReady&&api.readLiveSnapshot(ns).id==="prepare","fresh B progress missing");
const boundary={state:api.freshState(),source:ns.source,container:ns.container,download:ns.download,
 historyRecording:false,runHistory:[],sessionRunIds:new Set(),sessionId:"boundary",lastExecutingTaskKey:"",
 lastExecutionProgressSignature:"",progressEpochReady:true};
const aTiming={task_id:"A",execution_epoch:7,revision:2,last_stage:"save",stages:{
 save:{elapsed:4,active:true,completed:false,run_count:1}}};
boundary.runTelemetry=t(A,"Saving",{stage_timing:aTiming});api.syncRunTelemetry(boundary);
api.applySnapshot(boundary,api.readLiveSnapshot(boundary));
const closedSave=boundary.state.records.save;
ok(closedSave.serverActive&&boundary.state.currentId==="save","Task A Save timing was not active");
mono+=1000;
const staleA={...aTiming,revision:3,stages:{save:{elapsed:5,active:true,completed:true,run_count:1}}};
boundary.runTelemetry=t(B,"Encoding Text Prompt",{server_time:11,stage_timing:staleA});api.syncRunTelemetry(boundary);
const bSnapshot=api.readLiveSnapshot(boundary);api.applySnapshot(boundary,bSnapshot);
ok(boundary.activeRun.queue_task_id==="B"&&bSnapshot.id==="encode"&&boundary.state.currentId==="encode","Task B did not display Encode");
ok(!boundary.state.records.save.hasRun&&!boundary.state.records.save.hasCompleted&&!boundary.state.records.save.isActive,"stale Task-A timing created Save in Task B");
ok(api.applyServerStageTiming(boundary,{stage_timing:staleA})===false,"stale Task-A timing was accepted");
const stopped=api.stageElapsedNow(closedSave);mono+=5000;
ok(api.stageElapsedNow(closedSave)===stopped&&!closedSave.serverActive,"Task A Save kept accumulating past its boundary");
const legacy={...t(A,"Preparing")};delete legacy.execution_task_known;delete legacy.executing_task;
legacy.active_task=A;ns.activeRun=null;sync(legacy);ok(ns.activeRun.queue_task_id==="A","legacy fallback regressed");
"""
        result = subprocess.run([node, "-"], input=javascript + "\n" + script,
                                text=True, encoding="utf-8", capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_authoritative_worker_outcomes_and_task_owned_late_outputs(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for task-outcome validation")
        javascript = _javascript_with_exports(
            "freshState", "startRun", "finishRun", "applyServerStageTiming",
            "normalizeRunMedia", "reconcileTaskOutcomes", "taskOutcomeForRun",
        )
        script = r'''
const api=globalThis.__statusProReleaseTest,ok=(v,m)=>{if(!v)throw new Error(m)};
globalThis.window={localStorage:{getItem:()=>null,setItem:()=>{}},sessionStorage:{getItem:()=>null,setItem:()=>{}},navigator:{}};
const source={querySelector:()=>null,querySelectorAll:()=>[]};
function make(id,epoch){
 const ns={state:api.freshState(),source,sessionId:"outcomes",historyRecording:false,runHistory:[],sessionRunIds:new Set(),
  recoverablePrompts:new Map(),promptMemory:true,historyPersistence:"browser"};
 const task={id,settings:{}};
 const telemetry={server_time:10,in_progress:true,execution_task_known:true,executing_task:task,
  output_records:[],stage_timing:{task_id:id,execution_epoch:epoch,revision:1,last_stage:"save",
   stages:{save:{elapsed:3.2,active:false,completed:true,run_count:1}}}};
 api.startRun(ns,task,telemetry);api.applyServerStageTiming(ns,telemetry);return {ns,telemetry,run:ns.activeRun};
}
const success=make("A",11);
success.telemetry.task_outcomes=[{task_id:"A",execution_epoch:11,known:true,success:true,aborted:false,output_records:[]}];
api.finishRun(success.ns,"failed",11000,success.telemetry);
ok(success.run.status==="completed","worker success with delayed outputs was marked failed");
ok(success.run.stages.save&&success.run.stages.save.status==="complete","successful Save became failed");
const failed=make("F",12);failed.telemetry.task_outcomes=[{task_id:"F",execution_epoch:12,known:true,success:false,aborted:false,error:"worker exploded",output_records:[]}];
api.finishRun(failed.ns,"completed",11000,failed.telemetry);ok(failed.run.status==="failed"&&failed.run.failure_reason.includes("worker exploded"),"worker failure was lost");
const aborted=make("X",13);aborted.telemetry.task_outcomes=[{task_id:"X",execution_epoch:13,known:true,success:false,aborted:true,output_records:[]}];
api.finishRun(aborted.ns,"completed",11000,aborted.telemetry);ok(aborted.run.status==="aborted","worker abort was misclassified");
const legacy={status:"completed",completed_at:1000,settings:{},stages:{},outputs:[],output_records:[]};
api.normalizeRunMedia(legacy);ok(legacy.status==="completed","empty output discovery still implied failure");
const history={historyRecording:false,runHistory:[
 {id:"run-b",queue_task_id:"B",execution_epoch:22,status:"running",settings:{},stages:{},outputs:[],output_records:[]},
 {id:"run-a",queue_task_id:"A",execution_epoch:21,status:"completed",settings:{},stages:{},outputs:[],output_records:[]}
]};
api.reconcileTaskOutcomes(history,{task_outcomes:[
 {task_id:"A",execution_epoch:21,known:true,success:true,aborted:false,
  output_records:[{path:"outputs/a.png",media_type:"image",settings:{prompt:"A"}}]}
]});
ok(history.runHistory[1].outputs.join()==="outputs/a.png","late Task A output did not enrich Task A");
ok(history.runHistory[0].outputs.length===0,"late Task A output attached to Task B");
'''
        result = subprocess.run([node, "-"], input=javascript + "\n" + script,
                                text=True, encoding="utf-8", capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_repeated_pipeline_passes_preserve_monotonic_stage_completion(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for repeated-pass validation")
        javascript = _javascript_with_exports("freshState", "applySnapshot", "startRun")
        script = r"""
const api=globalThis.__statusProReleaseTest,ok=(v,m)=>{if(!v)throw new Error(m)};
globalThis.window={localStorage:{getItem:()=>null,setItem:()=>{}}};
const make=()=>({state:api.freshState(),activeRun:{settings:{},step_performance:[]},
 source:{querySelector:()=>null,querySelectorAll:()=>[]},sessionId:"test"});
const snap=(id,name,evidence="structured",aborting=false)=>({id,rawName:name,rawMessage:name,
 progress:null,steps:{current:null,total:null,unit:null},stageElapsed:null,overallElapsed:null,
 transitionEvidence:evidence,aborting});
const ns=make();
api.applySnapshot(ns,snap("denoise","Denoising first phase"));
api.applySnapshot(ns,snap("decode","VAE Decoding"));
api.applySnapshot(ns,snap("post","Spatial refinement"));
api.applySnapshot(ns,snap("denoise","Denoising second phase"));
ok(ns.state.records.denoise.hasCompleted&&ns.state.records.denoise.isActive,"Generate cannot be complete and active");
ok(ns.state.records.decode.hasCompleted&&ns.state.records.post.hasCompleted,"later Generate erased downstream completion");
ok(ns.state.records.decode.state==="complete"&&ns.state.records.post.state==="complete","downstream ticks disappeared");
ok(ns.state.phaseOrder.filter(id=>ns.state.phases[id].stage==="denoise").length===2,"new Generate pass not tracked separately");
api.applySnapshot(ns,snap("decode","VAE Decoding second pass"));
api.applySnapshot(ns,snap("save","Saving"));
ok(ns.state.records.denoise.hasCompleted&&ns.state.records.decode.hasCompleted,"repeated stages lost completion");
ok(ns.state.records.save.isActive,"final Save not displayed");
const stale=make();
for(const s of [snap("encode","Encoding Text Prompt"),snap("denoise","Denoising"),snap("decode","VAE Decoding")]) api.applySnapshot(stale,s);
const phaseCount=stale.state.phaseOrder.length;
ok(api.applySnapshot(stale,snap("denoise","Denoising","legacy-dom"))===false,"stale Generate rewind accepted");
ok(stale.state.currentId==="decode"&&stale.state.phaseOrder.length===phaseCount,"stale Generate mutated phases");
ok(api.applySnapshot(stale,snap("encode","Encoding Text Prompt","legacy-dom"))===false,"stale Encode rewind accepted");
const aborting=make();
for(const s of [snap("denoise","Denoising first phase"),snap("decode","VAE Decoding"),
 snap("post","Spatial refinement"),snap("denoise","Denoising second phase")]) api.applySnapshot(aborting,s);
api.applySnapshot(aborting,snap("denoise","Aborting","structured",true));
ok(aborting.state.records.denoise.hasCompleted&&aborting.state.records.denoise.state==="aborting","repeat-pass abort erased completion");
ok(aborting.state.records.decode.hasCompleted&&aborting.state.records.post.hasCompleted,"abort erased earlier stages");
api.startRun(ns,{id:"B",settings:{}},{server_time:2});
ok(ns.state.currentId==="prepare"&&ns.state.records.prepare.isActive,"new task did not activate Prepare");
ok(Object.values(ns.state.records).filter(r=>r.id!=="prepare").every(r=>!r.hasRun&&!r.hasCompleted&&!r.isActive),"new task inherited stage history");
"""
        result = subprocess.run([node, "-"], input=javascript + "\n" + script,
                                text=True, encoding="utf-8", capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


    def test_stable_stage_slots_and_structured_v13_authority(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for stage-presentation validation")
        javascript = _javascript_with_exports(
            "freshState", "applySnapshot", "renderStages", "readLiveSnapshot", "structuredStageId",
            "applyServerStagePlan", "startRun", "normalizePlannedStages", "STAGE_DEFS",
        )
        script = r"""
const api=globalThis.__statusProReleaseTest,ok=(v,m)=>{if(!v)throw new Error(m)};
globalThis.window={localStorage:{getItem:()=>null,setItem:()=>{}}};
function element(tag="div"){
 const node={tagName:tag,dataset:{},style:{},children:[],attributes:{},className:"",disabled:false,title:"",
  classList:{values:new Set(),toggle(name,on){if(on)this.values.add(name);else this.values.delete(name)},contains(name){return this.values.has(name)}},
  append(...items){items.forEach(item=>this.children.push(item))},appendChild(item){this.children.push(item);return item},
  setAttribute(name,value){this.attributes[name]=String(value)},
  querySelectorAll(selector){return selector==="[data-stage-id]"?this.children.filter(x=>x.dataset.stageId):[]},
  querySelector(selector){const cls=selector.replace(/^\./,"");return this.children.find(x=>x.className===cls)||null}};
 return node;
}
globalThis.document={createElement:element};
ok(api.normalizePlannedStages(["Prepare","Generate","Enhance"]).join(",")==="prepare,denoise,post","display aliases created noncanonical stage IDs");
const container=element();container.clientWidth=2000;
const ns={state:api.freshState(),activeRun:{queue_task_id:"A",_stageTimingEpoch:1,_stagePlanEpoch:null,settings:{},step_performance:[]},
 panel:{querySelector:s=>s==="[data-sp-stages]"?container:null},
 source:{querySelector:()=>null,querySelectorAll:()=>[]},sessionId:"plans"};
api.renderStages(ns);
ok(container.children.map(x=>x.dataset.stageId).join(",")==="prepare,encode,denoise,decode","fallback layout has a global gap");
ok(container.children.map(x=>x.dataset.stagePosition).join(",")==="1,2,3,4","fallback numbering is not contiguous");
const plan=(id,epoch,stages)=>({planned_stages:stages,planned_stage_task_id:id,
 planned_stage_execution_epoch:epoch,planned_stage_revision:epoch,
 stage_timing:{task_id:id,execution_epoch:epoch,revision:1,last_stage:"prepare",stages:{}}});
ok(api.applyServerStagePlan(ns,plan("A",1,["prepare","encode","denoise","decode"])),"Task A plan rejected");
api.renderStages(ns);
let refs=[...container.children];
const snap=(id,name,evidence="structured")=>({id,rawName:name,rawMessage:name,progress:null,
 steps:{current:null,total:null,unit:null},stageElapsed:null,overallElapsed:null,transitionEvidence:evidence,aborting:false});
for(const s of [snap("encode","Encoding Text Prompt"),snap("denoise","Denoising"),snap("decode","VAE Decoding")]){
 api.applySnapshot(ns,s);api.renderStages(ns);
 ok(container.children.every((node,index)=>node===refs[index]),"existing stage node was reordered");
}
const before=container.children.length;
api.applySnapshot(ns,snap("denoise","Denoising second phase"));api.renderStages(ns);
ok(container.children.length===before&&ns.state.records.denoise.runCount===2,"repeated Generate duplicated its card");
api.applySnapshot(ns,snap("post","Unexpected upscaling"));api.renderStages(ns);
ok(container.children.slice(0,4).every((node,index)=>node===refs[index]),"unexpected Enhance reordered planned nodes");
ok(container.children[4].dataset.stageId==="post"&&container.children[4].dataset.stagePosition==="5","unexpected Enhance fallback was not deterministic");
api.startRun(ns,{id:"B",settings:{}},plan("B",2,["prepare","encode","input","denoise","decode","post"]));
api.renderStages(ns);
ok(container.children.map(x=>x.dataset.stageId).join(",")==="prepare,encode,input,denoise,decode,post","Task B plan was not rebuilt");
ok(container.children.map(x=>x.dataset.stagePosition).join(",")==="1,2,3,4,5,6","Task B numbering is not contiguous");
ok(api.applyServerStagePlan(ns,plan("A",1,["prepare","encode","denoise","decode"]))===false,"stale Task A plan was accepted by Task B");
refs=[...container.children];
for(const s of [snap("encode","Encoding Text Prompt"),snap("input","VAE Encoding"),snap("denoise","Denoising")]){
 ok(api.applySnapshot(ns,s)!==false,"Task B rejected its planned Encode to Inputs order");api.renderStages(ns);
 ok(container.children.every((node,index)=>node===refs[index]),"Task B planned nodes were recreated during progress");
}
api.applySnapshot(ns,snap("save","Saving"));api.renderStages(ns);
ok(container.children.slice(0,6).every((node,index)=>node===refs[index]),"runtime fallback reordered planned nodes");
ok(container.children[6].dataset.stageId==="save"&&container.children[6].dataset.stagePosition==="7","unexpected Save was not appended deterministically");
ok(api.structuredStageId("VAE Encoding")==="input","VAE Encode structured mapping");
ok(api.structuredStageId("Encoding Text Prompt 1/2")==="encode","prompt Encode structured mapping");
ok(api.structuredStageId("VAE Decoding")==="decode","VAE Decode structured mapping");
const live={state:api.freshState(),activeRun:{settings:{},step_performance:[]},download:{active:false,visible:false},
 source:{querySelector:()=>null,querySelectorAll:()=>[]}};
const phase=(name,status)=>({in_progress:true,execution_task_known:true,executing_task:{id:1},
 active_task:{id:1},native_progress:{phase:name,current:null,total:null,unit:null,progress:null},
 progress_phase:[name,null],status});
live.runTelemetry=phase("VAE Decoding","Generating");
ok(api.readLiveSnapshot(live).id==="decode","stale Generate text overrode structured Decode");
live.runTelemetry=phase("Denoising","Encoding prompt");
ok(api.readLiveSnapshot(live).id==="denoise","stale Encode text overrode structured Generate");
api.applySnapshot(live,snap("decode","VAE Decoding"));
live.runTelemetry=phase("Uncatalogued accelerator phase","Encoding prompt");
const unknown=api.readLiveSnapshot(live);
ok(unknown.id==="decode"&&unknown.transitionEvidence==="structured-unknown","unknown structured phase rewound");
"""
        result = subprocess.run([node, "-"], input=javascript + "\n" + script,
                                text=True, encoding="utf-8", capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


    def test_authoritative_stage_timing_recovery_and_interpolation(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for authoritative timing validation")
        source = _source()
        self.assertIn('stage_timing = self._stage_timing.snapshot', source)
        javascript = _javascript_with_exports(
            "freshState", "applySnapshot", "applyServerStageTiming", "stageElapsedNow", "stageTimeText", "startRun",
        )
        script = r"""
const api=globalThis.__statusProReleaseTest,ok=(v,m)=>{if(!v)throw new Error(m)};
let mono=30000;
Object.defineProperty(globalThis,"performance",{value:{now:()=>mono},configurable:true});
globalThis.window={localStorage:{getItem:()=>null,setItem:()=>{}}};
const ns={state:api.freshState(),activeRun:{queue_task_id:"A",_stageTimingEpoch:null,settings:{},step_performance:[]},sessionId:"timing",
 source:{querySelector:()=>null,querySelectorAll:()=>[]}};
const timing=(stages,last="denoise",revision=1)=>({stage_timing:{task_id:"A",execution_epoch:1,revision,last_stage:last,stages}});
api.applyServerStageTiming(ns,timing({
 input:{elapsed:2,active:false,completed:true,run_count:1},
 encode:{elapsed:7,active:false,completed:true,run_count:1},
 denoise:{elapsed:15,active:true,completed:false,run_count:1}
}));
ok(ns.state.records.input.hasCompleted&&ns.state.records.input.elapsed===2,"missed Inputs not reconstructed");
ok(ns.state.records.encode.hasCompleted&&ns.state.records.encode.elapsed===7,"polling delay inflated Encode");
ok(ns.state.currentId==="denoise"&&ns.state.records.denoise.elapsed===15,"active Generate not recovered");
ok(Object.values(ns.state.records).filter(r=>r.isActive).length===1,"authoritative timing left multiple stages active");
ok(api.stageTimeText(ns.state,ns.state.records.denoise).includes("15s elapsed"),"Generate card did not use denoise timing");
api.applySnapshot(ns,{id:"denoise",rawName:"Denoising",rawMessage:"Denoising",progress:12.5,
 steps:{current:1,total:8,unit:"steps"},stageElapsed:null,overallElapsed:null,transitionEvidence:"structured",aborting:false});
ok(Number.isFinite(ns.state.records.denoise.eta)&&ns.state.records.denoise.eta>0,"Generate ETA did not use authoritative denoise elapsed");
mono+=2000;
ok(Math.abs(api.stageElapsedNow(ns.state.records.denoise)-17)<0.001,"local monotonic interpolation failed");
api.applyServerStageTiming(ns,timing({denoise:{elapsed:50,active:true,completed:false,run_count:1}},"denoise",2));
ok(Math.abs(api.stageElapsedNow(ns.state.records.denoise)-50)<0.001,"fresh server timing did not reconcile");
mono+=2000;
ok(Math.abs(api.stageElapsedNow(ns.state.records.denoise)-52)<0.001,"interpolation did not resume");
api.applyServerStageTiming(ns,timing({denoise:{elapsed:18,active:false,completed:true,run_count:2}},"denoise",3));
ok(ns.state.records.denoise.elapsed===18&&ns.state.records.denoise.runCount===2,"repeated Generate timing lost");
const before=ns.state.records.denoise.runCount;
api.applySnapshot(ns,{id:"decode",rawName:"VAE Decoding",rawMessage:"",steps:{},progress:null,
 stageElapsed:null,overallElapsed:null,transitionEvidence:"structured",aborting:false});
const phaseCount=ns.state.phaseOrder.length;
ok(api.applySnapshot(ns,{id:"denoise",rawName:"Denoising",rawMessage:"",steps:{},progress:null,
 stageElapsed:null,overallElapsed:null,transitionEvidence:"legacy-dom",aborting:false})===false,"stale rewind accepted");
ok(ns.state.records.denoise.runCount===before&&ns.state.phaseOrder.length===phaseCount,"stale rewind altered timing");
api.applyServerStageTiming(ns,timing({save:{elapsed:3,active:false,completed:true,run_count:1}},"save",4));
mono+=1600;
ok(api.stageElapsedNow(ns.state.records.save)===3,"completion grace inflated Save");
api.applyServerStageTiming(ns,timing({denoise:{elapsed:12,active:false,completed:false,run_count:1}},"denoise",5));
api.applySnapshot(ns,{id:"denoise",rawName:"Aborting",rawMessage:"Aborting",steps:{},progress:null,
 stageElapsed:null,overallElapsed:null,transitionEvidence:"structured",aborting:true});
mono+=5000;
ok(api.stageElapsedNow(ns.state.records.denoise)===12&&ns.state.records.denoise.state==="aborting","abort timing continued");
api.startRun(ns,{id:"B",settings:{}},{server_time:2});
ok(ns.state.currentId==="prepare"&&ns.state.records.prepare.isActive&&!ns.state.records.prepare.serverTimed,"new task Prepare missing");
ok(Object.values(ns.state.records).filter(r=>r.id!=="prepare").every(r=>!r.serverTimed&&!r.hasRun),"task timing leaked");
const legacy={state:api.freshState(),activeRun:{},source:ns.source};
const oldNow=Date.now;let wall=1000;Date.now=()=>wall;
api.applySnapshot(legacy,{id:"encode",rawName:"Encoding prompt",rawMessage:"",steps:{},progress:null,
 stageElapsed:null,overallElapsed:null,aborting:false});
wall=11000;
api.applySnapshot(legacy,{id:"denoise",rawName:"Denoising",rawMessage:"",steps:{},progress:null,
 stageElapsed:null,overallElapsed:null,aborting:false});
Date.now=oldNow;
ok(Math.abs(legacy.state.records.encode.elapsed-10)<0.001,"legacy frontend timing fallback broke");
"""
        result = subprocess.run([node, "-"], input=javascript + "\n" + script,
                                text=True, encoding="utf-8", capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
