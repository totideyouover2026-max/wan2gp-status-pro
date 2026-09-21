import shutil
import subprocess
import unittest
from test_release_smoke import _javascript_with_exports, _source


class HistoryPhaseTests(unittest.TestCase):
    def test_pipeline_counters_roundtrip_and_existing_history_rendering(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required for History validation")
        javascript = _javascript_with_exports(
            "aggregateStageDurations", "recordedPhaseEntries", "historyPhaseLabel", "historyPhaseChipText",
            "stageDurations", "freshState", "applySnapshot", "finishHistoryPhases", "startRun", "finishRun",
            "buildExportRecord", "normalizeImportedExport", "createHistoryRun", "stageIdFor", "applyServerStageTiming",
        )
        script = r"""
const api = globalThis.__statusProReleaseTest;
const assert = (condition, message) => {if (!condition) throw new Error(message);};
const clone = value => JSON.parse(JSON.stringify(value));
const phases = {
    "prepare:loading": {stage: "prepare", label: "Loading model", duration_seconds: 29},
    "prepare:loaded": {stage: "prepare", label: "Model loaded", duration_seconds: 3},
    "input:refs": {stage: "input", label: "VAE Encoding", duration_seconds: 2},
    "encode:h3": {label: "Encoding H3 prompt and references", duration_seconds: 2},
    "encode:text": {stage: "encode", label: "Encoding Text Prompt", duration_seconds: 33, current: 32, total: 32, unit: "layers", progress: 100, status: "complete"},
    "denoise:main": {stage: "denoise", label: "Denoising", duration_seconds: 124, current: 8, total: 8, unit: "steps", progress: 100, status: "complete"},
    "decode:vae": {stage: "decode", label: "VAE Decoding", duration_seconds: 28, current: 12, total: 12, unit: "tiles", progress: 100, status: "complete"},
    "decode:h3": {stage: "decode", label: "Decoding H3 Stereo Audio", duration_seconds: 3},
    "save:file": {stage: "save", label: "Saving File", duration_seconds: 3}
};
const totals = api.aggregateStageDurations(phases);
assert(JSON.stringify(totals) === JSON.stringify({prepare:32, input:2, encode:35, denoise:124, decode:31, post:0, save:3}), "wrong pipeline totals");
assert(api.aggregateStageDurations({old: {label: "VAE Encoding", duration_seconds: 6}}).input === 6, "legacy stage classifier differs from live");
const nested = {...phases, encode: {label: "Encode", duration_seconds: 35}};
assert(api.aggregateStageDurations(nested).encode === 35, "parent total double-counted");
assert(api.aggregateStageDurations({prepare:{label:"Loading model",duration_seconds:29}, "prepare:loaded":{stage:"prepare",label:"Model loaded",duration_seconds:3}}).prepare===32, "distinct loading phase mistaken for parent total");
const duplicated = {a: {id:"same", stage:"encode", duration_seconds:2}, b: {id:"same", stage:"encode", duration_seconds:2}};
assert(api.aggregateStageDurations(duplicated).encode === 2, "duplicate phase ID counted twice");
assert(Object.values(api.aggregateStageDurations({bad:null, unknown:{label:"Other", duration_seconds:NaN}})).every(v=>v===0), "malformed legacy phases crash or invent duration");
assert(api.historyPhaseChipText(phases["encode:text"]).includes("32/32 layers"), "layer counter missing");
assert(api.historyPhaseChipText(phases["decode:vae"]).includes("12/12 tiles"), "tile counter missing");
assert(api.historyPhaseChipText(phases["denoise:main"]).includes("8/8 steps"), "step counter missing");
assert(!api.historyPhaseChipText(phases["prepare:loading"]).includes("/"), "fabricated counter");
const multi = {first:{stage:"encode", label:"Encoding Text Prompt 1/2", duration_seconds:3}, second:{stage:"encode", label:"Encoding Text Prompt 2/2", duration_seconds:4}};
assert(api.recordedPhaseEntries(multi).length === 2 && api.aggregateStageDurations(multi).encode === 7, "multi-prompt phases collapsed");
const filename = "2026-09-13-21h01m08s_seed155659142_integrated_multimodal_description [Shot 1] A five.mp4";
assert(api.historyPhaseLabel(`Saving File ${filename}`) === "Saving File", "dynamic save filename displayed");
assert(api.historyPhaseLabel("Saving File D:\\outputs\\render.mp4") === "Saving File", "save path displayed");
for (const label of ["Decoding H3 Stereo Audio", "Encoding Text Prompt 1/2", "Saving File metadata", "Loading model"]) {
    assert(api.historyPhaseLabel(label) === label, `legitimate label damaged: ${label}`);
}
globalThis.window = {localStorage: {getItem: () => null, setItem: () => {}}, sessionStorage: {getItem: () => null, setItem: () => {}}};
function ns() {return {state:api.freshState(), runHistory:[], sessionRunIds:new Set(), selectedRunIds:new Set(),
    openHistoryRuns:new Set(), recoverablePrompts:new Map(), galleryFeedback:new Map(),
    source:{querySelector:()=>null}, historyPersistence:"browser", sessionId:"session", promptMemory:false};}
for (const outcome of ["aborted", "failed", "incomplete"]) {
    const state = api.freshState();
    const context = {state, activeRun:{settings:{}}};
    api.applySnapshot(context, {id:"encode", rawName:"Encoding Text Prompt", rawMessage:"Encoding Text Prompt", progress:17/32*100,
        progressScope:"phase", steps:{current:17,total:32,unit:"layers"}});
    const phase = state.phases[state.currentPhaseId];
    api.finishHistoryPhases(state, outcome, phase.startedAt+5000);
    const ledger = api.stageDurations(state);
    const saved = Object.values(ledger).find(p=>p.stage==="encode");
    assert(saved.current===17 && saved.total===32 && saved.progress<100, "failed/aborted phase counter completed");
    assert(saved.status===outcome && saved.duration_seconds===5, "partial phase outcome/duration lost");
    assert(api.aggregateStageDurations(ledger).encode===5, "partial duration excluded");
}
for (const outcome of ["aborted", "failed"]) {
    const partial = ns();
    api.startRun(partial, {id:9,settings:{}}, {server_time:100});
    api.applySnapshot(partial,{id:"encode",rawName:"Encoding Text Prompt",rawMessage:"Encoding Text Prompt",
        progress:17/32*100, progressScope:"phase",steps:{current:17,total:32,unit:"layers"}});
    api.finishRun(partial,outcome,110000,{server_time:110,in_progress:false,output_records:[]});
    const recorded = partial.runHistory[0];
    const phase = Object.values(recorded.stages).find(p=>p.stage==="encode");
    assert(recorded.status===outcome && phase.current===17 && phase.total===32 && phase.status===outcome, "recording partial run completed its phase");
    const roundtrip = api.normalizeImportedExport({version:"1.1.0",runs:[api.buildExportRecord(recorded,new Set(["run_id","status","phase_timings"]))]})[0];
    assert(Object.values(roundtrip.stages).find(p=>p.stage==="encode").current===17, "partial roundtrip counter changed");
}
const context = ns();
api.startRun(context, {id:1,settings:{}}, {server_time:100, wangp_version:"13.0"});
assert(context.activeRun.wangp_version === "13.0", "native version not attached");
context.activeRun.step_performance = Array.from({length:8},(_,i)=>({step:i+1,total_steps:8,sequence:i+1,phase:0,duration_seconds:15.5}));
context.state.phases = Object.fromEntries(Object.entries(phases).map(([id,p])=>[id,{...p, id, elapsed:p.duration_seconds,state:"complete"}]));
context.state.phaseOrder = Object.keys(phases);
api.applySnapshot(context, {id:"save",rawName:`Saving File ${filename}`,rawMessage:`Saving File ${filename}`, steps:{current:null,total:null},progress:null});
const telemetry = {server_time:200,in_progress:false,output_records:[{path:filename,media_type:"video",settings:{}}]};
api.finishRun(context,"completed",200000,telemetry);
const run = context.runHistory[0];
assert(run.wangp_version === "13.0", "version lost at completion");
assert(run.step_performance.length === 8, "phase counters became Step Observations");
assert(Object.values(run.stages).some(p=>p.raw_label===`Saving File ${filename}` && p.label==="Saving File"), "raw save label not retained separately");
const exported = api.buildExportRecord(run,new Set(["run_id","status","started_at","completed_at","duration_seconds","phase_timings","step_performance","wangp_version","outputs","output_records"]));
const imported = api.normalizeImportedExport({version:"1.1.0",runs:[clone(exported)]})[0];
assert(imported.wangp_version==="13.0" && imported.step_performance.length===8, "roundtrip metadata/observations lost");
for (const [key,unit] of [["encode:text","layers"],["decode:vae","tiles"],["denoise:main","steps"]]) {
    assert(imported.stages[key].unit===unit && imported.stages[key].current===imported.stages[key].total, "roundtrip phase counters lost");
}
assert(imported.outputs.includes(filename), "output filename metadata lost");
const authoritative = ns();
api.startRun(authoritative,{id:2,settings:{}},{server_time:300});
const authoritativeTelemetry={server_time:310,in_progress:false,stage_timing:{task_id:"2",execution_epoch:1,revision:4,last_stage:"save",stages:{
 encode:{elapsed:7,active:false,completed:true,run_count:1},save:{elapsed:3,active:false,completed:true,run_count:1}}},
 output_records:[{path:"authoritative.mp4",media_type:"video",settings:{}}]};
api.applyServerStageTiming(authoritative,authoritativeTelemetry);
api.finishRun(authoritative,"completed",310000,authoritativeTelemetry);
const authoritativeRun=authoritative.runHistory[0];
assert(authoritativeRun.stages.encode.authoritative&&authoritativeRun.stages.encode.duration_seconds===7,
 "History did not use authoritative Encode timing");
assert(authoritativeRun.stages.save.authoritative&&authoritativeRun.stages.save.duration_seconds===3,
 "History completion grace changed Save timing");
class Element {
    constructor(tag) {this.tag=tag;this.children=[];this.dataset={};this.style={};this.classList={add(){},toggle(){}};this.textContent="";}
    append(...children) {this.children.push(...children);}
    appendChild(child) {this.children.push(child);return child;}
    setAttribute() {} addEventListener() {}
    get childElementCount() {return this.children.length;}
}
globalThis.document={createElement:tag=>new Element(tag)};
const tree=api.createHistoryRun(context,imported,1);
function flatten(e) {return [e,...e.children.flatMap(flatten)];}
const nodes=flatten(tree), text=nodes.map(e=>e.textContent).join("\n");
assert(text.includes("WanGP 13.0") && text.includes("32/32 layers") && text.includes("12/12 tiles"), "History details missing counters/version");
assert(text.indexOf("Pipeline timing")<text.indexOf("Observed timing composition"), "pipeline summary is out of order");
assert(nodes.find(e=>e.tag==="tbody").children.length===8, "rendered step table includes layer/tile data");
const chips=nodes.filter(e=>e.className==="status-pro__stage-breakdown");
assert(chips[0].children.length===7, "pipeline summary does not have seven stages");
assert(chips[0].children.map(e=>e.textContent).join("|").includes("Enhance: —"), "unused stage did not render as an em dash");
assert(!chips[0].children.some(e=>/Enhance: 0s/.test(e.textContent)), "unused stage rendered as zero seconds");
assert(!chips[1].children.some(e=>e.textContent.includes(filename)), "filename leaked into phase chip");
const old=api.normalizeImportedExport({version:"1.0.5",runs:[{run_id:"old",status:"completed",phase_timings:{encode:{label:"Encoding Text Prompt",duration_seconds:2}}}]})[0];
assert(!old.wangp_version && !old.stages.encode.current, "invented legacy metadata");
api.createHistoryRun(context,old,2);
"""
        result = subprocess.run([node,"-"],input=javascript+"\n"+script,text=True,encoding="utf-8",capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_optional_version_uses_application_global(self):
        source = _source()
        self.assertIn('self.request_global("WanGP_version")', source)
        self.assertIn('getattr(self, "WanGP_version", "")', source)
