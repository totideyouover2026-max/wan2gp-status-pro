import json
import os
import threading
import time
import uuid
from functools import wraps
from urllib.parse import unquote

import gradio as gr

try:
    import psutil
except Exception:  # Optional: Status Pro still works without process memory telemetry.
    psutil = None

from shared.utils.plugins import WAN2GPPlugin
from shared.utils import prompt_parser
from .download_telemetry import DOWNLOAD_TELEMETRY, install_download_observer


RUN_SETTING_KEYS = (
    "mode",
    "model_type",
    "base_model_type",
    "model_filename",
    "config",
    "skip_steps_cache_type",
    "skip_steps_multiplier",
    "skip_steps_start_step_perc",
    "image_mode",
    "resolution",
    "num_inference_steps",
    "video_length",
    "duration_seconds",
    "num_frames",
    "frame_num",
    "force_fps",
    "fps",
    "seed",
    "guidance_scale",
    "guidance2_scale",
    "guidance3_scale",
    "guidance_phases",
    "flow_shift",
    "sample_solver",
    "temporal_upsampling",
    "temporal_upsampling_method",
    "temporal_upsampling_multiplier",
    "spatial_upsampling",
    "spatial_upsampling_method",
    "spatial_upsampling_ratio",
    "activated_loras",
    "loras_multipliers",
    "video_prompt_type",
    "audio_prompt_type",
    "multi_prompts_gen_type",
    "window_no",
    "prompt",
    "negative_prompt",
)

MODEL_WEIGHT_EXTENSIONS = (
    ".safetensors",
    ".gguf",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
)
MAX_STEP_TELEMETRY = 300
_PROCESS = psutil.Process(os.getpid()) if psutil is not None else None


class _ModelLifecycleTelemetry:
    """Thread-safe, short-lived model release state for the browser bridge."""

    def __init__(self):
        self._lock = threading.RLock()
        self._event = None

    def begin_unload(self, model_type, model_name):
        token = str(time.time_ns())
        with self._lock:
            self._event = {
                "token": token,
                "state": "unloading",
                "model_type": str(model_type or "")[:300],
                "model_name": str(model_name or model_type or "Previously loaded model")[:500],
                "started_at": time.time(),
                "completed_at": None,
                "error": "",
            }
        return token

    def finish_unload(self, token, error=None):
        with self._lock:
            if not self._event or self._event.get("token") != token:
                return
            self._event["state"] = "failed" if error else "unloaded"
            self._event["completed_at"] = time.time()
            self._event["error"] = str(error or "")[:500]

    def snapshot(self):
        with self._lock:
            event = dict(self._event) if self._event else None
        if not event:
            return None
        completed_at = event.get("completed_at")
        if completed_at is not None and time.time() - float(completed_at) > 5:
            return None
        return event


MODEL_LIFECYCLE_TELEMETRY = _ModelLifecycleTelemetry()


def _telemetry_value(value, depth=0):
    """Return a small JSON-safe representation without copying media payloads."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:8000]
    if depth >= 2:
        return str(value)[:500]
    if isinstance(value, (list, tuple)):
        return [_telemetry_value(item, depth + 1) for item in value[:50]]
    if isinstance(value, dict):
        return {
            str(key)[:100]: _telemetry_value(item, depth + 1)
            for key, item in list(value.items())[:50]
        }
    return str(value)[:500]


def _memory_snapshot(torch_module=None):
    """Return non-synchronizing process and active-device memory counters."""
    sample = {"sampled_at": time.time()}
    if _PROCESS is not None:
        try:
            memory = _PROCESS.memory_info()
            sample["ram_rss_bytes"] = int(memory.rss)
            sample["ram_vms_bytes"] = int(memory.vms)
        except Exception:
            pass

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        return sample
    try:
        if not cuda.is_available() or not cuda.is_initialized():
            return sample
        device = int(cuda.current_device())
        sample["gpu_device_index"] = device
        sample["gpu_name"] = str(cuda.get_device_name(device))[:200]
        sample["vram_allocated_bytes"] = int(cuda.memory_allocated(device))
        sample["vram_reserved_bytes"] = int(cuda.memory_reserved(device))
        free_bytes, total_bytes = cuda.mem_get_info(device)
        sample["vram_device_free_bytes"] = int(free_bytes)
        sample["vram_device_total_bytes"] = int(total_bytes)
        sample["vram_device_used_bytes"] = int(total_bytes - free_bytes)
    except Exception:
        pass
    return sample


def _skip_count(pipe):
    cache = getattr(pipe, "cache", None)
    value = getattr(cache, "skipped_steps", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _performance_snapshot(gen):
    source = gen.get("status_pro_performance") if isinstance(gen, dict) else None
    if not isinstance(source, dict):
        return None
    steps = []
    for item in list(source.get("steps") or [])[-MAX_STEP_TELEMETRY:]:
        if isinstance(item, dict):
            steps.append({str(key)[:80]: _telemetry_value(value) for key, value in item.items()})
    return {
        "id": str(source.get("id") or "")[:120],
        "started_at": _telemetry_value(source.get("started_at")),
        "callback_phase": _telemetry_value(source.get("callback_phase", 0)),
        "phase_started_at": _telemetry_value(source.get("phase_started_at")),
        "steps": steps,
        "steps_truncated": bool(source.get("steps_truncated")),
    }


def _component_filename(value):
    """Reduce a local path or download URL to its display-safe filename."""
    if not isinstance(value, str) or not value.strip():
        return ""
    clean = value.strip().split("|", 1)[0].split("?", 1)[0].split("#", 1)[0]
    clean = clean.replace("\\", "/").rstrip("/")
    return unquote(clean.rsplit("/", 1)[-1])[:500]


def _component_filenames(values):
    names = []
    seen = set()

    def visit(value, depth=0):
        if depth > 5:
            return
        if isinstance(value, str):
            name = _component_filename(value)
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                names.append(name)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item, depth + 1)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item, depth + 1)

    visit(values)
    return names


def _vae_weight_names(values, include_upsamplers=False):
    names = []
    for name in _component_filenames(values):
        lowered = name.lower()
        if not lowered.endswith(MODEL_WEIGHT_EXTENSIONS):
            continue
        if "vae" not in lowered and "autoencoder" not in lowered:
            continue
        if not include_upsamplers and ("upscale" in lowered or "upsampler" in lowered):
            continue
        names.append(name)
    return names


def _model_components(
    settings,
    get_model_def=None,
    get_base_model_type=None,
    get_model_handler=None,
    get_model_config_groups=None,
    model_config_groups=None,
    get_model_recursive_prop=None,
    get_model_filename=None,
    transformer_quantization="",
    transformer_dtype_policy="",
    text_encoder_quantization="",
):
    """Resolve stage component filenames without exposing local paths or URLs."""
    if not isinstance(settings, dict):
        return {}
    model_type = settings.get("model_type") or settings.get("base_model_type")
    if not model_type:
        return {}

    model_def = None
    if callable(get_model_def):
        try:
            model_def = get_model_def(model_type)
        except Exception:
            pass
    model_def = model_def if isinstance(model_def, dict) else {}
    config_id = settings.get("config")
    if config_id and callable(get_model_config_groups) and model_config_groups is not None:
        select_configs = getattr(model_config_groups, "selected_model_configs", None)
        if callable(select_configs):
            try:
                resolved_def = model_def.copy()
                groups = get_model_config_groups(model_type, model_def)
                for _, _, selected_config in select_configs(groups, config_id):
                    if isinstance(selected_config, dict):
                        resolved_def.update(selected_config)
                model_def = resolved_def
            except Exception:
                pass

    prepare = _component_filenames(settings.get("model_filename"))
    if not prepare and callable(get_model_filename):
        try:
            prepare = _component_filenames(get_model_filename(
                model_type=model_type,
                quantization=str(transformer_quantization or ""),
                dtype_policy=transformer_dtype_policy or "",
                model_def=model_def,
            ))
        except Exception:
            pass

    encode = []
    if callable(get_model_recursive_prop) and callable(get_model_filename):
        try:
            encoder_urls = get_model_recursive_prop(
                model_type,
                "text_encoder_URLs",
                return_list=True,
                model_def=model_def,
            )
            if encoder_urls:
                encode = _component_filenames(get_model_filename(
                    model_type=model_type,
                    quantization=str(text_encoder_quantization or ""),
                    dtype_policy=transformer_dtype_policy or "",
                    URLs=encoder_urls,
                ))
        except Exception:
            pass

    spatial_upsampling = str(settings.get("spatial_upsampling") or "").strip()
    include_upsamplers = bool(spatial_upsampling)
    generic_vae_overrides = []
    for key in ("VAE_URLs", "vae_URLs", "vae_URL"):
        if model_def.get(key):
            generic_vae_overrides.extend(_vae_weight_names(
                model_def.get(key),
                include_upsamplers=include_upsamplers,
            ))

    decode = []
    if generic_vae_overrides:
        decode.extend(generic_vae_overrides)
    else:
        handler = None
        base_model_type = settings.get("base_model_type")
        if not base_model_type and callable(get_base_model_type):
            try:
                base_model_type = get_base_model_type(model_type)
            except Exception:
                pass
        if callable(get_model_handler):
            try:
                handler = get_model_handler(model_type)
            except Exception:
                pass
        query_files = getattr(handler, "query_model_files", None)
        if callable(query_files) and base_model_type:
            try:
                download_defs = query_files([], base_model_type, model_def=model_def)
            except TypeError:
                try:
                    download_defs = query_files([], base_model_type)
                except Exception:
                    download_defs = []
            except Exception:
                download_defs = []
            for download_def in download_defs or []:
                if isinstance(download_def, dict):
                    decode.extend(_vae_weight_names(
                        download_def.get("fileList"),
                        include_upsamplers=include_upsamplers,
                    ))

        for key in (
            "video_vae_file",
            "audio_vae_file",
            "ltx2_video_vae_file",
            "vae_file",
            "vae_filename",
        ):
            decode.extend(_vae_weight_names(
                model_def.get(key),
                include_upsamplers=include_upsamplers,
            ))

    components = {}
    for stage, values in (
        ("prepare", prepare),
        ("input", decode),
        ("encode", encode),
        ("decode", decode),
    ):
        unique = []
        seen = set()
        for value in values:
            key = value.lower()
            if key not in seen:
                seen.add(key)
                unique.append(value)
        if unique:
            components[stage] = unique
    return components


def _task_telemetry(
    task,
    get_model_name=None,
    get_model_family=None,
    families_infos=None,
    component_resolver=None,
):
    if not isinstance(task, dict):
        return None
    params = task.get("params") if isinstance(task.get("params"), dict) else {}
    settings = {
        key: _telemetry_value(params.get(key))
        for key in RUN_SETTING_KEYS
        if key in params and params.get(key) is not None
    }
    settings.setdefault("num_inference_steps", _telemetry_value(task.get("steps")))
    settings.setdefault("video_length", _telemetry_value(task.get("length")))
    if "prompt" not in settings and task.get("prompt") is not None:
        settings["prompt"] = _telemetry_value(task.get("prompt"))
    model_type = settings.get("model_type") or settings.get("base_model_type")
    if model_type and callable(get_model_name):
        try:
            settings["model_name"] = _telemetry_value(get_model_name(model_type))
        except Exception:
            pass
    if model_type and callable(get_model_family) and isinstance(families_infos, dict):
        try:
            family_key = get_model_family(model_type, for_ui=True)
            family_info = families_infos.get(family_key)
            if isinstance(family_info, (list, tuple)) and len(family_info) > 1:
                settings["model_family"] = _telemetry_value(family_info[1])
        except Exception:
            pass
    if callable(component_resolver):
        try:
            component_models = component_resolver(settings)
            if component_models:
                settings["component_models"] = _telemetry_value(component_models)
        except Exception:
            pass
    return {
        "id": _telemetry_value(task.get("id")),
        "client_id": str(params.get("client_id") or "")[:200],
        "repeats": _telemetry_value(task.get("repeats", 1)),
        "settings": settings,
    }


def _window_prompt(prompt, multi_prompts_gen_type, window_no):
    if not isinstance(prompt, str) or not isinstance(window_no, int) or window_no < 1:
        return None
    mode = prompt_parser.normalize_multi_prompts_mode(multi_prompts_gen_type)
    if "W" not in mode:
        return None
    prompts = prompt_parser.split_prompt_units(prompt, mode)
    if not prompts:
        return None
    return prompts[min(window_no - 1, len(prompts) - 1)]


def _window_prompts(prompt, multi_prompts_gen_type):
    if not isinstance(prompt, str):
        return []
    mode = prompt_parser.normalize_multi_prompts_mode(multi_prompts_gen_type)
    if "W" not in mode:
        return []
    return [_telemetry_value(value) for value in prompt_parser.split_prompt_units(prompt, mode)]


def _output_paths(values):
    paths = []
    for value in list(values or [])[-200:]:
        candidate = value
        if isinstance(value, (list, tuple)) and value:
            candidate = value[0]
        if isinstance(candidate, dict):
            candidate = candidate.get("path") or candidate.get("name")
        if candidate is not None:
            paths.append(str(candidate)[:2000])
    return paths


def _media_type_from_path(path, audio_hint=False):
    if audio_hint:
        return "audio"
    extension = os.path.splitext(str(path or ""))[1].lower()
    if extension in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
        return "image"
    if extension in {".wav", ".mp3", ".aac", ".flac", ".m4a", ".ogg", ".opus", ".wma"}:
        return "audio"
    if extension in {".mp4", ".mkv", ".mov", ".webm", ".avi", ".ogv"}:
        return "video"
    return "unknown"


def _output_records(paths, settings_values, audio_hint=False):
    normalized_paths = _output_paths(paths)
    settings_values = list(settings_values or [])
    records = []
    for index, path in enumerate(normalized_paths):
        raw_settings = settings_values[index] if index < len(settings_values) else None
        raw_settings = raw_settings if isinstance(raw_settings, dict) else {}
        resolved = {
            key: _telemetry_value(raw_settings.get(key))
            for key in RUN_SETTING_KEYS
            if key in raw_settings and raw_settings.get(key) is not None
        }
        for key in ("creation_date", "creation_timestamp", "generation_time"):
            if key in raw_settings and raw_settings.get(key) is not None:
                resolved[key] = _telemetry_value(raw_settings.get(key))
        records.append({
            "path": path,
            "media_type": _media_type_from_path(path, audio_hint=audio_hint),
            "settings": resolved,
        })
    return records


def _gallery_entry_path(value):
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    return str(value or "").strip()


def _gallery_path_keys(value):
    path = _gallery_entry_path(value)
    if not path:
        return set()
    normalized = os.path.normcase(os.path.normpath(path))
    keys = {normalized}
    try:
        keys.add(os.path.normcase(os.path.abspath(normalized)))
    except Exception:
        pass
    return keys


def _find_gallery_index(requested_paths, available_paths):
    available = list(available_paths or [])
    available_keys = [_gallery_path_keys(value) for value in available]
    requested = [_gallery_entry_path(value) for value in requested_paths or []]
    for path in requested:
        keys = _gallery_path_keys(path)
        for index, candidate_keys in enumerate(available_keys):
            if keys & candidate_keys:
                return index

    # Relative and absolute paths can occasionally arrive with different roots.
    # Fall back to a basename only when it identifies exactly one gallery item.
    basename_indexes = {}
    for index, value in enumerate(available):
        basename = os.path.normcase(os.path.basename(_gallery_entry_path(value)))
        if basename:
            basename_indexes.setdefault(basename, []).append(index)
    for path in requested:
        basename = os.path.normcase(os.path.basename(path))
        indexes = basename_indexes.get(basename, [])
        if len(indexes) == 1:
            return indexes[0]
    return None


def _resolve_gallery_import_path(value, output_roots):
    """Resolve an imported history path without allowing arbitrary file access."""
    raw_path = _gallery_entry_path(value)
    if not raw_path:
        raise ValueError("No output path was included in this history record.")

    media_type = _media_type_from_path(raw_path)
    if media_type == "unknown":
        raise ValueError("The recorded output is not a supported gallery media file.")

    roots = []
    for root in output_roots or []:
        root_path = str(root or "").strip()
        if not root_path:
            continue
        canonical_root = os.path.normcase(os.path.realpath(os.path.abspath(root_path)))
        if canonical_root not in roots:
            roots.append(canonical_root)
    if not roots:
        roots.append(os.path.normcase(os.path.realpath(os.path.join(os.getcwd(), "outputs"))))

    candidate = raw_path if os.path.isabs(raw_path) else os.path.join(os.getcwd(), raw_path)
    resolved_candidate = os.path.realpath(os.path.abspath(candidate))
    canonical = os.path.normcase(resolved_candidate)
    allowed = False
    for root in roots:
        try:
            if os.path.commonpath((canonical, root)) == root:
                allowed = True
                break
        except (OSError, ValueError):
            continue
    if not allowed:
        raise PermissionError("Status Pro can only import media from WanGP's configured output folders.")
    if not os.path.isfile(canonical):
        raise FileNotFoundError(
            "The recorded file is no longer available at its output path. Check the Outputs folder."
        )
    return resolved_candidate, media_type


class StatusProPlugin(WAN2GPPlugin):
    """A richer, browser-side presentation for Wan2GP generation progress."""

    def __init__(self):
        super().__init__()
        self.name = "Status Pro"
        self.version = "1.0.2"
        self.description = (
            "Selectable pipeline timeline with stage timings and live ETA estimates."
        )
        self._runtime_id = str(uuid.uuid4())
        self._insertion_registered = False
        self._step_observer_installed = False
        self._model_lifecycle_observer_installed = False

    def setup_ui(self):
        try:
            install_download_observer()
        except Exception as exc:
            # Download detail is optional. A changed Wan2GP/Hugging Face API
            # must not prevent the core status and history UI from loading.
            print(f"[Status Pro] Download telemetry unavailable: {exc}")
        self.request_component("gen_status")
        self.request_component("state")
        for component_id in (
            "gallery_tabs",
            "current_gallery_tab",
            "output",
            "last_choice",
            "audio_files_paths",
            "audio_file_selected",
            "audio_gallery_refresh_trigger",
        ):
            self.request_component(component_id)
        self.request_global("get_model_name")
        self.request_global("get_model_family")
        self.request_global("families_infos")
        self.request_global("get_model_def")
        self.request_global("get_base_model_type")
        self.request_global("get_model_handler")
        self.request_global("get_model_config_groups")
        self.request_global("model_config_groups")
        self.request_global("get_model_recursive_prop")
        self.request_global("get_model_filename")
        self.request_global("transformer_quantization")
        self.request_global("transformer_dtype_policy")
        self.request_global("text_encoder_quantization")
        self.request_global("build_callback")
        self.request_global("release_model")
        self.request_global("get_settings_from_file")
        self.request_global("save_path")
        self.request_global("image_save_path")
        self.request_global("audio_save_path")
        self.request_global("torch")
        self.add_custom_js(self._javascript())

    def _install_model_lifecycle_observer(self):
        if self._model_lifecycle_observer_installed:
            return
        original_release = getattr(self, "release_model", None)
        if not callable(original_release):
            return
        if getattr(original_release, "_status_pro_model_lifecycle_observer", False):
            self._model_lifecycle_observer_installed = True
            return

        @wraps(original_release)
        def observed_release(*args, **kwargs):
            global_values = getattr(original_release, "__globals__", {})
            model_type = global_values.get("transformer_type")
            has_loaded_model = (
                global_values.get("wan_model") is not None
                or global_values.get("offloadobj") is not None
            )
            if not has_loaded_model:
                return original_release(*args, **kwargs)

            model_name = model_type
            resolver = getattr(self, "get_model_name", None)
            if model_type and callable(resolver):
                try:
                    model_name = resolver(model_type)
                except Exception:
                    pass
            token = MODEL_LIFECYCLE_TELEMETRY.begin_unload(model_type, model_name)
            try:
                result = original_release(*args, **kwargs)
            except Exception as exc:
                MODEL_LIFECYCLE_TELEMETRY.finish_unload(token, error=exc)
                raise
            MODEL_LIFECYCLE_TELEMETRY.finish_unload(token)
            return result

        observed_release._status_pro_model_lifecycle_observer = True
        self.set_global("release_model", observed_release)
        self.release_model = observed_release
        self._model_lifecycle_observer_installed = True

    def _install_step_observer(self):
        if self._step_observer_installed:
            return
        original_builder = getattr(self, "build_callback", None)
        if not callable(original_builder):
            return
        if getattr(original_builder, "_status_pro_step_observer", False):
            self._step_observer_installed = True
            return
        torch_module = getattr(self, "torch", None)

        @wraps(original_builder)
        def observed_builder(state, pipe, *args, **kwargs):
            callback = original_builder(state, pipe, *args, **kwargs)
            state = state if isinstance(state, dict) else {}
            gen = state.get("gen") if isinstance(state.get("gen"), dict) else {}
            observer_id = f"{time.time_ns()}"
            performance = {
                "id": observer_id,
                "started_at": time.time(),
                "callback_phase": 0,
                "phase_started_at": None,
                "steps": [],
                "steps_truncated": False,
            }
            gen["status_pro_performance"] = performance
            last_step_at = time.perf_counter()
            last_skip_count = _skip_count(pipe)
            phase_index = 0
            next_sequence = 0
            default_total = kwargs.get("num_inference_steps")
            if default_total is None and len(args) >= 3:
                default_total = args[2]
            try:
                current_total = int(default_total) if default_total is not None and int(default_total) > 0 else None
            except (TypeError, ValueError):
                current_total = None

            @wraps(callback)
            def observed_callback(*callback_args, **callback_kwargs):
                nonlocal last_step_at, last_skip_count, phase_index, next_sequence, current_total
                step_idx = callback_kwargs.get("step_idx", callback_args[0] if callback_args else -1)
                force_refresh = callback_kwargs.get(
                    "force_refresh",
                    callback_args[2] if len(callback_args) > 2 else True,
                )
                try:
                    step_idx = int(step_idx)
                except (TypeError, ValueError):
                    step_idx = -1

                override_total = callback_kwargs.get(
                    "override_num_inference_steps",
                    callback_args[4] if len(callback_args) > 4 else None,
                )
                try:
                    override_total = int(override_total) if override_total is not None and int(override_total) > 0 else None
                except (TypeError, ValueError):
                    override_total = None
                if override_total is not None:
                    current_total = override_total

                now = time.perf_counter()
                if step_idx >= 0:
                    current_skip_count = _skip_count(pipe)
                    skip_delta = None
                    if current_skip_count is not None and last_skip_count is not None:
                        skip_delta = max(0, current_skip_count - last_skip_count)
                    cache = getattr(pipe, "cache", None)
                    next_sequence += 1
                    sample = {
                        "sequence": next_sequence,
                        "phase": phase_index,
                        "step": step_idx + 1,
                        "total_steps": current_total,
                        "pass_no": _telemetry_value(callback_kwargs.get("pass_no", -1)),
                        "duration_seconds": round(max(0.0, now - last_step_at), 4),
                        "skip_method": _telemetry_value(getattr(cache, "cache_type", None)),
                        "skipped": bool(skip_delta) if skip_delta is not None else None,
                        "skipped_delta": skip_delta,
                        "skipped_total": current_skip_count,
                        "completed_at": time.time(),
                        "memory": _memory_snapshot(torch_module),
                    }
                    extra = callback_kwargs.get("denoising_extra")
                    if extra:
                        sample["label"] = str(extra)[:300]
                    performance["steps"].append(sample)
                    if len(performance["steps"]) > MAX_STEP_TELEMETRY:
                        performance["steps"] = performance["steps"][-MAX_STEP_TELEMETRY:]
                        performance["steps_truncated"] = True
                    last_step_at = now
                    last_skip_count = current_skip_count
                elif force_refresh:
                    phase_index += 1
                    performance["callback_phase"] = phase_index
                    performance["phase_started_at"] = time.time()
                    last_step_at = now
                    last_skip_count = _skip_count(pipe)
                return callback(*callback_args, **callback_kwargs)

            return observed_callback

        observed_builder._status_pro_step_observer = True
        self.set_global("build_callback", observed_builder)
        self._step_observer_installed = True

    def _run_snapshot_json(self, state):
        try:
            state = state if isinstance(state, dict) else {}
            gen = state.get("gen") if isinstance(state.get("gen"), dict) else {}
            queue = gen.get("queue") if isinstance(gen.get("queue"), list) else []
            active_task = _task_telemetry(
                queue[0],
                get_model_name=getattr(self, "get_model_name", None),
                get_model_family=getattr(self, "get_model_family", None),
                families_infos=getattr(self, "families_infos", None),
                component_resolver=lambda settings: _model_components(
                    settings,
                    get_model_def=getattr(self, "get_model_def", None),
                    get_base_model_type=getattr(self, "get_base_model_type", None),
                    get_model_handler=getattr(self, "get_model_handler", None),
                    get_model_config_groups=getattr(self, "get_model_config_groups", None),
                    model_config_groups=getattr(self, "model_config_groups", None),
                    get_model_recursive_prop=getattr(self, "get_model_recursive_prop", None),
                    get_model_filename=getattr(self, "get_model_filename", None),
                    transformer_quantization=getattr(self, "transformer_quantization", ""),
                    transformer_dtype_policy=getattr(self, "transformer_dtype_policy", ""),
                    text_encoder_quantization=getattr(self, "text_encoder_quantization", ""),
                ),
            ) if gen.get("in_progress") and queue else None
            if active_task and gen.get("sliding_window"):
                window_prompts = _window_prompts(
                    active_task["settings"].get("prompt"),
                    active_task["settings"].get("multi_prompts_gen_type"),
                )
                if window_prompts:
                    active_task["window_prompts"] = window_prompts
                window_prompt = _window_prompt(
                    active_task["settings"].get("prompt"),
                    active_task["settings"].get("multi_prompts_gen_type"),
                    gen.get("window_no"),
                )
                if window_prompt:
                    active_task["window_prompt"] = _telemetry_value(window_prompt)
            video_records = _output_records(gen.get("file_list"), gen.get("file_settings_list"))
            audio_records = _output_records(
                gen.get("audio_file_list"),
                gen.get("audio_file_settings_list"),
                audio_hint=True,
            )
            payload = {
                "server_time": time.time(),
                "runtime_id": self._runtime_id,
                "in_progress": bool(gen.get("in_progress")),
                "queue_length": len(queue),
                "queue_task_ids": [_telemetry_value(task.get("id")) for task in queue if isinstance(task, dict)],
                "active_task": active_task,
                "sliding_window": bool(gen.get("sliding_window")),
                "window_no": _telemetry_value(gen.get("window_no")),
                "total_windows": _telemetry_value(gen.get("total_windows")),
                "video_outputs": [record["path"] for record in video_records],
                "audio_outputs": [record["path"] for record in audio_records],
                "output_records": video_records + audio_records,
                "status": str(gen.get("status") or "")[:2000],
                "progress_phase": _telemetry_value(gen.get("progress_phase")),
                "queue_errors": _telemetry_value(gen.get("queue_errors") or {}),
                "resource_sample": _memory_snapshot(getattr(self, "torch", None)) if gen.get("in_progress") else None,
                "performance": _performance_snapshot(gen),
                "model_lifecycle": MODEL_LIFECYCLE_TELEMETRY.snapshot(),
            }
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except Exception as exc:
            return json.dumps(
                {"server_time": time.time(), "runtime_id": self._runtime_id, "error": str(exc)[:500]},
                ensure_ascii=False,
                separators=(",", ":"),
            )

    def post_ui_setup(self, components: dict):
        if self._insertion_registered or components.get("gen_status") is None:
            return

        self._install_step_observer()
        self._install_model_lifecycle_observer()

        state_component = components.get("state")
        gallery_tabs = components.get("gallery_tabs")
        current_gallery_tab = components.get("current_gallery_tab")
        video_gallery = components.get("output")
        last_choice = components.get("last_choice")
        audio_files_paths = components.get("audio_files_paths")
        audio_file_selected = components.get("audio_file_selected")
        audio_gallery_refresh_trigger = components.get("audio_gallery_refresh_trigger")
        gallery_navigation_enabled = all(component is not None for component in (
            state_component,
            gallery_tabs,
            current_gallery_tab,
            video_gallery,
            last_choice,
            audio_files_paths,
            audio_file_selected,
            audio_gallery_refresh_trigger,
        ))

        def navigate_to_history_output(state_value, request_value):
            untouched = [gr.update()] * 7
            try:
                request = json.loads(request_value or "{}")
            except Exception:
                request = {}
            token = str(request.get("token") or "")[:200]
            run_id = str(request.get("run_id") or "")[:500]

            def response(status, message, gallery=None):
                return json.dumps({
                    "token": token,
                    "run_id": run_id,
                    "status": status,
                    "message": message,
                    "gallery": gallery,
                }, ensure_ascii=False, separators=(",", ":"))

            if not gallery_navigation_enabled:
                return (*untouched, gr.update(), response("error", "Gallery navigation is unavailable in this Wan2GP session."))

            state_value = state_value if isinstance(state_value, dict) else {}
            gen = state_value.get("gen") if isinstance(state_value.get("gen"), dict) else {}
            state_value["gen"] = gen
            video_paths = list(gen.get("file_list") or [])
            video_settings = list(gen.get("file_settings_list") or [])
            audio_paths = list(gen.get("audio_file_list") or [])
            audio_settings = list(gen.get("audio_file_settings_list") or [])
            records = request.get("outputs") if isinstance(request.get("outputs"), list) else []
            operation = str(request.get("operation") or "navigate").lower()

            if operation == "import":
                if not records:
                    return (*untouched, state_value, response("error", "No output path was included in this history record."))
                record = records[0] if isinstance(records[0], dict) else {"path": records[0]}
                try:
                    path, media_type = _resolve_gallery_import_path(
                        record.get("path"),
                        (
                            getattr(self, "save_path", None),
                            getattr(self, "image_save_path", None),
                            getattr(self, "audio_save_path", None),
                        ),
                    )
                except (ValueError, PermissionError, FileNotFoundError) as exc:
                    return (*untouched, state_value, response("missing", str(exc)))

                settings = None
                settings_reader = getattr(self, "get_settings_from_file", None)
                if callable(settings_reader):
                    try:
                        settings_result = settings_reader(state_value, path, False, False, False)
                        if isinstance(settings_result, (list, tuple)) and settings_result:
                            settings = settings_result[0]
                    except Exception:
                        settings = None

                if media_type == "audio":
                    index = _find_gallery_index([path], audio_paths)
                    added = index is None
                    if added:
                        while len(audio_settings) < len(audio_paths):
                            audio_settings.append(None)
                        audio_paths.append(path)
                        audio_settings.append(settings)
                        index = len(audio_paths) - 1
                    gen["audio_file_list"] = audio_paths
                    gen["audio_file_settings_list"] = audio_settings
                    gen["audio_selected"] = index
                    gen["audio_last_selected"] = (index + 1) >= len(audio_paths)
                    gen["current_gallery_source"] = "audio"
                    message = (
                        f"Imported {os.path.basename(path)} into Audio Files Gallery."
                        if added else f"{os.path.basename(path)} is already in Audio Files Gallery."
                    )
                    return (
                        gr.Tabs(selected="audio"),
                        1,
                        gr.update(),
                        gr.update(),
                        json.dumps(audio_paths),
                        index,
                        f"status-pro-{time.time_ns()}",
                        state_value,
                        response("imported", message, "audio"),
                    )

                index = _find_gallery_index([path], video_paths)
                added = index is None
                if added:
                    while len(video_settings) < len(video_paths):
                        video_settings.append(None)
                    video_paths.append(path)
                    video_settings.append(settings)
                    index = len(video_paths) - 1
                gen["file_list"] = video_paths
                gen["file_settings_list"] = video_settings
                gen["selected"] = index
                gen["last_selected"] = (index + 1) >= len(video_paths)
                gen["current_gallery_source"] = "video"
                gen["selected_video_time"] = 0.0 if media_type == "video" else None
                gallery_label = "Video / Images Gallery"
                message = (
                    f"Imported {os.path.basename(path)} into {gallery_label}."
                    if added else f"{os.path.basename(path)} is already in {gallery_label}."
                )
                return (
                    gr.Tabs(selected="video_images"),
                    0,
                    gr.Gallery(value=video_paths, selected_index=index),
                    index,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    state_value,
                    response("imported", message, "video_images"),
                )

            for record in records[:50]:
                record = record if isinstance(record, dict) else {"path": record}
                path = _gallery_entry_path(record.get("path"))
                media_type = str(record.get("media_type") or "").lower()
                if not path:
                    continue
                targets = []
                if media_type == "audio":
                    targets.append(("audio", audio_paths))
                elif media_type in {"video", "image"}:
                    targets.append(("video_images", video_paths))
                else:
                    targets.extend((("video_images", video_paths), ("audio", audio_paths)))

                for gallery_name, paths in targets:
                    index = _find_gallery_index([path], paths)
                    if index is None:
                        continue
                    if gallery_name == "audio":
                        gen["audio_selected"] = index
                        gen["audio_last_selected"] = (index + 1) >= len(audio_paths)
                        gen["current_gallery_source"] = "audio"
                        return (
                            gr.Tabs(selected="audio"),
                            1,
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            index,
                            f"status-pro-{time.time_ns()}",
                            state_value,
                            response("selected", "Selected in Audio Files Gallery.", "audio"),
                        )

                    gen["selected"] = index
                    gen["last_selected"] = (index + 1) >= len(video_paths)
                    gen["current_gallery_source"] = "video"
                    extension = os.path.splitext(_gallery_entry_path(video_paths[index]))[1].lower()
                    gen["selected_video_time"] = 0.0 if extension in {".mp4", ".mkv", ".mov", ".webm", ".avi", ".ogv"} else None
                    return (
                        gr.Tabs(selected="video_images"),
                        0,
                        gr.Gallery(value=video_paths, selected_index=index),
                        index,
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        state_value,
                        response("selected", "Selected in Video / Images Gallery.", "video_images"),
                    )

            return (*untouched, state_value, response(
                "missing",
                "Not currently in the gallery — check the Outputs folder.",
            ))

        def create_status_pro_host():
            with gr.Column(elem_id="status-pro-container") as container:
                gr.HTML(
                    value=self._markup(),
                    elem_id="status-pro-host",
                    show_label=False,
                )
                download_bridge = gr.Textbox(
                    value="{}",
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_id="status-pro-download-bridge",
                    elem_classes=["status-pro-download-bridge"],
                )
                run_bridge = gr.Textbox(
                    value=json.dumps(
                        {"server_time": time.time(), "runtime_id": self._runtime_id},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_id="status-pro-run-bridge",
                    elem_classes=["status-pro-run-bridge"],
                )
                gallery_request_bridge = gr.Textbox(
                    value="{}",
                    interactive=True,
                    show_label=False,
                    container=False,
                    elem_id="status-pro-gallery-request-bridge",
                    elem_classes=["status-pro-gallery-bridge"],
                )
                gallery_request_trigger = gr.Button(
                    value="Navigate gallery",
                    visible=False,
                    elem_id="status-pro-gallery-request-trigger",
                    elem_classes=["status-pro-gallery-bridge"],
                )
                gallery_result_bridge = gr.Textbox(
                    value="{}",
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_id="status-pro-gallery-result-bridge",
                    elem_classes=["status-pro-gallery-bridge"],
                )
                if gallery_navigation_enabled:
                    gallery_request_trigger.click(
                        fn=navigate_to_history_output,
                        inputs=[state_component, gallery_request_bridge],
                        outputs=[
                            gallery_tabs,
                            current_gallery_tab,
                            video_gallery,
                            last_choice,
                            audio_files_paths,
                            audio_file_selected,
                            audio_gallery_refresh_trigger,
                            state_component,
                            gallery_result_bridge,
                        ],
                        queue=False,
                        show_progress="hidden",
                        api_name=False,
                        show_api=False,
                    )
                download_timer = gr.Timer(value=0.5, active=True)
                download_timer.tick(
                    fn=DOWNLOAD_TELEMETRY.snapshot_json,
                    inputs=None,
                    outputs=[download_bridge],
                    queue=False,
                    show_progress="hidden",
                    api_name=False,
                    show_api=False,
                    trigger_mode="always_last",
                )
                if state_component is not None:
                    run_timer = gr.Timer(value=0.5, active=True)
                    run_timer.tick(
                        fn=self._run_snapshot_json,
                        inputs=[state_component],
                        outputs=[run_bridge],
                        queue=False,
                        show_progress="hidden",
                        api_name=False,
                        show_api=False,
                        trigger_mode="always_last",
                    )
            return container

        self.insert_after(
            target_component_id="gen_status",
            new_component_constructor=create_status_pro_host,
        )
        self._insertion_registered = True

    @staticmethod
    def _markup() -> str:
        return """
<section class="status-pro" data-status-pro hidden aria-label="Generation status">
  <header class="status-pro__header">
    <div class="status-pro__heading">
      <span class="status-pro__badge">Status Pro</span>
      <span class="status-pro__live" data-sp-live>Waiting for progress</span>
    </div>
    <div class="status-pro__header-actions">
      <div class="status-pro__summary" aria-live="polite">
        <span data-sp-steps></span>
        <span data-sp-overall></span>
        <span data-sp-eta></span>
      </div>
      <button class="status-pro__history-toggle" data-sp-history-toggle type="button" aria-expanded="false" aria-controls="status-pro-history-drawer">
        <span data-sp-history-label>History</span> <span data-sp-history-count>0</span>
      </button>
    <button class="status-pro__collapse" data-sp-collapse type="button" aria-expanded="true" title="Collapse Status Pro">▼</button>
    </div>
  </header>
  <div class="status-pro__body" data-sp-body>
    <section class="status-pro__idle" data-sp-idle hidden aria-live="polite">
      <div class="status-pro__idle-copy">
        <strong data-sp-idle-title>Ready</strong>
        <span data-sp-idle-message>Generation history will appear here.</span>
      </div>
    </section>
    <div data-sp-running>
    <div class="status-pro__stages" data-sp-stages role="tablist" aria-label="Generation stages"></div>
    <section class="status-pro__downloads" data-sp-downloads hidden aria-label="Model downloads">
    <div class="status-pro__downloads-header">
      <div class="status-pro__downloads-title">
        <span class="status-pro__download-indicator" aria-hidden="true">↓</span>
        <div>
          <strong data-sp-download-title>Downloading model files</strong>
          <span data-sp-download-summary>Preparing download information…</span>
        </div>
      </div>
      <span class="status-pro__download-total" data-sp-download-total></span>
    </div>
    <div class="status-pro__download-overall" aria-hidden="true">
      <div data-sp-download-overall-fill></div>
    </div>
    <div class="status-pro__download-files" data-sp-download-files></div>
    </section>
    <div class="status-pro__detail" data-sp-detail role="tabpanel" aria-live="polite">
      <div class="status-pro__detail-copy">
        <strong data-sp-detail-name>Preparing</strong>
        <span class="status-pro__detail-activities" data-sp-detail-activities hidden></span>
        <span class="status-pro__detail-model" data-sp-detail-model hidden></span>
        <span data-sp-detail-message>Waiting for generation progress.</span>
      </div>
      <dl class="status-pro__metrics">
        <div><dt>Status</dt><dd data-sp-detail-state>Pending</dd></div>
        <div><dt>Elapsed</dt><dd data-sp-detail-elapsed>—</dd></div>
        <div data-sp-eta-metric><dt>Expected left</dt><dd data-sp-detail-eta>—</dd></div>
        <div data-sp-progress-metric><dt>Progress</dt><dd data-sp-detail-progress>—</dd></div>
        <div data-sp-step-metric hidden><dt>Avg step time</dt><dd data-sp-detail-step-time>—</dd></div>
      </dl>
    </div>
    <div class="status-pro__overall-track" aria-hidden="true">
      <div class="status-pro__overall-fill" data-sp-overall-fill></div>
    </div>
    </div>
    <span data-sp-history-home hidden></span>
    <section class="status-pro__history-drawer" id="status-pro-history-drawer" data-sp-history-drawer hidden aria-label="Generation history">
      <div class="status-pro__history-drawer-header">
        <div class="status-pro__history-scope" role="group" aria-label="History range">
          <button type="button" data-sp-history-scope="session" aria-pressed="false">This session</button>
          <button type="button" data-sp-history-scope="all" aria-pressed="true">All history</button>
        </div>
        <div class="status-pro__history-actions">
          <button type="button" class="status-pro__history-import" data-sp-import-button>Import</button>
          <input type="file" data-sp-import-file accept=".json,application/json" hidden aria-label="Choose a Status Pro JSON export">
          <button type="button" class="status-pro__history-export" data-sp-export-button>Export</button>
          <span class="status-pro__history-separator" aria-hidden="true"></span>
          <label class="status-pro__history-select-all" title="Select all visible history entries">
            <span>Select all</span>
            <input type="checkbox" data-sp-select-all-history aria-label="Select all visible history entries">
          </label>
          <span class="status-pro__history-separator" aria-hidden="true"></span>
          <button type="button" data-sp-clear-selected disabled>Clear selected</button>
          <button type="button" class="status-pro__clear-history" data-sp-clear-history>Clear history</button>
          <span class="status-pro__history-separator" aria-hidden="true"></span>
          <button type="button" class="status-pro__history-settings" data-sp-settings-button aria-label="History settings" title="History settings"><span aria-hidden="true">⚙</span></button>
          <span class="status-pro__history-separator" aria-hidden="true"></span>
          <button type="button" class="status-pro__history-expand" data-sp-history-expand aria-label="Expand generation history" title="Expand generation history"><span aria-hidden="true">⛶</span></button>
        </div>
      </div>
      <div class="status-pro__history-storage-note" data-sp-history-storage-note hidden role="status"></div>
      <div class="status-pro__history-empty" data-sp-history-empty>No generations recorded yet.</div>
      <div class="status-pro__history" data-sp-history></div>
    </section>
    <dialog class="status-pro__history-modal" data-sp-history-modal aria-labelledby="status-pro-history-modal-title">
      <div class="status-pro__history-modal-shell">
        <header class="status-pro__history-modal-header">
          <div>
            <strong id="status-pro-history-modal-title">Generation history</strong>
            <span data-sp-history-modal-summary>Browse recorded runs in a larger workspace.</span>
          </div>
          <button type="button" class="status-pro__history-modal-close" data-sp-history-modal-close aria-label="Close expanded history" title="Return to embedded history">×</button>
        </header>
        <div class="status-pro__history-modal-content" data-sp-history-modal-content></div>
      </div>
    </dialog>
    <dialog class="status-pro__export-modal" data-sp-export-modal aria-labelledby="status-pro-export-title">
      <div class="status-pro__export-shell">
        <header class="status-pro__export-header">
          <div>
            <strong id="status-pro-export-title">History settings</strong>
            <span data-sp-export-scope>Choose history, privacy, and export defaults.</span>
          </div>
          <div class="status-pro__export-header-actions">
            <label class="status-pro__history-persistence" title="Choose whether Status Pro records completed runs and how long it keeps them">
              <span>History</span>
              <select data-sp-history-persistence aria-label="History persistence">
                <option value="off">Do not record new runs</option>
                <option value="persistent">Until manually cleared</option>
                <option value="browser">Until browser tab closes</option>
                <option value="runtime">Until WanGP restarts</option>
              </select>
            </label>
            <button type="button" class="status-pro__export-info" data-sp-export-info aria-expanded="false" aria-controls="status-pro-export-guide" title="About export presets">i</button>
            <button type="button" class="status-pro__export-close" data-sp-export-close aria-label="Close history settings">×</button>
          </div>
        </header>
        <section class="status-pro__export-guide" id="status-pro-export-guide" data-sp-export-guide hidden>
          <strong>Preset guide</strong>
          <dl>
            <div><dt>Standard</dt><dd>A complete local archive with every available field except prompts.</dd></div>
            <div><dt>Performance</dt><dd>Benchmark models using per-step timings, cache skips, phase totals, and RAM/VRAM observations.</dd></div>
            <div><dt>Reproducibility</dt><dd>Recreate a run from its checkpoint, generation and step-skipping settings, LoRAs, seed, and available prompts.</dd></div>
            <div><dt>Share-safe</dt><dd>Share timing and resource summaries without prompts, paths, exact checkpoints, settings objects, or browser/session IDs.</dd></div>
            <div><dt>Custom presets</dt><dd>Save several named field combinations in this browser for repeated export workflows.</dd></div>
          </dl>
        </section>
        <div class="status-pro__export-controls">
          <div class="status-pro__export-preset-control">
            <span>Preset</span>
            <span class="status-pro__export-preset-picker">
              <select data-sp-export-preset>
                <option value="standard">Standard</option>
                <option value="performance">Performance</option>
                <option value="reproducibility">Reproducibility</option>
                <option value="share-safe">Share-safe</option>
              </select>
              <button type="button" data-sp-export-delete-preset hidden>Delete</button>
            </span>
          </div>
          <label>
            <span>Format</span>
            <select data-sp-export-format>
              <option value="json">JSON</option>
              <option value="csv">CSV</option>
              <option value="md">Markdown</option>
            </select>
          </label>
          <div class="status-pro__export-quick-actions">
            <button type="button" data-sp-export-select-all>Select all</button>
            <button type="button" data-sp-export-clear-fields>Clear</button>
          </div>
        </div>
        <div class="status-pro__export-fields" data-sp-export-fields></div>
        <div class="status-pro__export-note" data-sp-export-prompt-note></div>
        <footer class="status-pro__export-footer">
          <div>
            <button type="button" data-sp-export-reset>Reset to default</button>
            <button type="button" data-sp-export-save>Save as preset…</button>
          </div>
          <div>
            <button type="button" data-sp-export-cancel>Cancel</button>
            <button type="button" class="status-pro__export-confirm" data-sp-export-confirm>Save settings</button>
          </div>
        </footer>
      </div>
    </dialog>
  </div>
</section>
"""

    @staticmethod
    def _javascript() -> str:
        return r"""
(function () {
    const NAMESPACE = "__wangpStatusPro";
    const HISTORY_KEY = "wangp.status-pro.stage-history.v1";
    const RUN_HISTORY_KEY = "wangp.status-pro.run-history.v1";
    const SESSION_RUN_HISTORY_KEY = "wangp.status-pro.run-history.session.v1";
    const RUNTIME_RUN_HISTORY_KEY = "wangp.status-pro.run-history.runtime.v1";
    const HISTORY_RUNTIME_ID_KEY = "wangp.status-pro.history-runtime-id.v1";
    const HISTORY_PERSISTENCE_KEY = "wangp.status-pro.history-persistence.v1";
    const HISTORY_RECORDING_KEY = "wangp.status-pro.history-recording.v1";
    const COLLAPSED_KEY = "wangp.status-pro.collapsed.v1";
    const EXPORT_FIELDS_KEY = "wangp.status-pro.export-fields.v1";
    const EXPORT_SETTINGS_KEY = "wangp.status-pro.export-settings.v1";
    const EXPORT_PRESETS_KEY = "wangp.status-pro.export-presets.v2";
    const PROMPT_MEMORY_KEY = "wangp.status-pro.prompt-memory.v1";
    const MAX_CUSTOM_EXPORT_PRESETS = 20;
    const MAX_RUN_HISTORY = 100;
    const MAX_STEP_RECORDS = 300;
    const MAX_IMPORT_BYTES = 20 * 1024 * 1024;
    const TICK_MS = 250;
    const IDLE_GRACE_MS = 1600;
    const RESET_AFTER_MS = 3000;

    const EXPORT_FIELD_DEFS = [
        { id: "run_id", label: "Run ID", group: "Run" },
        { id: "session_id", label: "Session ID", group: "Run" },
        { id: "queue_task_id", label: "Queue task ID", group: "Run" },
        { id: "repeats", label: "Repeat count", group: "Run" },
        { id: "status", label: "Status", group: "Run" },
        { id: "outcome", label: "Outcome / failure detail", group: "Run" },
        { id: "started_at", label: "Started time", group: "Timing" },
        { id: "completed_at", label: "Completed time", group: "Timing" },
        { id: "duration_seconds", label: "Total wall-clock time", group: "Timing" },
        { id: "generation_time", label: "Wan2GP generation time", group: "Timing" },
        { id: "phase_timings", label: "Phase and stage timings", group: "Timing" },
        { id: "step_performance", label: "Per-step performance", group: "Performance" },
        { id: "resource_usage", label: "RAM / VRAM usage", group: "Performance" },
        { id: "step_skipping", label: "Step-skipping configuration and results", group: "Performance" },
        { id: "model_summary", label: "Model summary", group: "Model & settings" },
        { id: "model_name", label: "Model display name", group: "Model & settings" },
        { id: "checkpoint", label: "Checkpoint / source", group: "Model & settings" },
        { id: "resolution", label: "Resolution", group: "Model & settings" },
        { id: "steps", label: "Inference steps", group: "Model & settings" },
        { id: "fps", label: "FPS", group: "Model & settings" },
        { id: "seed", label: "Seed", group: "Model & settings" },
        { id: "guidance", label: "Guidance", group: "Model & settings" },
        { id: "guidance2", label: "Guidance 2", group: "Model & settings" },
        { id: "guidance3", label: "Guidance 3", group: "Model & settings" },
        { id: "flow_shift", label: "Flow shift", group: "Model & settings" },
        { id: "sampler", label: "Sampler", group: "Model & settings" },
        { id: "loras", label: "LoRAs", group: "Model & settings" },
        { id: "settings", label: "Complete settings object", group: "Model & settings" },
        { id: "media_type", label: "Media type", group: "Media & output" },
        { id: "frame_count", label: "Frame count", group: "Media & output" },
        { id: "output_count", label: "Output count", group: "Media & output" },
        { id: "outputs", label: "Output paths", group: "Media & output" },
        { id: "output_records", label: "Resolved output metadata", group: "Media & output" },
        { id: "prompt", label: "Prompt", group: "Prompts", prompt: true },
        { id: "negative_prompt", label: "Negative prompt", group: "Prompts", prompt: true }
    ];
    const EXPORT_GROUP_HELP = {
        "Run": "Identifiers, repeat information, and the final outcome for each generation.",
        "Timing": "Start, finish, wall-clock, generation, and individual stage timings.",
        "Performance": "Per-step speed, memory observations, and step-skipping behaviour.",
        "Model & settings": "Model and generation parameters useful for comparison or reproduction.",
        "Media & output": "Generated media characteristics, file counts, paths, and per-output metadata.",
        "Prompts": "Optional prompt text held only in this page when prompt memory is enabled."
    };
    const EXPORT_FIELD_HELP = {
        run_id: "Status Pro's unique identifier for this recorded run.",
        session_id: "Identifies the page session so related runs can be grouped.",
        queue_task_id: "The task number assigned by WanGP's generation queue.",
        repeats: "The number of repeated generations requested for this task.",
        status: "Whether the run completed, was aborted, or failed.",
        outcome: "The abort reason or failure detail when one was reported.",
        started_at: "The date and time at which Status Pro first observed the run.",
        completed_at: "The date and time at which the run finished.",
        duration_seconds: "Total elapsed time from the start of the run to its finish.",
        generation_time: "Generation duration reported directly by WanGP.",
        phase_timings: "Durations for observed phases such as loading, encoding, generating, and decoding.",
        step_performance: "Individual step durations, pass numbers, skips, and memory samples.",
        resource_usage: "Observed process RAM and GPU memory averages and peaks.",
        step_skipping: "Configured cache or skipping method and the skips Status Pro observed.",
        model_summary: "A compact human-readable model, variant, media type, and resolution summary.",
        model_name: "WanGP's display name for the selected model.",
        checkpoint: "The selected checkpoint filename or source reference.",
        resolution: "Requested output width and height.",
        steps: "Configured inference-step count for each generation pass.",
        fps: "Requested playback frame rate, when applicable.",
        seed: "Random seed used to reproduce the generation.",
        guidance: "Primary prompt-guidance value.",
        guidance2: "Secondary guidance value used by multi-guidance models.",
        guidance3: "Third guidance value used by multi-guidance models.",
        flow_shift: "Scheduler flow-shift value used during generation.",
        sampler: "Sampling or solver method selected for denoising.",
        loras: "Names of the LoRAs activated for the run.",
        settings: "The complete captured settings object; useful for detailed analysis but verbose.",
        media_type: "Whether the output is an image, video, audio file, or a mixture.",
        frame_count: "Resolved number of frames in the generated media.",
        output_count: "Number of output files associated with the run.",
        outputs: "Paths recorded for the generated output files.",
        output_records: "Per-file media type and resolved settings captured with each output.",
        prompt: "Positive prompt text, available only while page prompt memory retains it.",
        negative_prompt: "Negative prompt text, available only while page prompt memory retains it."
    };
    const EXPORT_FIELD_IDS = new Set(EXPORT_FIELD_DEFS.map(field => field.id));
    const EXPORT_PRESETS = {
        standard: EXPORT_FIELD_DEFS.filter(field => !field.prompt).map(field => field.id),
        performance: ["queue_task_id", "status", "started_at", "completed_at", "duration_seconds", "generation_time", "phase_timings", "step_performance", "resource_usage", "step_skipping", "model_summary", "media_type", "resolution", "frame_count", "steps"],
        reproducibility: ["queue_task_id", "model_name", "checkpoint", "media_type", "resolution", "frame_count", "fps", "steps", "seed", "guidance", "guidance2", "guidance3", "flow_shift", "sampler", "step_skipping", "loras", "prompt", "negative_prompt"],
        "share-safe": ["queue_task_id", "status", "duration_seconds", "generation_time", "phase_timings", "resource_usage", "step_skipping", "model_summary", "media_type", "resolution", "frame_count", "steps"]
    };

    const STAGE_DEFS = [
        { id: "prepare", label: "Prepare" },
        { id: "input", label: "Inputs", optional: true },
        { id: "encode", label: "Encode" },
        { id: "denoise", label: "Generate" },
        { id: "decode", label: "Decode" },
        { id: "post", label: "Enhance", optional: true },
        { id: "save", label: "Save", optional: true }
    ];

    const STYLE_TEXT = `
#status-pro-container {
    display: none;
    min-width: 0;
}
#status-pro-container.status-pro-container--active {
    display: flex;
}
#status-pro-host {
    display: none;
    min-width: 0;
}
#status-pro-host.status-pro-host--active {
    display: block;
}
.status-pro-source--active {
    display: none !important;
}
#status-pro-download-bridge,
.status-pro-download-bridge,
#status-pro-run-bridge,
.status-pro-run-bridge,
#status-pro-gallery-request-bridge,
#status-pro-gallery-request-trigger,
#status-pro-gallery-result-bridge,
.status-pro-gallery-bridge {
    display: none !important;
}
.status-pro {
    --sp-accent: var(--color-accent, var(--primary-500, #0ea5e9));
    --sp-accent-strong: var(--primary-600, #0284c7);
    --sp-good: #22c55e;
    --sp-muted: var(--body-text-color-subdued, #94a3b8);
    --sp-text: var(--body-text-color, #f8fafc);
    --sp-panel: var(--block-background-fill, #1e293b);
    --sp-panel-soft: var(--background-fill-secondary, #0f172a);
    --sp-border: var(--border-color-primary, #334155);
    box-sizing: border-box;
    container-name: status-pro;
    container-type: inline-size;
    width: 100%;
    padding: 12px;
    border: 1px solid var(--sp-border);
    border-radius: var(--block-radius, 8px);
    background: var(--sp-panel);
    color: var(--body-text-color, #f8fafc);
    box-shadow: var(--block-shadow, none);
}
.status-pro *, .status-pro *::before, .status-pro *::after { box-sizing: border-box; }
.status-pro__header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
}
.status-pro__heading, .status-pro__summary, .status-pro__header-actions {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}
.status-pro__header-actions {
    justify-content: flex-end;
    flex-wrap: nowrap;
    margin-left: auto;
}
.status-pro__collapse,
.status-pro__history-toggle {
    display: inline-grid;
    place-items: center;
    height: 28px;
    flex: 0 0 auto;
    border: 1px solid var(--sp-border);
    border-radius: 7px;
    background: var(--sp-panel-soft);
    color: var(--sp-muted);
    font-size: 1rem;
    line-height: 1;
    cursor: pointer;
    transition: border-color 160ms ease, color 160ms ease, background-color 160ms ease;
}
.status-pro__collapse {
    width: 28px;
    padding: 0;
}
.status-pro__history-toggle {
    display: inline-flex;
    gap: 6px;
    padding: 0 9px;
    font-size: .7rem;
    font-weight: 650;
}
.status-pro__history-toggle [data-sp-history-count] {
    display: inline-grid;
    min-width: 18px;
    min-height: 18px;
    place-items: center;
    padding: 0 5px;
    border-radius: 99px;
    background: color-mix(in srgb, var(--sp-accent) 18%, transparent);
    color: var(--body-text-color, #f8fafc);
    font-variant-numeric: tabular-nums;
}
.status-pro__history-toggle[data-recording="off"] {
    border-style: dashed;
    color: var(--sp-muted);
}
.status-pro__history-toggle[aria-expanded="true"] {
    border-color: var(--sp-accent);
    color: var(--body-text-color, #f8fafc);
}
.status-pro__collapse:hover,
.status-pro__history-toggle:hover {
    border-color: var(--sp-accent);
    color: var(--body-text-color, #f8fafc);
}
.status-pro__collapse:focus-visible,
.status-pro__history-toggle:focus-visible { outline: 2px solid var(--sp-accent); outline-offset: 2px; }
.status-pro__body[hidden] { display: none !important; }
.status-pro__body [hidden] { display: none !important; }
.status-pro--collapsed .status-pro__header { margin-bottom: 0; }
.status-pro__idle {
    display: grid;
    gap: 10px;
}
.status-pro__idle-copy {
    display: grid;
    gap: 2px;
    padding: 2px 1px;
}
.status-pro__idle-copy strong { font-size: .9rem; }
.status-pro__idle-copy span,
.status-pro__history-empty {
    color: var(--sp-muted);
    font-size: .75rem;
}
.status-pro__history-drawer {
    display: grid;
    gap: 10px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--sp-border);
}
.status-pro__history-drawer-header {
    --sp-history-control-height: 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px 16px;
    flex-wrap: wrap;
    padding: 8px 9px;
    border: 1px solid var(--sp-border);
    border-radius: 8px;
    background: color-mix(in srgb, var(--sp-panel-soft) 72%, transparent);
}
.status-pro__history-storage-note {
    padding: 8px 10px;
    border: 1px solid color-mix(in srgb, #f59e0b 52%, var(--sp-border));
    border-radius: 8px;
    background: color-mix(in srgb, #f59e0b 10%, var(--sp-panel));
    color: var(--sp-text);
    font-size: .75rem;
}
.status-pro__history-scope {
    display: inline-flex;
    align-items: center;
    box-sizing: border-box;
    height: var(--sp-history-control-height);
    padding: 2px;
    border: 1px solid var(--sp-border);
    border-radius: 7px;
    background: var(--sp-panel-soft);
}
.status-pro__history-scope button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
    height: calc(var(--sp-history-control-height) - 6px);
    min-height: 0;
    padding: 3px 8px;
    border: 0;
    border-radius: 5px;
    background: transparent;
    color: var(--sp-muted);
    font-size: .69rem;
    line-height: 1;
    cursor: pointer;
}
.status-pro__history-scope button[aria-pressed="true"] {
    background: color-mix(in srgb, var(--sp-accent) 18%, var(--sp-panel));
    color: var(--body-text-color, #f8fafc);
}
.status-pro__history-actions {
    display: flex;
    align-items: center;
    gap: 7px;
    flex-wrap: nowrap;
    min-height: var(--sp-history-control-height);
}
.status-pro__history-separator {
    flex: 0 0 1px;
    width: 1px;
    height: 22px;
    margin: 0 2px;
    background: color-mix(in srgb, var(--sp-border) 82%, transparent);
}
.status-pro__history-select-all {
    display: inline-flex;
    align-items: center;
    box-sizing: border-box;
    height: var(--sp-history-control-height);
    gap: 5px;
    color: var(--sp-muted);
    font-size: .69rem;
    line-height: 1;
    cursor: pointer;
}
.status-pro__history-select-all input {
    width: 16px;
    height: 16px;
    margin: 0;
    accent-color: var(--sp-accent);
}
.status-pro__history-actions button,
.status-pro__history-actions select {
    box-sizing: border-box;
    height: var(--sp-history-control-height);
    min-height: 0;
    padding: 4px 9px;
    border: 1px solid var(--sp-border);
    border-radius: 6px;
    background: var(--sp-panel);
    color: inherit;
    font-size: .69rem;
    line-height: 1;
}
.status-pro__history-actions button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
}
.status-pro__history-actions button:hover {
    border-color: var(--sp-accent);
}
.status-pro__history-actions .status-pro__history-export {
    border-color: color-mix(in srgb, var(--sp-accent) 70%, var(--sp-border));
    color: color-mix(in srgb, var(--sp-accent) 82%, var(--sp-text));
}
.status-pro__history-actions .status-pro__history-settings {
    display: inline-grid;
    flex: 0 0 var(--sp-history-control-height);
    width: var(--sp-history-control-height);
    padding: 0;
    place-items: center;
    color: var(--sp-muted);
    font-size: 1rem;
    line-height: 1;
}
.status-pro__history-actions button:disabled,
.status-pro__history-actions select:disabled {
    cursor: default;
    opacity: .45;
}
.status-pro__history-actions .status-pro__clear-history {
    color: var(--sp-muted);
}
.status-pro__history-actions .status-pro__history-expand {
    display: inline-grid;
    flex: 0 0 var(--sp-history-control-height);
    width: var(--sp-history-control-height);
    padding: 0;
    place-items: center;
    color: var(--sp-muted);
    font-size: 1rem;
    line-height: 1;
}
.status-pro__history-modal {
    position: fixed;
    z-index: 10000;
    box-sizing: border-box;
    width: min(94vw, 2600px);
    height: min(90dvh, 1500px);
    max-width: none;
    max-height: none;
    padding: 0;
    overflow: hidden;
    border: 1px solid var(--sp-border);
    border-radius: 12px;
    background: var(--sp-panel);
    color: var(--body-text-color, #f8fafc);
    box-shadow: 0 24px 80px rgb(0 0 0 / .48);
}
.status-pro__history-modal::backdrop {
    background: rgb(2 6 23 / .7);
    backdrop-filter: blur(2px);
}
.status-pro__history-modal-shell {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    height: 100%;
    min-height: 0;
}
.status-pro__history-modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 13px 16px;
    border-bottom: 1px solid var(--sp-border);
    background: color-mix(in srgb, var(--sp-panel-soft) 72%, var(--sp-panel));
}
.status-pro__history-modal-header > div {
    display: grid;
    gap: 2px;
    min-width: 0;
}
.status-pro__history-modal-header strong { font-size: .95rem; }
.status-pro__history-modal-header span {
    overflow: hidden;
    color: var(--sp-muted);
    font-size: .72rem;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-pro__history-modal-close {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 34px;
    width: 34px;
    height: 34px;
    min-height: 34px;
    padding: 0;
    border: 1px solid var(--sp-border);
    border-radius: 8px;
    background: var(--sp-panel-soft);
    color: inherit;
    font-size: 1rem;
    cursor: pointer;
}
.status-pro__history-modal-close:hover { border-color: var(--sp-accent); }
.status-pro__history-modal-content {
    display: grid;
    min-height: 0;
    padding: 0 14px 14px;
    overflow: hidden;
}
.status-pro__history-drawer[data-expanded="true"] {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    margin-top: 0;
    padding-top: 12px;
    border-top: 0;
}
.status-pro__history-drawer[data-expanded="true"] .status-pro__history {
    flex: 1 1 auto;
    max-height: none;
    min-height: 0;
    padding-right: 4px;
}
.status-pro__history-drawer[data-expanded="true"] .status-pro__history-expand span {
    display: inline-block;
    transform: rotate(180deg);
}
.status-pro__export-modal {
    position: fixed;
    box-sizing: border-box;
    width: min(880px, calc(100vw - 28px));
    height: min(820px, calc(100dvh - 32px));
    max-width: none;
    max-height: calc(100dvh - 32px);
    padding: 0;
    overflow: hidden;
    border: 1px solid var(--sp-border);
    border-radius: 10px;
    background: var(--sp-panel);
    color: var(--body-text-color, #f8fafc);
    box-shadow: 0 22px 70px rgb(0 0 0 / .42);
}
.status-pro__export-modal::backdrop {
    background: rgb(2 6 23 / .68);
    backdrop-filter: blur(2px);
}
.status-pro__export-shell {
    display: grid;
    grid-template-rows: auto auto auto minmax(0, 1fr) auto;
    height: 100%;
    min-height: 0;
}
.status-pro__export-header,
.status-pro__export-footer,
.status-pro__export-controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px 16px;
}
.status-pro__export-header {
    flex-wrap: wrap;
    padding: 14px 16px;
    border-bottom: 1px solid var(--sp-border);
    cursor: move;
    touch-action: none;
    user-select: none;
}
.status-pro__export-header > div:first-child { display: grid; gap: 2px; }
.status-pro__export-header strong { font-size: .95rem; }
.status-pro__export-header span,
.status-pro__export-note {
    color: var(--sp-muted);
    font-size: .72rem;
}
.status-pro__export-header-actions {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-left: auto;
}
.status-pro__history-persistence {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--sp-muted);
    font-size: .68rem;
    cursor: default;
}
.status-pro__history-persistence select {
    min-height: 30px;
    padding: 4px 26px 4px 8px;
    border: 1px solid var(--sp-border);
    border-radius: 7px;
    background: var(--sp-panel-soft);
    color: var(--body-text-color, #f8fafc);
    font-size: .69rem;
    cursor: pointer;
}
.status-pro__export-info,
.status-pro__export-close {
    width: 30px;
    height: 30px;
    padding: 0;
    border: 1px solid var(--sp-border);
    border-radius: 7px;
    background: var(--sp-panel-soft);
    color: inherit;
    font-size: 1.1rem;
    cursor: pointer;
}
.status-pro__export-info {
    border-radius: 999px;
    color: var(--sp-accent);
    font-family: Georgia, serif;
    font-size: .86rem;
    font-weight: 700;
}
.status-pro__export-info[aria-expanded="true"] {
    border-color: var(--sp-accent);
    background: color-mix(in srgb, var(--sp-accent) 14%, var(--sp-panel-soft));
}
.status-pro__export-guide {
    display: grid;
    gap: 8px;
    padding: 11px 16px;
    border-bottom: 1px solid var(--sp-border);
    background: color-mix(in srgb, var(--sp-accent) 7%, var(--sp-panel-soft));
}
.status-pro__export-guide > strong { font-size: .76rem; }
.status-pro__export-guide dl {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 8px 14px;
    margin: 0;
}
.status-pro__export-guide dl > div { display: grid; gap: 1px; }
.status-pro__export-guide dt { font-size: .69rem; font-weight: 700; }
.status-pro__export-guide dd {
    margin: 0;
    color: var(--sp-muted);
    font-size: .68rem;
    line-height: 1.3;
}
.status-pro__export-controls {
    justify-content: flex-start;
    padding: 10px 16px;
    border-bottom: 1px solid color-mix(in srgb, var(--sp-border) 72%, transparent);
}
.status-pro__export-controls label,
.status-pro__export-preset-control {
    display: grid;
    gap: 3px;
    min-width: 150px;
    color: var(--sp-muted);
    font-size: .64rem;
    text-transform: uppercase;
}
.status-pro__export-controls select,
.status-pro__export-modal button {
    min-height: 30px;
    padding: 5px 10px;
    border: 1px solid var(--sp-border);
    border-radius: 7px;
    background: var(--sp-panel-soft);
    color: inherit;
    font-size: .72rem;
}
.status-pro__export-preset-picker {
    display: flex;
    gap: 6px;
}
.status-pro__export-preset-picker select { flex: 1 1 auto; min-width: 0; }
.status-pro__export-modal button { cursor: pointer; }
.status-pro__export-modal .status-pro__export-info,
.status-pro__export-modal .status-pro__export-close { flex: 0 0 auto; padding: 0; }
.status-pro__export-modal button:hover { border-color: var(--sp-accent); }
.status-pro__export-modal button:disabled { cursor: default; opacity: .45; }
.status-pro__export-quick-actions {
    display: flex;
    align-self: flex-end;
    gap: 6px;
}
.status-pro__export-fields {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    align-items: start;
    gap: 10px;
    min-height: 0;
    padding: 12px 16px 28px;
    overflow-y: auto;
    scroll-padding-block: 12px 28px;
    scrollbar-width: thin;
}
.status-pro__export-group {
    align-content: start;
    display: grid;
    gap: 6px;
    padding: 10px;
    border: 1px solid var(--sp-border);
    border-radius: 8px;
    background: color-mix(in srgb, var(--sp-panel-soft) 70%, transparent);
}
.status-pro__export-group > strong {
    margin-bottom: 2px;
    font-size: .7rem;
    letter-spacing: .02em;
    text-transform: uppercase;
    cursor: help;
    text-decoration: underline dotted color-mix(in srgb, currentColor 42%, transparent);
    text-underline-offset: 3px;
}
.status-pro__export-field {
    display: flex;
    align-items: flex-start;
    gap: 7px;
    color: var(--sp-muted);
    font-size: .72rem;
    line-height: 1.25;
    cursor: pointer;
}
.status-pro__export-field input {
    flex: 0 0 auto;
    margin-top: 1px;
    accent-color: var(--sp-accent);
}
.status-pro__export-field:has(input:checked) { color: inherit; }
.status-pro__export-note {
    min-height: 0;
    margin-top: 5px;
    padding: 8px 9px;
    overflow-wrap: anywhere;
    border: 1px solid color-mix(in srgb, var(--sp-border) 76%, transparent);
    border-radius: 7px;
    background: color-mix(in srgb, var(--sp-panel) 54%, transparent);
    font-size: .68rem;
    line-height: 1.35;
}
.status-pro__export-note[data-state="available"] { color: var(--sp-good); }
.status-pro__export-note[data-state="partial"] { color: #f59e0b; }
.status-pro__prompt-memory {
    display: flex;
    align-items: flex-start;
    gap: 7px;
    padding: 8px 9px;
    border: 1px solid color-mix(in srgb, var(--sp-accent) 42%, var(--sp-border));
    border-radius: 7px;
    background: color-mix(in srgb, var(--sp-accent) 7%, var(--sp-panel));
    color: var(--sp-text);
    font-size: .7rem;
    line-height: 1.3;
    cursor: pointer;
}
.status-pro__prompt-memory input {
    flex: 0 0 auto;
    margin-top: 1px;
    accent-color: var(--sp-accent);
}
.status-pro__export-footer {
    align-self: end;
    min-height: 54px;
    padding: 11px 16px;
    border-top: 1px solid var(--sp-border);
    background: var(--sp-panel);
}
.status-pro__export-footer > div { display: flex; gap: 7px; flex-wrap: wrap; }
.status-pro__export-footer .status-pro__export-confirm {
    border-color: var(--sp-accent);
    background: var(--sp-accent-strong);
    color: white;
    font-weight: 700;
}
.status-pro__history-empty {
    padding: 14px;
    border: 1px dashed var(--sp-border);
    border-radius: 8px;
    text-align: center;
}
.status-pro__history {
    display: grid;
    gap: 6px;
    grid-auto-rows: max-content;
    align-content: start;
    max-height: 360px;
    overflow-y: auto;
    scrollbar-width: thin;
}
.status-pro__run {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--sp-border);
    border-radius: 8px;
    background: color-mix(in srgb, var(--sp-panel-soft) 78%, transparent);
}
.status-pro__run > summary {
    display: grid;
    grid-template-columns: 18px minmax(72px, auto) minmax(120px, 1fr) repeat(3, minmax(80px, auto)) minmax(52px, auto) 18px;
    gap: 10px 16px;
    align-items: center;
    padding: 9px 10px;
    cursor: pointer;
    list-style: none;
}
.status-pro__run-select {
    margin: 0;
    accent-color: var(--sp-accent);
    cursor: pointer;
}
.status-pro__run > summary::-webkit-details-marker { display: none; }
.status-pro__run > summary::after {
    content: "⌄";
    color: var(--sp-muted);
    font-size: .8rem;
}
.status-pro__run[open] > summary::after { content: "⌃"; }
.status-pro__run-title {
    font-size: .76rem;
    font-weight: 750;
}
.status-pro__run-model {
    overflow: hidden;
    font-size: .75rem;
    font-weight: 650;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-pro__run-summary-value {
    color: var(--sp-muted);
    font-size: .7rem;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
.status-pro__run-status {
    display: inline-flex;
    width: fit-content;
    padding: 2px 6px;
    border-radius: 99px;
    background: color-mix(in srgb, var(--sp-good) 16%, transparent);
    color: var(--sp-good);
}
.status-pro__run-status[data-status="aborted"] {
    background: color-mix(in srgb, #f59e0b 15%, transparent);
    color: #f59e0b;
}
.status-pro__run-status[data-status="window"] {
    background: color-mix(in srgb, var(--sp-accent) 16%, transparent);
    color: var(--sp-accent);
}
.status-pro__run-status[data-status="failed"] {
    background: color-mix(in srgb, #ef4444 15%, transparent);
    color: #ef4444;
}
.status-pro__run-status[data-status="incomplete"] {
    background: color-mix(in srgb, #f59e0b 15%, transparent);
    color: #f59e0b;
}
.status-pro__task-count {
    color: var(--sp-accent);
    font-size: .69rem;
    font-weight: 700;
    white-space: nowrap;
}
.status-pro__task-runs {
    display: grid;
    gap: 6px;
    padding: 8px;
    border-top: 1px solid color-mix(in srgb, var(--sp-border) 70%, transparent);
    background: color-mix(in srgb, var(--sp-panel) 32%, transparent);
}
.status-pro__run--grouped {
    background: color-mix(in srgb, var(--sp-panel-soft) 56%, transparent);
}
.status-pro__gallery-view {
    min-height: 26px;
    padding: 3px 9px;
    border: 1px solid color-mix(in srgb, var(--sp-accent) 68%, var(--sp-border));
    border-radius: 6px;
    background: color-mix(in srgb, var(--sp-accent) 10%, transparent);
    color: var(--sp-accent);
    font-size: .68rem;
    font-weight: 700;
    cursor: pointer;
}
.status-pro__gallery-view:hover:not(:disabled) {
    background: color-mix(in srgb, var(--sp-accent) 18%, transparent);
}
.status-pro__gallery-view:disabled {
    opacity: .45;
    cursor: default;
}
.status-pro__gallery-feedback {
    grid-column: 2 / -2;
    grid-row: 2;
    color: var(--sp-muted);
    font-size: .68rem;
    line-height: 1.25;
}
.status-pro__gallery-feedback[data-status="selected"] { color: var(--sp-good); }
.status-pro__gallery-feedback[data-status="imported"] { color: var(--sp-good); }
.status-pro__gallery-feedback[data-status="missing"],
.status-pro__gallery-feedback[data-status="error"] { color: #f59e0b; }
.status-pro__run-detail {
    display: grid;
    gap: 9px;
    padding: 0 10px 10px;
    border-top: 1px solid color-mix(in srgb, var(--sp-border) 70%, transparent);
}
.status-pro__run-fields {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
    align-items: start;
    gap: 9px 12px;
    padding-top: 9px;
}
.status-pro__run-field {
    display: grid;
    align-content: start;
    align-self: start;
    gap: 4px;
    min-width: 0;
}
.status-pro__run-field dt {
    min-height: 2.15em;
    padding: 3px 6px;
    border-radius: 4px;
    background: color-mix(in srgb, var(--sp-accent) 9%, var(--sp-panel-soft));
    color: color-mix(in srgb, var(--sp-accent) 62%, var(--sp-text));
    font-size: .62rem;
    font-weight: 700;
    line-height: 1.15;
    text-transform: uppercase;
}
.status-pro__run-field dd {
    overflow-wrap: anywhere;
    margin: 0;
    font-size: .72rem;
    line-height: 1.3;
    white-space: pre-line;
}
.status-pro__run-field-actions {
    display: flex;
    align-items: center;
    gap: 5px;
    flex-wrap: wrap;
}
.status-pro__run-field-actions .status-pro__gallery-view {
    min-height: 25px;
    padding-inline: 8px;
}
.status-pro__stage-breakdown {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}
.status-pro__stage-breakdown span {
    --sp-stage-color: var(--sp-muted);
    padding: 3px 7px;
    border: 1px solid color-mix(in srgb, var(--sp-stage-color) 55%, var(--sp-border));
    border-radius: 99px;
    background: color-mix(in srgb, var(--sp-stage-color) 11%, var(--sp-panel-soft));
    color: color-mix(in srgb, var(--sp-stage-color) 72%, var(--sp-text));
    font-size: .67rem;
    font-variant-numeric: tabular-nums;
}
.status-pro__stage-breakdown span[data-stage="prepare"] { --sp-stage-color: #3b82f6; }
.status-pro__stage-breakdown span[data-stage="input"] { --sp-stage-color: #f59e0b; }
.status-pro__stage-breakdown span[data-stage="encode"] { --sp-stage-color: #8b5cf6; }
.status-pro__stage-breakdown span[data-stage="denoise"] { --sp-stage-color: #06b6d4; }
.status-pro__stage-breakdown span[data-stage="decode"] { --sp-stage-color: #a855f7; }
.status-pro__stage-breakdown span[data-stage="enhance"] { --sp-stage-color: #ec4899; }
.status-pro__stage-breakdown span[data-stage="save"] { --sp-stage-color: #14b8a6; }
.status-pro__stage-breakdown span[data-stage="unaccounted"] {
    border-color: color-mix(in srgb, var(--sp-text) 25%, var(--sp-border));
    background: repeating-linear-gradient(
        135deg,
        color-mix(in srgb, var(--sp-text) 12%, var(--sp-panel-soft)) 0 4px,
        color-mix(in srgb, var(--sp-text) 3%, var(--sp-panel-soft)) 4px 8px
    );
    color: var(--sp-muted);
}
.status-pro__timing-overview {
    display: grid;
    gap: 5px;
}
.status-pro__timing-overview-label {
    color: var(--sp-muted);
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: .02em;
    text-transform: uppercase;
}
.status-pro__timing-bar {
    display: flex;
    width: 100%;
    height: 8px;
    overflow: hidden;
    border: 1px solid color-mix(in srgb, var(--sp-border) 82%, transparent);
    border-radius: 99px;
    background: var(--sp-panel-soft);
}
.status-pro__timing-segment { min-width: 2px; }
.status-pro__timing-segment[data-stage="prepare"] { background: #3b82f6; }
.status-pro__timing-segment[data-stage="input"] { background: #f59e0b; }
.status-pro__timing-segment[data-stage="encode"] { background: #8b5cf6; }
.status-pro__timing-segment[data-stage="denoise"] { background: #06b6d4; }
.status-pro__timing-segment[data-stage="decode"] { background: #a855f7; }
.status-pro__timing-segment[data-stage="enhance"] { background: #ec4899; }
.status-pro__timing-segment[data-stage="save"] { background: #14b8a6; }
.status-pro__timing-segment[data-stage="unaccounted"] {
    background: repeating-linear-gradient(
        135deg,
        color-mix(in srgb, var(--sp-text) 24%, var(--sp-panel-soft)) 0 4px,
        color-mix(in srgb, var(--sp-text) 6%, var(--sp-panel-soft)) 4px 8px
    );
}
.status-pro__output-actions {
    display: grid;
    gap: 5px;
    padding: 7px 8px;
    border: 1px solid var(--sp-border);
    border-radius: 7px;
}
.status-pro__output-actions-title {
    color: var(--sp-muted);
    font-size: .64rem;
    font-weight: 700;
    text-transform: uppercase;
}
.status-pro__output-action {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 8px;
    align-items: center;
}
.status-pro__output-action-name {
    overflow: hidden;
    color: var(--sp-muted);
    font-size: .7rem;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-pro__step-log {
    overflow: hidden;
    border: 1px solid var(--sp-border);
    border-radius: 8px;
}
.status-pro__step-log > summary {
    display: flex;
    justify-content: space-between;
    padding: 7px 9px;
    color: var(--sp-muted);
    font-size: .7rem;
    font-weight: 700;
    cursor: pointer;
    list-style: none;
}
.status-pro__step-log > summary::-webkit-details-marker { display: none; }
.status-pro__step-log > summary::after { content: "⌄"; margin-left: auto; }
.status-pro__step-log[open] > summary::after { content: "⌃"; }
.status-pro__step-log-table-wrap {
    max-height: 260px;
    overflow: auto;
    border-top: 1px solid var(--sp-border);
    scrollbar-width: thin;
}
.status-pro__step-log table {
    width: 100%;
    border-collapse: collapse;
    font-size: .67rem;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
.status-pro__step-log th,
.status-pro__step-log td {
    padding: 5px 8px;
    border-bottom: 1px solid color-mix(in srgb, var(--sp-border) 58%, transparent);
    text-align: left;
}
.status-pro__step-log th {
    position: sticky;
    top: 0;
    z-index: 1;
    background: var(--sp-panel-soft);
    color: var(--sp-muted);
    font-size: .6rem;
    text-transform: uppercase;
}
.status-pro__step-skipped { color: #f59e0b; font-weight: 700; }
.status-pro__step-fastest {
    background: color-mix(in srgb, #22c55e 17%, transparent);
    color: color-mix(in srgb, #22c55e 78%, var(--sp-text));
    font-weight: 700;
}
.status-pro__step-slowest {
    background: color-mix(in srgb, #ef4444 15%, transparent);
    color: color-mix(in srgb, #ef4444 72%, var(--sp-text));
    font-weight: 700;
}
.status-pro__badge {
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    padding: 3px 9px;
    border-radius: 7px;
    background: var(--sp-accent-strong);
    color: white;
    font-weight: 700;
    letter-spacing: .01em;
}
.status-pro__live { font-weight: 650; }
.status-pro__summary {
    justify-content: flex-end;
    color: var(--sp-muted);
    font-size: .82rem;
    font-variant-numeric: tabular-nums;
}
.status-pro__summary span:not(:empty) + span:not(:empty)::before {
    content: "•";
    margin-right: 8px;
    opacity: .55;
}
.status-pro__stages {
    display: flex;
    align-items: stretch;
    justify-content: center;
    justify-content: safe center;
    gap: 7px;
    width: 100%;
    overflow-x: auto;
    padding: 1px 1px 5px;
    scrollbar-width: thin;
}
.status-pro__stage {
    position: relative;
    display: grid;
    grid-template-columns: 24px minmax(0, 1fr);
    grid-template-areas: "icon name" "icon time";
    grid-template-rows: auto auto;
    column-gap: 9px;
    row-gap: 3px;
    align-content: center;
    align-items: center;
    flex: 1 1 150px;
    max-width: 260px;
    min-width: 108px;
    min-height: 52px;
    padding: 7px 10px;
    overflow: hidden;
    border: 1px solid var(--sp-border);
    border-radius: 8px;
    background: var(--sp-panel-soft);
    color: inherit;
    text-align: left;
    cursor: pointer;
    opacity: .68;
    transition: flex-grow 280ms ease, min-width 280ms ease, opacity 180ms ease,
                border-color 180ms ease, background-color 180ms ease;
}
.status-pro__stages--inline .status-pro__stage {
    grid-template-columns: 24px minmax(0, auto) minmax(0, auto);
    grid-template-areas: "icon name time";
    grid-template-rows: auto;
    justify-content: center;
    min-height: 44px;
}
.status-pro__stage:hover { opacity: .92; }
.status-pro__stage:focus-visible { outline: 2px solid var(--sp-accent); outline-offset: 2px; }
.status-pro__stage--complete { opacity: .82; }
.status-pro__stage--current {
    flex-grow: 1.7;
    flex-basis: 230px;
    max-width: 420px;
    min-width: 178px;
    opacity: 1;
    border-color: var(--sp-accent);
    background: color-mix(in srgb, var(--sp-accent) 16%, var(--sp-panel-soft));
}
.status-pro__stage--selected:not(.status-pro__stage--current) {
    flex-grow: 1.35;
    max-width: 320px;
    min-width: 150px;
    opacity: 1;
    border-color: color-mix(in srgb, var(--sp-accent) 68%, var(--sp-border));
}
.status-pro__stage--selected::after {
    content: "";
    position: absolute;
    left: 9px;
    right: 9px;
    bottom: 0;
    height: 2px;
    border-radius: 2px;
    background: var(--sp-accent);
}
.status-pro__stage-icon {
    grid-area: icon;
    display: inline-grid;
    place-items: center;
    width: 24px;
    height: 24px;
    border: 1px solid var(--sp-border);
    border-radius: 999px;
    color: var(--sp-muted);
    font-size: .7rem;
    font-weight: 750;
}
.status-pro__stage--complete .status-pro__stage-icon {
    border-color: color-mix(in srgb, var(--sp-good) 70%, var(--sp-border));
    background: color-mix(in srgb, var(--sp-good) 18%, transparent);
    color: var(--sp-good);
}
.status-pro__stage--current .status-pro__stage-icon {
    border-color: var(--sp-accent);
    background: var(--sp-accent);
    color: white;
    animation: status-pro-pulse 1.8s ease-in-out infinite;
}
.status-pro__stage-name {
    grid-area: name;
    overflow: hidden;
    font-size: .82rem;
    font-weight: 700;
    line-height: 1.2;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-pro__stage-time {
    grid-area: time;
    overflow: hidden;
    color: var(--sp-muted);
    font-size: .72rem;
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-pro__downloads {
    margin-top: 7px;
    padding: 10px;
    border: 1px solid color-mix(in srgb, var(--sp-accent) 48%, var(--sp-border));
    border-radius: 8px;
    background: color-mix(in srgb, var(--sp-accent) 7%, var(--sp-panel-soft));
}
.status-pro__downloads-header,
.status-pro__downloads-title {
    display: flex;
    align-items: center;
    gap: 10px;
}
.status-pro__downloads-header {
    justify-content: space-between;
}
.status-pro__downloads-title > div {
    display: grid;
    gap: 2px;
}
.status-pro__downloads-title strong { font-size: .84rem; }
.status-pro__downloads-title span,
.status-pro__download-total {
    color: var(--sp-muted);
    font-size: .72rem;
    font-variant-numeric: tabular-nums;
}
.status-pro__download-indicator {
    display: inline-grid;
    place-items: center;
    width: 27px;
    height: 27px;
    flex: 0 0 auto;
    border-radius: 999px;
    background: var(--sp-accent);
    color: white !important;
    font-size: 1rem !important;
    font-weight: 800;
}
.status-pro__downloads[data-active="true"] .status-pro__download-indicator {
    animation: status-pro-download-pulse 1.4s ease-in-out infinite;
}
.status-pro__download-overall {
    height: 4px;
    margin: 9px 0;
    overflow: hidden;
    border-radius: 99px;
    background: color-mix(in srgb, var(--sp-border) 78%, transparent);
}
.status-pro__download-overall > div {
    width: 0;
    height: 100%;
    border-radius: inherit;
    background: var(--sp-accent);
    transition: width 220ms linear;
}
.status-pro__download-files {
    display: grid;
    gap: 5px;
    max-height: 224px;
    overflow-y: auto;
    padding-right: 2px;
    scrollbar-width: thin;
}
.status-pro__download-file {
    display: grid;
    grid-template-columns: 20px minmax(150px, .8fr) minmax(280px, 1.2fr);
    grid-template-areas: "icon name stats" "icon freshness cycles" "icon bar bar";
    gap: 4px 9px;
    align-items: center;
    min-height: 34px;
    padding: 5px 7px;
    border-radius: 6px;
    background: color-mix(in srgb, var(--sp-panel) 58%, transparent);
}
.status-pro__download-file-icon {
    grid-area: icon;
    display: inline-grid;
    place-items: center;
    width: 20px;
    height: 20px;
    border: 1px solid var(--sp-border);
    border-radius: 999px;
    color: var(--sp-muted);
    font-size: .68rem;
    font-weight: 750;
}
.status-pro__download-file[data-state="downloading"] .status-pro__download-file-icon,
.status-pro__download-file[data-state="retrying"] .status-pro__download-file-icon {
    border-color: var(--sp-accent);
    color: var(--sp-accent);
}
.status-pro__download-file[data-state="complete"] .status-pro__download-file-icon {
    border-color: color-mix(in srgb, var(--sp-good) 72%, var(--sp-border));
    color: var(--sp-good);
}
.status-pro__download-file[data-state="failed"] .status-pro__download-file-icon {
    border-color: #ef4444;
    color: #ef4444;
}
.status-pro__download-file-name {
    grid-area: name;
    overflow: hidden;
    font-size: .74rem;
    font-weight: 650;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-pro__download-file-stats {
    grid-area: stats;
    color: var(--sp-muted);
    font-size: .69rem;
    font-variant-numeric: tabular-nums;
    text-align: right;
    white-space: nowrap;
}
.status-pro__download-file-freshness {
    grid-area: freshness;
    overflow: hidden;
    color: var(--sp-muted);
    font-size: .64rem;
    font-variant-numeric: tabular-nums;
    line-height: 1.15;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-pro__download-file-cycles {
    grid-area: cycles;
    overflow: hidden;
    color: color-mix(in srgb, var(--sp-accent) 74%, var(--sp-muted));
    font-size: .64rem;
    font-variant-numeric: tabular-nums;
    line-height: 1.15;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-pro__download-file-bar {
    grid-area: bar;
    height: 2px;
    overflow: hidden;
    border-radius: 99px;
    background: color-mix(in srgb, var(--sp-border) 70%, transparent);
}
.status-pro__download-file-bar > span {
    display: block;
    width: 0;
    height: 100%;
    border-radius: inherit;
    background: var(--sp-accent);
    transition: width 180ms linear;
}
.status-pro__download-file[data-state="complete"] .status-pro__download-file-bar > span {
    background: var(--sp-good);
}
.status-pro__detail {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    min-height: 56px;
    margin-top: 7px;
    padding: 9px 10px;
    border: 1px solid var(--sp-border);
    border-radius: 8px;
    background: color-mix(in srgb, var(--sp-panel-soft) 82%, transparent);
}
.status-pro__detail-copy {
    display: grid;
    flex: 1 1 auto;
    gap: 2px;
    min-width: 0;
}
.status-pro__detail-copy strong { font-size: .86rem; }
.status-pro__detail-copy span {
    max-width: 100%;
    overflow: hidden;
    color: var(--sp-muted);
    font-size: .74rem;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-pro__detail-copy .status-pro__detail-model {
    color: color-mix(in srgb, var(--sp-accent) 76%, var(--sp-muted));
    font-weight: 650;
}
.status-pro__detail-copy .status-pro__detail-model--list {
    display: grid;
    gap: 2px;
    max-width: none;
    overflow: visible;
    text-overflow: clip;
    white-space: normal;
}
.status-pro__detail-copy .status-pro__detail-activities {
    display: grid;
    gap: 2px;
    max-width: none;
    overflow: visible;
    color: var(--sp-text);
    text-overflow: clip;
    white-space: normal;
}
.status-pro__detail-activity-line {
    display: block;
    min-width: 0;
    overflow-wrap: anywhere;
}
.status-pro__detail-model-line {
    display: block;
    min-width: 0;
    overflow-wrap: anywhere;
}
.status-pro__metrics {
    display: grid;
    flex: 0 0 auto;
    grid-template-columns: repeat(5, minmax(72px, auto));
    gap: 9px 16px;
    margin: 0;
}
.status-pro__metrics div { display: grid; gap: 1px; }
.status-pro__metrics dt {
    color: var(--sp-muted);
    font-size: .64rem;
    line-height: 1.1;
    text-transform: uppercase;
}
.status-pro__metrics dd {
    margin: 0;
    font-size: .76rem;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
.status-pro__overall-track {
    height: 3px;
    margin-top: 8px;
    overflow: hidden;
    border-radius: 99px;
    background: color-mix(in srgb, var(--sp-border) 75%, transparent);
}
.status-pro__overall-fill {
    width: 0;
    height: 100%;
    border-radius: inherit;
    background: var(--sp-accent);
    transition: width 180ms linear;
}
.status-pro__overall-fill--indeterminate {
    width: 28% !important;
    animation: status-pro-indeterminate 1.35s ease-in-out infinite;
}
@keyframes status-pro-indeterminate {
    0% { transform: translateX(-110%); }
    50% { transform: translateX(135%); }
    100% { transform: translateX(360%); }
}
@keyframes status-pro-pulse {
    0%, 100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--sp-accent) 35%, transparent); }
    50% { box-shadow: 0 0 0 4px transparent; }
}
@keyframes status-pro-download-pulse {
    0%, 100% { transform: translateY(0); opacity: 1; }
    50% { transform: translateY(2px); opacity: .72; }
}
@container status-pro (max-width: 760px) {
    .status-pro__stage {
        flex-basis: 128px;
        max-width: none;
        min-width: 118px;
    }
    .status-pro__stage--current {
        flex-basis: 210px;
        max-width: none;
    }
    .status-pro__detail {
        align-items: stretch;
        flex-direction: column;
    }
    .status-pro__detail-copy { width: 100%; }
    .status-pro__detail-copy span {
        max-width: none;
        white-space: normal;
    }
    .status-pro__metrics {
        grid-template-columns: repeat(auto-fit, minmax(84px, 1fr));
        width: 100%;
    }
    .status-pro__run > summary {
        grid-template-columns: 18px minmax(72px, auto) minmax(0, 1fr) minmax(80px, auto) minmax(52px, auto) 18px;
        gap: 5px 12px;
    }
    .status-pro__run > summary > :nth-child(5) { grid-column: 3; grid-row: 2; }
    .status-pro__run > summary > :nth-child(6) { grid-column: 4; grid-row: 2; }
    .status-pro__run > summary > .status-pro__gallery-view { grid-column: 5; grid-row: 1 / span 2; }
    .status-pro__run > summary > .status-pro__task-count { grid-column: 5; grid-row: 1 / span 2; align-self: center; }
    .status-pro__run > summary > .status-pro__gallery-feedback { grid-column: 2 / -2; grid-row: 3; }
    .status-pro__run > summary::after { grid-column: 6; grid-row: 1 / span 2; }
}
@container status-pro (max-width: 650px) {
    .status-pro__header { align-items: flex-start; flex-direction: column; }
    .status-pro__header-actions { justify-content: space-between; width: 100%; }
    .status-pro__summary { justify-content: flex-start; }
    .status-pro__stage { flex-basis: 104px; min-width: 104px; }
    .status-pro__stage--current { min-width: 174px; }
}
@container status-pro (max-width: 520px) {
    .status-pro__metrics { grid-template-columns: repeat(2, minmax(90px, 1fr)); }
    .status-pro__download-file {
        grid-template-columns: 20px minmax(0, 1fr);
        grid-template-areas: "icon name" "icon stats" "icon freshness" "icon cycles" "icon bar";
    }
    .status-pro__download-file-stats { text-align: left; }
    .status-pro__history-drawer-header { align-items: stretch; flex-direction: column; }
    .status-pro__history-actions { flex-wrap: wrap; width: 100%; }
    .status-pro__history-actions button:not(.status-pro__history-settings):not(.status-pro__history-expand) { flex: 1 1 auto; }
    .status-pro__history-separator { display: none; }
    .status-pro__export-controls { align-items: stretch; flex-direction: column; }
    .status-pro__export-controls label,
    .status-pro__export-preset-control { width: 100%; }
    .status-pro__export-quick-actions { align-self: stretch; }
    .status-pro__export-quick-actions button { flex: 1; }
    .status-pro__export-footer { align-items: stretch; flex-direction: column; }
    .status-pro__export-footer > div { width: 100%; }
    .status-pro__export-footer button { flex: 1; }
    .status-pro__export-header-actions { width: 100%; }
    .status-pro__history-persistence { flex: 1 1 auto; }
    .status-pro__export-privacy { margin-right: auto; text-align: left; }
    .status-pro__history-modal {
        width: calc(100vw - 12px);
        height: calc(100dvh - 12px);
    }
    .status-pro__history-modal-content { padding-inline: 7px; }
    .status-pro__history-modal-header { padding-inline: 10px; }
}
@container status-pro (max-width: 900px) {
    .status-pro { padding: 10px; }
    .status-pro__header {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto auto;
        grid-template-rows: auto auto;
        gap: 8px 10px;
        align-items: center;
        margin-bottom: 9px;
    }
    .status-pro__heading {
        grid-column: 1;
        grid-row: 1;
        min-width: 0;
        flex-wrap: nowrap;
    }
    .status-pro__live {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .status-pro__header-actions { display: contents; }
    .status-pro__summary {
        grid-column: 1 / -1;
        grid-row: 2;
        justify-content: center;
        min-width: 0;
        text-align: center;
    }
    .status-pro__history-toggle {
        grid-column: 2;
        grid-row: 1;
    }
    .status-pro__collapse {
        grid-column: 3;
        grid-row: 1;
    }
    .status-pro__stages { gap: 6px; }
    .status-pro__stages .status-pro__stage {
        grid-template-columns: 24px;
        grid-template-areas: "icon";
        grid-template-rows: auto;
        justify-content: center;
        flex: 0 0 44px;
        width: 44px;
        min-width: 44px;
        max-width: 44px;
        min-height: 58px;
        padding: 7px 9px;
        column-gap: 0;
    }
    .status-pro__stages .status-pro__stage:not(.status-pro__stage--selected) .status-pro__stage-name,
    .status-pro__stages .status-pro__stage:not(.status-pro__stage--selected) .status-pro__stage-time {
        display: none;
    }
    .status-pro__stages .status-pro__stage--selected {
        grid-template-columns: 24px minmax(0, auto);
        grid-template-areas: "icon name" "icon time";
        grid-template-rows: auto auto;
        justify-content: center;
        flex: 1 1 220px;
        width: auto;
        min-width: 170px;
        max-width: none;
        column-gap: 9px;
    }
    .status-pro__stages .status-pro__stage--selected .status-pro__stage-name,
    .status-pro__stages .status-pro__stage--selected .status-pro__stage-time {
        display: block;
    }
    .status-pro__history-drawer-header {
        display: grid;
        grid-template-columns: minmax(0, 1fr);
        align-items: center;
        gap: 8px;
        padding: 7px 8px;
    }
    .status-pro__history-actions {
        justify-content: flex-start;
        width: 100%;
    }
}
@media (min-width: 901px) {
    .status-pro__history-drawer[data-expanded="true"] .status-pro__history-drawer-header {
        display: flex;
        align-items: center;
        flex-direction: row;
        justify-content: space-between;
        gap: 10px 16px;
        padding: 8px 9px;
    }
    .status-pro__history-drawer[data-expanded="true"] .status-pro__history-actions {
        justify-content: flex-end;
        width: auto;
        flex-wrap: nowrap;
    }
    .status-pro__history-drawer[data-expanded="true"] .status-pro__history-actions button:not(.status-pro__history-settings):not(.status-pro__history-expand) {
        flex: 0 1 auto;
    }
    .status-pro__history-drawer[data-expanded="true"] .status-pro__history-separator { display: block; }
    .status-pro__history-drawer[data-expanded="true"] .status-pro__run > summary {
        grid-template-columns: 18px minmax(72px, auto) minmax(120px, 1fr) repeat(3, minmax(80px, auto)) minmax(52px, auto) 18px;
        gap: 10px 16px;
    }
    .status-pro__history-drawer[data-expanded="true"] .status-pro__run > summary > :nth-child(5),
    .status-pro__history-drawer[data-expanded="true"] .status-pro__run > summary > :nth-child(6),
    .status-pro__history-drawer[data-expanded="true"] .status-pro__run > summary > .status-pro__gallery-view,
    .status-pro__history-drawer[data-expanded="true"] .status-pro__run > summary > .status-pro__task-count {
        grid-column: auto;
        grid-row: auto;
    }
    .status-pro__history-drawer[data-expanded="true"] .status-pro__run > summary > .status-pro__gallery-feedback {
        grid-column: 2 / -2;
        grid-row: 2;
    }
    .status-pro__history-drawer[data-expanded="true"] .status-pro__run > summary::after {
        grid-column: auto;
        grid-row: auto;
    }
}
@media (prefers-reduced-motion: reduce) {
    .status-pro__stage, .status-pro__overall-fill { transition: none; }
    .status-pro__stage--current .status-pro__stage-icon { animation: none; }
    .status-pro__overall-fill--indeterminate { width: 100% !important; animation: none; opacity: .45; }
    .status-pro__downloads[data-active="true"] .status-pro__download-indicator { animation: none; }
}
`;

    function appRoot() {
        if (window.gradioApp) return window.gradioApp();
        const app = document.querySelector("gradio-app");
        return app ? (app.shadowRoot || app) : document;
    }

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function optionalNumber(value) {
        if (value === null || value === undefined || value === "") return null;
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    }

    function formatDuration(seconds, approximate = false) {
        if (!Number.isFinite(seconds) || seconds < 0) return "—";
        const rounded = Math.max(0, Math.round(seconds));
        const hours = Math.floor(rounded / 3600);
        const minutes = Math.floor((rounded % 3600) / 60);
        const secs = rounded % 60;
        let value;
        if (hours > 0) value = `${hours}h ${minutes}m`;
        else if (minutes > 0) value = `${minutes}m ${secs}s`;
        else value = `${secs}s`;
        return approximate ? `~${value}` : value;
    }

    function formatStepDuration(seconds) {
        if (!Number.isFinite(seconds) || seconds < 0) return "—";
        if (seconds < 10) return `${seconds.toFixed(2)}s`;
        if (seconds < 60) return `${seconds.toFixed(1)}s`;
        return formatDuration(seconds);
    }

    function formatLiveStepDuration(seconds) {
        if (!Number.isFinite(seconds) || seconds < 0) return "—";
        if (seconds < 0.1) return `${seconds.toFixed(2)}s`;
        if (seconds < 1) return `${seconds.toFixed(1)}s`;
        return formatDuration(seconds);
    }

    function formatClock(timestamp) {
        if (!Number.isFinite(timestamp)) return "—";
        return new Intl.DateTimeFormat(undefined, {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit"
        }).format(new Date(timestamp));
    }

    function formatDateTime(timestamp) {
        if (!Number.isFinite(timestamp)) return "—";
        return new Intl.DateTimeFormat(undefined, {
            year: "numeric",
            month: "short",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit"
        }).format(new Date(timestamp));
    }

    function localIsoTimestamp(date = new Date()) {
        const offset = -date.getTimezoneOffset();
        const sign = offset >= 0 ? "+" : "-";
        const pad = value => String(Math.abs(value)).padStart(2, "0");
        return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
            `T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}` +
            `${sign}${pad(Math.floor(offset / 60))}:${pad(offset % 60)}`;
    }

    function formatBytes(bytes) {
        if (!Number.isFinite(bytes) || bytes < 0) return "—";
        const units = ["B", "KB", "MB", "GB", "TB"];
        let value = bytes;
        let index = 0;
        while (value >= 1024 && index < units.length - 1) {
            value /= 1024;
            index += 1;
        }
        const digits = value >= 100 || index === 0 ? 0 : (value >= 10 ? 1 : 2);
        return `${value.toFixed(digits)} ${units[index]}`;
    }

    function formatRate(bytesPerSecond) {
        return Number.isFinite(bytesPerSecond) && bytesPerSecond > 0
            ? `${formatBytes(bytesPerSecond)}/s`
            : "";
    }

    const RESOURCE_FIELDS = [
        "ram_rss_bytes",
        "ram_vms_bytes",
        "vram_allocated_bytes",
        "vram_reserved_bytes",
        "vram_device_used_bytes",
        "vram_device_free_bytes"
    ];

    function observeResourceSample(run, sample, includeInAverage = true) {
        if (!run || !sample || typeof sample !== "object") return;
        const sampledAt = optionalNumber(sample.sampled_at);
        if (Number.isFinite(sampledAt) && sampledAt === run._last_resource_sample_at) return;
        if (Number.isFinite(sampledAt)) run._last_resource_sample_at = sampledAt;
        const resources = run.resources || {
            sample_count: 0,
            observation_count: 0,
            sampling_interval_seconds: 0.5,
            scope: "Wan2GP process and active CUDA device",
            gpu_device_index: null,
            gpu_name: null,
            vram_device_total_bytes: null,
            metrics: {}
        };
        resources.observation_count += 1;
        if (includeInAverage) resources.sample_count += 1;
        if (sample.gpu_device_index !== null && sample.gpu_device_index !== undefined) {
            resources.gpu_device_index = sample.gpu_device_index;
        }
        if (sample.gpu_name) resources.gpu_name = String(sample.gpu_name);
        const gpuTotal = optionalNumber(sample.vram_device_total_bytes);
        if (Number.isFinite(gpuTotal)) resources.vram_device_total_bytes = gpuTotal;
        RESOURCE_FIELDS.forEach(field => {
            const value = optionalNumber(sample[field]);
            if (!Number.isFinite(value)) return;
            const metric = resources.metrics[field] || {
                start_bytes: value,
                end_bytes: value,
                peak_bytes: value,
                total_bytes: 0,
                sample_count: 0
            };
            metric.end_bytes = value;
            metric.peak_bytes = Math.max(metric.peak_bytes, value);
            if (includeInAverage) {
                metric.total_bytes += value;
                metric.sample_count += 1;
                metric.average_bytes = Math.round(metric.total_bytes / metric.sample_count);
            }
            resources.metrics[field] = metric;
        });
        run.resources = resources;
    }

    function observePerformanceTelemetry(run, telemetry) {
        if (!run || !telemetry) return;
        observeResourceSample(run, telemetry.resource_sample);
        const performance = telemetry.performance;
        if (!performance || typeof performance !== "object") return;
        const observerId = String(performance.id || "observer");
        run.step_performance = Array.isArray(run.step_performance) ? run.step_performance : [];
        run._performance_step_keys = run._performance_step_keys || {};
        (Array.isArray(performance.steps) ? performance.steps : []).forEach(step => {
            if (!step || typeof step !== "object") return;
            const completedAt = optionalNumber(step.completed_at);
            const completedMs = Number.isFinite(completedAt) ? completedAt * 1000 : null;
            if (Number.isFinite(completedMs) && Number.isFinite(run.started_at) && completedMs < run.started_at - 1000) return;
            if (Number.isFinite(completedMs) && Number.isFinite(run.completed_at) && completedMs > run.completed_at + 1000) return;
            const key = `${observerId}:${step.sequence}`;
            if (run._performance_step_keys[key]) return;
            run._performance_step_keys[key] = true;
            const copy = cloneJson(step, {});
            copy.observer_id = observerId;
            run.step_performance.push(copy);
            if (run.step_performance.length > MAX_STEP_RECORDS) {
                run.step_performance = run.step_performance.slice(-MAX_STEP_RECORDS);
                run._performance_step_keys = Object.fromEntries(run.step_performance.map(record => [`${record.observer_id}:${record.sequence}`, true]));
                run.step_performance_source_truncated = true;
            }
            observeResourceSample(run, copy.memory, false);
        });
        if (performance.steps_truncated) run.step_performance_source_truncated = true;
    }

    function finalizePerformance(run) {
        if (!run) return;
        delete run._last_resource_sample_at;
        delete run._performance_step_keys;
        const resources = run.resources;
        if (resources && resources.metrics) {
            Object.values(resources.metrics).forEach(metric => {
                delete metric.total_bytes;
                delete metric.sample_count;
            });
        }
        const steps = Array.isArray(run.step_performance) ? run.step_performance : [];
        const validDurations = steps.map(step => optionalNumber(step && step.duration_seconds)).filter(Number.isFinite);
        const skipObservations = steps.map(step => optionalNumber(step && step.skipped_delta)).filter(Number.isFinite);
        const skipped = skipObservations.length ? skipObservations.reduce((total, value) => total + value, 0) : null;
        const passMap = new Map();
        steps.forEach(step => {
            const observerId = String(step && step.observer_id || "observer");
            const passNo = optionalNumber(step && step.pass_no);
            const phaseNo = optionalNumber(step && step.phase);
            const identity = Number.isFinite(passNo) && passNo > 0
                ? `pass:${passNo}`
                : `phase:${Number.isFinite(phaseNo) ? phaseNo : 0}`;
            const key = `${observerId}:${identity}`;
            let pass = passMap.get(key);
            if (!pass) {
                pass = {
                    observer_id: observerId,
                    phase: Number.isFinite(phaseNo) ? phaseNo : null,
                    pass_no: Number.isFinite(passNo) && passNo > 0 ? passNo : null,
                    label: "",
                    observed_steps: 0,
                    unique_steps: new Set(),
                    configured_steps: null,
                    skipped_steps: 0,
                    duration_seconds: 0
                };
                passMap.set(key, pass);
            }
            if (step && step.label) pass.label = String(step.label);
            pass.observed_steps += 1;
            const stepNo = optionalNumber(step && step.step);
            if (Number.isFinite(stepNo)) pass.unique_steps.add(stepNo);
            const totalSteps = optionalNumber(step && step.total_steps);
            if (Number.isFinite(totalSteps)) pass.configured_steps = Math.max(pass.configured_steps || 0, totalSteps);
            const skippedDelta = optionalNumber(step && step.skipped_delta);
            if (Number.isFinite(skippedDelta)) pass.skipped_steps += skippedDelta;
            const duration = optionalNumber(step && step.duration_seconds);
            if (Number.isFinite(duration)) pass.duration_seconds += duration;
        });
        const passes = Array.from(passMap.values()).map((pass, index) => ({
            observer_id: pass.observer_id,
            phase: pass.phase,
            pass_no: pass.pass_no,
            label: pass.label || (Number.isFinite(pass.pass_no) ? `Pass ${pass.pass_no}` : `Phase ${Number.isFinite(pass.phase) && pass.phase > 0 ? pass.phase : index + 1}`),
            observed_steps: pass.observed_steps,
            unique_steps: pass.unique_steps.size,
            configured_steps: pass.configured_steps,
            skipped_steps: pass.skipped_steps,
            duration_seconds: Math.round(pass.duration_seconds * 10000) / 10000
        }));
        run.step_summary = {
            recorded_steps: steps.length,
            observed_passes: passes.length,
            passes,
            skipped_steps: skipped,
            average_seconds: validDurations.length ? validDurations.reduce((sum, value) => sum + value, 0) / validDurations.length : null,
            fastest_seconds: validDurations.length ? Math.min(...validDurations) : null,
            slowest_seconds: validDurations.length ? Math.max(...validDurations) : null,
            truncated: Boolean(run.step_performance_source_truncated)
        };
    }

    function parseDuration(text) {
        const source = String(text || "");
        let seconds = 0;
        let matched = false;
        const hours = source.match(/(\d+(?:\.\d+)?)\s*h(?:ours?)?\b/i);
        const minutes = source.match(/(\d+(?:\.\d+)?)\s*m(?:in(?:utes?)?)?\b/i);
        const secs = source.match(/(\d+(?:\.\d+)?)\s*s(?:ec(?:onds?)?)?\b/i);
        if (hours) { seconds += Number(hours[1]) * 3600; matched = true; }
        if (minutes) { seconds += Number(minutes[1]) * 60; matched = true; }
        if (secs) { seconds += Number(secs[1]); matched = true; }
        if (matched) return seconds;
        const clock = source.match(/\b(?:(\d+):)?(\d{1,2}):(\d{2})\b/);
        if (!clock) return null;
        return Number(clock[1] || 0) * 3600 + Number(clock[2]) * 60 + Number(clock[3]);
    }

    function parsePercent(levelText, progressBar) {
        const match = String(levelText || "").match(/(\d+(?:\.\d+)?)\s*%/);
        if (match) return clamp(Number(match[1]), 0, 100);
        if (progressBar) {
            const width = String(progressBar.style.width || "").match(/(\d+(?:\.\d+)?)\s*%/);
            if (width) return clamp(Number(width[1]), 0, 100);
        }
        return null;
    }

    function parseSteps(metaText) {
        const match = String(metaText || "").match(/(\d+)\s*\/\s*(\d+)(?:\s*steps?)?/i);
        if (!match) return { current: null, total: null };
        return { current: Number(match[1]), total: Number(match[2]) };
    }

    function parseProgressTiming(levelText) {
        const timing = String(levelText || "").split("|").pop().trim();
        const values = timing.split(/\s*\/\s*/).map(parseDuration).filter(Number.isFinite);
        return {
            elapsed: values[0] ?? null,
            total: values.length >= 2 ? values[1] : null
        };
    }

    function stageIdFor(rawName) {
        const name = String(rawName || "").toLowerCase();
        const modelLifecycle = /\b(?:prepar|load|loading|loaded|unload|unloading|unloaded|releas|download|queue|cache|compil|warm.?up|initializ|abort|cancel|interrupt)\w*\b/.test(name) &&
            /\b(?:model|weight|checkpoint|transformer|encoder|vae|whisper|vocoder|lora|file|asset|prompt enhancer)\w*\b/.test(name);
        if (modelLifecycle || /\b(?:initializ|abort|cancel|interrupt)\w*\b/.test(name)) return "prepare";
        if (/\b(?:sav(?:e|ing|ed)?|export\w*|writ(?:e|ing|ten)?|mux\w*|remux\w*|finaliz\w*)\b/.test(name)) return "save";

        // Semantic prompt/text work belongs to Encode even when it uses words such as
        // "enhancing" or mentions references. Check it before media preprocessing.
        if (/(?:enhanc\w*\s+prompt|prompt\s+enhanc\w*|(?:prompt|text|caption)\s+(?:and\s+references?\s+)?encod\w*|encod\w*\s+(?:(?:h3\s+)?prompt|speaker\s+\d*\s*reference)|condition\w*|embed\w*|token\w*|text\s+feature)/.test(name)) return "encode";

        const inputContext = /\b(?:input|control|source|guide|mask|pose|depth|canny|face|movement|reference|background)\w*\b/.test(name);
        const inputWork = /\b(?:prepar|pre.?process|load|extract|remov|resiz|crop|trim|normaliz|separat|align|encod|decod)\w*\b/.test(name);
        const mediaPreprocessing = /\b(?:prepar|pre.?process|load|extract|remov|resiz|crop|trim|normaliz|separat|align)\w*\b.*\b(?:frame|image|video|audio)\w*\b|\b(?:frame|image|video|audio)\w*\b.*\b(?:prepar|pre.?process|load|extract|remov|resiz|crop|trim|normaliz|separat|align)\w*\b/.test(name);
        if (/\bvae\s+encod\w*\b/.test(name) ||
            /\b(?:pre.?process|extract)\w*\b/.test(name) ||
            (inputContext && inputWork) || mediaPreprocessing ||
            /\b(?:extracting\s+(?:pose|depth|face)|removing\s+(?:image\s+)?references?\s+background)\b/.test(name)) return "input";

        if (/\b(?:denois|diffus|sampl|synthesis|synthes|generating\s+(?:audio|waveform|speech)|spectrum\s+smoothing\s+replay)/.test(name)) return "denoise";
        if (/(?:vae\s*decod|decod|reconstruct)/.test(name)) return "decode";
        if (/(?:post.?process|upscal|upsampl|interpol|color correction|film grain|tcdecoder|seedvc|voice replacement|audio post|soundtrack|enhanc)/.test(name)) return "post";
        if (/(?:encod|prompt|condition|embed|feature|token)/.test(name)) return "encode";
        if (/(?:prepar|initial|load|download|queue|cache|start)/.test(name)) return "prepare";
        return "prepare";
    }

    function phaseInfo(rawName) {
        const label = String(rawName || "Preparing")
            .replace(/^(?:(?:prompt|sample|sliding window)\s+\d+\s*\/\s*\d+\s*,?\s*)+/i, "")
            .replace(/^\s*-\s*/, "")
            .trim() || "Preparing";
        const kind = stageIdFor(label);
        const phaseMatch = label.match(/\bdenoising\s+(first|second|third|\d+(?:st|nd|rd|th)?)\s+phase\b/i);
        const phaseNumber = phaseMatch
            ? ({ first: 1, second: 2, third: 3 }[phaseMatch[1].toLowerCase()] || Number.parseInt(phaseMatch[1], 10))
            : null;
        const key = label.toLowerCase()
            .replace(/[^a-z0-9]+/g, "-")
            .replace(/^-+|-+$/g, "") || kind;
        return { id: `${kind}:${key}`, kind, label, phaseNumber };
    }

    function stageNameFrom(levelText) {
        const first = String(levelText || "").split("|")[0].trim();
        return first.replace(/\s+-\s+\d+(?:\.\d+)?\s*%\s*$/, "").trim() || "Preparing";
    }

    function loadHistory() {
        try {
            const parsed = JSON.parse(window.localStorage.getItem(HISTORY_KEY) || "{}");
            return parsed && typeof parsed === "object" ? parsed : {};
        } catch (_) {
            return {};
        }
    }

    function saveHistory(history) {
        try {
            window.localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
        } catch (_) {
            // History is optional; private browsing and locked-down browsers may reject it.
        }
    }

    function normalizeHistoryPersistence(mode) {
        return ["persistent", "browser", "runtime"].includes(mode) ? mode : "runtime";
    }

    function runHistoryStorage(mode) {
        return normalizeHistoryPersistence(mode) === "browser" ? window.sessionStorage : window.localStorage;
    }

    function runHistoryKey(mode) {
        const normalized = normalizeHistoryPersistence(mode);
        if (normalized === "persistent") return RUN_HISTORY_KEY;
        if (normalized === "browser") return SESSION_RUN_HISTORY_KEY;
        return RUNTIME_RUN_HISTORY_KEY;
    }

    function modeStoresPrompts(mode) {
        return normalizeHistoryPersistence(mode) === "browser";
    }

    function historyPersistenceLabel(mode) {
        const normalized = normalizeHistoryPersistence(mode);
        if (normalized === "persistent") return "Until manually cleared";
        if (normalized === "browser") return "Until browser tab closes";
        return "Until WanGP restarts";
    }

    function historyPersistenceConfirmation(mode) {
        const normalized = normalizeHistoryPersistence(mode);
        if (normalized === "persistent") {
            return "Keep history until manually cleared?\n\nPrompt-free history will remain in this browser after WanGP and the browser are restarted. You can remove it at any time with Clear selected or Clear history.";
        }
        if (normalized === "browser") {
            return "Keep history until this browser tab closes?\n\nHistory will survive page reloads and WanGP restarts while this same tab or app webview remains open. Closing that browsing context clears it.";
        }
        return "Keep history until WanGP restarts?\n\nHistory will be associated with the currently running WanGP process and cleared automatically when Status Pro detects a new WanGP launch. It may survive closing and reopening the browser while WanGP keeps running.";
    }

    function loadHistoryRecordingPreference() {
        try {
            return window.localStorage.getItem(HISTORY_RECORDING_KEY) !== "0";
        } catch (_) {
            return true;
        }
    }

    function saveHistoryRecordingPreference(enabled) {
        try {
            window.localStorage.setItem(HISTORY_RECORDING_KEY, enabled ? "1" : "0");
        } catch (_) {
            // The selected state remains active for this page if storage is unavailable.
        }
    }

    function historyRecordingConfirmation(enabled) {
        return enabled
            ? "Start recording new runs?\n\nCompleted, aborted, and failed runs will be added to History using the selected retention lifetime."
            : "Stop recording new runs?\n\nLive Status Pro tracking will continue, but runs completed while this is off will not be added to History. Existing records remain available until cleared under their current retention setting.";
    }

    function setHistoryRecording(namespace, enabled) {
        const next = Boolean(enabled);
        if (next === (namespace.historyRecording !== false)) return;
        namespace.historyRecording = next;
        saveHistoryRecordingPreference(next);
        namespace.historyStorageNotice = next
            ? "Automatic history recording is on. Future completed, aborted, and failed runs will be recorded."
            : "Automatic history recording is off. Existing records are unchanged.";
        namespace.historyRenderKey = null;
    }

    function prepareRuntimeHistory(runtimeId) {
        const current = String(runtimeId || "").trim();
        if (!current) return false;
        try {
            const previous = String(window.localStorage.getItem(HISTORY_RUNTIME_ID_KEY) || "").trim();
            const changed = Boolean(previous && previous !== current);
            if (changed) window.localStorage.removeItem(RUNTIME_RUN_HISTORY_KEY);
            window.localStorage.setItem(HISTORY_RUNTIME_ID_KEY, current);
            return changed;
        } catch (_) {
            return false;
        }
    }

    function migrateHistoryValue(value, targetMode, runtimeId) {
        if (!value || value === "[]") return;
        const normalized = normalizeHistoryPersistence(targetMode);
        if (normalized === "runtime") prepareRuntimeHistory(runtimeId);
        let migrated = value;
        if (!modeStoresPrompts(normalized)) {
            try {
                const parsed = JSON.parse(value);
                if (Array.isArray(parsed)) {
                    migrated = JSON.stringify(parsed.map(run => stripRunPrompts(cloneJson(run, {}))));
                }
            } catch (_) {
                // Let the normal history loader reject malformed legacy data.
            }
        }
        runHistoryStorage(normalized).setItem(runHistoryKey(normalized), migrated);
    }

    function loadHistoryPersistence(runtimeId = "") {
        try {
            const stored = window.localStorage.getItem(HISTORY_PERSISTENCE_KEY);
            window.localStorage.removeItem("wangp.status-pro.capture-prompts.v1");
            if (["persistent", "browser", "runtime"].includes(stored)) {
                if (stored === "runtime") prepareRuntimeHistory(runtimeId);
                return stored;
            }

            // The former "session" setting was described as a WanGP session, but
            // technically used sessionStorage. Migrate it to the now-explicit
            // WanGP-runtime mode so existing users get the behaviour they expected.
            const legacySession = window.sessionStorage.getItem(SESSION_RUN_HISTORY_KEY);
            const legacyPersistent = window.localStorage.getItem(RUN_HISTORY_KEY);
            const legacy = legacySession && legacySession !== "[]" ? legacySession : legacyPersistent;
            migrateHistoryValue(legacy, "runtime", runtimeId);
            window.sessionStorage.removeItem(SESSION_RUN_HISTORY_KEY);
            window.localStorage.removeItem(RUN_HISTORY_KEY);
            window.localStorage.setItem(HISTORY_PERSISTENCE_KEY, "runtime");
        } catch (_) {
            // WanGP-runtime mode remains the in-memory default when storage is restricted.
        }
        return "runtime";
    }

    function saveHistoryPersistence(mode) {
        try {
            window.localStorage.setItem(HISTORY_PERSISTENCE_KEY, normalizeHistoryPersistence(mode));
        } catch (_) {
            // The selected mode remains active for this page even if its preference cannot be saved.
        }
    }

    function loadRunHistory(mode, runtimeId = "") {
        try {
            const normalized = normalizeHistoryPersistence(mode);
            if (normalized === "runtime") prepareRuntimeHistory(runtimeId);
            const storage = runHistoryStorage(normalized);
            const parsed = JSON.parse(storage.getItem(runHistoryKey(normalized)) || "[]");
            return Array.isArray(parsed) ? parsed.filter(run => run && typeof run === "object").slice(0, MAX_RUN_HISTORY) : [];
        } catch (_) {
            return [];
        }
    }

    function saveRunHistory(history, mode = "runtime") {
        const retained = history.slice(0, MAX_RUN_HISTORY);
        const requestedCount = retained.length;
        let storage = null;
        const key = runHistoryKey(mode);
        try {
            storage = runHistoryStorage(mode);
        } catch (_) {
            return { persisted: false, retained, dropped: 0 };
        }
        while (retained.length) {
            try {
                storage.setItem(key, JSON.stringify(retained));
                return {
                    persisted: true,
                    retained,
                    dropped: requestedCount - retained.length
                };
            } catch (_) {
                // Prefer retaining newer detailed runs when browser storage is full.
                retained.pop();
            }
        }
        try {
            storage.setItem(key, "[]");
            return { persisted: true, retained: [], dropped: requestedCount };
        } catch (_) {
            // Run history storage may be unavailable in private browsing.
            return { persisted: false, retained: history.slice(0, MAX_RUN_HISTORY), dropped: 0 };
        }
    }

    function clearRunHistoryStorage(mode) {
        try {
            const storage = runHistoryStorage(mode);
            storage.removeItem(runHistoryKey(mode));
        } catch (_) {
            // A blocked old store is harmless; the active mode remains authoritative.
        }
    }

    function persistRunHistory(namespace) {
        const before = namespace.runHistory.slice(0, MAX_RUN_HISTORY);
        const storedHistory = namespace.promptMemory !== false && modeStoresPrompts(namespace.historyPersistence)
            ? before
            : before.map(run => stripRunPrompts(cloneJson(run, {})));
        const result = saveRunHistory(storedHistory, namespace.historyPersistence);
        if (!result.persisted) {
            namespace.historyStorageNotice = namespace.historyPersistence === "browser"
                ? "Browser-tab session storage is unavailable. History will remain available only until this page is closed or reloaded."
                : "Browser local storage is unavailable. History will remain available only until this page is closed or reloaded.";
            return result;
        }
        namespace.runHistory = result.retained;
        if (result.dropped > 0) {
            const retainedIds = new Set(result.retained.map(run => String(run.id)));
            namespace.sessionRunIds = new Set(Array.from(namespace.sessionRunIds).filter(id => retainedIds.has(String(id))));
            namespace.selectedRunIds = new Set(Array.from(namespace.selectedRunIds).filter(id => retainedIds.has(String(id))));
            Array.from(namespace.recoverablePrompts.keys()).forEach(id => {
                if (!retainedIds.has(String(id))) namespace.recoverablePrompts.delete(id);
            });
            if (namespace.openHistoryRuns) {
                namespace.openHistoryRuns = new Set(Array.from(namespace.openHistoryRuns).filter(id => retainedIds.has(String(id))));
            }
            const storageLabel = namespace.historyPersistence === "browser"
                ? "browser-tab session storage"
                : namespace.historyPersistence === "runtime"
                    ? "WanGP-launch browser storage"
                    : "persistent browser storage";
            namespace.historyStorageNotice = `${result.dropped} older histor${result.dropped === 1 ? "y entry was" : "y entries were"} removed because ${storageLabel} is full.`;
        }
        return result;
    }

    function setHistoryPersistence(namespace, nextMode) {
        const next = normalizeHistoryPersistence(nextMode);
        const previous = namespace.historyPersistence;
        if (next === previous) return true;
        if (next === "runtime" && !String(namespace.runtimeId || "").trim()) {
            namespace.historyStorageNotice = "WanGP restart-aware history is not available until Status Pro connects to the running WanGP process.";
            return false;
        }

        const previousHistory = namespace.runHistory;
        let nextHistory = previousHistory.map(run => cloneJson(run, {}));
        if (namespace.promptMemory === false || !modeStoresPrompts(next)) {
            if (namespace.promptMemory !== false) nextHistory.forEach(run => cacheRunPrompts(namespace, run));
            nextHistory = nextHistory.map(run => stripRunPrompts(run));
        } else {
            nextHistory = nextHistory.map(run => runWithRecoverablePrompts(namespace, run));
        }

        if (next === "runtime") prepareRuntimeHistory(namespace.runtimeId);
        namespace.historyPersistence = next;
        namespace.runHistory = nextHistory;
        namespace.historyStorageNotice = "";
        const result = persistRunHistory(namespace);
        if (!result.persisted) {
            namespace.historyPersistence = previous;
            namespace.runHistory = previousHistory;
            namespace.historyStorageNotice = `Could not switch history to ${historyPersistenceLabel(next).toLowerCase()} storage in this browser.`;
            return false;
        }

        clearRunHistoryStorage(previous);
        saveHistoryPersistence(next);
        if (next === "browser") namespace.recoverablePrompts.clear();
        namespace.historyRenderKey = null;
        return true;
    }

    function setPromptMemory(namespace, enabled) {
        const next = Boolean(enabled);
        if (next === (namespace.promptMemory !== false)) return;
        namespace.promptMemory = next;
        savePromptMemoryPreference(next);
        if (!next) {
            namespace.recoverablePrompts.clear();
            namespace.runHistory = namespace.runHistory.map(run => stripRunPrompts(cloneJson(run, {})));
            if (namespace.exportFields instanceof Set) {
                namespace.exportFields.delete("prompt");
                namespace.exportFields.delete("negative_prompt");
                namespace.exportPreset = inferExportPreset(namespace, namespace.exportFields);
                saveExportSettings(namespace);
            }
            namespace.historyRenderKey = null;
            persistRunHistory(namespace);
        }
    }

    function loadCollapsedPreference() {
        try {
            return window.localStorage.getItem(COLLAPSED_KEY) === "1";
        } catch (_) {
            return false;
        }
    }

    function saveCollapsedPreference(collapsed) {
        try {
            window.localStorage.setItem(COLLAPSED_KEY, collapsed ? "1" : "0");
        } catch (_) {
            // Compact mode still works when local storage is unavailable.
        }
    }

    function loadPromptMemoryPreference() {
        try {
            const stored = window.localStorage.getItem(PROMPT_MEMORY_KEY);
            return stored === null ? true : stored === "1";
        } catch (_) {
            return true;
        }
    }

    function savePromptMemoryPreference(enabled) {
        try {
            window.localStorage.setItem(PROMPT_MEMORY_KEY, enabled ? "1" : "0");
        } catch (_) {
            // The in-page privacy choice remains active when preferences cannot be saved.
        }
    }

    function normalizeExportFormat(format) {
        return ["json", "csv", "md"].includes(String(format || "").toLowerCase())
            ? String(format).toLowerCase()
            : "json";
    }

    function loadExportSettings() {
        try {
            const parsed = JSON.parse(window.localStorage.getItem(EXPORT_SETTINGS_KEY) || "{}");
            const fields = Array.isArray(parsed.fields)
                ? parsed.fields.filter(field => EXPORT_FIELD_IDS.has(field))
                : [];
            return {
                fields: fields.length ? fields : Array.from(EXPORT_PRESETS.standard),
                preset: String(parsed.preset || "standard"),
                format: normalizeExportFormat(parsed.format)
            };
        } catch (_) {
            return { fields: Array.from(EXPORT_PRESETS.standard), preset: "standard", format: "json" };
        }
    }

    function saveExportSettings(namespace) {
        try {
            window.localStorage.setItem(EXPORT_SETTINGS_KEY, JSON.stringify({
                fields: Array.from(namespace.exportFields),
                preset: namespace.exportPreset,
                format: normalizeExportFormat(namespace.exportFormat)
            }));
        } catch (_) {
            // Export defaults remain active for this page when local storage is unavailable.
        }
    }

    function normalizeCustomExportPresets(value) {
        if (!Array.isArray(value)) return [];
        const seen = new Set();
        return value.map((preset, index) => {
            if (!preset || typeof preset !== "object") return null;
            const name = String(preset.name || "").trim().slice(0, 60);
            const fields = Array.isArray(preset.fields) ? preset.fields.filter(field => EXPORT_FIELD_IDS.has(field)) : [];
            let id = String(preset.id || `preset-${index + 1}`).replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 80);
            if (!name || !fields.length || !id || seen.has(id)) return null;
            seen.add(id);
            return { id, name, fields: Array.from(new Set(fields)) };
        }).filter(Boolean).slice(0, MAX_CUSTOM_EXPORT_PRESETS);
    }

    function saveCustomExportPresets(presets) {
        try {
            window.localStorage.setItem(EXPORT_PRESETS_KEY, JSON.stringify(normalizeCustomExportPresets(presets)));
        } catch (_) {
            // Named presets remain usable for the current page session.
        }
    }

    function loadCustomExportPresets() {
        try {
            const stored = normalizeCustomExportPresets(JSON.parse(window.localStorage.getItem(EXPORT_PRESETS_KEY) || "[]"));
            if (stored.length) return stored;
            const legacy = JSON.parse(window.localStorage.getItem(EXPORT_FIELDS_KEY) || "[]");
            const fields = Array.isArray(legacy) ? legacy.filter(field => EXPORT_FIELD_IDS.has(field)) : [];
            if (!fields.length) return [];
            const migrated = [{ id: "migrated-custom", name: "My preset", fields: Array.from(new Set(fields)) }];
            saveCustomExportPresets(migrated);
            return migrated;
        } catch (_) {
            return [];
        }
    }

    function recordDuration(state, id, seconds) {
        if (!Number.isFinite(seconds) || seconds < 0.5 || seconds > 86400) return;
        const values = Array.isArray(state.history[id]) ? state.history[id] : [];
        values.push(Math.round(seconds * 10) / 10);
        state.history[id] = values.slice(-20);
        saveHistory(state.history);
    }

    function createStageRecords() {
        return Object.fromEntries(STAGE_DEFS.map(def => [def.id, {
            id: def.id,
            label: def.label,
            visible: !def.optional,
            state: "pending",
            preloaded: false,
            unreported: false,
            activity: null,
            activityModel: "",
            rawName: "",
            rawMessage: "",
            startedAt: null,
            elapsed: null,
            elapsedBase: 0,
            reportedElapsed: null,
            reportedAt: null,
            nativeEta: null,
            progress: null,
            eta: null,
            samples: [],
            stepCurrent: null,
            stepTotal: null,
            lastStepCurrent: null,
            lastStepElapsed: null,
            lastStepAt: null,
            stepSamples: [],
            stepSeconds: null
        }]));
    }

    function freshState() {
        return {
            records: createStageRecords(),
            phases: {},
            phaseOrder: [],
            currentPhaseId: null,
            activeDenoisePhase: null,
            lastDenoisePhase: null,
            currentId: null,
            selectedId: null,
            selectionIsManual: false,
            history: loadHistory(),
            overallElapsed: null,
            steps: { current: null, total: null },
            lastSeenAt: 0,
            inactiveSince: 0,
            jobStartedAt: Date.now()
        };
    }

    function resetJob(namespace) {
        const history = namespace.state.history;
        namespace.state = freshState();
        namespace.state.history = history;
    }

    function cloneJson(value, fallback = {}) {
        try {
            return JSON.parse(JSON.stringify(value));
        } catch (_) {
            return fallback;
        }
    }

    function promptFields(settings) {
        const fields = {};
        ["prompt", "negative_prompt"].forEach(key => {
            if (settings && settings[key] !== null && settings[key] !== undefined && settings[key] !== "") {
                fields[key] = settings[key];
            }
        });
        return fields;
    }

    function cacheRunPrompts(namespace, run) {
        const settings = promptFields(run.settings);
        const outputRecords = (run.output_records || []).map(record => promptFields(record && record.settings));
        if (!Object.keys(settings).length && !outputRecords.some(record => Object.keys(record).length)) return;
        namespace.recoverablePrompts.set(String(run.id), { settings, outputRecords });
    }

    function stripRunPrompts(run) {
        run.settings = run.settings && typeof run.settings === "object" ? run.settings : {};
        delete run.settings.prompt;
        delete run.settings.negative_prompt;
        (run.output_records || []).forEach(record => {
            if (!record || !record.settings || typeof record.settings !== "object") return;
            delete record.settings.prompt;
            delete record.settings.negative_prompt;
        });
        return run;
    }

    function runtimeIdFromTelemetry(telemetry) {
        return String(telemetry && telemetry.runtime_id || "").trim();
    }

    function parseRunBridge(container) {
        const field = container && container.querySelector(
            "#status-pro-run-bridge textarea, #status-pro-run-bridge input"
        );
        const raw = String(field ? field.value : "").trim();
        if (!raw) return { raw: "", telemetry: null };
        try {
            const parsed = JSON.parse(raw);
            const telemetry = parsed && typeof parsed === "object" && Number.isFinite(Number(parsed.server_time))
                ? parsed
                : null;
            return { raw, telemetry };
        } catch (_) {
            return { raw, telemetry: null };
        }
    }

    function clearRuntimeScopedHistory(namespace) {
        namespace.runHistory = [];
        namespace.sessionRunIds.clear();
        namespace.selectedRunIds.clear();
        namespace.recoverablePrompts.clear();
        namespace.openHistoryGroups.clear();
        namespace.openHistoryRuns.clear();
        namespace.visibleHistoryGroups.clear();
        namespace.historyRenderKey = null;
    }

    function syncRuntimeId(namespace, runtimeId) {
        const next = String(runtimeId || "").trim();
        if (!next || next === namespace.runtimeId) return false;
        const changed = prepareRuntimeHistory(next);
        namespace.runtimeId = next;
        if (namespace.historyPersistence === "runtime" && changed) {
            clearRuntimeScopedHistory(namespace);
            namespace.historyStorageNotice = "History from the previous WanGP launch was cleared automatically.";
            if (namespace.activeRun) render(namespace);
            else renderIdle(namespace);
        }
        return changed;
    }

    function readRunSnapshot(namespace) {
        const field = namespace.container.querySelector(
            "#status-pro-run-bridge textarea, #status-pro-run-bridge input"
        );
        const raw = String(field ? field.value : "").trim();
        if (!raw || raw === namespace.runRaw) return namespace.runTelemetry;
        namespace.runRaw = raw;
        try {
            const parsed = JSON.parse(raw);
            namespace.runTelemetry = parsed && typeof parsed === "object" && Number.isFinite(Number(parsed.server_time))
                ? parsed
                : null;
            if (namespace.runTelemetry) syncRuntimeId(namespace, runtimeIdFromTelemetry(namespace.runTelemetry));
        } catch (_) {
            namespace.runTelemetry = null;
        }
        return namespace.runTelemetry;
    }

    function runTaskKey(task) {
        return task && task.id !== null && task.id !== undefined ? String(task.id) : "";
    }

    function windowDetails(telemetry) {
        if (!telemetry) return { number: null, total: null };
        let number = optionalNumber(telemetry.window_no);
        let total = optionalNumber(telemetry.total_windows);
        const status = String(telemetry.status || "");
        const match = status.match(/sliding\s+window\s+(\d+)\s*\/\s*(\d+)/i);
        if (match) {
            number = number ?? Number(match[1]);
            total = total ?? Number(match[2]);
        }
        if (!telemetry.sliding_window && !match) return { number: null, total: null };
        return {
            number: Number.isFinite(number) && number > 0 ? Math.floor(number) : null,
            total: Number.isFinite(total) && total > 0 ? Math.floor(total) : null
        };
    }

    function isNextSlidingWindow(run, telemetry) {
        const next = windowDetails(telemetry);
        return Number.isFinite(run && run.window_no) && Number.isFinite(next.number) && next.number > run.window_no;
    }

    function windowPromptFor(task, windowNo) {
        const prompts = task && Array.isArray(task.window_prompts) ? task.window_prompts : [];
        if (Number.isFinite(windowNo) && prompts.length) {
            const prompt = prompts[Math.min(Math.max(0, Math.floor(windowNo) - 1), prompts.length - 1)];
            if (prompt !== null && prompt !== undefined && String(prompt)) return String(prompt);
        }
        return task && task.window_prompt ? String(task.window_prompt) : null;
    }

    function relevantQueueError(telemetry, run) {
        const errors = telemetry && telemetry.queue_errors;
        const clientId = String(run && run.client_id || "");
        return clientId && errors && typeof errors === "object" ? errors[clientId] : "";
    }

    function stageDurations(state) {
        const stages = {};
        const prepare = state.records && state.records.prepare;
        if (prepare && prepare.preloaded && prepare.state === "complete") {
            stages["prepare:preloaded"] = {
                label: prepare.label,
                duration_seconds: 0,
                status: "complete",
                stage: "prepare",
                preloaded: true
            };
        }
        const encode = state.records && state.records.encode;
        if (encode && encode.unreported && encode.state === "complete") {
            stages["encode:unreported"] = {
                label: encode.label,
                status: "unreported",
                stage: "encode",
                unreported: true
            };
        }
        state.phaseOrder.forEach(id => {
            const record = state.phases[id];
            if (!record || record.state === "pending" || !Number.isFinite(record.elapsed)) return;
            stages[id] = {
                label: record.label,
                duration_seconds: Math.round(record.elapsed * 10) / 10,
                status: record.state === "aborting" ? "aborted" : record.state,
                stage: record.stage
            };
        });
        return stages;
    }

    function finishPhase(state) {
        const record = state.phases[state.currentPhaseId];
        if (!record || record.state !== "current") return;
        record.elapsed = (Date.now() - record.startedAt) / 1000;
        record.state = "complete";
    }

    function resetStageForNextDenoisePhase(state, id) {
        const record = state.records[id];
        if (!record) return;
        const definition = STAGE_DEFS.find(def => def.id === id);
        record.visible = !definition || !definition.optional;
        record.state = "pending";
        record.preloaded = false;
        record.unreported = false;
        record.activity = null;
        record.activityModel = "";
        record.rawName = "";
        record.rawMessage = "";
        record.startedAt = null;
        record.elapsed = null;
        record.elapsedBase = 0;
        record.reportedElapsed = null;
        record.reportedAt = null;
        record.progress = null;
        record.eta = null;
        record.samples = [];
        record.stepCurrent = null;
        record.stepTotal = null;
        record.lastStepCurrent = null;
        record.lastStepElapsed = null;
        record.lastStepAt = null;
        record.stepSamples = [];
        record.stepSeconds = null;
    }

    function resetDownstreamStagesForNextDenoisePhase(state) {
        ["decode", "post", "save"].forEach(id => resetStageForNextDenoisePhase(state, id));
        if (!state.selectionIsManual) state.selectedId = "denoise";
    }

    function applyPhase(state, snapshot, now) {
        const phase = phaseInfo(snapshot.rawName);
        if (state.currentPhaseId !== phase.id) {
            finishPhase(state);
            const isLaterDenoisePhase = phase.kind === "denoise" &&
                Number.isFinite(phase.phaseNumber) &&
                Number.isFinite(state.lastDenoisePhase) &&
                phase.phaseNumber > state.lastDenoisePhase;
            if (isLaterDenoisePhase) resetDownstreamStagesForNextDenoisePhase(state);
            state.currentPhaseId = phase.id;
            state.phases[phase.id] = {
                id: phase.id,
                label: phase.label,
                stage: snapshot.id,
                startedAt: now,
                elapsed: null,
                state: snapshot.aborting ? "aborting" : "current"
            };
            state.phaseOrder.push(phase.id);
        }
        if (phase.kind === "denoise" && Number.isFinite(phase.phaseNumber)) {
            state.activeDenoisePhase = phase.phaseNumber;
            state.lastDenoisePhase = phase.phaseNumber;
        }
        const record = state.phases[state.currentPhaseId];
        record.state = snapshot.aborting ? "aborting" : "current";
        record.elapsed = (now - record.startedAt) / 1000;
    }

    function runStatusFrom(namespace, telemetry) {
        const field = namespace.source.querySelector("textarea, input");
        const message = `${String(telemetry && telemetry.status || "")} ${String(field && field.value || "")}`;
        if (/\b(abort(?:ed|ing)?|cancel(?:led|ling)?|interrupt(?:ed|ing)?)\b/i.test(message)) return "aborted";
        if (/\b(error|failed|failure|exception)\b/i.test(message)) return "failed";
        return "completed";
    }

    function observeRunOutcome(namespace, ...messages) {
        const run = namespace.activeRun;
        if (!run) return;
        const message = messages
            .filter(value => value !== null && value !== undefined && value !== "")
            .map(value => typeof value === "string" ? value : JSON.stringify(value))
            .join(" ")
            .replace(/\s+/g, " ")
            .trim();
        if (!message) return;
        if (run.notice_baseline && message === run.notice_baseline) return;

        const aborted = /\b(abort(?:ed|ing)?|cancel(?:led|ling)?|interrupt(?:ed|ing)?)\b/i.test(message);
        const failed = /\b(error|failed|failure|exception|traceback|out of memory|oom|insufficient|unsufficient|tried to allocate)\b/i.test(message);
        if (failed) {
            run.outcome_status = "failed";
            if (/\b(cuda out of memory|out of memory|oom|tried to allocate|insufficient vram|unsufficient vram)\b/i.test(message)) {
                run.status_reason = "Out of GPU memory (VRAM). Reduce resolution, frame count, or memory use.";
            } else if (/\b(insufficient ram|unsufficient ram|reserved ram)\b/i.test(message)) {
                run.status_reason = "Insufficient system or reserved RAM.";
            } else {
                run.status_reason = message.slice(0, 500);
            }
            return;
        }
        if (aborted && run.outcome_status !== "failed") {
            run.outcome_status = "aborted";
            run.status_reason = "Cancelled before completion.";
        }
    }

    function visibleFailureNotice(namespace) {
        if (!namespace.root || !namespace.root.querySelectorAll) return "";
        const notices = Array.from(namespace.root.querySelectorAll('[role="alert"], .toast, .error'));
        for (let index = notices.length - 1; index >= Math.max(0, notices.length - 12); index -= 1) {
            const message = String(notices[index].textContent || "").trim();
            if (/\b(error|failed|failure|out of memory|oom|insufficient|unsufficient|abort|cancel)\b/i.test(message)) {
                return message;
            }
        }
        return "";
    }

    function currentOutputPaths(telemetry) {
        return [
            ...(Array.isArray(telemetry && telemetry.video_outputs) ? telemetry.video_outputs : []),
            ...(Array.isArray(telemetry && telemetry.audio_outputs) ? telemetry.audio_outputs : [])
        ].map(value => String(value));
    }

    function mediaTypeFromPath(path) {
        const clean = String(path || "").split(/[?#]/)[0].toLowerCase();
        if (/\.(?:jpe?g|png|webp|bmp|gif|tiff?)$/.test(clean)) return "image";
        if (/\.(?:wav|mp3|aac|flac|m4a|ogg|opus|wma)$/.test(clean)) return "audio";
        if (/\.(?:mp4|mkv|mov|webm|avi|ogv)$/.test(clean)) return "video";
        return "unknown";
    }

    function currentOutputRecords(telemetry) {
        if (Array.isArray(telemetry && telemetry.output_records)) {
            return cloneJson(telemetry.output_records, []);
        }
        return currentOutputPaths(telemetry).map(path => ({
            path,
            media_type: mediaTypeFromPath(path),
            settings: {}
        }));
    }

    function outputWindowNumber(record) {
        return optionalNumber(record && record.settings && record.settings.window_no);
    }

    function outputBoundaryTime(run, record, fallback) {
        const settings = record && record.settings || {};
        let timestamp = optionalNumber(settings.creation_timestamp);
        if (Number.isFinite(timestamp)) timestamp *= timestamp < 100000000000 ? 1000 : 1;
        else timestamp = Date.parse(settings.creation_date || "");
        if (!Number.isFinite(timestamp) || timestamp < run.started_at || timestamp > fallback) return fallback;
        return timestamp;
    }

    function nextSlidingOutputBoundary(run, telemetry) {
        if (!Number.isFinite(run && run.window_no) || !Number.isFinite(run && run.total_windows) || run.window_no >= run.total_windows) {
            return null;
        }
        const outputRecords = currentOutputRecords(telemetry);
        const baseline = Math.min(Math.max(0, Number(run.output_baseline) || 0), outputRecords.length);
        for (let index = baseline + 1; index < outputRecords.length; index += 1) {
            const outputWindow = outputWindowNumber(outputRecords[index]);
            if (Number.isFinite(outputWindow) && outputWindow > run.window_no) {
                return { outputEnd: index, nextWindowNo: outputWindow, record: outputRecords[index - 1] };
            }
        }
        if (outputRecords.length - baseline >= 2) {
            return { outputEnd: baseline + 1, nextWindowNo: run.window_no + 1, record: outputRecords[baseline] };
        }
        return null;
    }

    function repairInheritedPassTotals(run) {
        const steps = Array.isArray(run && run.step_performance) ? run.step_performance : [];
        const completed = run && (run.status === "completed" || run.status === "window");
        const truncated = Boolean(run && (run.step_performance_source_truncated || (run.step_summary && run.step_summary.truncated)));
        if (!completed || truncated || !steps.length) return false;
        const groups = new Map();
        steps.forEach(step => {
            const passNo = optionalNumber(step && step.pass_no);
            if (!(passNo > 1)) return;
            const observerId = String(step && step.observer_id || "observer");
            const key = `${observerId}:pass:${passNo}`;
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push(step);
        });
        let repaired = false;
        groups.forEach(samples => {
            const stepNumbers = samples.map(step => optionalNumber(step && step.step));
            const totals = samples.map(step => optionalNumber(step && step.total_steps));
            if (!stepNumbers.every(Number.isFinite) || !totals.every(Number.isFinite)) return;
            const uniqueSteps = Array.from(new Set(stepNumbers)).sort((left, right) => left - right);
            const uniqueTotals = Array.from(new Set(totals));
            if (uniqueSteps.length !== samples.length || uniqueTotals.length !== 1) return;
            if (!uniqueSteps.every((value, index) => value === index + 1)) return;
            const inheritedTotal = uniqueTotals[0];
            if (!(inheritedTotal > uniqueSteps.length)) return;
            samples.forEach(step => { step.total_steps = uniqueSteps.length; });
            repaired = true;
        });
        return repaired;
    }

    function normalizeRunMedia(run) {
        const declaredMediaType = String(run && run.media_type || "").toLowerCase();
        const declaredFrameCount = optionalNumber(run && run.frame_count);
        const declaredOutputCount = optionalNumber(run && run.output_count);
        run.settings = run.settings && typeof run.settings === "object" ? run.settings : {};
        run.step_performance = Array.isArray(run.step_performance) ? run.step_performance : [];
        run.resources = run.resources && typeof run.resources === "object" ? run.resources : null;
        run.step_summary = run.step_summary && typeof run.step_summary === "object" ? run.step_summary : null;
        const repairedStepTotals = repairInheritedPassTotals(run);
        if (run.step_performance.length && (repairedStepTotals || !run.step_summary || !Array.isArray(run.step_summary.passes))) {
            finalizePerformance(run);
        }
        run.outputs = Array.isArray(run.outputs) ? run.outputs.map(value => String(value)) : [];
        run.output_records = Array.isArray(run.output_records)
            ? run.output_records
            : run.outputs.map(path => ({ path, media_type: mediaTypeFromPath(path), settings: {} }));
        run.output_records.forEach(record => {
            record.path = String(record.path || "");
            record.media_type = record.media_type && record.media_type !== "unknown"
                ? String(record.media_type)
                : mediaTypeFromPath(record.path);
            record.settings = record.settings && typeof record.settings === "object" ? record.settings : {};
        });
        const mediaTypes = Array.from(new Set(run.output_records.map(record => record.media_type).filter(type => type && type !== "unknown")));
        if (!mediaTypes.length && ["image", "video", "audio"].includes(declaredMediaType)) mediaTypes.push(declaredMediaType);
        if (!mediaTypes.length && Number(run.settings.image_mode) > 0) mediaTypes.push("image");
        const primaryType = mediaTypes.includes("video")
            ? "video"
            : (mediaTypes.includes("image") ? "image" : (mediaTypes.includes("audio") ? "audio" : "unknown"));
        run.media_type = primaryType;
        run.media_types = mediaTypes;
        run.output_count = run.output_records.length || run.outputs.length || declaredOutputCount || 0;
        if (primaryType === "image") run.frame_count = 1;
        else if (primaryType === "video") {
            run.frame_count = optionalNumber(setting(run.settings, "num_frames", "frame_num", "video_length")) || declaredFrameCount;
        } else run.frame_count = null;
        if (primaryType === "image" || primaryType === "audio") {
            delete run.settings.video_length;
            delete run.settings.num_frames;
            delete run.settings.frame_num;
            run.output_records.forEach(record => {
                delete record.settings.video_length;
                delete record.settings.num_frames;
                delete record.settings.frame_num;
            });
        }
        const abortedStage = Object.values(run.stages || {}).some(stage => stage && stage.status === "aborted");
        if (run.status === "completed" && abortedStage) {
            run.status = "aborted";
            run.status_reason = run.status_reason || "Cancelled before completion.";
        } else if (run.status === "completed" && run.completed_at && run.output_count === 0 && !run.imported) {
            run.status = "failed";
            run.status_reason = run.status_reason || "Generation ended without producing an output.";
            run.failure_reason = run.failure_reason || run.status_reason;
        }
        return run;
    }

    function startRun(namespace, task, telemetry, options = {}) {
        resetJob(namespace);
        const observedNow = Number(telemetry && telemetry.server_time) * 1000 || Date.now();
        const now = Number.isFinite(options.startedAt) ? options.startedAt : observedNow;
        const settings = cloneJson(task && task.settings, {});
        const observedWindow = windowDetails(telemetry);
        const window = {
            number: Number.isFinite(options.windowNo) ? options.windowNo : observedWindow.number,
            total: Number.isFinite(options.totalWindows) ? options.totalWindows : observedWindow.total
        };
        const windowPrompt = windowPromptFor(task, window.number);
        if (windowPrompt) settings.prompt = windowPrompt;
        namespace.activeRun = {
            id: `${namespace.sessionId}-${runTaskKey(task) || "observed"}-${Math.round(now)}`,
            session_id: namespace.sessionId,
            queue_task_id: task && task.id !== undefined ? task.id : null,
            client_id: String(task && task.client_id || ""),
            status: "running",
            started_at: now,
            completed_at: null,
            duration_seconds: null,
            settings,
            stages: {},
            step_performance: [],
            resources: null,
            step_summary: null,
            _performance_step_keys: {},
            outputs: [],
            output_baseline: Number.isFinite(options.outputBaseline)
                ? options.outputBaseline
                : currentOutputRecords(telemetry).length,
            repeats: optionalNumber(task && task.repeats) || 1,
            window_no: window.number,
            total_windows: window.total,
            window_prompt: windowPrompt,
            window_prompts: cloneJson(task && task.window_prompts, []),
            outcome_status: null,
            status_reason: null,
            notice_baseline: visibleFailureNotice(namespace)
        };
        observePerformanceTelemetry(namespace.activeRun, telemetry);
    }

    function updateActiveRun(namespace, task, telemetry) {
        if (!namespace.activeRun || !task) return;
        namespace.activeRun.settings = {
            ...namespace.activeRun.settings,
            ...cloneJson(task.settings, {})
        };
        const windowPrompt = windowPromptFor(task, namespace.activeRun.window_no);
        if (windowPrompt) {
            namespace.activeRun.window_prompt = windowPrompt;
            namespace.activeRun.settings.prompt = namespace.activeRun.window_prompt;
        }
        namespace.activeRun.repeats = optionalNumber(task.repeats) || namespace.activeRun.repeats || 1;
        const window = windowDetails(telemetry);
        if (!Number.isFinite(namespace.activeRun.window_no) && Number.isFinite(window.number)) {
            namespace.activeRun.window_no = window.number;
        }
        if (Number.isFinite(window.total)) namespace.activeRun.total_windows = window.total;
        observePerformanceTelemetry(namespace.activeRun, telemetry);
    }

    function finishRun(namespace, status, completedAt, telemetry, outputEnd) {
        const run = namespace.activeRun;
        if (!run) return;
        finishStage(namespace.state, namespace.state.currentId);
        finishPhase(namespace.state);
        const ended = Number.isFinite(completedAt) ? completedAt : Date.now();
        run.completed_at = ended;
        run.duration_seconds = Math.max(0, Math.round((ended - run.started_at) / 100) / 10);
        run.stages = stageDurations(namespace.state);
        observePerformanceTelemetry(run, telemetry || namespace.runTelemetry);
        finalizePerformance(run);
        const outputRecords = currentOutputRecords(telemetry || namespace.runTelemetry);
        const outputStart = Math.min(run.output_baseline || 0, outputRecords.length);
        const outputLimit = Number.isFinite(outputEnd)
            ? Math.min(Math.max(outputStart, Math.floor(outputEnd)), outputRecords.length)
            : outputRecords.length;
        run.output_records = outputRecords.slice(outputStart, outputLimit);
        run.outputs = run.output_records.map(record => String(record.path || "")).filter(Boolean);
        if (run.output_records.length && run.output_records[0].settings) {
            run.settings = {
                ...run.settings,
                ...cloneJson(run.output_records[0].settings, {})
            };
        }
        if (run.window_prompt) {
            run.settings.prompt = run.window_prompt;
            run.output_records.forEach(record => {
                record.settings = { ...(record.settings || {}), prompt: run.window_prompt };
            });
        }
        delete run.output_baseline;
        normalizeRunMedia(run);
        run.status = status === "window" ? "window" : (run.outcome_status || status || "completed");
        if (run.status === "completed" && run.outputs.length === 0) {
            run.status = "failed";
            run.status_reason = run.status_reason || "Generation ended without producing an output.";
        }
        if (run.status === "failed") run.failure_reason = run.status_reason || "Generation failed.";
        delete run.outcome_status;
        delete run.notice_baseline;
        delete run.client_id;
        delete run.window_prompt;
        delete run.window_prompts;
        if (namespace.historyRecording === false) {
            namespace.lastCompletedAt = ended;
            namespace.activeRun = null;
            resetJob(namespace);
            return;
        }
        if (run.settings && (namespace.promptMemory === false || !modeStoresPrompts(namespace.historyPersistence))) {
            if (namespace.promptMemory !== false) cacheRunPrompts(namespace, run);
            stripRunPrompts(run);
        }
        namespace.runHistory.unshift(cloneJson(run, {}));
        namespace.runHistory = namespace.runHistory.slice(0, MAX_RUN_HISTORY);
        namespace.sessionRunIds.add(run.id);
        namespace.lastCompletedAt = ended;
        persistRunHistory(namespace);
        namespace.activeRun = null;
        resetJob(namespace);
    }

    function taskFromRun(run) {
        return {
            id: run.queue_task_id,
            client_id: run.client_id,
            repeats: run.repeats,
            settings: cloneJson(run.settings, {}),
            window_prompts: cloneJson(run.window_prompts, [])
        };
    }

    function splitMissedSlidingWindows(namespace, task, telemetry, now) {
        let boundary = nextSlidingOutputBoundary(namespace.activeRun, telemetry);
        while (namespace.activeRun && boundary) {
            const run = namespace.activeRun;
            const continuationTask = task || taskFromRun(run);
            const boundaryTime = outputBoundaryTime(run, boundary.record, now);
            finishRun(namespace, "window", boundaryTime, telemetry, boundary.outputEnd);
            startRun(namespace, continuationTask, telemetry, {
                startedAt: boundaryTime,
                outputBaseline: boundary.outputEnd,
                windowNo: boundary.nextWindowNo,
                totalWindows: run.total_windows
            });
            boundary = nextSlidingOutputBoundary(namespace.activeRun, telemetry);
        }
    }

    function syncRunTelemetry(namespace) {
        const telemetry = readRunSnapshot(namespace);
        if (!telemetry) return;
        const task = telemetry.active_task && typeof telemetry.active_task === "object"
            ? telemetry.active_task
            : null;
        const nextKey = runTaskKey(task);
        const activeKey = namespace.activeRun && namespace.activeRun.queue_task_id !== null
            ? String(namespace.activeRun.queue_task_id)
            : "";
        const now = Number(telemetry.server_time) * 1000 || Date.now();
        if (namespace.activeRun) observePerformanceTelemetry(namespace.activeRun, telemetry);

        if (task) {
            splitMissedSlidingWindows(namespace, task, telemetry, now);
            if (namespace.activeRun && isNextSlidingWindow(namespace.activeRun, telemetry)) {
                finishRun(namespace, "window", now, telemetry);
            }
            if (namespace.activeRun && activeKey && activeKey !== nextKey) {
                finishRun(namespace, "completed", now, telemetry);
            }
            if (!namespace.activeRun) startRun(namespace, task, telemetry);
            updateActiveRun(namespace, task, telemetry);
            observeRunOutcome(namespace, telemetry.status, relevantQueueError(telemetry, namespace.activeRun));
            return;
        }

        if (namespace.activeRun) {
            splitMissedSlidingWindows(namespace, null, telemetry, now);
            observeRunOutcome(namespace, telemetry.status, relevantQueueError(telemetry, namespace.activeRun));
            finishRun(namespace, runStatusFrom(namespace, telemetry), now, telemetry);
        }
    }

    function findTracker(namespace) {
        if (namespace.tracker && namespace.tracker.isConnected && namespace.tracker.querySelector(".progress-level-inner")) {
            return namespace.tracker;
        }
        const wrappers = Array.from(namespace.source.querySelectorAll(".wrap.default, .wrap"));
        namespace.tracker = wrappers.find(wrapper =>
            wrapper.querySelector(".progress-level-inner") && wrapper.querySelector(".progress-text")
        ) || null;
        return namespace.tracker;
    }

    function statusSnapshot(namespace, message) {
        message = String(message || "").trim();
        if (!message) return null;
        const aborting = /\b(abort(?:ing|ed)?|cancel(?:ling|ed)?|interrupt(?:ing|ed)?)\b/i.test(message);
        const loadWord = "(?:load(?:ing|ed)?|download(?:ing|ed)?)";
        const assetWord = "(?:models?|weights?|files?|assets?)";
        const modelActivity = new RegExp(
            `\\b${loadWord}\\b.*\\b${assetWord}\\b|\\b${assetWord}\\b.*\\b${loadWord}\\b`,
            "i"
        ).test(message);
        if (!aborting && !modelActivity) return null;
        const loadComplete = /\b(?:loaded|downloaded)\b/i.test(message);
        const id = aborting && namespace.state.currentId
            ? namespace.state.currentId
            : stageIdFor(message);
        return {
            id,
            rawName: aborting ? "Aborting" : (loadComplete ? "Model loaded" : "Loading model"),
            rawMessage: message,
            metaText: "",
            stageElapsed: null,
            overallElapsed: namespace.state.overallElapsed,
            progress: loadComplete ? 100 : null,
            steps: namespace.state.steps,
            aborting,
            activity: loadComplete ? "load-complete" : "load",
            activityModel: "",
            textOnly: true
        };
    }

    function modelLifecycleSnapshot(namespace) {
        const telemetry = namespace.runTelemetry;
        const lifecycle = telemetry && telemetry.model_lifecycle;
        if (!lifecycle || typeof lifecycle !== "object") return null;
        const lifecycleState = String(lifecycle.state || "");
        if (!/^(?:unloading|unloaded|failed)$/.test(lifecycleState)) return null;
        const serverTime = optionalNumber(telemetry.server_time);
        const startedAt = optionalNumber(lifecycle.started_at);
        const completedAt = optionalNumber(lifecycle.completed_at);
        const stageElapsed = Number.isFinite(startedAt) && Number.isFinite(serverTime)
            ? Math.max(0, (Number.isFinite(completedAt) ? completedAt : serverTime) - startedAt)
            : null;
        const modelName = String(lifecycle.model_name || lifecycle.model_type || "Previously loaded model");
        const failed = lifecycleState === "failed";
        const complete = lifecycleState === "unloaded";
        return {
            id: "prepare",
            rawName: failed ? "Model unload failed" : (complete ? "Model unloaded" : `Unloading ${modelName}`),
            rawMessage: failed
                ? (String(lifecycle.error || "The previous model could not be unloaded."))
                : (complete
                    ? `${modelName} was released from RAM and VRAM.`
                    : `Releasing ${modelName} from RAM and VRAM before the next model loads.`),
            metaText: "",
            stageElapsed,
            overallElapsed: namespace.state.overallElapsed,
            progress: null,
            steps: namespace.state.steps,
            aborting: false,
            activity: failed ? "unload-failed" : (complete ? "unload-complete" : "unload"),
            activityModel: modelName,
            textOnly: true
        };
    }

    function readStatusField(namespace) {
        const field = namespace.source.querySelector("textarea, input");
        return statusSnapshot(namespace, field ? field.value : "");
    }

    function readPrepareStatus(namespace) {
        const telemetry = namespace.runTelemetry;
        const lifecycleSnapshot = modelLifecycleSnapshot(namespace);
        if (lifecycleSnapshot && lifecycleSnapshot.activity === "unload") return lifecycleSnapshot;
        const phase = Array.isArray(telemetry && telemetry.progress_phase)
            ? String(telemetry.progress_phase[0] || "")
            : "";
        if (phase && stageIdFor(phase) !== "prepare") {
            return null;
        }
        const nativeSnapshot = statusSnapshot(namespace, telemetry && telemetry.status);
        if (nativeSnapshot) return nativeSnapshot;
        return lifecycleSnapshot;
    }

    function readReportedPreGenerationStatus(namespace) {
        if (!namespace.activeRun) return null;
        const telemetry = namespace.runTelemetry;
        const reported = telemetry && telemetry.progress_phase;
        const phase = Array.isArray(reported) ? String(reported[0] || "").trim() : "";
        const id = stageIdFor(phase);
        if (!phase || (id !== "input" && id !== "encode")) return null;
        const currentId = namespace.state && namespace.state.currentId;
        if (id === "encode" && currentId && !["prepare", "input", "encode"].includes(currentId)) return null;
        if (id === "input" && currentId && ["decode", "post", "save"].includes(currentId)) return null;
        return {
            id,
            rawName: phase,
            rawMessage: phase,
            metaText: "",
            stageElapsed: null,
            nativeEta: null,
            overallElapsed: namespace.state.overallElapsed,
            progress: null,
            steps: { current: null, total: null },
            aborting: false,
            textOnly: true
        };
    }

    function readSnapshot(namespace) {
        const tracker = findTracker(namespace);
        if (!tracker) return readStatusField(namespace);
        const level = tracker.querySelector(".progress-level-inner");
        const meta = tracker.querySelector(".progress-text");
        if (!level || !meta) return readStatusField(namespace);
        const levelText = String(level.textContent || "").trim();
        const metaText = String(meta.textContent || "").trim();
        if (!levelText && !metaText) return readStatusField(namespace);
        const rawName = stageNameFrom(levelText);
        const timing = parseProgressTiming(levelText);
        const stageElapsed = timing.elapsed;
        const overallElapsed = parseDuration(metaText.split("|").slice(-1)[0]);
        return {
            id: stageIdFor(rawName),
            rawName,
            rawMessage: levelText,
            metaText,
            stageElapsed,
            nativeEta: Number.isFinite(timing.elapsed) && Number.isFinite(timing.total) && timing.total >= timing.elapsed
                ? timing.total - timing.elapsed
                : null,
            overallElapsed,
            progress: parsePercent(levelText, tracker.querySelector(".progress-bar")),
            steps: parseSteps(metaText),
            aborting: false,
            textOnly: false
        };
    }

    function reinterpretQwenSilentEncode(namespace, snapshot) {
        if (!snapshot || snapshot.id !== "denoise") return snapshot;
        const telemetry = namespace.runTelemetry;
        const task = telemetry && telemetry.active_task;
        const settings = task && task.settings;
        const modelType = String(settings && (settings.base_model_type || settings.model_type) || "").toLowerCase();
        if (!modelType.startsWith("qwen_image")) return snapshot;

        const performance = telemetry && telemetry.performance;
        const hasPerformance = performance && typeof performance === "object";
        if (!hasPerformance) {
            if (!namespace.qwenEncodeFallbackStartedAt) namespace.qwenEncodeFallbackStartedAt = Date.now();
            if (Date.now() - namespace.qwenEncodeFallbackStartedAt > 3000) return snapshot;
        } else {
            namespace.qwenEncodeFallbackStartedAt = 0;
        }
        const callbackPhase = optionalNumber(performance && performance.callback_phase);
        const observedSteps = Array.isArray(performance && performance.steps) ? performance.steps : [];
        if ((Number.isFinite(callbackPhase) && callbackPhase >= 2) || observedSteps.length > 0) return snapshot;

        const serverTime = optionalNumber(telemetry && telemetry.server_time);
        const phaseStartedAt = optionalNumber(performance && performance.phase_started_at);
        const encodingElapsed = Number.isFinite(serverTime) && Number.isFinite(phaseStartedAt)
            ? Math.max(0, serverTime - phaseStartedAt)
            : null;
        return {
            ...snapshot,
            id: "encode",
            rawName: "Encoding prompt and references",
            rawMessage: "Qwen is encoding prompt and reference-image conditioning.",
            stageElapsed: encodingElapsed,
            nativeEta: null,
            progress: null,
            steps: { current: null, total: null },
            textOnly: true
        };
    }

    function finishStage(state, id) {
        if (!id) return;
        const record = state.records[id];
        if (!record || record.state !== "current") return;
        if (record.startedAt) {
            record.elapsed = (Number.isFinite(record.elapsedBase) ? record.elapsedBase : 0) +
                (Date.now() - record.startedAt) / 1000;
        }
        record.state = "complete";
        record.progress = 100;
        record.eta = 0;
        if (id === "denoise" && Number.isFinite(record.stepTotal)) {
            record.stepCurrent = record.stepTotal;
        }
        recordDuration(state, id, record.elapsed);
    }

    function markPreparePreloaded(state) {
        const record = state.records.prepare;
        if (!record || record.state !== "pending") return;
        record.visible = true;
        record.state = "complete";
        record.preloaded = true;
        record.unreported = false;
        record.rawName = "Model preloaded";
        record.rawMessage = "The required model was already loaded and ready for this run.";
        record.startedAt = null;
        record.elapsed = 0;
        record.elapsedBase = 0;
        record.reportedElapsed = 0;
        record.progress = 100;
        record.eta = 0;
    }

    function markEncodeUnreported(state) {
        const record = state.records.encode;
        if (!record || record.state !== "pending") return;
        record.visible = true;
        record.state = "complete";
        record.preloaded = false;
        record.unreported = true;
        record.rawName = "Encode status not reported";
        record.rawMessage = "Wan2GP did not expose a separately measurable Encode phase for this run. Any required prompt, reference, or input conditioning completed before generation began.";
        record.startedAt = null;
        record.elapsed = null;
        record.reportedElapsed = null;
        record.reportedAt = null;
        record.progress = 100;
        record.eta = 0;
    }

    function updateEta(record) {
        if (!Number.isFinite(record.progress) || record.progress <= 0 || record.progress >= 100 || !Number.isFinite(record.elapsed) || record.elapsed <= 0) {
            record.eta = record.progress >= 100 ? 0 : null;
            return;
        }
        const overallEstimate = record.elapsed * (100 - record.progress) / record.progress;
        let estimate = overallEstimate;
        const useful = record.samples.filter(sample => sample.progress <= record.progress);
        if (useful.length >= 2) {
            const latest = useful[useful.length - 1];
            let earlier = useful[0];
            for (let index = useful.length - 2; index >= 0; index -= 1) {
                if (latest.elapsed - useful[index].elapsed >= 2 && latest.progress - useful[index].progress >= 0.2) {
                    earlier = useful[index];
                    break;
                }
            }
            const elapsedDelta = latest.elapsed - earlier.elapsed;
            const progressDelta = latest.progress - earlier.progress;
            if (elapsedDelta > 0 && progressDelta > 0) {
                const recentEstimate = (100 - record.progress) / (progressDelta / elapsedDelta);
                estimate = recentEstimate * 0.65 + overallEstimate * 0.35;
            }
        }
        estimate = clamp(estimate, 0, 86400);
        record.eta = Number.isFinite(record.eta) ? record.eta * 0.72 + estimate * 0.28 : estimate;
    }

    function updateDenoiseEta(record) {
        if (!Number.isFinite(record.stepCurrent) || !Number.isFinite(record.stepTotal) ||
            !Number.isFinite(record.elapsed) || record.stepCurrent <= 0 || record.elapsed <= 0) {
            record.eta = null;
            return;
        }
        if (record.stepCurrent >= record.stepTotal) {
            record.eta = 0;
            return;
        }
        const fallbackStepSeconds = record.elapsed / record.stepCurrent;
        const fallbackEta = fallbackStepSeconds * (record.stepTotal - record.stepCurrent);
        const nativeEta = optionalNumber(record.nativeEta);
        record.eta = clamp(Number.isFinite(nativeEta) ? nativeEta : fallbackEta, 0, 86400);
        record.stepSeconds = (record.elapsed + record.eta) / record.stepTotal;
    }

    function updateStepTiming(record, steps, now = Date.now()) {
        if (record.id !== "denoise" && record.id !== "post") return;
        const current = optionalNumber(steps && steps.current);
        const total = optionalNumber(steps && steps.total);
        record.stepCurrent = current;
        record.stepTotal = total;
        if (!Number.isFinite(current) || !Number.isFinite(record.elapsed)) return;

        if (!Number.isFinite(record.lastStepCurrent) || !Number.isFinite(record.lastStepElapsed)) {
            record.lastStepCurrent = current;
            record.lastStepElapsed = record.elapsed;
            record.lastStepAt = now;
            record.stepSeconds = null;
            return;
        }

        if (current < record.lastStepCurrent) {
            record.lastStepCurrent = current;
            record.lastStepElapsed = record.elapsed;
            record.lastStepAt = now;
            record.stepSamples = [];
            record.stepSeconds = null;
            return;
        }

        if (current === record.lastStepCurrent) return;
        const elapsedDelta = Number.isFinite(record.lastStepAt)
            ? (now - record.lastStepAt) / 1000
            : record.elapsed - record.lastStepElapsed;
        const stepDelta = current - record.lastStepCurrent;
        record.lastStepCurrent = current;
        record.lastStepElapsed = record.elapsed;
        record.lastStepAt = now;
        if (elapsedDelta <= 0 || stepDelta <= 0) return;

        record.stepSamples.push(elapsedDelta / stepDelta);
        record.stepSamples = record.stepSamples.slice(-6);
        const sorted = record.stepSamples.slice().sort((left, right) => left - right);
        const middle = Math.floor(sorted.length / 2);
        record.stepSeconds = sorted.length % 2
            ? sorted[middle]
            : (sorted[middle - 1] + sorted[middle]) / 2;
    }

    function applySnapshot(namespace, snapshot) {
        const state = namespace.state;
        const now = Date.now();
        if (state.inactiveSince && now - state.inactiveSince > RESET_AFTER_MS) {
            finishStage(state, state.currentId);
            resetJob(namespace);
        }
        const activeState = namespace.state;
        activeState.inactiveSince = 0;
        activeState.lastSeenAt = now;
        if (namespace.activeRun && activeState.currentId === null &&
            (snapshot.id === "input" || snapshot.id === "encode" || snapshot.id === "denoise")) {
            markPreparePreloaded(activeState);
        }
        if (namespace.activeRun && snapshot.id === "denoise") markEncodeUnreported(activeState);
        applyPhase(activeState, snapshot, now);

        if (activeState.currentId !== snapshot.id) {
            finishStage(activeState, activeState.currentId);
            activeState.currentId = snapshot.id;
            const next = activeState.records[snapshot.id];
            const elapsedBase = snapshot.id === "input" && next.state === "complete" && Number.isFinite(next.elapsed)
                ? next.elapsed
                : 0;
            next.state = snapshot.aborting ? "aborting" : "current";
            next.preloaded = false;
            next.unreported = false;
            next.activity = null;
            next.activityModel = "";
            next.visible = true;
            next.startedAt = now;
            next.elapsed = elapsedBase || null;
            next.elapsedBase = elapsedBase;
            next.reportedElapsed = null;
            next.reportedAt = null;
            next.nativeEta = null;
            next.progress = null;
            next.eta = null;
            next.samples = [];
            next.stepCurrent = null;
            next.stepTotal = null;
            next.lastStepCurrent = null;
            next.lastStepElapsed = null;
            next.lastStepAt = null;
            next.stepSamples = [];
            next.stepSeconds = null;
            if (!activeState.selectionIsManual || !activeState.selectedId) activeState.selectedId = snapshot.id;
        }

        const record = activeState.records[snapshot.id];
        record.visible = true;
        record.state = snapshot.aborting ? "aborting" : "current";
        record.rawName = snapshot.rawName;
        record.rawMessage = snapshot.rawMessage;
        record.activity = snapshot.activity || null;
        record.activityModel = String(snapshot.activityModel || "");
        // WanGP exposes Decode as a blocking VAE operation. Its progress value is
        // either a static 0% or the completed denoising bar, not decoder progress.
        record.progress = snapshot.id === "decode" ? null : snapshot.progress;
        record.elapsed = (Number.isFinite(record.elapsedBase) ? record.elapsedBase : 0) +
            (now - record.startedAt) / 1000;
        record.reportedElapsed = Number.isFinite(snapshot.stageElapsed) ? snapshot.stageElapsed : null;
        record.reportedAt = now;
        record.nativeEta = optionalNumber(snapshot.nativeEta);
        const lastSample = record.samples[record.samples.length - 1];
        if (Number.isFinite(record.progress) && Number.isFinite(record.elapsed) &&
            (!lastSample || lastSample.progress !== record.progress || Math.abs(lastSample.elapsed - record.elapsed) >= 1)) {
            record.samples.push({ progress: record.progress, elapsed: record.elapsed });
            record.samples = record.samples.slice(-30);
        }
        updateStepTiming(record, snapshot.steps, now);
        if (snapshot.aborting) record.eta = null;
        else if (record.id === "denoise") updateDenoiseEta(record);
        else updateEta(record);
        activeState.overallElapsed = snapshot.overallElapsed;
        const denoise = activeState.records.denoise;
        activeState.steps = snapshot.id === "decode" && Number.isFinite(denoise.stepTotal)
            ? { current: denoise.stepTotal, total: denoise.stepTotal }
            : snapshot.steps;
    }

    function stageTimeText(state, record) {
        if (record.preloaded) return "Preloaded";
        if (record.unreported) return "Not reported";
        if (record.state === "complete") return formatDuration(record.elapsed);
        if (record.state === "aborting") return Number.isFinite(record.elapsed) ? `${formatDuration(record.elapsed)} elapsed` : "Stopping…";
        if (record.state === "current") {
            const remaining = remainingEstimate(state, record);
            if (Number.isFinite(remaining)) return `${formatDuration(remaining, true)} left`;
            return Number.isFinite(record.elapsed) ? `${formatDuration(record.elapsed)} elapsed` : "Calculating…";
        }
        return "—";
    }

    function stageDisplayLabel(namespace, record) {
        if (!record || record.id !== "denoise" || !Number.isFinite(namespace.state.activeDenoisePhase)) {
            return record ? record.label : "";
        }
        const total = optionalNumber(namespace.activeRun && namespace.activeRun.settings && namespace.activeRun.settings.guidance_phases);
        const suffix = Number.isFinite(total) && total >= 2
            ? `Phase ${namespace.state.activeDenoisePhase} of ${Math.floor(total)}`
            : `Phase ${namespace.state.activeDenoisePhase}`;
        return `${record.label} · ${suffix}`;
    }

    function statusLabel(record) {
        if (record.preloaded) return "Preloaded";
        if (record.unreported) return "Not reported";
        if (record.activity === "unload") return "Unloading";
        if (record.activity === "unload-complete") return "Unloaded";
        if (record.activity === "unload-failed") return "Failed";
        if (record.state === "complete") return "Completed";
        if (record.state === "aborting") return "Aborting";
        if (record.state === "current") return "Running";
        return "Pending";
    }

    function stageSupportsEta(record) {
        if (!record) return false;
        if (record.id === "denoise") return true;
        return record.id === "post" &&
            ((Number.isFinite(record.progress) && record.progress > 0 && record.progress < 100) ||
                (Number.isFinite(record.eta) && record.eta > 0));
    }

    function remainingEstimate(state, record) {
        if (!stageSupportsEta(record)) return null;
        return Number.isFinite(record.eta) && record.eta > 0 ? record.eta : null;
    }

    function totalEta(state) {
        const current = state.records[state.currentId];
        return current ? remainingEstimate(state, current) : null;
    }

    function text(root, selector, value) {
        const element = root.querySelector(selector);
        if (element) element.textContent = value == null ? "" : String(value);
    }

    function setting(settings, ...keys) {
        for (const key of keys) {
            if (settings && settings[key] !== null && settings[key] !== undefined && settings[key] !== "") {
                return settings[key];
            }
        }
        return null;
    }

    function settingText(value) {
        if (value === null || value === undefined || value === "") return "—";
        if (Array.isArray(value)) return value.map(item => settingText(item)).join(", ");
        if (typeof value === "object") return JSON.stringify(value);
        return String(value);
    }

    function compactModelName(value) {
        if (Array.isArray(value)) return value.map(item => compactModelName(item)).join(", ");
        const raw = settingText(value);
        if (raw === "—") return raw;
        const clean = raw.split(/[?#]/, 1)[0].replace(/\\/g, "/").replace(/\/+$/, "");
        const filename = clean.slice(clean.lastIndexOf("/") + 1);
        if (!filename) return raw;
        try {
            return decodeURIComponent(filename);
        } catch (_) {
            return filename;
        }
    }

    function compactLoraNames(value) {
        const values = Array.isArray(value) ? value : [value];
        return values
            .map(item => compactModelName(item).replace(/\.safetensors$/i, ""))
            .filter(item => item && item !== "—")
            .join("\n");
    }

    function stageModelInfo(namespace, record) {
        if (record && /^unload/.test(String(record.activity || "")) && record.activityModel) {
            const name = compactModelName(record.activityModel);
            return {
                role: "Outgoing model",
                itemRole: "Outgoing model",
                names: [name],
                text: `Outgoing model: ${name}`
            };
        }
        const settings = namespace.activeRun && namespace.activeRun.settings;
        const components = settings && settings.component_models;
        if (record && record.id === "input" &&
            !/(?:vae|latent|auto.?encod|encod|decod)/i.test(`${record.rawName || ""} ${record.rawMessage || ""}`)) {
            return null;
        }
        const raw = components && record ? components[record.id] : null;
        const values = Array.isArray(raw) ? raw : (raw ? [raw] : []);
        const names = values
            .map(value => compactModelName(value))
            .filter(value => value && value !== "—");
        if (!names.length) return null;
        const roles = {
            prepare: ["Transformer", "Transformers"],
            input: ["Input VAE", "Input VAEs"],
            encode: ["Text encoder", "Text encoders"],
            decode: ["VAE", "VAEs"]
        };
        const roleNames = roles[record.id] || ["Model", "Models"];
        const itemRole = roleNames[0];
        const role = names.length > 1 ? roleNames[1] : itemRole;
        return {
            role,
            itemRole,
            names,
            text: `${role}: ${names.join(" + ")}`
        };
    }

    function runModel(run) {
        return compactModelName(setting(run.settings, "model_filename", "model_type", "base_model_type"));
    }

    function runModelLabel(run) {
        const name = settingText(setting(run.settings, "model_name"));
        return name === "—" ? runModel(run) : name;
    }

    function outputLabel(run) {
        const outputs = Array.isArray(run.outputs) ? run.outputs : [];
        if (outputs.length !== 1) return `${outputs.length} ${run.media_type || "output"}${outputs.length === 1 ? "" : "s"}`;
        return compactModelName(outputs[0]);
    }

    function galleryOutputRecords(run) {
        const records = Array.isArray(run && run.output_records) && run.output_records.length
            ? run.output_records
            : (Array.isArray(run && run.outputs)
                ? run.outputs.map(path => ({ path, media_type: mediaTypeFromPath(path) }))
                : []);
        return records
            .map(record => ({
                path: String(record && record.path || ""),
                media_type: String(record && record.media_type || mediaTypeFromPath(record && record.path)).toLowerCase()
            }))
            .filter(record => record.path && ["video", "image", "audio"].includes(record.media_type));
    }

    function applyGalleryFeedback(namespace, runId) {
        const id = String(runId || "");
        const feedback = namespace.galleryFeedback.get(id) || null;
        namespace.panel.querySelectorAll("[data-sp-gallery-feedback]").forEach(element => {
            if (String(element.dataset.spGalleryFeedback || "") !== id) return;
            element.hidden = !feedback;
            element.textContent = feedback ? feedback.message : "";
            element.dataset.status = feedback ? feedback.status : "";
        });
        namespace.panel.querySelectorAll("[data-sp-view-run], [data-sp-import-media]").forEach(button => {
            const buttonRunId = button.dataset.spViewRun || button.dataset.spImportMedia;
            if (String(buttonRunId || "") !== id) return;
            button.disabled = button.dataset.spHasOutput !== "true" || Boolean(feedback && feedback.status === "pending");
        });
    }

    function setGalleryFeedback(namespace, runId, status, message, timeoutMs = 0) {
        const id = String(runId || "");
        const feedback = { status, message, token: `${Date.now()}-${Math.random()}` };
        namespace.galleryFeedback.set(id, feedback);
        applyGalleryFeedback(namespace, id);
        if (timeoutMs > 0) {
            window.setTimeout(() => {
                if (namespace.galleryFeedback.get(id) !== feedback) return;
                namespace.galleryFeedback.delete(id);
                applyGalleryFeedback(namespace, id);
            }, timeoutMs);
        }
    }

    function requestGalleryAction(namespace, runId, outputIndex = null, operation = "navigate") {
        const id = String(runId || "");
        const run = namespace.runHistory.find(candidate => String(candidate.id) === id);
        const records = galleryOutputRecords(run);
        const selected = Number.isFinite(Number(outputIndex)) && outputIndex !== null
            ? records.slice(Number(outputIndex), Number(outputIndex) + 1)
            : records;
        if (!selected.length) {
            setGalleryFeedback(namespace, id, "missing", "No gallery output was recorded for this history entry.", 8000);
            return;
        }
        const field = namespace.container.querySelector(
            "#status-pro-gallery-request-bridge textarea, #status-pro-gallery-request-bridge input"
        );
        const trigger = namespace.container.querySelector("#status-pro-gallery-request-trigger");
        if (!field || !trigger) {
            setGalleryFeedback(namespace, id, "error", "Gallery navigation is unavailable in this Wan2GP session.", 8000);
            return;
        }
        const token = (window.crypto && window.crypto.randomUUID)
            ? window.crypto.randomUUID()
            : `gallery-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        namespace.galleryRequests.set(token, { runId: id, operation });
        setGalleryFeedback(
            namespace,
            id,
            "pending",
            operation === "import" ? "Checking the recorded output file…" : "Finding this output in the current gallery…"
        );
        field.value = JSON.stringify({ token, run_id: id, operation, outputs: selected });
        field.dispatchEvent(new Event("input", { bubbles: true }));
        trigger.click();
    }

    function requestGalleryNavigation(namespace, runId, outputIndex = null) {
        requestGalleryAction(namespace, runId, outputIndex, "navigate");
    }

    function requestGalleryImport(namespace, runId, outputIndex = null) {
        requestGalleryAction(namespace, runId, outputIndex, "import");
    }

    function readGalleryNavigationResult(namespace) {
        const field = namespace.container.querySelector(
            "#status-pro-gallery-result-bridge textarea, #status-pro-gallery-result-bridge input"
        );
        const raw = String(field ? field.value : "").trim();
        if (!raw || raw === "{}" || raw === namespace.galleryResultRaw) return;
        namespace.galleryResultRaw = raw;
        let result = null;
        try {
            result = JSON.parse(raw);
        } catch (_) {
            return;
        }
        const token = String(result && result.token || "");
        const pendingRequest = namespace.galleryRequests.get(token);
        const runId = pendingRequest && typeof pendingRequest === "object"
            ? String(pendingRequest.runId || "")
            : String(pendingRequest || result && result.run_id || "");
        const operation = pendingRequest && typeof pendingRequest === "object"
            ? String(pendingRequest.operation || "navigate")
            : "navigate";
        if (!runId) return;
        namespace.galleryRequests.delete(token);
        const status = String(result.status || "error");
        const message = String(result.message || "Gallery navigation did not complete.");
        setGalleryFeedback(namespace, runId, status, message, ["selected", "imported"].includes(status) ? 3500 : 8000);
        if (operation === "import" && ["imported", "missing", "error"].includes(status)) {
            window.alert(message);
        }
    }

    function phaseTimingSummary(run) {
        const stages = Object.values(run.stages || {});
        const total = stages.reduce((sum, stage) => {
            const duration = Number(stage && stage.duration_seconds);
            return Number.isFinite(duration) ? sum + duration : sum;
        }, 0);
        const setup = stages.reduce((sum, stage) => {
            const duration = Number(stage && stage.duration_seconds);
            return stage && stage.stage === "prepare" && Number.isFinite(duration) ? sum + duration : sum;
        }, 0);
        return { total, setup };
    }

    function humanizeModelToken(token) {
        const value = String(token || "");
        if (!value) return "";
        if (/^(i2v|t2v|ti2v|fl2v|vace|raw|cfg|h3)$/i.test(value)) return value.toUpperCase();
        if (/\d/.test(value)) return value.toUpperCase();
        return value.charAt(0).toUpperCase() + value.slice(1).toLowerCase();
    }

    function fallbackModelParts(modelType) {
        const raw = String(modelType || "").toLowerCase();
        let platform = "";
        let remainder = raw;
        const families = [
            [/^krea2(?:_|$)/, "Krea 2"],
            [/^wan_?2_2(?:_|$)/, "Wan 2.2"],
            [/^wan(?:_|$)/, "Wan 2.1"],
            [/^ltx2(?:_|$)/, "LTX-2"],
            [/^ltxv(?:_|$)/, "LTX Video"],
            [/^minimax_h3(?:_|$)/, "MiniMax H3"],
            [/^ideogram4(?:_|$)/, "Ideogram 4"],
            [/^qwen(?:_|$)/, "Qwen"],
            [/^flux(?:_|$)/, "FLUX"],
            [/^hunyuan(?:_|$)/, "Hunyuan"]
        ];
        for (const [pattern, label] of families) {
            if (!pattern.test(raw)) continue;
            platform = label;
            remainder = raw.replace(pattern, "");
            break;
        }
        const variant = remainder.split(/[_\s-]+/)
            .filter(token => token && token !== "identity" && token !== "edit")
            .map(humanizeModelToken)
            .join(" ");
        return { platform, variant };
    }

    function runDescriptor(run) {
        const importedSummary = String(run && run.imported_model_summary || "").trim();
        if (importedSummary) return importedSummary.slice(0, 500);
        const settings = run.settings || {};
        const modelType = setting(settings, "model_type", "base_model_type");
        const fallback = fallbackModelParts(modelType);
        let platform = settingText(setting(settings, "model_family"));
        if (platform === "—") platform = fallback.platform;
        platform = String(platform || "").replace(/\s+Identity\s+Edit$/i, "").trim();

        let variant = settingText(setting(settings, "model_name"));
        if (variant === "—" || /^Unknown model\b/i.test(variant)) variant = "";
        variant = String(variant || "")
            .replace(/\s*\(experimental\)\s*/gi, " ")
            .replace(/\bIdentity\s+Edit\b/gi, " ")
            .trim();
        if (platform && variant.toLowerCase().startsWith(platform.toLowerCase())) {
            variant = variant.slice(platform.length).trim();
        }
        variant = variant.replace(/^[\s—–-]+|[\s—–-]+$/g, "").trim() || fallback.variant;

        const media = String(run.media_type || "");
        const mediaLabel = media && media !== "unknown"
            ? media.charAt(0).toUpperCase() + media.slice(1).toLowerCase()
            : "";
        const resolution = settingText(setting(settings, "resolution"));
        const parts = [platform, variant, mediaLabel, resolution === "—" ? "" : resolution]
            .filter((value, index, values) => value && values.indexOf(value) === index);
        return parts.join(" - ") || runModel(run);
    }

    function addRunField(container, label, value, displayValue = null, fullValue = null) {
        if (value === null || value === undefined || value === "" || (Array.isArray(value) && value.length === 0)) return;
        const wrapper = document.createElement("div");
        wrapper.className = "status-pro__run-field";
        const term = document.createElement("dt");
        term.textContent = label;
        const description = document.createElement("dd");
        description.textContent = displayValue === null ? settingText(value) : settingText(displayValue);
        if (fullValue !== null && fullValue !== undefined && fullValue !== "") description.title = settingText(fullValue);
        wrapper.append(term, description);
        container.appendChild(wrapper);
    }

    function timingStageId(value, label = "") {
        const source = `${String(value || "")} ${String(label || "")}`.toLowerCase();
        if (/input|control|preprocess/.test(source)) return "input";
        if (/prepare|setup|load|model/.test(source)) return "prepare";
        if (/encode|prompt|text/.test(source)) return "encode";
        if (/denois|generat|sampl/.test(source)) return "denoise";
        if (/decode|vae/.test(source)) return "decode";
        if (/enhance|upscal|post|interpol/.test(source)) return "enhance";
        if (/save|mux|output/.test(source)) return "save";
        return "unaccounted";
    }

    function timingOverviewSegments(run) {
        const segments = Object.entries(run.stages || {})
            .map(([key, stage]) => ({
                stage: timingStageId(stage && stage.stage || String(key).split(":")[0], stage && stage.label),
                label: String(stage && stage.label || "Stage"),
                seconds: optionalNumber(stage && stage.duration_seconds)
            }))
            .filter(segment => Number.isFinite(segment.seconds) && segment.seconds > 0);
        const observed = segments.reduce((sum, segment) => sum + segment.seconds, 0);
        const wall = optionalNumber(run.duration_seconds);
        const unaccounted = Number.isFinite(wall) ? Math.max(0, wall - observed) : 0;
        if (unaccounted >= 0.5) segments.push({stage: "unaccounted", label: "Unaccounted", seconds: unaccounted});
        return segments;
    }

    function appendTimingOverview(body, run) {
        const segments = timingOverviewSegments(run);
        const total = segments.reduce((sum, segment) => sum + segment.seconds, 0);
        if (!(total > 0)) return [];

        const overview = document.createElement("div");
        overview.className = "status-pro__timing-overview";
        const label = document.createElement("span");
        label.className = "status-pro__timing-overview-label";
        label.textContent = "Observed timing composition";
        const bar = document.createElement("div");
        bar.className = "status-pro__timing-bar";
        bar.setAttribute("role", "img");
        bar.setAttribute("aria-label", segments.map(segment => `${segment.label}: ${formatDuration(segment.seconds, true)}`).join("; "));
        segments.forEach(segment => {
            const item = document.createElement("span");
            item.className = "status-pro__timing-segment";
            item.dataset.stage = segment.stage;
            item.style.width = `${(segment.seconds / total) * 100}%`;
            item.title = `${segment.label}: ${formatDuration(segment.seconds, true)} (${Math.round(segment.seconds / total * 1000) / 10}%)`;
            bar.appendChild(item);
        });
        overview.append(label, bar);
        body.appendChild(overview);
        return segments;
    }

    function stepTimingOutliers(steps) {
        const groups = new Map();
        steps.forEach((step, index) => {
            const duration = optionalNumber(step && step.duration_seconds);
            if (!Number.isFinite(duration) || duration < 0 || step.skipped === true) return;
            const passNo = optionalNumber(step.pass_no);
            const phaseNo = optionalNumber(step.phase);
            const key = Number.isFinite(passNo) && passNo > 0
                ? `pass:${passNo}`
                : (Number.isFinite(phaseNo) && phaseNo > 0 ? `phase:${phaseNo}` : `label:${String(step.label || "generate")}`);
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push({index, duration});
        });
        const fastest = new Set();
        const slowest = new Set();
        groups.forEach(records => {
            if (records.length < 2) return;
            const minimum = Math.min(...records.map(record => record.duration));
            const maximum = Math.max(...records.map(record => record.duration));
            if (minimum === maximum) return;
            records.forEach(record => {
                if (record.duration === minimum) fastest.add(record.index);
                if (record.duration === maximum) slowest.add(record.index);
            });
        });
        return {fastest, slowest};
    }

    function addImportedMediaField(container, run, records, pending = false) {
        if (!records.length) return;
        const wrapper = document.createElement("div");
        wrapper.className = "status-pro__run-field";
        const term = document.createElement("dt");
        term.textContent = records.length === 1 ? "Gallery import" : "Gallery imports";
        const actions = document.createElement("dd");
        actions.className = "status-pro__run-field-actions";
        records.forEach((record, outputIndex) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "status-pro__gallery-view";
            button.dataset.spImportMedia = run.id;
            button.dataset.spOutputIndex = String(outputIndex);
            button.dataset.spHasOutput = "true";
            button.textContent = records.length === 1 ? "Import media" : `Import ${outputIndex + 1}`;
            button.title = `Add ${compactModelName(record.path)} to its Wan2GP gallery`;
            button.disabled = pending;
            actions.appendChild(button);
        });
        wrapper.append(term, actions);
        container.appendChild(wrapper);
    }

    function resourceMetric(run, field, statistic = "peak_bytes") {
        return optionalNumber(run && run.resources && run.resources.metrics && run.resources.metrics[field] && run.resources.metrics[field][statistic]);
    }

    function stepSkippingLabel(run) {
        const settings = run.settings || {};
        const method = settingText(setting(settings, "skip_steps_cache_type"));
        if (method === "—") return "None";
        const names = { tea: "TeaCache", mag: "MagCache", spectrum: "Spectrum", first_block: "First Block Cache" };
        const parts = [names[String(method).toLowerCase()] || method];
        const multiplier = optionalNumber(setting(settings, "skip_steps_multiplier"));
        if (Number.isFinite(multiplier)) parts.push(String(method).toLowerCase() === "first_block" ? `threshold ${multiplier}` : `×${multiplier}`);
        const start = optionalNumber(setting(settings, "skip_steps_start_step_perc"));
        if (Number.isFinite(start)) parts.push(`from ${start}%`);
        return parts.join(" · ");
    }

    function passObservationLabel(stepSummary) {
        const observedPasses = optionalNumber(stepSummary && stepSummary.observed_passes);
        if (!(observedPasses > 1)) return observedPasses;
        const passes = Array.isArray(stepSummary && stepSummary.passes) ? stepSummary.passes : [];
        const observedCounts = passes.map(pass => optionalNumber(pass && pass.observed_steps));
        const configuredCounts = passes.map(pass => optionalNumber(pass && pass.configured_steps));
        const completeCounts = passes.length === observedPasses &&
            observedCounts.every(Number.isFinite) && configuredCounts.every(Number.isFinite) &&
            observedCounts.every((count, index) => count === configuredCounts[index]);
        const commonConfigured = completeCounts && configuredCounts.every(value => value === configuredCounts[0])
            ? configuredCounts[0]
            : null;
        if (Number.isFinite(commonConfigured)) return `${observedPasses} × ${commonConfigured} configured steps`;
        if (passes.length === observedPasses && observedCounts.every(Number.isFinite)) {
            return `${observedPasses} passes · ${observedCounts.join(" + ")} observations`;
        }
        return `${observedPasses} passes`;
    }

    function appendStepPerformance(body, run) {
        const steps = Array.isArray(run.step_performance) ? run.step_performance : [];
        if (!steps.length) return;
        const details = document.createElement("details");
        details.className = "status-pro__step-log";
        const summary = document.createElement("summary");
        const label = document.createElement("span");
        label.textContent = `Step observations (${steps.length})`;
        const hint = document.createElement("span");
        const observedPasses = optionalNumber(run.step_summary && run.step_summary.observed_passes);
        hint.textContent = run.step_summary && run.step_summary.truncated
            ? "Latest observations · source truncated"
            : `${observedPasses > 1 ? `${observedPasses} passes · ` : ""}Time · skipping · memory`;
        summary.append(label, hint);
        const wrap = document.createElement("div");
        wrap.className = "status-pro__step-log-table-wrap";
        const table = document.createElement("table");
        const head = document.createElement("thead");
        const header = document.createElement("tr");
        ["Phase", "Step", "Time", "Skipped", "RAM", "VRAM allocated", "VRAM reserved", "GPU used"].forEach(value => {
            const cell = document.createElement("th");
            cell.textContent = value;
            header.appendChild(cell);
        });
        head.appendChild(header);
        const tableBody = document.createElement("tbody");
        const outliers = stepTimingOutliers(steps);
        steps.forEach((step, stepIndex) => {
            const row = document.createElement("tr");
            const memory = step && step.memory || {};
            const skipped = step.skipped === true ? (optionalNumber(step.skipped_delta) > 1 ? `Yes (×${step.skipped_delta})` : "Yes") : (step.skipped === false ? "No" : "—");
            const passNo = optionalNumber(step.pass_no);
            const phaseNo = optionalNumber(step.phase);
            const values = [
                step.label || (Number.isFinite(passNo) && passNo > 0 ? `Pass ${passNo}` : (Number.isFinite(phaseNo) && phaseNo > 0 ? `Phase ${phaseNo}` : "Generate")),
                Number.isFinite(optionalNumber(step.total_steps)) ? `${step.step}/${step.total_steps}` : step.step,
                formatStepDuration(optionalNumber(step.duration_seconds)),
                skipped,
                formatBytes(optionalNumber(memory.ram_rss_bytes)),
                formatBytes(optionalNumber(memory.vram_allocated_bytes)),
                formatBytes(optionalNumber(memory.vram_reserved_bytes)),
                formatBytes(optionalNumber(memory.vram_device_used_bytes))
            ];
            values.forEach((value, index) => {
                const cell = document.createElement("td");
                cell.textContent = value === null || value === undefined ? "—" : String(value);
                if (index === 3 && step.skipped === true) cell.className = "status-pro__step-skipped";
                if (index === 2 && outliers.fastest.has(stepIndex)) {
                    cell.className = "status-pro__step-fastest";
                    cell.title = "Fastest observed step in this pass";
                } else if (index === 2 && outliers.slowest.has(stepIndex)) {
                    cell.className = "status-pro__step-slowest";
                    cell.title = "Slowest observed step in this pass";
                }
                row.appendChild(cell);
            });
            tableBody.appendChild(row);
        });
        table.append(head, tableBody);
        wrap.appendChild(table);
        details.append(summary, wrap);
        body.appendChild(details);
    }

    function historyRuns(namespace) {
        return namespace.historyScope === "session"
            ? namespace.runHistory.filter(run => namespace.sessionRunIds.has(run.id))
            : namespace.runHistory;
    }

    function selectedHistoryRuns(namespace) {
        return namespace.runHistory.filter(run => namespace.selectedRunIds.has(String(run.id)));
    }

    function createHistoryRun(namespace, run, fallbackNumber, grouped = false) {
            const details = document.createElement("details");
            details.className = grouped ? "status-pro__run status-pro__run--grouped" : "status-pro__run";
            details.dataset.spRunDetails = String(run.id);
            details.open = namespace.openHistoryRuns.has(String(run.id));
            details.addEventListener("toggle", () => {
                if (details.open) namespace.openHistoryRuns.add(String(run.id));
                else namespace.openHistoryRuns.delete(String(run.id));
            });
            const summary = document.createElement("summary");
            const imported = Boolean(run.imported);
            const galleryRecords = galleryOutputRecords(run);
            const nativeGalleryRecords = imported ? [] : galleryRecords;
            const select = document.createElement("input");
            select.type = "checkbox";
            select.className = "status-pro__run-select";
            select.dataset.spRunSelect = run.id;
            select.checked = namespace.selectedRunIds.has(String(run.id));
            const childLabel = Number.isFinite(optionalNumber(run.window_no))
                ? `window ${Math.floor(Number(run.window_no))}`
                : `run ${fallbackNumber}`;
            select.setAttribute("aria-label", `Select ${grouped ? childLabel : (run.queue_task_id !== null && run.queue_task_id !== undefined ? `task ${run.queue_task_id}` : childLabel)} for export`);
            const title = document.createElement("span");
            title.className = "status-pro__run-title";
            title.textContent = grouped
                ? (Number.isFinite(optionalNumber(run.window_no)) ? `Window ${Math.floor(Number(run.window_no))}` : `Run ${fallbackNumber}`)
                : run.queue_task_id !== null && run.queue_task_id !== undefined
                ? `Task #${run.queue_task_id}`
                : `Run ${fallbackNumber}`;
            const model = document.createElement("span");
            model.className = "status-pro__run-model";
            model.textContent = runDescriptor(run);
            model.title = `${model.textContent} · ${runModel(run)}`;
            const status = document.createElement("span");
            status.className = "status-pro__run-summary-value status-pro__run-status";
            const displayedStatus = grouped && run.status === "window" ? "completed" : String(run.status || "completed");
            status.dataset.status = displayedStatus;
            status.textContent = run.status === "window" && !grouped
                ? `Window ${run.window_no || ""}`.trim()
                : displayedStatus;
            const duration = document.createElement("span");
            duration.className = "status-pro__run-summary-value";
            duration.textContent = formatDuration(Number(run.duration_seconds));
            const completed = document.createElement("span");
            completed.className = "status-pro__run-summary-value";
            completed.textContent = formatClock(Number(run.completed_at));
            const viewOutput = document.createElement("button");
            viewOutput.type = "button";
            viewOutput.className = "status-pro__gallery-view";
            viewOutput.dataset.spViewRun = run.id;
            viewOutput.dataset.spHasOutput = nativeGalleryRecords.length ? "true" : "false";
            viewOutput.textContent = "View";
            viewOutput.disabled = !nativeGalleryRecords.length;
            viewOutput.title = imported
                ? "Expand this record and use Import media to add its recorded file to a gallery"
                : nativeGalleryRecords.length
                ? "Select this output in its Wan2GP gallery"
                : "No gallery output was recorded for this run";
            viewOutput.setAttribute("aria-label", imported
                ? `${title.textContent} can be added with Import media in its expanded details`
                : nativeGalleryRecords.length
                ? `View ${title.textContent} output in gallery`
                : `${title.textContent} has no recorded gallery output`);
            const feedbackState = namespace.galleryFeedback.get(String(run.id));
            const galleryFeedback = document.createElement("span");
            galleryFeedback.className = "status-pro__gallery-feedback";
            galleryFeedback.dataset.spGalleryFeedback = run.id;
            galleryFeedback.hidden = !feedbackState;
            galleryFeedback.textContent = feedbackState ? feedbackState.message : "";
            galleryFeedback.dataset.status = feedbackState ? feedbackState.status : "";
            if (feedbackState && feedbackState.status === "pending") viewOutput.disabled = true;
            summary.append(select, title, model, status, duration, completed, viewOutput, galleryFeedback);

            const body = document.createElement("div");
            body.className = "status-pro__run-detail";
            const fields = document.createElement("dl");
            fields.className = "status-pro__run-fields";
            const settings = run.settings || {};
            addRunField(fields, "Started", formatDateTime(Number(run.started_at)));
            addRunField(fields, "Completed", formatDateTime(Number(run.completed_at)));
            if (imported) {
                addRunField(fields, "History source", "Imported Status Pro JSON");
                addRunField(fields, "Imported", formatDateTime(Number(run.imported_at)));
                addRunField(fields, "Original export", formatDateTime(Number(run.import_source && run.import_source.exported_at)));
                addRunField(fields, "Export version", run.import_source && run.import_source.version);
            }
            addRunField(fields, "Total time", formatDuration(Number(run.duration_seconds)));
            const generationTime = optionalNumber(setting(settings, "generation_time"));
            const phaseTiming = phaseTimingSummary(run);
            if (Number.isFinite(generationTime) && generationTime >= 0) {
                addRunField(fields, "Generation", formatDuration(generationTime));
            }
            if (phaseTiming.setup >= 0.5) addRunField(fields, "Model loading", formatDuration(phaseTiming.setup));
            const unaccounted = Number(run.duration_seconds) - phaseTiming.total;
            if (Number.isFinite(unaccounted) && unaccounted >= 1) addRunField(fields, "Unaccounted", formatDuration(unaccounted, true));
            addRunField(fields, "Outcome detail", run.status_reason || run.failure_reason);
            if (run.status === "window") addRunField(fields, "Window", run.total_windows ? `${run.window_no}/${run.total_windows}` : run.window_no);
            addRunField(fields, "Media type", run.media_type);
            addRunField(fields, "Frames", run.frame_count);
            addRunField(fields, "Output count", run.output_count);
            addRunField(fields, "Model", runModelLabel(run));
            addRunField(fields, "Resolution", setting(settings, "resolution"));
            addRunField(fields, "Steps", setting(settings, "num_inference_steps"));
            addRunField(fields, "FPS", setting(settings, "force_fps", "fps"));
            addRunField(fields, "Seed", setting(settings, "seed"));
            addRunField(fields, "Guidance", setting(settings, "guidance_scale"));
            addRunField(fields, "Guidance 2", setting(settings, "guidance2_scale"));
            addRunField(fields, "Guidance 3", setting(settings, "guidance3_scale"));
            addRunField(fields, "Flow shift", setting(settings, "flow_shift"));
            addRunField(fields, "Sampler", setting(settings, "sample_solver"));
            addRunField(fields, "Step skipping", stepSkippingLabel(run));
            const stepSummary = run.step_summary || {};
            if (optionalNumber(stepSummary.recorded_steps) > 0) {
                const observedPasses = optionalNumber(stepSummary.observed_passes);
                if (observedPasses > 1) {
                    addRunField(fields, "Passes observed", passObservationLabel(stepSummary));
                }
                addRunField(fields, "Step observations", stepSummary.recorded_steps);
                addRunField(fields, "Steps skipped", stepSummary.skipped_steps);
                addRunField(fields, "Average step", formatStepDuration(optionalNumber(stepSummary.average_seconds)));
                addRunField(fields, "Fastest step", formatStepDuration(optionalNumber(stepSummary.fastest_seconds)));
                addRunField(fields, "Slowest step", formatStepDuration(optionalNumber(stepSummary.slowest_seconds)));
            }
            const peakRam = resourceMetric(run, "ram_rss_bytes");
            const averageRam = resourceMetric(run, "ram_rss_bytes", "average_bytes");
            const peakVramAllocated = resourceMetric(run, "vram_allocated_bytes");
            const averageVramAllocated = resourceMetric(run, "vram_allocated_bytes", "average_bytes");
            const peakVramReserved = resourceMetric(run, "vram_reserved_bytes");
            const peakGpuUsed = resourceMetric(run, "vram_device_used_bytes");
            if (Number.isFinite(peakRam)) addRunField(fields, "Observed peak process RAM", formatBytes(peakRam));
            if (Number.isFinite(averageRam)) addRunField(fields, "Average process RAM", formatBytes(averageRam));
            if (Number.isFinite(peakVramAllocated)) addRunField(fields, "Observed peak VRAM allocated", formatBytes(peakVramAllocated));
            if (Number.isFinite(averageVramAllocated)) addRunField(fields, "Average VRAM allocated", formatBytes(averageVramAllocated));
            if (Number.isFinite(peakVramReserved)) addRunField(fields, "Observed peak VRAM reserved", formatBytes(peakVramReserved));
            if (Number.isFinite(peakGpuUsed)) addRunField(fields, "Observed peak GPU memory used", formatBytes(peakGpuUsed));
            addRunField(fields, "GPU", run.resources && run.resources.gpu_name);
            const loras = setting(settings, "activated_loras");
            addRunField(fields, "LoRAs", loras, compactLoraNames(loras), loras);
            addRunField(fields, "Output", outputLabel(run));
            if (imported) {
                addImportedMediaField(
                    fields,
                    run,
                    galleryRecords,
                    Boolean(feedbackState && feedbackState.status === "pending")
                );
            }
            body.appendChild(fields);

            if (!imported && galleryRecords.length > 1) {
                const outputActions = document.createElement("div");
                outputActions.className = "status-pro__output-actions";
                const outputActionsTitle = document.createElement("span");
                outputActionsTitle.className = "status-pro__output-actions-title";
                outputActionsTitle.textContent = "Gallery outputs";
                outputActions.appendChild(outputActionsTitle);
                galleryRecords.forEach((record, outputIndex) => {
                    const row = document.createElement("div");
                    row.className = "status-pro__output-action";
                    const name = document.createElement("span");
                    name.className = "status-pro__output-action-name";
                    const mediaLabel = record.media_type.charAt(0).toUpperCase() + record.media_type.slice(1);
                    name.textContent = `${mediaLabel} ${outputIndex + 1}: ${compactModelName(record.path)}`;
                    name.title = record.path;
                    const button = document.createElement("button");
                    button.type = "button";
                    button.className = "status-pro__gallery-view";
                    button.dataset.spViewRun = run.id;
                    button.dataset.spOutputIndex = String(outputIndex);
                    button.dataset.spHasOutput = "true";
                    button.textContent = "View";
                    button.title = `Select ${mediaLabel.toLowerCase()} ${outputIndex + 1} in its Wan2GP gallery`;
                    if (feedbackState && feedbackState.status === "pending") button.disabled = true;
                    row.append(name, button);
                    outputActions.appendChild(row);
                });
                body.appendChild(outputActions);
            }

            const timingSegments = appendTimingOverview(body, run);
            const stages = document.createElement("div");
            stages.className = "status-pro__stage-breakdown";
            Object.values(run.stages || {}).forEach(stage => {
                if (Number(stage && stage.duration_seconds) < 1 && !stage.preloaded && !stage.unreported) return;
                const chip = document.createElement("span");
                const stageTime = stage.preloaded
                    ? "Preloaded"
                    : (stage.unreported ? "Not reported" : formatDuration(Number(stage.duration_seconds)));
                chip.textContent = `${stage.label}: ${stageTime}`;
                chip.dataset.stage = timingStageId(stage.stage, stage.label);
                stages.appendChild(chip);
            });
            const unaccountedTiming = timingSegments.find(segment => segment.stage === "unaccounted" && segment.label === "Unaccounted");
            if (unaccountedTiming) {
                const chip = document.createElement("span");
                chip.dataset.stage = "unaccounted";
                chip.textContent = `Unaccounted: ${formatDuration(unaccountedTiming.seconds)}`;
                chip.title = "Wall-clock time not assigned to a reported Wan2GP stage.";
                stages.appendChild(chip);
            }
            if (stages.childElementCount) body.appendChild(stages);
            appendStepPerformance(body, run);
            details.append(summary, body);
            return details;
    }

    function historyTaskKey(run) {
        const taskId = run && run.queue_task_id;
        if (taskId === null || taskId === undefined || taskId === "") return `run:${String(run && run.id || "unknown")}`;
        return `task:${String(run.session_id || "legacy")}:${String(taskId)}`;
    }

    function groupHistoryRuns(runs) {
        const groups = [];
        const byKey = new Map();
        runs.forEach(run => {
            const key = historyTaskKey(run);
            let group = byKey.get(key);
            if (!group) {
                group = { key, runs: [] };
                byKey.set(key, group);
                groups.push(group);
            }
            group.runs.push(run);
        });
        groups.forEach(group => {
            group.runs.sort((left, right) => {
                const leftWindow = optionalNumber(left.window_no);
                const rightWindow = optionalNumber(right.window_no);
                if (Number.isFinite(leftWindow) && Number.isFinite(rightWindow) && leftWindow !== rightWindow) {
                    return leftWindow - rightWindow;
                }
                return (Number(left.started_at) || 0) - (Number(right.started_at) || 0);
            });
        });
        return groups;
    }

    function historyTaskSummary(group) {
        const runs = group.runs;
        const representative = runs.slice().sort((left, right) => (Number(right.completed_at) || 0) - (Number(left.completed_at) || 0))[0];
        const expectedWindows = runs.reduce((maximum, run) => Math.max(maximum, optionalNumber(run.total_windows) || 0), 0);
        const observedWindows = new Set(runs
            .map(run => optionalNumber(run.window_no))
            .filter(value => Number.isFinite(value) && value > 0)).size;
        const isWindowed = expectedWindows > 1 || observedWindows > 0 || runs.some(run => run.status === "window");
        const completedUnits = isWindowed ? (observedWindows || runs.length) : runs.length;
        const expectedUnits = isWindowed ? Math.max(expectedWindows, completedUnits) : runs.length;
        const statuses = runs.map(run => String(run.status || "completed"));
        let status = statuses.includes("failed")
            ? "failed"
            : (statuses.includes("aborted") ? "aborted" : "completed");
        if (status === "completed" && isWindowed && completedUnits < expectedUnits) status = "incomplete";
        const started = runs.map(run => Number(run.started_at)).filter(Number.isFinite);
        const completed = runs.map(run => Number(run.completed_at)).filter(Number.isFinite);
        const firstStarted = started.length ? Math.min(...started) : null;
        const lastCompleted = completed.length ? Math.max(...completed) : null;
        const wallSeconds = Number.isFinite(firstStarted) && Number.isFinite(lastCompleted) && lastCompleted >= firstStarted
            ? (lastCompleted - firstStarted) / 1000
            : runs.reduce((sum, run) => sum + Math.max(0, Number(run.duration_seconds) || 0), 0);
        return {
            representative,
            status,
            duration: wallSeconds,
            completedAt: lastCompleted,
            unitLabel: isWindowed
                ? `${completedUnits < expectedUnits ? `${completedUnits}/${expectedUnits}` : expectedUnits} window${expectedUnits === 1 ? "" : "s"}`
                : `${runs.length} run${runs.length === 1 ? "" : "s"}`
        };
    }

    function createHistoryTaskGroup(namespace, group) {
        const aggregate = historyTaskSummary(group);
        const representative = aggregate.representative;
        const details = document.createElement("details");
        details.className = "status-pro__run status-pro__task-group";
        details.dataset.spTaskGroup = group.key;
        details.open = namespace.openHistoryGroups.has(group.key);
        details.addEventListener("toggle", () => {
            if (details.open) namespace.openHistoryGroups.add(group.key);
            else namespace.openHistoryGroups.delete(group.key);
        });

        const summary = document.createElement("summary");
        const select = document.createElement("input");
        select.type = "checkbox";
        select.className = "status-pro__run-select";
        select.dataset.spTaskSelect = group.key;
        const selectedCount = group.runs.filter(run => namespace.selectedRunIds.has(String(run.id))).length;
        select.checked = selectedCount === group.runs.length;
        select.indeterminate = selectedCount > 0 && selectedCount < group.runs.length;
        select.setAttribute("aria-label", `Select all ${aggregate.unitLabel} in task ${representative.queue_task_id} for export`);

        const title = document.createElement("span");
        title.className = "status-pro__run-title";
        title.textContent = `Task #${representative.queue_task_id}`;
        const model = document.createElement("span");
        model.className = "status-pro__run-model";
        model.textContent = runDescriptor(representative);
        model.title = `${model.textContent} · ${runModel(representative)}`;
        const status = document.createElement("span");
        status.className = "status-pro__run-summary-value status-pro__run-status";
        status.dataset.status = aggregate.status;
        status.textContent = aggregate.status;
        const duration = document.createElement("span");
        duration.className = "status-pro__run-summary-value";
        duration.textContent = formatDuration(aggregate.duration);
        duration.title = "Wall-clock time from the first recorded window/run to the last completion";
        const completed = document.createElement("span");
        completed.className = "status-pro__run-summary-value";
        completed.textContent = formatClock(aggregate.completedAt);
        const count = document.createElement("span");
        count.className = "status-pro__task-count";
        count.textContent = aggregate.unitLabel;
        summary.append(select, title, model, status, duration, completed, count);

        const children = document.createElement("div");
        children.className = "status-pro__task-runs";
        group.runs.forEach((run, index) => children.appendChild(createHistoryRun(namespace, run, index + 1, true)));
        details.append(summary, children);
        return details;
    }

    function renderHistory(namespace) {
        const container = namespace.panel.querySelector("[data-sp-history]");
        const empty = namespace.panel.querySelector("[data-sp-history-empty]");
        if (!container || !empty) return;
        const runs = historyRuns(namespace);
        const key = `${namespace.historyRecording === false ? "off:" : ""}${namespace.historyScope}:${runs.map(run => run.id).join("|")}`;
        if (namespace.historyRenderKey === key) return;
        namespace.historyRenderKey = key;
        empty.hidden = runs.length > 0;
        empty.textContent = namespace.historyRecording === false && !runs.length
            ? "Automatic history recording is off. Existing records are unchanged."
            : namespace.historyScope === "session"
            ? "No generations recorded for this session yet."
            : "No generations recorded yet.";
        const fragment = document.createDocumentFragment();
        const groups = groupHistoryRuns(runs);
        namespace.visibleHistoryGroups = new Map(groups.map(group => [group.key, group]));
        groups.forEach((group, index) => {
            fragment.appendChild(group.runs.length > 1
                ? createHistoryTaskGroup(namespace, group)
                : createHistoryRun(namespace, group.runs[0], groups.length - index, false));
        });
        container.replaceChildren(fragment);
    }

    function exportScopeRuns(namespace) {
        const selected = selectedHistoryRuns(namespace);
        return selected.length ? selected : historyRuns(namespace);
    }

    function runWithRecoverablePrompts(namespace, sourceRun) {
        const run = cloneJson(sourceRun, {});
        const cached = namespace.recoverablePrompts.get(String(sourceRun.id));
        if (!cached) return run;
        run.settings = { ...(run.settings || {}), ...cloneJson(cached.settings, {}) };
        (run.output_records || []).forEach((record, index) => {
            const recovered = cached.outputRecords && cached.outputRecords[index];
            if (!recovered) return;
            record.settings = { ...(record.settings || {}), ...cloneJson(recovered, {}) };
        });
        return run;
    }

    function exportSourceRuns(namespace) {
        return exportScopeRuns(namespace).map(run => runWithRecoverablePrompts(namespace, run));
    }

    function cleanPromptFields(settings, selectedFields) {
        const cleaned = cloneJson(settings, {});
        if (!selectedFields.has("prompt")) delete cleaned.prompt;
        if (!selectedFields.has("negative_prompt")) delete cleaned.negative_prompt;
        return cleaned;
    }

    function exportFieldValue(run, fieldId, selectedFields) {
        const settings = run.settings || {};
        if (fieldId === "run_id") return run.id;
        if (fieldId === "session_id") return run.session_id;
        if (fieldId === "queue_task_id") return run.queue_task_id;
        if (fieldId === "repeats") return run.repeats;
        if (fieldId === "status") return run.status;
        if (fieldId === "outcome") return run.status_reason || run.failure_reason || null;
        if (fieldId === "started_at") return Number.isFinite(Number(run.started_at)) ? new Date(Number(run.started_at)).toISOString() : null;
        if (fieldId === "completed_at") return Number.isFinite(Number(run.completed_at)) ? new Date(Number(run.completed_at)).toISOString() : null;
        if (fieldId === "duration_seconds") return optionalNumber(run.duration_seconds);
        if (fieldId === "generation_time") return optionalNumber(setting(settings, "generation_time"));
        if (fieldId === "phase_timings") return cloneJson(run.stages, {});
        if (fieldId === "step_performance") return cloneJson(run.step_performance, []);
        if (fieldId === "resource_usage") return cloneJson(run.resources, null);
        if (fieldId === "step_skipping") return {
            method: setting(settings, "skip_steps_cache_type"),
            multiplier_or_threshold: setting(settings, "skip_steps_multiplier"),
            start_percent: setting(settings, "skip_steps_start_step_perc"),
            recorded_skipped_steps: optionalNumber(run.step_summary && run.step_summary.skipped_steps),
            recorded_steps: optionalNumber(run.step_summary && run.step_summary.recorded_steps),
            step_observations: optionalNumber(run.step_summary && run.step_summary.recorded_steps),
            observed_passes: optionalNumber(run.step_summary && run.step_summary.observed_passes),
            pass_summaries: cloneJson(run.step_summary && run.step_summary.passes, [])
        };
        if (fieldId === "model_summary") return runDescriptor(run);
        if (fieldId === "model_name") return runModelLabel(run);
        if (fieldId === "checkpoint") return setting(settings, "model_filename", "model_type", "base_model_type");
        if (fieldId === "resolution") return setting(settings, "resolution");
        if (fieldId === "steps") return setting(settings, "num_inference_steps");
        if (fieldId === "fps") return setting(settings, "force_fps", "fps");
        if (fieldId === "seed") return setting(settings, "seed");
        if (fieldId === "guidance") return setting(settings, "guidance_scale");
        if (fieldId === "guidance2") return setting(settings, "guidance2_scale");
        if (fieldId === "guidance3") return setting(settings, "guidance3_scale");
        if (fieldId === "flow_shift") return setting(settings, "flow_shift");
        if (fieldId === "sampler") return setting(settings, "sample_solver");
        if (fieldId === "loras") return cloneJson(setting(settings, "activated_loras"), null);
        if (fieldId === "settings") return cleanPromptFields(settings, selectedFields);
        if (fieldId === "media_type") return run.media_type;
        if (fieldId === "frame_count") return run.frame_count;
        if (fieldId === "output_count") return run.output_count;
        if (fieldId === "outputs") return cloneJson(run.outputs, []);
        if (fieldId === "output_records") {
            return (run.output_records || []).map(record => ({
                ...cloneJson(record, {}),
                settings: cleanPromptFields(record && record.settings, selectedFields)
            }));
        }
        if (fieldId === "prompt" || fieldId === "negative_prompt") {
            const direct = setting(settings, fieldId);
            if (direct !== null) return direct;
            for (const record of (run.output_records || [])) {
                const resolved = setting(record && record.settings, fieldId);
                if (resolved !== null) return resolved;
            }
            return null;
        }
        return null;
    }

    function buildExportRecord(run, selectedFields) {
        const record = {};
        EXPORT_FIELD_DEFS.forEach(field => {
            if (selectedFields.has(field.id)) record[field.id] = exportFieldValue(run, field.id, selectedFields);
        });
        return record;
    }

    function exportableRuns(namespace, selectedFields) {
        return exportSourceRuns(namespace).map(run => buildExportRecord(run, selectedFields));
    }

    function csvCell(value) {
        const string = value === null || value === undefined ? "" : String(value);
        return `"${string.replace(/"/g, '""')}"`;
    }

    function exportCsv(records, selectedFields) {
        const selectedColumns = EXPORT_FIELD_DEFS.filter(field => selectedFields.has(field.id)).map(field => field.id);
        const stageColumns = ["prepare", "input", "encode", "denoise", "decode", "post", "save"];
        const columns = selectedColumns.flatMap(column => column === "phase_timings"
            ? [...stageColumns.map(stage => `${stage}_seconds`), "phase_timings"]
            : [column]);
        const rows = records.map(record => {
            const stageTotals = Object.values(record.phase_timings || {}).reduce((totals, stage) => {
                const duration = Number(stage && stage.duration_seconds);
                if (stage && stage.stage && Number.isFinite(duration)) totals[stage.stage] = (totals[stage.stage] || 0) + duration;
                return totals;
            }, {});
            return columns.map(column => {
                const stageMatch = column.match(/^(prepare|input|encode|denoise|decode|post|save)_seconds$/);
                const value = stageMatch ? stageTotals[stageMatch[1]] : record[column];
                return csvCell(value && typeof value === "object" ? JSON.stringify(value) : value);
            }).join(",");
        });
        return [columns.map(csvCell).join(","), ...rows].join("\r\n");
    }

    function markdownCell(value) {
        return String(value === null || value === undefined ? "" : value).replace(/\|/g, "\\|").replace(/\r?\n/g, " ");
    }

    function exportMarkdown(records, selectedFields, metadata) {
        const fields = EXPORT_FIELD_DEFS.filter(field => selectedFields.has(field.id));
        const lines = [
            "# Status Pro generation history",
            "",
            `Exported: ${new Date().toISOString()}`,
            `Scope: ${metadata.scope}`,
            `Preset: ${metadata.preset}`,
            `Fields: ${fields.map(field => field.label).join(", ")}`
        ];
        records.forEach((record, index) => {
            const task = record.queue_task_id !== null && record.queue_task_id !== undefined ? record.queue_task_id : index + 1;
            lines.push("", `## Task ${markdownCell(task)}`, "");
            fields.forEach(field => {
                const value = record[field.id];
                if (value && typeof value === "object") {
                    lines.push(`### ${field.label}`, "", "```json", JSON.stringify(value, null, 2), "```");
                } else {
                    lines.push(`- ${field.label}: ${markdownCell(value)}`);
                }
            });
        });
        return lines.join("\n");
    }

    function downloadText(filename, mimeType, contents) {
        const url = URL.createObjectURL(new Blob([contents], { type: mimeType }));
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = filename;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    function exportHistory(namespace, format, selectedFields) {
        const records = exportableRuns(namespace, selectedFields);
        if (!records.length || !selectedFields.size) return;
        const stamp = new Date().toISOString().replace(/[:.]/g, "-");
        const metadata = {
            scope: selectedHistoryRuns(namespace).length ? "selected" : namespace.historyScope,
            preset: exportPresetLabel(namespace),
            fields: EXPORT_FIELD_DEFS.filter(field => selectedFields.has(field.id)).map(field => field.id)
        };
        if (format === "csv") downloadText(`status-pro-${stamp}.csv`, "text/csv;charset=utf-8", exportCsv(records, selectedFields));
        else if (format === "md") downloadText(`status-pro-${stamp}.md`, "text/markdown;charset=utf-8", exportMarkdown(records, selectedFields, metadata));
        else {
            const exportedAt = new Date();
            downloadText(`status-pro-${stamp}.json`, "application/json;charset=utf-8", JSON.stringify({
                exported_at: exportedAt.toISOString(),
                exported_at_local: localIsoTimestamp(exportedAt),
                version: "1.0.1",
                ...metadata,
                runs: records
            }, null, 2));
        }
    }

    function importedTimestamp(value) {
        const numeric = optionalNumber(value);
        if (Number.isFinite(numeric)) return numeric < 100000000000 ? numeric * 1000 : numeric;
        const parsed = Date.parse(String(value || ""));
        return Number.isFinite(parsed) ? parsed : null;
    }

    function setImportedSetting(settings, key, value) {
        if (value === null || value === undefined || value === "") return;
        settings[key] = cloneJson(value, value);
    }

    function normalizeImportedExport(payload, importedAt = Date.now()) {
        if (!payload || typeof payload !== "object" || Array.isArray(payload) || !Array.isArray(payload.runs)) {
            throw new Error("This is not a Status Pro JSON export: the runs list is missing.");
        }
        if (!payload.exported_at && !payload.version) {
            throw new Error("This JSON file does not contain Status Pro export metadata.");
        }
        if (!payload.runs.length) throw new Error("This Status Pro export contains no history records.");
        if (payload.runs.length > MAX_RUN_HISTORY) {
            throw new Error(`This export contains ${payload.runs.length} records; Status Pro can import at most ${MAX_RUN_HISTORY} at once.`);
        }

        const safeImportedAt = Number.isFinite(optionalNumber(importedAt)) ? Number(importedAt) : Date.now();
        const sourceExportedAt = importedTimestamp(payload.exported_at);
        const sourceVersion = String(payload.version || "Unknown").slice(0, 80);
        const sourceSession = `import-${sourceExportedAt || safeImportedAt}`;
        const seenIds = new Set();

        return payload.runs.map((source, index) => {
            if (!source || typeof source !== "object" || Array.isArray(source)) {
                throw new Error(`History record ${index + 1} is not a valid object.`);
            }
            const sourceId = source.run_id !== null && source.run_id !== undefined && source.run_id !== ""
                ? source.run_id
                : source.id;
            const rawId = sourceId === null || sourceId === undefined || sourceId === ""
                ? `${sourceSession}-run-${index + 1}`
                : String(sourceId).slice(0, 500);
            if (seenIds.has(rawId)) throw new Error(`History record ${index + 1} repeats run ID “${rawId}”.`);
            seenIds.add(rawId);

            const settings = source.settings && typeof source.settings === "object" && !Array.isArray(source.settings)
                ? cloneJson(source.settings, {})
                : {};
            setImportedSetting(settings, "generation_time", source.generation_time);
            setImportedSetting(settings, "model_name", source.model_name);
            setImportedSetting(settings, "model_filename", source.checkpoint);
            setImportedSetting(settings, "resolution", source.resolution);
            setImportedSetting(settings, "num_inference_steps", source.steps);
            setImportedSetting(settings, "force_fps", source.fps);
            setImportedSetting(settings, "seed", source.seed);
            setImportedSetting(settings, "guidance_scale", source.guidance);
            setImportedSetting(settings, "guidance2_scale", source.guidance2);
            setImportedSetting(settings, "guidance3_scale", source.guidance3);
            setImportedSetting(settings, "flow_shift", source.flow_shift);
            setImportedSetting(settings, "sample_solver", source.sampler);
            setImportedSetting(settings, "activated_loras", source.loras);
            setImportedSetting(settings, "prompt", source.prompt);
            setImportedSetting(settings, "negative_prompt", source.negative_prompt);

            const skipping = source.step_skipping && typeof source.step_skipping === "object" && !Array.isArray(source.step_skipping)
                ? source.step_skipping
                : {};
            setImportedSetting(settings, "skip_steps_cache_type", skipping.method);
            setImportedSetting(settings, "skip_steps_multiplier", skipping.multiplier_or_threshold);
            setImportedSetting(settings, "skip_steps_start_step_perc", skipping.start_percent);

            const outputRecords = Array.isArray(source.output_records)
                ? source.output_records.filter(record => record && typeof record === "object" && !Array.isArray(record)).map(record => cloneJson(record, {}))
                : [];
            const outputs = Array.isArray(source.outputs)
                ? source.outputs.map(value => String(value || "")).filter(Boolean)
                : outputRecords.map(record => String(record.path || "")).filter(Boolean);
            const startedAt = importedTimestamp(source.started_at);
            const completedAt = importedTimestamp(source.completed_at);
            const statusValue = String(source.status || "completed").toLowerCase();
            const status = ["completed", "window", "aborted", "failed", "incomplete"].includes(statusValue)
                ? statusValue
                : "completed";
            const legacyStepSummary = source.step_summary && typeof source.step_summary === "object" && !Array.isArray(source.step_summary)
                ? cloneJson(source.step_summary, null)
                : null;
            const passSummaries = Array.isArray(skipping.pass_summaries) ? cloneJson(skipping.pass_summaries, []) : [];
            const recordedSteps = optionalNumber(skipping.step_observations) || optionalNumber(skipping.recorded_steps);
            const stepSummary = legacyStepSummary || (recordedSteps !== null || passSummaries.length
                ? {
                    recorded_steps: recordedSteps,
                    observed_passes: optionalNumber(skipping.observed_passes) || passSummaries.length || null,
                    passes: passSummaries,
                    skipped_steps: optionalNumber(skipping.recorded_skipped_steps),
                    average_seconds: null,
                    fastest_seconds: null,
                    slowest_seconds: null,
                    truncated: false
                }
                : null);

            const firstOutputSettings = outputRecords[0] && outputRecords[0].settings && typeof outputRecords[0].settings === "object"
                ? outputRecords[0].settings
                : {};
            const run = {
                id: rawId,
                session_id: String(source.session_id || sourceSession).slice(0, 500),
                queue_task_id: source.queue_task_id === undefined ? null : source.queue_task_id,
                status,
                started_at: startedAt,
                completed_at: completedAt,
                duration_seconds: optionalNumber(source.duration_seconds),
                settings,
                stages: (source.phase_timings || source.stages) && typeof (source.phase_timings || source.stages) === "object" && !Array.isArray(source.phase_timings || source.stages)
                    ? cloneJson(source.phase_timings || source.stages, {})
                    : {},
                step_performance: Array.isArray(source.step_performance)
                    ? cloneJson(source.step_performance.slice(-MAX_STEP_RECORDS), [])
                    : [],
                step_summary: stepSummary,
                resources: (source.resource_usage || source.resources) && typeof (source.resource_usage || source.resources) === "object" && !Array.isArray(source.resource_usage || source.resources)
                    ? cloneJson(source.resource_usage || source.resources, null)
                    : null,
                outputs,
                output_records: outputRecords,
                output_count: optionalNumber(source.output_count),
                media_type: String(source.media_type || "unknown").toLowerCase(),
                frame_count: optionalNumber(source.frame_count),
                repeats: optionalNumber(source.repeats) || 1,
                window_no: optionalNumber(source.window_no) || optionalNumber(setting(settings, "window_no")) || optionalNumber(firstOutputSettings.window_no),
                total_windows: optionalNumber(source.total_windows) || optionalNumber(setting(settings, "total_windows")) || optionalNumber(firstOutputSettings.total_windows),
                status_reason: source.outcome !== null && source.outcome !== undefined
                    ? String(source.outcome).slice(0, 2000)
                    : (source.status_reason === null || source.status_reason === undefined ? null : String(source.status_reason).slice(0, 2000)),
                failure_reason: source.failure_reason !== null && source.failure_reason !== undefined
                    ? String(source.failure_reason).slice(0, 2000)
                    : (status === "failed" && source.outcome !== null && source.outcome !== undefined ? String(source.outcome).slice(0, 2000) : null),
                imported: true,
                imported_at: safeImportedAt,
                imported_model_summary: source.model_summary === null || source.model_summary === undefined
                    ? ""
                    : String(source.model_summary).slice(0, 500),
                import_source: {
                    exported_at: sourceExportedAt,
                    version: sourceVersion,
                    scope: payload.scope === undefined ? null : String(payload.scope).slice(0, 80),
                    preset: payload.preset === undefined ? null : String(payload.preset).slice(0, 200)
                }
            };
            return normalizeRunMedia(run);
        });
    }

    function importStatusProExport(namespace, payload, importedAt = Date.now()) {
        if (namespace.runHistory.length) {
            throw new Error("History must be empty before importing. Export anything you want to keep, clear history, then try again.");
        }
        const runs = normalizeImportedExport(payload, importedAt);
        namespace.recoverablePrompts.clear();
        runs.forEach(run => {
            if (namespace.promptMemory !== false) cacheRunPrompts(namespace, run);
            if (namespace.promptMemory === false || !modeStoresPrompts(namespace.historyPersistence)) stripRunPrompts(run);
        });
        namespace.runHistory = runs;
        namespace.sessionRunIds.clear();
        namespace.selectedRunIds.clear();
        namespace.openHistoryGroups.clear();
        namespace.openHistoryRuns.clear();
        namespace.visibleHistoryGroups.clear();
        namespace.galleryFeedback.clear();
        namespace.galleryRequests.clear();
        namespace.historyScope = "all";
        namespace.historyOpen = true;
        namespace.historyRenderKey = null;
        const result = persistRunHistory(namespace);
        return {
            requested: runs.length,
            imported: namespace.runHistory.length,
            dropped: result.dropped || 0,
            persisted: result.persisted
        };
    }

    function sameExportFields(left, right) {
        if (left.size !== right.size) return false;
        return Array.from(left).every(field => right.has(field));
    }

    function inferExportPreset(namespace, fields) {
        for (const [name, presetFields] of Object.entries(EXPORT_PRESETS)) {
            if (sameExportFields(fields, new Set(presetFields))) return name;
        }
        for (const preset of namespace.customExportPresets) {
            if (sameExportFields(fields, new Set(preset.fields))) return `custom:${preset.id}`;
        }
        return "custom:unsaved";
    }

    function exportPresetLabel(namespace, presetValue = namespace.exportPreset) {
        const builtIn = {
            standard: "Standard",
            performance: "Performance",
            reproducibility: "Reproducibility",
            "share-safe": "Share-safe"
        };
        if (builtIn[presetValue]) return builtIn[presetValue];
        if (String(presetValue).startsWith("custom:")) {
            const id = String(presetValue).slice(7);
            const preset = namespace.customExportPresets.find(candidate => candidate.id === id);
            return preset ? preset.name : "Custom (unsaved)";
        }
        return "Custom (unsaved)";
    }

    function modalSettings(namespace) {
        return namespace.settingsDraft || {
            historyRecording: namespace.historyRecording !== false,
            historyPersistence: namespace.historyPersistence,
            promptMemory: namespace.promptMemory !== false,
            exportFields: namespace.exportFields,
            exportPreset: namespace.exportPreset,
            exportFormat: namespace.exportFormat
        };
    }

    function ensureExportFieldControls(namespace) {
        const container = namespace.panel.querySelector("[data-sp-export-fields]");
        if (!container || container.childElementCount) return;
        const groups = new Map();
        EXPORT_FIELD_DEFS.forEach(field => {
            let group = groups.get(field.group);
            if (!group) {
                group = document.createElement("section");
                group.className = "status-pro__export-group";
                group.dataset.spExportGroup = field.group.toLowerCase();
                const heading = document.createElement("strong");
                heading.textContent = field.group;
                heading.title = EXPORT_GROUP_HELP[field.group] || field.group;
                group.appendChild(heading);
                groups.set(field.group, group);
                container.appendChild(group);
            }
            const label = document.createElement("label");
            label.className = "status-pro__export-field";
            label.title = EXPORT_FIELD_HELP[field.id] || field.label;
            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.dataset.spExportField = field.id;
            const caption = document.createElement("span");
            caption.textContent = field.label;
            label.append(checkbox, caption);
            group.appendChild(label);
        });
        const promptGroup = groups.get("Prompts");
        if (promptGroup) {
            const promptMemory = document.createElement("label");
            promptMemory.className = "status-pro__prompt-memory";
            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.dataset.spPromptMemory = "true";
            const caption = document.createElement("span");
            caption.textContent = "Remember prompts in this page until it closes. Prompt text is never added to longer-lived history.";
            promptMemory.title = "Keep prompt text in page memory so it can be selected for exports; it is discarded when the page closes.";
            promptMemory.append(checkbox, caption);
            promptGroup.insertBefore(promptMemory, promptGroup.children[1] || null);
        }
        const promptNote = namespace.panel.querySelector("[data-sp-export-prompt-note]");
        if (promptGroup && promptNote) promptGroup.appendChild(promptNote);
    }

    function availableExportPromptRuns(namespace, fieldId) {
        return exportSourceRuns(namespace).filter(run => exportFieldValue(run, fieldId, new Set([fieldId])) !== null).length;
    }

    function selectedExportFieldsFromModal(namespace) {
        return new Set(Array.from(namespace.panel.querySelectorAll("[data-sp-export-field]:checked:not(:disabled)"))
            .map(input => input.dataset.spExportField)
            .filter(field => EXPORT_FIELD_IDS.has(field)));
    }

    function setExportFields(namespace, fields, preset) {
        const settings = modalSettings(namespace);
        settings.exportFields = new Set(Array.from(fields).filter(field => EXPORT_FIELD_IDS.has(field)));
        settings.exportPreset = preset || inferExportPreset(namespace, settings.exportFields);
        renderExportModal(namespace);
    }

    function renderExportModal(namespace) {
        ensureExportFieldControls(namespace);
        const settings = modalSettings(namespace);
        const persistence = namespace.panel.querySelector("[data-sp-history-persistence]");
        if (persistence) persistence.value = settings.historyRecording ? settings.historyPersistence : "off";
        const promptMemory = namespace.panel.querySelector("[data-sp-prompt-memory]");
        if (promptMemory) {
            promptMemory.checked = settings.promptMemory;
            promptMemory.disabled = !settings.historyRecording;
            const promptMemoryLabel = promptMemory.closest("label");
            if (promptMemoryLabel) promptMemoryLabel.title = settings.historyRecording
                ? "Keep prompt text only in this page so it can be explicitly included in exports"
                : "Prompt memory is paused while automatic history recording is off";
        }
        const format = namespace.panel.querySelector("[data-sp-export-format]");
        if (format) format.value = normalizeExportFormat(settings.exportFormat);
        const runs = exportSourceRuns(namespace);
        namespace.panel.querySelectorAll("[data-sp-export-field]").forEach(input => {
            const field = EXPORT_FIELD_DEFS.find(candidate => candidate.id === input.dataset.spExportField);
            const available = !field.prompt || settings.promptMemory;
            input.disabled = !available;
            input.checked = available && settings.exportFields.has(field.id);
            const help = EXPORT_FIELD_HELP[field.id] || field.label;
            input.closest("label").title = available ? help : `${help} Enable page prompt memory to include it.`;
        });
        const selectedFields = selectedExportFieldsFromModal(namespace);
        text(namespace.panel, "[data-sp-export-scope]", `${selectedFields.size} export field${selectedFields.size === 1 ? "" : "s"} selected · ${normalizeExportFormat(settings.exportFormat).toUpperCase()} default`);

        const promptRuns = runs.filter(run => ["prompt", "negative_prompt"].some(field => exportFieldValue(run, field, new Set([field])) !== null)).length;
        const cachedRuns = exportScopeRuns(namespace).filter(run => namespace.recoverablePrompts.has(String(run.id))).length;
        const promptNote = namespace.panel.querySelector("[data-sp-export-prompt-note]");
        if (promptNote) {
            promptNote.dataset.state = !settings.promptMemory ? "unavailable" : (promptRuns === runs.length && runs.length ? "available" : (promptRuns ? "partial" : "unavailable"));
            promptNote.textContent = !settings.historyRecording
                ? "Automatic history recording is off. Prompt memory is paused; existing records remain available for review and export."
                : !settings.promptMemory
                ? "Prompt memory is off. Future prompts will not be retained for History exports; saving this change also removes prompt text currently held by Status Pro."
                : !runs.length
                    ? "Prompt memory is on. Prompts from future runs will remain available only while this page stays open."
                : !promptRuns
                    ? "Prompt memory is on, but no page-session prompts are available for the current history scope."
                    : !modeStoresPrompts(namespace.historyPersistence)
                        ? `Prompts are available for ${promptRuns} of ${runs.length} run${runs.length === 1 ? "" : "s"} from this page session${cachedRuns ? ` (${cachedRuns} held in page memory)` : ""}; they are not saved with retained history.`
                        : `Prompts are available for ${promptRuns} of ${runs.length} run${runs.length === 1 ? "" : "s"} and will clear when this browser-tab session ends.`;
        }

        const preset = namespace.panel.querySelector("[data-sp-export-preset]");
        if (preset) {
            preset.querySelectorAll("[data-sp-custom-preset]").forEach(option => option.remove());
            const oldGroup = preset.querySelector("[data-sp-custom-preset-group]");
            if (oldGroup) oldGroup.remove();
            if (namespace.customExportPresets.length) {
                const group = document.createElement("optgroup");
                group.label = "Custom presets";
                group.dataset.spCustomPresetGroup = "true";
                namespace.customExportPresets.forEach(customPreset => {
                    const option = document.createElement("option");
                    option.value = `custom:${customPreset.id}`;
                    option.textContent = customPreset.name;
                    option.dataset.spCustomPreset = "true";
                    group.appendChild(option);
                });
                preset.appendChild(group);
            }
            const knownCustom = namespace.customExportPresets.some(candidate => `custom:${candidate.id}` === settings.exportPreset);
            if (settings.exportPreset === "custom:unsaved" || (String(settings.exportPreset).startsWith("custom:") && !knownCustom)) {
                const option = document.createElement("option");
                option.value = "custom:unsaved";
                option.textContent = "Custom (unsaved)";
                option.dataset.spCustomPreset = "true";
                preset.appendChild(option);
                settings.exportPreset = "custom:unsaved";
            }
            preset.value = settings.exportPreset;
        }
        const deletePreset = namespace.panel.querySelector("[data-sp-export-delete-preset]");
        if (deletePreset) deletePreset.hidden = !String(settings.exportPreset).startsWith("custom:") || settings.exportPreset === "custom:unsaved";
        const confirm = namespace.panel.querySelector("[data-sp-export-confirm]");
        if (confirm) {
            confirm.disabled = !selectedFields.size;
            confirm.textContent = "Save settings";
        }
    }

    function positionExportModal(namespace, left, top) {
        const modal = namespace.panel.querySelector("[data-sp-export-modal]");
        if (!modal) return;
        const rect = modal.getBoundingClientRect();
        const margin = 8;
        const maxLeft = Math.max(margin, window.innerWidth - rect.width - margin);
        const maxTop = Math.max(margin, window.innerHeight - rect.height - margin);
        modal.style.margin = "0";
        modal.style.left = `${clamp(left, margin, maxLeft)}px`;
        modal.style.top = `${clamp(top, margin, maxTop)}px`;
        modal.style.right = "auto";
        modal.style.bottom = "auto";
        modal.dataset.positioned = "true";
    }

    function clampExportModalToViewport(namespace) {
        const modal = namespace.panel.querySelector("[data-sp-export-modal]");
        if (!modal || !modal.open) return;
        const rect = modal.getBoundingClientRect();
        positionExportModal(namespace, rect.left, rect.top);
    }

    function openExportModal(namespace) {
        const modal = namespace.panel.querySelector("[data-sp-export-modal]");
        if (!modal) return;
        namespace.settingsDraft = {
            historyRecording: namespace.historyRecording !== false,
            historyPersistence: namespace.historyPersistence,
            promptMemory: namespace.promptMemory !== false,
            exportFields: new Set(namespace.exportFields),
            exportPreset: namespace.exportPreset,
            exportFormat: namespace.exportFormat
        };
        renderExportModal(namespace);
        if (typeof modal.showModal === "function") modal.showModal();
        else modal.setAttribute("open", "");
        const rect = modal.getBoundingClientRect();
        if (modal.dataset.positioned === "true") positionExportModal(namespace, rect.left, rect.top);
        else positionExportModal(namespace, (window.innerWidth - rect.width) / 2, (window.innerHeight - rect.height) / 2);
    }

    function closeExportModal(namespace) {
        const modal = namespace.panel.querySelector("[data-sp-export-modal]");
        if (!modal) return;
        namespace.settingsDraft = null;
        if (typeof modal.close === "function" && modal.open) modal.close();
        else modal.removeAttribute("open");
    }

    function restoreHistoryDrawer(namespace) {
        const drawer = namespace.panel.querySelector("[data-sp-history-drawer]");
        const home = namespace.panel.querySelector("[data-sp-history-home]");
        if (drawer && home) home.after(drawer);
        if (drawer) drawer.removeAttribute("data-expanded");
        namespace.historyExpanded = false;
        renderHistoryDrawer(namespace);
    }

    function openHistoryModal(namespace) {
        const modal = namespace.panel.querySelector("[data-sp-history-modal]");
        const content = namespace.panel.querySelector("[data-sp-history-modal-content]");
        const drawer = namespace.panel.querySelector("[data-sp-history-drawer]");
        if (!modal || !content || !drawer || namespace.historyExpanded) return;
        namespace.historyOpen = true;
        namespace.historyExpanded = true;
        drawer.dataset.expanded = "true";
        drawer.hidden = false;
        content.appendChild(drawer);
        if (typeof modal.showModal === "function") modal.showModal();
        else modal.setAttribute("open", "");
        renderHistoryDrawer(namespace);
        const close = modal.querySelector("[data-sp-history-modal-close]");
        if (close) close.focus({ preventScroll: true });
    }

    function closeHistoryModal(namespace) {
        const modal = namespace.panel.querySelector("[data-sp-history-modal]");
        if (!modal) return;
        if (typeof modal.close === "function" && modal.open) modal.close();
        else {
            modal.removeAttribute("open");
            restoreHistoryDrawer(namespace);
        }
    }

    function renderHistoryDrawer(namespace) {
        const drawer = namespace.panel.querySelector("[data-sp-history-drawer]");
        const toggle = namespace.panel.querySelector("[data-sp-history-toggle]");
        if (!drawer || !toggle) return;
        const sessionRuns = namespace.runHistory.filter(run => namespace.sessionRunIds.has(run.id));
        const sessionCount = groupHistoryRuns(sessionRuns).length;
        const allCount = groupHistoryRuns(namespace.runHistory).length;
        const scopedRuns = historyRuns(namespace);
        const selectedCount = selectedHistoryRuns(namespace).length;
        drawer.hidden = !namespace.historyOpen;
        toggle.setAttribute("aria-expanded", namespace.historyOpen ? "true" : "false");
        toggle.setAttribute("data-recording", namespace.historyRecording === false ? "off" : "on");
        toggle.title = `${namespace.historyOpen ? "Hide" : "Show"} generation history · ${namespace.historyRecording === false ? "automatic recording off · " : ""}${allCount} task${allCount === 1 ? "" : "s"}, ${namespace.runHistory.length} recorded run${namespace.runHistory.length === 1 ? "" : "s"}`;
        text(toggle, "[data-sp-history-label]", namespace.historyRecording === false ? "History off" : "History");
        text(toggle, "[data-sp-history-count]", allCount);
        const modalSummary = namespace.panel.querySelector("[data-sp-history-modal-summary]");
        if (modalSummary) {
            modalSummary.textContent = `${namespace.historyRecording === false ? "Automatic recording off · " : ""}${allCount} task${allCount === 1 ? "" : "s"} · ${namespace.runHistory.length} recorded run${namespace.runHistory.length === 1 ? "" : "s"}`;
        }
        namespace.panel.querySelectorAll("[data-sp-history-scope]").forEach(button => {
            const selected = button.dataset.spHistoryScope === namespace.historyScope;
            button.setAttribute("aria-pressed", selected ? "true" : "false");
            button.textContent = button.dataset.spHistoryScope === "session"
                ? `This session (${sessionCount})`
                : `All history (${allCount})`;
        });
        const settingsButton = namespace.panel.querySelector("[data-sp-settings-button]");
        if (settingsButton) {
            settingsButton.setAttribute("aria-label", "History settings");
            settingsButton.title = "History settings";
        }
        const importButton = namespace.panel.querySelector("[data-sp-import-button]");
        if (importButton) {
            importButton.title = namespace.runHistory.length
                ? "History must be cleared before importing a Status Pro JSON export"
                : "Import a Status Pro JSON export into empty history";
        }
        const expandButton = namespace.panel.querySelector("[data-sp-history-expand]");
        if (expandButton) {
            expandButton.setAttribute("aria-expanded", namespace.historyExpanded ? "true" : "false");
            expandButton.setAttribute("aria-label", namespace.historyExpanded ? "Return to embedded history" : "Expand generation history");
            expandButton.title = namespace.historyExpanded ? "Return to embedded history" : "Expand generation history";
        }
        const exportButton = namespace.panel.querySelector("[data-sp-export-button]");
        if (exportButton) {
            exportButton.disabled = scopedRuns.length === 0 || !namespace.exportFields.size;
            exportButton.textContent = selectedCount ? `Export (${selectedCount})` : "Export";
            exportButton.title = `Export using ${exportPresetLabel(namespace)} as ${normalizeExportFormat(namespace.exportFormat).toUpperCase()}`;
        }
        const clearSelected = namespace.panel.querySelector("[data-sp-clear-selected]");
        if (clearSelected) clearSelected.disabled = selectedCount === 0;
        const selectAll = namespace.panel.querySelector("[data-sp-select-all-history]");
        if (selectAll) {
            const selectedInScope = scopedRuns.filter(run => namespace.selectedRunIds.has(String(run.id))).length;
            selectAll.disabled = scopedRuns.length === 0;
            selectAll.checked = scopedRuns.length > 0 && selectedInScope === scopedRuns.length;
            selectAll.indeterminate = selectedInScope > 0 && selectedInScope < scopedRuns.length;
        }
        const clear = namespace.panel.querySelector("[data-sp-clear-history]");
        if (clear) clear.disabled = namespace.runHistory.length === 0;
        const storageNote = namespace.panel.querySelector("[data-sp-history-storage-note]");
        if (storageNote) {
            const recordingNote = namespace.historyRecording === false
                ? "Automatic history recording is off. Existing records are unchanged."
                : "";
            storageNote.textContent = namespace.historyStorageNotice || recordingNote;
            storageNote.hidden = !(namespace.historyStorageNotice || recordingNote);
        }
        if (namespace.historyOpen) renderHistory(namespace);
    }

    function sessionCompletionSummary(sessionRuns) {
        const runs = Array.isArray(sessionRuns) ? sessionRuns : [];
        const latestTask = groupHistoryRuns(runs).map(historyTaskSummary).reduce((latestTask, task) => {
            if (!latestTask) return task;
            return (Number(task && task.completedAt) || 0) >= (Number(latestTask.completedAt) || 0) ? task : latestTask;
        }, null);
        return {
            generationCount: runs.reduce((sum, run) => sum + Math.max(1, Number(run && run.repeats) || 1), 0),
            totalDuration: runs.reduce((sum, run) => sum + (Number(run && run.duration_seconds) || 0), 0),
            latestFinishedAt: Number(latestTask && latestTask.completedAt) || 0,
            latestDuration: Number(latestTask && latestTask.duration) || 0
        };
    }

    function renderIdle(namespace) {
        const idle = namespace.panel.querySelector("[data-sp-idle]");
        const running = namespace.panel.querySelector("[data-sp-running]");
        if (!idle || !running) return;
        idle.hidden = false;
        running.hidden = true;
        const sessionRuns = namespace.runHistory.filter(run => namespace.sessionRunIds.has(run.id));
        const completed = namespace.historyRecording !== false && sessionRuns.length > 0;
        const {generationCount, totalDuration, latestFinishedAt, latestDuration} = sessionCompletionSummary(sessionRuns);
        text(namespace.panel, "[data-sp-live]", completed ? "Complete" : "Ready");
        text(namespace.panel, "[data-sp-steps]", "");
        text(namespace.panel, "[data-sp-overall]", completed ? `${formatDuration(latestDuration)} last run` : "");
        text(namespace.panel, "[data-sp-eta]", "");
        text(namespace.panel, "[data-sp-idle-title]", completed ? "All generations complete" : "Ready to generate");
        text(namespace.panel, "[data-sp-idle-message]", completed
            ? `${generationCount} generation${generationCount === 1 ? "" : "s"}${sessionRuns.length !== generationCount ? ` across ${sessionRuns.length} queued runs` : ""} completed in ${formatDuration(totalDuration)}. Most recent generation finished at ${formatClock(latestFinishedAt)}.`
            : namespace.historyRecording === false
                ? "Live generation timing will appear here. Automatic history recording is off."
            : (namespace.runHistory.length
                ? `${namespace.runHistory.length} saved generation${namespace.runHistory.length === 1 ? "" : "s"} available below.`
                : "Generation timing and settings will appear here after the first run."));
        renderHistoryDrawer(namespace);
    }

    function readDownloadSnapshot(namespace) {
        const field = namespace.container.querySelector(
            "#status-pro-download-bridge textarea, #status-pro-download-bridge input"
        );
        const raw = String(field ? field.value : "").trim();
        if (!raw || raw === namespace.downloadRaw) return namespace.download;
        namespace.downloadRaw = raw;
        try {
            const parsed = JSON.parse(raw);
            namespace.download = parsed && typeof parsed === "object" ? parsed : null;
        } catch (_) {
            namespace.download = null;
        }
        return namespace.download;
    }

    function downloadPanelVisible(namespace) {
        const download = namespace.download;
        if (!download || !download.visible || !Array.isArray(download.files) || !download.files.length) return false;
        return Boolean(download.active || namespace.state.currentId === "prepare");
    }

    function downloadFileStats(file) {
        const state = String(file.state || "pending");
        const downloaded = optionalNumber(file.downloaded);
        const total = optionalNumber(file.total);
        if (state === "failed") return String(file.error || "Download failed");
        if (state === "complete") return Number.isFinite(total) && total > 0 ? formatBytes(total) : "Complete";
        if (state === "pending") return Number.isFinite(total) && total > 0 ? `${formatBytes(total)} · Pending` : "Pending";
        const parts = [];
        if (Number.isFinite(downloaded)) {
            parts.push(Number.isFinite(total) && total > 0
                ? `${formatBytes(downloaded)} / ${formatBytes(total)}`
                : formatBytes(downloaded));
        }
        return parts.join(" · ") || "Starting…";
    }

    function downloadFreshnessText(file, nowSeconds) {
        const state = String(file.state || "pending");
        const startedAt = optionalNumber(file.started_at);
        const lastByteAt = optionalNumber(file.last_byte_at);
        const elapsed = Number.isFinite(startedAt) ? Math.max(0, nowSeconds - startedAt) : null;
        if (state === "pending") return "Waiting to start";
        if (state === "complete" || state === "failed") return "";
        const elapsedText = Number.isFinite(elapsed) ? `${formatDuration(elapsed)} elapsed` : "Starting…";
        if (!Number.isFinite(lastByteAt)) return `${elapsedText} · Connecting`;
        const quietFor = Math.max(0, nowSeconds - lastByteAt);
        if (quietFor < 3) return `${elapsedText} · Receiving data`;
        if (quietFor < 8) return `${elapsedText} · Last byte ${formatDuration(quietFor, true)} ago`;
        return `${elapsedText} · Waiting for next transfer update (${formatDuration(quietFor, true)} since last byte)`;
    }

    function downloadCycleText(file) {
        const state = String(file.state || "pending");
        if (state === "complete" || state === "failed" || state === "pending") return "";
        const cycles = optionalNumber(file.transfer_cycles) || 0;
        if (cycles < 1) return "ETA learning transfer pattern…";
        const parts = [`${cycles} transfer ${cycles === 1 ? "cycle" : "cycles"} observed`];
        const averageBytes = optionalNumber(file.cycle_average_bytes);
        const averageSeconds = optionalNumber(file.cycle_average_seconds);
        if (Number.isFinite(averageBytes) && Number.isFinite(averageSeconds)) {
            parts.push(`Avg cycle: ${formatBytes(averageBytes)} / ${formatDuration(averageSeconds, true)}`);
        }
        const effectiveRate = optionalNumber(file.effective_rate);
        if (Number.isFinite(effectiveRate) && effectiveRate > 0) parts.push(`Effective rate: ${formatRate(effectiveRate)}`);
        const eta = optionalNumber(file.effective_eta);
        if (Number.isFinite(eta) && eta > 0) parts.push(`Estimated remaining: ${formatDuration(eta, true)}`);
        return parts.join(" · ");
    }

    function downloadHeaderEta(download) {
        const files = Array.isArray(download && download.files) ? download.files : [];
        const remaining = files.filter(file => ["pending", "downloading", "retrying"].includes(String(file && file.state || "")));
        if (!remaining.length) return null;
        const estimates = remaining.map(file => optionalNumber(file && file.effective_eta));
        if (estimates.some(estimate => !Number.isFinite(estimate) || estimate < 0)) return null;
        return Math.max(...estimates);
    }

    function renderDownloads(namespace) {
        const panel = namespace.panel.querySelector("[data-sp-downloads]");
        const detail = namespace.panel.querySelector("[data-sp-detail]");
        if (!panel) return;
        const visible = downloadPanelVisible(namespace);
        panel.hidden = !visible;
        if (detail) detail.hidden = visible;
        if (!visible) return;

        const download = namespace.download;
        const totals = download.totals || {};
        const failed = Number(totals.failed || 0);
        const title = download.active
            ? "Downloading model files"
            : (failed > 0 ? "Download completed with errors" : "Downloads complete");
        text(panel, "[data-sp-download-title]", title);
        panel.dataset.active = download.active ? "true" : "false";

        const count = Number(totals.file_count || download.files.length || 0);
        const completed = Number(totals.completed || 0);
        const summary = [`${completed}/${count} files`];
        const knownDownloaded = optionalNumber(totals.known_downloaded);
        const knownTotal = optionalNumber(totals.known_total);
        if (Number.isFinite(knownDownloaded) && Number.isFinite(knownTotal) && knownTotal > 0) {
            summary.push(`${formatBytes(knownDownloaded)} / ${formatBytes(knownTotal)}`);
        } else if (optionalNumber(totals.downloaded) > 0) {
            summary.push(`${formatBytes(optionalNumber(totals.downloaded))} received`);
        }
        text(panel, "[data-sp-download-summary]", summary.join(" · "));

        const totalDetails = [];
        const transferStartedAt = optionalNumber(download.started_at);
        if (download.active && Number.isFinite(transferStartedAt)) {
            totalDetails.push(`${formatDuration(Math.max(0, Date.now() / 1000 - transferStartedAt))} elapsed`);
        }
        text(panel, "[data-sp-download-total]", totalDetails.join(" · "));

        const overallFill = panel.querySelector("[data-sp-download-overall-fill]");
        if (overallFill) {
            const percent = Number.isFinite(knownDownloaded) && Number.isFinite(knownTotal) && knownTotal > 0
                ? clamp(knownDownloaded / knownTotal * 100, 0, 100)
                : (download.active ? 0 : 100);
            overallFill.style.width = `${percent}%`;
        }

        const stateOrder = { downloading: 0, retrying: 1, failed: 2, pending: 3, complete: 4 };
        const files = download.files.slice().sort((left, right) =>
            (stateOrder[left.state] ?? 9) - (stateOrder[right.state] ?? 9)
        );
        const nowSeconds = Date.now() / 1000;
        const fileContainer = panel.querySelector("[data-sp-download-files]");
        const fragment = document.createDocumentFragment();
        files.forEach(file => {
            const row = document.createElement("div");
            const state = String(file.state || "pending");
            row.className = "status-pro__download-file";
            row.dataset.state = state;
            const icon = document.createElement("span");
            icon.className = "status-pro__download-file-icon";
            icon.setAttribute("aria-hidden", "true");
            icon.textContent = state === "complete" ? "✓" : (state === "failed" ? "!" : (state === "downloading" || state === "retrying" ? "↓" : "○"));
            const name = document.createElement("span");
            name.className = "status-pro__download-file-name";
            name.textContent = String(file.name || "Unknown file");
            name.title = String(file.path || file.name || "");
            const stats = document.createElement("span");
            stats.className = "status-pro__download-file-stats";
            stats.textContent = downloadFileStats(file);
            stats.title = String(file.error || "");
            const freshness = document.createElement("span");
            freshness.className = "status-pro__download-file-freshness";
            freshness.textContent = downloadFreshnessText(file, nowSeconds);
            const cycles = document.createElement("span");
            cycles.className = "status-pro__download-file-cycles";
            cycles.textContent = downloadCycleText(file);
            const bar = document.createElement("span");
            bar.className = "status-pro__download-file-bar";
            const fill = document.createElement("span");
            const fileTotal = optionalNumber(file.total);
            const fileDownloaded = optionalNumber(file.downloaded);
            const filePercent = state === "complete"
                ? 100
                : (Number.isFinite(fileTotal) && fileTotal > 0 && Number.isFinite(fileDownloaded)
                    ? clamp(fileDownloaded / fileTotal * 100, 0, 100)
                    : 0);
            fill.style.width = `${filePercent}%`;
            bar.appendChild(fill);
            row.append(icon, name, stats, freshness, cycles, bar);
            fragment.appendChild(row);
        });
        fileContainer.replaceChildren(fragment);
    }

    function stageActivities(state, record) {
        if (!record || record.id !== "input") return [];
        const seen = new Set();
        return state.phaseOrder
            .filter(id => {
                if (seen.has(id)) return false;
                seen.add(id);
                return true;
            })
            .map(id => state.phases[id])
            .filter(phase => phase && phase.stage === "input")
            .map(phase => ({
                label: phase.label,
                state: phase.state,
                elapsed: phase.elapsed
            }));
    }

    function detailMessage(record, hasActivities = false) {
        if (record.preloaded) return record.rawMessage || "The required model was already loaded and ready for this run.";
        if (record.unreported) return record.rawMessage || "Wan2GP did not expose a separately measurable status for this stage.";
        if (record.state === "complete") {
            if (record.id === "input" && hasActivities) return "";
            return Number.isFinite(record.elapsed) ? `Completed in ${formatDuration(record.elapsed)}.` : "Completed.";
        }
        if (record.state === "aborting") return record.rawMessage || "Wan2GP is stopping the current run.";
        if (/^unload/.test(String(record.activity || ""))) return record.rawMessage;
        if (record.id === "decode" && record.state === "current") {
            return "Decoding is running. Wan2GP does not report intermediate VAE progress for this model.";
        }
        let message = String(record.rawMessage || "").trim();
        const rawName = String(record.rawName || "").trim();
        if (rawName && message.toLowerCase().startsWith(rawName.toLowerCase())) {
            message = message.slice(rawName.length).replace(/^[\s|:\-–—]+/, "").trim();
        }
        message = message.replace(/\s+-\s+/g, " · ");
        if (message) return message;
        if (record.id === "input" && hasActivities) return "";
        if (record.state === "pending") return "This stage has not started.";
        return "Live stage timing.";
    }

    function renderStages(namespace) {
        const state = namespace.state;
        const container = namespace.panel.querySelector("[data-sp-stages]");
        const existing = new Map(Array.from(container.querySelectorAll("[data-stage-id]")).map(button => [button.dataset.stageId, button]));

        const visibleDefs = STAGE_DEFS.filter(def => state.records[def.id].visible);
        container.dataset.stageCount = String(visibleDefs.length);
        visibleDefs.forEach((def, visibleIndex) => {
            const record = state.records[def.id];
            let button = existing.get(def.id);
            if (!button) {
                button = document.createElement("button");
                button.type = "button";
                button.className = "status-pro__stage";
                button.dataset.stageId = def.id;
                button.setAttribute("role", "tab");
                const icon = document.createElement("span");
                icon.className = "status-pro__stage-icon";
                icon.setAttribute("aria-hidden", "true");
                const name = document.createElement("span");
                name.className = "status-pro__stage-name";
                const timing = document.createElement("span");
                timing.className = "status-pro__stage-time";
                button.append(icon, name, timing);
            }
            const selected = state.selectedId === def.id;
            button.classList.toggle("status-pro__stage--complete", record.state === "complete");
            const active = record.state === "current" || record.state === "aborting";
            button.classList.toggle("status-pro__stage--current", active);
            button.classList.toggle("status-pro__stage--selected", selected);
            button.setAttribute("aria-selected", selected ? "true" : "false");
            const label = stageDisplayLabel(namespace, record);
            const modelInfo = stageModelInfo(namespace, record);
            const modelSummary = modelInfo ? `, ${modelInfo.text}` : "";
            const accessibleLabel = `${record.rawName || label}: ${statusLabel(record)}, ${stageTimeText(state, record)}${modelSummary}`;
            button.setAttribute("aria-label", accessibleLabel);
            button.title = accessibleLabel;
            button.querySelector(".status-pro__stage-icon").textContent = record.state === "complete" ? "✓" : (record.state === "aborting" ? "!" : (record.state === "current" ? "●" : String(visibleIndex + 1)));
            button.querySelector(".status-pro__stage-name").textContent = label;
            button.querySelector(".status-pro__stage-time").textContent = stageTimeText(state, record);
            const buttonAtIndex = container.children[visibleIndex] || null;
            if (buttonAtIndex !== button) container.insertBefore(button, buttonAtIndex);
        });
        existing.forEach((button, id) => {
            if (!state.records[id] || !state.records[id].visible) button.remove();
        });

        // A fixed breakpoint made four-card runs stack unnecessarily on half-width
        // layouts. Use the actual card count so inline contents are retained whenever
        // every card can keep its intended flex basis without crowding.
        const inlineWidth = visibleDefs.length
            ? (visibleDefs.length * 150) + 80 + (Math.max(0, visibleDefs.length - 1) * 7)
            : 0;
        container.classList.toggle("status-pro__stages--inline", container.clientWidth >= inlineWidth);
    }

    function renderDetail(namespace) {
        const state = namespace.state;
        const selected = state.records[state.selectedId] || state.records[state.currentId] || state.records.prepare;
        const activities = stageActivities(state, selected);
        let etaText = "—";
        const remaining = remainingEstimate(state, selected);
        if (selected.state === "current" && Number.isFinite(remaining)) etaText = formatDuration(remaining, true);
        else if (selected.state === "current" && stageSupportsEta(selected)) etaText = "Calculating…";

        text(namespace.panel, "[data-sp-detail-name]", stageDisplayLabel(namespace, selected));
        const activityElement = namespace.panel.querySelector("[data-sp-detail-activities]");
        if (activityElement) {
            activityElement.hidden = activities.length === 0;
            activityElement.replaceChildren();
            activities.forEach(activity => {
                const line = document.createElement("span");
                line.className = "status-pro__detail-activity-line";
                const active = activity.state === "current" || activity.state === "aborting";
                const icon = activity.state === "complete" ? "✓" : (activity.state === "aborting" ? "!" : "●");
                const timing = Number.isFinite(activity.elapsed)
                    ? `${formatDuration(activity.elapsed)}${active ? " elapsed" : ""}`
                    : (active ? "Running" : "Completed");
                line.textContent = `${icon} ${activity.label} · ${timing}`;
                activityElement.appendChild(line);
            });
        }
        const detail = detailMessage(selected, activities.length > 0);
        const detailElement = namespace.panel.querySelector("[data-sp-detail-message]");
        if (detailElement) {
            detailElement.hidden = !detail;
            detailElement.textContent = detail;
        }
        const modelInfo = stageModelInfo(namespace, selected);
        const modelElement = namespace.panel.querySelector("[data-sp-detail-model]");
        if (modelElement) {
            modelElement.hidden = !modelInfo;
            modelElement.replaceChildren();
            modelElement.classList.toggle("status-pro__detail-model--list", Boolean(modelInfo && modelInfo.names.length > 1));
            if (modelInfo && modelInfo.names.length > 1) {
                modelElement.setAttribute("role", "list");
                for (const name of modelInfo.names) {
                    const line = document.createElement("span");
                    line.className = "status-pro__detail-model-line";
                    line.setAttribute("role", "listitem");
                    line.textContent = `${modelInfo.itemRole}: ${name}`;
                    line.title = name;
                    modelElement.append(line);
                }
            } else {
                modelElement.removeAttribute("role");
                modelElement.textContent = modelInfo ? modelInfo.text : "";
            }
            modelElement.title = modelInfo ? modelInfo.text : "";
        }
        text(namespace.panel, "[data-sp-detail-state]", statusLabel(selected));
        text(namespace.panel, "[data-sp-detail-elapsed]", Number.isFinite(selected.elapsed) ? formatDuration(selected.elapsed) : "—");
        text(namespace.panel, "[data-sp-detail-eta]", etaText);
        text(namespace.panel, "[data-sp-detail-progress]", Number.isFinite(selected.progress) ? `${selected.progress.toFixed(1)}%` : "—");
        const etaMetric = namespace.panel.querySelector("[data-sp-eta-metric]");
        if (etaMetric) etaMetric.hidden = selected.state !== "current" || !stageSupportsEta(selected);
        const progressMetric = namespace.panel.querySelector("[data-sp-progress-metric]");
        if (progressMetric) progressMetric.hidden = selected.id === "decode" || /^unload/.test(String(selected.activity || ""));
        const stepMetric = namespace.panel.querySelector("[data-sp-step-metric]");
        if (stepMetric) {
            const showStepTime = (selected.id === "denoise" || selected.id === "post") &&
                (Number.isFinite(selected.stepCurrent) || Number.isFinite(selected.stepTotal));
            stepMetric.hidden = !showStepTime;
            text(stepMetric, "[data-sp-detail-step-time]", Number.isFinite(selected.stepSeconds)
                ? `${formatLiveStepDuration(selected.stepSeconds)}/step`
                : "Calculating…");
        }
    }

    function applyCollapsed(namespace) {
        const body = namespace.panel.querySelector("[data-sp-body]");
        const button = namespace.panel.querySelector("[data-sp-collapse]");
        if (!body || !button) return;
        body.hidden = namespace.collapsed;
        namespace.panel.classList.toggle("status-pro--collapsed", namespace.collapsed);
        button.setAttribute("aria-expanded", namespace.collapsed ? "false" : "true");
        button.setAttribute("aria-label", namespace.collapsed ? "Expand Status Pro" : "Collapse Status Pro");
        button.title = namespace.collapsed ? "Expand Status Pro" : "Collapse Status Pro";
        button.textContent = namespace.collapsed ? "◀" : "▼";
    }

    function render(namespace) {
        const state = namespace.state;
        const current = state.records[state.currentId];
        const downloading = namespace.download && namespace.download.active;
        const idle = namespace.panel.querySelector("[data-sp-idle]");
        const running = namespace.panel.querySelector("[data-sp-running]");
        if (idle) idle.hidden = true;
        if (running) running.hidden = false;
        text(namespace.panel, "[data-sp-live]", downloading ? "Downloading model files" : (current ? (current.rawName || current.label) : "Waiting for progress"));
        text(namespace.panel, "[data-sp-steps]", Number.isFinite(state.steps.current) && Number.isFinite(state.steps.total) ? `${state.steps.current}/${state.steps.total} steps` : "");
        text(namespace.panel, "[data-sp-overall]", Number.isFinite(state.overallElapsed) ? `${formatDuration(state.overallElapsed)} elapsed` : "");
        const downloadEta = downloading ? downloadHeaderEta(namespace.download) : null;
        const eta = downloading ? downloadEta : totalEta(state);
        const showEta = downloading ? Number.isFinite(downloadEta) : stageSupportsEta(current);
        text(namespace.panel, "[data-sp-eta]", showEta
            ? (downloading ? `~${formatDuration(eta)} transfer ETA` : `${formatDuration(eta, true)} ETA`)
            : "");
        const overallFill = namespace.panel.querySelector("[data-sp-overall-fill]");
        if (overallFill) {
            const stageIsIndeterminate = current && current.state === "current" &&
                (current.id === "decode" || current.activity === "unload");
            overallFill.classList.toggle("status-pro__overall-fill--indeterminate", Boolean(stageIsIndeterminate));
            overallFill.style.width = !stageIsIndeterminate && current && Number.isFinite(current.progress)
                ? `${current.progress}%`
                : "0%";
        }
        renderStages(namespace);
        renderDetail(namespace);
        renderDownloads(namespace);
        renderHistoryDrawer(namespace);
    }

    function setActive(namespace, active) {
        namespace.container.classList.toggle("status-pro-container--active", active);
        namespace.host.classList.toggle("status-pro-host--active", active);
        namespace.source.classList.toggle("status-pro-source--active", active);
        namespace.panel.hidden = !active;
    }

    function tick(namespace) {
        if (!namespace.host.isConnected || !namespace.source.isConnected) return;
        observeRunOutcome(namespace, visibleFailureNotice(namespace));
        syncRunTelemetry(namespace);
        observeRunOutcome(namespace, visibleFailureNotice(namespace));
        readGalleryNavigationResult(namespace);
        readDownloadSnapshot(namespace);
        let snapshot = readPrepareStatus(namespace) || readReportedPreGenerationStatus(namespace) || readSnapshot(namespace);
        snapshot = reinterpretQwenSilentEncode(namespace, snapshot);
        if (!snapshot && namespace.download && namespace.download.visible &&
            (namespace.download.active || namespace.state.currentId === "prepare")) {
            snapshot = {
                id: "prepare",
                rawName: namespace.download.active ? "Downloading model files" : "Downloads complete",
                rawMessage: namespace.download.active ? "Downloading required model assets" : "Required model assets downloaded",
                metaText: "",
                stageElapsed: null,
                overallElapsed: namespace.state.overallElapsed,
                progress: null,
                steps: namespace.state.steps,
                aborting: false,
                textOnly: true
            };
        }
        if (snapshot) {
            observeRunOutcome(namespace, snapshot.rawName, snapshot.rawMessage);
            applySnapshot(namespace, snapshot);
            setActive(namespace, true);
            render(namespace);
            return;
        }
        const now = Date.now();
        if (!namespace.state.inactiveSince) namespace.state.inactiveSince = now;
        if (now - namespace.state.inactiveSince >= RESET_AFTER_MS) {
            finishStage(namespace.state, namespace.state.currentId);
        }
        setActive(namespace, true);
        if (namespace.activeRun && now - namespace.state.inactiveSince < RESET_AFTER_MS) render(namespace);
        else renderIdle(namespace);
    }

    function installStyle(root) {
        if (root.querySelector("#status-pro-styles")) return;
        const style = document.createElement("style");
        style.id = "status-pro-styles";
        style.textContent = STYLE_TEXT;
        root.appendChild(style);
    }

    function bind() {
        const root = appRoot();
        const container = root.querySelector("#status-pro-container");
        const host = root.querySelector("#status-pro-host");
        if (!container || !host) return false;
        const panel = host.querySelector("[data-status-pro]");
        const source = container.previousElementSibling;
        if (!panel || !source) return false;

        installStyle(root);
        const previous = window[NAMESPACE];
        if (previous && previous.timer) window.clearInterval(previous.timer);
        if (previous && previous.exportResizeHandler) window.removeEventListener("resize", previous.exportResizeHandler);
        if (previous && previous.historyExpanded) closeHistoryModal(previous);
        const customExportPresets = loadCustomExportPresets();
        const savedExportSettings = loadExportSettings();
        const promptMemory = loadPromptMemoryPreference();
        const historyRecording = loadHistoryRecordingPreference();
        const initialRunBridge = parseRunBridge(container);
        const runtimeId = runtimeIdFromTelemetry(initialRunBridge.telemetry);
        const historyPersistence = loadHistoryPersistence(runtimeId);
        const runHistory = loadRunHistory(historyPersistence, runtimeId)
            .map(normalizeRunMedia)
            .map(run => promptMemory ? run : stripRunPrompts(run));

        const namespace = {
            root,
            container,
            host,
            panel,
            source,
            tracker: null,
            state: freshState(),
            download: null,
            downloadRaw: "",
            runTelemetry: initialRunBridge.telemetry,
            runRaw: initialRunBridge.raw,
            qwenEncodeFallbackStartedAt: 0,
            activeRun: null,
            runHistory,
            historyPersistence,
            historyRecording,
            promptMemory,
            runtimeId,
            historyStorageNotice: "",
            sessionId: (window.crypto && window.crypto.randomUUID)
                ? window.crypto.randomUUID()
                : `session-${Date.now()}-${Math.random().toString(16).slice(2)}`,
            sessionRunIds: new Set(historyPersistence === "persistent" ? [] : runHistory.map(run => run.id)),
            lastCompletedAt: null,
            recoverablePrompts: new Map(),
            historyRenderKey: null,
            openHistoryGroups: new Set(),
            openHistoryRuns: new Set(),
            visibleHistoryGroups: new Map(),
            historyOpen: false,
            historyExpanded: false,
            historyScope: "all",
            selectedRunIds: new Set(),
            galleryFeedback: new Map(),
            galleryRequests: new Map(),
            galleryResultRaw: "",
            customExportPresets,
            exportFields: new Set(savedExportSettings.fields),
            exportPreset: savedExportSettings.preset,
            exportFormat: savedExportSettings.format,
            settingsDraft: null,
            exportDrag: null,
            exportResizeHandler: null,
            collapsed: loadCollapsedPreference(),
            timer: null
        };
        if (!promptMemory) {
            namespace.exportFields.delete("prompt");
            namespace.exportFields.delete("negative_prompt");
        }
        namespace.exportPreset = inferExportPreset(namespace, namespace.exportFields);
        window[NAMESPACE] = namespace;
        if (!promptMemory) persistRunHistory(namespace);

        panel.querySelector("[data-sp-stages]").addEventListener("click", event => {
            const button = event.target.closest("[data-stage-id]");
            if (!button || !panel.contains(button)) return;
            namespace.state.selectedId = button.dataset.stageId;
            namespace.state.selectionIsManual = true;
            render(namespace);
        });

        panel.querySelector("[data-sp-collapse]").addEventListener("click", () => {
            namespace.collapsed = !namespace.collapsed;
            saveCollapsedPreference(namespace.collapsed);
            applyCollapsed(namespace);
        });

        panel.querySelector("[data-sp-history-toggle]").addEventListener("click", () => {
            if (namespace.collapsed) {
                namespace.collapsed = false;
                namespace.historyOpen = true;
                saveCollapsedPreference(false);
                applyCollapsed(namespace);
            } else {
                namespace.historyOpen = !namespace.historyOpen;
            }
            renderHistoryDrawer(namespace);
        });

        const historyExpand = panel.querySelector("[data-sp-history-expand]");
        if (historyExpand) {
            historyExpand.addEventListener("click", () => {
                if (namespace.historyExpanded) closeHistoryModal(namespace);
                else openHistoryModal(namespace);
            });
        }
        const historyModal = panel.querySelector("[data-sp-history-modal]");
        if (historyModal) {
            historyModal.addEventListener("click", event => {
                if (event.target === historyModal) closeHistoryModal(namespace);
            });
            historyModal.addEventListener("cancel", event => {
                event.preventDefault();
                closeHistoryModal(namespace);
            });
            historyModal.addEventListener("close", () => restoreHistoryDrawer(namespace));
        }
        const historyModalClose = panel.querySelector("[data-sp-history-modal-close]");
        if (historyModalClose) historyModalClose.addEventListener("click", () => closeHistoryModal(namespace));

        panel.querySelectorAll("[data-sp-history-scope]").forEach(button => {
            button.addEventListener("click", () => {
                namespace.historyScope = button.dataset.spHistoryScope === "session" ? "session" : "all";
                namespace.historyRenderKey = null;
                renderHistoryDrawer(namespace);
            });
        });

        const history = panel.querySelector("[data-sp-history]");
        if (history) {
            history.addEventListener("click", event => {
                const importMedia = event.target.closest("[data-sp-import-media]");
                if (importMedia && history.contains(importMedia)) {
                    event.preventDefault();
                    event.stopPropagation();
                    const outputIndex = importMedia.dataset.spOutputIndex === undefined
                        ? null
                        : Number(importMedia.dataset.spOutputIndex);
                    requestGalleryImport(namespace, importMedia.dataset.spImportMedia, outputIndex);
                    return;
                }
                const viewOutput = event.target.closest("[data-sp-view-run]");
                if (viewOutput && history.contains(viewOutput)) {
                    event.preventDefault();
                    event.stopPropagation();
                    const outputIndex = viewOutput.dataset.spOutputIndex === undefined
                        ? null
                        : Number(viewOutput.dataset.spOutputIndex);
                    requestGalleryNavigation(namespace, viewOutput.dataset.spViewRun, outputIndex);
                    return;
                }
                if (event.target.closest("[data-sp-run-select], [data-sp-task-select]")) event.stopPropagation();
            });
            history.addEventListener("change", event => {
                const taskSelect = event.target.closest("[data-sp-task-select]");
                if (taskSelect && history.contains(taskSelect)) {
                    const group = namespace.visibleHistoryGroups.get(taskSelect.dataset.spTaskSelect);
                    if (!group) return;
                    group.runs.forEach(run => {
                        if (taskSelect.checked) namespace.selectedRunIds.add(String(run.id));
                        else namespace.selectedRunIds.delete(String(run.id));
                    });
                    namespace.historyRenderKey = null;
                    renderHistoryDrawer(namespace);
                    return;
                }
                const select = event.target.closest("[data-sp-run-select]");
                if (!select || !history.contains(select)) return;
                if (select.checked) namespace.selectedRunIds.add(select.dataset.spRunSelect);
                else namespace.selectedRunIds.delete(select.dataset.spRunSelect);
                namespace.historyRenderKey = null;
                renderHistoryDrawer(namespace);
            });
        }

        const selectAllHistory = panel.querySelector("[data-sp-select-all-history]");
        if (selectAllHistory) {
            selectAllHistory.addEventListener("change", () => {
                historyRuns(namespace).forEach(run => {
                    if (selectAllHistory.checked) namespace.selectedRunIds.add(String(run.id));
                    else namespace.selectedRunIds.delete(String(run.id));
                });
                namespace.historyRenderKey = null;
                renderHistoryDrawer(namespace);
            });
        }

        const settingsButton = panel.querySelector("[data-sp-settings-button]");
        if (settingsButton) settingsButton.addEventListener("click", () => openExportModal(namespace));
        const importButton = panel.querySelector("[data-sp-import-button]");
        const importFile = panel.querySelector("[data-sp-import-file]");
        if (importButton && importFile) {
            importButton.addEventListener("click", () => {
                if (namespace.runHistory.length) {
                    window.alert("History must be empty before importing.\n\nExport any current history you want to keep, then use Clear history and try Import again.");
                    return;
                }
                importFile.value = "";
                importFile.click();
            });
            importFile.addEventListener("change", async () => {
                const file = importFile.files && importFile.files[0];
                if (!file) return;
                try {
                    if (!String(file.name || "").toLowerCase().endsWith(".json")) {
                        throw new Error("Only Status Pro JSON exports can be imported.");
                    }
                    if (Number(file.size) > MAX_IMPORT_BYTES) {
                        throw new Error(`This JSON file is larger than ${Math.round(MAX_IMPORT_BYTES / 1024 / 1024)} MB and cannot be imported.`);
                    }
                    let payload;
                    try {
                        payload = JSON.parse(await file.text());
                    } catch (_) {
                        throw new Error("The selected file is not valid JSON.");
                    }
                    const result = importStatusProExport(namespace, payload);
                    if (namespace.activeRun) render(namespace);
                    else renderIdle(namespace);
                    const storageNote = result.persisted
                        ? ""
                        : " Browser storage was unavailable, so the imported history will last only until this page closes or reloads.";
                    const droppedNote = result.dropped
                        ? ` ${result.dropped} record${result.dropped === 1 ? " was" : "s were"} omitted because browser storage was full.`
                        : "";
                    window.alert(`Imported ${result.imported} history record${result.imported === 1 ? "" : "s"}.${droppedNote}${storageNote}`);
                } catch (error) {
                    window.alert(`Status Pro could not import this file.\n\n${error && error.message ? error.message : "The JSON export is not supported."}`);
                } finally {
                    importFile.value = "";
                }
            });
        }
        const exportButton = panel.querySelector("[data-sp-export-button]");
        if (exportButton) {
            exportButton.addEventListener("click", () => {
                exportHistory(namespace, namespace.exportFormat, namespace.exportFields);
            });
        }

        const historyPersistenceSelect = panel.querySelector("[data-sp-history-persistence]");
        if (historyPersistenceSelect) {
            historyPersistenceSelect.value = namespace.historyRecording === false ? "off" : namespace.historyPersistence;
            historyPersistenceSelect.addEventListener("change", () => {
                const settings = modalSettings(namespace);
                settings.historyRecording = historyPersistenceSelect.value !== "off";
                if (settings.historyRecording) settings.historyPersistence = normalizeHistoryPersistence(historyPersistenceSelect.value);
                renderExportModal(namespace);
            });
        }

        const exportModal = panel.querySelector("[data-sp-export-modal]");
        if (exportModal) {
            exportModal.addEventListener("click", event => {
                if (event.target === exportModal) closeExportModal(namespace);
            });
            exportModal.addEventListener("close", () => {
                namespace.settingsDraft = null;
            });
        }
        const exportInfo = panel.querySelector("[data-sp-export-info]");
        const exportGuide = panel.querySelector("[data-sp-export-guide]");
        if (exportInfo && exportGuide) {
            exportInfo.addEventListener("click", () => {
                const open = exportGuide.hidden;
                exportGuide.hidden = !open;
                exportInfo.setAttribute("aria-expanded", open ? "true" : "false");
                window.requestAnimationFrame(() => clampExportModalToViewport(namespace));
            });
        }
        const exportHeader = panel.querySelector(".status-pro__export-header");
        if (exportHeader) {
            exportHeader.addEventListener("pointerdown", event => {
                if (event.button !== 0 || event.target.closest("button, input, select, a, label")) return;
                const modal = panel.querySelector("[data-sp-export-modal]");
                if (!modal || !modal.open) return;
                const rect = modal.getBoundingClientRect();
                namespace.exportDrag = {
                    pointerId: event.pointerId,
                    offsetX: event.clientX - rect.left,
                    offsetY: event.clientY - rect.top
                };
                exportHeader.setPointerCapture(event.pointerId);
                exportHeader.style.cursor = "grabbing";
                event.preventDefault();
            });
            exportHeader.addEventListener("pointermove", event => {
                const drag = namespace.exportDrag;
                if (!drag || drag.pointerId !== event.pointerId) return;
                positionExportModal(namespace, event.clientX - drag.offsetX, event.clientY - drag.offsetY);
            });
            const stopExportDrag = event => {
                const drag = namespace.exportDrag;
                if (!drag || drag.pointerId !== event.pointerId) return;
                namespace.exportDrag = null;
                exportHeader.style.cursor = "move";
                if (exportHeader.hasPointerCapture(event.pointerId)) exportHeader.releasePointerCapture(event.pointerId);
            };
            exportHeader.addEventListener("pointerup", stopExportDrag);
            exportHeader.addEventListener("pointercancel", stopExportDrag);
        }
        namespace.exportResizeHandler = () => clampExportModalToViewport(namespace);
        window.addEventListener("resize", namespace.exportResizeHandler);
        panel.querySelectorAll("[data-sp-export-close], [data-sp-export-cancel]").forEach(button => {
            button.addEventListener("click", () => closeExportModal(namespace));
        });

        const exportFields = panel.querySelector("[data-sp-export-fields]");
        if (exportFields) {
            ensureExportFieldControls(namespace);
            exportFields.addEventListener("change", event => {
                if (!event.target.closest("[data-sp-export-field]")) return;
                const settings = modalSettings(namespace);
                settings.exportFields = selectedExportFieldsFromModal(namespace);
                settings.exportPreset = inferExportPreset(namespace, settings.exportFields);
                renderExportModal(namespace);
            });
        }

        const promptMemoryToggle = panel.querySelector("[data-sp-prompt-memory]");
        if (promptMemoryToggle) {
            promptMemoryToggle.addEventListener("change", () => {
                const settings = modalSettings(namespace);
                settings.promptMemory = promptMemoryToggle.checked;
                if (!settings.promptMemory) {
                    settings.exportFields.delete("prompt");
                    settings.exportFields.delete("negative_prompt");
                    settings.exportPreset = inferExportPreset(namespace, settings.exportFields);
                }
                renderExportModal(namespace);
            });
        }

        const exportFormat = panel.querySelector("[data-sp-export-format]");
        if (exportFormat) {
            exportFormat.addEventListener("change", () => {
                modalSettings(namespace).exportFormat = normalizeExportFormat(exportFormat.value);
                renderExportModal(namespace);
            });
        }

        const exportPreset = panel.querySelector("[data-sp-export-preset]");
        if (exportPreset) {
            exportPreset.addEventListener("change", () => {
                const settings = modalSettings(namespace);
                if (exportPreset.value.startsWith("custom:")) {
                    const id = exportPreset.value.slice(7);
                    const custom = namespace.customExportPresets.find(preset => preset.id === id);
                    if (custom) setExportFields(namespace, custom.fields, exportPreset.value);
                    else renderExportModal(namespace);
                    return;
                }
                setExportFields(namespace, EXPORT_PRESETS[exportPreset.value] || EXPORT_PRESETS.standard, exportPreset.value);
            });
        }

        const selectAllExport = panel.querySelector("[data-sp-export-select-all]");
        if (selectAllExport) {
            selectAllExport.addEventListener("click", () => {
                const available = new Set(Array.from(panel.querySelectorAll("[data-sp-export-field]:not(:disabled)"))
                    .map(input => input.dataset.spExportField));
                setExportFields(namespace, available, "custom:unsaved");
            });
        }
        const clearExport = panel.querySelector("[data-sp-export-clear-fields]");
        if (clearExport) clearExport.addEventListener("click", () => setExportFields(namespace, [], "custom:unsaved"));
        const resetExport = panel.querySelector("[data-sp-export-reset]");
        if (resetExport) resetExport.addEventListener("click", () => setExportFields(namespace, EXPORT_PRESETS.standard, "standard"));
        const saveExport = panel.querySelector("[data-sp-export-save]");
        if (saveExport) {
            saveExport.addEventListener("click", () => {
                const settings = modalSettings(namespace);
                settings.exportFields = selectedExportFieldsFromModal(namespace);
                if (!settings.exportFields.size) {
                    window.alert("Select at least one export field before saving a preset.");
                    return;
                }
                const activeId = String(settings.exportPreset).startsWith("custom:")
                    ? String(settings.exportPreset).slice(7)
                    : "";
                const active = namespace.customExportPresets.find(preset => preset.id === activeId);
                const entered = window.prompt("Name this export preset:", active ? active.name : "");
                if (entered === null) return;
                const name = entered.trim().slice(0, 60);
                if (!name) {
                    window.alert("Enter a name for the export preset.");
                    return;
                }
                const sameName = namespace.customExportPresets.find(preset => preset.name.toLowerCase() === name.toLowerCase());
                if (sameName && sameName.id !== activeId && !window.confirm(`Replace the existing preset “${sameName.name}”?`)) return;
                const reuseActive = active && active.name.toLowerCase() === name.toLowerCase();
                if (!sameName && !reuseActive && namespace.customExportPresets.length >= MAX_CUSTOM_EXPORT_PRESETS) {
                    window.alert(`Status Pro supports up to ${MAX_CUSTOM_EXPORT_PRESETS} custom export presets.`);
                    return;
                }
                const id = sameName
                    ? sameName.id
                    : (reuseActive ? active.id : ((window.crypto && window.crypto.randomUUID)
                        ? window.crypto.randomUUID()
                        : `preset-${Date.now()}-${Math.random().toString(16).slice(2)}`));
                const saved = { id, name, fields: Array.from(settings.exportFields) };
                const index = namespace.customExportPresets.findIndex(preset => preset.id === id);
                if (index >= 0) namespace.customExportPresets[index] = saved;
                else namespace.customExportPresets.push(saved);
                namespace.customExportPresets = normalizeCustomExportPresets(namespace.customExportPresets);
                settings.exportPreset = `custom:${id}`;
                saveCustomExportPresets(namespace.customExportPresets);
                renderExportModal(namespace);
                saveExport.textContent = "Saved";
                window.setTimeout(() => { saveExport.textContent = "Save as preset…"; }, 1200);
            });
        }
        const deleteExportPreset = panel.querySelector("[data-sp-export-delete-preset]");
        if (deleteExportPreset) {
            deleteExportPreset.addEventListener("click", () => {
                const settings = modalSettings(namespace);
                if (!String(settings.exportPreset).startsWith("custom:") || settings.exportPreset === "custom:unsaved") return;
                const id = settings.exportPreset.slice(7);
                const preset = namespace.customExportPresets.find(candidate => candidate.id === id);
                if (!preset || !window.confirm(`Delete the export preset “${preset.name}”?`)) return;
                namespace.customExportPresets = namespace.customExportPresets.filter(candidate => candidate.id !== id);
                settings.exportPreset = "custom:unsaved";
                saveCustomExportPresets(namespace.customExportPresets);
                renderExportModal(namespace);
            });
        }
        const confirmExport = panel.querySelector("[data-sp-export-confirm]");
        if (confirmExport) {
            confirmExport.addEventListener("click", () => {
                const settings = modalSettings(namespace);
                const fields = selectedExportFieldsFromModal(namespace);
                if (!fields.size) return;
                const nextRecording = settings.historyRecording !== false;
                const nextHistory = normalizeHistoryPersistence(settings.historyPersistence);
                if (nextRecording !== (namespace.historyRecording !== false) && !window.confirm(historyRecordingConfirmation(nextRecording))) return;
                if (nextHistory !== namespace.historyPersistence && !window.confirm(historyPersistenceConfirmation(nextHistory))) return;
                if (!settings.promptMemory && namespace.promptMemory !== false &&
                    !window.confirm("Turn off page-session prompt memory?\n\nPrompt text currently held by Status Pro will be removed, and future prompts will not be available in History exports until this setting is turned on again.")) return;
                if (nextHistory !== namespace.historyPersistence && !setHistoryPersistence(namespace, nextHistory)) {
                    window.alert(namespace.historyStorageNotice || "Status Pro could not change the history storage setting.");
                    return;
                }
                setHistoryRecording(namespace, nextRecording);
                setPromptMemory(namespace, settings.promptMemory);
                namespace.exportFields = new Set(fields);
                namespace.exportPreset = inferExportPreset(namespace, fields);
                namespace.exportFormat = normalizeExportFormat(settings.exportFormat);
                saveExportSettings(namespace);
                closeExportModal(namespace);
                renderHistoryDrawer(namespace);
            });
        }

        const clearSelected = panel.querySelector("[data-sp-clear-selected]");
        if (clearSelected) {
            clearSelected.addEventListener("click", () => {
                const selectedIds = new Set(namespace.selectedRunIds);
                const selectedCount = selectedIds.size;
                if (!selectedCount) return;
                if (!window.confirm(`Clear ${selectedCount} selected locally stored Status Pro history ${selectedCount === 1 ? "entry" : "entries"}?`)) return;
                namespace.runHistory = namespace.runHistory.filter(run => !selectedIds.has(String(run.id)));
                selectedIds.forEach(runId => {
                    namespace.sessionRunIds.delete(runId);
                    namespace.recoverablePrompts.delete(runId);
                    namespace.openHistoryRuns.delete(runId);
                });
                namespace.selectedRunIds.clear();
                const remainingGroupKeys = new Set(groupHistoryRuns(namespace.runHistory).map(group => group.key));
                namespace.openHistoryGroups = new Set(Array.from(namespace.openHistoryGroups).filter(key => remainingGroupKeys.has(key)));
                namespace.historyRenderKey = null;
                persistRunHistory(namespace);
                if (namespace.activeRun) render(namespace);
                else renderIdle(namespace);
            });
        }

        const clearHistory = panel.querySelector("[data-sp-clear-history]");
        if (clearHistory) {
            clearHistory.addEventListener("click", () => {
                if (!window.confirm("Clear all locally stored Status Pro generation history?")) return;
                namespace.runHistory = [];
                namespace.sessionRunIds.clear();
                namespace.selectedRunIds.clear();
                namespace.recoverablePrompts.clear();
                namespace.openHistoryGroups.clear();
                namespace.openHistoryRuns.clear();
                namespace.visibleHistoryGroups.clear();
                namespace.historyRenderKey = null;
                namespace.historyStorageNotice = "";
                persistRunHistory(namespace);
                if (namespace.activeRun) render(namespace);
                else renderIdle(namespace);
            });
        }

        applyCollapsed(namespace);
        setActive(namespace, true);
        renderIdle(namespace);
        namespace.timer = window.setInterval(() => tick(namespace), TICK_MS);
        tick(namespace);
        console.info("[Status Pro] Progress timeline initialized");
        return true;
    }

    function boot() {
        if (bind()) return;
        window.setTimeout(boot, 500);
    }

    boot();
})();
"""
