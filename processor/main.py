"""
spike-sort: preprocessed recordings (ephys-prep) -> spike sorting.

Variant `gpu`: Kilosort4 (CUDA; exits 3 without a GPU). Variant `cpu`: SpikeInterface's Lupin (default)
or SpykingCircus2. Per session, under OUTPUT_DIR/spike-sort/<session_id>/:

  sorting/         SpikeInterface numpy_folder; spike times in samples of the ephys-prep recording
  sorter_output/   the sorter's own output (Kilosort4: the Phy-compatible folder) minus temporary files
  sort.json        sorter, version, parameters, n_units, wall time, drift correction, GPU name / peak VRAM
  <session_id>_units.png

and spike-sort/summary.json, provenance.json.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from processor import common as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
log = logging.getLogger("spike-sort")
log.setLevel(logging.INFO)
for noisy in ("matplotlib", "PIL", "fontTools", "numba", "h5py"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP = "spike-sort"
REPO_ROOT = Path(__file__).resolve().parent.parent
SORTERS = ("kilosort4", "lupin", "spykingcircus2")

DEFAULTS = {
    "SORTER": "",                 # empty: kilosort4 in the gpu image, lupin in the cpu image
    "DRIFT_CORRECTION": "auto",
    "MOTION_THRESHOLD_UM": "10",
    "KS_BATCH_SIZE": "60000",
    "KS_NBLOCKS": "5",
    "KS_TH_UNIVERSAL": "9",
    "KS_TH_LEARNED": "8",
    "KS_CLEAR_CACHE": "false",
    "DETECT_THRESHOLD": "5",
    "THREADS": "0",
    "SEED": "42",
}
PARAMETERS = tuple(DEFAULTS)
CITATIONS = {"spikeinterface": "10.7554/eLife.61834", "aind_pipeline": "10.7554/eLife.110170"}


class ParameterError(ValueError):
    pass


def image_flavour() -> str:
    """The variant the image was built for (Dockerfile ARG TORCH_FLAVOUR, kept as an env var): cuda or cpu.
    Outside an image (tests, a dev venv): from what is installed."""
    return os.environ.get("TORCH_FLAVOUR") or ("cuda" if _has("kilosort") else "cpu")


def variant_name() -> str:
    flavour = os.environ.get("TORCH_FLAVOUR")
    return {"cuda": "gpu", "cpu": "cpu"}.get(flavour or "", f"none (not an image build; {image_flavour()} environment)")


def _has(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


def read_parameters(env=os.environ) -> dict:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p: dict = {}
    s = raw["SORTER"].strip().lower()
    p["SORTER"] = s or ("kilosort4" if image_flavour() == "cuda" else "lupin")
    if p["SORTER"] not in SORTERS:
        raise ParameterError(f"SORTER must be one of {', '.join(SORTERS)}; got {raw['SORTER']!r}")
    p["DRIFT_CORRECTION"] = raw["DRIFT_CORRECTION"].strip().lower()
    if p["DRIFT_CORRECTION"] not in ("auto", "on", "off"):
        raise ParameterError("DRIFT_CORRECTION must be auto, on or off")
    try:
        for k in ("KS_BATCH_SIZE", "KS_NBLOCKS", "THREADS", "SEED"):
            p[k] = int(raw[k])
        for k in ("KS_TH_UNIVERSAL", "KS_TH_LEARNED", "DETECT_THRESHOLD", "MOTION_THRESHOLD_UM"):
            p[k] = float(raw[k])
    except ValueError as exc:
        raise ParameterError(f"bad numeric parameter: {exc}") from exc
    p["KS_CLEAR_CACHE"] = raw["KS_CLEAR_CACHE"].strip().lower() in ("1", "true", "yes")
    if p["KS_BATCH_SIZE"] < 1000 or p["KS_NBLOCKS"] < 0 or p["THREADS"] < 0 or p["DETECT_THRESHOLD"] <= 0:
        raise ParameterError("KS_BATCH_SIZE >= 1000, KS_NBLOCKS >= 0, THREADS >= 0, DETECT_THRESHOLD > 0")
    return p


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    for pkg in ("spikeinterface", "kilosort", "torch", "faiss-cpu", "probeinterface", "numpy", "scipy", "numba", "scikit-learn",
                "hdbscan", "matplotlib"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def _rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------------- inputs

def find_sessions(input_dir: Path) -> list[dict]:
    """ephys-prep sessions (<sid>/session.json + <sid>/recording/), else bare SpikeInterface recording folders."""
    found, seen = [], set()
    for d, dirs, files in C.walk(input_dir):
        if "session.json" in files and (d / "recording").is_dir():
            try:
                meta = json.loads((d / "session.json").read_text())
            except (OSError, ValueError):
                meta = {}
            sid = meta.get("session_id") or d.name
            found.append({"session_id": C.SAFE_ID.sub("-", sid), "dir": d, "recording": d / "recording", "session": meta,
                          "session_json": d / "session.json"})
            seen.add(os.path.realpath(d / "recording"))
            dirs[:] = []
        elif "si_folder.json" in files and os.path.realpath(d) not in seen:
            try:
                obj = json.loads((d / "si_folder.json").read_text()).get("object", "")
            except (OSError, ValueError):
                obj = ""
            if "Sorting" in str(obj):
                dirs[:] = []
                continue
            name = d.parent.name if d.name == "recording" else d.name
            found.append({"session_id": C.SAFE_ID.sub("-", name), "dir": d, "recording": d, "session": {}, "session_json": None})
            dirs[:] = []
    ids = set()
    for f in found:
        sid, k = f["session_id"], 2
        while sid in ids:
            sid = f"{f['session_id']}-{k}"
            k += 1
        f["session_id"] = sid
        ids.add(sid)
    return found


# --------------------------------------------------------------------------------------------- GPU

def gpu_check(sorter: str) -> tuple[int, str | None, dict | None]:
    """(exit code, message, gpu info). Kilosort4 needs CUDA: exit 3 when it can't run."""
    if sorter != "kilosort4":
        return 0, None, None
    if not _has("kilosort"):
        return 3, ("SORTER=kilosort4 needs the gpu variant of spike-sort (this image has no Kilosort4). "
                   "Pick the gpu variant, or SORTER=lupin / spykingcircus2 to sort on CPU."), None
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return 0, None, {"name": torch.cuda.get_device_name(0), "total_vram_mb": round(props.total_memory / 2**20),
                         "cuda": torch.version.cuda, "torch": torch.__version__}
    if os.environ.get("ALLOW_CPU_KILOSORT") == "1":     # developer tests only: Kilosort4 on CPU is far too slow for real data
        log.warning("ALLOW_CPU_KILOSORT=1: running Kilosort4 on the CPU (testing only)")
        return 0, None, {"name": "cpu (ALLOW_CPU_KILOSORT)", "total_vram_mb": 0}
    return 3, ("Kilosort4 needs an NVIDIA GPU and none is visible (torch.cuda.is_available() is False). Run the gpu variant "
               "on a GPU compute node, or use the cpu variant of spike-sort (SORTER=lupin, SpikeInterface's CPU sorter)."), None


def _vram_now() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"GPU memory {torch.cuda.memory_allocated() / 2**20:.0f} MB (peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB)"
    except Exception:   # noqa: BLE001
        pass
    return ""


# --------------------------------------------------------------------------------------------- one session

def sort_session(item: dict, p: dict, out_root: Path, input_dir: Path, job_kwargs: dict, gpu: dict | None) -> dict:
    import numpy as np
    import spikeinterface.core as sc
    import spikeinterface.sorters as ss

    from processor import figures as F
    from processor import sorters as S

    sid = item["session_id"]
    sdir = out_root / sid
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    rec = sc.load(item["recording"])
    meta = item["session"]
    n_ch = rec.get_num_channels()
    if not rec.has_probe():
        raise RuntimeError("the recording has no probe / channel locations (sorters need them)")
    motion_corrected = bool(meta.get("motion_corrected", False))
    # Session-clock times stay in the main process: worker processes get the recording without its time vector
    # (8 bytes per sample, copied into every worker otherwise). Spike times are sample indices anyway.
    times = np.asarray(rec.get_times(), dtype="float64") if rec.has_time_vector() else None
    if times is not None:
        rec.reset_times()
    t_rec0 = rec.get_start_time()
    to_time = (lambda i: times[np.clip(np.asarray(i, dtype=np.int64), 0, len(times) - 1)]) if times is not None else \
        (lambda i: rec.sample_index_to_time(np.asarray(i)))
    sorter = p["SORTER"]
    motion_dir = item["dir"] / "motion"
    plan = S.drift_plan(sorter, p["DRIFT_CORRECTION"], meta, n_ch, p["MOTION_THRESHOLD_UM"], motion_dir.is_dir())
    if p["DRIFT_CORRECTION"] == "on" and motion_corrected:
        log.warning("session %s: %s", sid, plan["reason"])
    params = S.sorter_params(sorter, p, plan["sorter_corrects"], p["SEED"])
    rec_sort = rec
    if plan["interpolate"]:
        import spikeinterface.preprocessing as spre
        from spikeinterface.core import Motion
        from spikeinterface.sortingcomponents.motion import interpolate_motion

        m = Motion.load(motion_dir)
        if times is not None:      # the motion's bins are on the session clock: back to this recording's sample clock
            fs = rec.get_sampling_frequency()
            m = Motion(displacement=m.displacement, spatial_bins_um=m.spatial_bins_um, direction=m.direction,
                       temporal_bins_s=[t_rec0 + np.interp(np.asarray(tb), times, np.arange(len(times))) / fs for tb in m.temporal_bins_s])
        rec_sort = interpolate_motion(spre.astype(rec, "float32"), m, border_mode="force_extrapolate", spatial_interpolation_method="kriging",
                                      sigma_um=20.0, p=2, dtype="float32")
    log.info("session %s: %s on %d channels, %.1f s at %.0f Hz (t = %.3f .. %.3f s on the session clock); drift: %s", sid, sorter,
             n_ch, rec.get_total_duration(), rec.get_sampling_frequency(), float(to_time(0)), float(to_time(rec.get_num_samples() - 1)),
             plan["reason"])

    if gpu and sorter == "kilosort4":
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    _seed_everything(p["SEED"])
    run_dir = sdir / "_run"
    t0 = time.monotonic()
    with C.Heartbeat(f"{sid}: {sorter} sorting", extra=_vram_now if sorter == "kilosort4" else None):
        sorting = ss.run_sorter(sorter, rec_sort, folder=run_dir, remove_existing_folder=True, delete_output_folder=False,
                                verbose=True, raise_error=True, **({} if sorter == "kilosort4" else {"job_kwargs": job_kwargs}), **params)
    wall = time.monotonic() - t0
    peak_vram = None
    if sorter == "kilosort4":
        try:
            import torch

            if torch.cuda.is_available():
                peak_vram = round(torch.cuda.max_memory_reserved() / 2**20)
        except Exception:   # noqa: BLE001
            pass
    n_raw = len(sorting.unit_ids)
    sorting = sorting.remove_empty_units()
    from spikeinterface.curation import remove_excess_spikes

    sorting = remove_excess_spikes(sorting, rec)
    n_units = len(sorting.unit_ids)
    counts = sorting.count_num_spikes_per_unit(outputs="array") if n_units else np.array([])
    sorting.save(folder=sdir / "sorting", format="numpy_folder", overwrite=True)

    motion_ks = _collect_sorter_output(run_dir, sdir / "sorter_output", sorter, rec, item, to_time)
    shutil.rmtree(run_dir, ignore_errors=True)

    figs = []
    if n_units:
        cap, stats = F.units_figure(sdir / f"{sid}_units.png", sid, rec, sorting, sorter, to_time)
        figs.append({"path": f"{sid}/{sid}_units.png", "caption": cap})
    else:
        stats = {}
    from importlib.metadata import version

    try:
        sorter_version = ss.sorter_dict[sorter].get_sorter_version()
    except Exception:   # noqa: BLE001
        sorter_version = "unknown"
    # duration on the session clock (the timestamps when the recording has them), not samples / nominal rate
    # span on the session clock (includes pauses kept by ephys-prep) vs time actually recorded (samples / rate)
    dur = float(to_time(rec.get_num_samples() - 1) - to_time(0)) + 1.0 / rec.get_sampling_frequency()
    recorded = rec.get_num_samples() / rec.get_sampling_frequency()
    sort_json = {
        "session_id": sid,
        "sorter": sorter,
        "sorter_version": str(sorter_version),
        "spikeinterface": version("spikeinterface"),
        "parameters": params,
        "drift_correction": {"by_sorter": plan["sorter_corrects"], "interpolated_with_ephys_prep_motion": plan["interpolate"],
                             "reason": plan["reason"], "threshold_um": p["MOTION_THRESHOLD_UM"],
                             "ephys_prep_max_drift_range_um": (meta.get("motion") or {}).get("max_drift_range_um"),
                             "ephys_prep_motion_corrected": motion_corrected},
        "n_units": n_units,
        "n_units_before_cleaning": n_raw,
        "n_spikes": int(counts.sum()) if n_units else 0,
        "median_firing_rate_hz": float(np.median(counts / recorded)) if n_units else None,
        "n_channels": n_ch,
        "sampling_frequency": rec.get_sampling_frequency(),
        "t_start": float(to_time(0)),
        "duration_s": dur,
        "recorded_s": recorded,
        "spike_times": "samples of the ephys-prep recording (sorting/); seconds = recording times at those samples",
        "wall_seconds": round(wall, 1),
        "realtime_factor": round(wall / recorded, 2) if recorded else None,
        "gpu": ({"name": gpu.get("name"), "total_vram_mb": gpu.get("total_vram_mb"), "peak_vram_mb": peak_vram} if gpu else None),
        "recording": _rel(item["recording"], input_dir),
        "kilosort_motion_saved": motion_ks,
        **({"sheet": meta["sheet"]} if meta.get("sheet") else {}),
        "app": APP, "version": app_version(),
    }
    C.write_json(sdir / "sort.json", sort_json)
    log.info("session %s: %s found %d units (%d before removing empty ones), %d spikes, in %.0f s (%.2f x real time)%s", sid, sorter, n_units,
             n_raw, sort_json["n_spikes"], wall, wall / recorded if recorded else 0, f", peak VRAM {peak_vram} MB" if peak_vram else "")
    return {"sort": sort_json, "figures": figs, "stats": stats}


def _seed_everything(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _log_sorter_trace(run_dir: Path):
    try:
        trace = json.loads((run_dir / "spikeinterface_log.json").read_text()).get("error_trace")
        if trace:
            lines = trace if isinstance(trace, list) else str(trace).splitlines()
            log.error("sorter trace (last lines):\n%s", "\n".join(lines[-12:]))
    except (OSError, ValueError):
        pass


KS_TEMP = {"temp_wh.dat", "recording.dat"}


def _collect_sorter_output(run_dir: Path, dest: Path, sorter: str, rec, item, to_time) -> bool:
    """Keep the sorter's own folder (Kilosort4: Phy-compatible) minus big temporary files; Kilosort4's drift as a Motion."""
    src = run_dir / "sorter_output"
    dest.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        for f in src.iterdir():
            if f.name in KS_TEMP or f.name.startswith("cache_preprocessing"):
                continue
            shutil.move(str(f), dest / f.name)
    for f in run_dir.glob("spikeinterface_*.json"):
        shutil.copy(f, dest / f.name)
    if sorter == "kilosort4" and (dest / "params.py").exists():
        # Phy: point dat_path at the ephys-prep binary (relative paths differ between ports, so the file name only)
        txt = (dest / "params.py").read_text().splitlines()
        txt = [("dat_path = 'traces_cached_seg0.raw'  # ephys-prep recording/ (copy it here, or edit this path, to view raw data in Phy)"
                if line.startswith("dat_path") else line) for line in txt]
        (dest / "params.py").write_text("\n".join(txt) + "\n")
    saved = False
    if sorter == "kilosort4" and (dest / "ops.npy").exists():
        try:
            import numpy as np
            from spikeinterface.core import Motion

            ops = np.load(dest / "ops.npy", allow_pickle=True).item()
            yblk, dshift = ops.get("yblk"), ops.get("dshift")
            if yblk is not None and dshift is not None and np.size(dshift):
                dshift = np.asarray(dshift)
                tb = ops["batch_size"] / ops["fs"]
                t0, t1 = float(to_time(0)), float(to_time(rec.get_num_samples() - 1))
                bins = np.linspace(t0 + tb / 2, t1 - tb / 2, dshift.shape[0])
                Motion(displacement=dshift, temporal_bins_s=bins, spatial_bins_um=np.asarray(yblk)).save(dest.parent / "kilosort_motion")
                saved = True
        except Exception as exc:   # noqa: BLE001 - optional
            log.info("Kilosort4 drift estimate not saved: %s", exc)
    return saved


# --------------------------------------------------------------------------------------------- main

def main() -> int:
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    try:
        p = read_parameters()
    except ParameterError as exc:
        log.error("%s", exc)
        return 2
    threads = C.set_thread_env(p["THREADS"])
    C.setup_writable_dirs()
    code, msg, gpu = gpu_check(p["SORTER"])
    if code:
        log.error("%s", msg)
        return code
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 1
    import spikeinterface.core as sc

    # spawn: worker processes start clean (forking after torch / OpenMP has run, e.g. in the sorter's own drift
    # correction, can crash or hang); the recordings are folders on disk, so workers re-open them cheaply.
    job_kwargs = C.job_kwargs_for(threads)
    sc.set_global_job_kwargs(**job_kwargs)
    logging.getLogger("kilosort").setLevel(logging.INFO)
    logging.getLogger("kilosort").propagate = False     # Kilosort4 prints its progress with its own handler
    log.info("%s %s (%s image): SORTER=%s, THREADS=%d, GPU %s, parameters %s", APP, app_version(), image_flavour(), p["SORTER"], threads,
             (gpu or {}).get("name", "none"), p)

    items = find_sessions(input_dir)
    if not items:
        log.error("no ephys-prep sessions (<session_id>/session.json + recording/) or SpikeInterface recording folders under %s", input_dir)
        return 1
    log.info("found %d session(s): %s", len(items), ", ".join(i["session_id"] for i in items))
    # Two output ports (conventions.md): `sorting` (the data, plus the small files) and `qc` (only the small files:
    # session / sort json, figures, summary, provenance), so the report can be linked without the data.
    out_root = output_dir / "sorting" / APP
    out_root.mkdir(parents=True, exist_ok=True)
    results, figures, warnings = {}, [], []
    for it in items:
        ts = time.monotonic()
        try:
            r = sort_session(it, p, out_root, input_dir, job_kwargs, gpu)
            results[it["session_id"]] = {"status": "ok", "seconds": round(time.monotonic() - ts, 1), **r}
            figures += r["figures"]
        except Exception as exc:   # noqa: BLE001 - a failed session is logged and skipped
            msg = f"{type(exc).__name__}: {exc}"
            log.error("session %s failed: %s", it["session_id"], msg)
            _log_sorter_trace(out_root / it["session_id"] / "_run")
            shutil.rmtree(out_root / it["session_id"], ignore_errors=True)
            results[it["session_id"]] = {"status": "failed", "seconds": round(time.monotonic() - ts, 1), "error": msg}
            warnings.append(f"{it['session_id']}: failed ({msg})")
            if "out of memory" in msg.lower():
                warnings.append(f"{it['session_id']}: GPU out of memory: try KS_CLEAR_CACHE=true or KS_BATCH_SIZE=30000, or a larger GPU")

    def headline(sid, r):
        if r["status"] != "ok":
            return {"status": r["status"], "error": r.get("error")}
        s = r["sort"]
        h = {"status": "ok", "sorter": s["sorter"], "sorter_version": s["sorter_version"], "n_units": s["n_units"], "n_spikes": s["n_spikes"],
                "median_firing_rate_hz": s["median_firing_rate_hz"], "duration_s": s["duration_s"], "recorded_s": s["recorded_s"], "wall_seconds": s["wall_seconds"],
                "realtime_factor": s["realtime_factor"], "drift_correction_by_sorter": s["drift_correction"]["by_sorter"],
                "drift_interpolated": s["drift_correction"]["interpolated_with_ephys_prep_motion"], "drift_decision": s["drift_correction"]["reason"],
                "gpu": (s["gpu"] or {}).get("name"), "peak_vram_mb": (s["gpu"] or {}).get("peak_vram_mb"), **r.get("stats", {}),
                }
        if s.get("sheet"):
            h["sheet"] = s["sheet"]                 # the session sheet's columns, kept apart from measured values
        return h

    C.write_json(out_root / "summary.json", {
        "app": APP, "version": app_version(),
        "sessions": {sid: headline(sid, r) for sid, r in results.items()},
        "figures": figures, "tables": [], "warnings": warnings,
    })
    from processor.sorters import SORTER_DOIS

    ok = [s for s, r in results.items() if r["status"] == "ok"]
    peak_vram = max([r["sort"]["gpu"]["peak_vram_mb"] or 0 for r in results.values() if r["status"] == "ok" and r["sort"]["gpu"]] or [0])
    finished = datetime.now(timezone.utc)
    C.write_json(out_root / "provenance.json", {
        "app": APP, "version": app_version(), "variant": variant_name(),
        "tools": tool_versions(),
        "parameters": p,
        "inputs": sorted(_rel(i["session_json"] or i["recording"], input_dir) for i in items),
        "started": started.isoformat().replace("+00:00", "Z"), "finished": finished.isoformat().replace("+00:00", "Z"),
        "wall_seconds": round(time.monotonic() - t0, 1),
        "peak_rss_mb": C.peak_rss_mb(),
        "gpu": {"name": gpu["name"], "peak_vram_mb": peak_vram} if gpu else None,
        "sessions": {sid: {"status": r["status"], "seconds": r["seconds"], **({"error": r["error"]} if r.get("error") else {})}
                     for sid, r in results.items()},
        "citations": [CITATIONS["spikeinterface"]] + SORTER_DOIS[p["SORTER"]] + [CITATIONS["aind_pipeline"]],
    })
    C.copy_small_files(out_root, output_dir / "qc" / APP)
    log.info("%s done: %d of %d sessions ok in %.0f s, peak RSS %s MB", APP, len(ok), len(results), time.monotonic() - t0, C.peak_rss_mb())
    return 0 if ok else 1
