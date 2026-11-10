"""spike-sort without Docker: parameters, the GPU contract, drift rules, and sorting accuracy on planted units."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from processor import main as M
from processor import sorters as S
from tests import make_fixture


def run(monkeypatch, input_dir: Path, output_dir: Path, **params) -> int:
    for k in M.PARAMETERS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("THREADS", "2")
    for k, v in params.items():
        monkeypatch.setenv(k, str(v))
    return M.main()


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("fixture")
    make_fixture.write(out, seconds=30.0)
    return out


def accuracy(gt_npz: Path, sorting_folder: Path):
    import spikeinterface.comparison as scmp
    import spikeinterface.core as sc

    gt = np.load(gt_npz)
    gts = sc.NumpySorting.from_samples_and_labels([gt["sample_index"]], [gt["unit_index"]], float(gt["sampling_frequency"]))
    cmp = scmp.compare_sorter_to_ground_truth(gts, sc.load(sorting_folder), exhaustive_gt=True)
    return cmp.get_performance()["accuracy"].to_numpy()


def test_bad_parameters(monkeypatch, tmp_path):
    assert run(monkeypatch, tmp_path, tmp_path / "o", SORTER="mountainsort9") == 2
    assert run(monkeypatch, tmp_path, tmp_path / "o", DRIFT_CORRECTION="maybe") == 2
    assert run(monkeypatch, tmp_path, tmp_path / "o", KS_BATCH_SIZE="10") == 2


def test_no_input(monkeypatch, tmp_path):
    (tmp_path / "in").mkdir()
    assert run(monkeypatch, tmp_path / "in", tmp_path / "o", SORTER="lupin") == 1


def test_kilosort_without_gpu_exits_3(monkeypatch, tmp_path):
    monkeypatch.delenv("ALLOW_CPU_KILOSORT", raising=False)
    code, msg, _ = M.gpu_check("kilosort4")
    if importlib.util.find_spec("torch") and importlib.util.find_spec("kilosort"):
        import torch

        if torch.cuda.is_available():
            pytest.skip("a GPU is visible")
    assert code == 3 and ("GPU" in msg or "gpu variant" in msg)
    assert run(monkeypatch, tmp_path, tmp_path / "o", SORTER="kilosort4") == 3


def test_drift_rules():
    plan = S.drift_plan
    big, small = {"motion": {"max_drift_range_um": 25.0}}, {"motion": {"max_drift_range_um": 4.0}}
    assert plan("kilosort4", "auto", big, 384, 10, True)["sorter_corrects"] is True          # KS4: its own correction
    assert plan("kilosort4", "auto", small, 384, 10, True)["sorter_corrects"] is True
    assert plan("kilosort4", "auto", {"motion_corrected": True}, 384, 10, True)["sorter_corrects"] is False   # never twice
    assert plan("kilosort4", "auto", big, 64, 10, True)["sorter_corrects"] is False           # AIND: < 96 channels
    assert plan("lupin", "auto", big, 384, 10, True) == {"sorter_corrects": False, "interpolate": True, "reason": plan("lupin", "auto", big, 384, 10, True)["reason"]}
    assert plan("lupin", "auto", small, 384, 10, True)["interpolate"] is False               # below the threshold
    assert plan("lupin", "on", small, 384, 10, True)["interpolate"] is True
    assert plan("lupin", "auto", {}, 384, 10, False)["sorter_corrects"] is True              # no ephys-prep motion
    assert plan("lupin", "off", big, 384, 10, True)["interpolate"] is False
    ks = S.sorter_params("kilosort4", {"KS_BATCH_SIZE": 60000, "KS_TH_UNIVERSAL": 9, "KS_TH_LEARNED": 8, "KS_CLEAR_CACHE": False,
                                       "KS_NBLOCKS": 5}, False, 1)
    assert ks["nblocks"] == 0 and ks["do_correction"] is False and ks["batch_size"] == 60000 and ks["Th_universal"] == 9
    ks = S.sorter_params("kilosort4", {"KS_BATCH_SIZE": 60000, "KS_TH_UNIVERSAL": 9, "KS_TH_LEARNED": 8, "KS_CLEAR_CACHE": False,
                                       "KS_NBLOCKS": 5}, True, 1)
    assert ks["nblocks"] == 5 and ks["do_correction"] is True
    lu = S.sorter_params("lupin", {"DETECT_THRESHOLD": 5.0}, False, 1)
    assert lu["apply_motion_correction"] is False and lu["preprocessing_dict"] == {}


@pytest.fixture(scope="session")
def lupin_run(fixture_dir, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("out-lupin")
    mp = pytest.MonkeyPatch()
    try:
        assert run(mp, fixture_dir / "input", out, SORTER="lupin") == 0
    finally:
        mp.undo()
    return out


def test_lupin_outputs(lupin_run):
    top = lupin_run / "sorting" / "spike-sort"
    sdir = top / "sub-fixture_ses-01"
    for f in ("sorting/si_folder.json", "sort.json", "sub-fixture_ses-01_units.png"):
        assert (sdir / f).exists(), f
    sort = json.loads((sdir / "sort.json").read_text())
    for k in ("sorter", "sorter_version", "parameters", "n_units", "wall_seconds", "gpu", "drift_correction"):
        assert k in sort
    assert sort["sorter"] == "lupin" and sort["drift_correction"]["by_sorter"] is False   # no motion folder, 64 channels < 96
    assert abs(sort["t_start"] - 100.0) < 1e-9
    prov = json.loads((top / "provenance.json").read_text())
    assert set(prov) >= {"app", "version", "variant", "tools", "parameters", "inputs", "started", "finished", "wall_seconds",
                         "peak_rss_mb", "gpu", "sessions", "citations"}
    summ = json.loads((top / "summary.json").read_text())
    qc = top.parent.parent / "qc" / "spike-sort"
    assert (qc / "sub-fixture_ses-01" / "sort.json").exists() and not (qc / "sub-fixture_ses-01" / "sorting").exists()
    assert all((qc / f["path"]).exists() for f in summ["figures"])
    assert summ["sessions"]["sub-fixture_ses-01"]["n_units"] == sort["n_units"]
    assert summ["figures"][0]["caption"]


def test_lupin_recovers_planted_units(lupin_run, fixture_dir):
    acc = accuracy(fixture_dir / "ground_truth" / "sub-fixture_ses-01_gt.npz", lupin_run / "sorting" / "spike-sort" / "sub-fixture_ses-01" / "sorting")
    assert (acc >= 0.9).sum() >= 10, acc


def test_lupin_on_spikeinterface_ground_truth(monkeypatch, tmp_path):
    """SpikeInterface's generate_ground_truth_recording (Neuropixels-like 2-column probe, 15 units, 40 s)."""
    import spikeinterface.comparison as scmp
    import spikeinterface.core as sc

    rec, gt = sc.generate_ground_truth_recording(durations=[40.0], sampling_frequency=30000.0, num_channels=48, num_units=15,
                                                 generate_probe_kwargs={"num_columns": 2, "xpitch": 32, "ypitch": 20,
                                                                        "contact_shapes": "square", "contact_shape_params": {"width": 12}},
                                                 generate_templates_kwargs={"unit_params": {"alpha": (150.0, 400.0)}},
                                                 noise_kwargs={"noise_levels": 8.0, "strategy": "on_the_fly"}, seed=7)
    folder = tmp_path / "in" / "gt_session" / "recording"
    rec.save(folder=folder, format="binary", dtype="float32", n_jobs=1)
    assert run(monkeypatch, tmp_path / "in", tmp_path / "out", SORTER="lupin") == 0
    sorting = sc.load(tmp_path / "out" / "sorting" / "spike-sort" / "gt_session" / "sorting")
    acc = scmp.compare_sorter_to_ground_truth(gt, sorting, exhaustive_gt=True).get_performance()["accuracy"].to_numpy()
    assert (acc >= 0.8).sum() >= 12, acc          # most units, with high accuracy


@pytest.mark.skipif(importlib.util.find_spec("kilosort") is None, reason="Kilosort4 is only in the gpu image / a dev install")
def test_kilosort4_on_cpu_for_testing(monkeypatch, fixture_dir, tmp_path):
    """The Kilosort4 code path, on CPU (ALLOW_CPU_KILOSORT=1, developer only): outputs and accuracy."""
    monkeypatch.setenv("ALLOW_CPU_KILOSORT", "1")
    assert run(monkeypatch, fixture_dir / "input", tmp_path, SORTER="kilosort4") == 0
    sdir = tmp_path / "sorting" / "spike-sort" / "sub-fixture_ses-01"
    for f in ("spike_times.npy", "spike_clusters.npy", "templates.npy", "params.py", "cluster_KSLabel.tsv"):
        assert (sdir / "sorter_output" / f).exists(), f
    assert not (sdir / "sorter_output" / "temp_wh.dat").exists()
    sort = json.loads((sdir / "sort.json").read_text())
    assert sort["parameters"]["nblocks"] == 0 and sort["parameters"]["Th_universal"] == 9
    acc = accuracy(fixture_dir / "ground_truth" / "sub-fixture_ses-01_gt.npz", sdir / "sorting")
    assert (acc >= 0.8).sum() >= 9, acc


def test_lupin_interpolates_with_ephys_prep_motion(monkeypatch, fixture_dir, tmp_path):
    """Drift above MOTION_THRESHOLD_UM: the recording is interpolated with ephys-prep's motion before Lupin."""
    import shutil

    from spikeinterface.core import Motion

    src = fixture_dir / "input" / "ephys-prep" / "sub-fixture_ses-01"
    dst = tmp_path / "in" / "ephys-prep" / "sub-fixture_ses-01"
    shutil.copytree(src, dst)
    bins = np.arange(100.5, 130.0, 1.0)                                    # session clock (t_start = 100 s)
    disp = -10.0 * (bins - 100.0)[:, None] / 30.0                           # the planted 10 um drift (units move down)
    Motion(displacement=disp, temporal_bins_s=bins, spatial_bins_um=np.array([300.0])).save(dst / "motion")
    meta = json.loads((dst / "session.json").read_text())
    meta["motion"] = {"max_drift_range_um": 10.0}
    meta["motion_estimated"] = True
    (dst / "session.json").write_text(json.dumps(meta))
    assert run(monkeypatch, tmp_path / "in", tmp_path / "out", SORTER="lupin", MOTION_THRESHOLD_UM="5") == 0
    sdir = tmp_path / "out" / "sorting" / "spike-sort" / "sub-fixture_ses-01"
    sort = json.loads((sdir / "sort.json").read_text())
    assert sort["drift_correction"]["interpolated_with_ephys_prep_motion"] is True and sort["drift_correction"]["by_sorter"] is False
    acc = accuracy(fixture_dir / "ground_truth" / "sub-fixture_ses-01_gt.npz", sdir / "sorting")
    assert (acc >= 0.85).sum() >= 10, acc


def test_entry_without_dev_shm(fixture_dir, tmp_path):
    """The container entry point, with SpikeInterface's shared memory backed by files in $TMPDIR (no /dev/shm):
    Lupin's template matching and template estimation run in 2 spawned worker processes."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, INPUT_DIR=str(fixture_dir / "input"), OUTPUT_DIR=str(tmp_path / "out"), TMPDIR=str(tmp_path / "tmp"),
               HOME=str(tmp_path / "home"), SORTER="lupin", THREADS="2")
    r = subprocess.run([sys.executable, "-m", "processor.entry"], env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert (tmp_path / "tmp" / "shm").is_dir() and not list((tmp_path / "tmp" / "shm").iterdir())   # used, and cleaned up
    summ = json.loads((tmp_path / "out" / "sorting" / "spike-sort" / "summary.json").read_text())
    assert summ["sessions"]["sub-fixture_ses-01"]["n_units"] >= 10


def test_fileshm_roundtrip(tmp_path, monkeypatch):
    from processor import fileshm

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(fileshm, "_DIR", None)
    a = fileshm.FileSharedMemory(create=True, size=64)
    b = fileshm.FileSharedMemory(a.name)
    np.ndarray((8,), "float64", buffer=a.buf)[:] = 3.0
    assert np.ndarray((8,), "float64", buffer=b.buf)[0] == 3.0
    b.close()
    a.close()
    a.unlink()
    assert not list((tmp_path / "shm").iterdir())


def test_pause_rates_use_recorded_time(monkeypatch, fixture_dir, tmp_path):
    """A recording whose session-clock times carry a 12 s pause: duration_s is the span (42 s), recorded_s the 30 s
    recorded; rates and the real-time factor use recorded_s. Sheet columns stay under 'sheet' in summary.json."""
    import shutil

    from spikeinterface.core import load

    src = fixture_dir / "input" / "ephys-prep" / "sub-fixture_ses-01"
    dst = tmp_path / "in" / "ephys-prep" / "sub-fixture_ses-01"
    shutil.copytree(src, dst)
    rec = load(dst / "recording")
    n, fs = rec.get_num_samples(), rec.get_sampling_frequency()
    t = 100.0 + np.arange(n) / fs
    t[n // 2:] += 12.0
    t.astype("<f8").tofile(dst / "recording" / "times_cached_seg0.raw")
    b = json.loads((dst / "recording" / "binary.json").read_text())
    b["kwargs"]["file_timestamps_paths"] = ["times_cached_seg0.raw"]
    b["kwargs"]["t_starts"] = None
    (dst / "recording" / "binary.json").write_text(json.dumps(b))
    meta = json.loads((dst / "session.json").read_text())
    meta["sheet"] = {"subject_id": "fix01", "t_start": "999", "probe": "Probe01"}
    (dst / "session.json").write_text(json.dumps(meta))
    assert run(monkeypatch, tmp_path / "in", tmp_path / "out", SORTER="lupin") == 0
    top = tmp_path / "out" / "sorting" / "spike-sort"
    sort = json.loads((top / "sub-fixture_ses-01" / "sort.json").read_text())
    assert sort["duration_s"] == pytest.approx(42.0, abs=1e-3) and sort["recorded_s"] == pytest.approx(30.0, abs=1e-3)
    assert sort["realtime_factor"] == pytest.approx(round(sort["wall_seconds"] / 30.0, 2), abs=0.011)
    gt = np.load(fixture_dir / "ground_truth" / "sub-fixture_ses-01_gt.npz")
    assert sort["median_firing_rate_hz"] == pytest.approx(np.median(np.bincount(gt["unit_index"])) / 30.0, rel=0.25)
    h = json.loads((top / "summary.json").read_text())["sessions"]["sub-fixture_ses-01"]
    assert h["sheet"] == {"subject_id": "fix01", "t_start": "999", "probe": "Probe01"} and "subject_id" not in h and "probe" not in h


def test_fileshm_sweep_at_exit(tmp_path):
    """Segments a process created and never unlinked (a crash) are removed when it exits."""
    import subprocess
    import sys

    code = ("from processor import fileshm; fileshm.install(); from multiprocessing.shared_memory import SharedMemory; "
            "s = SharedMemory(create=True, size=1024); import os; print(os.listdir(fileshm._dir()))")
    r = subprocess.run([sys.executable, "-c", code], env=dict(__import__("os").environ, TMPDIR=str(tmp_path)), capture_output=True, text=True)
    assert r.returncode == 0 and "si_" in r.stdout
    assert not list((tmp_path / "shm").iterdir())
