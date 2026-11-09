"""
A small ephys-prep output with known units, written with numpy + json only (the platform's review
generator image has no SpikeInterface): a SpikeInterface binary_folder exactly as ephys-prep saves it.

  <out>/input/ephys-prep/sub-fixture_ses-01/recording/    int16, 64 channels (2 columns x 32 rows, 20 um pitch),
                                                           30 kHz, 30 s, t_start = 100 s (session clock)
  <out>/input/ephys-prep/sub-fixture_ses-01/session.json   motion_corrected false
  <out>/ground_truth/sub-fixture_ses-01_gt.npz             spike samples / unit ids of the 12 planted units

The traces are already "preprocessed": white noise (9 uV) plus biphasic spikes (80-220 uV), no common
noise, a slow 10 um drift.

    python -m tests.make_fixture OUT [--seconds 30] [--units 12]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FS = 30_000.0
GAIN_UV = 2.34375
N_ROWS, N_COLS, Y_PITCH, X_PITCH = 32, 2, 20.0, 32.0
T_START = 100.0
SI_VERSION = "0.105.4"


def probe_geometry() -> np.ndarray:
    xs = np.repeat(np.arange(N_COLS) * X_PITCH, N_ROWS)
    ys = np.tile(np.arange(N_ROWS) * Y_PITCH, N_COLS)
    order = np.lexsort((xs, ys))
    return np.column_stack([xs[order], ys[order]])


def waveform(n_before: int, n_after: int, width: float) -> np.ndarray:
    t = np.arange(-n_before, n_after) / FS * 1000
    return -np.exp(-0.5 * (t / (0.12 * width)) ** 2) + 0.35 * np.exp(-0.5 * ((t - 0.45 * width) / (0.3 * width)) ** 2)


def synth(seconds: float, n_units: int, seed: int = 0, drift_um: float = 10.0):
    rng = np.random.default_rng(seed)
    locs = probe_geometry()
    n_ch, n = len(locs), int(seconds * FS)
    x = rng.normal(0, 9.0, size=(n, n_ch)).astype(np.float32)
    nb, na = 30, 60
    unit_y = np.linspace(60, locs[:, 1].max() - 60, n_units) + rng.uniform(-10, 10, n_units)
    unit_x = rng.uniform(0, X_PITCH, n_units)
    amps = rng.uniform(80, 220, n_units)
    widths = rng.uniform(0.8, 1.4, n_units)
    rates = rng.uniform(3, 12, n_units)
    spikes, labels = [], []
    for u in range(n_units):
        st = np.cumsum(rng.exponential(1.0 / rates[u], int(rates[u] * seconds * 2) + 10) + 0.002)
        s = (st[(st > 0.01) & (st < seconds - 0.01)] * FS).astype(np.int64)
        wf = waveform(nb, na, widths[u])
        for k in s:
            y = unit_y[u] - drift_um * (k / n)
            d2 = (locs[:, 0] - unit_x[u]) ** 2 + (locs[:, 1] - y) ** 2
            x[k - nb:k + na] += np.outer(wf, amps[u] / (1 + d2 / 30.0 ** 2) ** 1.5).astype(np.float32)
        spikes.append(s)
        labels.append(np.full(len(s), u))
    traces = np.clip(np.round(x / GAIN_UV), -32768, 32767).astype("<i2")
    sp, lab = np.concatenate(spikes), np.concatenate(labels)
    o = np.argsort(sp, kind="stable")
    return traces, locs, {"sample_index": sp[o], "unit_index": lab[o], "unit_y_um": unit_y, "amplitude_uV": amps,
                          "firing_rate_hz": rates, "t_start": T_START, "sampling_frequency": FS}


def write_binary_folder(folder: Path, traces: np.ndarray, locs: np.ndarray, t_start: float = T_START) -> None:
    """A SpikeInterface 0.105 binary_folder (what recording.save(format="binary") writes), by hand."""
    folder.mkdir(parents=True, exist_ok=True)
    n_ch = traces.shape[1]
    traces.tofile(folder / "traces_cached_seg0.raw")
    ids = list(range(n_ch))
    gains, offsets = [GAIN_UV] * n_ch, [0.0] * n_ch
    props = {"group": None, "gain_to_uV": gains, "offset_to_uV": offsets, "gain_to_physical_unit": None,
             "offset_to_physical_unit": None, "physical_unit": None}
    (folder / "si_folder.json").write_text(json.dumps({"class": "spikeinterface.core.binaryfolder.BinaryFolderRecording",
                                                       "module": "spikeinterface", "version": SI_VERSION, "kwargs": {"folder_path": "."},
                                                       "annotations": {"is_filtered": True},
                                                       "properties": dict(props, group=[0] * n_ch), "relative_paths": True}, indent=1))
    (folder / "binary.json").write_text(json.dumps({
        "class": "spikeinterface.core.binaryrecordingextractor.BinaryRecordingExtractor", "module": "spikeinterface", "version": SI_VERSION,
        "kwargs": {"file_paths": ["traces_cached_seg0.raw"], "sampling_frequency": FS, "t_starts": [t_start], "num_channels": n_ch,
                   "dtype": "<i2", "channel_ids": ids, "time_axis": 0, "file_offset": 0, "gain_to_uV": gains, "offset_to_uV": offsets,
                   "is_filtered": True, "file_timestamps_paths": None},
        "annotations": {"is_filtered": True}, "properties": props, "relative_paths": True}, indent=1))
    (folder / "annotations.json").write_text(json.dumps({"is_filtered": True}))
    (folder / "properties").mkdir(exist_ok=True)
    np.save(folder / "properties" / "gain_to_uV.npy", np.array(gains))
    np.save(folder / "properties" / "offset_to_uV.npy", np.array(offsets))
    np.save(folder / "properties" / "group.npy", np.zeros(n_ch, dtype="int64"))
    (folder / "probegroup.json").write_text(json.dumps({"specification": "probeinterface", "version": "0.4.2", "probes": [{
        "ndim": 2, "si_units": "um", "annotations": {"model_name": "synthetic 2x32", "manufacturer": "synthetic"},
        "contact_annotations": {}, "contact_positions": locs.tolist(), "contact_plane_axes": [[[1.0, 0.0], [0.0, 1.0]]] * n_ch,
        "contact_shapes": ["square"] * n_ch, "contact_shape_params": [{"width": 12}] * n_ch,
        "device_channel_indices": ids, "contact_ids": [f"e{i}" for i in ids]}]}, indent=1))


def write(out: Path, seconds: float = 30.0, n_units: int = 12) -> dict:
    out = Path(out)
    traces, locs, gt = synth(seconds, n_units)
    sid = "sub-fixture_ses-01"
    sdir = out / "input" / "ephys-prep" / sid
    write_binary_folder(sdir / "recording", traces, locs)
    (sdir / "session.json").write_text(json.dumps({
        "session_id": sid, "source": "synthetic", "source_format": "synthetic", "stream": "synthetic",
        "probe": {"model": "synthetic 2x32 (20 um pitch)"}, "sampling_frequency": FS, "t_start": T_START,
        "t_stop": T_START + traces.shape[0] / FS, "duration_s": traces.shape[0] / FS, "n_channels": int(traces.shape[1]),
        "bad_channels": [], "motion_corrected": False, "motion_estimated": False, "motion_method": "none",
        "time_source": "t_start + n / rate", "gain_to_uV": GAIN_UV, "dtype": "int16", "app": "ephys-prep", "version": "fixture"}, indent=2))
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)
    np.savez(out / "ground_truth" / f"{sid}_gt.npz", **gt)
    return gt


def cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--units", type=int, default=12)
    a = ap.parse_args()
    write(a.out, a.seconds, a.units)
    print(f"fixture written to {a.out}")


if __name__ == "__main__":
    cli()
