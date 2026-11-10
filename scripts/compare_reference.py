"""Agreement of a spike-sort sorting with a reference sorting in an NWB units table (e.g. IBL's own sorting in the
session's processed NWB), on the session clock.

    python scripts/compare_reference.py RECORDING_FOLDER SORTING_FOLDER REFERENCE.nwb [--probe Probe01] [--min-quality 1]

Reference spike times (seconds, session clock) are mapped to samples of the ephys-prep recording with its own
timestamps; units outside the recording's window are dropped. Matching: SpikeInterface compare_two_sorters
(0.4 ms window, Hungarian). Prints how many reference units have a match with agreement >= 0.5 / 0.8.
"""
import argparse
import json

import h5py
import numpy as np
import spikeinterface.comparison as scmp
import spikeinterface.core as sc

ap = argparse.ArgumentParser()
ap.add_argument("recording")
ap.add_argument("sorting")
ap.add_argument("reference")
ap.add_argument("--probe", default=None, help="keep reference units whose electrode group / probe name contains this")
ap.add_argument("--min-quality", type=float, default=None, help="keep units with ibl_quality_score >= this (IBL 'good' = 1)")
a = ap.parse_args()
rec = sc.load(a.recording)
times = rec.get_times()
t0, t1 = times[0], times[-1]
ours = sc.load(a.sorting)
with h5py.File(a.reference, "r") as f:
    u = f["units"]
    st, idx = u["spike_times"][:], u["spike_times_index"][:]
    cols = {}
    qual = u["ibl_quality_score"][:] if "ibl_quality_score" in u else None
    for c in ("probe_name", "electrode_group"):
        if c in u:
            v = u[c][:]
            cols[c] = [x.decode() if isinstance(x, bytes) else str(x) for x in v]
starts = np.concatenate([[0], idx[:-1]])
keep = np.ones(len(idx), bool)
if a.probe:
    names = cols.get("probe_name") or cols.get("electrode_group")
    keep &= np.array([a.probe.lower() in n.lower() for n in names])
if a.min_quality is not None and qual is not None:
    keep &= qual >= a.min_quality
samples, labels = [], []
for k in np.flatnonzero(keep):
    s = st[starts[k]:idx[k]]
    s = s[(s >= t0) & (s <= t1)]
    if len(s) < 10:
        continue
    samples.append(np.searchsorted(times, s))
    labels.append(np.full(len(s), k))
ref = sc.NumpySorting.from_samples_and_labels([np.concatenate(samples)], [np.concatenate(labels)], rec.get_sampling_frequency())
cmp = scmp.compare_two_sorters(ref, ours, sorting1_name="reference", sorting2_name="ours")
best = cmp.hungarian_match_12
agr = np.array([cmp.agreement_scores.loc[u1, u2] if u2 != -1 else 0.0 for u1, u2 in best.items()])
print(json.dumps({"window_s": [float(t0), float(t1)], "n_reference_units": len(ref.unit_ids), "n_sorted_units": len(ours.unit_ids),
                  "matched_ge_0.5": int((agr >= 0.5).sum()), "matched_ge_0.8": int((agr >= 0.8).sum()),
                  "median_agreement_of_matched": float(np.median(agr[agr >= 0.5])) if (agr >= 0.5).any() else None}, indent=2))
