"""Accuracy of a spike-sort sorting against the DANDI:001250 ground truth (units table of the same NWB file).

    python scripts/score_ground_truth.py GT.nwb SORTING_FOLDER [--min-spikes 10] [--out scores.csv]

Scores units with at least --min-spikes ground-truth spikes (788 of 1836 in the demo file), with
SpikeInterface's compare_sorter_to_ground_truth (match window 0.4 ms, Hungarian matching).
"""
import argparse
import json

import numpy as np
import spikeinterface.comparison as scmp
import spikeinterface.core as sc
import spikeinterface.extractors as se

ap = argparse.ArgumentParser()
ap.add_argument("gt_nwb")
ap.add_argument("sorting")
ap.add_argument("--min-spikes", type=int, default=10)
ap.add_argument("--out")
a = ap.parse_args()
sorting = sc.load(a.sorting)
fs = sorting.get_sampling_frequency()
gt = se.read_nwb_sorting(a.gt_nwb, sampling_frequency=fs, t_start=0.0)
counts = gt.count_num_spikes_per_unit()
keep = [u for u, n in counts.items() if n >= a.min_spikes]
gt = gt.select_units(keep)
cmp = scmp.compare_sorter_to_ground_truth(gt, sorting, exhaustive_gt=False, match_score=0.5)
perf = cmp.get_performance()
res = {"n_gt_units_scored": len(keep), "n_sorted_units": len(sorting.unit_ids),
       "well_detected_acc_ge_0.8": int((perf["accuracy"] >= 0.8).sum()),
       "detected_acc_ge_0.5": int((perf["accuracy"] >= 0.5).sum()),
       "median_accuracy_of_matched": float(perf["accuracy"][perf["accuracy"] > 0].median()),
       "mean_accuracy_all": float(perf["accuracy"].mean()),
       **{k: int(v) for k, v in cmp.count_units_categories(well_detected_score=0.8).items()}}
# by ground-truth firing: accuracy of units with >= 100 spikes (well sampled)
big = [u for u in keep if counts[u] >= 100]
res["n_gt_units_ge_100_spikes"] = len(big)
res["well_detected_of_ge_100_spikes"] = int((perf.loc[big, "accuracy"] >= 0.8).sum()) if big else 0
print(json.dumps(res, indent=2))
if a.out:
    perf.assign(n_gt_spikes=[counts[u] for u in perf.index]).to_csv(a.out)
