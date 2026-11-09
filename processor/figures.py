"""Per-session figure: where the sorted units are on the probe and how they fire."""
from __future__ import annotations

from pathlib import Path

import numpy as np

DPI = 150


def units_figure(path: Path, sid: str, rec, sorting, sorter: str, max_spikes_per_unit: int = 50, max_points: int = 200_000) -> tuple[str, dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import spikeinterface.core as sc

    an = sc.create_sorting_analyzer(sorting, rec, sparse=False, format="memory", return_in_uV=True)
    an.compute("random_spikes", max_spikes_per_unit=max_spikes_per_unit, seed=0)
    an.compute("templates", ms_before=1.0, ms_after=2.0)
    tpl = an.get_extension("templates").get_data()                 # units x samples x channels (uV)
    ptp = np.ptp(tpl, axis=1)
    peak = ptp.argmax(axis=1)
    amp = ptp[np.arange(len(peak)), peak]
    depth = rec.get_channel_locations()[peak, 1]
    dur = rec.get_total_duration()
    counts = sorting.count_num_spikes_per_unit(outputs="array")
    rate = counts / dur

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), constrained_layout=True, gridspec_kw={"width_ratios": [1, 2.2]}, sharey=True)
    sca = axes[0].scatter(rate, depth, c=amp, s=18, cmap="viridis", vmin=0, vmax=np.percentile(amp, 95) if len(amp) else 1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("firing rate (Hz, log)")
    axes[0].set_ylabel("depth of the unit's peak channel (um)")
    fig.colorbar(sca, ax=axes[0], label="peak-to-peak amplitude (uV)", location="bottom", shrink=0.8)
    sv = sorting.to_spike_vector()
    if len(sv) > max_points:
        sv = sv[np.sort(np.random.default_rng(0).choice(len(sv), max_points, replace=False))]
    t = rec.sample_index_to_time(sv["sample_index"])
    order = np.argsort(amp[sv["unit_index"]])
    axes[1].scatter(t[order], depth[sv["unit_index"]][order] + np.random.default_rng(1).uniform(-4, 4, len(order)), s=0.5,
                    c=amp[sv["unit_index"]][order], cmap="viridis", alpha=0.5, rasterized=True)
    axes[1].set_xlabel("time (s, session clock)")
    fig.suptitle(f"{sid}: {len(rate)} units from {sorter}", fontsize=12)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    stats = {"median_amplitude_uV": float(np.median(amp)) if len(amp) else None,
             "n_units_above_50uV": int((amp >= 50).sum()), "depth_range_um": [float(depth.min()), float(depth.max())] if len(depth) else None}
    cap = (f"The {len(rate)} units the sorter found, before any curation: left, each unit at the depth of its largest "
           "waveform, by firing rate, coloured by amplitude; right, every spike (a sample when there are many) at its unit's "
           "depth over time. Units should spread over the part of the probe in the brain, and their bands should stay "
           "present over the session; a band that stops or starts abruptly can be drift or a unit split in two. Many of "
           "these units are multi-unit activity or noise until ephys-curate labels them.")
    return cap, stats
