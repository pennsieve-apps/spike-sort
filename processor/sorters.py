"""
Sorter parameters and the call into SpikeInterface's run_sorter.

Defaults follow the AIND ephys pipeline v1.5.0 (`pipeline/default_params.json`, Buccino et al. 2026),
which runs every sorter on a recording already preprocessed like ephys-prep's (phase shift, 300 Hz
high-pass, bad channels removed, common median reference):

- Kilosort4 4.1.7: batch_size 60000, nblocks 5, Th_universal 9, Th_learned 8, dminx 32,
  whitening_range 32, highpass_cutoff 300, do_CAR true, torch_device auto; drift correction off when
  the probe has fewer than 96 channels (AIND's min_drift_channels). Other parameters: Kilosort 4.1.7's
  own defaults.
- Lupin (SpikeInterface 0.105.4): apply_preprocessing with an empty preprocessing_dict (whitening only,
  the input is already filtered and referenced), template_matching_engine circus-omp,
  template_sparsify_threshold 1.5, cache_preprocessing_mode no-cache, apply_motion_correction
  (dredge_fast) when drift correction is on.
- SpykingCircus2: apply_preprocessing false, cache_preprocessing no-cache, apply_motion_correction
  when drift correction is on.

Drift is corrected exactly once: by ephys-prep (session.json motion_corrected = true, then never
again here) or by the sorter (the default: ephys-prep only estimates motion, as AIND does).
"""
from __future__ import annotations

from copy import deepcopy

MIN_DRIFT_CHANNELS = 96

KS4_AIND = {
    "batch_size": 60000,
    "nblocks": 5,
    "Th_universal": 9,
    "Th_learned": 8,
    "do_CAR": True,
    "invert_sign": False,
    "nskip": 25,
    "whitening_range": 32,
    "highpass_cutoff": 300,
    "binning_depth": 5,
    "sig_interp": 20,
    "drift_smoothing": [0.5, 0.5, 0.5],
    "dminx": 32,
    "min_template_size": 10,
    "template_sizes": 5,
    "nearest_chans": 10,
    "nearest_templates": 100,
    "templates_from_data": True,
    "n_templates": 6,
    "n_pcs": 6,
    "Th_single_ch": 6,
    "acg_threshold": 0.2,
    "ccg_threshold": 0.25,
    "cluster_downsampling": 20,
    "duplicate_spike_ms": 0.25,
    "save_preprocessed_copy": False,
    "torch_device": "auto",
    "clear_cache": False,
    "save_extra_vars": False,
    "do_correction": True,
    "keep_good_only": False,
    "skip_kilosort_preprocessing": False,
    "use_binary_file": None,
    "delete_recording_dat": True,
}

LUPIN_AIND = {
    "apply_preprocessing": True,
    "preprocessing_dict": {},
    "apply_motion_correction": True,
    "motion_correction_preset": "dredge_fast",
    "template_sparsify_threshold": 1.5,
    "template_matching_engine": "circus-omp",
    "cache_preprocessing_mode": "no-cache",
    "detect_threshold": 5.0,
    "save_array": False,
}

SC2_AIND = {
    "apply_preprocessing": False,
    "cache_preprocessing": {"mode": "no-cache"},
    "apply_motion_correction": True,
    "motion_correction": {"preset": "dredge_fast"},
}

SORTER_DOIS = {
    "kilosort4": ["10.1038/s41592-024-02232-7"],
    "lupin": ["10.7554/eLife.110588"],
    "spykingcircus2": ["10.7554/eLife.34518", "10.7554/eLife.110588"],
}


def drift_plan(sorter: str, mode: str, meta: dict, n_channels: int, threshold_um: float, has_motion: bool) -> dict:
    """How drift is corrected for this session, exactly once. Returns {"sorter_corrects", "interpolate", "reason"}.

    - ephys-prep already interpolated (session.json motion_corrected): nothing more.
    - Kilosort4: its own correction (nblocks) unless DRIFT_CORRECTION=off or < 96 channels (AIND). It does not use
      ephys-prep's estimate, and no threshold applies: KS4 is built around its drift correction.
    - Lupin / SpykingCircus2 (Garcia et al. 2026): interpolate the recording with ephys-prep's DREDge motion when
      its largest drift range exceeds MOTION_THRESHOLD_UM (~10 um); below that, interpolation costs more than it
      gains, so no correction. Without an ephys-prep motion estimate, the sorter's own DREDge correction (>= 96 ch).
    """
    off = {"sorter_corrects": False, "interpolate": False}
    if meta.get("motion_corrected"):
        return {**off, "reason": "ephys-prep already interpolated the recording (motion_corrected); not corrected twice"}
    if mode == "off":
        return {**off, "reason": "DRIFT_CORRECTION=off"}
    drift = (meta.get("motion") or {}).get("max_drift_range_um")
    if sorter == "kilosort4":
        if mode == "auto" and n_channels < MIN_DRIFT_CHANNELS:
            return {**off, "reason": f"{n_channels} channels < {MIN_DRIFT_CHANNELS}: too few to estimate drift (AIND rule)"}
        return {"sorter_corrects": True, "interpolate": False, "reason": "Kilosort4's own drift correction (nblocks)"}
    if has_motion and drift is not None:
        if mode == "on" or drift > threshold_um:
            return {"sorter_corrects": False, "interpolate": True,
                    "reason": f"interpolated with ephys-prep's DREDge motion (largest drift range {drift:.1f} um"
                              + (f" > MOTION_THRESHOLD_UM={threshold_um:g})" if drift > threshold_um else "; DRIFT_CORRECTION=on)")}
        return {**off, "reason": f"largest drift range {drift:.1f} um <= MOTION_THRESHOLD_UM={threshold_um:g}: not worth interpolating"}
    if mode == "auto" and n_channels < MIN_DRIFT_CHANNELS:
        return {**off, "reason": f"no ephys-prep motion estimate and {n_channels} channels < {MIN_DRIFT_CHANNELS}"}
    return {"sorter_corrects": True, "interpolate": False, "reason": "no ephys-prep motion estimate: the sorter's own DREDge correction"}


def sorter_params(sorter: str, p: dict, drift_on: bool, seed: int) -> dict:
    if sorter == "kilosort4":
        k = deepcopy(KS4_AIND)
        k.update(batch_size=p["KS_BATCH_SIZE"], Th_universal=p["KS_TH_UNIVERSAL"], Th_learned=p["KS_TH_LEARNED"],
                 clear_cache=p["KS_CLEAR_CACHE"], cluster_init_seed=seed)
        k["do_correction"] = drift_on
        k["nblocks"] = p["KS_NBLOCKS"] if drift_on else 0
        return k
    if sorter == "lupin":
        k = deepcopy(LUPIN_AIND)
        k.update(apply_motion_correction=drift_on, detect_threshold=p["DETECT_THRESHOLD"], seed=seed)
        return k
    if sorter == "spykingcircus2":
        k = deepcopy(SC2_AIND)
        k["apply_motion_correction"] = drift_on
        k["seed"] = seed
        k["detection"] = {"method": "matched_filtering", "method_kwargs": {"peak_sign": "neg", "detect_threshold": p["DETECT_THRESHOLD"]},
                          "pipeline_kwargs": {}}
        return k
    raise ValueError(sorter)
