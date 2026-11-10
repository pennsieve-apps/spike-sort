# spike-sort

This app runs **[Kilosort4](https://github.com/MouseLand/Kilosort)**, developed by Marius Pachitariu, Shashwat
Sridhar, Jacob Pennington and Carsen Stringer at HHMI Janelia Research Campus
([Pachitariu et al. 2024, Nature Methods](https://doi.org/10.1038/s41592-024-02232-7)), and, on CPUs,
**Lupin** and **SpykingCircus2**, the sorters built into
**[SpikeInterface](https://github.com/SpikeInterface/spikeinterface)** by Samuel Garcia (CNRS, Lyon), Pierre Yger
(Lille Neuroscience & Cognition, INSERM / Univ. Lille), Alessio Buccino (Allen Institute for Neural Dynamics /
CatalystNeuro) and colleagues ([Garcia et al. 2026, eLife](https://doi.org/10.7554/eLife.110588);
[Yger et al. 2018, eLife](https://doi.org/10.7554/eLife.34518); SpikeInterface:
[Buccino et al. 2020, eLife](https://doi.org/10.7554/eLife.61834)), with the parameters of the
**[AIND ephys pipeline](https://github.com/AllenNeuralDynamics/aind-ephys-pipeline)** (Allen Institute for Neural
Dynamics; [Buccino et al. 2026, eLife](https://doi.org/10.7554/eLife.110170)). **Pennsieve packages these
methods as a platform app; it did not develop them.** The demo data are the International Brain Laboratory's
Brain Wide Map recordings ([DANDI:000409](https://doi.org/10.48324/dandi.000409/0.260309.1324)) and Laquitaine et
al.'s simulated ground truth ([DANDI:001250](https://doi.org/10.48324/dandi.001250/0.261108.1908)).

**What it does.** An extracellular probe picks up the spikes of many neurons at once. Spike sorting finds the
spikes in the preprocessed signal and groups them into putative neurons ("units") by their waveform across
neighbouring channels, keeping track of the brain moving along the probe (drift). This is the second step of
the **Neuropixels Spike Sorting** workflow:

`ephys-prep → spike-sort → ephys-curate → neuro-report`

The output is the *uncurated* sorting: Kilosort4 typically finds hundreds of units on a Neuropixels probe, many of
them multi-unit activity or noise. `ephys-curate` computes waveforms and quality metrics and labels them.

**Licence: GPL-3.0-or-later.** Kilosort4 is GPL-3.0 and this app imports it in the same process, so the app is
GPL too (see [Licences](#licences)).

## Variants

| Variant | Sorter | Compute | When |
|---|---|---|---|
| `gpu` (default) | Kilosort4 4.1.7 (PyTorch 2.14.1, CUDA 12.6) | NVIDIA GPU (T4 16 GB or larger) | The standard choice for Neuropixels: fast and accurate. Without a visible GPU it exits with code 3 and says to use the `cpu` variant (Kilosort4 on a CPU is "only for testing", per its authors). |
| `cpu` | Lupin (default) or SpykingCircus2 (`SORTER`) | standard (CPU) | No GPU available, or a second opinion. **Lupin** is SpikeInterface's own modular sorter; in its authors' benchmarks (Garcia et al. 2026: simulated recordings and proxy metrics) it is comparable to Kilosort4, with fewer false positives. It has not been benchmarked independently yet, so Kilosort4 stays the default. SpykingCircus2 is the second CPU option (accurate but slower in the same benchmarks). MountainSort5 is not offered for Neuropixels: it has no drift correction. |

Both images also contain Lupin and SpykingCircus2, so `SORTER=lupin` works on the gpu image too (on CPU).

**Why Kilosort4 is the default.** It is the reference sorter of the AIND pipeline (Buccino et al. 2026), which
found more matched units with Kilosort4 than with Kilosort2.5 on hybrid ground truth, and of most Neuropixels labs.

**Kilosort4 on a T4.** With `batch_size=60000` (2 s at 30 kHz) a 384-channel probe uses a few GB of the T4's 16 GB
(the peak is in clustering and grows with the number of spikes; it is recorded in `sort.json` as
`peak_vram_mb`). If a long or very active recording runs out of GPU memory: `KS_CLEAR_CACHE=true` (slower) or
`KS_BATCH_SIZE=30000`. On a larger GPU nothing needs to change. GPU numbers in this README are *to be measured on
the dev compute node*.

## Method (per session)

1. **Input**: `ephys-prep/<session_id>/recording/` (int16, preprocessed: phase shift, 300 Hz high-pass, bad
   channels removed, common median reference) with its `session.json`; or any SpikeInterface recording folder.
2. **Drift is corrected exactly once** (`DRIFT_CORRECTION=auto`, after Garcia et al. 2026 and AIND). ephys-prep
   always estimates the motion with DREDge and, by default, leaves the data as they are.
   - **Kilosort4** keeps its own drift correction (`nblocks=5`, AIND), whatever the drift; off with fewer than 96
     channels (AIND's `min_drift_channels`).
   - **Lupin / SpykingCircus2**: the recording is interpolated (kriging) with **ephys-prep's DREDge estimate** only
     when its largest drift range exceeds `MOTION_THRESHOLD_UM` (10 µm); below that, interpolation costs more
     accuracy than it recovers, so nothing is corrected. The sorter's own correction is off (no re-estimation).
     Without an ephys-prep estimate (a bare recording folder), the sorter's own DREDge correction is used (≥ 96
     channels).
   - If ephys-prep already interpolated (`session.json` `motion_corrected: true`), nothing is corrected
     (Kilosort4 `nblocks=0, do_correction=false`).
   `sort.json` and `summary.json` record the decision and why.
3. **Kilosort4** (AIND `default_params.json`, v1.5.0): `batch_size 60000, nblocks 5, Th_universal 9, Th_learned 8,
   dminx 32, whitening_range 32, highpass_cutoff 300, do_CAR true, nt 61, nskip 25, binning_depth 5, sig_interp 20,
   drift_smoothing [0.5, 0.5, 0.5], min_template_size 10, template_sizes 5, nearest_chans 10, nearest_templates 100,
   templates_from_data true, n_templates 6, n_pcs 6, Th_single_ch 6, acg_threshold 0.2, ccg_threshold 0.25,
   cluster_downsampling 20, duplicate_spike_ms 0.25`; Kilosort 4.1.7's own defaults otherwise
   (`max_channel_distance 32`, `artifact_threshold inf`); `cluster_init_seed = SEED`. Kilosort4 reads the int16
   binary in place (no copy) and applies its own high-pass, CAR and whitening.
4. **Lupin** (AIND): `apply_preprocessing` with an empty `preprocessing_dict` (the input is already filtered and
   referenced; Lupin whitens), `template_matching_engine circus-omp`, `template_sparsify_threshold 1.5`,
   `detect_threshold 5`, no preprocessing cache (streams from disk). **SpykingCircus2** (AIND):
   `apply_preprocessing false`, matched-filter detection at 5 × MAD, iterative HDBSCAN, circus-omp matching.
5. Empty units are removed and spikes beyond the end of the recording dropped (AIND); the sorting is saved as a
   SpikeInterface `numpy_folder`. Spike times are **sample indices of the ephys-prep recording**: seconds on the
   session clock are that recording's times at those samples (`recording.sample_index_to_time`), which
   `ephys-curate` applies. A heartbeat line every minute (elapsed time, memory, GPU memory) shows a long sort is
   alive; Kilosort4's own progress is logged too.

**No /dev/shm needed.** SpikeInterface shares arrays between its worker processes through POSIX shared memory
(circus-omp / wobble template overlaps, template accumulators). Containers get only 64 MB of `/dev/shm` (ECS tasks
unless `sharedMemorySize` is set) and Lambda has none, so Lupin and SpykingCircus2 crashed on real Neuropixels data
(SIGBUS, BrokenProcessPool). The app replaces `multiprocessing.shared_memory.SharedMemory` with a file-backed
equivalent in `$TMPDIR/shm` (`processor/fileshm.py`, installed in the main process and in every spawned worker
by `processor/entry.py`) and also turns shared memory off in SpykingCircus2's matching. Kilosort4 uses no
multiprocessing or shared memory (single process; PyTorch on the GPU); to be confirmed on the GPU node with the
default 64 MB `/dev/shm`.

## Ports

| Port | Kind | Files | Notes |
|---|---|---|---|
| `recording` | input | `ephys-prep/<session_id>/{session.json, recording/}` | Found anywhere below (symlinks followed); every session is sorted. |
| `sorting` | output | `spike-sort/...` (below) | The only output: the whole of `OUTPUT_DIR`. |

## Parameters

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `SORTER` | string | `kilosort4` (gpu) / `lupin` (cpu) | `kilosort4` (gpu variant only), `lupin`, `spykingcircus2`. |
| `DRIFT_CORRECTION` | string | `auto` | `auto` (see Method 2), `on` (CPU sorters: interpolate whatever the drift), `off`. Never twice. |
| `MOTION_THRESHOLD_UM` | number | `10` | CPU sorters: interpolate with ephys-prep's motion only above this drift (µm). Not used by Kilosort4. |
| `KS_BATCH_SIZE` | integer | `60000` | Kilosort4 batch (samples). Lower for GPU memory. |
| `KS_NBLOCKS` | integer | `5` | Kilosort4 non-rigid drift blocks when drift correction is on (`1` = rigid). |
| `KS_TH_UNIVERSAL` | number | `9` | Kilosort4 threshold, universal templates. |
| `KS_TH_LEARNED` | number | `8` | Kilosort4 threshold, learned templates. |
| `KS_CLEAR_CACHE` | boolean | `false` | Free the PyTorch GPU cache between steps (for out-of-memory errors). |
| `DETECT_THRESHOLD` | number | `5` | Lupin / SpykingCircus2 detection threshold (× noise MAD). |
| `THREADS` | integer | `0` | CPU worker processes; `0` = the container's CPU quota. |
| `SEED` | integer | `42` | Seeds Kilosort4's clustering, Lupin / SC2, torch and numpy. |

Exit codes: 0 ok; 1 no input or every session failed; 2 bad parameters; **3 Kilosort4 requested but no GPU (or
the cpu image)**.

## Outputs (`OUTPUT_DIR/spike-sort/`)

| File | Contents |
|---|---|
| `<session_id>/sorting/` | SpikeInterface `numpy_folder` (`spikeinterface.load`): `spikes.npy` (sample_index, unit_index, segment_index), unit ids, sampling frequency. |
| `<session_id>/sorter_output/` | Kilosort4: its Phy-compatible output (`spike_times.npy`, `spike_clusters.npy`, `templates.npy`, `amplitudes.npy`, `pc_features.npy`, `cluster_KSLabel.tsv`, `cluster_ContamPct.tsv`, `ops.npy`, `kilosort4.log`, `diagnostics.png`, `params.py` with `dat_path` set to the ephys-prep binary's file name). Lupin / SC2: SpikeInterface's sorter log and parameters. |
| `<session_id>/kilosort_motion/` | Kilosort4's own drift estimate as a SpikeInterface `Motion` (when it corrected drift). |
| `<session_id>/sort.json` | `sorter, sorter_version, spikeinterface, parameters` (all, as run), `drift_correction {by_sorter, interpolated_with_ephys_prep_motion, reason, threshold_um, ephys_prep_max_drift_range_um, ephys_prep_motion_corrected}, n_units, n_units_before_cleaning, n_spikes, median_firing_rate_hz, n_channels, sampling_frequency, t_start, duration_s, wall_seconds, realtime_factor, gpu {name, total_vram_mb, peak_vram_mb}`, the session's sheet columns. |
| `<session_id>/<session_id>_units.png` | Units by depth and firing rate, and every spike at its unit's depth over time. |
| `summary.json` | Per session: sorter, units, spikes, firing rate, wall time, real-time factor, GPU and peak VRAM; figures with captions. |
| `provenance.json` | App and tool versions, variant, parameters, inputs, timings, peak memory and VRAM, citations. |

To open a Kilosort4 result in Phy, copy (or link) `ephys-prep/<session_id>/recording/traces_cached_seg0.raw` next
to `sorter_output/params.py`.

## Results on the demo data

**Synthetic fixture** (`tests/make_fixture.py`: 64 channels, 30 s, 12 planted units, 80-220 µV): Lupin finds all
12 with accuracy 0.98-1.0 (2 extra small units); Kilosort4 (run on CPU for the test only) finds 12 units, 1441
spikes for 1403 planted. **SpikeInterface ground truth** (`generate_ground_truth_recording`, 48 channels, 15 units,
40 s): Lupin recovers at least 12 of 15 units with accuracy ≥ 0.8 (test).

**Ground truth, DANDI:001250** (`sub-demo-npx-bio-evoked`, a biophysically detailed simulation of a Neuropixels
1.0 shank in rat cortex, 120 channels, 20 kHz, 60 s, 1836 simulated neurons of which 788 fire ≥ 10 spikes; ephys-prep
with `BAD_CHANNELS=dead-noisy`, then Lupin, cpu; `scripts/score_ground_truth.py`; drift 0.2 µm, so nothing is interpolated): Lupin found 180 units;
**36 of the 788 ground-truth neurons are recovered with accuracy ≥ 0.8** (52 with ≥ 0.5; median accuracy of the
matched ones 0.96). Recovery rises with activity: 10 of 347 neurons with 10-49 spikes, 5 of 191 with 50-99, 20 of
243 with 100-299 and 1 of 7 with ≥ 300 spikes (21 of the 250 with ≥ 100 spikes). Lupin is not bit-reproducible
across machines: an independent rerun on another machine gave 190 units and 39 of 788 at ≥ 0.8. That is the point of Laquitaine et al.: in a
realistic cortex most neurons are small, sparse or overlapping and no sorter recovers them; sorted units are
biased towards large, active neurons. Kilosort4 on the same data: to be measured on the dev compute node.

**IBL, `sub-NR-0031_ses-caa5dddc`, Probe01** (10 min, NP 1.0, ephys-prep defaults; drift 8.9 µm, below the threshold, so not
interpolated): **Lupin (cpu) found 686 units** (2.72 M spikes, median 2.7 Hz) in the workflow repo's
`data/work/ibl/spike-sort/`. Against IBL's own sorting of the same probe and window (processed NWB units table,
`scripts/compare_reference.py`, 0.4 ms matching): **48 of IBL's 70 best units** (`ibl_quality_score` = 1, ≥ 10 spikes in the window; 83 have ≥ 1) have a
Lupin unit with agreement ≥ 0.8 (55 with ≥ 0.5, median agreement 0.97); over all 534 IBL units active in the
window, 105 match at ≥ 0.8 and 178 at ≥ 0.5. IBL's sorting is itself a (pyKilosort) sorting, not ground truth.
Kilosort4 (gpu) on the same data: to be measured on the dev compute node.

## Runtime

Measured on an M-series Mac (CPU only). **Native arm64, Lupin, 6 workers**: the IBL 10 min probe took **47 min**
(4.7 × real time; peak 3.2 GB in the main process, 4.3 GB with its workers per `time -l`); 120 s took 19 min;
the ground-truth file (120 channels, 60 s) 3.7 min. **Docker, linux/amd64 under emulation** (5-10 × slower):
the fixture (64 channels, 20 s) in the Lambda-style read-only container (2 CPUs, 4 GB) took 14 min with 2.0 GB
peak; the gpu image without a GPU exits 3 within seconds (1.4 GB peak at import). Images: cpu 1.9 GB, gpu 11.3 GB
(uncompressed; CUDA 12.6 runtime wheels).

Chosen: **cpu variant 8 vCPU, 16 GB**, timeout 12 h (Lupin ~3-5 × real time on 8 vCPU: ~3-5 h per
probe-hour, est.); **gpu variant**: T4, 30 GB host memory (Kilosort4 asks for 32 GB+ for long recordings), timeout
6 h. GPU wall time and VRAM: to be measured on the dev compute node. Not a Lambda step (no GPU, > 15 min).

## Limitations

- These are *uncurated* units: many are multi-unit activity or noise. Use `ephys-curate` before analysing them.
- A sorter's accuracy depends on the data: dense, small or bursting neurons, and fast drift, lower it. Ground-truth
  accuracy on a simulation (above) is an upper bound for what to expect on that kind of tissue, not a guarantee.
- Kilosort4 needs a CUDA GPU; on the cpu variant it is not available (exit 3). Kilosort4 results are not
  bit-for-bit reproducible across GPU models.
- Multi-segment recordings are not supported (ephys-prep writes one segment per session).
- Lupin's evidence so far is its authors' own simulations and proxy metrics; treat it as a CPU alternative, and
  prefer Kilosort4 where a GPU is available.
- The 10 µm interpolation threshold is a rule of thumb from simulations (Garcia et al. 2026).

## Licences

**GPL-3.0-or-later** (this repository, [`LICENSE`](LICENSE)): the gpu image runs Kilosort4 (GPL-3.0, MouseLand,
[`licenses/kilosort-LICENSE.txt`](licenses/kilosort-LICENSE.txt)) in the same Python process, so this wrapper is
distributed under the GPL as well. (Kilosort's PyPI metadata says "BSD"; its repository's LICENSE is GPL-3.0,
which we follow.) The rest of the image is permissive (SpikeInterface MIT, PyTorch BSD-3, faiss MIT, NVIDIA CUDA
runtime wheels under NVIDIA's licence): see [`THIRD_PARTY.md`](THIRD_PARTY.md). The cpu image contains no GPL code.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full author lists in
[`CITATION.cff`](CITATION.cff).

- **Kilosort4**: Pachitariu M, Sridhar S, Pennington J, Stringer C. Spike sorting with Kilosort4. *Nature Methods*
  2024;21:914-921. doi:10.1038/s41592-024-02232-7. PMID 38589517. (Kilosort asks users of Kilosort 1-4 to cite
  this paper.)
- **Lupin / modular spike sorting**: Garcia S, Halcrow C, Windolf C, McKenzie ZM, Adkisson-Floro P, Mayorquin HR,
  Dichter BK, Buccino AP, Yger P. Opening the black box toward a modular approach to spike sorting. *eLife*
  2026;15:RP110588. doi:10.7554/eLife.110588. PMID 42789325.
- **SpyKING CIRCUS**: Yger P, Spampinato GL, Esposito E, et al. A spike sorting toolbox for up to thousands of
  electrodes validated with ground truth recordings in vitro and in vivo. *eLife* 2018;7:e34518.
  doi:10.7554/eLife.34518. PMID 29557782.
- **SpikeInterface**: Buccino AP, Hurwitz CL, Garcia S, Magland J, Siegle JH, Hurwitz R, Hennig MH. SpikeInterface,
  a unified framework for spike sorting. *eLife* 2020;9:e61834. doi:10.7554/eLife.61834. PMID 33170122.
- **AIND pipeline** (parameters): Buccino AP, Sridhar A, Feng D, Svoboda K, Siegle JH. Efficient and reproducible
  pipelines for spike sorting large-scale electrophysiology data. *eLife* 2026;15:RP110170.
  doi:10.7554/eLife.110170. PMID 42554034.
- **DREDge** (Lupin / SC2 drift correction): Windolf C, Yu H, Paulk AC, et al. DREDge: robust motion correction for
  high-density extracellular recordings across species. *Nature Methods* 2025;22:788-800.
  doi:10.1038/s41592-025-02614-5. PMID 40050699. Garcia S, Windolf C, Boussard J, Dichter B, Buccino AP, Yger P. A
  modular implementation to handle and benchmark drift correction for high-density extracellular recordings.
  *eNeuro* 2024;11:ENEURO.0229-23.2023. doi:10.1523/ENEURO.0229-23.2023. PMID 38238082.
- **Demo data**: International Brain Laboratory. IBL - Brain Wide Map. DANDI Archive 2026.
  doi:10.48324/dandi.000409/0.260309.1324; International Brain Laboratory, et al. A brain-wide map of neural activity
  during complex behaviour. *Nature* 2025;645:177-191. doi:10.1038/s41586-025-09235-0. PMID 40903598.
  Laquitaine S, et al. Spike sorting biases and information loss in a detailed cortical model. DANDI Archive 2026.
  doi:10.48324/dandi.001250/0.261108.1908; bioRxiv 2024. doi:10.1101/2024.12.04.626805.
- **Software**: [Kilosort](https://github.com/MouseLand/Kilosort), [SpikeInterface](https://github.com/SpikeInterface/spikeinterface),
  [PyTorch](https://github.com/pytorch/pytorch).

## Acknowledgements

The MouseLand team at HHMI Janelia for Kilosort; the SpikeInterface team and contributors for Lupin,
SpykingCircus2 and the framework; the Allen Institute for Neural Dynamics for the open AIND ephys pipeline and its
parameters; the International Brain Laboratory and the DANDI Archive (BRAIN Initiative) for the open demo data;
Steeve Laquitaine and colleagues for the ground-truth simulation.
