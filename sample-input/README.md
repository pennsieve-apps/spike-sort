# sample-input

spike-sort reads `ephys-prep`'s output (`ephys-prep/<session_id>/session.json` + `recording/`).

## Synthetic fixture (no download)

```sh
make fixture PYTHON=.venv/bin/python   # data/fixture/input/ephys-prep/sub-fixture_ses-01/
make run PYTHON=.venv/bin/python       # Docker (cpu variant, Lupin) -> data/output/
```

`tests/make_fixture.py` (numpy only) writes a SpikeInterface binary folder exactly as ephys-prep saves it: 64
channels (2 × 32, 20 µm pitch), 30 s at 30 kHz from 100 s on the session clock, 12 planted units (80-220 µV) with
10 µm of drift, already "preprocessed" (white noise only), and `ground_truth/sub-fixture_ses-01_gt.npz`.

## Real data

Run `ephys-prep` on a raw recording first (see its `sample-input/README.md`). The workflow's demo:

- **IBL Brain Wide Map, DANDI:000409** (CC-BY-4.0, doi:10.48324/dandi.000409/0.260309.1324),
  `sub-NR-0031_ses-caa5dddc`, Probe01, 10 min: ephys-prep output in the workflow repo's
  `data/work/ibl/ephys-prep/`, this app's cpu output in `data/work/ibl/spike-sort/`. IBL's own sorting of the
  session is in the processed NWB (`desc-processed_behavior+ecephys.nwb`, units table) for comparison.
- **Ground truth, DANDI:001250** (CC-BY-4.0, doi:10.48324/dandi.001250/0.261108.1908): ephys-prep with
  `BAD_CHANNELS=dead-noisy`, then this app; score with
  `python scripts/score_ground_truth.py <the NWB> spike-sort/<session_id>/sorting` (units with ≥ 10 spikes).
