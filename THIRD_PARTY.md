# Third-party software

This app is **GPL-3.0-or-later** (see `LICENSE`) because the gpu image imports **Kilosort4 (GPL-3.0)** in the same
Python process (SpikeInterface's `run_sorter("kilosort4")` calls `kilosort.run_kilosort`). Kilosort's own
licence is in [`licenses/kilosort-LICENSE.txt`](licenses/kilosort-LICENSE.txt); its source is at
https://github.com/MouseLand/Kilosort (tag v4.1.7) and on PyPI (`kilosort==4.1.7`). Kilosort's PyPI metadata says
"BSD", but its PyPI classifier and its repository's LICENSE say GPL-3.0, which we follow.

The **cpu image** contains no GPL code (no Kilosort): only the permissive components below.

Components (pinned in `requirements.txt` and `requirements-cpu.txt` / `requirements-cuda.txt`; licences read
from the package metadata on PyPI / in the built images):

| Component | Image | What it does here | Authors / copyright holder | Licence | Source |
|---|---|---|---|---|---|
| Kilosort 4.1.7 | gpu | The GPU spike sorter | Marius Pachitariu, Shashwat Sridhar, Jacob Pennington, Carsen Stringer (MouseLand, HHMI Janelia) | GPL-3.0 | https://github.com/MouseLand/Kilosort |
| SpikeInterface 0.105.4 | both | Loading recordings, `run_sorter`, Lupin and SpykingCircus2 sorters, DREDge, sorting folders | Alessio Buccino, Samuel Garcia and the SpikeInterface contributors | MIT | https://github.com/SpikeInterface/spikeinterface |
| probeinterface 0.4.2, Neo 0.14.6 | both | Probe geometry; dependency of SpikeInterface | Samuel Garcia, Alessio Buccino, Chris Halcrow; Neo authors and contributors | MIT; BSD-3-Clause | https://github.com/SpikeInterface/probeinterface, https://github.com/NeuralEnsemble/python-neo |
| PyTorch 2.14.1 (CPU build in cpu, CUDA 12.6 build in gpu) | both | Kilosort4 (GPU), Lupin, DREDge | PyTorch Team (Meta and contributors) | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT (package metadata; PyTorch itself is BSD-3-Clause) | https://github.com/pytorch/pytorch |
| NVIDIA CUDA runtime wheels (nvidia-cublas-cu12, -cuda-runtime, -cudnn, -cufft, -curand, -cusolver, -cusparse, -cusparselt, -nccl, -nvjitlink, -nvtx, -nvshmem, -cufile, -cuda-nvrtc, -cuda-cupti; cuda-toolkit 12.6.3) | gpu | CUDA libraries PyTorch loads | NVIDIA Corporation | NVIDIA proprietary licence (redistributable with applications under the CUDA EULA) | https://pypi.org/project/nvidia-cuda-runtime-cu12/ |
| cuda-bindings 12.9.10, cuda-pathfinder 1.8.6 | gpu | CUDA Python bindings (PyTorch dependency) | NVIDIA Corporation | Apache-2.0 | https://github.com/NVIDIA/cuda-python |
| triton 3.8.0 | gpu | PyTorch kernel compiler | Philippe Tillet and contributors | MIT | https://github.com/triton-lang/triton |
| faiss-cpu 1.15.1 | gpu | Nearest neighbours in Kilosort4's clustering | Meta AI Research | MIT | https://github.com/facebookresearch/faiss |
| psutil 7.2.2 | gpu | Kilosort4's resource logging | Giampaolo Rodola | BSD-3-Clause | https://github.com/giampaolo/psutil |
| hdbscan 0.8.44 | both | SpykingCircus2 clustering | Leland McInnes and contributors | BSD (package metadata) | https://github.com/scikit-learn-contrib/hdbscan |
| NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1 | both | Arrays, signal processing, clustering | NumPy, SciPy, scikit-learn developers | BSD-3-Clause | numpy.org, scipy.org, scikit-learn.org |
| numba 0.68.1, llvmlite 0.50.0 | both | JIT kernels of SpikeInterface / Kilosort | Numba developers (Anaconda, Inc.) | BSD-2-Clause; llvmlite BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | https://numba.pydata.org |
| Matplotlib 3.11.2 | both | Units figure; Kilosort4's diagnostics.png | Matplotlib Development Team | Matplotlib licence (PSF-based, BSD-compatible) | https://matplotlib.org |
| zarr 2.18.7, numcodecs 0.15.2 | both | Dependencies of SpikeInterface | zarr developers | MIT | https://github.com/zarr-developers |
| certifi, tqdm | both | Dependencies | — | MPL-2.0 (tqdm: MPL-2.0 AND MIT) | PyPI |
| Other transitive dependencies (cloudpickle, joblib, narwhals, networkx, pydantic, quantities, sympy, mpmath, threadpoolctl, filelock, fsspec, Jinja2, pillow, requests, ...) | both | Versions in `requirements.txt` | their authors | MIT / BSD / Apache-2.0 / PSF-2.0 / MIT-CMU (Pillow); quantities BSD-style | PyPI |
| python:3.12.16-slim-bookworm | both | Base image (Debian 12 + CPython 3.12) | Python Software Foundation; Debian | PSF (Python); Debian packages under their own free licences | https://hub.docker.com/_/python |

## Methods used, not bundled as separate code

- The **AIND ephys pipeline** (Allen Institute for Neural Dynamics, MIT,
  https://github.com/AllenNeuralDynamics/aind-ephys-pipeline): `pipeline/default_params.json` (v1.5.0) gives the
  sorter parameters and the "fewer than 96 channels: no drift correction" rule; no code is copied.
- **DREDge** (Windolf et al. 2025; https://github.com/evarol/dredge, MIT) as implemented in SpikeInterface.

## Not bundled

Demo data: IBL Brain Wide Map, DANDI:000409 (CC-BY-4.0); Laquitaine et al. ground-truth simulation, DANDI:001250
(CC-BY-4.0).

No base image from `spikeinterface/kilosort4-base` is used: its tags are mutable and its contents (a ~6.8 GB
CUDA devel image) are larger than needed; this image is built from `python:3.12-slim` pinned by digest, with the
CUDA runtime from NVIDIA's pip wheels.
