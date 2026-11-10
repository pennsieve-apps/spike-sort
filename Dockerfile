FROM python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /processor

# cpu (default for a builder that passes no build argument) or cuda; app.yml's variants set it.
#   cpu:  CPU-only PyTorch, Lupin / SpykingCircus2, no Kilosort4, no NVIDIA libraries.
#   cuda: Kilosort4 4.1.7 + PyTorch 2.14.1 for CUDA 12.6 (NVIDIA runtime from pip wheels; no CUDA base image needed:
#         Kilosort4 runs only PyTorch kernels, and the platform's driver 550 runs CUDA 12.x wheels).
ARG TORCH_FLAVOUR=cpu

COPY requirements.txt requirements-${TORCH_FLAVOUR}.txt /processor/
RUN pip install --no-deps --only-binary=:all: -r /processor/requirements.txt -r /processor/requirements-${TORCH_FLAVOUR}.txt \
    && find /usr/local/lib/python3.12/site-packages -name '*.pyc' -delete \
    && find /usr/local/lib/python3.12/site-packages -path '*/torch/*' -prune -o -type d -name tests -prune -exec rm -rf {} + \
    && rm -rf /usr/local/lib/python3.12/site-packages/torch/include

COPY processor/ /processor/processor/
COPY VERSION LICENSE THIRD_PARTY.md /processor/
COPY licenses/ /processor/licenses/

# Lambda-safe: no cache paths here (HOME and /tmp are replaced at run time); processor.common.setup_writable_dirs
# points matplotlib / numba / triton caches at $TMPDIR at startup. USER: getpass with an arbitrary uid.
ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    USER=app \
    TORCH_FLAVOUR=${TORCH_FLAVOUR}

RUN python -c "import matplotlib.pyplot, spikeinterface.full, spikeinterface.sorters as ss, torch; \
assert ss.sorter_dict['lupin'].is_installed() and ss.sorter_dict['spykingcircus2'].is_installed(); \
import os; os.environ['TORCH_FLAVOUR'] == 'cuda' and __import__('kilosort')"

RUN useradd --uid 1000 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin processor
USER 1000

ENTRYPOINT ["python3", "-m", "processor.entry"]
