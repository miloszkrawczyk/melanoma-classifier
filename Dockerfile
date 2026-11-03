# Debian with Python already on it, stripped of the compilers and tooling the
# full python:3.12 image carries. Around 150 MB against roughly a gigabyte.
#
# 3.12 rather than 3.11 because numpy 2.5 dropped support for anything older.
# Development happens on 3.14; the gap is fine, since pyproject only asks for
# 3.11 or newer and nothing here uses syntax added after 3.12.
#
# Not alpine, which looks smaller and is a trap for anything involving
# scientific Python: it ships musl instead of glibc, so the prebuilt wheels on
# PyPI do not fit and pip compiles numpy and torch from source. The build goes
# from two minutes to forty and the image comes out bigger anyway.
FROM python:3.12-slim

# uv, taken from its own published image rather than installed with a script.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Hugging Face Spaces runs containers as user 1000, not root. Files copied in
# as root would be unwritable, and Gradio needs somewhere to put the uploads it
# is handed. Creating the user here means the image behaves the same locally
# and deployed, rather than working on my laptop and falling over on theirs.
RUN useradd --create-home --uid 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# The single most important line in this file. On Linux, plain "pip install
# torch" fetches the CUDA build and drags in a dozen nvidia-* packages worth
# some 2.5 GB. This service runs on a CPU and has nothing to point them at.
#
# The cpu index holds the same torch built without any of it. It is still not
# small, 596 MB unpacked against a 200 MB download, but the nvidia packages
# are gone entirely. Check with: torch.__version__ ends in "+cpu".
RUN uv pip install --system --no-cache \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0 torchvision==0.28.0

# Everything else comes from PyPI as usual. Listed out rather than installed
# from uv.lock on purpose: the lock also carries pandas, kaggle, matplotlib and
# pytest, none of which the web service imports. This list is what it actually
# needs, and saying so in one place is worth more than the reproducibility of
# resolving the lock again.
RUN uv pip install --system --no-cache \
    fastapi==0.141.1 \
    "uvicorn[standard]==0.52.1" \
    gradio==6.22.0 \
    pillow==12.3.0 \
    numpy==2.5.1 \
    python-multipart==0.0.32

# Code and weights last, because Docker caches each step and reuses it while
# the step above has not changed. Installing 200 MB of torch again on every
# edit to a Python file would make each rebuild a several-minute wait.
COPY src/ src/
COPY api/ api/
COPY frontend/ frontend/

# Not in the GitHub repo: 43 MB of weights are gitignored, so a plain clone
# followed by "docker build" stops here saying the file is missing. Either
# rebuild them with "uv run python -m src.train resnet", or convert the ones
# you already have with "uv run python -m scripts.convert_weights".
#
# safetensors, not the .pth. Naming the old file here would have shipped
# whatever was trained in August while the code outside the image read the
# newer weights, and nothing would have failed to say so.
COPY weights/resnet.safetensors weights/resnet.safetensors

# Ownership handed over after the copying, since the user cannot write into
# /app otherwise. Everything below this line runs unprivileged.
RUN chown -R user:user /app
USER user

# 7860 is what Hugging Face Spaces expects. Any port works elsewhere.
EXPOSE 7860

# 0.0.0.0 rather than 127.0.0.1: inside a container, localhost means the
# container itself, so a server bound to it would be unreachable from outside.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
