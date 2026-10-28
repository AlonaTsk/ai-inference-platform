# Build from the repo root:
#   CPU:  docker build -f gpu/Dockerfile -t ai-inference-platform/gpu-bench:cpu .
#   CUDA: docker build -f gpu/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 \
#           -t ai-inference-platform/gpu-bench:cu124 .
# CUDA wheels bundle the CUDA runtime; the node only needs the NVIDIA driver and
# the NVIDIA container toolkit / device plugin, not a CUDA base image.
FROM python:3.11-slim

ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" numpy --index-url "${TORCH_INDEX}" --extra-index-url https://pypi.org/simple

WORKDIR /srv
COPY inference/app/__init__.py inference/app/data.py inference/app/
COPY gpu/textclf.py gpu/benchmark.py gpu/
ENV PYTHONPATH=/srv/inference

RUN useradd --uid 10001 --no-create-home app
USER 10001
WORKDIR /srv/gpu
ENTRYPOINT ["python", "benchmark.py"]
