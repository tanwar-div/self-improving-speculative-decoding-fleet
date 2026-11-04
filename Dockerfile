# CPU-only serving image. This host has no nvidia-container-toolkit, so the
# containerized fleet runs the target model on CPU with a small model; GPU
# numbers come from the host benchmark. Installing the CPU torch wheel keeps
# the image ~1GB instead of the ~7GB a CUDA base image would cost.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/hf \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libpq5 curl \
 && rm -rf /var/lib/apt/lists/*

# torch first, from the CPU index, so the big layer is cached independently of
# application dependency churn.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements-container.txt /app/
RUN pip install --no-cache-dir -r requirements-container.txt

COPY sql /app/sql
COPY src /app/src

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=180s --retries=10 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

ENTRYPOINT ["python", "-m", "fleet.cli"]
CMD ["serve"]
