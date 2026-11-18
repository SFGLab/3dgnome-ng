# Versions as on the team-frodo CUDA venv the smoke test ran on: Python 3.12, numpy 2.4.4,
# numba 0.65.1, scipy 1.17.1, jax 0.10.1.
# The 3D-GNOME site and its worker in one image, the frontend built in a Node stage. Build from
# the repository root:
#   docker build -f web/Dockerfile -t gnome3d-web .
#   docker build -f web/Dockerfile --build-arg GPU=1 -t gnome3d-web:gpu .
FROM node:22-slim AS frontend
# One dependency of the lockfile is a GitHub repository named by its ssh address. npm needs git
# for it, and the image has no ssh key, so git reads GitHub over https.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && git config --global url."https://github.com/".insteadOf "ssh://git@github.com/"
WORKDIR /web/frontend
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci
COPY web/frontend ./
# The path the site is served under, /new for a site at /new/. Empty is the root.
ARG BASE_PATH=
RUN BASE_PATH=$BASE_PATH npm run build

FROM python:3.12-slim

# hic-straw has no wheel for this Python and is compiled here, against libcurl and zlib.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tabix g++ libcurl4-openssl-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ARG GPU=0
RUN pip install --no-cache-dir pip==26.3 \
 && pip install --no-cache-dir numpy==2.4.4 numba==0.65.1 scipy==1.17.1 threadpoolctl==3.6.0 \
 && if [ "$GPU" = "1" ]; then pip install --no-cache-dir "jax[cuda12]==0.10.1"; else pip install --no-cache-dir jax==0.10.1; fi

# HiCDiffusion at a fixed commit with its own interpreter, Python 3.11 and the packages its
# requirements pin, which clash with the engine's. Five of those pins carry published
# advisories and are raised by an override file. On the raised set the model predicts the map it
# predicts on its own pins, within 3e-4 of a largest value of 4.5.
# HICDIFFUSION=0 leaves the step out of the image, and the site then does not offer it.
ARG HICDIFFUSION=1
ARG HICDIFFUSION_COMMIT=f892fb4df826158f0d398fbccc62832d7534f6db
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN if [ "$HICDIFFUSION" = "1" ]; then \
      apt-get update \
   && apt-get install -y --no-install-recommends git ca-certificates \
   && rm -rf /var/lib/apt/lists/* \
   && pip install --no-cache-dir uv==0.13.4 \
   && git clone -q https://github.com/SFGLab/HiCDiffusion /opt/hicdiffusion \
   && git -C /opt/hicdiffusion checkout -q "$HICDIFFUSION_COMMIT" \
   && uv venv -q --python 3.11 /opt/hicdiffusion-venv \
   && printf '%s\n' torch==2.15.0 torchvision==0.29.2 lightning==2.6.6 fsspec==2026.9.0 tqdm==4.66.3 > /opt/hicdiffusion-raised.txt \
   && uv pip install -q --no-cache --python /opt/hicdiffusion-venv/bin/python -r /opt/hicdiffusion/requirements.txt \
        --override /opt/hicdiffusion-raised.txt ; \
    fi

COPY pyproject.toml README.md SETTINGS.md ./
COPY gnome3d ./gnome3d
RUN pip install --no-cache-dir --no-deps .

COPY web/requirements.txt ./web/requirements.txt
RUN pip install --no-cache-dir -r web/requirements.txt
COPY web ./web
COPY --from=frontend /web/frontend/dist ./web/frontend/dist

ENV PYTHONPATH=/app/web:/app/web/vendor \
    ENGINE_PYTHON=python \
    DATA_DIR=/data \
    RESULTS_DIR=/results \
    GNOME3D_JAX_CACHE=/results/.jax-cache \
    HICDIFFUSION_DIR=/opt/hicdiffusion \
    HICDIFFUSION_PYTHON=/opt/hicdiffusion-venv/bin/python \
    HICDIFFUSION_MODELS=/data/hicdiffusion \
    REFERENCE_FASTA=/data/reference/hg38.fa

EXPOSE 8000
CMD ["uvicorn", "gnome3d_web.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
