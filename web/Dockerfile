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
RUN pip install --no-cache-dir numpy==2.4.4 numba==0.65.1 scipy==1.17.1 threadpoolctl==3.6.0 \
 && if [ "$GPU" = "1" ]; then pip install --no-cache-dir "jax[cuda12]==0.10.1"; else pip install --no-cache-dir jax==0.10.1; fi

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
    GNOME3D_JAX_CACHE=/results/.jax-cache

EXPOSE 8000
CMD ["uvicorn", "gnome3d_web.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
