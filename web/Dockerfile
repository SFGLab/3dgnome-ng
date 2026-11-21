# The 3D-GNOME site and its worker in one image, the frontend built in a Node stage. Python 3.12,
# numpy 2.4.4, numba 0.65.1, scipy 1.17.1, jax 0.10.1. Build from the repository root:
#   docker build -f web/Dockerfile -t gnome3d-web .
#   docker build -f web/Dockerfile --build-arg GPU=1 -t gnome3d-web:gpu .
FROM node:22-slim AS frontend
# One dependency of the lockfile is a GitHub repository at its ssh address. The image has no ssh key
# and git reads GitHub over https.
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

# libcurl and htslib are built here from their current releases. Debian 13 ships versions with open
# advisories and the site opens contact maps at addresses a user gives. This libcurl speaks http and
# https alone. hic-straw has no wheel for this Python and is compiled against this libcurl. The
# compiler and the headers stay in this stage.
FROM python:3.12-slim AS native
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ make curl ca-certificates xz-utils bzip2 \
      libssl-dev zlib1g-dev libbz2-dev liblzma-dev \
 && rm -rf /var/lib/apt/lists/*
ARG CURL_VERSION=8.23.0
ARG CURL_SHA256=8dadd7d2536a314f982ad05a651b713f426bf7d02cc2c76cffb9796f2218e6c5
ARG HTSLIB_VERSION=1.25
ARG HTSLIB_SHA256=158c271e4e2986bb2dd11283b640edc10c56695b293df77a109ce99480303730
WORKDIR /build
RUN curl -fsSLO https://curl.se/download/curl-$CURL_VERSION.tar.xz \
 && echo "$CURL_SHA256  curl-$CURL_VERSION.tar.xz" | sha256sum -c - \
 && tar xf curl-$CURL_VERSION.tar.xz && cd curl-$CURL_VERSION \
 && ./configure --prefix=/opt/native --with-openssl --with-ca-bundle=/etc/ssl/certs/ca-certificates.crt \
      --disable-static --disable-manual --disable-docs \
      --disable-ldap --disable-ldaps --disable-ftp --disable-file --disable-dict --disable-telnet \
      --disable-tftp --disable-pop3 --disable-imap --disable-smb --disable-smtp --disable-gopher \
      --disable-mqtt --disable-rtsp --disable-ipfs --disable-websockets \
      --without-libpsl --without-brotli --without-zstd --without-nghttp2 --without-libidn2 \
      --without-librtmp --without-libssh2 \
 && make -j"$(nproc)" >/dev/null && make install >/dev/null
RUN curl -fsSLO https://github.com/samtools/htslib/releases/download/$HTSLIB_VERSION/htslib-$HTSLIB_VERSION.tar.bz2 \
 && echo "$HTSLIB_SHA256  htslib-$HTSLIB_VERSION.tar.bz2" | sha256sum -c - \
 && tar xf htslib-$HTSLIB_VERSION.tar.bz2 && cd htslib-$HTSLIB_VERSION \
 && ./configure --prefix=/opt/native --disable-libcurl --disable-s3 --disable-gcs --disable-plugins \
      LDFLAGS="-Wl,-rpath,/opt/native/lib" \
 && make -j"$(nproc)" >/dev/null && make install >/dev/null
RUN CPPFLAGS=-I/opt/native/include LDFLAGS="-L/opt/native/lib -Wl,-rpath,/opt/native/lib" \
    pip wheel --no-cache-dir --no-deps -w /wheels hic-straw==1.3.1

FROM python:3.12-slim

RUN apt-get update \
 && apt-get upgrade -y \
 && rm -rf /var/lib/apt/lists/*
COPY --from=native /opt/native/lib/libcurl.so* /opt/native/lib/libhts.so* /opt/native/lib/
COPY --from=native /opt/native/bin/tabix /opt/native/bin/bgzip /usr/local/bin/

WORKDIR /app
ARG GPU=0
RUN pip install --no-cache-dir pip==26.3 \
 && pip install --no-cache-dir numpy==2.4.4 numba==0.65.1 scipy==1.17.1 threadpoolctl==3.6.0 \
 && if [ "$GPU" = "1" ]; then pip install --no-cache-dir "jax[cuda12]==0.10.1"; else pip install --no-cache-dir jax==0.10.1; fi

# HiCDiffusion at a fixed commit with its own interpreter, Python 3.11 and its pinned packages. Five
# pins with published advisories are raised by an override file. HICDIFFUSION=0 leaves the step out
# of the image.
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
        --override /opt/hicdiffusion-raised.txt \
   && apt-get purge -y git && apt-get autoremove -y ; \
    fi

COPY pyproject.toml README.md SETTINGS.md LICENSE ./
COPY gnome3d ./gnome3d
# The build context has no git history. The version of the engine is the build argument.
ARG GNOME3D_VERSION=0.0.0
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_GNOME3D=$GNOME3D_VERSION pip install --no-cache-dir --no-deps .

COPY web/requirements.txt ./web/requirements.txt
COPY --from=native /wheels /wheels
RUN pip install --no-cache-dir --find-links /wheels -r web/requirements.txt
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
