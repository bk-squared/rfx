# syntax=docker/dockerfile:1
#
# openEMS solver-environment image for rfx's external-openEMS VESSL referee
# lanes (scripts/vessl_coax_two_port_referee.yaml,
# scripts/vessl_msl_phase_referee.yaml, scripts/vessl_crossval_external.yaml,
# scripts/vessl_coax_two_port_referee_mesh_refinement.yaml). Each of those
# lanes today pays apt-get + a source build of openEMS-Project (fparser +
# CSXCAD + openEMS, ~30-60 min) on EVERY run. Motivation: 2026-08-04, the
# cluster's archive.ubuntu.com route was unreachable and killed 4
# provisioning attempts across these lanes before any physics ran; a
# kakao-mirror + ForceIPv4 workaround (see
# scripts/vessl_coax_two_port_referee_mesh_refinement.yaml) is the bridge,
# this image is the fix -- GitHub Actions runners have healthy network to
# archive.ubuntu.com, so the apt+build work happens here, once, instead of
# on the cluster on every job.
#
# BOUNDARY (read before adding anything to this file): this image is the
# SOLVER environment ONLY -- openEMS-Project built from source, plus the
# python bindings' OWN runtime deps (cython, numpy<2, matplotlib, h5py,
# scipy). It does NOT install rfx or rfx's python deps (jax, etc). The
# lanes fetch/mount the rfx checkout and `pip install` rfx's own deps
# per-run, exactly as they do today -- baking rfx in here would force a
# full multi-hour image rebuild on every rfx commit, defeating the purpose
# of amortizing the openEMS build across runs. Do not add rfx or its deps
# to this file.
#
# Base image is intentionally the SAME tag every affected lane already
# pins (pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime) -- this image is a
# pure infra swap, not a dependency bump.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# openEMS-Project (thliebig/openEMS-Project) commit to build. Default is the
# HEAD of `master` at the time this image was authored (2026-08-04) --
# resolved via `gh api repos/thliebig/openEMS-Project/commits/master`. The
# lanes today clone unpinned HEAD on every run; pinning here makes the
# image's solver version reproducible. Override via
# `--build-arg OPENEMS_COMMIT=<sha>` (also exposed as the workflow_dispatch
# `openems_sha` input in .github/workflows/build-openems-image.yml).
ARG OPENEMS_COMMIT=5b423bdfe0c84064cf9028167bb759007c33b182
ARG BUILD_DATE

LABEL org.opencontainers.image.source="https://github.com/bk-squared/rfx" \
      org.opencontainers.image.description="Source-built openEMS-Project (fparser+CSXCAD+openEMS) solver environment for rfx's external-openEMS VESSL referee lanes -- does NOT include rfx itself" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      rfx.openems-project.commit="${OPENEMS_COMMIT}"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    CSXCAD_INSTALL_PATH=/usr/local \
    OPENEMS_INSTALL_PATH=/usr/local

# apt superset of the four lanes' dependency lists -- verified identical
# across all four at authoring time (diffed against
# scripts/vessl_coax_two_port_referee.yaml, scripts/vessl_msl_phase_referee.yaml,
# scripts/vessl_crossval_external.yaml, and PR #561's
# scripts/vessl_coax_two_port_referee_mesh_refinement.yaml -- no
# reconciliation across differing lists was needed, they already agree).
# Plain archive.ubuntu.com: GitHub Actions runners have healthy network, so
# the lanes' kakao-mirror/ForceIPv4 workaround (a cluster-network fix, not a
# package-list fix) is deliberately NOT baked into this image build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    libhdf5-dev libtinyxml-dev \
    libboost-thread-dev libboost-system-dev libboost-date-time-dev \
    libboost-serialization-dev \
    libcgal-dev libvtk7-dev \
    python3-dev python3-numpy python3-scipy python3-matplotlib \
    python3-setuptools python3-h5py \
    && rm -rf /var/lib/apt/lists/*

# Fetch openEMS-Project pinned to OPENEMS_COMMIT.
WORKDIR /tmp
RUN git clone --recursive https://github.com/thliebig/openEMS-Project.git \
    && cd openEMS-Project \
    && git checkout "${OPENEMS_COMMIT}" \
    && git submodule update --init --recursive

# fparser / CSXCAD / openEMS build+install -- mirrors the lanes' own cmake
# invocations exactly (CSXCAD: -DCMAKE_DISABLE_FIND_PACKAGE_MPI=TRUE;
# openEMS: -DWITH_MPI=OFF; both: -DCMAKE_BUILD_TYPE=Release,
# -DCMAKE_INSTALL_PREFIX=/usr/local).
RUN cd /tmp/openEMS-Project/fparser \
    && mkdir -p build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release \
    && make -j"$(nproc)" && make install

RUN cd /tmp/openEMS-Project/CSXCAD \
    && mkdir -p build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release -DCMAKE_DISABLE_FIND_PACKAGE_MPI=TRUE \
    && make -j"$(nproc)" && make install

RUN cd /tmp/openEMS-Project/openEMS \
    && mkdir -p build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release -DWITH_MPI=OFF \
    && make -j"$(nproc)" && make install

RUN ldconfig

# CSXCAD/openEMS python bindings. The lanes use the base image's conda
# `python` (>=3.10), never /usr/bin/python3 (3.8 on this base image, whose
# pip resolver can't satisfy the bindings' setup.py) -- same rationale as
# every lane YAML's own `PY=python; command -v "$PY" || PY=/opt/conda/bin/python`
# fallback. These pip installs are the BINDINGS' own runtime deps (cython,
# numpy<2, matplotlib, h5py, scipy) -- not rfx's deps; see file header.
RUN python -m pip install --no-cache-dir --upgrade pip "cython" "numpy<2" "matplotlib" "h5py" "scipy"
RUN cd /tmp/openEMS-Project/CSXCAD/python && python setup.py install
RUN cd /tmp/openEMS-Project/openEMS/python && python setup.py install

# Sanity check at BUILD time -- fail the image build itself (not a later,
# more expensive job) if the bindings don't import.
RUN python3 -c "from openEMS.openEMS import openEMS; from CSXCAD.CSXCAD import ContinuousStructure; print('openEMS/CSXCAD bindings import OK')"

WORKDIR /root
CMD ["/bin/bash"]
