# syntax=docker/dockerfile:1
#
# Two-solver environment image for the #717 option-(b) weekly crossval lane
# (scripts/vessl_crossval_solvers_lane.yaml).
#
# WHY THIS IMAGE EXISTS. The four gpu-marked crossval test files
#
#   tests/crossval/test_crossval_comprehensive.py
#   tests/crossval/test_meep_crossval.py
#   tests/crossval/test_meep_crossval_dielectric_cavity.py
#   tests/crossval/test_openems_crossval.py
#
# carry 18 tests, of which 13 are external legs that have never executed
# anywhere (docs/design_notes/717_crossval_lane_decision.md Sec. 1). The PI
# chose option (b) on 2026-09-15: a recurring solver-carrying lane, not the
# Sec. 4 "$IGN deletion" trick. Those 13 legs need meep AND openEMS importable
# in ONE interpreter, because test_crossval_comprehensive.py's TestPECCavity
# holds test_rfx_vs_meep (:163) and test_rfx_vs_openems (:171) in the same
# class -- one pytest process, both solvers. A lane that installs them into two
# separate environments cannot run that file.
#
# WHY IT EXTENDS THE openEMS IMAGE rather than rebuilding both from scratch:
# docker/openems-lane/Dockerfile already pays the 30-60 min fparser + CSXCAD +
# openEMS source build and pins the solver commit. Rebuilding that here would
# duplicate the build and let the two pins drift. This file adds the SECOND
# solver to that environment and nothing else.
#
# BOUNDARY (same rule as the parent image, read before adding anything):
# this image is the SOLVER environment ONLY -- openEMS-Project (inherited) plus
# conda-forge pymeep, plus those two bindings' own runtime deps. It does NOT
# install rfx or rfx's python deps (jax etc). The lane fetches the rfx checkout
# and pip-installs rfx's deps per run, exactly as every other referee lane does.
# Baking rfx in would force a multi-hour rebuild on every rfx commit.
#
# THE numpy CONSTRAINT IS STRUCTURAL, NOT A PREFERENCE. Every conda-forge
# pymeep build through 1.34.0 declares `numpy >=1.11.3,<2.0a0` -- there is no
# numpy-2 pymeep for any python version. The parent image's openEMS bindings
# were likewise built against `numpy<2`. So this image is a numpy-1 environment
# and the lane must pin a jax that still supports numpy 1.x; that is why
# scripts/vessl_crossval_solvers_lane.yaml installs `jax[cpu]==0.6.2` with
# an explicit `numpy<2`, the same pair scripts/vessl_crossval_external.yaml has
# been running on this cluster since PR #563.

ARG OPENEMS_IMAGE=ghcr.io/bk-squared/rfx-openems
ARG OPENEMS_IMAGE_TAG=5b423bdfe0c8
FROM ${OPENEMS_IMAGE}:${OPENEMS_IMAGE_TAG}

# Re-declare after FROM (ARGs before FROM are out of scope in the build stage)
# so the provenance stamp below can record what this was built on top of.
ARG OPENEMS_IMAGE=ghcr.io/bk-squared/rfx-openems
ARG OPENEMS_IMAGE_TAG=5b423bdfe0c8
ARG PYMEEP_VERSION=1.34.0
ARG BUILD_DATE

LABEL org.opencontainers.image.source="https://github.com/bk-squared/rfx" \
      org.opencontainers.image.description="openEMS-Project + conda-forge pymeep in one interpreter, for rfx's weekly two-solver crossval lane (#717 option b) -- does NOT include rfx itself" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      rfx.parent-image="${OPENEMS_IMAGE}:${OPENEMS_IMAGE_TAG}" \
      rfx.pymeep.version="${PYMEEP_VERSION}"

# Runtime provenance stamps, as real ENV rather than only LABELs: a VESSL job's
# own stdout can echo these, `docker inspect` cannot be run from inside a lane.
# Same reasoning as the parent image's RFX_OPENEMS_COMMIT (which is inherited
# and still stamped). ghcr tags are mutable, so a tag repush stays visible in
# the RUN LOG itself rather than only in a registry API call.
ENV RFX_PYMEEP_VERSION=${PYMEEP_VERSION} \
    RFX_SOLVER_PARENT_IMAGE=${OPENEMS_IMAGE}:${OPENEMS_IMAGE_TAG}

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

# conda-forge pymeep into the SAME conda interpreter that carries the openEMS
# bindings. Two deliberate pins:
#
#   python=<the interpreter that is already here>
#     Without it conda is free to move python to satisfy pymeep, which would
#     leave every compiled extension in this image (the openEMS/CSXCAD
#     bindings, torch) built for an interpreter that is no longer the default.
#     Pinning it makes an unsatisfiable solve a loud build failure here instead
#     of a broken import on the cluster.
#
#   numpy<2
#     Belt-and-braces. pymeep's own metadata already forbids numpy 2, but the
#     parent image pip-installed numpy<2 for the openEMS bindings and conda
#     must not "helpfully" resolve that pip-installed numpy away.
#
#
# MEASURED ON THE CLUSTER, TWICE, BEFORE THIS IMAGE WAS EVER BUILT -- and the
# first reading was wrong. VESSL run 369367261409 ran the conda line into this
# parent image WITH --override-channels and died:
#
#     ResolvePackageNotFound:
#       - conda=23.5.2
#
# That was attributed to --override-channels restricting the solve to
# conda-forge. Run 369367261425 then ran it WITHOUT --override-channels and
# failed identically, which falsifies that reading. The channel restriction is
# not the cause.
#
# What the two runs together say: conda must keep its OWN package satisfiable to
# install anything into the env it lives in, and `conda=23.5.2` is not findable
# from this image on this cluster with or without the restriction -- the channel
# it originally came from is either unconfigured in the image or unreachable from
# remilab-c0. The failure is about the base environment being uninstallable-into,
# not about which channels pymeep is drawn from.
#
# CONSEQUENCE, and it is why this image exists rather than a runtime install:
# the conda layer has to be applied where the channel situation is healthy, i.e.
# on a GitHub Actions runner via build-crossval-solvers-image.yml, not in a pod.
# A runtime install into the parent image is not a shortcut around building this;
# it was tried and it does not work.
#
# A SEPARATE ENV IS NOT THE ANSWER HERE, though it is elsewhere: cv02 needs only
# Meep and scripts/vessl_cv02_record_regen.yaml takes that route successfully.
# These four test files need meep AND openEMS in ONE interpreter, and openEMS
# lives in the base env's python.
#
# IF THIS SOLVE STILL FAILS ON THE RUNNER, in order, and none is a silent
# fallback --
# each changes what the image IS, so make the change, rebuild, and say which:
#   1. pymeep has no build for this python. Bump PYMEEP_VERSION (the build
#      workflow exposes it as a workflow_dispatch input) or drop the python pin
#      and let conda-forge choose; the binding rebuild below already handles a
#      moved interpreter, and the final import assertion still guards it.
#   2. conda-forge and the base channel disagree on something other than conda
#      itself. Read the conflict; do not widen the channel list blindly.
# Do NOT work around a failure by installing pymeep into a second conda env:
# the whole point of this image is ONE interpreter, because
# test_crossval_comprehensive.py needs meep and openEMS in one pytest process.
# (A second env IS the right answer where only one solver is needed -- see
# scripts/vessl_cv02_record_regen.yaml, which takes that route deliberately.)
RUN PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" \
    && echo "extending the openEMS image's python ${PYVER} with pymeep ${PYMEEP_VERSION}" \
    && conda install -y -c conda-forge \
         "python=${PYVER}" "pymeep=${PYMEEP_VERSION}" "numpy<2" \
    && conda clean -afy

# The conda solve above can relink numpy underneath the parent image's
# pip-installed openEMS/CSXCAD bindings. They are compiled C extensions, so a
# numpy ABI move breaks them at IMPORT time, not at solve time. The parent
# image leaves its source tree at /tmp/openEMS-Project (it never rm -rf's it),
# so rebuilding the two binding packages against the post-solve interpreter is
# cheap -- the expensive part, the C++ libraries under /usr/local, is untouched
# and is not rebuilt here. Unconditional rather than conditional: a rebuild that
# was not needed costs ~2 min, a skipped one that WAS needed costs a red lane
# on the cluster a week later.
RUN test -d /tmp/openEMS-Project/CSXCAD/python \
      || { echo "parent image no longer ships /tmp/openEMS-Project; this file must be rewritten to build the bindings itself" >&2; exit 1; } \
    && python -m pip install --no-cache-dir --upgrade "cython" "numpy<2" \
    && cd /tmp/openEMS-Project/CSXCAD/python && python setup.py install \
    && cd /tmp/openEMS-Project/openEMS/python && python setup.py install

# Sanity check at BUILD time, and it is the whole point of this file: BOTH
# solvers must import in ONE interpreter, because TestPECCavity needs both in
# one pytest process. Failing here fails the image build; the alternative is
# discovering it as a red weekly lane holding a lab GPU seat.
RUN python -c "\
import numpy, sys; \
print('python', sys.version.split()[0], 'numpy', numpy.__version__); \
assert numpy.__version__.startswith('1.'), f'numpy {numpy.__version__} is not the numpy-1 ABI pymeep and the openEMS bindings are built against'; \
import meep; print('meep', meep.__version__); \
from openEMS.openEMS import openEMS; \
from CSXCAD.CSXCAD import ContinuousStructure; \
print('openEMS/CSXCAD bindings import OK alongside meep')"

WORKDIR /root
CMD ["/bin/bash"]
