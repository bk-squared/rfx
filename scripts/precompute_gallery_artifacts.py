#!/usr/bin/env python3
"""Precompute the rfx web-gallery artifact bundles (S-param PNG/Smith/Touchstone/manifest).

This script powers a **precomputed** showcase gallery: it runs a small set of
canonical rfx validation cases once, on a capable machine, and emits static
artifacts (plots, Touchstone files, a provenance ``manifest.json``) that the
public docs site serves as-is. The site never runs a live solver.

GPU note
--------
The full validated runs are FDTD and GPU-heavy. ``--quick`` exists so the
*pipeline* (registry -> builder -> plot -> Touchstone -> manifest) can be
smoke-tested in seconds on CPU; quick runs use reduced grids/steps, are tagged
``quick_smoke: true`` in the manifest, and set ``validation.passed = null`` so
they can never be mistaken for a validated result. Produce the real,
gate-passing artifacts on VESSL (see ``scripts/vessl_gpu_suite.yaml``), then
commit the small ones and sync the large media (see deliverable 4/5 deploy
drafts under ``gallery-deploy/``).

Manifest provenance shape mirrors :func:`rfx.artifacts._provenance`
(``reproducibility.status = "provenance-only"``): the bundle records *how* a
result was produced, not a deterministic replay.

Usage
-----
::

    # smoke-test the whole pipeline fast on CPU (one fast case)
    python scripts/precompute_gallery_artifacts.py --case multilayer_fresnel --quick

    # full (GPU/VESSL) run of every case into the committed assets tree
    python scripts/precompute_gallery_artifacts.py --case all

Artifacts land under ``--out`` (default ``docs/public/gallery/assets/<case_id>/``)
and are referenced from the gallery pages by the absolute served URL
``/rfx/gallery/assets/<case_id>/<file>``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

C0 = 2.998e8
SERVED_URL_PREFIX = "/rfx/gallery/assets"
MANIFEST_SCHEMA_VERSION = "rfx-gallery-manifest-v1"
DEFAULT_OUT = Path("docs/public/gallery/assets")


# ---------------------------------------------------------------------------
# Provenance helpers (mirror rfx.artifacts shape)
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_sha() -> str | None:
    """Return ``git rev-parse HEAD`` for the current checkout, or ``None``."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _rfx_version() -> str:
    try:
        import rfx

        return str(getattr(rfx, "__version__", "unknown"))
    except Exception:
        return "unknown"


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GalleryCase:
    """One showcase case: metadata + a builder that runs the rfx sim.

    The ``builder`` is a callable ``builder(quick: bool) -> CaseResult`` that
    constructs and runs the rfx simulation and returns its S-parameters plus
    metric values. Builders must be import-safe (no work at import time).
    """

    id: str
    title: str
    description: str
    validation_tier: str  # "E5" | "E4"
    metric: str  # what is compared
    tolerance: str  # human-readable tolerance
    reference_solver: str  # e.g. "analytic transfer matrix", "OpenEMS"
    builder: Callable[[bool], "CaseResult"]
    n_ports: int = 2
    has_smith: bool = True
    # --- Gallery-v2 visual config ---
    geometry_mode: str = "slice2d"  # "slice2d" (εr cross-section) | "geom3d"
    geometry_slice_axis: int = 2  # axis index for the εr slice (slice2d only)
    anim_slice_axis: int = 2  # SnapshotSpec.slice_axis: INT 0/1/2 (capture)
    anim_render_axis: str = "z"  # save_field_animation slice_axis: STR x/y/z
    anim_component: str = "ez"
    has_animation: bool = True
    is_optimization: bool = False  # opt cases skip S-param/Smith/Touchstone


@dataclass
class CaseResult:
    """What a builder returns: S-params plus a validation summary.

    The optional fields support the richer Gallery-v2 assets:

    * ``sim`` — the constructed :class:`~rfx.api.Simulation`, attached by each
      builder so the emitter can render a geometry image and run a second
      snapshot forward for the field animation. ``None`` for cases that do not
      expose a Simulation (e.g. the raw-functional Fresnel path).
    * ``geometry_path`` / ``animation_path`` — pre-rendered asset paths a
      builder may emit itself; normally the emitter fills these in.
    * Optimization-case extras (``loss_history`` … ``r_reference``) carry the
      convergence trace, per-iteration design state, and reflection spectra so
      the emitter can draw the convergence / design-evolution / spectrum plots.
    """

    freqs_hz: np.ndarray
    s_params: np.ndarray  # (n_ports, n_ports, n_freqs) complex
    passed: bool | None
    metric_value: str  # e.g. "max|S11| = 0.012"
    params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    # --- Gallery-v2 optional plumbing ---
    sim: Any = None  # rfx.api.Simulation, for geometry image + animation forward
    anim_sim: Any = None  # optional dedicated Simulation for the field animation
    #                       (used when the validated `sim` cannot snapshot —
    #                       e.g. a 2-port waveguide or a non-uniform-mesh run)
    geometry_path: Path | None = None
    animation_path: Path | None = None
    # --- optimization-case extras ---
    loss_history: np.ndarray | None = None  # (n_iters,) cost per Adam step
    eps_trace: np.ndarray | None = None  # (n_iters, n_layers) design εr per step
    eps_reference: np.ndarray | None = None  # (n_layers,) analytic-optimum εr
    freqs_spectrum_hz: np.ndarray | None = None  # (n_f,) for the R(f) spectra
    r_initial: np.ndarray | None = None  # (n_f,) reflection R(f), start of run
    r_final: np.ndarray | None = None  # (n_f,) reflection R(f), end of run
    r_reference: np.ndarray | None = None  # (n_f,) analytic-optimum R(f)
    reference_cost: float | None = None  # analytic-optimum scalar cost


# ---------------------------------------------------------------------------
# Builders — each constructs + runs an rfx sim and returns a CaseResult
# ---------------------------------------------------------------------------


def _build_multilayer_fresnel(quick: bool) -> CaseResult:
    """Normal-incidence dielectric slab R/T vs the exact transfer matrix.

    Replicates the rfx-only path of ``examples/crossval/04_multilayer_fresnel.py``
    (2D TMz TFSF plane wave, single-run scattered/total field measurement) and
    maps the lossless-reciprocal-slab physics onto a 2-port S-matrix:
    ``|S11|^2 = R``, ``|S21|^2 = T``. The complex analytic ``(r, t)`` from the
    transfer matrix is exported so the Touchstone carries phase, and the gallery
    plot compares rfx |S| against the analytic curve.

    This is the designated fast case: ``--quick`` shrinks the interior grid and
    step count so the full pipeline runs in seconds on CPU.
    """
    from rfx.grid import Grid
    from rfx.core.yee import init_state, init_materials, update_e, update_h
    from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
    from rfx.sources.tfsf import (
        init_tfsf,
        update_tfsf_1d_h,
        update_tfsf_1d_e,
        apply_tfsf_e,
        apply_tfsf_h,
    )

    eps_slab = 4.0
    d_slab = 10.0e-3
    f0 = 10.0e9
    dx = 1.0e-3
    bw = 0.5
    n_cpml = 12 if quick else 20
    nx_interior = 240 if quick else 600
    max_steps = 2000 if quick else 8000
    probe_gap = 12 if quick else 30  # cells from slab edge to each probe

    def fresnel_slab_rt_complex(freqs: np.ndarray, eps_r: float, d: float):
        n = math.sqrt(eps_r)
        r = np.zeros_like(freqs, dtype=complex)
        t = np.zeros_like(freqs, dtype=complex)
        for i, f in enumerate(freqs):
            if f <= 0:
                t[i] = 1.0
                continue
            delta = 2 * np.pi * f * n * d / C0
            cos_d, sin_d = np.cos(delta), np.sin(delta)
            m00 = cos_d
            m01 = 1j * sin_d / n
            m10 = 1j * n * sin_d
            m11 = cos_d
            num = m00 + m01 - m10 - m11
            den = m00 + m01 + m10 + m11
            r[i] = num / den
            t[i] = 2.0 / den
        return r, t

    grid = Grid(
        freq_max=20e9,
        domain=(nx_interior * dx, 0.004, dx),
        dx=dx,
        cpml_layers=n_cpml,
        mode="2d_tmz",
    )
    dt = grid.dt
    periodic = (False, True, True)

    tfsf_cfg, tfsf_st = init_tfsf(
        grid.nx,
        dx,
        dt,
        cpml_layers=n_cpml,
        tfsf_margin=5,
        f0=f0,
        bandwidth=bw,
        amplitude=1.0,
        polarization="ez",
        direction="+x",
        ny=grid.ny,
        nz=grid.nz,
    )
    x_lo, x_hi = tfsf_cfg.x_lo, tfsf_cfg.x_hi
    i0 = tfsf_cfg.i0

    slab_lo_g = grid.nx // 2 - int(d_slab / (2 * dx))
    slab_hi_g = grid.nx // 2 + int(d_slab / (2 * dx))

    probe_refl_x = max(slab_lo_g - probe_gap, x_lo + 11)
    probe_trans_x = min(slab_hi_g + probe_gap, x_hi - 11)
    probe_refl = (probe_refl_x, grid.ny // 2, 0)
    probe_trans = (probe_trans_x, grid.ny // 2, 0)
    ref_1d_refl = i0 + (probe_refl_x - x_lo)
    ref_1d_trans = i0 + (probe_trans_x - x_lo)

    v_cells = C0 * dt / dx
    dist_to_cpml_hi = grid.nx - n_cpml - probe_trans_x
    dist_to_cpml_lo = probe_refl_x - n_cpml
    t_safe_hi = int(2 * dist_to_cpml_hi / v_cells * 0.95)
    t_safe_lo = int(2 * dist_to_cpml_lo / v_cells * 0.95)
    n_steps = min(t_safe_hi, t_safe_lo, max_steps)

    materials = init_materials(grid.shape)
    materials = materials._replace(
        eps_r=materials.eps_r.at[slab_lo_g:slab_hi_g, :, :].set(eps_slab)
    )
    state = init_state(grid.shape)
    cp, cs = init_cpml(grid)

    ts_refl = np.zeros(n_steps)
    ts_trans = np.zeros(n_steps)
    ts_inc_refl = np.zeros(n_steps)
    ts_inc_trans = np.zeros(n_steps)

    for step in range(n_steps):
        t = step * dt
        state = update_h(state, materials, dt, dx, periodic)
        state = apply_tfsf_h(state, tfsf_cfg, tfsf_st, dx, dt)
        state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
        tfsf_st = update_tfsf_1d_h(tfsf_cfg, tfsf_st, dx, dt)

        state = update_e(state, materials, dt, dx, periodic)
        state = apply_tfsf_e(state, tfsf_cfg, tfsf_st, dx, dt)
        state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
        tfsf_st = update_tfsf_1d_e(tfsf_cfg, tfsf_st, dx, dt, t)

        ts_refl[step] = float(state.ez[probe_refl])
        ts_trans[step] = float(state.ez[probe_trans])
        ts_inc_refl[step] = float(tfsf_st.e1d[ref_1d_refl])
        ts_inc_trans[step] = float(tfsf_st.e1d[ref_1d_trans])

    ts_scattered_refl = ts_refl - ts_inc_refl

    nfft = int(2 ** np.ceil(np.log2(max(n_steps, 2))) * 8)
    freqs = np.fft.rfftfreq(nfft, d=dt)
    s_inc_t = np.fft.rfft(ts_inc_trans, n=nfft)
    s_inc_r = np.fft.rfft(ts_inc_refl, n=nfft)
    s_total_t = np.fft.rfft(ts_trans, n=nfft)
    s_scat_r = np.fft.rfft(ts_scattered_refl, n=nfft)

    inc_power = np.abs(s_inc_t)
    mask = (freqs > 3e9) & (freqs < 15e9) & (inc_power > inc_power.max() * 0.02)
    f_band = freqs[mask]

    t_rfx = np.abs(s_total_t[mask]) ** 2 / np.abs(s_inc_t[mask]) ** 2
    r_rfx = np.abs(s_scat_r[mask]) ** 2 / np.abs(s_inc_r[mask]) ** 2
    r_an_c, t_an_c = fresnel_slab_rt_complex(f_band, eps_slab, d_slab)
    r_an = np.abs(r_an_c) ** 2
    t_an = np.abs(t_an_c) ** 2

    # Build a 2-port S-matrix. rfx provides |S| from R/T; analytic provides
    # the complex reference (used for Touchstone phase + the plotted overlay).
    n_f = len(f_band)
    s = np.zeros((2, 2, n_f), dtype=complex)
    s11_rfx_mag = np.sqrt(np.clip(r_rfx, 0.0, None))
    s21_rfx_mag = np.sqrt(np.clip(t_rfx, 0.0, None))
    # Use the analytic phase on the rfx magnitudes (rfx R/T are power-only here).
    s[0, 0, :] = s11_rfx_mag * np.exp(1j * np.angle(r_an_c))
    s[1, 1, :] = s11_rfx_mag * np.exp(1j * np.angle(r_an_c))
    s[0, 1, :] = s21_rfx_mag * np.exp(1j * np.angle(t_an_c))
    s[1, 0, :] = s21_rfx_mag * np.exp(1j * np.angle(t_an_c))

    t_err = float(np.abs(t_rfx - t_an).mean())
    r_err = float(np.abs(r_rfx - r_an).mean())
    cons = float(np.abs(r_rfx + t_rfx - 1).mean())
    passed = None if quick else bool(t_err < 0.05 and r_err < 0.05 and cons < 0.05)

    # Companion Simulation (visuals only — geometry image + field animation).
    # The validated S-params above come from the raw-functional TFSF path; this
    # high-level Simulation reproduces the same slab geometry so the gallery can
    # show *what the structure is* and *what the wave does* without touching the
    # measured physics.
    from rfx import Simulation, Box
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.sources.sources import GaussianPulse

    dom_x = nx_interior * dx
    sim_vis = Simulation(
        freq_max=20e9,
        domain=(dom_x, 0.004, dx),
        dx=dx,
        cpml_layers=n_cpml,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="periodic", hi="periodic"),
            z=Boundary(lo="periodic", hi="periodic"),
        ),
    )
    sim_vis.add_material("slab", eps_r=eps_slab, sigma=0.0)
    sim_vis.add(
        Box((dom_x / 2 - d_slab / 2, 0.0, 0.0), (dom_x / 2 + d_slab / 2, 0.004, dx)),
        material="slab",
    )
    sim_vis.add_source(
        (dom_x * 0.15, 0.002, dx / 2),
        "ez",
        waveform=GaussianPulse(f0=f0, bandwidth=bw, amplitude=1.0),
    )

    return CaseResult(
        freqs_hz=f_band,
        s_params=s,
        passed=passed,
        sim=sim_vis,
        metric_value=(
            f"mean|T_rfx-T_an|={t_err:.4f}, mean|R_rfx-R_an|={r_err:.4f}, "
            f"R+T dev={cons:.4f}"
        ),
        params={
            "eps_slab": eps_slab,
            "slab_thickness_m": d_slab,
            "dx_m": dx,
            "cpml_layers": n_cpml,
            "nx_interior": nx_interior,
            "n_steps": int(n_steps),
            "mode": "2d_tmz",
        },
        notes=(
            "Lossless reciprocal slab; |S11|^2=R, |S21|^2=T. rfx magnitudes "
            "from FDTD R/T; phase taken from the exact transfer matrix."
        ),
    )


def _build_waveguide_wr90(quick: bool) -> CaseResult:
    """Empty WR-90 waveguide port: matched-load reference (|S11|=0, |S21|=1).

    Reuses the canonical ``add_waveguide_port`` + ``compute_waveguide_s_matrix``
    pipeline from ``examples/crossval/11_waveguide_port_wr90.py`` for the
    empty-guide geometry. The matched-load analytic reference is exact, so this
    is an E5 case.
    """
    import jax.numpy as jnp

    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec

    a_wg = 0.02286
    b_wg = 0.01016
    f_cutoff = C0 / (2.0 * a_wg)
    n_freqs = 7 if quick else 21
    freqs = np.linspace(8.2e9, 12.4e9, n_freqs)
    dx = 0.001
    cpml_layers = 12 if quick else 20
    num_periods = 30 if quick else 200
    domain_x = 0.120 if quick else 0.200
    port_left_x = 0.030 if quick else 0.040
    port_right_x = domain_x - 0.040
    ref_left = port_left_x + 0.010
    ref_right = port_right_x - 0.010

    f0 = float(freqs.mean())
    bandwidth = min(0.6, max(0.2, float(freqs[-1] - freqs[0]) / max(f0, 1.0)))

    sim = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(domain_x, a_wg, b_wg),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=cpml_layers,
        dx=dx,
    )
    port_freqs = jnp.asarray(freqs)
    sim.add_waveguide_port(
        port_left_x,
        direction="+x",
        mode=(1, 0),
        mode_type="TE",
        freqs=port_freqs,
        f0=f0,
        bandwidth=bandwidth,
        waveform="modulated_gaussian",
        reference_plane=ref_left,
        name="left",
    )
    sim.add_waveguide_port(
        port_right_x,
        direction="-x",
        mode=(1, 0),
        mode_type="TE",
        freqs=port_freqs,
        f0=f0,
        bandwidth=bandwidth,
        waveform="modulated_gaussian",
        reference_plane=ref_right,
        name="right",
    )

    result = sim.compute_waveguide_s_matrix(num_periods=num_periods, normalize=True)
    s_raw = np.asarray(result.s_params)
    port_idx = {name: i for i, name in enumerate(result.port_names)}
    f_hz = np.asarray(result.freqs)
    il = port_idx["left"]
    ir = port_idx["right"]
    s = np.zeros((2, 2, len(f_hz)), dtype=complex)
    s[0, 0, :] = s_raw[il, il, :]
    s[1, 0, :] = s_raw[ir, il, :]
    s[0, 1, :] = s_raw[il, ir, :]
    s[1, 1, :] = s_raw[ir, ir, :]

    max_s11 = float(np.abs(s[0, 0, :]).max())
    min_s21 = float(np.abs(s[1, 0, :]).min())
    passed = None if quick else bool(max_s11 < 0.02 and min_s21 > 0.97)

    # Companion Simulation for the field animation. The validated S-matrix is
    # produced by compute_waveguide_s_matrix() with two waveguide ports, which
    # Simulation.run() (the snapshot path) cannot drive — so a source-fed empty
    # guide reproduces the same TE10 propagation purely for the visual.
    #
    # CPML bug note: Simulation._build_grid() sets face_layers[y/z] = cpml_layers
    # regardless of boundary type (PEC y,z walls get pad=0 but the CPML arrays
    # are still allocated with size cpml_layers). When cpml_layers (e.g. 20)
    # exceeds the guide cross-section cell count (b_wg/dx ≈ 10), XLA's shape
    # inference fails with a broadcasting error. Fix: pad b_wg so nz > 2*cpml
    # (still hollow air — only the domain extent changes, not the physics).
    from rfx.sources.sources import GaussianPulse

    anim_cpml = 8  # fixed, safe for the guide cross-section in both quick/full runs
    # Ensure the guide cross-section fits at least 2*anim_cpml+4 cells in each
    # transverse direction so CPML arrays never exceed the grid extent.
    b_wg_anim = max(b_wg, (2 * anim_cpml + 4) * dx)
    a_wg_anim = max(a_wg, (2 * anim_cpml + 4) * dx)
    anim_domain_x = 0.100  # compact x: enough for one clear pulse transit
    anim_src_x = anim_domain_x * 0.15  # source near the left CPML exit

    sim_anim = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(anim_domain_x, a_wg_anim, b_wg_anim),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=anim_cpml,
        dx=dx,
    )
    sim_anim.add_source(
        (anim_src_x, a_wg_anim / 2, b_wg_anim / 2),
        "ez",
        waveform=GaussianPulse(f0=f0, bandwidth=bandwidth, amplitude=1.0),
    )

    return CaseResult(
        freqs_hz=f_hz,
        s_params=s,
        passed=passed,
        sim=sim,
        anim_sim=sim_anim,
        metric_value=f"max|S11|={max_s11:.4f}, min|S21|={min_s21:.4f}",
        params={
            "band": "X-band (8.2-12.4 GHz)",
            "waveguide": "WR-90 TE10",
            "f_cutoff_hz": float(f_cutoff),
            "dx_m": dx,
            "cpml_layers": cpml_layers,
            "num_periods": num_periods,
            "domain_x_m": domain_x,
        },
        notes="Empty matched-load guide; analytic reference |S11|=0, |S21|=1.",
    )


def _build_patch_antenna(quick: bool) -> CaseResult:
    """Probe-fed 2.4 GHz patch antenna on FR4 (lumped-port S11 sweep).

    Replicates the rfx-only lumped-port path of
    ``examples/crossval/05_patch_antenna.py`` (no OpenEMS dependency). The
    cross-validation reference is OpenEMS Harminv resonance frequency
    (tier E4 — coarse-mesh resonance agreement, not absolute calibrated S).
    """
    import jax.numpy as jnp

    from rfx import Simulation, Box
    from rfx.boundaries.spec import BoundarySpec
    from rfx.sources.sources import GaussianPulse
    from rfx.auto_config import smooth_grading

    f_design = 2.4e9
    eps_r = 4.3
    h_sub = 1.5e-3
    w_patch = 38.0e-3
    l_patch = 29.5e-3
    gx = 60.0e-3
    gy = 55.0e-3
    dx = 1.0e-3
    n_cpml = 8
    n_sub = 6
    dz_sub = h_sub / n_sub
    air_below = 12.0e-3
    air_above = 25.0e-3
    n_below = int(math.ceil(air_below / dx))
    n_above = int(math.ceil(air_above / dx))

    dom_x = gx + 2 * 10e-3
    dom_y = gy + 2 * 10e-3
    gx_lo = (dom_x - gx) / 2
    gx_hi = gx_lo + gx
    gy_lo = (dom_y - gy) / 2
    gy_hi = gy_lo + gy
    patch_x_lo = dom_x / 2 - l_patch / 2
    patch_x_hi = dom_x / 2 + l_patch / 2
    patch_y_lo = dom_y / 2 - w_patch / 2
    patch_y_hi = dom_y / 2 + w_patch / 2
    probe_inset = 8.0e-3
    feed_x = patch_x_lo + probe_inset
    feed_y = dom_y / 2

    z_gnd_lo = air_below - dz_sub
    z_gnd_hi = air_below
    z_sub_lo = air_below
    z_sub_hi = air_below + h_sub
    z_patch_lo = z_sub_hi
    z_patch_hi = z_sub_hi + dz_sub

    raw_dz = np.concatenate(
        [
            np.full(n_below, dx),
            np.full(1, dz_sub),
            np.full(n_sub, dz_sub),
            np.full(n_above, dx),
        ]
    )
    dz_profile = smooth_grading(raw_dz, max_ratio=1.3)

    sim = Simulation(
        freq_max=4e9,
        domain=(dom_x, dom_y, 0),
        dx=dx,
        dz_profile=dz_profile,
        boundary=BoundarySpec.uniform("cpml"),
        cpml_layers=n_cpml,
    )
    sim.add_material("fr4", eps_r=eps_r, sigma=0.0)
    sim.add(Box((gx_lo, gy_lo, z_gnd_lo), (gx_hi, gy_hi, z_gnd_hi)), material="pec")
    sim.add(Box((gx_lo, gy_lo, z_sub_lo), (gx_hi, gy_hi, z_sub_hi)), material="fr4")
    sim.add(
        Box((patch_x_lo, patch_y_lo, z_patch_lo), (patch_x_hi, patch_y_hi, z_patch_hi)),
        material="pec",
    )
    port_z0 = z_sub_lo + dz_sub * 1.5
    port_extent = z_sub_hi - port_z0
    sim.add_port(
        position=(feed_x, feed_y, port_z0),
        component="ez",
        impedance=50.0,
        extent=port_extent,
        waveform=GaussianPulse(f0=f_design, bandwidth=0.8),
    )

    n_freqs = 31 if quick else 101
    s_param_n_steps = 2000 if quick else 12000
    freqs_s = jnp.linspace(1.5e9, 3.5e9, n_freqs)
    result = sim.run(
        n_steps=s_param_n_steps,
        compute_s_params=True,
        s_param_freqs=freqs_s,
        s_param_n_steps=s_param_n_steps,
    )

    s_raw = np.asarray(result.s_params)  # (1, 1, n_f) single port
    f_hz = np.asarray(freqs_s)
    s = np.zeros((1, 1, len(f_hz)), dtype=complex)
    s[0, 0, :] = s_raw[0, 0, :]

    s11 = s[0, 0, :]
    passive = bool(np.all(np.abs(s11) < 1.05))
    # Local dip near analytic resonance (Balanis TL ~2.42 GHz).
    eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h_sub / w_patch) ** (-0.5)
    delta_l = (
        0.412
        * h_sub
        * ((eps_eff + 0.3) * (w_patch / h_sub + 0.264))
        / ((eps_eff - 0.258) * (w_patch / h_sub + 0.8))
    )
    f_res_an = C0 / (2 * (l_patch + 2 * delta_l) * math.sqrt(eps_eff))
    s11_db = 20 * np.log10(np.maximum(np.abs(s11), 1e-6))
    lo = int(np.searchsorted(f_hz, f_res_an * 0.90))
    hi = int(np.searchsorted(f_hz, f_res_an * 1.10))
    if hi > lo:
        local_idx = lo + int(np.argmin(s11_db[lo:hi]))
        f_res_rfx = float(f_hz[local_idx])
        f_err = 100 * abs(f_res_rfx - f_res_an) / f_res_an
    else:
        f_res_rfx = float("nan")
        f_err = float("inf")
    passed = None if quick else bool(passive and f_err < 20.0)

    # Companion Simulation for the field animation. The validated S11 run uses a
    # non-uniform z stack (graded dz) on which the snapshot path is not wired, so
    # a uniform-mesh twin of the same patch — soft-fed instead of lumped-port —
    # reproduces the field spreading under the patch purely for the visual.
    n_z_unif = max(4, int(math.ceil((air_below + h_sub + air_above) / dx)))
    dom_z = n_z_unif * dx
    z0_gnd = air_below
    sim_anim = Simulation(
        freq_max=4e9,
        domain=(dom_x, dom_y, dom_z),
        dx=dx,
        boundary=BoundarySpec.uniform("cpml"),
        cpml_layers=n_cpml,
    )
    sim_anim.add_material("fr4", eps_r=eps_r, sigma=0.0)
    sim_anim.add(
        Box((gx_lo, gy_lo, z0_gnd - dx), (gx_hi, gy_hi, z0_gnd)), material="pec"
    )
    sim_anim.add(
        Box((gx_lo, gy_lo, z0_gnd), (gx_hi, gy_hi, z0_gnd + dx)), material="fr4"
    )
    sim_anim.add(
        Box(
            (patch_x_lo, patch_y_lo, z0_gnd + dx),
            (patch_x_hi, patch_y_hi, z0_gnd + 2 * dx),
        ),
        material="pec",
    )
    sim_anim.add_source(
        (feed_x, feed_y, z0_gnd + 0.5 * dx),
        "ez",
        waveform=GaussianPulse(f0=f_design, bandwidth=0.8),
    )

    return CaseResult(
        freqs_hz=f_hz,
        s_params=s,
        passed=passed,
        sim=sim,
        anim_sim=sim_anim,
        metric_value=(
            f"resonance dip f={f_res_rfx / 1e9:.3f} GHz vs analytic "
            f"{f_res_an / 1e9:.3f} GHz ({f_err:.1f}% err); passive={passive}"
        ),
        params={
            "design_freq_hz": f_design,
            "substrate": "FR4 (eps_r=4.3)",
            "patch_mm": [w_patch * 1e3, l_patch * 1e3],
            "dx_m": dx,
            "cpml_layers": n_cpml,
            "s_param_n_steps": s_param_n_steps,
        },
        notes=(
            "Single-cell lumped-port S11 has parasitic reactance, so the patch "
            "resonance appears as a local dip. Reference is OpenEMS Harminv "
            "resonance frequency (E4 resonance agreement, not calibrated S)."
        ),
    )


def _build_ar_coating_design(quick: bool) -> CaseResult:
    """Inverse-designed 3-layer anti-reflection coating (differentiable FDTD).

    Ports ``examples/inverse_design/multilayer_ar_coating.py``: a high-εr
    substrate (εr=12) is matched over the X-band by three quarter-wave coating
    layers whose permittivities are optimized by gradient descent *through* the
    FDTD solver (``jax.value_and_grad`` + Adam), and cross-validated against the
    exact transfer-matrix-method (TMM) optimum.

    Returns the loss history, per-iteration design εr, the initial/final/exact
    reflection spectra, and a companion :class:`Simulation` (the matched stack)
    so the emitter can draw convergence, design-evolution, spectrum, and
    geometry images. This case has no S-matrix/Touchstone.
    """
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize as _scimin

    from rfx import Simulation, Box
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.sources.sources import GaussianPulse

    eps_sub = 12.0
    n_layers = 3
    f_lo, f_hi = 8e9, 12e9
    f0 = 0.5 * (f_lo + f_hi)
    f_max = 15e9
    lambda0 = C0 / f0
    n_ref = eps_sub ** (0.5 / (n_layers + 1))
    layer_thk = lambda0 / (4.0 * n_ref)
    dx = 0.5e-3

    x_src = 10e-3
    x_refl = 22e-3
    x_design_lo = 35e-3
    x_design_hi = x_design_lo + n_layers * layer_thk
    x_trans = x_design_hi + 8e-3
    lx = x_trans + 18e-3
    ly = lz = dx

    n_periods = 30
    # --quick: fewer Adam iterations and shorter time-marching for a fast CPU
    # smoke; the full validated run keeps n_iters=60.
    n_iters = 3 if quick else 60
    lr = 0.15

    def _make_sim() -> Simulation:
        sim = Simulation(
            freq_max=f_max,
            domain=(lx, ly, lz),
            dx=dx,
            cpml_layers=10,
            boundary=BoundarySpec(
                x="cpml",
                y=Boundary(lo="periodic", hi="periodic"),
                z=Boundary(lo="periodic", hi="periodic"),
            ),
        )
        bw = (f_hi - f_lo) / f0 * 1.6
        sim.add_source(
            (x_src, ly / 2, lz / 2),
            "ez",
            waveform=GaussianPulse(f0=f0, bandwidth=bw, amplitude=1.0),
        )
        sim.add_probe((x_refl, ly / 2, lz / 2), "ez")
        sim.add_probe((x_trans, ly / 2, lz / 2), "ez")
        return sim

    sim_probe = _make_sim()
    grid = sim_probe._build_grid()
    nx = grid.shape[0]
    dt = float(grid.dt)
    n_steps = int(round(n_periods / f0 / dt))
    if quick:
        n_steps = min(n_steps, 600)

    layer_edges_c = [
        int(round((x_design_lo + i * layer_thk) / dx)) for i in range(n_layers + 1)
    ]
    i_sub_lo = layer_edges_c[-1]
    i_sub_hi = nx

    def _base_eps() -> jnp.ndarray:
        eps = jnp.ones((nx, 1, 1), dtype=jnp.float32)
        return eps.at[i_sub_lo:i_sub_hi, :, :].set(eps_sub)

    def _vacuum_eps() -> jnp.ndarray:
        return jnp.ones((nx, 1, 1), dtype=jnp.float32)

    def _render_design_eps(layer_eps: jnp.ndarray) -> jnp.ndarray:
        eps = _base_eps()
        for i in range(n_layers):
            eps = eps.at[layer_edges_c[i]:layer_edges_c[i + 1], :, :].set(layer_eps[i])
        return eps

    def _latent_to_eps(latent: jnp.ndarray) -> jnp.ndarray:
        return 1.0 + (eps_sub - 1.0) * jax.nn.sigmoid(latent)

    # ---- analytic TMM reference ----
    def tmm_R(layer_eps: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        n0 = 1.0
        n_sub = float(np.sqrt(eps_sub))
        R = np.zeros_like(freqs, dtype=float)
        for fi, f in enumerate(freqs):
            if f <= 0:
                R[fi] = 0.0
                continue
            M = np.eye(2, dtype=complex)
            for li in range(len(layer_eps)):
                n_l = float(np.sqrt(layer_eps[li]))
                beta = 2.0 * np.pi * f * n_l * layer_thk / C0
                cos_b, sin_b = np.cos(beta), np.sin(beta)
                ML = np.array(
                    [[cos_b, 1j * sin_b / n_l], [1j * n_l * sin_b, cos_b]],
                    dtype=complex,
                )
                M = M @ ML
            num = n0 * M[0, 0] + n0 * n_sub * M[0, 1] - M[1, 0] - n_sub * M[1, 1]
            den = n0 * M[0, 0] + n0 * n_sub * M[0, 1] + M[1, 0] + n_sub * M[1, 1]
            R[fi] = float(np.abs(num / den) ** 2)
        return R

    def tmm_band_cost(layer_eps: np.ndarray) -> float:
        fs = np.linspace(f_lo, f_hi, 51)
        return float(np.mean(tmm_R(layer_eps, fs)))

    res_tmm = _scimin(
        tmm_band_cost,
        x0=np.array([2.5, 4.0, 7.0]),
        bounds=[(1.0, eps_sub)] * n_layers,
        method="L-BFGS-B",
    )
    tmm_eps = res_tmm.x
    tmm_cost = float(res_tmm.fun)
    geo_ladder = np.array(
        [eps_sub ** ((i + 1) / (n_layers + 1)) for i in range(n_layers)]
    )

    # ---- vacuum reference forward (incident-wave subtraction) ----
    sim_ref = _make_sim()
    result_ref = sim_ref.forward(
        eps_override=_vacuum_eps(), n_steps=n_steps, skip_preflight=True
    )
    ts_inc_refl = jnp.asarray(result_ref.time_series[:, 0])

    nfft = int(2 ** np.ceil(np.log2(n_steps)))
    freqs_fft = jnp.fft.rfftfreq(nfft, d=dt)
    band_mask = (freqs_fft >= f_lo) & (freqs_fft <= f_hi)
    band_norm = float(jnp.sum(band_mask))

    sim_design = _make_sim()

    def cost_with_aux(latent: jnp.ndarray):
        layer_eps = _latent_to_eps(latent)
        res = sim_design.forward(
            eps_override=_render_design_eps(layer_eps),
            n_steps=n_steps,
            skip_preflight=True,
        )
        ts_scat = res.time_series[:, 0] - ts_inc_refl
        S_inc = jnp.fft.rfft(ts_inc_refl, n=nfft)
        S_scat = jnp.fft.rfft(ts_scat, n=nfft)
        R = (jnp.abs(S_scat) / (jnp.abs(S_inc) + 1e-30)) ** 2
        cost = jnp.sum(R * band_mask) / band_norm
        return cost, R

    grad_fn = jax.value_and_grad(lambda lat: cost_with_aux(lat)[0])

    # Warm start from the geometric-ladder closed-form approximation.
    p_geo = np.clip((geo_ladder - 1.0) / (eps_sub - 1.0), 1e-3, 1.0 - 1e-3)
    latent = jnp.asarray(np.log(p_geo / (1.0 - p_geo)), dtype=jnp.float32)
    m = jnp.zeros_like(latent)
    v = jnp.zeros_like(latent)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    losses: list[float] = []
    eps_trace: list[np.ndarray] = []
    initial_R: np.ndarray | None = None
    for it in range(n_iters):
        cost, grad = grad_fn(latent)
        losses.append(float(cost))
        eps_trace.append(np.asarray(_latent_to_eps(latent)))
        if it == 0:
            _, R0 = cost_with_aux(latent)
            initial_R = np.asarray(R0)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v / (1 - beta2 ** (it + 1))
        latent = latent - lr * m_hat / (jnp.sqrt(v_hat) + adam_eps)

    final_eps = np.asarray(_latent_to_eps(latent))
    _, R_final = cost_with_aux(latent)
    R_final = np.asarray(R_final)
    final_cost = losses[-1]

    freqs_fft_arr = np.asarray(freqs_fft)
    mask = (freqs_fft_arr >= f_lo) & (freqs_fft_arr <= f_hi)
    fs_band = freqs_fft_arr[mask]
    r_initial_band = initial_R[mask] if initial_R is not None else R_final[mask]
    r_final_band = R_final[mask]
    r_ref_band = tmm_R(tmm_eps, fs_band)

    cost_ratio_to_tmm = final_cost / max(tmm_cost, 1e-20)
    geo_cost = tmm_band_cost(geo_ladder)
    cost_ratio_to_geo = final_cost / max(geo_cost, 1e-20)
    n_intermediate = int(np.sum((final_eps > 1.5) & (final_eps < eps_sub - 0.5)))
    passed = (
        None
        if quick
        else bool(
            cost_ratio_to_tmm <= 1.2
            and cost_ratio_to_geo <= 0.7
            and n_intermediate >= 2
        )
    )

    # Companion Simulation at the final design (geometry image only).
    sim_vis = _make_sim()
    sim_vis.add_material("substrate", eps_r=eps_sub, sigma=0.0)
    sim_vis.add(Box((x_design_hi, 0.0, 0.0), (lx, ly, lz)), material="substrate")
    for i in range(n_layers):
        lo_m = layer_edges_c[i] * dx
        hi_m = layer_edges_c[i + 1] * dx
        name = f"layer{i + 1}"
        sim_vis.add_material(name, eps_r=float(final_eps[i]), sigma=0.0)
        sim_vis.add(Box((lo_m, 0.0, 0.0), (hi_m, ly, lz)), material=name)

    return CaseResult(
        freqs_hz=fs_band,
        s_params=np.zeros((1, 1, len(fs_band)), dtype=complex),
        passed=passed,
        sim=sim_vis,
        metric_value=(
            f"final cost {final_cost:.2e} = {cost_ratio_to_tmm:.2f}x the "
            f"exact-theory optimum (passes if <=1.2x)"
        ),
        params={
            "eps_substrate": eps_sub,
            "n_layers": n_layers,
            "band": "X-band (8-12 GHz)",
            "n_iters": n_iters,
            "n_steps": int(n_steps),
            "dx_m": dx,
            "final_eps": [float(e) for e in final_eps],
            "tmm_eps": [float(e) for e in tmm_eps],
            "cost_ratio_to_tmm": cost_ratio_to_tmm,
            "cost_ratio_to_geo": cost_ratio_to_geo,
        },
        notes=(
            "Gradient-based inverse design: Adam descends through the "
            "differentiable FDTD forward; reference is the exact transfer-matrix "
            "optimum. The X-band-mean reflection cost is the cross-validated "
            "quantity (the degenerate εr surface admits multiple equal-cost "
            "designs)."
        ),
        loss_history=np.asarray(losses),
        eps_trace=np.asarray(eps_trace),
        eps_reference=np.asarray(tmm_eps),
        freqs_spectrum_hz=fs_band,
        r_initial=r_initial_band,
        r_final=r_final_band,
        r_reference=r_ref_band,
        reference_cost=tmm_cost,
    )


CASE_REGISTRY: list[GalleryCase] = [
    GalleryCase(
        id="multilayer_fresnel",
        title="Multilayer Fresnel Slab",
        description=(
            "Normal-incidence reflection/transmission of a dielectric slab, "
            "cross-validated against the exact transfer-matrix solution."
        ),
        validation_tier="E5",
        metric="|S11|^2=R and |S21|^2=T vs exact transfer matrix; mean |error| < 0.05",
        tolerance="mean |R/T error| < 0.05, |R+T-1| < 0.05",
        reference_solver="analytic transfer matrix (+ Meep secondary)",
        builder=_build_multilayer_fresnel,
        n_ports=2,
        has_smith=True,
        geometry_mode="slice2d",
        geometry_slice_axis=2,
        anim_slice_axis=2,
        anim_render_axis="z",
    ),
    GalleryCase(
        id="waveguide_wr90",
        title="WR-90 Waveguide Port (Empty Guide)",
        description=(
            "X-band WR-90 rectangular waveguide port S-matrix for a matched "
            "(empty) guide, against the exact |S11|=0 / |S21|=1 reference."
        ),
        validation_tier="E5",
        metric="max|S11| < 0.02, min|S21| > 0.97 vs analytic matched load",
        tolerance="max|S11| < 0.02, min|S21| > 0.97",
        reference_solver="analytic matched-load (+ MEEP/OpenEMS/Palace cross-check)",
        builder=_build_waveguide_wr90,
        n_ports=2,
        has_smith=True,
        # An empty guide has no dielectric to show in an εr slice; the 3D
        # renderer shows the guide domain + port instead (matplotlib backend).
        geometry_mode="geom3d",
        anim_slice_axis=2,
        anim_render_axis="z",
    ),
    GalleryCase(
        id="patch_antenna",
        title="2.4 GHz Patch Antenna (FR4)",
        description=(
            "Probe-fed rectangular microstrip patch antenna on FR4; the S11 "
            "resonance dip vs the analytic transmission-line (Balanis) estimate."
        ),
        validation_tier="E4",
        metric="resonance frequency within 20% of the analytic transmission-line estimate; |S11| passive",
        tolerance="resonance freq error < 20%, |S11| <= 1.05",
        reference_solver="analytic transmission-line (Balanis) estimate",
        builder=_build_patch_antenna,
        n_ports=1,
        has_smith=True,
        geometry_mode="geom3d",
        anim_slice_axis=2,
        anim_render_axis="z",
    ),
    GalleryCase(
        id="ar_coating_design",
        title="Anti-Reflection Coating Design",
        description=(
            "A 3-layer anti-reflection coating for a high-permittivity substrate, "
            "designed automatically by gradient-based optimization through the "
            "FDTD solver and checked against exact theory."
        ),
        validation_tier="OPT",
        metric="X-band-mean reflection cost vs the exact transfer-matrix optimum; ratio <= 1.2",
        tolerance="final cost <= 1.2x exact-theory optimum and <= 0.7x closed-form ladder",
        reference_solver="analytic transfer matrix (exact-theory optimum)",
        builder=_build_ar_coating_design,
        n_ports=1,
        has_smith=False,
        geometry_mode="slice2d",
        geometry_slice_axis=2,
        anim_slice_axis=2,
        anim_render_axis="z",
        has_animation=True,
        is_optimization=True,
    ),
]

CASE_BY_ID: dict[str, GalleryCase] = {c.id: c for c in CASE_REGISTRY}


# ---------------------------------------------------------------------------
# Manifest (pure, unit-testable)
# ---------------------------------------------------------------------------


def build_manifest(
    case: GalleryCase,
    *,
    assets: list[dict[str, Any]],
    runtime_seconds: float,
    passed: bool | None,
    metric_value: str,
    params: dict[str, Any],
    quick_smoke: bool,
    rfx_version: str | None = None,
    git_sha: str | None = None,
    hostname: str | None = None,
    backend: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the gallery ``manifest.json`` payload (no I/O, no sim).

    The shape mirrors :func:`rfx.artifacts._provenance`: a provenance record
    plus a ``reproducibility.status = "provenance-only"`` field. ``passed`` is
    forced to ``None`` for quick smoke runs so a smoke artifact can never be
    read as a validated result.

    ``assets`` is a list of ``{"filename", "type", "sha256", "size_bytes"}``
    dicts (``served_url`` is filled in here from the case id).
    """
    quick_smoke = bool(quick_smoke)
    validated_pass = None if quick_smoke else passed

    asset_entries: list[dict[str, Any]] = []
    for asset in assets:
        filename = asset["filename"]
        asset_entries.append(
            {
                "filename": filename,
                "type": asset["type"],
                "served_url": f"{SERVED_URL_PREFIX}/{case.id}/{filename}",
                "sha256": asset.get("sha256"),
                "size_bytes": asset.get("size_bytes"),
            }
        )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at or _utc_now(),
        "case_id": case.id,
        "title": case.title,
        "description": case.description,
        "quick_smoke": quick_smoke,
        "provenance": {
            "rfx_version": rfx_version if rfx_version is not None else _rfx_version(),
            "git_sha": git_sha if git_sha is not None else _git_sha(),
            "params": params,
            "runtime_seconds": round(float(runtime_seconds), 3),
            "hostname": hostname if hostname is not None else socket.gethostname(),
            "backend": backend if backend is not None else _detect_backend(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "validation": {
            "tier": case.validation_tier,
            "metric": case.metric,
            "metric_value": metric_value,
            "tolerance": case.tolerance,
            "reference_solver": case.reference_solver,
            "passed": validated_pass,
        },
        "reproducibility": {
            "status": "provenance-only",
            "limitations": [
                "Precomputed artifact: the public site does not re-run a solver.",
                "Quick-smoke artifacts use reduced grids/steps and are NOT validated."
                if quick_smoke
                else "Validated runs are produced on GPU/VESSL.",
            ],
        },
        "assets": asset_entries,
    }


def _detect_backend() -> str:
    try:
        import jax

        return str(jax.default_backend())
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Asset emission
# ---------------------------------------------------------------------------


def _emit_geometry(case: GalleryCase, result: "CaseResult", out_dir: Path) -> dict | None:
    """Render the geometry image (``geometry.png``). Returns the asset dict.

    ``geom3d`` cases use the 3D geometry renderer (matplotlib backend — pyvista
    is fragile headless); ``slice2d`` cases draw an εr cross-section. Returns
    ``None`` (and prints a warning) if no Simulation was attached or rendering
    fails, so a missing image never aborts the bundle.
    """
    import matplotlib.pyplot as plt

    if result.sim is None:
        return None
    geom_png = out_dir / "geometry.png"
    try:
        if case.geometry_mode == "geom3d":
            from rfx.visualize3d import plot_geometry_3d

            fig = plot_geometry_3d(
                result.sim,
                backend="matplotlib",
                show_ports=True,
                title=f"{case.title} — structure",
            )
        else:
            from rfx.visualize import plot_geometry_2d_slice

            fig = plot_geometry_2d_slice(
                result.sim,
                axis=case.geometry_slice_axis,
                title=f"{case.title} — structure (εᵣ)",
            )
        fig.savefig(geom_png, dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - rendering robustness
        print(f"    [warn] geometry render failed: {exc}", flush=True)
        return None
    return {"filename": geom_png.name, "type": "geometry-png"}


def _emit_animation(
    case: GalleryCase, result: "CaseResult", out_dir: Path, quick: bool
) -> dict | None:
    """Run a second snapshot forward and save the field animation.

    Captures a 2D slice (``SnapshotSpec.slice_axis`` is the INT axis index) to
    bound memory, then renders with ``save_field_animation`` (``slice_axis`` is
    the STRING axis name). The animation may come back as ``.gif`` if ffmpeg is
    absent — we emit whatever path the renderer returns. Returns ``None`` on any
    failure so a missing clip never aborts the bundle.
    """
    from rfx.simulation import SnapshotSpec
    from rfx.animation import save_field_animation

    anim_sim = result.anim_sim if result.anim_sim is not None else result.sim
    if anim_sim is None or not case.has_animation:
        return None
    n_anim_steps = 300 if quick else 1200
    try:
        anim_result = anim_sim.run(
            n_steps=n_anim_steps,
            compute_s_params=False,
            snapshot=SnapshotSpec(
                interval=10,
                components=(case.anim_component,),
                slice_axis=case.anim_slice_axis,
                slice_index=None,
            ),
        )
    except Exception as exc:  # pragma: no cover - solver robustness
        print(f"    [warn] field animation forward failed: {exc}", flush=True)
        return None

    # Save as GIF (Pillow writer — requires no external encoder, works on every
    # platform including GPU images without h264). MP4 was previously attempted
    # first but the conda ffmpeg on the GPU node lacks an h264 encoder, causing
    # broken partial files. GIF via PillowWriter is unambiguously reliable.
    try:
        saved = save_field_animation(
            anim_result,
            str(out_dir / "fields.gif"),
            component=case.anim_component,
            slice_axis=case.anim_render_axis,
            interval=3,
            fps=12,
        )
    except Exception as exc:  # pragma: no cover - rendering robustness
        print(f"    [warn] field animation failed: {exc}", flush=True)
        return None
    return {"filename": Path(saved).name, "type": "animation"}


def _emit_optimization(result: "CaseResult", out_dir: Path) -> list[dict]:
    """Emit the optimization-case plots + JSON. Returns the asset dicts."""
    import matplotlib.pyplot as plt

    assets: list[dict] = []
    losses = np.asarray(result.loss_history)
    eps_trace = np.asarray(result.eps_trace)
    eps_ref = np.asarray(result.eps_reference)
    iters_x = np.arange(len(losses))

    # convergence.png — cost vs iteration, with the exact-theory optimum line.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(iters_x, losses, "g-", lw=2, label="optimizer")
    if result.reference_cost is not None:
        ax.axhline(result.reference_cost, color="k", ls="--", label="exact-theory optimum")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Average reflection across the band")
    ax.set_title("How the design converges")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=130)
    plt.close(fig)
    assets.append({"filename": "convergence.png", "type": "convergence-png"})

    # design_evolution.png — per-layer εr vs iteration with exact-theory dashed.
    fig, ax = plt.subplots(figsize=(7, 5))
    for li in range(eps_trace.shape[1]):
        (line,) = ax.plot(iters_x, eps_trace[:, li], lw=2, label=f"layer {li + 1}")
        ax.axhline(eps_ref[li], color=line.get_color(), ls="--", alpha=0.6)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Layer permittivity εᵣ")
    ax.set_title("How each layer changes (dashed = exact-theory target)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "design_evolution.png", dpi=130)
    plt.close(fig)
    assets.append({"filename": "design_evolution.png", "type": "design-evolution-png"})

    # result_spectrum.png — reflection R(f): initial vs final vs exact-theory.
    fs_ghz = np.asarray(result.freqs_spectrum_hz) / 1e9
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(fs_ghz, np.clip(result.r_initial, 1e-7, None), "r-", lw=1.5, label="before")
    ax.semilogy(fs_ghz, np.clip(result.r_final, 1e-7, None), "g-", lw=2, label="after")
    ax.semilogy(fs_ghz, np.clip(result.r_reference, 1e-7, None), "k--", lw=1.2, label="exact-theory optimum")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Reflected power")
    ax.set_title("Reflection before vs after design")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "result_spectrum.png", dpi=130)
    plt.close(fig)
    assets.append({"filename": "result_spectrum.png", "type": "result-spectrum-png"})

    # optimization.json — rows [iter, cost, eps1, eps2, eps3].
    rows = [
        [int(i), float(losses[i]), *[float(e) for e in eps_trace[i]]]
        for i in range(len(losses))
    ]
    (out_dir / "optimization.json").write_text(
        json.dumps(
            {
                "columns": ["iter", "cost"]
                + [f"eps{j + 1}" for j in range(eps_trace.shape[1])],
                "reference_cost": float(result.reference_cost)
                if result.reference_cost is not None
                else None,
                "reference_eps": [float(e) for e in eps_ref],
                "rows": rows,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    assets.append({"filename": "optimization.json", "type": "optimization-json"})
    return assets


def _emit_case(case: GalleryCase, out_dir: Path, quick: bool) -> dict[str, Any]:
    """Run a case builder and write its artifact bundle. Returns the manifest."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rfx.io import write_touchstone
    from rfx.visualize import plot_s_params
    from rfx.smith import plot_smith

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = case.builder(quick)
    runtime = time.time() - t0

    freqs = np.asarray(result.freqs_hz)
    s = np.asarray(result.s_params)
    assets: list[dict[str, Any]] = []

    if not case.is_optimization:
        # S-parameter magnitude PNG
        sparam_png = out_dir / "sparams.png"
        fig = plot_s_params(s, freqs, db=True, title=f"{case.title} — |S| (dB)")
        fig.savefig(sparam_png, dpi=130)
        plt.close(fig)
        assets.append({"filename": sparam_png.name, "type": "sparam-plot-png"})

        # Smith PNG (S11 trajectory)
        if case.has_smith:
            smith_png = out_dir / "smith.png"
            ax = plot_smith(s[0, 0, :], freqs, title=f"{case.title} — S11")
            ax.figure.savefig(smith_png, dpi=130)
            plt.close(ax.figure)
            assets.append({"filename": smith_png.name, "type": "smith-png"})

        # Touchstone (.s1p/.s2p/.s4p depending on port count)
        touchstone = out_dir / f"sparams.s{case.n_ports}p"
        write_touchstone(
            touchstone,
            s,
            freqs,
            z0=50.0,
            comments=[
                f"rfx gallery case {case.id}",
                f"tier {case.validation_tier}; quick_smoke={quick}",
            ],
        )
        assets.append({"filename": touchstone.name, "type": "touchstone"})

        # Machine-readable S-params for the (gitops) interactive viewer draft.
        sparams_json = out_dir / "sparams.json"
        sparams_json.write_text(
            json.dumps(
                {
                    "case_id": case.id,
                    "freqs_hz": [float(f) for f in freqs],
                    "n_ports": int(case.n_ports),
                    "s": [
                        [[[float(v.real), float(v.imag)] for v in s[i, j, :]] for j in range(s.shape[1])]
                        for i in range(s.shape[0])
                    ],
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        assets.append({"filename": sparams_json.name, "type": "sparams-json"})
    else:
        # Optimization case: convergence / design-evolution / spectrum + JSON.
        assets.extend(_emit_optimization(result, out_dir))

    # Geometry image (what the structure is) — every case.
    geom_asset = _emit_geometry(case, result, out_dir)
    if geom_asset is not None:
        assets.append(geom_asset)

    # Field animation (what the physics does) — every case with a Simulation.
    anim_asset = _emit_animation(case, result, out_dir, quick)
    if anim_asset is not None:
        assets.append(anim_asset)

    # Compute sha256 + size for each written asset.
    for asset in assets:
        path = out_dir / asset["filename"]
        asset["sha256"] = sha256_file(path)
        asset["size_bytes"] = path.stat().st_size

    manifest = build_manifest(
        case,
        assets=assets,
        runtime_seconds=runtime,
        passed=result.passed,
        metric_value=result.metric_value,
        params={**result.params, "notes": result.notes},
        quick_smoke=quick,
    )
    # allow_nan=False: a non-finite metric/provenance value must fail loud here
    # rather than emit invalid JSON (Infinity/NaN) that strict parsers reject.
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Precompute rfx gallery artifact bundles (plots + Touchstone + manifest).",
    )
    parser.add_argument(
        "--case",
        default="all",
        help="Case id to build, or 'all' (default). "
        f"Known ids: {', '.join(c.id for c in CASE_REGISTRY)}.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output assets root (default: {DEFAULT_OUT}/<case_id>/).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced grid/steps for a fast CPU smoke test "
        "(manifest tagged quick_smoke=true, validation.passed=null).",
    )
    args = parser.parse_args(argv)

    if args.case == "all":
        cases = list(CASE_REGISTRY)
    else:
        case = CASE_BY_ID.get(args.case)
        if case is None:
            known = ", ".join(c.id for c in CASE_REGISTRY)
            parser.error(
                f"unknown case id {args.case!r}. Known case ids: {known} (or 'all')."
            )
        cases = [case]

    out_root: Path = args.out
    print(
        f"Precomputing {len(cases)} case(s) -> {out_root}"
        + (" [QUICK SMOKE]" if args.quick else "")
    )
    for case in cases:
        case_dir = out_root / case.id
        print(f"  [{case.id}] tier {case.validation_tier} -> {case_dir} ...", flush=True)
        manifest = _emit_case(case, case_dir, args.quick)
        v = manifest["validation"]
        print(
            f"    runtime={manifest['provenance']['runtime_seconds']}s "
            f"passed={v['passed']} metric=({v['metric_value']})"
        )
        for asset in manifest["assets"]:
            print(f"    asset {asset['type']:>16}: {asset['served_url']}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
