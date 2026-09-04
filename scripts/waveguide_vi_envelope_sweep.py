"""Harness for the waveguide `normalize=False` validity-envelope sweep.

PRE-MEASUREMENT. Nothing this script produces is a gate, a fixture or a
validated number. It runs the case list that
``docs/design_notes/waveguide_vi_envelope_sweep_predeclaration.md`` binds, and
writes one JSON per case the moment that case finishes.

Usage
-----
    python scripts/waveguide_vi_envelope_sweep.py <cases.json> <out_dir>
        [--only ID ...] [--dry] [--redo]

``cases.json`` is a list of case dicts (see
``scripts/waveguide_vi_envelope_cases.py``, which emits them). ``out_dir``
receives ``<case_id>.json`` per case plus ``<case_id>.log`` is left to the
caller/VESSL block.

Persistence rule (pre-declaration §8, and the two 8.5-hour runs that died in an
optional stage before their save): the per-case JSON is written by
``run_case``'s caller BEFORE anything optional runs, and there is no
aggregation stage in this file at all. Printing is not persisting.

Geometry, absorber and drive
----------------------------
The guide, the coarse lattice and the ladder come from the committed builder
``tests/_waveguide_chain_battery_fixture.py``; this module never restates
``A_M`` / ``B_M`` / ``DX_COARSE`` / the rung table. What it adds is the
band-scaled layout the pre-declaration §4 asks for (every clearance held at a
fixed number of ``lam_g(f_low)``, the fixture's 48-coarse-cell layout as the
unit, rounded to the coarse lattice), the K-rule absorber of §1.1, the two
discrete-cutoff locks of §4, and the y-asymmetric / y-centred blade DUTs of §5.2.

Precision
---------
float32 everywhere except the one float64 case (F2). ``jax_enable_x64`` is
process-global, so a float64 case is REFUSED unless the process was started
with ``RFX_SWEEP_X64=1``; the VESSL block runs that case list in its own
process. A silently downcast "float64 control" is a no-op, which is exactly
what a first pass of the scouting did.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

# The worktree must win over any installed rfx-fdtd (there IS one on this box,
# /usr/local/lib/python3.10/dist-packages). RFX_WT names it; without the env we
# fall back to this file's own repo root, which is the same thing when the
# script is run from the checkout.
_REPO_ROOT = os.environ.get("RFX_WT") or str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import jax  # noqa: E402

if os.environ.get("RFX_SWEEP_X64") == "1":
    # Process-global on purpose: this is a standalone measurement script, never
    # a pytest module (rfx CLAUDE.md forbids the module-level flip in tests).
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import rfx  # noqa: E402
from rfx import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402
import rfx.sources.waveguide_port as _wgp  # noqa: E402

import tests._waveguide_chain_battery_fixture as F  # noqa: E402

C0 = 299_792_458.0
A_M = F.A_M
B_M = F.B_M
DX_COARSE = F.DX_COARSE
FC_CONTINUOUS_HZ = C0 / (2.0 * A_M)          # 6.557140376e9

# rung -> literal cell size, from the fixture's own sweep ladder. Never A_M/N:
# the literal values keep b/dx integral and the reference/probe planes on
# integer cell counts (pre-declaration §5.1).
DX_BY_N = dict(zip(F.N_LADDER_SWEEP, F.DX_LADDER_SWEEP))

# The committed fixture's x-layout, read back as coarse-cell integers so this
# module carries no second copy of the numbers.
K_DOMAIN = int(round(F.DOMAIN_X_M / DX_COARSE))          # 48
K_PORT = int(round(F.PORT_LEFT_X_M / DX_COARSE))         # 5
K_REF = int(round(F.D_REF_M / DX_COARSE))                # 3
K_PROBE = int(round(F.D_PROBE_M / DX_COARSE))            # 10
LAMG_REF_M = None  # set below, after lam_g is defined

# Drive bandwidth rule, from the scouting the pre-declaration is written on:
# fractional bandwidth = 2.9 x (r_hi - r_lo) / r_centre.
BANDWIDTH_SHAPE_FACTOR = 2.9


# --------------------------------------------------------------------------
# guide arithmetic
# --------------------------------------------------------------------------

def lam_g(f_hz: float, fc_hz: float = FC_CONTINUOUS_HZ) -> float:
    """TE guide wavelength lam_0 / sqrt(1 - (fc/f)^2)."""
    return (C0 / float(f_hz)) / math.sqrt(1.0 - (float(fc_hz) / float(f_hz)) ** 2)


def v_group(f_hz: float, fc_hz: float = FC_CONTINUOUS_HZ) -> float:
    return C0 * math.sqrt(1.0 - (float(fc_hz) / float(f_hz)) ** 2)


LAMG_REF_M = lam_g(float(F.FREQS[0]))        # 0.0571 m, the committed band's low edge


def sinc_te10(N: int) -> float:
    """fc_discrete / fc_continuous for TE10 on an N-cell PEC guide: sinc(pi/2N)."""
    x = math.pi / (2 * N)
    return math.sin(x) / x


def sinc_te20(N: int) -> float:
    """fc_discrete(TE20) / (2 fc_continuous): sinc(pi/N)."""
    x = math.pi / N
    return math.sin(x) / x


def bandwidth_for(r_lo: float, r_hi: float) -> float:
    return BANDWIDTH_SHAPE_FACTOR * (r_hi - r_lo) / (0.5 * (r_lo + r_hi))


# --------------------------------------------------------------------------
# band placement — the three lock conventions of the pre-declaration
# --------------------------------------------------------------------------

def band_freqs(case: dict, N: int) -> np.ndarray:
    """Realized frequency bins for one case at one rung.

    ``lock`` selects the convention:

    * ``"none"``   — bins fixed on the continuous axis, f = r * fc_continuous.
    * ``"te10"``   — each rung sits at the same distance from ITS OWN discrete
      TE10 cutoff as the N=9 rung does, f -> f * sinc(pi/2N) / sinc(pi/18)
      (pre-declaration §4). Geometry, absorber and record length are untouched;
      only the frequency list moves, by < 0.6 %.
    * ``"te20"``   — the same construction against the discrete TE20 onset,
      f -> f * sinc(pi/N) / sinc(pi/9). Used by R6/R7, whose bands straddle a
      ceiling that is 2 % away from 2.000 f_c at N=9 (§4).
    * ``"te20_ratio"`` — the bins ARE ratios of the rung's own discrete TE20
      cutoff, f = ratio * 2 * fc_continuous * sinc(pi/N). The C leg (§5.2).

    ``freqs_hz`` in the case overrides everything (R5 and its descendants use
    the committed fixture band verbatim so §3.5 can compare against CHECK 1).
    """
    lock = case.get("lock", "none")
    if case.get("freqs_hz") is not None:
        f = np.asarray(case["freqs_hz"], dtype=float)
        if lock not in ("none", None):
            raise ValueError("explicit freqs_hz cannot also carry a lock")
        return f
    if lock == "te20_ratio":
        ratios = np.asarray(case["te20_ratios"], dtype=float)
        return ratios * 2.0 * FC_CONTINUOUS_HZ * sinc_te20(N)
    f = np.linspace(case["r_lo"], case["r_hi"], int(case["n_bins"])) * FC_CONTINUOUS_HZ
    if lock in ("none", None):
        return f
    if lock == "te10":
        return f * (sinc_te10(N) / sinc_te10(9))
    if lock == "te20":
        return f * (sinc_te20(N) / sinc_te20(9))
    raise ValueError(f"unknown lock {lock!r}")


def layout(r_lo: float) -> dict:
    """Coarse-cell x-layout scaled to this band's low-edge guide wavelength.

    Scaled from the NOMINAL band low edge, never from a rung's locked value, so
    every rung of a band realizes ONE geometry.
    """
    s = lam_g(r_lo * FC_CONTINUOUS_HZ) / LAMG_REF_M
    K = int(round(K_DOMAIN * s))
    K += K % 2                                   # keep the domain even
    f = K / K_DOMAIN
    k_port = int(round(K_PORT * f))
    k_ref = max(1, int(round(K_REF * f)))
    k_probe = max(k_ref + 1, int(round(K_PROBE * f)))
    return dict(scale=s, k_domain=K, k_port=k_port, k_ref=k_ref, k_probe=k_probe,
                lam_g_low_m=lam_g(r_lo * FC_CONTINUOUS_HZ))


def num_periods_for(case: dict, lay: dict, fc_hz: float, f_lo: float,
                    f_hi: float) -> tuple[float, dict]:
    """Record length = 2*t0 (full modulated-gaussian support) + n_trav domain
    traversals at the slowest in-band group velocity, expressed in periods of
    freq_max (``grid.num_timesteps``' own convention)."""
    f0 = 0.5 * (f_lo + f_hi)
    fwidth = f0 * float(case["bandwidth"])
    t0 = 5.0 / fwidth
    L = lay["k_domain"] * DX_COARSE
    t_trav = L / v_group(f_lo, fc_hz)
    T = 2.0 * t0 + float(case["n_trav"]) * t_trav
    return T * f_hi, dict(t0_s=t0, t_traverse_s=t_trav, t_record_s=T,
                          fwidth_hz=fwidth, v_group_low_m_s=v_group(f_lo, fc_hz))


# --------------------------------------------------------------------------
# the blade DUTs (pre-declaration §5.2)
# --------------------------------------------------------------------------
# Defined on the N=18 lattice: width, thickness and y-offset are integer
# multiples of dx(N=18), so N = 18/36/72 rasterize the IDENTICAL solid and the
# TE20 excitation amplitude cannot drift with rung. The pre-declaration fixes
# the lattice but not the numbers; these are the numbers, declared here.
DX18 = DX_BY_N[18]
BLADE_THICK_CELLS18 = 2          # 2 * 1.27 mm = 2.54 mm = one coarse cell
BLADE_WIDTH_CELLS18 = 6          # 6 * 1.27 mm = 7.62 mm = a/3 of the broad wall
BLADE_Y_LO_CELLS18 = {"blade_offset": 0, "blade_centred": 6}
DUTS = ("thru", "blade_offset", "blade_centred")


def blade_extent(dut: str, domain_x_m: float) -> tuple[tuple[float, float, float],
                                                       tuple[float, float, float]]:
    """[lo, hi) corners of the blade, absolute metres."""
    x_c = 0.5 * round(domain_x_m / DX_COARSE) * DX_COARSE
    half = 0.5 * BLADE_THICK_CELLS18 * DX18
    y_lo = BLADE_Y_LO_CELLS18[dut] * DX18
    y_hi = y_lo + BLADE_WIDTH_CELLS18 * DX18
    return (x_c - half, y_lo, 0.0), (x_c + half, y_hi, B_M)


def _add_dut(sim: Simulation, dut: str, domain_x_m: float, dx: float) -> dict:
    if dut == "thru":
        return {"dut": "thru"}
    if dut not in BLADE_Y_LO_CELLS18:
        raise ValueError(f"unknown dut {dut!r}; expected one of {DUTS}")
    lo, hi = blade_extent(dut, domain_x_m)
    for v in (*lo, *hi):
        n = v / dx
        if abs(n - round(n)) > 1e-9:
            raise AssertionError(
                f"blade face {v!r} is not on the dx={dx} lattice (n={n}) — the "
                "three rungs would rasterize different solids")
    sim.add_material("pec_like", eps_r=1.0, sigma=1e10)
    sim.add(Box(lo, hi), material="pec_like")
    return {"dut": dut, "blade_lo_m": list(lo), "blade_hi_m": list(hi),
            "blade_cells_at_dx": [(hi[k] - lo[k]) / dx for k in range(3)]}


# --------------------------------------------------------------------------
# the CHECK 2 mirror-covariance instrument (case F1 only)
# --------------------------------------------------------------------------
_ORIG_APPLY_E = _wgp.apply_waveguide_port_e


def _instrumented_apply_e(state, cfg, step, dt, dx):
    """The one-cell E-plane change CHECK 2 identified, applied at RUN time only.

    Shipped code corrects E at ``cfg.x_index`` for a ``+`` port and at
    ``cfg.x_index + 1`` for a ``-`` port, so the E-plane index sum is ``nx``
    where every other port index sums to the mirror-covariant ``nx-1``. Handing
    the ``-`` port a cfg with ``x_index - 1`` puts its correction back on
    ``x_index`` and restores the sum. Nothing in the library is edited; the
    shipped lane and this lane differ by this wrapper alone.
    """
    if cfg.direction.startswith("-") and cfg.h_inc_table.shape[0] > 1:
        cfg = cfg._replace(x_index=cfg.x_index - 1)
    return _ORIG_APPLY_E(state, cfg, step, dt, dx)


@contextlib.contextmanager
def port_variant(variant: str):
    if variant == "shipped":
        yield
        return
    if variant != "instrumented_e_plane":
        raise ValueError(f"unknown port variant {variant!r}")
    _wgp.apply_waveguide_port_e = _instrumented_apply_e
    try:
        yield
    finally:
        _wgp.apply_waveguide_port_e = _ORIG_APPLY_E


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def _boundary() -> BoundarySpec:
    return BoundarySpec(x="cpml", y=Boundary(lo="pec", hi="pec"),
                        z=Boundary(lo="pec", hi="pec"))


def _build(case: dict, freqs: np.ndarray, lay: dict, cpml_layers: int):
    N = int(case["N"])
    dx = DX_BY_N[N]
    b_m = A_M * float(case.get("b_over_a", 4.0 / 9.0))
    mult = N // 9
    domain_x = lay["k_domain"] * DX_COARSE
    x_port_l = lay["k_port"] * DX_COARSE
    x_port_r = domain_x - x_port_l
    ref_off = lay["k_ref"] * mult
    probe_off = lay["k_probe"] * mult
    assert abs(ref_off * dx - lay["k_ref"] * DX_COARSE) < 1e-15
    assert abs(probe_off * dx - lay["k_probe"] * DX_COARSE) < 1e-15
    f0 = float(case.get("f0_hz") or 0.5 * (freqs[0] + freqs[-1]))
    sim = Simulation(
        freq_max=float(freqs[-1]),
        domain=(domain_x, A_M, b_m),
        dx=dx,
        boundary=_boundary(),
        cpml_layers=int(cpml_layers),
        # Pinned in writing (pre-declaration §1.4): kappa_max > 1 is
        # monotonically harmful for a normally incident propagating TE10.
        cpml_kappa_max=1.0,
        precision=case.get("precision", "float32"),
    )
    dut_meta = _add_dut(sim, case.get("dut", "thru"), domain_x, dx)
    fj = jnp.asarray(freqs)
    sim.add_waveguide_port(x_port_l, direction="+x", mode=(1, 0), mode_type="TE",
                           freqs=fj, f0=f0, bandwidth=float(case["bandwidth"]),
                           ref_offset=ref_off, probe_offset=probe_off, name="left",
                           reference_plane=x_port_l + lay["k_ref"] * DX_COARSE)
    sim.add_waveguide_port(x_port_r, direction="-x", mode=(1, 0), mode_type="TE",
                           freqs=fj, f0=f0, bandwidth=float(case["bandwidth"]),
                           ref_offset=ref_off, probe_offset=probe_off, name="right",
                           reference_plane=x_port_r - lay["k_ref"] * DX_COARSE)
    meta = dict(dx_m=dx, N=N, b_m=b_m, b_over_a=b_m / A_M, domain_x_m=domain_x,
                port_x_m=[x_port_l, x_port_r], ref_offset_cells=ref_off,
                probe_offset_cells=probe_off, f0_hz=f0,
                bandwidth=float(case["bandwidth"]), layout=lay, **dut_meta)
    return sim, meta


def absorber_layers(case: dict, lay: dict, fc_numerical_hz: float) -> dict:
    """Pre-declaration §1.1: ``cpml_layers = ceil(K * lam_g(f_low) / dx)`` with
    lam_g at the band's own low edge and fc the REALIZED numerical TE10 cutoff.

    ``f_low`` is the band's NOMINAL low edge, not a rung's locked value, so one
    band has one absorber thickness across its ladder. The exactly-scaled
    1x/2x/4x count is recorded beside the realized one so the ceil's rung
    drift stays visible rather than being assumed away.
    """
    N = int(case["N"])
    dx = DX_BY_N[N]
    K = float(case["K"])
    f_low = float(case.get("absorber_f_low_hz") or case["r_lo"] * FC_CONTINUOUS_HZ)
    lg = lam_g(f_low, fc_numerical_hz)
    layers = int(math.ceil(K * lg / dx))
    layers9 = int(math.ceil(K * lg / DX_COARSE))
    return dict(cpml_layers=layers, cpml_layers_exact_scaled=layers9 * (N // 9),
                absorber_f_low_hz=f_low, absorber_lam_g_m=lg,
                K_declared=K, K_realized=layers * dx / lg,
                absorber_thickness_m=layers * dx)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def rfx_provenance() -> dict:
    path = os.path.dirname(rfx.__file__)
    sha = os.environ.get("RFX_EXPECT_SHA")
    try:
        sha_live = subprocess.run(
            ["git", "-C", os.path.dirname(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:                                        # noqa: BLE001
        sha_live = None
    return dict(rfx_file=rfx.__file__, rfx_package_dir=path,
                rfx_version=getattr(rfx, "__version__", None),
                rfx_sha_git=sha_live, rfx_sha_env=sha,
                jax_version=jax.__version__, jax_backend=jax.default_backend(),
                jax_devices=[str(d) for d in jax.devices()],
                jax_enable_x64=bool(jax.config.jax_enable_x64),
                sys_path0=sys.path[0])


def assert_worktree_rfx() -> dict:
    prov = rfx_provenance()
    root = os.path.realpath(_REPO_ROOT)
    got = os.path.realpath(prov["rfx_package_dir"])
    if not got.startswith(root):
        raise SystemExit(
            f"FATAL: rfx resolved to {got}, not the worktree {root}. A stale "
            "editable install has shadowed the checkout here before — refusing "
            "to measure the wrong code.")
    return prov


# --------------------------------------------------------------------------
# run one case
# --------------------------------------------------------------------------

def run_case(case: dict) -> dict:
    t_start = time.time()
    N = int(case["N"])
    if N not in DX_BY_N:
        raise ValueError(f"N={N} is not a ladder rung {sorted(DX_BY_N)}")
    precision = case.get("precision", "float32")
    if precision == "float64" and not jax.config.jax_enable_x64:
        raise SystemExit(
            "FATAL: a float64 case was handed to a process without x64. Set "
            "RFX_SWEEP_X64=1 and run this case list in its own process; a "
            "silently downcast float64 control is a no-op.")
    out: dict = dict(case)
    out["provenance"] = rfx_provenance()

    r_lo = float(case.get("r_lo") or np.asarray(case["freqs_hz"]).min() / FC_CONTINUOUS_HZ)
    lay = layout(r_lo)
    freqs = band_freqs(case, N)

    # Pass 1 — realized numerical TE10 cutoff of the rasterized guide, read at
    # cpml_layers=8 so the absorber cannot feed back into its own rule.
    probe_case = dict(case, K=None)
    probe_sim, _ = _build(probe_case, freqs, lay, cpml_layers=8)
    fc_num = float(F.numerical_te10_cutoff_hz(probe_sim))
    probe_grid = probe_sim._build_grid()
    out["probe_grid_shape_at_8_layers"] = [int(v) for v in probe_grid.shape]
    out["fc_numerical_hz"] = fc_num
    out["fc_continuous_hz"] = FC_CONTINUOUS_HZ
    out["fc_numerical_over_continuous"] = fc_num / FC_CONTINUOUS_HZ
    del probe_sim, probe_grid

    # Pass 2 — the absorber, then the real build.
    absorber = absorber_layers(dict(case, r_lo=r_lo), lay, fc_num)
    out.update(absorber)
    sim, meta = _build(case, freqs, lay, absorber["cpml_layers"])
    out.update(meta)
    grid = sim._build_grid()
    npp, timing = num_periods_for(case, lay, fc_num, float(freqs[0]), float(freqs[-1]))
    num_periods = float(case.get("num_periods") or math.ceil(npp))
    n_steps = int(grid.num_timesteps(num_periods))
    out.update(num_periods=num_periods, n_steps=n_steps, dt_s=float(grid.dt),
               timing=timing, grid_shape=[int(v) for v in grid.shape],
               cell_steps=float(np.prod(grid.shape)) * n_steps * 2,
               freqs_hz=[float(v) for v in freqs],
               r_bins_continuous=[float(v / FC_CONTINUOUS_HZ) for v in freqs],
               precision=precision,
               port_variant=case.get("port_variant", "shipped"))
    print(json.dumps({k: out[k] for k in
                      ("case_id", "N", "dx_m", "cpml_layers", "K_realized",
                       "grid_shape", "n_steps", "num_periods", "cell_steps")},
                     indent=1), flush=True)
    if case.get("dry"):
        out["wall_time_s"] = time.time() - t_start
        return out

    # ---- preflight, captured verbatim (banner included) -------------------
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(buf):
            report = sim.preflight()
    out["preflight_banner"] = buf.getvalue()
    out["preflight_findings"] = [
        dict(severity=str(getattr(i, "severity", "warning")),
             code=str(getattr(i, "code", "uncoded")),
             source=(None if getattr(i, "source", None) is None else str(i.source)),
             message=str(i))
        for i in report]
    out["preflight_warnings_raised"] = [
        f"{w.category.__name__}: {w.message}" for w in wl]
    for p in out["preflight_findings"]:
        print("PREFLIGHT", p["severity"], p["code"], p["message"], flush=True)

    # the port's own discrete cutoff, per rung
    try:
        cfg0 = sim._build_waveguide_port_config(
            sim._waveguide_ports[0], grid, jnp.asarray(freqs), n_steps)
        cfg1 = sim._build_waveguide_port_config(
            sim._waveguide_ports[1], grid, jnp.asarray(freqs), n_steps)
        out["port_f_cutoff_hz"] = float(cfg0.f_cutoff)
        out["r_bins_discrete"] = [float(f / float(cfg0.f_cutoff)) for f in freqs]
        out["port_indices"] = dict(
            nx=int(grid.shape[0]),
            x_index=[int(cfg0.x_index), int(cfg1.x_index)],
            ref_x=[int(cfg0.ref_x), int(cfg1.ref_x)],
            probe_x=[int(cfg0.probe_x), int(cfg1.probe_x)])
        out["discrete_te20_cutoff_hz"] = 2.0 * FC_CONTINUOUS_HZ * sinc_te20(N)
        out["r_bins_over_discrete_te20"] = [
            float(f / out["discrete_te20_cutoff_hz"]) for f in freqs]
        del cfg0, cfg1
    except Exception as e:                                   # noqa: BLE001
        out["port_f_cutoff_hz"] = None
        out["port_cutoff_error"] = repr(e)
        out["r_bins_discrete"] = None

    # ---- the solve -------------------------------------------------------
    t_solve = time.time()
    with port_variant(case.get("port_variant", "shipped")):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            res = sim.compute_waveguide_s_matrix(n_steps=n_steps, normalize=False)
            S = np.asarray(res.s_params).astype(np.complex128)
    out["solve_wall_time_s"] = time.time() - t_solve
    out["solve_warnings"] = sorted({f"{w.category.__name__}: {w.message}" for w in wl})

    sd = res.settling_db
    out["settling_db"] = (None if sd is None else
                          {"left": float(np.asarray(sd).ravel()[0]),
                           "right": float(np.asarray(sd).ravel()[1])})
    out["reference_planes_m"] = [float(v) for v in np.asarray(res.reference_planes).ravel()]

    s11, s12, s21, s22 = S[0, 0], S[0, 1], S[1, 0], S[1, 1]
    a11, a12, a21, a22 = (np.abs(x) for x in (s11, s12, s21, s22))
    sym = 0.5 * (a11 + a22)
    asy = 0.5 * np.abs(a11 - a22)
    headline = np.maximum(a11, a22)                 # == sym + asy, per bin
    col0 = a11 ** 2 + a21 ** 2
    col1 = a22 ** 2 + a12 ** 2
    out.update(
        abs_s11=[float(v) for v in a11], abs_s22=[float(v) for v in a22],
        abs_s21=[float(v) for v in a21], abs_s12=[float(v) for v in a12],
        per_bin_sym=[float(v) for v in sym], per_bin_asy=[float(v) for v in asy],
        per_bin_headline=[float(v) for v in headline],
        band_mean_headline=float(headline.mean()),
        band_max_headline=float(headline.max()),
        band_argmax_headline_bin=int(np.argmax(headline)),
        band_mean_sym=float(sym.mean()), band_mean_asy=float(asy.mean()),
        band_max_sym=float(sym.max()), band_max_asy=float(asy.max()),
        band_mean_abs_s11=float(a11.mean()), band_max_abs_s11=float(a11.max()),
        band_mean_abs_s22=float(a22.mean()), band_max_abs_s22=float(a22.max()),
        two_port_residual_D=float(np.abs(a11 - a22).max()),
        column_power=[[float(v) for v in col0], [float(v) for v in col1]],
        column_power_max=float(max(col0.max(), col1.max())),
        column_power_min=float(min(col0.min(), col1.min())),
        headline_identity_max_err=float(np.abs(headline - (sym + asy)).max()),
        n_nan=int(np.isnan(S).sum()),
        arg_s21_deg=[float(v) for v in np.degrees(np.angle(s21))],
        arg_s12_deg=[float(v) for v in np.degrees(np.angle(s12))],
    )
    out["wall_time_s"] = time.time() - t_start
    return out


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", help="case-list JSON (a list of case dicts)")
    ap.add_argument("out_dir", help="directory receiving one JSON per case")
    ap.add_argument("--only", nargs="*", default=None, help="run only these case ids")
    ap.add_argument("--lane", default=None, choices=["local", "vessl"],
                    help="run only the cases tagged for this lane (§8)")
    ap.add_argument("--dry", action="store_true", help="build and size, do not solve")
    ap.add_argument("--redo", action="store_true", help="re-run cases already persisted")
    a = ap.parse_args(argv)

    prov = assert_worktree_rfx()
    print("rfx provenance: " + json.dumps(prov), flush=True)

    cases = json.loads(Path(a.cases).read_text())
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fails = 0
    for case in cases:
        cid = case["case_id"]
        if a.only and cid not in a.only:
            continue
        if a.lane and case.get("lane") != a.lane:
            continue
        dest = out_dir / f"{cid}.json"
        if dest.exists() and not a.redo:
            print(f"SKIP {cid} (already persisted)", flush=True)
            continue
        print(f"===== case {cid} =====", flush=True)
        try:
            rec = run_case(dict(case, dry=case.get("dry", False) or a.dry))
        except SystemExit:
            raise
        except Exception as e:                               # noqa: BLE001
            traceback.print_exc()
            rec = dict(case, error=f"{type(e).__name__}: {e}",
                       traceback=traceback.format_exc(),
                       provenance=prov, wall_time_s=None)
            fails += 1
        # PERSIST FIRST. Nothing optional runs before this line.
        tmp = dest.with_suffix(".json.part")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(dest)
        if "error" in rec:
            print(f"FAILED {cid}: {rec['error']}", flush=True)
            continue
        if case.get("dry") or a.dry:
            print(f"DRY {cid} persisted", flush=True)
            continue
        print("DONE {} mean_headline={:.6e} max_headline={:.6e} mean_sym={:.6e} "
              "mean_asy={:.6e} settling={} colpow=[{:.7f},{:.7f}] wall={:.1f}s"
              .format(cid, rec["band_mean_headline"], rec["band_max_headline"],
                      rec["band_mean_sym"], rec["band_mean_asy"],
                      rec["settling_db"], rec["column_power_min"],
                      rec["column_power_max"], rec["wall_time_s"]), flush=True)
    print(f"vi_envelope_failed_cases={fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
