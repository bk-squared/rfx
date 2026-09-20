"""What the near field at the driven port's own reference plane is made of (#873).

Pre-declaration and decision rule:
``docs/design_notes/waveguide_driven_plane_near_field_composition_predeclaration.md``
(committed before any projection coefficient here was computed).

The two closed lanes on this issue modelled the empty-guide ``normalize=False``
excess as a spurious REFLECTION (PR #880, non-closing, and on the N+1-cell port
#868/#889 has since fixed) and as a port-local V/I TRANSFER error (PR #1081,
refuted by a bound: ``a1`` and ``b2`` are the same ``(V + Z*I)/2`` combination
read by two identical ports, so a common transfer error cancels in |S21|).
PR #1081 located the excess at the driven port's own reference plane, 7.62 mm
from an active TFSF launch, but did not measure what the field there is made of.
This run does: it projects the recorded transverse E and H at that plane onto the
discrete TE/TM eigenmodes of the port cross-section built on the arm's own grid.

Stages
------
``control``    CPU FDTD, unpatched.  The three thru cells as the battery runs
               them.  Reproduce-gate against the frozen artifact
               ``tests/fixtures/waveguide_chain_battery/fixture_v18_close.json``,
               plus the port's own modal V/I time series, which the ``compose``
               stage's capture must reproduce.
``compose``    CPU FDTD, one driven run per rung.  The SAME solve: only the
               recording changes.  ``modal_voltage`` / ``modal_current`` are
               replaced by functions that store the whole aperture slice they
               would otherwise contract, so the per-step record holds
               ``(E_y, E_z)`` and the co-located ``(H_y, H_z)`` at each of the
               four recorded planes.  Post-scan: the port's own rectangular
               full-record DFT per aperture cell, then a least-squares modal
               projection in the aperture measure.
``falsifier``  CPU FDTD.  Section 6 of the pre-declaration: the coarse thru cell
               with the record plane moved off 3 cells (7.62 mm) to 4 cells
               (10.16 mm) and 11 cells (27.94 mm), against reduction factors
               tabulated per candidate mode BEFORE the run.

Usage (from the repository root, no editable install needed)::

    PYTHONPATH=. python scripts/diagnostics/waveguide_driven_plane_near_field_composition.py \
        --out tests/fixtures/waveguide_false_lane_column_power/near_field_composition.json \
        --figure docs/design_notes/figures/waveguide_driven_plane_near_field_composition.png

This writes a NEW artifact.  ``suspects.json`` (PR #880) and
``transmission_tilt.json`` (PR #1081) in the same directory are neither read for
their measurements nor overwritten; the latter's plane table is quoted by the
results note and cited by key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests/fixtures/waveguide_chain_battery/fixture_v18_close.json"
PREDECLARATION = (
    "docs/design_notes/waveguide_driven_plane_near_field_composition_predeclaration.md"
)
C0 = 299_792_458.0
RUNGS = ("coarse", "mid", "fine")
# The four recorded planes, in the order the report lists them.
PLANES = ("drive_ref", "drive_probe", "recv_probe", "recv_ref")
# Modes whose coefficient is tabulated per bin whatever its size (the named
# candidates of the pre-declaration, section 2.2).  Everything above the
# reporting floor is dumped as well.
NAMED = (("TE", 2, 0), ("TE", 0, 1), ("TE", 1, 1), ("TM", 1, 1),
         ("TE", 3, 0), ("TE", 5, 0), ("TE", 7, 0))


# --------------------------------------------------------------------------
# provenance helpers
# --------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is reported, never guessed
        return "unknown"


# --------------------------------------------------------------------------
# the discrete transverse basis, on the arm's own grid
# --------------------------------------------------------------------------
def _eig1d(widths: np.ndarray, bc: str) -> tuple[np.ndarray, np.ndarray]:
    """Unit-mass-normalised eigenpairs of the 1-D Galerkin transverse operator."""
    import rfx.sources._waveguide_modes as wm

    K, m = wm._galerkin_stiffness_mass_1d(np.asarray(widths, float), bc=bc)
    s = 1.0 / np.sqrt(m)
    A = (s[:, None] * K) * s[None, :]
    A = 0.5 * (A + A.T)
    evals, psi = np.linalg.eigh(A)
    return np.maximum(evals, 0.0), psi * s[:, None]


def build_basis(a: float, b: float, u_widths, v_widths, dA, h_offset) -> dict:
    """Separable discrete TE + TM basis with unambiguous (m, n) labels.

    ``_discrete_te_mode_profiles`` / ``_discrete_tm_mode_profiles`` pick an
    eigenvector out of the 2-D spectrum by overlap with an analytic shape, and on
    this grid several requested (m, n) land on the SAME eigenvector (at the
    coarse rung TE03, TE13, TE23 and TE70 all return kc = 945.01), which makes a
    basis assembled that way singular.  The operator is separable, so the basis
    is built here as the tensor product of the two 1-D eigenproblems --
    Hx_mn = phi_m(u) (x) chi_n(v), kc^2 = lambda_m + mu_n -- giving every (m, n)
    a unique vector and no picker.  Everything else (the transverse gradient, the
    dual co-location stencil, the amplitude normalisation) is the port's own.
    """
    import rfx.sources._waveguide_modes as wm

    u_widths = np.asarray(u_widths, float)
    v_widths = np.asarray(v_widths, float)
    dA = np.asarray(dA, float)
    nu, nv = len(u_widths), len(v_widths)
    u_c = np.cumsum(u_widths) - 0.5 * u_widths
    v_c = np.cumsum(v_widths) - 0.5 * v_widths

    def _finish(ey, ez, kc):
        hy = -wm._shift_profile_to_dual(ez, h_offset)
        hz = wm._shift_profile_to_dual(ey, h_offset)
        e_norm = np.sqrt(float(np.sum((ey ** 2 + ez ** 2) * dA)))
        h_norm = np.sqrt(float(np.sum((hy ** 2 + hz ** 2) * dA)))
        if e_norm <= 0 or h_norm <= 0:
            return None
        return dict(ey=ey / e_norm, ez=ez / e_norm,
                    hy=hy / h_norm, hz=hz / h_norm, kc=float(kc))

    out: dict = {}
    lu, Pu = _eig1d(u_widths, "neumann")
    lv, Pv = _eig1d(v_widths, "neumann")
    for m in range(nu):
        phi = Pu[:, m]
        if float(np.dot(phi, np.cos(m * np.pi * u_c / a))) < 0:
            phi = -phi
        for n in range(nv):
            if m == 0 and n == 0:
                continue
            chi = Pv[:, n]
            if float(np.dot(chi, np.cos(n * np.pi * v_c / b))) < 0:
                chi = -chi
            hx = phi[:, None] * chi[None, :]
            ey = wm._cell_centred_gradient(hx, v_widths, axis=1, bc="neumann")
            ez = -wm._cell_centred_gradient(hx, u_widths, axis=0, bc="neumann")
            entry = _finish(ey, ez, np.sqrt(lu[m] + lv[n]))
            if entry is not None:
                out[("TE", m, n)] = entry
    lu, Pu = _eig1d(u_widths, "dirichlet")
    lv, Pv = _eig1d(v_widths, "dirichlet")
    for m in range(nu):
        phi = Pu[:, m]
        if float(np.dot(phi, np.sin((m + 1) * np.pi * u_c / a))) < 0:
            phi = -phi
        for n in range(nv):
            chi = Pv[:, n]
            if float(np.dot(chi, np.sin((n + 1) * np.pi * v_c / b))) < 0:
                chi = -chi
            ex = phi[:, None] * chi[None, :]
            ey = wm._cell_centred_gradient(ex, u_widths, axis=0, bc="dirichlet")
            ez = wm._cell_centred_gradient(ex, v_widths, axis=1, bc="dirichlet")
            entry = _finish(ey, ez, np.sqrt(lu[m] + lv[n]))
            if entry is not None:
                out[("TM", m + 1, n + 1)] = entry
    return out


def label(key) -> str:
    return f"{key[0]}{key[1]}{key[2]}"


# --------------------------------------------------------------------------
# fixture capture
# --------------------------------------------------------------------------
class _Stop(Exception):
    pass


def _capture_setup(dut: str, dx: float, num_periods: float = 40.0) -> dict:
    """Run the Simulation front end far enough to get (grid, materials, cfgs)."""
    import rfx.sparams.waveguide as swg
    from tests import _waveguide_chain_battery_fixture as F

    held: dict = {}

    def stop(grid, materials, cfgs, n_steps, **kw):
        held.update(grid=grid, materials=materials, cfgs=list(cfgs),
                    n_steps=int(n_steps), kw=dict(kw))
        raise _Stop()

    original = swg.extract_waveguide_s_matrix
    swg.extract_waveguide_s_matrix = stop
    try:
        sim = F.build_simulation(dut, dx)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                sim.compute_waveguide_s_matrix(num_periods=num_periods,
                                               normalize=False)
            except _Stop:
                pass
        held["warnings"] = sorted(
            {f"{w.category.__name__}: {w.message}" for w in caught})
    finally:
        swg.extract_waveguide_s_matrix = original
    return held


def _run_kwargs(kw: dict) -> dict:
    """The subset of the extractor's kwargs that ``rfx.simulation.run`` takes."""
    keys = ("boundary", "cpml_axes", "pec_axes", "periodic", "debye", "lorentz",
            "aniso_eps", "conformal_weights", "aniso_inv_eps", "pec_edge_masks",
            "sheet_impedance")
    return {k: kw[k] for k in keys if k in kw}


# --------------------------------------------------------------------------
# stage "control"
# --------------------------------------------------------------------------
def stage_control(num_periods: float = 40.0) -> dict:
    """The battery's own run, unpatched: reproduce-gate + reference V/I series."""
    import rfx.sources.waveguide_port as wp
    import rfx.sparams.waveguide as swg
    from tests import _waveguide_chain_battery_fixture as F

    frozen = json.loads(FIXTURE.read_text())
    out: dict = {}
    stash: list = []
    original = wp.extract_waveguide_port_waves

    def spy(cfg, *, ref_shift=0.0):
        stash.append(cfg)
        return original(cfg, ref_shift=ref_shift)

    wp.extract_waveguide_port_waves = spy
    swg.extract_waveguide_port_waves = spy
    try:
        for rung, dx in zip(RUNGS, F.DX_LADDER):
            stash.clear()
            sim = F.build_simulation("thru", dx)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = sim.compute_waveguide_s_matrix(num_periods=num_periods,
                                                     normalize=False)
            texts = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
            s = np.asarray(res.s_params)
            col0 = np.abs(s[0, 0]) ** 2 + np.abs(s[1, 0]) ** 2
            cell = next(c for c in frozen["cells"]
                        if c["dut"] == "thru" and c["lane"] == "false"
                        and c["rung"] == rung)
            frozen_col = np.asarray(cell["column_power_per_bin"], float)[0]
            drive_cfg = stash[0]
            out[rung] = {
                "dx_m": float(dx),
                "column_power_minus_1_col0": (col0 - 1.0).tolist(),
                "max_positive_column_power_minus_1": float(np.max(col0 - 1.0)),
                "frozen_max_positive_column_power_minus_1":
                    float(np.max(frozen_col - 1.0)),
                "reproduce_gate_rel_diff": float(
                    abs(np.max(col0 - 1.0) / np.max(frozen_col - 1.0) - 1.0)),
                "max_abs_per_bin_diff_vs_frozen": float(
                    np.max(np.abs(col0 - frozen_col))),
                "band_mean_s11_mag": float(np.mean(np.abs(s[0, 0]))),
                "s11_mag": np.abs(s[0, 0]).tolist(),
                "s21_mag2_minus_1": (np.abs(s[1, 0]) ** 2 - 1.0).tolist(),
                "settling_db": [float(x) for x in np.asarray(res.settling_db)],
                "warnings_verbatim": texts,
                "_v_ref_t": np.asarray(drive_cfg.v_ref_t, float),
                "_i_ref_t": np.asarray(drive_cfg.i_ref_t, float),
                "_n_steps_recorded": int(drive_cfg.n_steps_recorded),
            }
    finally:
        wp.extract_waveguide_port_waves = original
        swg.extract_waveguide_port_waves = original
    return out


# --------------------------------------------------------------------------
# stage "compose": the aperture capture
# --------------------------------------------------------------------------
def _aperture_capture_run(setup: dict, drive_idx: int = 0):
    """One driven run whose per-step record holds the whole aperture slice."""
    import jax.numpy as jnp
    import rfx.sources.waveguide_port as wp
    from rfx.simulation import run as run_simulation

    def modal_voltage_capture(state, cfg, x_idx, dx):
        e_u = wp._plane_field(getattr(state, cfg.e_u_component), cfg, x_idx)
        e_v = wp._plane_field(getattr(state, cfg.e_v_component), cfg, x_idx)
        return jnp.stack([e_u, e_v])

    def modal_current_capture(state, cfg, x_idx, dx):
        h_u = wp._plane_h_field_at_dual(
            getattr(state, cfg.h_u_component), cfg, x_idx, cfg.h_offset)
        h_v = wp._plane_h_field_at_dual(
            getattr(state, cfg.h_v_component), cfg, x_idx, cfg.h_offset)
        return jnp.stack([h_u, h_v])

    cfgs = setup["cfgs"]
    driven = []
    for idx, cfg in enumerate(cfgs):
        n_t = int(cfg.v_probe_t.shape[0])
        nu = int(cfg.u_hi - cfg.u_lo)
        nv = int(cfg.v_hi - cfg.v_lo)
        wide = jnp.zeros((n_t, 2, nu, nv), dtype=cfg.v_probe_t.dtype)
        driven.append(cfg._replace(
            src_amp=cfg.src_amp if idx == drive_idx else 0.0,
            v_probe_t=wide, v_ref_t=wide, i_probe_t=wide, i_ref_t=wide,
            v_inc_t=jnp.zeros_like(cfg.v_inc_t),
            n_steps_recorded=jnp.zeros((), dtype=jnp.int32),
        ))

    saved = (wp.modal_voltage, wp.modal_current)
    wp.modal_voltage, wp.modal_current = (modal_voltage_capture,
                                          modal_current_capture)
    try:
        result = run_simulation(
            setup["grid"], setup["materials"], setup["n_steps"],
            waveguide_ports=driven, **_run_kwargs(setup["kw"]))
    finally:
        wp.modal_voltage, wp.modal_current = saved
    return list(result.waveguide_ports)


def _rect_dft_field(series: np.ndarray, freqs: np.ndarray, dt: float,
                    n_valid: int) -> np.ndarray:
    """``_rect_dft`` applied per aperture cell, in float64.

    Same kernel as ``rfx.sources.waveguide_port._rect_dft``:
    ``2*dt*sum_n f(t_n) exp(-j w t_n)`` over ``n < n_valid``, no window.
    """
    series = np.asarray(series, float)
    n_t = series.shape[0]
    t = np.arange(n_t, dtype=float) * float(dt)
    masked = series.reshape(n_t, -1).copy()
    masked[n_valid:] = 0.0
    phase = np.exp(-1j * 2.0 * np.pi * freqs[None, :] * t[:, None])
    spec = 2.0 * float(dt) * (phase.T @ masked)
    return spec.reshape((len(freqs),) + series.shape[1:])


def _project(field_spec: np.ndarray, design: np.ndarray, w: np.ndarray,
             pinv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares modal coefficients and the unexplained residual fraction.

    ``design`` is the UNWEIGHTED (K, M) profile matrix; ``pinv`` is the
    pseudo-inverse of its ``sqrt(dA)``-weighted transpose, so the solve
    minimises the residual in the aperture measure while the reconstruction is
    compared against the field itself.
    """
    n_f = field_spec.shape[0]
    flat = field_spec.reshape(n_f, -1).T                  # (M, n_f)
    coeffs = pinv @ (flat * np.sqrt(w)[:, None])          # (K, n_f)
    recon = design.T @ coeffs                             # (M, n_f)
    num = np.sqrt(np.sum(np.abs(flat - recon) ** 2 * w[:, None], axis=0))
    den = np.sqrt(np.sum(np.abs(flat) ** 2 * w[:, None], axis=0))
    return coeffs, num / np.maximum(den, 1e-300)


def _calibrated_split(planes: dict, ey10, ez10, hy10, hz10, dA, omega, dt,
                      alt_ref: str | None = None) -> None:
    """Split each plane's field into "what a clean TE10 wave looks like" and the rest.

    Basis-free, and forced by a measured fact: the recorded ``Ez``/``Hy`` are
    NODE-registered in u while the port's transverse template is cell-centred, so
    the field's u-profile lives in a space the cell-centred mode family does not
    span (its wall node is pinned to zero and no cell-centred gradient is). A
    least-squares decomposition in that family is therefore ill-conditioned, and
    its TE10 coefficient inherits the conditioning.

    The receiving reference plane, 88.9 mm from the driven source, carries every
    evanescent mode below 1e-4 by its own alpha, so the shape it presents IS a
    clean single-TE10 wave as this extractor sees it. Taking that measured shape
    as the reference needs no basis and no inversion: at every plane the field is
    split into its component along the reference shape and the orthogonal rest,
    and the extractor's V and I are split linearly with it.
    """
    w = np.concatenate([dA.ravel(), dA.ravel()])
    prof_e = np.concatenate([ey10.ravel(), ez10.ravel()])
    prof_h = np.concatenate([hy10.ravel(), hz10.ravel()])

    def flat(spec):
        n_f = spec.shape[0]
        return spec.reshape(n_f, -1).T                    # (M, n_f)

    ref_e = flat(planes["recv_ref"]["_e_spec"])
    ref_h = flat(planes["recv_ref"]["_h_spec"])
    ne = np.sqrt(np.sum(np.abs(ref_e) ** 2 * w[:, None], axis=0))
    nh = np.sqrt(np.sum(np.abs(ref_h) ** 2 * w[:, None], axis=0))
    ue, uh = ref_e / ne, ref_h / nh                       # unit reference shapes
    ve_ref = (prof_e * w) @ ue                            # V of one unit of it
    ih_ref = ((prof_h * w) @ uh) * np.exp(+1j * omega * 0.5 * dt)

    if alt_ref is not None:
        alt_e = flat(planes[alt_ref]["_e_spec"])
        alt_h = flat(planes[alt_ref]["_h_spec"])
        ue = alt_e / np.sqrt(np.sum(np.abs(alt_e) ** 2 * w[:, None], axis=0))
        uh = alt_h / np.sqrt(np.sum(np.abs(alt_h) ** 2 * w[:, None], axis=0))
        ve_ref = (prof_e * w) @ ue
        ih_ref = ((prof_h * w) @ uh) * np.exp(+1j * omega * 0.5 * dt)

    for p in planes.values():
        fe, fh = flat(p["_e_spec"]), flat(p["_h_spec"])
        a = np.sum(fe * np.conj(ue) * w[:, None], axis=0)
        c = np.sum(fh * np.conj(uh) * w[:, None], axis=0)
        e_perp = fe - ue * a
        h_perp = fh - uh * c
        p["perp_fraction_e"] = np.sqrt(
            np.sum(np.abs(e_perp) ** 2 * w[:, None], axis=0)) / np.maximum(
                np.sqrt(np.sum(np.abs(fe) ** 2 * w[:, None], axis=0)), 1e-300)
        p["perp_fraction_h"] = np.sqrt(
            np.sum(np.abs(h_perp) ** 2 * w[:, None], axis=0)) / np.maximum(
                np.sqrt(np.sum(np.abs(fh) ** 2 * w[:, None], axis=0)), 1e-300)
        p["V_te10_cal"] = a * ve_ref
        p["I_te10_cal"] = c * ih_ref
        p["V_perp_cal"] = (prof_e * w) @ e_perp
        p["I_perp_cal"] = ((prof_h * w) @ h_perp) * np.exp(+1j * omega * 0.5 * dt)
        p["P10_cal"] = 0.5 * np.real(p["V_te10_cal"] * np.conj(p["I_te10_cal"]))
        p["D_pred_cal"] = p["P"] / p["P10_cal"] - 1.0
        p["_e_perp"] = e_perp
        p["_h_perp"] = h_perp
        p["_e_perp_norm"] = np.sqrt(
            np.sum(np.abs(e_perp) ** 2 * w[:, None], axis=0))
        p["_e_norm"] = np.sqrt(np.sum(np.abs(fe) ** 2 * w[:, None], axis=0))


def stage_compose(num_periods: float = 40.0, control: dict | None = None) -> dict:
    """Project the recorded aperture field at each plane onto the discrete modes."""
    from tests import _waveguide_chain_battery_fixture as F

    freqs = np.asarray(F.FREQS, float)
    omega = 2 * np.pi * freqs
    out: dict = {}
    for rung, dx in zip(RUNGS, F.DX_LADDER):
        setup = _capture_setup("thru", dx, num_periods)
        cfgs = setup["cfgs"]
        cfg0 = cfgs[0]
        dt = float(cfg0.dt)
        dA = np.asarray(cfg0.aperture_dA, float)
        u_w = np.asarray(cfg0.u_widths, float)
        v_w = np.asarray(cfg0.v_widths, float)
        a, b = float(cfg0.a), float(cfg0.b)
        h_offset = tuple(float(x) for x in cfg0.h_offset)
        basis = build_basis(a, b, u_w, v_w, dA, h_offset)
        keys = list(basis)
        w = np.concatenate([dA.ravel(), dA.ravel()])
        De = np.array([np.concatenate([basis[k]["ey"].ravel(),
                                       basis[k]["ez"].ravel()]) for k in keys])
        Dh = np.array([np.concatenate([basis[k]["hy"].ravel(),
                                       basis[k]["hz"].ravel()]) for k in keys])
        We = De * np.sqrt(w)[None, :]
        Wh = Dh * np.sqrt(w)[None, :]
        pinv_e = np.linalg.pinv(We.T, rcond=1e-10)
        pinv_h = np.linalg.pinv(Wh.T, rcond=1e-10)

        # the extractor's own weights, from the port's stored profiles
        ey10 = np.asarray(cfg0.ey_profile, float)
        ez10 = np.asarray(cfg0.ez_profile, float)
        hy10 = np.asarray(cfg0.hy_profile, float)
        hz10 = np.asarray(cfg0.hz_profile, float)
        ge = np.array([float(np.sum((basis[k]["ey"] * ey10
                                     + basis[k]["ez"] * ez10) * dA)) for k in keys])
        gh = np.array([float(np.sum((basis[k]["hy"] * hy10
                                     + basis[k]["hz"] * hz10) * dA)) for k in keys])
        i10 = keys.index(("TE", 1, 0))

        finals = _aperture_capture_run(setup)
        n_valid = int(finals[0].n_steps_recorded)
        records = {
            "drive_ref": (finals[0].v_ref_t, finals[0].i_ref_t,
                          float(finals[0].reference_x_m)),
            "drive_probe": (finals[0].v_probe_t, finals[0].i_probe_t,
                            float(finals[0].probe_x_m)),
            "recv_probe": (finals[1].v_probe_t, finals[1].i_probe_t,
                           float(finals[1].probe_x_m)),
            "recv_ref": (finals[1].v_ref_t, finals[1].i_ref_t,
                         float(finals[1].reference_x_m)),
        }

        planes: dict = {}
        for name in PLANES:
            e_t, h_t, x_m = records[name]
            e_spec = _rect_dft_field(np.asarray(e_t), freqs, dt, n_valid)
            h_spec = _rect_dft_field(np.asarray(h_t), freqs, dt, n_valid)
            # the extractor's own two integrals, from the captured slice
            V = (np.sum(e_spec[:, 0] * ey10[None] * dA[None], axis=(1, 2))
                 + np.sum(e_spec[:, 1] * ez10[None] * dA[None], axis=(1, 2)))
            I = (np.sum(h_spec[:, 0] * hy10[None] * dA[None], axis=(1, 2))
                 + np.sum(h_spec[:, 1] * hz10[None] * dA[None], axis=(1, 2)))
            I = I * np.exp(+1j * omega * 0.5 * dt)
            A, res_e = _project(e_spec, De, w, pinv_e)
            B, res_h = _project(h_spec, Dh, w, pinv_h)
            V_k = A * ge[:, None]
            I_k = B * gh[:, None] * np.exp(+1j * omega * 0.5 * dt)[None, :]
            V10, I10 = V_k[i10], I_k[i10]
            P = 0.5 * np.real(V * np.conj(I))
            P10 = 0.5 * np.real(V10 * np.conj(I10))
            # Registration witness, no basis involved: the index-flip symmetry of
            # the recorded slice against a template that IS index-flip symmetric.
            ez_bin = e_spec[-1, 1]
            hy_bin = h_spec[-1, 0]
            planes[name] = {
                "x_m": x_m,
                "distance_from_driven_source_m":
                    abs(x_m - float(cfg0.source_x_m)),
                "ez_index_flip_antisym_fraction_bin16": float(
                    np.linalg.norm(0.5 * (ez_bin - ez_bin[::-1]))
                    / max(float(np.linalg.norm(0.5 * (ez_bin + ez_bin[::-1]))), 1e-300)),
                "hy_index_flip_antisym_fraction_bin16": float(
                    np.linalg.norm(0.5 * (hy_bin - hy_bin[::-1]))
                    / max(float(np.linalg.norm(0.5 * (hy_bin + hy_bin[::-1]))), 1e-300)),
                "ez_profile_u_normalised_bin16": (
                    np.abs(ez_bin[:, 0]) / max(float(np.abs(ez_bin[:, 0]).max()),
                                               1e-300)).tolist(),
                "V": V, "I": I, "P": P, "P10": P10,
                "_e_spec": e_spec, "_h_spec": h_spec,
                "A": A, "B": B, "V_k": V_k, "I_k": I_k,
                "residual_fraction_e": res_e, "residual_fraction_h": res_h,
                "reconstruction_rel_err_V": float(np.max(
                    np.abs(V - V_k.sum(axis=0)) / np.maximum(np.abs(V), 1e-300))),
                "reconstruction_rel_err_I": float(np.max(
                    np.abs(I - I_k.sum(axis=0)) / np.maximum(np.abs(I), 1e-300))),
            }
        _calibrated_split(planes, ey10, ez10, hy10, hz10, dA, omega, dt)
        # Robustness of the calibration reference: repeat the split on the OTHER
        # far plane (71.12 mm from the driven source, also never used as its own
        # reference) and keep the driven plane's answer from each.
        alt = {n: dict(p) for n, p in planes.items()}
        _calibrated_split(alt, ey10, ez10, hy10, hz10, dA, omega, dt,
                          alt_ref="recv_probe")
        planes["drive_ref"]["D_pred_cal_alt_ref"] = alt["drive_ref"]["D_pred_cal"]
        planes["drive_ref"]["perp_fraction_e_alt_ref"] = (
            alt["drive_ref"]["perp_fraction_e"])
        planes["recv_ref"]["D_pred_cal_alt_ref"] = alt["recv_ref"]["D_pred_cal"]
        planes["recv_ref"]["perp_fraction_e_alt_ref"] = (
            alt["recv_ref"]["perp_fraction_e"])
        out[rung] = {
            "setup": setup, "basis": basis, "keys": keys, "i10": i10,
            "ge": ge, "gh": gh, "dA": dA, "freqs": freqs, "dt": dt,
            "planes": planes, "n_valid": n_valid,
            "warnings_verbatim": setup["warnings"],
            "grid_shape": [int(v) for v in np.asarray(
                getattr(setup["grid"], "shape", (0, 0, 0)))],
        }

        # capture witness: the reconstructed modal voltage against the
        # unpatched run's own recorded series (same solve, different recording)
        if control is not None and rung in control:
            ref_v = control[rung]["_v_ref_t"][:control[rung]["_n_steps_recorded"]]
            e_t = np.asarray(records["drive_ref"][0], float)[:len(ref_v)]
            recon_v = np.sum((e_t[:, 0] * ey10[None] + e_t[:, 1] * ez10[None])
                             * dA[None], axis=(1, 2))
            scale = max(float(np.max(np.abs(ref_v))), 1e-300)
            out[rung]["capture_witness_rel_err_v_ref_t"] = float(
                np.max(np.abs(recon_v - ref_v)) / scale)
            ref_i = control[rung]["_i_ref_t"][:control[rung]["_n_steps_recorded"]]
            h_t = np.asarray(records["drive_ref"][1], float)[:len(ref_i)]
            recon_i = np.sum((h_t[:, 0] * hy10[None] + h_t[:, 1] * hz10[None])
                             * dA[None], axis=(1, 2))
            out[rung]["capture_witness_rel_err_i_ref_t"] = float(
                np.max(np.abs(recon_i - ref_i))
                / max(float(np.max(np.abs(ref_i))), 1e-300))
    return out


# --------------------------------------------------------------------------
# stage "falsifier"
# --------------------------------------------------------------------------
def stage_falsifier(compose: dict, num_periods: float = 40.0) -> dict:
    """Pre-declaration section 6: the record plane moved off 3 cells."""
    from tests import _waveguide_chain_battery_fixture as F

    out: dict = {}
    saved = (F.D_REF_M, F.REF_LEFT_DEFAULT_M, F.REF_RIGHT_DEFAULT_M)
    try:
        for cells in (3, 4, 11):
            F.D_REF_M = cells * F.DX_COARSE
            F.REF_LEFT_DEFAULT_M = F.PORT_LEFT_X_M + F.D_REF_M
            F.REF_RIGHT_DEFAULT_M = F.PORT_RIGHT_X_M - F.D_REF_M
            dx = F.DX_LADDER[0]
            sim = F.build_simulation("thru", dx)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = sim.compute_waveguide_s_matrix(num_periods=num_periods,
                                                     normalize=False)
            texts = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
            s = np.asarray(res.s_params)
            col0 = np.abs(s[0, 0]) ** 2 + np.abs(s[1, 0]) ** 2
            entry = {
                "ref_cells": cells,
                "distance_from_source_m": float(F.D_REF_M),
                "reference_planes_m": [float(F.REF_LEFT_DEFAULT_M),
                                       float(F.REF_RIGHT_DEFAULT_M)],
                "column_power_minus_1_col0": (col0 - 1.0).tolist(),
                "max_positive_column_power_minus_1": float(np.max(col0 - 1.0)),
                "band_mean_s11_mag": float(np.mean(np.abs(s[0, 0]))),
                "settling_db": [float(x) for x in np.asarray(res.settling_db)],
                "warnings_verbatim": texts,
            }
            # the contamination term at the same plane, re-measured
            setup = _capture_setup("thru", dx, num_periods)
            comp = _compose_one(setup, compose["coarse"])
            entry.update(comp)
            out[f"ref{cells}cells"] = entry
    finally:
        F.D_REF_M, F.REF_LEFT_DEFAULT_M, F.REF_RIGHT_DEFAULT_M = saved
    return out


def _compose_one(setup: dict, template: dict) -> dict:
    """Drive-plane D_pred / D_meas for one already-built setup (falsifier reuse)."""
    freqs = template["freqs"]
    omega = 2 * np.pi * freqs
    cfg0 = setup["cfgs"][0]
    dt = float(cfg0.dt)
    dA = np.asarray(cfg0.aperture_dA, float)
    basis = build_basis(float(cfg0.a), float(cfg0.b),
                        np.asarray(cfg0.u_widths, float),
                        np.asarray(cfg0.v_widths, float), dA,
                        tuple(float(x) for x in cfg0.h_offset))
    keys = list(basis)
    w = np.concatenate([dA.ravel(), dA.ravel()])
    De = np.array([np.concatenate([basis[k]["ey"].ravel(),
                                   basis[k]["ez"].ravel()]) for k in keys])
    Dh = np.array([np.concatenate([basis[k]["hy"].ravel(),
                                   basis[k]["hz"].ravel()]) for k in keys])
    We, Wh = De * np.sqrt(w)[None, :], Dh * np.sqrt(w)[None, :]
    pinv_e = np.linalg.pinv(We.T, rcond=1e-10)
    pinv_h = np.linalg.pinv(Wh.T, rcond=1e-10)
    ey10 = np.asarray(cfg0.ey_profile, float)
    ez10 = np.asarray(cfg0.ez_profile, float)
    hy10 = np.asarray(cfg0.hy_profile, float)
    hz10 = np.asarray(cfg0.hz_profile, float)
    ge = np.array([float(np.sum((basis[k]["ey"] * ey10
                                 + basis[k]["ez"] * ez10) * dA)) for k in keys])
    gh = np.array([float(np.sum((basis[k]["hy"] * hy10
                                 + basis[k]["hz"] * hz10) * dA)) for k in keys])
    i10 = keys.index(("TE", 1, 0))
    finals = _aperture_capture_run(setup)
    n_valid = int(finals[0].n_steps_recorded)
    planes = {}
    for name, (e_t, h_t) in (("drive_ref", (finals[0].v_ref_t, finals[0].i_ref_t)),
                             ("recv_ref", (finals[1].v_ref_t, finals[1].i_ref_t))):
        e_spec = _rect_dft_field(np.asarray(e_t), freqs, dt, n_valid)
        h_spec = _rect_dft_field(np.asarray(h_t), freqs, dt, n_valid)
        V = (np.sum(e_spec[:, 0] * ey10[None] * dA[None], axis=(1, 2))
             + np.sum(e_spec[:, 1] * ez10[None] * dA[None], axis=(1, 2)))
        I = (np.sum(h_spec[:, 0] * hy10[None] * dA[None], axis=(1, 2))
             + np.sum(h_spec[:, 1] * hz10[None] * dA[None], axis=(1, 2)))
        I = I * np.exp(+1j * omega * 0.5 * dt)
        A, _ = _project(e_spec, De, w, pinv_e)
        B, _ = _project(h_spec, Dh, w, pinv_h)
        planes[name] = {
            "_e_spec": e_spec, "_h_spec": h_spec, "A": A, "B": B,
            "V": V, "I": I, "P": 0.5 * np.real(V * np.conj(I)),
            "P10": 0.5 * np.real(A[i10] * ge[i10]
                                 * np.conj(B[i10] * gh[i10]
                                           * np.exp(+1j * omega * 0.5 * dt))),
        }
    _calibrated_split(planes, ey10, ez10, hy10, hz10, dA, omega, dt)
    drive, recv = planes["drive_ref"], planes["recv_ref"]
    d_meas = drive["P"] / recv["P"] - 1.0
    d_pred = drive["P"] / drive["P10"] - 1.0
    d_pred_cal = drive["D_pred_cal"]
    d_te10_cal = drive["P10_cal"] / recv["P"] - 1.0
    w2 = np.concatenate([dA.ravel(), dA.ravel()])
    named = {}
    for k in NAMED:
        if k in keys:
            j = keys.index(k)
            ov = float(np.abs(((De[j] * w2) @ drive["_e_perp"])[-1]))
            named[label(k)] = ov / max(float(np.abs(drive["A"][i10][-1])), 1e-300)
    return {
        "D_pred_per_bin": d_pred.tolist(),
        "D_meas_per_bin": d_meas.tolist(),
        "D_pred_cal_per_bin": d_pred_cal.tolist(),
        "D_te10_cal_per_bin": d_te10_cal.tolist(),
        "D_pred_bin16": float(d_pred[-1]),
        "D_meas_bin16": float(d_meas[-1]),
        "D_pred_cal_bin16": float(d_pred_cal[-1]),
        "D_te10_cal_bin16": float(d_te10_cal[-1]),
        "perp_fraction_e_bin16": float(drive["perp_fraction_e"][-1]),
        "perp_fraction_h_bin16": float(drive["perp_fraction_h"][-1]),
        "named_mode_perp_overlap_over_te10_bin16": named,
    }


# --------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------
def _serialise_rung(rung: str, data: dict, control_rung: dict) -> dict:
    keys, i10 = data["keys"], data["i10"]
    ge, gh, dA = data["ge"], data["gh"], data["dA"]
    freqs = data["freqs"]
    k = 2 * np.pi * freqs / C0
    basis = data["basis"]
    planes = data["planes"]
    drive = planes["drive_ref"]
    recv = planes["recv_ref"]

    d_pred = drive["P"] / drive["P10"] - 1.0
    d_meas = drive["P"] / recv["P"] - 1.0
    ref_diff = drive["P10"] / recv["P"] - 1.0
    # The registration baseline. The port's transverse template is cell-centred
    # while the Ez / Hy it integrates are node-registered in u (measured: the
    # far-plane Ez slice is sin(pi*j/nu), zero at index 0, against a template
    # sin(pi*(j+1/2)/nu)). A clean single-TE10 plane therefore projects onto a
    # FIXED spectrum of higher coefficients, present at every plane. It cancels
    # in D_meas, which is a ratio of two planes, but not in D_pred as
    # pre-declared, so the differential form is carried beside it.
    d_pred_recv = recv["P"] / recv["P10"] - 1.0
    d_pred_diff = (1.0 + d_pred) / (1.0 + d_pred_recv) - 1.0
    d_te10_only = drive["P10"] / recv["P10"] - 1.0
    # The calibrated split: TE10 reference taken from the clean far plane.
    d_pred_cal = drive["D_pred_cal"]
    d_te10_cal = drive["P10_cal"] / recv["P"] - 1.0

    # Name the orthogonal content by direct overlap onto the mode family (no
    # inversion: the residual is small and the overlaps are what the extractor
    # weights act on).  The leak into I is exactly I_perp; the per-mode sum
    # below is checked against it.
    w = np.concatenate([dA.ravel(), dA.ravel()])
    e_flat = np.array([np.concatenate([basis[kk]["ey"].ravel(),
                                       basis[kk]["ez"].ravel()]) for kk in keys])
    h_flat = np.array([np.concatenate([basis[kk]["hy"].ravel(),
                                       basis[kk]["hz"].ravel()]) for kk in keys])
    perp_overlap = {}
    for name in PLANES:
        p = planes[name]
        perp_overlap[name] = (
            (e_flat * w) @ p["_e_perp"],      # (K, n_f)
            (h_flat * w) @ p["_h_perp"],
        )
    _half_step = np.exp(+1j * 2 * np.pi * freqs * 0.5 * data["dt"])
    i_perp_from_modes = (
        (perp_overlap["drive_ref"][1] * gh[:, None]).sum(axis=0) * _half_step)
    v_perp_from_modes = (perp_overlap["drive_ref"][0] * ge[:, None]).sum(axis=0)

    # per-mode linear attribution of P - P10 at the driven plane
    V10, I10 = drive["V_k"][i10], drive["I_k"][i10]
    attribution = {}
    for j, key in enumerate(keys):
        if j == i10:
            continue
        contrib = (0.5 * np.real(V10 * np.conj(drive["I_k"][j]))
                   + 0.5 * np.real(drive["V_k"][j] * np.conj(I10)))
        if np.max(np.abs(contrib)) <= 1e-12 * np.max(np.abs(drive["P10"])):
            continue
        attribution[label(key)] = contrib

    # Coefficient census: the named candidates, TE10, and every mode carrying
    # more than 1e-5 of the TE10 amplitude in the ORTHOGONAL (non-TE10) field at
    # the driven plane. The floor is applied to the orthogonal part, not to the
    # raw least-squares coefficient: the cell-centred family is
    # near-rank-deficient against a node-registered field (section 3 of the
    # results note), so the raw coefficients put hundreds of modes above any
    # floor and none of them is content.
    a10_drive = np.abs(drive["A"][i10])
    perp_rel = np.abs(perp_overlap["drive_ref"][0]) / np.maximum(a10_drive, 1e-300)
    reported = set(NAMED) | {("TE", 1, 0)}
    ranked = sorted(range(len(keys)), key=lambda jj: -float(np.max(perp_rel[jj])))
    for j in ranked:
        if float(np.max(perp_rel[j])) <= 1e-5 or len(reported) >= 40:
            break
        reported.add(keys[j])
    modes = {}
    for key in sorted(reported, key=lambda kk: basis[kk]["kc"] if kk in basis else 0):
        if key not in keys:
            continue
        j = keys.index(key)
        kc = basis[key]["kc"]
        alpha = np.sqrt(np.maximum(kc ** 2 - k ** 2, 0.0))
        entry = {
            "kc_discrete_per_m": kc,
            "fc_discrete_hz": kc * C0 / (2 * np.pi),
            "alpha_per_m": alpha.tolist(),
            "overlap_e_with_te10": float(ge[j]),
            "overlap_h_with_te10": float(gh[j]),
            "predicted_first_hop_ratio": float(np.exp(alpha[-1] * 0.01778)),
        }
        # Ratios to TE10, per plane, and the registration baseline subtracted.
        # `r_k` is the same ratio at the receiving reference plane, 88.9 mm from
        # the driven source, where every evanescent mode is below 1e-4 by its own
        # alpha: whatever spectrum the projection assigns there is the
        # registration signature of a clean TE10 wave, not content.
        ratios = {name: planes[name]["A"][j] / planes[name]["A"][i10]
                  for name in PLANES}
        r_k = ratios["recv_ref"]
        for name in PLANES:
            p = planes[name]
            entry[f"A_mag_{name}"] = np.abs(p["A"][j]).tolist()
            entry[f"A_phase_deg_{name}"] = np.degrees(
                np.angle(p["A"][j])).tolist()
            entry[f"B_mag_{name}"] = np.abs(p["B"][j]).tolist()
            entry[f"A_over_te10_mag_{name}"] = np.abs(ratios[name]).tolist()
            entry[f"near_field_excess_mag_{name}"] = np.abs(
                ratios[name] - r_k).tolist()
        entry["baseline_spread_recv_probe_vs_recv_ref"] = np.abs(
            ratios["recv_probe"] - r_k).tolist()
        # The calibrated census: direct overlap of the orthogonal field onto
        # this mode, relative to the plane's TE10 amplitude.  No inversion.
        for name in PLANES:
            perp_e, perp_h = perp_overlap[name]
            entry[f"perp_overlap_e_{name}"] = np.abs(
                perp_e[j] / planes[name]["A"][i10]).tolist()
            entry[f"perp_overlap_h_{name}"] = np.abs(
                perp_h[j] / planes[name]["B"][i10]).tolist()
        pe_d = np.abs(perp_overlap["drive_ref"][0][j])
        pe_p = np.abs(perp_overlap["drive_probe"][0][j])
        # The mirror control has to use recv_PROBE: recv_ref is the calibration
        # reference, so its orthogonal part is zero by construction and a ratio
        # against it would be vacuous.
        pe_r = np.abs(perp_overlap["recv_probe"][0][j])
        entry["perp_first_hop_ratio"] = (
            pe_d / np.maximum(pe_p, 1e-300)).tolist()
        entry["perp_first_hop_ratio_bin16"] = float(pe_d[-1] / max(pe_p[-1], 1e-300))
        entry["perp_mirror_control_bin16"] = float(pe_r[-1] / max(pe_d[-1], 1e-300))
        entry["perp_share_of_orthogonal_field_bin16"] = float(
            pe_d[-1] / max(float(drive["_e_perp_norm"][-1]), 1e-300))
        entry["i_leak_over_i_abs"] = np.abs(
            perp_overlap["drive_ref"][1][j] * gh[j] * _half_step
            / np.maximum(np.abs(drive["I"]), 1e-300)).tolist()
        a_drive = np.abs(ratios["drive_ref"] - r_k)
        a_probe = np.abs(ratios["drive_probe"] - r_k)
        entry["measured_first_hop_ratio"] = (
            a_drive / np.maximum(a_probe, 1e-300)).tolist()
        entry["measured_first_hop_ratio_bin16"] = float(
            a_drive[-1] / max(a_probe[-1], 1e-300))
        entry["near_field_excess_bin16"] = float(a_drive[-1])
        entry["mirror_control_recv_over_drive"] = float(
            np.abs(ratios["recv_probe"][-1] - r_k[-1]) / max(a_drive[-1], 1e-300))
        if label(key) in attribution:
            entry["contamination_share_bin16"] = float(
                attribution[label(key)][-1] / (drive["P"][-1] - drive["P10"][-1])
                if abs(drive["P"][-1] - drive["P10"][-1]) > 0 else 0.0)
            entry["contamination_per_bin_over_P10"] = (
                attribution[label(key)] / drive["P10"]).tolist()
        modes[label(key)] = entry

    # Mechanism check, derivation only (no field): what does a NODE-registered
    # TE_m0 read as through the port's CELL-CENTRED templates?  The measured
    # far-plane slice is sin(pi*m*j/nu) on j = 0..nu-1, so that is the shape a
    # mode of this guide actually presents. The cell-centred parity argument
    # that makes <e_20, e_10> vanish does not apply to it.
    import rfx.sources._waveguide_modes as wm

    nu, nv = dA.shape
    jj = np.arange(nu)
    ez10_t = np.asarray(data["setup"]["cfgs"][0].ez_profile, float)
    hy10_t = np.asarray(data["setup"]["cfgs"][0].hy_profile, float)
    h_off = tuple(float(x) for x in data["setup"]["cfgs"][0].h_offset)
    registration = {}
    base_e = base_h = None
    for m in range(1, min(nu, 6)):
        node = np.repeat(np.sin(m * np.pi * jj / nu)[:, None], nv, axis=1)
        node = node / np.sqrt(np.sum(node ** 2 * dA))
        o_e = float(np.sum(node * ez10_t * dA))
        o_h = float(np.sum(wm._shift_profile_to_dual(node, h_off) * hy10_t * dA))
        if m == 1:
            base_e, base_h = o_e, o_h
        registration[f"TE{m}0_node"] = {
            "overlap_with_ez10_template": o_e,
            "overlap_with_hy10_template": o_h,
            "read_as_te10_fraction_e": o_e / base_e,
            "read_as_te10_fraction_h": o_h / base_h,
        }

    # TE10 plane independence, propagation phase removed
    beta = None
    cfg0 = data["setup"]["cfgs"][0]
    fc = float(cfg0.f_cutoff)
    dt = data["dt"]
    s_t = np.sin(2 * np.pi * freqs * 0.5 * dt) / (C0 * 0.5 * dt)
    kc10 = 2 * np.pi * fc / C0
    beta = (2.0 / data["setup"]["cfgs"][0].dx) * np.arcsin(np.clip(
        0.5 * data["setup"]["cfgs"][0].dx
        * np.sqrt(np.maximum(s_t ** 2 - kc10 ** 2, 0.0)), -1.0, 1.0))
    te10_plane = {}
    x0 = planes["recv_ref"]["x_m"]
    for name in PLANES:
        p = planes[name]
        phase = np.exp(+1j * beta * (p["x_m"] - x0))
        te10_plane[name] = {
            "x_m": p["x_m"],
            "A10_mag": np.abs(p["A"][i10]).tolist(),
            "A10_dephased_over_recv_ref": (
                np.abs(p["A"][i10] * phase / planes["recv_ref"]["A"][i10]) - 1.0
            ).tolist(),
            "B10_mag": np.abs(p["B"][i10]).tolist(),
            "P_over_recv_ref_minus_1": (p["P"] / recv["P"] - 1.0).tolist(),
            "P10_over_recv_ref_minus_1": (p["P10"] / recv["P"] - 1.0).tolist(),
            "residual_fraction_e": p["residual_fraction_e"].tolist(),
            "residual_fraction_h": p["residual_fraction_h"].tolist(),
            "reconstruction_rel_err_V": p["reconstruction_rel_err_V"],
            "reconstruction_rel_err_I": p["reconstruction_rel_err_I"],
            "ez_index_flip_antisym_fraction_bin16":
                p["ez_index_flip_antisym_fraction_bin16"],
            "hy_index_flip_antisym_fraction_bin16":
                p["hy_index_flip_antisym_fraction_bin16"],
            "ez_profile_u_normalised_bin16": p["ez_profile_u_normalised_bin16"],
            "D_pred_at_this_plane": (p["P"] / p["P10"] - 1.0).tolist(),
            "D_pred_cal_at_this_plane": p["D_pred_cal"].tolist(),
            "perp_fraction_e": p["perp_fraction_e"].tolist(),
            "perp_fraction_h": p["perp_fraction_h"].tolist(),
            "P10_cal_over_recv_ref_minus_1": (p["P10_cal"] / recv["P"] - 1.0).tolist(),
        }

    return {
        "dx_m": float(cfg0.dx),
        "dt_s": dt,
        "n_steps": int(data["setup"]["n_steps"]),
        "n_steps_recorded": int(data["n_valid"]),
        "grid_shape": data["grid_shape"],
        "nu": int(dA.shape[0]), "nv": int(dA.shape[1]),
        "n_basis_modes": len(keys),
        "port_f_cutoff_hz": fc,
        "source_x_m": float(cfg0.source_x_m),
        "capture_witness_rel_err_v_ref_t": data.get(
            "capture_witness_rel_err_v_ref_t"),
        "capture_witness_rel_err_i_ref_t": data.get(
            "capture_witness_rel_err_i_ref_t"),
        "D_meas_per_bin": d_meas.tolist(),
        "D_pred_per_bin": d_pred.tolist(),
        "reference_difference_per_bin": ref_diff.tolist(),
        "D_pred_recv_ref_per_bin": d_pred_recv.tolist(),
        "D_pred_differential_per_bin": d_pred_diff.tolist(),
        "D_te10_only_per_bin": d_te10_only.tolist(),
        "D_meas_bin16": float(d_meas[-1]),
        "D_pred_bin16": float(d_pred[-1]),
        "reference_difference_bin16": float(ref_diff[-1]),
        "D_pred_recv_ref_bin16": float(d_pred_recv[-1]),
        "D_pred_differential_bin16": float(d_pred_diff[-1]),
        "D_te10_only_bin16": float(d_te10_only[-1]),
        "D_pred_cal_per_bin": d_pred_cal.tolist(),
        "D_te10_cal_per_bin": d_te10_cal.tolist(),
        "D_pred_cal_bin16": float(d_pred_cal[-1]),
        "D_te10_cal_bin16": float(d_te10_cal[-1]),
        "D_pred_cal_alt_ref_bin16": float(drive["D_pred_cal_alt_ref"][-1]),
        "perp_fraction_e_alt_ref_bin16": float(
            drive["perp_fraction_e_alt_ref"][-1]),
        "far_plane_shape_disagreement_bin16": float(
            recv["perp_fraction_e_alt_ref"][-1]),
        "leg1_ratio_bin16": float(d_pred[-1] / d_meas[-1])
        if abs(d_meas[-1]) > 0 else float("nan"),
        "leg1_ratio_differential_bin16": float(d_pred_diff[-1] / d_meas[-1])
        if abs(d_meas[-1]) > 0 else float("nan"),
        "leg1_ratio_cal_bin16": float(d_pred_cal[-1] / d_meas[-1])
        if abs(d_meas[-1]) > 0 else float("nan"),
        "leg1_ratio_cal_per_bin": (d_pred_cal / d_meas).tolist(),
        "identity_residual_bin16": float(
            (1.0 + d_pred_diff[-1]) * (1.0 + d_te10_only[-1]) - (1.0 + d_meas[-1])),
        "identity_residual_cal_bin16": float(
            (1.0 + d_pred_cal[-1]) * (1.0 + d_te10_cal[-1]) - (1.0 + d_meas[-1])),
        "V_perp_over_V_bin16": float(
            abs(drive["V_perp_cal"][-1]) / abs(drive["V"][-1])),
        "I_perp_over_I_bin16": float(
            abs(drive["I_perp_cal"][-1]) / abs(drive["I"][-1])),
        "I_perp_mode_sum_closure_bin16": float(
            abs(i_perp_from_modes[-1] - drive["I_perp_cal"][-1])
            / max(abs(drive["I_perp_cal"][-1]), 1e-300)),
        "V_perp_mode_sum_closure_bin16": float(
            abs(v_perp_from_modes[-1] - drive["V_perp_cal"][-1])
            / max(abs(drive["V_perp_cal"][-1]), 1e-300)),
        "te10_by_plane": te10_plane,
        "node_registration_readthrough": registration,
        "modes": modes,
        "warnings_verbatim": data["warnings_verbatim"],
        "control_settling_db": control_rung["settling_db"],
        "control_warnings_verbatim": control_rung["warnings_verbatim"],
        "reproduce_gate_rel_diff": control_rung["reproduce_gate_rel_diff"],
        "max_abs_per_bin_diff_vs_frozen":
            control_rung["max_abs_per_bin_diff_vs_frozen"],
    }


def make_figure(compose: dict, path: Path, falsifier: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharex=True)
    for col, rung in enumerate(RUNGS):
        data = compose[rung]
        keys, i10 = data["keys"], data["i10"]
        freqs = data["freqs"] / 1e9
        drive = data["planes"]["drive_ref"]
        recv = data["planes"]["recv_ref"]
        dA = data["dA"]
        w = np.concatenate([dA.ravel(), dA.ravel()])
        basis = data["basis"]
        ax = axes[0, col]
        a10 = np.abs(drive["A"][i10])
        for key in NAMED:
            if key not in keys:
                continue
            ek = np.concatenate([basis[key]["ey"].ravel(),
                                 basis[key]["ez"].ravel()])
            ax.semilogy(freqs, np.abs((ek * w) @ drive["_e_perp"]) / a10,
                        label=label(key))
        ax.semilogy(freqs, drive["perp_fraction_e"], "k:", label="all non-TE10")
        ax.set_title(f"{rung}: near-field content / TE10, driven ref plane")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("modal content, relative to TE10")
        ax = axes[1, col]
        d_meas = drive["P"] / recv["P"] - 1.0
        d_pred = drive["D_pred_cal"]
        ref = drive["P10_cal"] / recv["P"] - 1.0
        ax.plot(freqs, d_meas, "o-", label="measured offset")
        ax.plot(freqs, d_pred, "s--", label="higher-mode part")
        ax.plot(freqs, ref, "^:", label="TE10-only plane difference")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xlabel("frequency (GHz)")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("modal power offset")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)

    if falsifier is None:
        return
    # The decay figure: TE20's measured amplitude against the alpha the
    # pre-declaration tabulated, at every plane position this run produced.
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    data = compose["coarse"]
    keys, i10 = data["keys"], data["i10"]
    basis, dA = data["basis"], data["dA"]
    w = np.concatenate([dA.ravel(), dA.ravel()])
    e20 = np.concatenate([basis[("TE", 2, 0)]["ey"].ravel(),
                          basis[("TE", 2, 0)]["ez"].ravel()])
    xs, ys = [], []
    for name in PLANES:
        if name == "recv_ref":
            continue        # the calibration reference: zero by construction
        p = data["planes"][name]
        xs.append(p["distance_from_driven_source_m"] * 1e3)
        ys.append(abs(((e20 * w) @ p["_e_perp"])[-1]) / abs(p["A"][i10][-1]))
    ax.semilogy(xs[:2], ys[:2], "o", ms=8, label="TE20 content, recorded planes")
    ax.semilogy(xs[2:], ys[2:], "o", ms=8, mfc="none",
                label="floor of this measurement (71.1 mm plane)")
    fx, fy = [], []
    for k in sorted(falsifier, key=lambda kk: falsifier[kk]["distance_from_source_m"]):
        v = falsifier[k]
        fx.append(v["distance_from_source_m"] * 1e3)
        fy.append(v["named_mode_perp_overlap_over_te10_bin16"]["TE20"])
    ax.semilogy(fx, fy, "s", ms=7, mfc="none",
                label="TE20 content, moved record plane")
    kc = basis[("TE", 2, 0)]["kc"]
    k116 = 2 * np.pi * data["freqs"][-1] / C0
    alpha = np.sqrt(max(kc ** 2 - k116 ** 2, 0.0))
    dd = np.linspace(5.0, 90.0, 200)
    ax.semilogy(dd, ys[0] * np.exp(-alpha * (dd - xs[0]) * 1e-3), "k--", lw=1,
                label=f"exp(-{alpha:.1f}/m . d), pre-declared alpha")
    ax.set_ylim(1e-5, 1e-1)
    ax.set_xlim(0.0, 80.0)
    ax.set_xlabel("distance from the driven source plane (mm)")
    ax.set_ylabel("TE20 content / TE10, bin 16 (11.6 GHz)")
    ax.set_title("coarse rung: the near field at the driven plane is TE20")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    decay_path = path.with_name(path.stem + "_decay" + path.suffix)
    fig.savefig(decay_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--stages", default="control,compose,falsifier")
    args = parser.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    report: dict = {
        "schema": "rfx.waveguide_driven_plane_near_field_composition",
        "schema_version": 1,
        "issue": 873,
        "hypothesis": "source near-field modal composition at the driven "
                      "reference plane",
        "attempts_on_this_hypothesis": 1,
        "prior_closed_lanes": ["PR #880 (reflection factors)",
                               "PR #1081 (port-local V/I transfer)"],
        "predeclaration": PREDECLARATION,
        "sibling_artifacts_not_touched": [
            "tests/fixtures/waveguide_false_lane_column_power/suspects.json",
            "tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json",
        ],
        "provenance": {
            "commit": _git_commit(),
            "source_fixture":
                "tests/fixtures/waveguide_chain_battery/fixture_v18_close.json",
            "source_fixture_sha256": _sha256(FIXTURE),
            "recapture_entry_point":
                "scripts/diagnostics/waveguide_driven_plane_near_field_composition.py",
            "precision": "float32 fields, float64 post-processing",
            "num_periods": 40.0,
            "stages": stages,
        },
    }

    control = stage_control() if "control" in stages else None
    compose = stage_compose(control=control) if "compose" in stages else None

    if control is not None:
        report["control"] = {
            r: {k: v for k, v in d.items() if not k.startswith("_")}
            for r, d in control.items()
        }
    if compose is not None:
        report["read"] = {
            "freqs_hz": np.asarray(compose["coarse"]["freqs"]).tolist(),
            "per_rung": {r: _serialise_rung(r, compose[r], control[r])
                         for r in RUNGS},
        }
    if "falsifier" in stages and compose is not None:
        report["falsifier"] = stage_falsifier(compose)
    if compose is not None and args.figure is not None:
        make_figure(compose, args.figure, report.get("falsifier"))
        report["figure"] = str(args.figure)
        if "falsifier" in report:
            report["figure_decay"] = str(
                args.figure.with_name(args.figure.stem + "_decay"
                                      + args.figure.suffix))

    import jax
    report["provenance"]["jax_version"] = jax.__version__
    report["provenance"]["jax_default_backend"] = jax.default_backend()
    report["provenance"]["numpy_version"] = np.__version__

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
