"""Issue #770 adjudication harness: off-diagonal receive channel vs external physics.

Pre-declaration (binding, committed BEFORE this file existed):
    docs/design_notes/issue770_offdiag_adjudication_predeclaration.md

Measurement-only (F-A6): spy capture + offline algebra; no shipped
extractor is edited here.  Every gate below is evaluated verbatim from
the pre-declaration; nothing here may widen a window.  A miss is
reported as a miss with the residual mechanism named.

Run (from THIS worktree):

  PYTHONPATH=$PWD JAX_PLATFORMS=cpu .venv/bin/python \
      validation/research/issue770_offdiag_adjudication.py          # main arm
  ... issue770_offdiag_adjudication.py --dc                         # F-A4 arm
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp

import rfx
from rfx import Simulation, Box
from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
from rfx.sources.sources import GaussianPulse

C0 = 299792458.0

# ---------------------------------------------------------------- FIX-T ----
# Canonical THRU battery geometry, verbatim constants
# (tests/test_lumped_twoport_vi_validation_battery.py).
DX = 0.5e-3
DOMAIN = (0.032, 0.020, 0.010)
NZ = int(round(DOMAIN[2] / DX))          # 20
FREQ_MAX = 10e9
CPML_LAYERS = 8
H = 1.0e-3
W = 5.0e-3
X1, X2 = 0.008, 0.024
L = X2 - X1
Y_MID = DOMAIN[1] / 2
N_STEPS = 4000
FREQS = np.linspace(3e9, 7e9, 9)
Z0 = 50.0
PINNED_CODES = ["pec_faces_finite_pec",
                "wire_port_dead_extent_cells",
                "wire_port_dead_extent_cells"]

# FIX-T-DC arm (committed DC-anchor constants).
DC_FREQS = np.array([0.5e9, 1.0e9])
DC_N_STEPS = 12000
DCA_BAND = (-0.25, +0.10)

# Pre-declared gates (section 4 of the pre-declaration).
A1_CEIL = 1.02
A2_WINDOW = (0.90, 1.02)
A3_ABS = 1.5e-2
A3_REL = 0.10
A5_ATOL = 1e-3
A6_WIRING = 1e-5

FAILS: list[str] = []


def gate(ok, label, detail):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {label}: {detail}")
    if not ok:
        FAILS.append(label)


def build_fix_t(*, nu: bool, drive: int | None, pulse=None):
    """FIX-T; ``drive`` selects the excited port on the NU lane
    (None = both excite=True, the uniform battery fixture verbatim)."""
    from rfx.boundaries.spec import Boundary, BoundarySpec
    kw = {"dz_profile": np.full(NZ, DX)} if nu else {}
    sim = Simulation(freq_max=FREQ_MAX, domain=DOMAIN, dx=DX,
                     boundary=BoundarySpec(x="cpml", y="cpml",
                                           z=Boundary(lo="pec", hi="cpml")),
                     cpml_layers=CPML_LAYERS, **kw)
    sim.add(Box((X1 - DX, Y_MID - W / 2, H),
                (X2 + DX, Y_MID + W / 2, H + DX)), material="pec")
    if pulse is None:
        pulse = GaussianPulse(f0=5e9, bandwidth=0.8)
    for idx, (x, d) in enumerate(((X1, "-x"), (X2, "+x"))):
        exc = (drive is None) or (idx == drive)
        sim.add_port(position=(x, Y_MID, 0.0), component="ez",
                     impedance=Z0, extent=H, excite=exc,
                     waveform=(pulse if exc else None), direction=d)
    return sim


def preflight_verbatim(sim, label):
    report = sim.preflight()
    issues = [str(i) for i in report]
    for msg in issues:
        print(f"[{label}] preflight (verbatim): {msg}")
    codes = sorted(getattr(i, "code", None) for i in report)
    return codes, issues


def run_nu(sim, n_steps, freqs):
    """Run on the NU lane; capture wire specs + result (764-harness spy)."""
    import rfx.runners.nonuniform as nur
    cap = {}
    orig = nur.run_nonuniform

    def spy(grid, materials, n_steps_, **kwargs):
        r = orig(grid, materials, n_steps_, **kwargs)
        cap.update(grid=grid, wire_ports=kwargs.get("wire_ports"), r=r)
        return r

    nur.run_nonuniform = spy
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            sim.run(n_steps=n_steps, compute_s_params=True,
                    s_param_freqs=jnp.asarray(freqs))
    finally:
        nur.run_nonuniform = orig
    for wmsg in wlist:
        print(f"[nu run] warning (verbatim): {wmsg.category.__name__}: "
              f"{wmsg.message}")
    return cap


def nu_raw(cap):
    """(n_ports, 4, n_freqs) complex128 raw accumulators + n_live tuple."""
    raw = [tuple(np.asarray(a, dtype=np.complex128) for a in accs)
           for accs in cap["r"]["wire_sparams_raw"]]
    n_live = tuple(w["n_live"] for w in cap["wire_ports"])
    return raw, n_live


# --------------------------------------------------------- frame algebra ---
def frame_w_column(raw_j, j):
    """Frame W column for drive j from that run's per-port (v,i,vinc,vp)."""
    vpj, ij = raw_j[j][3], raw_j[j][1]
    a_j = (vpj + Z0 * ij) / (2.0 * np.sqrt(Z0))
    col = {}
    for p, (v, i, _vinc, vp) in enumerate(raw_j):
        if p == j:
            b = (vp - Z0 * i) / (2.0 * np.sqrt(Z0))
        else:
            b = (vp - Z0 * i) / (2.0 * np.sqrt(Z0))   # global sign s=+1 here;
            # s is pinned in the --dc arm and is magnitude/reciprocity-
            # invariant (common to S21 and S12).
        col[p] = b / a_j
    return col


def frame_p_column(v_all, i_all, vref_all, n_live, j):
    """Frame P (shipped per-cell #308) column for drive j from the
    uniform bundle's raw accumulators."""
    z0c_j = Z0 / n_live[j]
    a_j = (-vref_all[j, j] + z0c_j * i_all[j, j]) / (2.0 * np.sqrt(z0c_j))
    col = {}
    for p in range(v_all.shape[1]):
        if p == j:
            continue
        z0c = Z0 / n_live[p]
        b = (v_all[j, p] - z0c * i_all[j, p]) / (2.0 * np.sqrt(z0c))
        col[p] = b / a_j
    return col


def fmt(x):
    return np.array2string(np.asarray(x), precision=4, suppress_small=False)


# ------------------------------------------------------------- main arm ----
def main():
    print(f"rfx from {rfx.__file__}")
    assert "issue-770-offdiag" in rfx.__file__, "wrong rfx on sys.path"

    # ---- NU lane (primary): drive port 0, then drive port 1 ----
    nu_cols_w = {}       # drive -> {port: S^W}
    nu_diag_ship = {}    # drive -> shipped NU S_jj
    nu_offdiag_ship = {}  # drive -> shipped NU mixed-frame off-diagonal
    raws = {}
    for j in (0, 1):
        sim = build_fix_t(nu=True, drive=j)
        codes, issues = preflight_verbatim(sim, f"FIX-T NU drive{j}")
        if codes != PINNED_CODES:
            print(f"[FIXTURE] advisory set {codes} != pinned {PINNED_CODES} "
                  f"— recorded (G0 clause; geometry constants are verbatim "
                  f"battery values, so a lane-specific advisory is a "
                  f"deviation to record, a geometry drift is INVALID)")
        cap = run_nu(sim, N_STEPS, FREQS)
        raw, n_live = nu_raw(cap)
        raws[j] = (raw, n_live)
        assert all(n == 2 for n in n_live), f"n_live {n_live} != 2 (G0)"
        S_ship = np.asarray(cap["r"]["s_params"], dtype=np.complex128)
        nu_diag_ship[j] = S_ship[j, j]
        nu_offdiag_ship[j] = S_ship[1 - j, j]
        nu_cols_w[j] = frame_w_column(raw, j)
        # F-A6 wiring: shipped NU diagonal must be reproduced from raw accs.
        vp, i = raw[j][3], raw[j][1]
        s_jj = (vp - Z0 * i) / (vp + Z0 * i)
        d = np.max(np.abs(s_jj - nu_diag_ship[j]))
        gate(d <= A6_WIRING, f"F-A6 NU wiring drive{j}",
             f"max|S_jj(raw) - S_jj(shipped)| = {d:.3e} (gate {A6_WIRING})")

    # ---- uniform lane bundle: frame P (v_ref lives here) + parity ----
    sim_u = build_fix_t(nu=False, drive=None)
    codes_u, _ = preflight_verbatim(sim_u, "FIX-T uniform")
    if codes_u != PINNED_CODES:
        print(f"[FIXTURE] uniform advisory set {codes_u} != pinned "
              f"{PINNED_CODES}")
    bundle = compute_lumped_wire_s_matrix_via_scan(
        sim_u, FREQS, n_steps=N_STEPS, return_vi_dump=True)
    S_u = np.asarray(bundle.s_params, dtype=np.complex128)
    v_all = np.asarray(bundle.raw_voltages_fdt, dtype=np.complex128)
    i_all = np.asarray(bundle.raw_currents, dtype=np.complex128)
    vp_all = np.asarray(bundle.raw_port_voltages_fdt, dtype=np.complex128)
    vref_all = np.asarray(bundle.raw_drive_ref_voltages_fdt,
                          dtype=np.complex128)
    n_live_u = tuple(int(n) for n in bundle.port_cell_counts)
    assert n_live_u == (2, 2), f"uniform n_live {n_live_u} (G0)"

    # F-A6 wiring on the uniform bundle: reproduce the shipped S.
    from rfx.probes.probes import decompose_wire_s_matrix
    S_re = np.asarray(decompose_wire_s_matrix(
        v_all, i_all, np.array([Z0, Z0]), np.array(n_live_u),
        v_port=vp_all, v_ref=vref_all), dtype=np.complex128)
    d = np.max(np.abs(S_re - S_u))
    gate(d <= A6_WIRING, "F-A6 uniform wiring",
         f"max|S(decompose(raw)) - S(bundle)| = {d:.3e} (gate {A6_WIRING})")

    # Frame P columns (shipped off-diagonal, recomputed for the record).
    p_cols = {j: frame_p_column(v_all, i_all, vref_all, n_live_u, j)
              for j in (0, 1)}
    for j in (0, 1):
        d = np.max(np.abs(p_cols[j][1 - j] - S_u[1 - j, j]))
        gate(d <= A6_WIRING, f"F-A6 frame-P wiring drive{j}",
             f"max|b/a(raw) - S_offdiag(bundle)| = {d:.3e}")

    # Frame W from the uniform bundle (for F-A5 parity).
    u_cols_w = {}
    for j in (0, 1):
        raw_u = [(v_all[j, p], i_all[j, p], None, vp_all[j, p])
                 for p in range(2)]
        u_cols_w[j] = frame_w_column(raw_u, j)

    # ---------------------------------------------------------- report ----
    print(f"\nfreqs (GHz): {FREQS / 1e9}")
    for j in (0, 1):
        i = 1 - j
        s_jj = nu_cols_w[j][j]
        s_ij_w = nu_cols_w[j][i]
        s_ij_p = p_cols[j][i]
        print(f"\n--- drive {j} ---")
        print(f"|S_jj| physical (NU, frame W diag): {fmt(np.abs(s_jj))}")
        print(f"|S_ij| frame W (NU):                {fmt(np.abs(s_ij_w))}")
        print(f"|S_ij| frame P (uniform, shipped):  {fmt(np.abs(s_ij_p))}")
        print(f"|S_ij| NU shipped mixed _ab:        "
              f"{fmt(np.abs(nu_offdiag_ship[j]))}")
        ratio = np.abs(s_ij_w) / np.abs(s_ij_p)
        print(f"frame ratio |S^W/S^P| (predicted sqrt(2)*a_P/a_W): "
              f"{fmt(ratio)}")

        # F-A1 per frame.
        for name, s_ij in (("W", s_ij_w), ("P", s_ij_p)):
            cp = np.abs(s_jj) ** 2 + np.abs(s_ij) ** 2
            gate(np.all(cp <= A1_CEIL), f"F-A1 frame {name} drive{j}",
                 f"column power {fmt(cp)} (ceil {A1_CEIL})")
        # F-A2 per frame.
        denom = 1.0 - np.abs(s_jj) ** 2
        for name, s_ij in (("W", s_ij_w), ("P", s_ij_p)):
            T = np.abs(s_ij) ** 2 / denom
            ok = np.all((T >= A2_WINDOW[0]) & (T <= A2_WINDOW[1]))
            gate(ok, f"F-A2 frame {name} drive{j}",
                 f"net-through fraction T = {fmt(T)} (window {A2_WINDOW})")

    # F-A3 reciprocity per frame (NU frame W primary; frame P from bundle).
    r_w = np.abs(nu_cols_w[0][1] - nu_cols_w[1][0])
    r_p = np.abs(p_cols[0][1] - p_cols[1][0])
    for name, r, s21 in (("W", r_w, nu_cols_w[0][1]),
                         ("P", r_p, p_cols[0][1])):
        rel = r / np.abs(s21)
        ok = (np.max(r) <= A3_ABS) or (np.max(rel) <= A3_REL)
        gate(ok, f"F-A3 frame {name}",
             f"max|S21-S12| = {np.max(r):.4e} (abs gate {A3_ABS}), "
             f"max rel = {np.max(rel):.4e} (rel gate {A3_REL}); "
             f"locked class 7.5277e-3")

    # F-A5 lane parity, frame W, all four entries.
    dmax = 0.0
    for j in (0, 1):
        for p in (0, 1):
            dmax = max(dmax, float(np.max(np.abs(
                nu_cols_w[j][p] - u_cols_w[j][p]))))
    gate(dmax <= A5_ATOL, "F-A5 lane parity frame W",
         f"max|S^W_NU - S^W_uniform| = {dmax:.3e} (gate {A5_ATOL})")

    print("\n==== verdict inputs complete ====")
    print(f"failures: {FAILS if FAILS else 'none'}")


# --------------------------------------------------------------- DC arm ----
def dc_arm():
    print(f"rfx from {rfx.__file__}")
    assert "issue-770-offdiag" in rfx.__file__, "wrong rfx on sys.path"
    sim = build_fix_t(nu=True, drive=0,
                      pulse=GaussianPulse(f0=2.5e9, bandwidth=1.0))
    preflight_verbatim(sim, "FIX-T-DC NU drive0")
    cap = run_nu(sim, DC_N_STEPS, DC_FREQS)
    raw, n_live = nu_raw(cap)
    assert all(n == 2 for n in n_live), f"n_live {n_live} != 2 (G0)"
    col = frame_w_column(raw, 0)
    s21 = col[1]
    expected = np.exp(-1j * 2 * np.pi * DC_FREQS * L / C0)
    dev_plus = np.angle(s21 / expected)
    dev_minus = np.angle(-s21 / expected)
    # Global sign pin (declared): s = the sign whose 0.5 GHz dev is
    # nearest 0.  Then the F-A4 band applies to the pinned channel and
    # the flipped channel must leave it.
    s = +1 if abs(dev_plus[0]) <= abs(dev_minus[0]) else -1
    dev = dev_plus if s == +1 else dev_minus
    dev_f = dev_minus if s == +1 else dev_plus
    print(f"|S21^W| = {fmt(np.abs(s21))} at {DC_FREQS/1e9} GHz")
    print(f"pinned global receive sign s = {s:+d}")
    print(f"dev (pinned) = {fmt(dev)} rad; dev (flipped) = {fmt(dev_f)} rad")
    lo, hi = DCA_BAND
    gate(np.all((dev > lo) & (dev < hi)), "F-A4 DC anchor frame W",
         f"dev = {fmt(dev)} rad, band ({lo}, {hi})")
    gate(not np.all((dev_f > lo) & (dev_f < hi)),
         "F-A4 pi-discrimination",
         f"flipped dev = {fmt(dev_f)} rad must leave the band")
    print(f"failures: {FAILS if FAILS else 'none'}")


if __name__ == "__main__":
    if "--dc" in sys.argv:
        dc_arm()
    else:
        main()
