"""A slab against a high cavity wall solves as its mirror image against the low wall.

Physics checked. The anchor: on the PMC cavities with a static eps_r 4 slab,
both the high-wall and the low-wall solve resonate within 1 % of the exact
root of the 2-D layered transverse-resonance equation, solved here in float64
without rfx. The mirror invariant: a slab against the high wall gives the same
probe record as its mirror image against the low wall, up to float32 rounding.

Structure: a closed cavity 20 x 8 x 16 mm on 1 mm cells, PEC on x and z, PMC on y.
A dielectric slab fills 4 of the 8 cells across y over the full x and z extent:
against the high y wall in case H, against the low y wall in case L. An Ez
source and an Ez probe sit off every symmetry plane. The anchor is the physics
check; the mirror pair is the control that says which wall, if any, is wrong.

Variants (one parametrized test):

- the PMC-y cavity with an eps_r 4 slab (anchor and mirror);
- the same construction on x (PMC on x, PEC on y and z, the x and y lengths
  swapped, mirrored by x -> W - x; anchor and mirror);
- the PEC-y control cavity: PEC on every face, the same slab pair, an Ey
  source and probe at the same positions; the mode Ey ~ sin(pi x/A) sin(pi z/D)
  crosses the slab interface with Dy about uniform in y (series fill, eps_eff
  1.6), so f_est = (c/2) sqrt(A^-2 + D^-2) / sqrt(1.6), about 9.49 GHz, a mode
  selector only (measured 8.55 GHz, the same rule on both sides). PEC zeroes
  the wall edges, so this pair must agree with or without the repair.

The Debye slab uses the full-record DFT bar alone (producer Decision record 2).
Its pencil frequency and identification error are diagnostics only.

Mirror on the realized lattice. Case H's indices are the nearest nodes of the
requested positions (round half up, the rule of ``Grid.position_to_index``
for these closed cavities, which have no absorber pad). Along the mirror axis,
with N = W/h cells, case L takes j_L = N - j_H for a component that sits on
nodes along that axis (Ez on y or x) and j_L = N - j_H - 1 for one displaced
by half a cell (Ey on y: index j is the edge at (j + 1/2) h, so its image
(N - j - 1/2) h is index N - j - 1). Case L is requested at j_L h, and the test
asserts that every realized source and probe index is the intended one.

Measured on both sides by the same rule, from the two solves only:

- f_H, f_L: matrix pencil on samples [1800, 6000) of the probe record, the
  mode nearest f_est. For the Ez variants f_est = c/(2 L sqrt(2.5)), L the
  cavity length along the axis that carries the sin(pi . / L) variation,
  pencil band [0.5 f_est, 1.5 c/(2L)]; for the control the band is
  [0.5, 1.5] f_est. Precondition: that mode's identification error < 1e-6,
  else ``CannotJudge``.
- D(f): DFT of the full probe record at 41 bins, over 3-7 GHz for the Ez
  variants and over [0.6, 1.4] f_est for the control.

Bars (rfx-archive rfx/plans/idm/20261026-high-wall-mirror/DESIGN.md section 3
and its decision records 1-3):

- anchor (v2 accuracy bar item 4, resonances within 1 %):
  |f_H / f_closed - 1| <= 1e-2 and |f_L / f_closed - 1| <= 1e-2;
- mirror: a PMC mirror pair matches image theory within 6e-6 of the field
  peak after 800 steps in float32 (ledger entry "Magnetic (PMC) walls on x and
  on y are not mirror images...", scope-limited 2026-10-21), so the pair's
  remaining difference is rounding, and the bars are the repo's cross-trace
  bar for a quantity summed over the record:
  max over bins |D_H - D_L| <= 1e-4 * max |D_L|; |f_H / f_L - 1| <= 1e-4.

Before the high-wall repair the PMC-y cavity measured f_H / f_L - 1 = 2.17e-2
(mirror_shift.out, main 4b58aee42), 200 times the mirror bar.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.harminv import matrix_pencil

C0 = 299792458.0
H = 1e-3
A, B, D = 20 * H, 8 * H, 16 * H
N_STEPS = 6000
PENCIL_START = 1800
PENCIL_ERROR_MAX = 1e-6
N_BINS = 41
FREQ_BAR = 1e-4
DFT_BAR = 1e-4
ANCHOR_BAR = 1e-2
SLAB_CELLS = 4
SLAB_EPS = 4.0

# Ez mode sin(pi s / L), uniform across the slab: f = c / (2 L sqrt(2.5)).
F_EZ = C0 / (2.0 * A * np.sqrt(2.5))
# Control Ey mode sin(pi x / A) sin(pi z / D); it crosses the slab interface with
# Dy about uniform in y, so the series fill gives eps_eff = 1 / (0.5 / 4 + 0.5) = 1.6.
F_EY = 0.5 * C0 * np.sqrt(A ** -2 + D ** -2) / np.sqrt(1.6)

# name: (domain, boundary, mirror axis, sin axis, component, f_est,
#        pencil band, DFT band, freq_max, closed-form anchor)
VARIANTS = {
    "pmc_y": ((A, B, D), dict(x="pec", y="pmc", z="pec"), 1, 0, "ez", F_EZ,
              (0.5 * F_EZ, 1.5 * C0 / (2.0 * A)), (3e9, 7e9), 12e9, True),
    "pmc_x": ((B, A, D), dict(x="pmc", y="pec", z="pec"), 0, 1, "ez", F_EZ,
              (0.5 * F_EZ, 1.5 * C0 / (2.0 * A)), (3e9, 7e9), 12e9, True),
    "pmc_z": ((A, D, B), dict(x="pec", y="pec", z="pmc"), 2, 0, "ey", F_EZ,
              (0.5 * F_EZ, 1.5 * C0 / (2.0 * A)), (3e9, 7e9), 12e9, True),
    "pec_y_control": ((A, B, D), dict(x="pec", y="pec", z="pec"), 1, 0, "ey", F_EY,
                      (0.5 * F_EY, 1.5 * F_EY), (0.6 * F_EY, 1.4 * F_EY), 15e9, False),
}
VARIANTS["graded_pmc_y"] = VARIANTS["pmc_y"]
VARIANTS["debye_pmc_y"] = (*VARIANTS["pmc_y"][:9], False)
# Each half is 4 mm; the sequence is deliberately not palindromic.
GRADED_Y = H * np.array([0.90, 1.10, 1.05, 0.95, 1.08, 0.92, 1.10, 0.90])

# Case H positions as fractions of (mirror length W, sin length L, D).
SOURCE_H = (0.30, 0.31, 0.43)
PROBE_H = (0.20, 0.63, 0.57)


class CannotJudge(Exception):
    """The pencil cannot identify the mode: no verdict on the bar either way."""


def _nearest_node(x):
    return int(np.floor(x / H + 0.5))


def _indices(variant, frac, side):
    """Intended realized (i, j, k) of a point source or probe, mirrored on the lattice."""
    domain, _, mirror_ax, sin_ax, comp = VARIANTS[variant][:5]
    pos = [0.0, 0.0, 0.0]
    pos[mirror_ax] = frac[0] * domain[mirror_ax]
    pos[sin_ax] = frac[1] * domain[sin_ax]
    pos[3 - mirror_ax - sin_ax] = frac[2] * D
    idx = [_nearest_node(p) for p in pos]
    if variant == "graded_pmc_y":
        nodes = np.r_[0.0, np.cumsum(GRADED_Y)]
        idx[mirror_ax] = int(np.argmin(abs(nodes - pos[mirror_ax])))
    if side == "L":
        n = _nearest_node(domain[mirror_ax])
        displaced = comp == "e" + "xyz"[mirror_ax]
        idx[mirror_ax] = n - idx[mirror_ax] - (1 if displaced else 0)
    return tuple(idx)


def _point(variant, frac, side):
    """Requested position: case H as declared; case L at j_L h on the mirror axis."""
    domain, _, mirror_ax, sin_ax = VARIANTS[variant][:4]
    p = [0.0, 0.0, 0.0]
    p[mirror_ax] = frac[0] * domain[mirror_ax]
    p[sin_ax] = frac[1] * domain[sin_ax]
    p[3 - mirror_ax - sin_ax] = frac[2] * D
    if side == "L":
        p[mirror_ax] = _indices(variant, frac, "L")[mirror_ax] * H
    if variant == "graded_pmc_y":
        widths = GRADED_Y if side == "H" else GRADED_Y[::-1]
        p[mirror_ax] = float(np.r_[0.0, np.cumsum(widths)][_indices(variant, frac, side)[mirror_ax]])
    return tuple(p)


def _slab_bounds(variant, side):
    """Slab extent along the mirror axis: against the high wall in H, the low wall in L."""
    domain, _, mirror_ax = VARIANTS[variant][:3]
    w = domain[mirror_ax]
    if side == "H":
        return w - SLAB_CELLS * H, w
    return 0.0, SLAB_CELLS * H


def _solve(variant, side):
    domain, boundary, mirror_ax, _, comp, f_est, _, _, freq_max = VARIANTS[variant][:9]
    profiles = {}
    if variant == "graded_pmc_y":
        profiles = dict(dy_profile=(GRADED_Y if side == "H" else GRADED_Y[::-1]).copy(),
                        dz_profile=np.full(16, H))
    sim = Simulation(freq_max=freq_max, domain=domain, dx=H, boundary=BoundarySpec(**boundary), **profiles)
    poles = {}
    if variant == "debye_pmc_y":
        from rfx.materials.debye import DebyePole
        poles = dict(debye_poles=[DebyePole(delta_eps=3.0, tau=1 / (2 * np.pi * 50e9))])
    sim.add_material("d", eps_r=SLAB_EPS, **poles)
    lo, hi = _slab_bounds(variant, side)
    lo_c, hi_c = [0.0, 0.0, 0.0], list(domain)
    lo_c[mirror_ax], hi_c[mirror_ax] = lo, hi
    sim.add(Box(tuple(lo_c), tuple(hi_c)), material="d")
    sim.add_source(_point(variant, SOURCE_H, side), comp,
                   waveform=GaussianPulse(f0=f_est, bandwidth=0.8), amplitude_kind="current")
    sim.add_probe(_point(variant, PROBE_H, side), comp)
    r = sim.run(n_steps=N_STEPS, compute_s_params=False)
    rg = r.realized_geometry
    src = tuple(int(k) for k in rg.sources[0].edges[0])
    prb = tuple(int(k) for k in rg.probes[0].index)
    want_src, want_prb = _indices(variant, SOURCE_H, side), _indices(variant, PROBE_H, side)
    assert (src, prb) == (want_src, want_prb), (
        f"{variant} case {side}: realized source {src}, probe {prb}; "
        f"intended source {want_src}, probe {want_prb}")
    trace = np.asarray(r.time_series)[:, 0].astype(np.float64)
    return trace, float(rg.dt), src, prb


def _mode(trace, dt, variant, side):
    f_est, (f_lo, f_hi) = VARIANTS[variant][5:7]
    modes = matrix_pencil(trace[PENCIL_START:], dt, f_lo, f_hi)
    listing = "\n".join(f"  {m.freq:.6e} Hz  error {m.error:.2e}  amp {m.amplitude:.3e}"
                        for m in modes)
    if not modes:
        raise CannotJudge(f"case {side} has no mode in the pencil band")
    m = min(modes, key=lambda m: abs(m.freq - f_est))
    if not m.error < PENCIL_ERROR_MAX:
        raise CannotJudge(f"case {side} mode {m.freq:.6e} Hz has identification "
                          f"error {m.error:.2e} >= {PENCIL_ERROR_MAX:.0e}; modes:\n{listing}")
    return m


def _dft_freqs(variant):
    return np.linspace(*VARIANTS[variant][7], N_BINS)


def _dft(trace, dt, freqs):
    t = np.arange(len(trace)) * dt
    return np.exp(-2j * np.pi * freqs[:, None] * t[None, :]) @ trace * dt


def _layered_resonance(a, b, t, eps1):
    """Lowest resonance of the 2-D layered cavity, float64, independent of rfx.

    Assumptions: the field is uniform along z (the axis normal to the PEC faces
    that Ez meets); PEC at x = 0 and x = a gives Ez ~ sin(pi x / a); PMC at
    y = 0 and y = b gives dEz/dy = 0 there; a slab of eps1 fills 0 <= y <= t,
    air fills t < y <= b. Then Ez = cos(k1 y) in the slab and cos(k2 (b - y))
    in the air, with k_i^2 = eps_i k0^2 - (pi/a)^2, and continuity of Ez and
    dEz/dy at y = t gives the transverse-resonance equation

        k1 tan(k1 t) = -k2 tan(k2 (b - t)),

    with k2 = i kappa below the air cutoff (-k2 tan(k2 L) -> kappa tanh(kappa L)).
    The root is the lowest one above c / (2 a sqrt(eps1)), where k1 = 0. The
    function g = k1 tan(k1 t) + k2 tan(k2 (b - t)) rises with f between poles,
    so a root is a - to + crossing; a pole is a + to - jump and is skipped.
    Bracketed on a scan, then bisected.
    """
    kc = np.pi / a

    def g(f):
        k0 = 2.0 * np.pi * f / C0
        k1 = np.sqrt(eps1 * k0 ** 2 - kc ** 2)
        k2sq = k0 ** 2 - kc ** 2
        if k2sq >= 0.0:
            k2 = np.sqrt(k2sq)
            return k1 * np.tan(k1 * t) + k2 * np.tan(k2 * (b - t))
        kappa = np.sqrt(-k2sq)
        return k1 * np.tan(k1 * t) - kappa * np.tanh(kappa * (b - t))

    f_low = C0 / (2.0 * a * np.sqrt(eps1)) * (1.0 + 1e-9)
    scan = np.linspace(f_low, 2.0 * C0 / (2.0 * a), 20001)
    vals = [g(f) for f in scan]
    for f0, f1, g0, g1 in zip(scan[:-1], scan[1:], vals[:-1], vals[1:]):
        if g0 < 0.0 <= g1:
            lo, hi = float(f0), float(f1)
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if g(mid) < 0.0:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
    raise RuntimeError("no root of the transverse-resonance equation in the scan")


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_high_wall_slab_solves_as_its_low_wall_mirror(variant):
    tr_h, dt_h, src_h, prb_h = _solve(variant, "H")
    tr_l, dt_l, src_l, prb_l = _solve(variant, "L")
    assert dt_h == dt_l

    if variant == "debye_pmc_y":
        def diagnostic(trace, dt):
            from types import SimpleNamespace
            modes = matrix_pencil(trace[PENCIL_START:], dt, *VARIANTS[variant][6])
            return (min(modes, key=lambda m: abs(m.freq - VARIANTS[variant][5])) if modes
                    else SimpleNamespace(freq=float("nan"), error=float("nan")))
        m_h, m_l = diagnostic(tr_h, dt_h), diagnostic(tr_l, dt_l)
    else:
        m_h = _mode(tr_h, dt_h, variant, "H")
        m_l = _mode(tr_l, dt_l, variant, "L")
    ratio = m_h.freq / m_l.freq

    freqs = _dft_freqs(variant)
    d_h, d_l = _dft(tr_h, dt_h, freqs), _dft(tr_l, dt_l, freqs)
    diff = np.abs(d_h - d_l)
    worst = int(np.argmax(diff))
    rel = float(diff[worst] / np.max(np.abs(d_l)))

    failed = []
    report = (f"{variant}: dt {dt_h:.6e} s, {len(tr_h)} steps, f_est {VARIANTS[variant][5]:.6e} Hz; "
              f"source H {src_h} L {src_l}, probe H {prb_h} L {prb_l}; "
              f"f_H {m_h.freq:.6e} Hz (error {m_h.error:.1e}), "
              f"f_L {m_l.freq:.6e} Hz (error {m_l.error:.1e}), f_H/f_L - 1 = {ratio - 1:+.3e}; "
              f"max|D_H - D_L|/max|D_L| = {rel:.3e} at {freqs[worst] / 1e9:.3f} GHz")
    if VARIANTS[variant][9]:
        domain, _, mirror_ax, sin_ax = VARIANTS[variant][:4]
        f_closed = _layered_resonance(domain[sin_ax], domain[mirror_ax], SLAB_CELLS * H, SLAB_EPS)
        dev_h, dev_l = m_h.freq / f_closed - 1.0, m_l.freq / f_closed - 1.0
        report += (f"; f_closed {f_closed:.6e} Hz, f_H {m_h.freq:.6e} Hz, f_L {m_l.freq:.6e} Hz, "
                   f"f_H/f_closed - 1 = {dev_h:+.3e}, f_L/f_closed - 1 = {dev_l:+.3e}")
        if not abs(dev_h) <= ANCHOR_BAR:
            failed.append("anchor H")
        if not abs(dev_l) <= ANCHOR_BAR:
            failed.append("anchor L")
    if variant != "debye_pmc_y" and not abs(ratio - 1.0) <= FREQ_BAR:
        failed.append("mirror frequency")
    if not rel <= DFT_BAR:
        failed.append("mirror DFT")
    print(report, file=sys.stderr)
    assert not failed, f"failed: {', '.join(failed)}; {report}"
