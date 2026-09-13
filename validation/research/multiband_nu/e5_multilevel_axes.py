"""E5 — multi-level unequal bands on x, y and z (Lane 1 of the NU
full-functionality program, docs/design_notes/20260913_nu_full_functionality_program.md
section 4; this lane's own note is
docs/design_notes/20260913_nu_lane1_multilevel_xyz_predeclaration.md).

Question: does the z transition law measured by E1 (one fine + one coarse
size, symmetric cap ramps) extend to (a) a fine band between two DIFFERENT
coarse sizes traversed in both directions, (b) two fine bands of different
sizes in one column, and (c) the same structures on the x and y axes — and
what is its range there?

Instrument: the E1 two-run differencing (``w6_band_builder.e1_arm``'s
form, unchanged) on an axis-relabeled PEC-closed TE10 fixture. Reused
verbatim: ``chain_model.scattering`` / ``step_reflection`` / ``bloch_kz`` /
``s0_sy``; ``w2_w3_reflection.gaussian_sine`` / ``dft_at`` / ``vg_of``;
``w6_band_builder._cell_delay`` / ``_git_sha`` / ``_git_dirty`` /
``e1_stopband_edge_hz`` / ``_round_half_up`` and its E1 layout constants;
``harness.build_pec_fixture`` for the z fixture; ``rfx.make_band_profile``,
``rfx.make_nonuniform_grid``, ``run_nonuniform``. None of
``w6_band_builder.py``, ``chain_model.py``, ``w2_w3_reflection.py``,
``harness.py``, ``fixtures.py`` is edited by this lane.

Axis relabeling (cyclic, x -> y -> z -> x): z = the existing fixture
(``ex``, invariant x, sine y); x -> (``ey``, invariant y, sine z); y ->
(``ez``, invariant z, sine x). On x and y every profile is built with
``boundary_cell = 1.0 mm`` (G1: the x/y end-cell guard); the pin cells and
their ramps are instrumentation shared by the A and B runs and are
stripped before the chain solve.

Usage (declared in the note; run from the worktree with the pinned
PYTHONPATH):

    python -m validation.research.multiband_nu.e5_multilevel_axes --model-only \
        --out validation/research/multiband_nu/results/e5_model.json
    ... --axis z --patterns S --out .../results/e5_z.json                 (L1-0)
    ... --relabel --z-json .../results/e5_z.json --out .../results/e5_relabel.json  (L1-R)
    ... --axis z --patterns P1,P2,P3,P4,T --resume --out .../results/e5_z.json      (L1-Z)
    ... --axis x --out .../results/e5_x.json                              (L1-X)
    ... --pin-bridge --z-json .../results/e5_z.json --out .../results/e5_pinbridge.json
    ... --axis y --out .../results/e5_y.json                              (L1-Y)
    ... --note-tables .../results/e5_model.json      (markdown tables for the note)
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time

import numpy as np
import jax.numpy as jnp

import rfx
from rfx.core.yee import MaterialArrays
from rfx.nonuniform import make_band_profile, make_nonuniform_grid, run_nonuniform

from . import fixtures as fx
from .chain_model import bloch_kz, s0_sy, scattering, step_reflection
from .harness import build_pec_fixture
from .w2_w3_reflection import F0, SIGMA_T, T0, FS2_FLOOR, dft_at, gaussian_sine, vg_of
from .w6_band_builder import (
    B_N_FINE_PIN, C0_E1, E1_BOUND_SLACK, E1_C_FIT_MAX_DR, E1_C_TOL_DR,
    E1_CLASS_N_LAMBDA, E1_CLASS_RATIO, E1_LAW_SWEEP_MAX, E1_T_RUN, E1_T_RUN_PAD,
    E1_Z_B, E1_Z_LEAD, E1_Z_PRB, E1_Z_SRC, E1_Z_TAIL, _cell_delay, _git_dirty,
    _git_sha, _round_half_up, e1_stopband_edge_hz,
)

# --- fixed physical fixture (lane A / E1 values, unchanged) ---------------------
CAP = 1.4
D_F = fx.DZ_FINE                 # 1.0 mm fine cell (29.98 cells per lambda0)
D_F2 = 0.5e-3                    # pattern T second fine cell
DR = D_F * CAP                   # 1.4 mm   (E1 ramp_cell_m arithmetic)
DC = D_F * CAP * CAP             # 1.96 mm  (E1 coarse_cell_m arithmetic)
DC3 = D_F * CAP * CAP * CAP      # 2.744 mm (= 1.4^3 d_f)
D_T = fx.DXY                     # 1.5 mm transverse cell
N_LAMBDA_NOMINAL = 30            # E1's n_lambda of the 1.0 mm cell (29.98 per lambda0)
N_SIN = 20                       # sine-axis cells (b = 30 mm)
N_INV = 3                        # invariant-axis cells (a = 4.5 mm)
PIN = 1.0e-3                     # boundary_cell on x and y
WIDTHS = (2, 4, 8, 16, 32)
T_NB1 = 4
T_NC = (4, 16)
T_NB2 = (4, 8)
PATTERNS = ("S", "P1", "P2", "P3", "P4", "T")
FP_PATTERNS = ("S", "P1", "P2", "P3", "P4")      # patterns with a c window (W3)
STRUCTURE_ROLES = ("lead", "band", "mid", "tail", "ramp")
RELABEL_N_B = 4
RELABEL_B_TAIL = 4               # B_sym tail cells after the far fine pin
E1_JSON = "validation/research/multiband_nu/results/e1_band_law_sweep.json"

# --- frozen windows (program document 4.4; copied, never edited) ----------------
W1_FLOOR_MODEL = 2.5e-4          # E1 1f floor-reading rule ...
W1_FLOOR_ABS = 1.5e-4            # ... |R_meas - R_model| <= 1.5e-4 -> "fired at floor"
W4_MEAS_REL = 1e-6
W4_MODEL_REL = 1e-8
W5_REL = 1e-6
W5_PIN_ABS = 1.5e-4
W6_EXPECT_MIN_MARGIN_NS = 0.256


def rle(prof) -> list:
    """Run-length encoding ``[[value, count], ...]`` of a cell vector — exact
    (float64 values verbatim), so a 700-cell profile is a dozen entries."""
    out: list = []
    for c in np.asarray(prof, dtype=np.float64):
        c = float(c)
        if out and out[-1][0] == c:
            out[-1][1] += 1
        else:
            out.append([c, 1])
    return out


def unrle(pairs) -> np.ndarray:
    return np.asarray([v for v, n in pairs for _ in range(int(n))], dtype=np.float64)


def _sha(prof: np.ndarray, *extra) -> str:
    h = hashlib.sha1(np.ascontiguousarray(prof, dtype=np.float64).tobytes())
    h.update(repr(extra).encode())
    return h.hexdigest()[:12]


# --- axis relabeling --------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class AxisSpec:
    axis: str
    comp: str
    inv_axis: str
    sin_axis: str
    n_inv: int = N_INV
    n_sin: int = N_SIN
    d_t: float = D_T

    def ijk(self, graded: int, inv: int, sin: int) -> tuple[int, int, int]:
        d = {self.axis: graded, self.inv_axis: inv, self.sin_axis: sin}
        return (d["x"], d["y"], d["z"])

    @staticmethod
    def n_nodes(grid, axis: str) -> int:
        return {"x": grid.nx, "y": grid.ny, "z": grid.nz}[axis]

    def inv_arrays(self, grid):
        return {"x": (grid.inv_dx, grid.inv_dx_h), "y": (grid.inv_dy, grid.inv_dy_h),
                "z": (grid.inv_dz, grid.inv_dz_h)}[self.axis]


AXES = {
    "z": AxisSpec("z", "ex", "x", "y"),
    "x": AxisSpec("x", "ey", "y", "z"),
    "y": AxisSpec("y", "ez", "z", "x"),
}


def build_pec_fixture_axis(profile: np.ndarray, spec: AxisSpec):
    """z: the harness call verbatim. x/y: the rotated fixture with the
    transverse profiles passed explicitly (``dy_profile=None`` would make
    the y spacing equal the scalar ``dx``)."""
    prof = np.asarray(profile, dtype=np.float64)
    if spec.axis == "z":
        return build_pec_fixture(prof, (fx.A_X, fx.B_Y), fx.DXY)
    if spec.axis == "x":
        grid = make_nonuniform_grid(
            (0.0, 0.0), dz_profile=np.full(spec.n_sin, spec.d_t), dx=float(prof[0]),
            cpml_layers=0, dx_profile=prof, dy_profile=np.full(spec.n_inv, spec.d_t))
    elif spec.axis == "y":
        grid = make_nonuniform_grid(
            (0.0, 0.0), dz_profile=np.full(spec.n_inv, spec.d_t), dx=spec.d_t,
            cpml_layers=0, dx_profile=np.full(spec.n_sin, spec.d_t), dy_profile=prof)
    else:
        raise ValueError(spec.axis)
    shape = (grid.nx, grid.ny, grid.nz)
    mats = MaterialArrays(eps_r=jnp.ones(shape, dtype=jnp.float32),
                          mu_r=jnp.ones(shape, dtype=jnp.float32),
                          sigma=jnp.zeros(shape, dtype=jnp.float32))
    return grid, mats


def te10_sources_axis(grid, spec: AxisSpec, k_src: int, waveform: np.ndarray):
    """Same list, same order as ``w2_w3_reflection.te10_sources`` (invariant
    outer, sine inner), with the component and the slot relabeled."""
    n_inv = spec.n_nodes(grid, spec.inv_axis)
    n_sin = spec.n_nodes(grid, spec.sin_axis)
    d_t = spec.d_t
    b = (n_sin - 1) * d_t
    srcs = []
    for a in range(n_inv - 1):
        for s in range(1, n_sin - 1):
            amp = float(np.sin(np.pi * (s * d_t) / b))
            i, j, k = spec.ijk(k_src, a, s)
            srcs.append((i, j, k, spec.comp, waveform * np.float32(amp)))
    return srcs


def run_probe_axis(profile: np.ndarray, spec: AxisSpec, k_src: int, k_prb: int,
                   n_steps: int):
    grid, mats = build_pec_fixture_axis(profile, spec)
    wf = gaussian_sine(n_steps, float(grid.dt), SIGMA_T, T0)
    srcs = te10_sources_axis(grid, spec, k_src, wf)
    n_sin = spec.n_nodes(grid, spec.sin_axis)
    probe = spec.ijk(k_prb, 1, n_sin // 2) + (spec.comp,)
    out = run_nonuniform(grid, mats, n_steps, sources=srcs, probes=[probe])
    return grid, np.asarray(out["time_series"][:, 0], dtype=np.float64)


# --- declared vectors (the builder's own rule, G5) ------------------------------
@dataclasses.dataclass(frozen=True)
class Seg:
    role: str        # lead | band | mid | tail | dtpin | end
    target: float
    n: int           # TOTAL cells at `target` (a ramp's last cell counts)
    protected: bool

    @property
    def structure(self) -> bool:
        return self.role in ("lead", "band", "mid", "tail")


def _m_steps(ratio: float, cap: float) -> int:
    """``rfx.nonuniform._ramp_steps`` rule (1e-10 slack on the log)."""
    return max(1, int(np.ceil(np.log(ratio) / np.log(cap) - 1e-10)))


def ramp_cells(v: float, u: float, cap: float = CAP) -> list[float]:
    """The m - 1 intermediate cells of the builder's geometric ramp from the
    seam cell ``v`` up to the plateau ``u`` (the m-th cell is ``u`` itself
    and is counted with the plateau). ``rho = cap`` when the ratio is an
    exact power of the cap (E1's ``d_f x r`` arithmetic), else
    ``ratio^(1/m)`` — the builder's ``_ramp_cells``."""
    ratio = u / v
    if ratio <= 1.0 + 1e-12:
        return []
    m = _m_steps(ratio, cap)
    x = np.log(ratio) / np.log(cap)
    rho = cap if abs(x - round(x)) <= 1e-9 else ratio ** (1.0 / m)
    return [v * rho ** i for i in range(1, m)]


def expected_vector(segs: list[Seg], cap: float = CAP, pin: float | None = None):
    """Declared cells, per-cell roles, builder edges and protected flags.

    Ramps sit inside the coarser segment; a pin is a one-cell neighbour of
    the end segments. A ramp is structure only when both its segments are
    structure; pins, pin ramps, dt-pin bands and end segments are
    instrumentation (stripped before the chain solve). Edges are
    accumulated left to right, ``n x target`` as one product, so the S
    pattern on z reproduces E1's ``z1 = n_lead dc + dr`` arithmetic."""
    cells: list[float] = []
    roles: list[str] = []
    edges = [0.0]
    e = 0.0

    def add(vals, role):
        nonlocal e
        for c in vals:
            cells.append(float(c))
            roles.append(role)
            e += float(c)

    for i, s in enumerate(segs):
        lo_nb = pin if i == 0 else segs[i - 1].target
        hi_nb = pin if i == len(segs) - 1 else segs[i + 1].target
        lo_inst = i == 0 or not segs[i - 1].structure
        hi_inst = i == len(segs) - 1 or not segs[i + 1].structure
        if i == 0 and pin is not None:
            add([pin], "pin")
        if lo_nb is not None and lo_nb < s.target:
            add(ramp_cells(lo_nb, s.target, cap),
                "ramp" if (s.structure and not lo_inst) else "inst")
        cells.extend([float(s.target)] * s.n)
        roles.extend([s.role if s.structure else "inst"] * s.n)
        e += s.n * float(s.target)
        if hi_nb is not None and hi_nb < s.target:
            add(list(reversed(ramp_cells(hi_nb, s.target, cap))),
                "ramp" if (s.structure and not hi_inst) else "inst")
        if i == len(segs) - 1 and pin is not None:
            add([pin], "pin")
        edges.append(e)
    return (np.asarray(cells, dtype=np.float64), roles, edges,
            [s.protected for s in segs])


def build_declared(segs: list[Seg], cap: float = CAP, pin: float | None = None):
    """Builder output for the declared segments plus the declared vector,
    the structure slice bounds and the E1 1e-12 m match flag."""
    exp, roles, edges, prot = expected_vector(segs, cap, pin)
    prof = make_band_profile(edges, [s.target for s in segs], protected=prot,
                             max_ratio=cap, boundary_cell=pin)
    exact = len(prof) == len(exp) and bool(np.max(np.abs(prof - exp)) <= 1e-12)
    idx = [k for k, r in enumerate(roles) if r in STRUCTURE_ROLES]
    m_lo = idx[0]
    m_hi = len(roles) - 1 - idx[-1]
    assert idx == list(range(idx[0], idx[-1] + 1)), "structure cells not contiguous"
    return {"profile": prof, "expected": exp, "roles": roles, "edges": edges,
            "protected": prot, "exact": exact, "m_lo": m_lo, "m_hi": m_hi}


def b_profile_general(lead_cell: float, n_b: int, pin_cell: float) -> np.ndarray:
    """The B reference on z (E1's explicit form): ``[lead] x n_B + [pin] x 4``."""
    return np.asarray([lead_cell] * n_b + [pin_cell] * B_N_FINE_PIN, dtype=np.float64)


# --- layout in cells of the LEAD cell (E1 1c rule) --------------------------------
def layout(lead: float, tail: float) -> dict:
    return {"lead_cell_m": lead, "tail_cell_m": tail,
            "k_src": _round_half_up(E1_Z_SRC / lead), "k_prb": _round_half_up(E1_Z_PRB / lead),
            "n_lead": _round_half_up(E1_Z_LEAD / lead), "n_tail": _round_half_up(E1_Z_TAIL / tail),
            "n_B": _round_half_up(E1_Z_B / lead)}


def _seg_lead(lead: float) -> Seg:
    return Seg("lead", lead, layout(lead, lead)["n_lead"], False)


def _seg_tail(tail: float, protected: bool) -> Seg:
    return Seg("tail", tail, layout(tail, tail)["n_tail"], protected)


def pattern_spec(pattern: str) -> dict:
    """Structure segments of every arm of a pattern (axis-free)."""
    band = lambda d, n: Seg("band", d, n, True)   # noqa: E731
    if pattern == "S":
        c_l = c_r = DC
    elif pattern == "P1":
        c_l, c_r = DC, DC3
    elif pattern == "P2":
        c_l, c_r = DC3, DC
    elif pattern == "P3":
        c_l, c_r = DC, DR
    elif pattern == "P4":
        c_l, c_r = DR, DC
    elif pattern == "T":
        arms = {}
        for n_c in T_NC:
            for n_b2 in T_NB2:
                arms[f"T_c{n_c}_b{n_b2}"] = {
                    "kind": "band", "n_c": n_c, "n_b2": n_b2, "n_b": T_NB1,
                    "segs": [_seg_lead(DC), band(D_F, T_NB1), Seg("mid", DC, n_c, False),
                             band(D_F2, n_b2), _seg_tail(DC, False)]}
        singles = {
            "L1": {"kind": "single", "side": "L1",
                   "segs": [_seg_lead(DC), _seg_tail(D_F, True)]},
            "L2": {"kind": "single", "side": "L2",
                   "segs": [_seg_lead(DC), _seg_tail(D_F2, True)]},
        }
        # chain-only mirrors (the bound's four amplitudes; no FDTD)
        mirrors = {
            "R1": [_seg_lead(D_F), _seg_tail(DC, False)],
            "R2": [_seg_lead(D_F2), _seg_tail(DC, False)],
        }
        return {"pattern": "T", "d_min": D_F2, "singles": singles, "bands": arms,
                "mirrors": mirrors, "c_l": DC, "c_r": DC}
    else:
        raise ValueError(pattern)
    bands = {f"{pattern}_nb{n_b}": {"kind": "band", "n_b": n_b,
                                     "segs": [_seg_lead(c_l), band(D_F, n_b), _seg_tail(c_r, False)]}
             for n_b in WIDTHS}
    singles = {"L": {"kind": "single", "side": "L",
                     "segs": [_seg_lead(c_l), _seg_tail(D_F, True)]}}
    if pattern != "S":
        singles["R"] = {"kind": "single", "side": "R",
                        "segs": [_seg_lead(D_F), _seg_tail(c_r, False)]}
    mirrors = {} if pattern != "S" else {"R": [_seg_lead(D_F), _seg_tail(c_r, False)]}
    return {"pattern": pattern, "d_min": D_F, "singles": singles, "bands": bands,
            "mirrors": mirrors, "c_l": c_l, "c_r": c_r}


def axis_segments(segs: list[Seg], d_min: float, pinned: bool) -> list[Seg]:
    """Instrumentation added around the structure: a dt-pin band when the
    arm's own minimum sits above the family minimum (the T single L1), and
    on a pinned axis an end segment after a protected last segment so the
    pin has a free host (the builder refuses a protected end segment)."""
    out = list(segs)
    own_min = min(s.target for s in out)
    if d_min < own_min * (1 - 1e-12):
        out[-1] = dataclasses.replace(out[-1], protected=False)   # hosts the ramp down
        out.append(Seg("dtpin", d_min, B_N_FINE_PIN, True))
    if pinned and out[-1].protected:
        out.append(Seg("end", PIN, 2, False))
    return out


def b_segments(lead: float, n_b: int, d_min: float) -> list[Seg]:
    """B reference on a pinned axis: lead x n_B between the pins; the pin
    cells (1.0 mm) are the dt pin of the 1.0 mm family, and a realized
    dt-pin band plus a free end segment is added beyond the gate when the
    family minimum sits below the pin (pattern T)."""
    out = [Seg("lead", lead, n_b, False)]
    if d_min < PIN * (1 - 1e-12):
        out.append(Seg("dtpin", d_min, B_N_FINE_PIN, True))
        out.append(Seg("end", PIN, 2, False))
    return out


def make_b_profile(lead: float, n_b: int, d_min: float, pinned: bool) -> dict:
    if not pinned:
        prof = b_profile_general(lead, n_b, d_min)
        return {"profile": prof, "expected": prof, "exact": True, "m_lo": 0,
                "m_hi": B_N_FINE_PIN, "roles": ["lead"] * n_b + ["inst"] * B_N_FINE_PIN}
    return build_declared(b_segments(lead, n_b, d_min), CAP, PIN)


# --- gates (E1's five margins, generalized; plus the direct far-end return) --------
def gates_general(prof: np.ndarray, lead: float, k_src_phys: int, k_prb_phys: int,
                  n_lead: int, n_tail: int, m_lo: int, m_hi: int, n_b_ref: int,
                  dt: float, d_t: float, b: float, n_steps: int) -> dict:
    vg_l = vg_of(lead, dt, d_t, b)
    z_src, z_prb, z_tr = k_src_phys * lead, k_prb_phys * lead, n_lead * lead
    pin_lo = _cell_delay(prof[:m_lo], dt, d_t, b) if m_lo else 0.0
    t_r = T0 + (2 * z_tr - z_src - z_prb) / vg_l
    t_s = T0 + (z_src + 2 * z_tr - z_prb) / vg_l + 2 * pin_lo
    i_tr = m_lo + n_lead
    beyond = prof[i_tr:]
    t_f = T0 + (z_tr - z_src) / vg_l + 2 * _cell_delay(beyond, dt, d_t, b) + (z_tr - z_prb) / vg_l
    inner = prof[i_tr:len(prof) - m_hi - n_tail]
    t_inner = T0 + (z_tr - z_src) / vg_l + 2 * _cell_delay(inner, dt, d_t, b) + (z_tr - z_prb) / vg_l
    gate_end = min(t_s, t_f) - 4 * SIGMA_T
    n_gate = int(gate_end / dt)
    t_echo_prb = T0 + (z_src + z_prb) / vg_l + 2 * pin_lo
    t_inc_arr = T0 + (z_prb - z_src) / vg_l
    t_inc_end = min(t_inc_arr + 8 * SIGMA_T, t_echo_prb - 4 * SIGMA_T)
    t_bpin = T0 + (z_src + 2 * n_b_ref * lead - z_prb) / vg_l
    t_bfar = T0 + (2 * n_b_ref * lead - z_src - z_prb) / vg_l
    margins = {
        "reflection_inside": gate_end - (t_r + 4 * SIGMA_T),
        "inner_return_inside": gate_end - (t_inner + 4 * SIGMA_T),
        "run_covers_gate": n_steps * dt - gate_end,
        "b_pin_return_after_gate": t_bpin - gate_end,
        "incident_inside": t_inc_end - (t_inc_arr + 4 * SIGMA_T),
        "b_far_return_direct_after_gate": t_bfar - gate_end,
    }
    return {
        "gates_ns": {"t_r": t_r * 1e9, "t_inner_last": t_inner * 1e9, "t_s": t_s * 1e9,
                     "t_f": t_f * 1e9, "gate_end": gate_end * 1e9, "gate_steps": n_gate,
                     "t_inc_end": t_inc_end * 1e9, "t_b_pin_return": t_bpin * 1e9,
                     "t_b_far_direct": t_bfar * 1e9, "run_end": n_steps * dt * 1e9,
                     "pin_delay_lo": pin_lo * 1e9},
        "gate_margins_ns": {k: v * 1e9 for k, v in margins.items()},
        "gates_hold": bool(all(v > 0 for v in margins.values())),
        "_n_gate": n_gate, "_t_inc_end": t_inc_end, "_t_s": t_s,
    }


# --- law side ----------------------------------------------------------------------
def fit_c_two_amp(n_bs, r_vals, d_f: float, r_l: float, r_r: float, k_g: float,
                  dr: float) -> float:
    """Least-squares c in ``sqrt(R_L^2 + R_R^2 - 2 R_L R_R cos(2 k_g (n_b d_f
    + c)))``, R_L / R_R / k_g fixed; search [0, 3 dr], scan + two refinements
    (E1 (iii) verbatim). For R_L = R_R this is ``2 R |sin(k_g (L + c))|``."""
    L = np.asarray(n_bs, dtype=np.float64) * d_f
    R = np.asarray(r_vals, dtype=np.float64)

    def err(c):
        m = np.sqrt(np.maximum(r_l ** 2 + r_r ** 2 - 2 * r_l * r_r * np.cos(2 * k_g * (L + c)), 0.0))
        return float(np.sum((R - m) ** 2))

    c_max = E1_C_FIT_MAX_DR * dr
    lo, hi, n = 0.0, c_max, 4001
    for _ in range(3):
        cs = np.linspace(lo, hi, n)
        e = np.array([err(c) for c in cs])
        i = int(np.argmin(e))
        step = cs[1] - cs[0]
        lo, hi = max(0.0, cs[i] - step), min(c_max, cs[i] + step)
    return float(cs[i])


def fp_two_amp(n_b: int, c: float, d_f: float, r_l: float, r_r: float, k_g: float) -> float:
    return float(np.sqrt(max(r_l ** 2 + r_r ** 2 - 2 * r_l * r_r * np.cos(2 * k_g * (n_b * d_f + c)), 0.0)))


def side_centroid_mm(n_ramp_cells: int, r: float = CAP, d_f: float = D_F) -> float:
    """Hypothesis (reported, not gated): amplitude-weighted centroid of the
    step planes outside the band edge with the ``(local cell)^2`` weights
    of E1 1g. 0 ramp cells -> 0; 1 -> DR r^2/(1+r^2); 2 -> (r^2 DR + r^4 (DR
    + r DR)) / (1 + r^2 + r^4)."""
    dr = r * d_f
    if n_ramp_cells == 0:
        return 0.0
    planes = [0.0]
    w = [1.0]
    pos = 0.0
    for k in range(1, n_ramp_cells + 1):
        pos += dr * r ** (k - 1)
        planes.append(pos)
        w.append(r ** (2 * k))
    return float(np.dot(w, planes) / np.sum(w) * 1e3)


def chain_r(prof_struct: np.ndarray, n_lead: int, n_tail: int, dt: float) -> float:
    return abs(scattering(prof_struct, n_lead, n_tail, F0, dt, D_T, fx.B_Y)[0])


def w1_window(r_model: float) -> float:
    return 0.20 * r_model + FS2_FLOOR


def w1_verdict(r_meas: float, r_model: float) -> dict:
    half = w1_window(r_model)
    dev = abs(r_meas - r_model)
    fired = bool(dev > half)
    return {"deviation": dev, "deviation_rel": dev / r_model, "half": half,
            "window": [r_model - half, r_model + half], "fired": fired,
            "fired_at_floor": bool(fired and r_model < W1_FLOOR_MODEL and dev <= W1_FLOOR_ABS)}


# --- one (pattern, axis) cell ---------------------------------------------------------
class _RunCache:
    def __init__(self):
        self.runs: dict[str, tuple] = {}
        self.n_new = 0

    def run(self, spec: AxisSpec, prof: np.ndarray, k_src: int, k_prb: int, n_steps: int):
        key = _sha(prof, spec.axis, k_src, k_prb, n_steps)
        cached = key in self.runs
        if not cached:
            self.runs[key] = run_probe_axis(prof, spec, k_src, k_prb, n_steps)
            self.n_new += 1
        return key, cached, self.runs[key]


def arm_layout(arm: dict) -> dict:
    lead, tail = arm["segs"][0].target, arm["segs"][-1].target
    return layout(lead, tail)


def model_cell(pattern: str, spec: AxisSpec, pinned: bool) -> dict:
    """Chain-model side of one (pattern, axis) cell — zero FDTD. Frozen
    into results/e5_model.json before any run."""
    ps = pattern_spec(pattern)
    d_min = ps["d_min"]
    d_t, b = D_T, fx.B_Y
    # dt from the B reference of the band arms (the family's lead)
    lay_band = layout(ps["c_l"], ps["c_r"])
    b_band = make_b_profile(ps["c_l"], lay_band["n_B"], d_min, pinned)
    grid_b, _ = build_pec_fixture_axis(b_band["profile"], spec)
    dt = float(grid_b.dt)
    out = {"pattern": pattern, "axis": spec.axis, "pinned": pinned, "dt_s": dt,
           "d_min_m": d_min, "c_l_m": ps["c_l"], "c_r_m": ps["c_r"],
           "layout_band": lay_band, "b_band": _prof_record(b_band, lay_band["n_B"]),
           "singles": {}, "mirrors": {}, "bands": []}
    t_s_max = 0.0
    arms_for_steps = []
    # singles (FDTD arms) and mirrors (chain only)
    for name, arm in ps["singles"].items():
        rec = _model_arm(arm, spec, pinned, d_min, dt)
        out["singles"][name] = rec
        arms_for_steps.append(rec)
    for name, segs in ps["mirrors"].items():
        # chain-only mirrors (never run in FDTD): evaluated unpinned — the
        # chain solve strips the pins anyway, and a 0.5 mm lead cannot host
        # a coarser 1.0 mm pin (the builder would ramp DOWN from the pin)
        arm = {"kind": "single", "side": name, "segs": segs}
        out["mirrors"][name] = _model_arm(arm, spec, False, d_min, dt, chain_only=True)
    for name, arm in ps["bands"].items():
        rec = _model_arm(arm, spec, pinned, d_min, dt)
        rec["name"] = name
        out["bands"].append(rec)
        arms_for_steps.append(rec)
    # run length (E1 1c): the same for every arm of the cell
    for rec in arms_for_steps:
        t_s_max = max(t_s_max, rec["_t_s"])
    n_steps = int(np.ceil(max(E1_T_RUN, t_s_max + E1_T_RUN_PAD) / dt - 1e-9))
    out["n_steps"] = n_steps
    out["run_ns"] = n_steps * dt * 1e9
    # gates at the final n_steps
    for rec in arms_for_steps:
        _finish_gates(rec, n_steps, dt)
    for rec in out["mirrors"].values():
        _finish_gates(rec, n_steps, dt)
    # law side
    r_l = out["singles"]["L" if pattern != "T" else "L1"]["R_model"]
    if pattern == "T":
        amps = {"R_1L": out["singles"]["L1"]["R_model"], "R_1R": out["mirrors"]["R1"]["R_model"],
                "R_2L": out["singles"]["L2"]["R_model"], "R_2R": out["mirrors"]["R2"]["R_model"]}
        bound = (1 + E1_BOUND_SLACK) * sum(amps.values())
        out["law"] = {"amplitudes": amps, "bound_sum": bound,
                      "k_g_fine_per_m": bloch_kz(F0, dt, d_t, b, D_F),
                      "k_g_fine2_per_m": bloch_kz(F0, dt, d_t, b, D_F2)}
    else:
        r_r = out["singles"]["R"]["R_model"] if "R" in out["singles"] else out["mirrors"]["R"]["R_model"]
        k_g = bloch_kz(F0, dt, d_t, b, D_F)
        sweep_n = list(range(0, E1_LAW_SWEEP_MAX + 1))
        sweep_r = []
        lay = layout(ps["c_l"], ps["c_r"])
        for n_b in sweep_n:
            segs = [_seg_lead(ps["c_l"]), Seg("band", D_F, n_b, True), _seg_tail(ps["c_r"], False)]
            exp, _, _, _ = expected_vector(segs, CAP, None)
            sweep_r.append(chain_r(exp, lay["n_lead"], lay["n_tail"], dt))
        c_model = fit_c_two_amp(sweep_n, sweep_r, D_F, r_l, r_r, k_g, DR)
        fit = [fp_two_amp(n, c_model, D_F, r_l, r_r, k_g) for n in sweep_n]
        n_ramp_l = len(ramp_cells(D_F, ps["c_l"]))
        n_ramp_r = len(ramp_cells(D_F, ps["c_r"]))
        c_closed = (side_centroid_mm(n_ramp_l) + side_centroid_mm(n_ramp_r)) * 1e-3
        bound = (1 + E1_BOUND_SLACK) * (r_l + r_r)
        out["law"] = {
            "R_L": r_l, "R_R": r_r, "bound_sum": bound, "k_g_fine_per_m": k_g,
            "lambda_g_fine_mm": 2 * np.pi / k_g * 1e3,
            "c_model_m": c_model, "c_window_m": E1_C_TOL_DR * DR,
            "c_closed_form_m": c_closed, "n_ramp_cells_L": n_ramp_l, "n_ramp_cells_R": n_ramp_r,
            "c_fit_rms_0_80": float(np.sqrt(np.mean((np.asarray(sweep_r) - np.asarray(fit)) ** 2))),
            "c_fit_max_rel_0_80": float(np.max(np.abs(np.asarray(sweep_r) - np.asarray(fit))
                                               / np.maximum(np.asarray(sweep_r), 1e-300))),
            "chain_max_over_sum": float(max(sweep_r) / (r_l + r_r)),
            "chain_min_0_80": float(min(sweep_r)), "abs_RL_minus_RR": abs(r_l - r_r),
            "chain_sweep_R": sweep_r,
        }
        for rec in out["bands"]:
            rec["fp_form_model"] = fp_two_amp(rec["n_b"], c_model, D_F, r_l, r_r, k_g)
    for rec in out["bands"]:
        rec["bound_sum"] = out["law"]["bound_sum"]
    # runway diagnostics
    coarsest = max(ps["c_l"], ps["c_r"])
    out["runways"] = {
        "coarsest_cell_m": coarsest,
        "coarsest_cells_per_lambda0": (C0_E1 / F0) / coarsest,
        "fine_cells_per_lambda0": (C0_E1 / F0) / D_F,
        "stopband_edge_ghz_coarsest": (lambda f: None if f is None else f / 1e9)(
            e1_stopband_edge_hz(coarsest, dt, d_t, b)),
        "vg_over_c": {f"{c*1e3:.3f}mm": vg_of(c, dt, d_t, b) / C0_E1
                      for c in sorted({ps["c_l"], ps["c_r"], D_F, d_min})},
        "coarse_bloch_arg_coarsest": coarsest * np.sqrt(
            s0_sy(F0, dt, d_t, b)[0] ** 2 - s0_sy(F0, dt, d_t, b)[1] ** 2) / 2,
    }
    # E1's class rule verbatim: every adjacent ratio <= 1.4 (measured on the
    # builder output of every band arm) and the fine band at the nominal
    # 30 cells per lambda0 (E1's n_lambda for the 1.0 mm cell; T's 0.5 mm
    # band is the nominal 60).
    out["max_adjacent_ratio_bands"] = max(rec["max_adjacent_ratio"] for rec in out["bands"])
    out["in_accuracy_class"] = bool(out["max_adjacent_ratio_bands"] <= E1_CLASS_RATIO + 1e-9
                                    and N_LAMBDA_NOMINAL >= E1_CLASS_N_LAMBDA)
    # every B reference of the cell (one per distinct lead among its arms)
    leads = sorted({rec["layout"]["lead_cell_m"] for rec in list(out["singles"].values()) + out["bands"]})
    out["b_refs_model"] = {f"{lead*1e3:.3f}mm": _prof_record(
        make_b_profile(lead, layout(lead, lead)["n_B"], d_min, pinned), layout(lead, lead)["n_B"])
        for lead in leads}
    for rec in list(out["singles"].values()) + list(out["mirrors"].values()) + out["bands"]:
        rec.pop("_t_s", None)
    return out


def _max_ratio(cells) -> float:
    c = np.asarray(cells, dtype=np.float64)
    rr = c[1:] / c[:-1]
    return float(np.max(np.maximum(rr, 1 / rr)))


def _prof_record(built: dict, n_b: int) -> dict:
    return {"profile_rle": rle(built["profile"]),
            "builder_matches_declared_vector": bool(built["exact"]),
            "n_cells": int(len(built["profile"])), "m_lo": built["m_lo"], "m_hi": built["m_hi"],
            "n_B": n_b}


def _model_arm(arm: dict, spec: AxisSpec, pinned: bool, d_min: float, dt: float,
               chain_only: bool = False) -> dict:
    lay = arm_layout(arm)
    segs = axis_segments(arm["segs"], d_min, pinned)
    built = build_declared(segs, CAP, PIN if pinned else None)
    prof = built["profile"]
    m_lo, m_hi = built["m_lo"], built["m_hi"]
    struct = prof[m_lo:len(prof) - m_hi]
    struct_exp = built["expected"][m_lo:len(prof) - m_hi]
    r_model = chain_r(struct, lay["n_lead"], lay["n_tail"], dt)
    rec = {
        "kind": arm["kind"], "side": arm.get("side"), "n_b": arm.get("n_b"),
        "n_c": arm.get("n_c"), "n_b2": arm.get("n_b2"),
        "segments": [dataclasses.asdict(s) for s in segs],
        "layout": lay, "m_lo": m_lo, "m_hi": m_hi, "n_cells": int(len(prof)),
        "n_struct_cells": int(len(struct)),
        "profile_rle": rle(prof),
        "builder_matches_declared_vector": bool(built["exact"]),
        "max_abs_cell_dev_m": float(np.max(np.abs(prof - built["expected"]))
                                    if len(prof) == len(built["expected"]) else np.inf),
        "realized_min_m": float(prof.min()), "declared_min_m": float(min(s.target for s in segs)),
        "max_adjacent_ratio": _max_ratio(prof),
        "R_model": r_model, "R_model_db": 20 * np.log10(r_model),
        "R_model_on_declared": chain_r(struct_exp, lay["n_lead"], lay["n_tail"], dt),
        "half": w1_window(r_model), "window": [r_model - w1_window(r_model), r_model + w1_window(r_model)],
        "k_src_used": m_lo + lay["k_src"], "k_prb_used": m_lo + lay["k_prb"],
        "chain_only": chain_only,
    }
    if arm["kind"] == "single" and len(struct) > lay["n_lead"] + lay["n_tail"]:
        rec["step_reflections"] = [step_reflection(float(struct[k]), float(struct[k + 1]), F0, dt, D_T, fx.B_Y)
                                   for k in range(lay["n_lead"] - 1, len(struct) - lay["n_tail"])]
    elif arm["kind"] == "single":
        rec["step_reflections"] = [step_reflection(float(struct[lay["n_lead"] - 1]), float(struct[lay["n_lead"]]),
                                                   F0, dt, D_T, fx.B_Y)]
    # gates need n_steps; the E1 run-length rule needs t_s first
    g = gates_general(prof, lay["lead_cell_m"], lay["k_src"], lay["k_prb"], lay["n_lead"],
                      lay["n_tail"], m_lo, m_hi, lay["n_B"], dt, D_T, fx.B_Y, 1)
    rec["_t_s"] = g["_t_s"]
    rec["_gate_args"] = (m_lo, m_hi)
    return rec


def _finish_gates(rec: dict, n_steps: int, dt: float) -> None:
    lay = rec["layout"]
    prof = unrle(rec["profile_rle"])
    g = gates_general(prof, lay["lead_cell_m"], lay["k_src"], lay["k_prb"], lay["n_lead"],
                      lay["n_tail"], rec["m_lo"], rec["m_hi"], lay["n_B"], dt, D_T, fx.B_Y, n_steps)
    rec["gates_ns"], rec["gate_margins_ns"], rec["gates_hold"] = (
        g["gates_ns"], g["gate_margins_ns"], g["gates_hold"])
    rec["n_gate"], rec["n_inc"] = g["_n_gate"], int(g["_t_inc_end"] / dt)
    rec.pop("_gate_args", None)


def measure_arm(rec: dict, spec: AxisSpec, cache: _RunCache, b_key: str, grid_b, trace_b,
                n_steps: int) -> dict:
    """One FDTD arm against its B reference (E1's ``e1_arm`` form)."""
    prof = unrle(rec["profile_rle"])
    lay = rec["layout"]
    key, cached, (grid_a, trace_a) = cache.run(spec, prof, rec["k_src_used"], rec["k_prb_used"], n_steps)
    dt = float(grid_a.dt)
    dt_b = float(grid_b.dt)
    dt_match = abs(dt - dt_b) < 1e-20
    n_lead_tot = rec["m_lo"] + lay["n_lead"]
    lead = lay["lead_cell_m"]
    planes_ok = bool(rec["k_prb_used"] < n_lead_tot and rec["k_src_used"] >= rec["m_lo"]
                     and abs(prof[rec["k_src_used"]] - lead) <= 1e-12
                     and abs(prof[rec["k_prb_used"]] - lead) <= 1e-12)
    inv_a, inv_a_h = spec.inv_arrays(grid_a)
    inv_b, inv_b_h = spec.inv_arrays(grid_b)
    lead_f32 = bool(np.array_equal(np.asarray(inv_a[:n_lead_tot]), np.asarray(inv_b[:n_lead_tot]))
                    and np.array_equal(np.asarray(inv_a_h[:n_lead_tot]), np.asarray(inv_b_h[:n_lead_tot])))
    diff = trace_a - trace_b
    refl = dft_at(diff, dt, F0, 0, min(rec["n_gate"], n_steps))
    inc = dft_at(trace_b, dt, F0, 0, rec["n_inc"])
    r_meas = abs(refl) / abs(inc)
    v = w1_verdict(r_meas, rec["R_model"])
    out = {
        "run_id": key, "b_run_id": b_key, "fdtd_run_cached": cached,
        "dt_s": dt, "dt_b_s": dt_b, "dt_diff_s": abs(dt - dt_b), "dt_matches_b": dt_match,
        "R_meas": r_meas, "R_meas_db": 20 * np.log10(max(r_meas, 1e-300)),
        "inc_abs": abs(inc), "refl_abs": abs(refl),
        "source_probe_in_lead": planes_ok, "lead_f32_identical": lead_f32,
        "lead_f32_slice": n_lead_tot,
        "gates_hold": bool(rec["gates_hold"] and dt_match and planes_ok and lead_f32),
        "trace_max_abs_a": float(np.max(np.abs(trace_a))),
    }
    out.update(v)
    if rec["kind"] == "band":
        out["bound_sum"] = rec["bound_sum"]
        out["bound_fired"] = bool(r_meas > rec["bound_sum"])
    return out


def cell_verdicts(cell: dict) -> dict:
    """The declared conjunctions, from stored per-arm records (replayable)."""
    singles = cell["singles"]
    bands = cell["bands"]
    any_w1 = any(a["meas"]["fired"] for a in bands) or any(s["meas"]["fired"] for s in singles.values())
    any_w2 = any(a["meas"]["bound_fired"] for a in bands)
    all_gates = all(a["meas"]["gates_hold"] for a in bands) and all(s["meas"]["gates_hold"] for s in singles.values())
    v = {"any_w1_fired": bool(any_w1), "any_w2_fired": bool(any_w2), "all_gates_hold": bool(all_gates),
         "w1_fired_arms": [a["name"] for a in bands if a["meas"]["fired"]]
                          + [f"single_{k}" for k, s in singles.items() if s["meas"]["fired"]],
         "w1_fired_at_floor_arms": [a["name"] for a in bands if a["meas"]["fired_at_floor"]]}
    if cell["pattern"] in FP_PATTERNS:
        law = cell["law"]
        c_meas = fit_c_two_amp([a["n_b"] for a in bands], [a["meas"]["R_meas"] for a in bands],
                               D_F, law["R_L"], law["R_R"], law["k_g_fine_per_m"], DR)
        c_dev = abs(c_meas - law["c_model_m"])
        v.update({"c_meas_m": c_meas, "c_model_m": law["c_model_m"], "c_dev_m": c_dev,
                  "c_dev_over_ramp_cell": c_dev / DR, "c_window_m": law["c_window_m"],
                  "c_closed_form_m": law["c_closed_form_m"],
                  "c_closed_minus_model_m": law["c_closed_form_m"] - law["c_model_m"],
                  "w3_fired": bool(c_dev > law["c_window_m"])})
        v["in_law_domain"] = bool(not any_w1 and not any_w2 and not v["w3_fired"])
    else:
        v["w3_fired"] = None
        v["in_law_domain"] = bool(not any_w1 and not any_w2)
    return v


def run_cell(pattern: str, spec: AxisSpec, pinned: bool, cache: _RunCache) -> dict:
    cell = model_cell(pattern, spec, pinned)
    n_steps = cell["n_steps"]
    print(f"E5 {spec.axis}/{pattern}: dt={cell['dt_s']:.6e} n_steps={n_steps} "
          f"lead={cell['c_l_m']*1e3:.3f}mm tail={cell['c_r_m']*1e3:.3f}mm "
          f"K={cell['layout_band']['k_src']}/{cell['layout_band']['k_prb']} "
          f"lead/tail/B={cell['layout_band']['n_lead']}/{cell['layout_band']['n_tail']}/{cell['layout_band']['n_B']} "
          f"pinned={pinned}", flush=True)
    t0 = time.time()
    d_min = cell["d_min_m"]
    b_runs: dict[float, tuple] = {}

    def b_for(lead: float):
        if lead not in b_runs:
            lay = layout(lead, lead)
            built = make_b_profile(lead, lay["n_B"], d_min, pinned)
            key, cached, (grid_b, trace_b) = cache.run(spec, built["profile"], built["m_lo"] + lay["k_src"],
                                                        built["m_lo"] + lay["k_prb"], n_steps)
            assert abs(float(grid_b.dt) - cell["dt_s"]) < 1e-20, (float(grid_b.dt), cell["dt_s"])
            b_runs[lead] = (key, cached, grid_b, trace_b, built)
            print(f"   B lead={lead*1e3:.3f}mm cells={len(built['profile'])} "
                  f"exact={built['exact']} run={key} cached={cached}", flush=True)
        return b_runs[lead]

    for name, rec in cell["singles"].items():
        key, cached, grid_b, trace_b, _ = b_for(rec["layout"]["lead_cell_m"])
        rec["meas"] = measure_arm(rec, spec, cache, key, grid_b, trace_b, n_steps)
        m = rec["meas"]
        print(f"   single {name}: R_meas={m['R_meas']:.4e} ({m['R_meas_db']:.1f} dB) "
              f"model={rec['R_model']:.4e} dev={m['deviation_rel']*100:.2f}% fired={m['fired']} "
              f"gates_hold={m['gates_hold']} cached={m['fdtd_run_cached']}", flush=True)
    for rec in cell["bands"]:
        key, cached, grid_b, trace_b, _ = b_for(rec["layout"]["lead_cell_m"])
        rec["meas"] = measure_arm(rec, spec, cache, key, grid_b, trace_b, n_steps)
        m = rec["meas"]
        print(f"   {rec['name']}: R_meas={m['R_meas']:.4e} ({m['R_meas_db']:.1f} dB) "
              f"model={rec['R_model']:.4e} dev={m['deviation_rel']*100:.2f}% fired={m['fired']} "
              f"bound_fired={m['bound_fired']} gates_hold={m['gates_hold']} margins(ns)="
              + ", ".join(f"{k}={v:.3f}" for k, v in rec["gate_margins_ns"].items()), flush=True)
    cell["b_refs"] = {f"{lead*1e3:.3f}mm": {"run_id": v[0], "fdtd_run_cached": v[1],
                                            "dt_s": float(v[2].dt), **_prof_record(v[4], layout(lead, lead)["n_B"])}
                      for lead, v in b_runs.items()}
    cell["verdicts"] = cell_verdicts(cell)
    cell["wallclock_s"] = time.time() - t0
    v = cell["verdicts"]
    print(f"   verdicts: w1_fired={v['w1_fired_arms']} w2_fired={v['any_w2_fired']} "
          f"w3_fired={v['w3_fired']} gates={v['all_gates_hold']} in_law_domain={v['in_law_domain']} "
          + (f"c_meas={v['c_meas_m']*1e3:.3f}mm c_model={v['c_model_m']*1e3:.3f}mm dev={v['c_dev_m']*1e3:.3f}mm "
             if v["w3_fired"] is not None else "")
          + f"wallclock={cell['wallclock_s']:.1f}s", flush=True)
    if pattern == "S" and spec.axis == "z" and not pinned:
        # relabel reference traces (L1-R compares against these)
        for rec in cell["bands"]:
            if rec["n_b"] == RELABEL_N_B:
                _, _, (grid_a, trace_a) = cache.run(spec, unrle(rec["profile_rle"]), rec["k_src_used"],
                                                    rec["k_prb_used"], n_steps)
                _, _, grid_b, trace_b, _ = b_for(rec["layout"]["lead_cell_m"])
                cell["relabel_reference"] = {"n_b": RELABEL_N_B, "trace_a": trace_a.tolist(),
                                             "trace_b_e1": trace_b.tolist(),
                                             "R_meas": rec["meas"]["R_meas"], "dt_s": float(grid_a.dt)}
    return cell


def _provenance(model_only: bool) -> dict:
    return {"rfx_file": rfx.__file__, "argv": sys.argv[1:], "git_sha": _git_sha(),
            "git_dirty": _git_dirty(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_only": model_only, "F0_Hz": F0, "sigma_t_s": SIGMA_T, "t0_s": T0,
            "d_t_m": D_T, "b_m": fx.B_Y, "a_m": fx.A_X, "cap": CAP, "d_f_m": D_F, "d_f2_m": D_F2,
            "pin_m": PIN, "widths": list(WIDTHS), "t_nc": list(T_NC), "t_nb2": list(T_NB2),
            "windows": {"W1_per_arm": "|R_meas - R_model| <= 0.20 R_model + 3e-5",
                        "W1_floor_rule": "fired and R_model < 2.5e-4 and dev <= 1.5e-4 -> fired at floor",
                        "W2_bound": "R_meas <= 1.05 x (R_L + R_R) [T: four amplitudes]",
                        "W3_c_tol_ramp_cells": E1_C_TOL_DR, "W3_c_window_m": E1_C_TOL_DR * DR,
                        "W4_control_meas_rel": W4_MEAS_REL, "W4_control_model_rel": W4_MODEL_REL,
                        "W5_relabel_rel": W5_REL, "W5_pin_bridge_abs": W5_PIN_ABS,
                        "W6_expected_min_margin_ns": W6_EXPECT_MIN_MARGIN_NS}}


def run_model_only(out_path: str) -> dict:
    results = _provenance(True)
    results["axes"] = {}
    t0 = time.time()
    for axis, spec in AXES.items():
        results["axes"][axis] = {}
        for pattern in PATTERNS:
            cell = model_cell(pattern, spec, pinned=(axis != "z"))
            results["axes"][axis][pattern] = cell
            print(f"model {axis}/{pattern}: dt={cell['dt_s']:.6e} n_steps={cell['n_steps']} "
                  + (f"R_L={cell['law']['R_L']:.4e} R_R={cell['law']['R_R']:.4e} "
                     f"c_model={cell['law']['c_model_m']*1e3:.3f}mm c_closed={cell['law']['c_closed_form_m']*1e3:.3f}mm "
                     if pattern != "T" else
                     "amps=" + " ".join(f"{k}={v:.4e}" for k, v in cell["law"]["amplitudes"].items()) + " ")
                  + f"bound={cell['law']['bound_sum']:.4e} "
                  f"min_margin={min(min(r['gate_margins_ns'].values()) for r in cell['bands'] + list(cell['singles'].values())):.3f}ns",
                  flush=True)
            for rec in cell["bands"]:
                print(f"   {rec['name']}: exact={rec['builder_matches_declared_vector']} cells={rec['n_cells']} "
                      f"R_model={rec['R_model']:.4e} ({rec['R_model_db']:.1f} dB) window=[{rec['window'][0]:.4e}, "
                      f"{rec['window'][1]:.4e}] gates_hold={rec['gates_hold']}", flush=True)
    # pin-bridge and relabel model rows (z pinned; S n_b = 4 unpinned with B_sym)
    results["axes"]["z_pinned"] = {"S": model_cell("S", AXES["z"], pinned=True)}
    results["wallclock_s"] = time.time() - t0
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", out_path)
    return results


def run_axis(axis: str, patterns: list[str], out_path: str, resume: bool, pinned: bool | None = None) -> dict:
    spec = AXES[axis]
    pinned = (axis != "z") if pinned is None else pinned
    results = _provenance(False)
    results.update({"axis": axis, "pinned": pinned, "cells": {}})
    if resume:
        try:
            with open(out_path) as fh:
                prev = json.load(fh)
            results["cells"] = prev.get("cells", {})
            results["resumed_from"] = {"git_sha": prev.get("git_sha"), "started_utc": prev.get("started_utc"),
                                       "argv": prev.get("argv")}
        except (OSError, ValueError):
            pass
    cache = _RunCache()
    t0 = time.time()
    for pattern in patterns:
        if pattern in results["cells"]:
            print(f"E5 {axis}/{pattern}: already in {out_path}, skipped (resume)", flush=True)
            continue
        results["cells"][pattern] = run_cell(pattern, spec, pinned, cache)
        if pattern == "S" and axis == "z" and not pinned:
            with open(E1_JSON) as fh:
                results["w4_control"] = w4_control(results["cells"]["S"], json.load(fh))
            for r in results["w4_control"]["rows"]:
                print(f"   W4 {r['arm']}: R_meas={r['R_meas']:.10e} E1={r['R_meas_e1']:.10e} rel={r['meas_rel']:.3e} "
                      f"bit_identical={r['meas_bit_identical']} | R_model rel={r['model_rel']:.3e} "
                      f"fired={r['meas_fired'] or r['model_fired']}", flush=True)
            print(f"   W4 c_model diff={results['w4_control']['c_model_diff_m']:.3e} m "
                  f"c_meas diff={results['w4_control']['c_meas_diff_m']:.3e} m any_fired={results['w4_control']['any_fired']}",
                  flush=True)
        results["fdtd_runs_new_this_call"] = cache.n_new
        results["wallclock_s"] = time.time() - t0
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=1)
    results["wallclock_s"] = time.time() - t0
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {out_path} (new FDTD runs this call: {cache.n_new}, wallclock {results['wallclock_s']:.1f}s)")
    return results


# --- L1-R relabel identity ---------------------------------------------------------------
def relabel_arm(spec: AxisSpec, cache: _RunCache, n_steps: int) -> dict:
    """Pattern S, n_b = 4, UNPINNED on the given axis, against B_sym =
    ``[DC] x 400 | DR | [D_F] x 4 | DR | [DC] x 4`` (ends 1.96 mm on both
    sides so the x/y end-cell guard passes without pins)."""
    lay = layout(DC, DC)
    a_segs = [_seg_lead(DC), Seg("band", D_F, RELABEL_N_B, True), _seg_tail(DC, False)]
    a_built = build_declared(a_segs, CAP, None)
    b_segs = [Seg("lead", DC, lay["n_B"], False), Seg("dtpin", D_F, B_N_FINE_PIN, True),
              Seg("end", DC, RELABEL_B_TAIL, False)]
    b_built = build_declared(b_segs, CAP, None)
    key_a, cached_a, (grid_a, trace_a) = cache.run(spec, a_built["profile"], lay["k_src"], lay["k_prb"], n_steps)
    key_b, cached_b, (grid_b, trace_b) = cache.run(spec, b_built["profile"], lay["k_src"], lay["k_prb"], n_steps)
    dt = float(grid_a.dt)
    prof = a_built["profile"]
    g = gates_general(prof, DC, lay["k_src"], lay["k_prb"], lay["n_lead"], lay["n_tail"], 0, 0,
                      lay["n_B"], dt, D_T, fx.B_Y, n_steps)
    r_model = chain_r(prof, lay["n_lead"], lay["n_tail"], dt)
    refl = dft_at(trace_a - trace_b, dt, F0, 0, min(g["_n_gate"], n_steps))
    inc = dft_at(trace_b, dt, F0, 0, int(g["_t_inc_end"] / dt))
    r_meas = abs(refl) / abs(inc)
    return {"axis": spec.axis, "comp": spec.comp, "dt_s": dt, "dt_b_s": float(grid_b.dt),
            "grid_shape": [grid_a.nx, grid_a.ny, grid_a.nz],
            "a_profile_rle": rle(prof), "a_builder_exact": bool(a_built["exact"]),
            "b_sym_profile_rle": rle(b_built["profile"]), "b_builder_exact": bool(b_built["exact"]),
            "run_id_a": key_a, "run_id_b": key_b, "cached_a": cached_a, "cached_b": cached_b,
            "R_model": r_model, "R_meas": r_meas, **w1_verdict(r_meas, r_model),
            "gates_ns": g["gates_ns"], "gate_margins_ns": g["gate_margins_ns"], "gates_hold": g["gates_hold"],
            "trace_a": trace_a.tolist(), "trace_b_sym": trace_b.tolist(),
            "grid_dx_dy_scalar": [float(grid_a.dx), float(grid_a.dy)]}


def relabel_compare(ref: dict, arm: dict) -> dict:
    ta, tz = np.asarray(arm["trace_a"]), np.asarray(ref["trace_a"])
    tb, tbz = np.asarray(arm["trace_b_sym"]), np.asarray(ref["trace_b_sym"])
    out = {
        "trace_a_rel_maxdiff": float(np.max(np.abs(ta - tz)) / np.max(np.abs(tz))),
        "trace_b_rel_maxdiff": float(np.max(np.abs(tb - tbz)) / np.max(np.abs(tbz))),
        "trace_a_bit_identical": bool(np.array_equal(ta, tz)),
        "trace_b_bit_identical": bool(np.array_equal(tb, tbz)),
        "R_meas_rel_diff": abs(arm["R_meas"] - ref["R_meas"]) / ref["R_meas"],
        "dt_diff_s": abs(arm["dt_s"] - ref["dt_s"]),
    }
    out["w5_fired"] = bool(out["R_meas_rel_diff"] > W5_REL or out["trace_a_rel_maxdiff"] > W5_REL)
    return out


def run_relabel(out_path: str, z_json: str | None) -> dict:
    results = _provenance(False)
    results["sublane"] = "L1-R relabel identity"
    cache = _RunCache()
    n_steps = model_cell("S", AXES["z"], pinned=False)["n_steps"]
    results["n_steps"] = n_steps
    t0 = time.time()
    arms = {}
    for axis in ("z", "x", "y"):
        arms[axis] = relabel_arm(AXES[axis], cache, n_steps)
        a = arms[axis]
        print(f"L1-R {axis} ({a['comp']}): shape={a['grid_shape']} dt={a['dt_s']:.6e} R_meas={a['R_meas']:.6e} "
              f"model={a['R_model']:.6e} fired={a['fired']} gates_hold={a['gates_hold']}", flush=True)
    results["arms"] = arms
    results["compare"] = {axis: relabel_compare(arms["z"], arms[axis]) for axis in ("x", "y")}
    for axis, c in results["compare"].items():
        print(f"L1-R {axis} vs z: trace_a rel={c['trace_a_rel_maxdiff']:.3e} bit={c['trace_a_bit_identical']} "
              f"trace_b rel={c['trace_b_rel_maxdiff']:.3e} bit={c['trace_b_bit_identical']} "
              f"R_meas rel={c['R_meas_rel_diff']:.3e} dt_diff={c['dt_diff_s']:.3e} W5_fired={c['w5_fired']}", flush=True)
    if z_json:
        with open(z_json) as fh:
            zres = json.load(fh)
        ref = zres["cells"]["S"].get("relabel_reference")
        if ref is not None:
            ta = np.asarray(arms["z"]["trace_a"])
            tz = np.asarray(ref["trace_a"])
            results["z_bridge"] = {
                "R_meas_e1_b": ref["R_meas"], "R_meas_b_sym": arms["z"]["R_meas"],
                "rel_diff": abs(arms["z"]["R_meas"] - ref["R_meas"]) / ref["R_meas"],
                "trace_a_rerun_bit_identical": bool(np.array_equal(ta, tz)),
                "trace_a_rerun_rel_maxdiff": float(np.max(np.abs(ta - tz)) / np.max(np.abs(tz))),
                "trace_b_e1_vs_sym_rel_maxdiff_before_gate": float(
                    np.max(np.abs(np.asarray(ref["trace_b_e1"])[:arms["z"]["gates_ns"]["gate_steps"]]
                                  - np.asarray(arms["z"]["trace_b_sym"])[:arms["z"]["gates_ns"]["gate_steps"]]))
                    / np.max(np.abs(np.asarray(ref["trace_b_e1"])))),
            }
            zb = results["z_bridge"]
            print(f"L1-R z bridge: R_meas(E1 B)={zb['R_meas_e1_b']:.6e} R_meas(B_sym)={zb['R_meas_b_sym']:.6e} "
                  f"rel={zb['rel_diff']:.3e} rerun_bit_identical={zb['trace_a_rerun_bit_identical']} "
                  f"B traces before gate rel={zb['trace_b_e1_vs_sym_rel_maxdiff_before_gate']:.3e}", flush=True)
    results["fdtd_runs_new"] = cache.n_new
    results["wallclock_s"] = time.time() - t0
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", out_path)
    return results


# --- pin bridge on z ------------------------------------------------------------------------
def run_pin_bridge(out_path: str, z_json: str) -> dict:
    results = _provenance(False)
    results["sublane"] = "W5 pin bridge: pattern S pinned on z vs unpinned L1-0"
    cache = _RunCache()
    t0 = time.time()
    cell = run_cell("S", AXES["z"], True, cache)
    with open(z_json) as fh:
        zres = json.load(fh)
    unp = {a["n_b"]: a for a in zres["cells"]["S"]["bands"]}
    rows = []
    for rec in cell["bands"]:
        u = unp[rec["n_b"]]
        d = abs(rec["meas"]["R_meas"] - u["meas"]["R_meas"])
        rows.append({"n_b": rec["n_b"], "R_meas_pinned": rec["meas"]["R_meas"],
                     "R_meas_unpinned": u["meas"]["R_meas"], "abs_diff": d,
                     "rel_diff": d / u["meas"]["R_meas"], "fired": bool(d > W5_PIN_ABS),
                     "R_model_pinned": rec["R_model"], "R_model_unpinned": u["R_model"],
                     "R_model_rel_diff": abs(rec["R_model"] - u["R_model"]) / u["R_model"]})
        print(f"pin bridge n_b={rec['n_b']}: pinned={rec['meas']['R_meas']:.6e} unpinned={u['meas']['R_meas']:.6e} "
              f"abs_diff={d:.3e} fired={rows[-1]['fired']}", flush=True)
    results["cell"] = cell
    results["rows"] = rows
    results["any_fired"] = any(r["fired"] for r in rows)
    results["fdtd_runs_new"] = cache.n_new
    results["wallclock_s"] = time.time() - t0
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", out_path)
    return results


# --- W4 control against E1 ------------------------------------------------------------------
def w4_control(z_cell: dict, e1_json: dict) -> dict:
    e1 = e1_json["cells"]["N30_r1.4"]
    rows = []
    s = z_cell["singles"]["L"]
    rows.append({"arm": "single", "R_meas": s["meas"]["R_meas"], "R_meas_e1": e1["single"]["R_meas"],
                 "R_model": s["R_model"], "R_model_e1": e1["single"]["R_model"]})
    e1_arms = {a["n_b"]: a for a in e1["arms"]}
    for a in z_cell["bands"]:
        rows.append({"arm": f"n_b={a['n_b']}", "R_meas": a["meas"]["R_meas"], "R_meas_e1": e1_arms[a["n_b"]]["R_meas"],
                     "R_model": a["R_model"], "R_model_e1": e1_arms[a["n_b"]]["R_model"]})
    for r in rows:
        r["meas_rel"] = abs(r["R_meas"] - r["R_meas_e1"]) / r["R_meas_e1"]
        r["model_rel"] = abs(r["R_model"] - r["R_model_e1"]) / r["R_model_e1"]
        r["meas_bit_identical"] = r["R_meas"] == r["R_meas_e1"]
        r["meas_fired"] = bool(r["meas_rel"] > W4_MEAS_REL)
        r["model_fired"] = bool(r["model_rel"] > W4_MODEL_REL)
    c = z_cell["verdicts"]
    return {"rows": rows, "any_fired": any(r["meas_fired"] or r["model_fired"] for r in rows),
            "c_model_diff_m": abs(c["c_model_m"] - e1["c_model_m"]),
            "c_meas_diff_m": abs(c["c_meas_m"] - e1["c_meas_m"]),
            "e1_git_sha": e1_json["git_sha"]}


# --- note tables ----------------------------------------------------------------------------
def note_tables(model_path: str) -> str:
    with open(model_path) as fh:
        m = json.load(fh)
    lines = []
    lines.append("### Table S — settings per pattern (z; x/y differ by the pin cells only)\n")
    lines.append("| pattern | c_L (mm) | c_R (mm) | d_min (mm) | dt (s) | n_steps | k_src/k_prb | n_lead/n_tail/n_B | coarsest cells/lambda0 | stopband (GHz) | -54 dB class |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in PATTERNS:
        c = m["axes"]["z"][p]
        lb = c["layout_band"]
        lines.append(f"| {p} | {c['c_l_m']*1e3:.3f} | {c['c_r_m']*1e3:.3f} | {c['d_min_m']*1e3:.2f} | {c['dt_s']:.6e} | "
                     f"{c['n_steps']} | {lb['k_src']}/{lb['k_prb']} | {lb['n_lead']}/{lb['n_tail']}/{lb['n_B']} | "
                     f"{c['runways']['coarsest_cells_per_lambda0']:.2f} | {c['runways']['stopband_edge_ghz_coarsest']} | "
                     f"{'inside' if c['in_accuracy_class'] else 'outside'} |")
    lines.append("\n### Table L — law side per pattern (chain model, frozen; identical on x, y, z)\n")
    lines.append("| pattern | R_L | R_R | R_L + R_R | bound 1.05 (R_L + R_R) | k_g (1/mm) | c_model (mm) | c closed form (mm) | closed - model (mm) | c window +/- (mm) | chain max / (R_L+R_R) | chain min (0..80) | |R_L - R_R| |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in FP_PATTERNS:
        L = m["axes"]["z"][p]["law"]
        lines.append(f"| {p} | {L['R_L']:.4e} | {L['R_R']:.4e} | {L['R_L']+L['R_R']:.4e} | {L['bound_sum']:.4e} | "
                     f"{L['k_g_fine_per_m']/1e3:.5f} | {L['c_model_m']*1e3:.3f} | {L['c_closed_form_m']*1e3:.3f} | "
                     f"{(L['c_closed_form_m']-L['c_model_m'])*1e3:+.3f} | {L['c_window_m']*1e3:.3f} | "
                     f"{L['chain_max_over_sum']:.4f} | {L['chain_min_0_80']:.4e} | {L['abs_RL_minus_RR']:.4e} |")
    T = m["axes"]["z"]["T"]["law"]
    lines.append(f"\nPattern T amplitudes (chain, dt = {m['axes']['z']['T']['dt_s']:.6e} s): "
                 + ", ".join(f"{k} = {v:.4e}" for k, v in T["amplitudes"].items())
                 + f"; sum = {sum(T['amplitudes'].values()):.4e}; bound 1.05 x sum = {T['bound_sum']:.4e}.\n")
    lines.append("### Table R — per-arm chain-model predictions and frozen W1 windows (z; the same R_model on x and y)\n")
    lines.append("| pattern | arm | cells (z) | builder exact | R_model | dB | window (frozen) | W2 bound | fp form (two-amp) | min gate margin z / x / y (ns) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in PATTERNS:
        cz = m["axes"]["z"][p]
        for name, rec in list(cz["singles"].items()):
            mm_ = [min(m["axes"][ax][p]["singles"][name]["gate_margins_ns"].values()) for ax in ("z", "x", "y")]
            lines.append(f"| {p} | single {name} | {rec['n_cells']} | {rec['builder_matches_declared_vector']} | "
                         f"{rec['R_model']:.4e} | {rec['R_model_db']:.1f} | [{rec['window'][0]:.4e}, {rec['window'][1]:.4e}] | — | — | "
                         f"{mm_[0]:.3f} / {mm_[1]:.3f} / {mm_[2]:.3f} |")
        for i, rec in enumerate(cz["bands"]):
            mm_ = [min(m["axes"][ax][p]["bands"][i]["gate_margins_ns"].values()) for ax in ("z", "x", "y")]
            fp = f"{rec['fp_form_model']:.4e}" if "fp_form_model" in rec else "—"
            lines.append(f"| {p} | {rec['name']} | {rec['n_cells']} | {rec['builder_matches_declared_vector']} | "
                         f"{rec['R_model']:.4e} | {rec['R_model_db']:.1f} | [{rec['window'][0]:.4e}, {rec['window'][1]:.4e}] | "
                         f"{rec['bound_sum']:.4e} | {fp} | {mm_[0]:.3f} / {mm_[1]:.3f} / {mm_[2]:.3f} |")
    lines.append("\n### Table G — gate margins (ns) per arm and axis, the five E1 margins plus the direct B far-end return (all must be > 0 before the run)\n")
    keys = ["reflection_inside", "inner_return_inside", "run_covers_gate", "b_pin_return_after_gate",
            "incident_inside", "b_far_return_direct_after_gate"]
    lines.append("| axis | pattern | arm | " + " | ".join(keys) + " | gates_hold |")
    lines.append("|---|---|---|" + "---|" * len(keys) + "---|")
    for ax in ("z", "x", "y"):
        for p in PATTERNS:
            c = m["axes"][ax][p]
            for name, rec in list(c["singles"].items()) + [(r["name"], r) for r in c["bands"]]:
                lines.append(f"| {ax} | {p} | {name} | " + " | ".join(f"{rec['gate_margins_ns'][k]:.3f}" for k in keys)
                             + f" | {rec['gates_hold']} |")
    allm = [rec["gate_margins_ns"][k] for ax in ("z", "x", "y") for p in PATTERNS
            for rec in list(m["axes"][ax][p]["singles"].values()) + m["axes"][ax][p]["bands"] for k in keys]
    lines.append(f"\nSmallest margin over every arm and axis: {min(allm):.4f} ns (expectation >= {W6_EXPECT_MIN_MARGIN_NS} ns).")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-only", action="store_true")
    ap.add_argument("--axis", choices=list(AXES))
    ap.add_argument("--patterns", default=",".join(PATTERNS))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--relabel", action="store_true")
    ap.add_argument("--pin-bridge", action="store_true")
    ap.add_argument("--z-json", default=None)
    ap.add_argument("--note-tables", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.note_tables:
        print(note_tables(args.note_tables))
        return
    if args.model_only:
        run_model_only(args.out or "validation/research/multiband_nu/results/e5_model.json")
        return
    if args.relabel:
        run_relabel(args.out or "validation/research/multiband_nu/results/e5_relabel.json", args.z_json)
        return
    if args.pin_bridge:
        if not args.z_json:
            ap.error("--pin-bridge needs --z-json")
        run_pin_bridge(args.out or "validation/research/multiband_nu/results/e5_pinbridge.json", args.z_json)
        return
    if not args.axis:
        ap.error("one of --model-only / --axis / --relabel / --pin-bridge / --note-tables")
    patterns = [p for p in args.patterns.split(",") if p]
    run_axis(args.axis, patterns, args.out or f"validation/research/multiband_nu/results/e5_{args.axis}.json",
             args.resume)


if __name__ == "__main__":
    main()
