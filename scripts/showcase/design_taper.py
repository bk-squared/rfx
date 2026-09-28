"""The WR-90 dielectric-taper design loop at dx = a/36, every iterate recorded (issue 1359).

A TE10 wave in WR-90 (22.86 x 10.16 mm) meets an eps_r = 9 load that fills the
guide and runs into the absorber.  Thirty sections of graded dielectric ahead of
the load are the design variables, and the objective is the band-mean reflected
power <|S11|^2> over the 21 bins of 8.2-12.4 GHz from
``Simulation.compute_waveguide_s_matrix`` (``normalize=False``).  The model and
the optimiser are those of ``validation/tmtt_paper/waveguide_dielectric_taper.py``
with ``SMOKE=0`` (30 sections, 120 periods, 24 CPML layers,
``checkpoint_segments`` 120, Adam lr 0.2 for 120 steps from the flat eps_r = 5
block), imported by path and never edited.  Only the mesh changes: dx = a/36 =
0.635 mm divides both WR-90 walls, and every axial length is a whole number of
cells (the module's metre values rounded to this mesh).

The Klopfenstein baseline is built here (the module prints a Klopfenstein number
but carries no Klopfenstein code): ln Z_TE10 at the band centre follows
Klopfenstein's profile along electrical length, sampled at the centres of the
same 30 sections, with its one free parameter A chosen by the closed-form
section-cascade model on the same 21 bins.

Rules: ``docs/design_notes/20260928_design_films_predeclaration.md``, section
"Taper".  Stages, run in this order by ``scripts/vessl_showcase_design_taper.yaml``:

    python scripts/showcase/design_taper.py --stage timing   --out DIR
    python scripts/showcase/design_taper.py --stage main     --out DIR
    JAX_ENABLE_X64=1 python scripts/showcase/design_taper.py --stage x64fd --out DIR  # if DIR/x64_needed.json
    python scripts/showcase/design_taper.py --stage resolve  --out DIR
    python scripts/showcase/design_taper.py --stage finalize --out DIR
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))       # this checkout's rfx, not an installed one
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _record  # noqa: E402
import _design_common as dc  # noqa: E402

CASE_ID = "design-taper"
QUESTION = ("How do the 30 section permittivities, |S11| across 8.2-12.4 GHz and the "
            "gradient of the band-mean reflected power move together, iterate by iterate, "
            "when Adam designs a WR-90 dielectric taper through the FDTD solve at dx = a/36?")
MODULE = REPO / "validation" / "tmtt_paper" / "waveguide_dielectric_taper.py"
C0 = 299_792_458.0

# ------------------------------------------------------------ pre-declared
A_OVER_DX = 36                  # dx = a / 36 = 0.635 mm, commensurate with 22.86 x 10.16 mm
# The module's SMOKE=0 metre lengths, rounded to whole cells of a/36:
# 0.220 -> 346, 0.030 -> 47, 0.040 -> 63, 0.060 -> 94, 0.140 -> 220.
CELLS_A36 = {"domain": 346, "port_offset": 47, "ref_offset": 63,
             "taper_x0": 94, "fill_x": 220}
# Two settings move from the module's, each because the current code's own run-time
# advisory asks for it (pre-declaration, "Taper", deviations); the module's values
# are re-solved on the final design and reported beside them.
CPML_A36 = 72                   # module: 24 = 0.25 lambda_g at 8.2 GHz; 72 = 0.75 lambda_g
CPML_MODULE = 24
PERIODS = {1: 171, 2: 273}      # module: 120; T / tau_far >= 5 at a/36 and >= 8 at a/72
PERIODS_MODULE = 120
FINE_REFINE = 2                 # the re-solve mesh: a/72 = 0.3175 mm
FD_SECTIONS = {"section_1": 0, "section_16": 15, "section_30": 29}   # 0-based indices
FD_STEPS = (0.1, 0.05, 0.025)   # in eps_r
FD_JUDGED_STEP = 0.05
FD_REL_BAR = 0.05
FD_JUDGE_FRAC = 0.1
WITNESS_FACTOR = 1.5
WITNESS_TOL = 0.05
START_EPS = 5.0                 # the module's start: theta = 0, 1 + 8 * sigmoid(0)
KLOP_A_STEP = 0.25              # grid over A before the bounded refinement
KLOP_N_ELEC = 8001              # samples of the continuous profile along electrical length

_GEOMETRY_GLOBALS = ("DX_M", "CPML_LAYERS", "DOMAIN_X", "PORT_OFFSET", "REF_OFFSET",
                     "TAPER_X0", "FILL_X", "A_WG_REALIZED", "B_WG_REALIZED",
                     "F_CUTOFF_TE10", "Simulation")


def load_module(name: str = "_showcase_taper_module"):
    """The T-MTT taper module with its paper constants (``SMOKE=0``)."""
    if name in sys.modules:
        return sys.modules[name]
    os.environ["SMOKE"] = "0"
    spec = importlib.util.spec_from_file_location(name, MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def section_edges_cells(refine: int = 1) -> np.ndarray:
    """Section edges in cells from the domain's x = 0, on the a/(36 refine) mesh.

    The a/36 edges are the module's ``design_layout`` rule
    (``round(linspace(i_t0, i_fill, N + 1))``); the fine mesh doubles them, so
    each section keeps its physical extent instead of being re-rounded."""
    e = np.round(np.linspace(CELLS_A36["taper_x0"], CELLS_A36["fill_x"], 30 + 1)).astype(int)
    return refine * e


def _segments_for(n_steps: int, k_max: int) -> int:
    """The largest divisor of ``n_steps`` that is <= ``k_max``."""
    for k in range(min(k_max, n_steps), 0, -1):
        if n_steps % k == 0:
            return k
    return 1


class TaperModel:
    """The module's two-port guide on the a/(36 refine) mesh."""

    def __init__(self, mod, refine: int = 1, precision: str = "float32",
                 cpml_a36: int = CPML_A36):
        from rfx import Simulation
        from rfx.grid import cells_spanning
        self.mod, self.refine, self.precision = mod, refine, precision
        self.cpml_layers = cpml_a36 * refine
        dx = mod.A_WG / (A_OVER_DX * refine)
        saved = {k: getattr(mod, k) for k in _GEOMETRY_GLOBALS}
        try:
            mod.DX_M = dx
            mod.CPML_LAYERS = self.cpml_layers
            mod.DOMAIN_X = CELLS_A36["domain"] * refine * dx
            mod.PORT_OFFSET = CELLS_A36["port_offset"] * refine * dx
            mod.REF_OFFSET = CELLS_A36["ref_offset"] * refine * dx
            mod.TAPER_X0 = CELLS_A36["taper_x0"] * refine * dx
            mod.FILL_X = CELLS_A36["fill_x"] * refine * dx
            # the module's ceil(declared / dx) reads one ulp of 10.16 / 0.635 as a
            # 17th cell; the grid itself counts with cells_spanning (#1070)
            mod.A_WG_REALIZED = cells_spanning(mod.A_WG, dx) * dx
            mod.B_WG_REALIZED = cells_spanning(mod.B_WG, dx) * dx
            mod.F_CUTOFF_TE10 = C0 / (2.0 * mod.A_WG_REALIZED)
            if precision == "float64":
                mod.Simulation = functools.partial(Simulation, precision="float64")
            self.sim = mod.build_sim()
            self.grid = self.sim._build_grid()
            layout = mod.design_layout(self.grid)
        finally:
            for k, v in saved.items():
                setattr(mod, k, v)
        g = self.grid
        self.dx = float(g.dx)
        # the guide this mesh built must be WR-90 itself, not one rounded up
        built = {}
        for ax, name, declared in ((1, "y", mod.A_WG), (2, "z", mod.B_WG)):
            n = (g.ny, g.nz)[ax - 1] - 1 - g.face_pads[2 * ax] - g.face_pads[2 * ax + 1]
            built[name] = n * self.dx
            if abs(built[name] - declared) > 1e-12:
                raise ValueError(f"{name}: dx = {self.dx * 1e3:.4f} mm builds {n} cells = "
                                 f"{built[name] * 1e3:.4f} mm, not WR-90's {declared * 1e3:.4f} mm")
        self.walls_m = (built["y"], built["z"])
        self.pad = int(g.pad_x_lo)
        self.edges = self.pad + section_edges_cells(refine)
        self.i_fill = self.pad + refine * CELLS_A36["fill_x"]
        if refine == 1 and not (np.array_equal(layout["edges"], self.edges)
                                and layout["i_fill"] == self.i_fill):
            raise ValueError(f"section edges {self.edges} / fill {self.i_fill} differ from the "
                             f"module's design_layout {layout['edges']} / {layout['i_fill']}")
        self.freqs = np.asarray(mod.FREQS_HZ, dtype=float)
        self.eps_load = float(mod.EPS_LOAD)

    # ---- geometry, in mm from the domain's x = 0
    def section_edges_mm(self) -> np.ndarray:
        return (self.edges - self.pad) * self.dx * 1e3

    def n_steps(self, num_periods: float, segments: int | None) -> tuple[int, int | None]:
        """The module's record length: ``num_timesteps`` rounded up to a
        multiple of ``segments`` when the scan is checkpointed."""
        if segments is None:
            return int(self.grid.num_timesteps(num_periods=num_periods)), None
        return self.mod._checkpoint_n_steps(self.grid, num_periods, segments)

    def eps_grid(self, eps_sec):
        """The module's ``make_eps_builder``, taking the section eps_r directly."""
        import jax.numpy as jnp
        dtype = jnp.float64 if self.precision == "float64" else jnp.float32
        er = jnp.ones(self.grid.shape, dtype=dtype)
        for s in range(len(self.edges) - 1):
            lo, hi = int(self.edges[s]), int(self.edges[s + 1])
            er = er.at[lo:hi, :, :].set(jnp.asarray(eps_sec[s], dtype=dtype))
        return er.at[self.i_fill:, :, :].set(self.eps_load)

    def s11(self, eps_sec, n_steps: int, segments: int | None):
        """Complex S11 on the module's 21 bins, the module's call
        (``make_objective``: normalize=False, the full record, the left port)."""
        kw = {"checkpoint_segments": segments} if segments else {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*normalize=False.*")
            r = self.sim.compute_waveguide_s_matrix(n_steps=n_steps, normalize=False,
                                                    eps_override=self.eps_grid(eps_sec), **kw)
        left = list(r.port_names).index("left")
        return r.s_params[left, left, :]

    def preflight(self) -> list[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            issues = self.sim.preflight()
        return [str(s) for s in issues] + [f"warning: {w.message}" for w in caught]


# ------------------------------------------------------------ Klopfenstein
def te10_z(eps_r, f, fc):
    """TE10 wave impedance of a guide filled with ``eps_r``, in units of eta0."""
    return 1.0 / np.sqrt(np.asarray(eps_r, dtype=float) - (fc / f) ** 2)


def klopfenstein_lnz(u, A: float, z1: float, z2: float):
    """Klopfenstein's ln Z at normalized position ``u`` in [0, 1] (Pozar 5.9):
    ln Z = 1/2 ln(z1 z2) + (Gamma0 / cosh A) A^2 phi(2u - 1, A),
    phi(x, A) = int_0^x I1(A sqrt(1 - y^2)) / (A sqrt(1 - y^2)) dy,  Gamma0 = 1/2 ln(z2 / z1)."""
    from scipy.special import i1
    from scipy.integrate import cumulative_trapezoid
    g0 = 0.5 * np.log(z2 / z1)
    y = np.linspace(-1.0, 1.0, 20001)
    r = A * np.sqrt(np.clip(1.0 - y * y, 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        integrand = np.where(r > 1e-12, i1(r) / np.where(r > 1e-12, r, 1.0), 0.5)
    cum = cumulative_trapezoid(integrand, y, initial=0.0)
    cum -= np.interp(0.0, y, cum)                    # phi(0) = 0
    phi = np.interp(2.0 * np.asarray(u) - 1.0, y, cum)
    return 0.5 * np.log(z1 * z2) + g0 / np.cosh(A) * A * A * phi


def klopfenstein_sections(A: float, centres_m, length_m: float, f0: float, fc: float,
                          eps_load: float, n: int = KLOP_N_ELEC):
    """Section eps_r of the Klopfenstein taper laid along electrical length at f0.

    With Z in units of eta0, beta(f0) = k0 / Z, so the position of normalized
    electrical length s is x(s) = L int_0^s Z ds' / int_0^1 Z ds'.  Each section
    takes the profile's eps_r at its centre."""
    from scipy.integrate import cumulative_trapezoid
    z1, z2 = te10_z(1.0, f0, fc), te10_z(eps_load, f0, fc)
    s = np.linspace(0.0, 1.0, n)
    z = np.exp(klopfenstein_lnz(s, A, z1, z2))
    eps = 1.0 / z ** 2 + (fc / f0) ** 2
    cum = cumulative_trapezoid(z, s, initial=0.0)
    x = length_m * cum / cum[-1]
    return np.interp(np.asarray(centres_m), x, eps), {"s": s, "x_m": x, "eps_r": eps}


def cascade_s11(eps_sec, widths_m, freqs, fc: float, eps_load: float) -> np.ndarray:
    """Closed-form S11 of homogeneous full-cross-section sections between a
    vacuum-filled guide and an eps_load-filled guide (TE10, no higher modes:
    a transversely uniform section cannot couple TE10 to any other mode)."""
    out = []
    for f in np.atleast_1d(freqs):
        k0 = 2.0 * np.pi * f / C0
        zin = te10_z(eps_load, f, fc)
        for er, w in zip(np.asarray(eps_sec)[::-1], np.asarray(widths_m)[::-1]):
            zs = te10_z(er, f, fc)
            t = np.tan(k0 / zs * w)                 # beta = k0 / Z (Z in eta0)
            zin = zs * (zin + 1j * zs * t) / (zs + 1j * zin * t)
        z0 = te10_z(1.0, f, fc)
        out.append((zin - z0) / (zin + z0))
    return np.asarray(out)


def passband_A(centres_m, widths_m, length_m, f0, flo, fc, eps_load, iters: int = 200):
    """The textbook choice: A equal to the taper's electrical length at the
    lowest band frequency (passband edge at flo), found by fixed point."""
    A = 3.0
    for _ in range(iters):
        eps, _ = klopfenstein_sections(A, centres_m, length_m, f0, fc, eps_load)
        theta = float(np.sum(2.0 * np.pi * flo / C0
                             * np.sqrt(np.maximum(eps - (fc / flo) ** 2, 0.0)) * widths_m))
        if abs(theta - A) < 1e-9:
            break
        A = 0.5 * (A + theta)
    return A


def design_klopfenstein(model: TaperModel) -> dict:
    """Klopfenstein's profile on the model's sections, A by the cascade model."""
    from scipy.optimize import minimize_scalar
    edges_m = model.section_edges_mm() * 1e-3
    centres, widths = 0.5 * (edges_m[:-1] + edges_m[1:]), np.diff(edges_m)
    length = float(edges_m[-1] - edges_m[0])
    fc = C0 / (2.0 * model.walls_m[0])
    f0, flo = float(model.freqs.mean()), float(model.freqs.min())

    def cost(A):
        eps, _ = klopfenstein_sections(A, centres - edges_m[0], length, f0, fc, model.eps_load)
        s = cascade_s11(eps, widths, model.freqs, fc, model.eps_load)
        return float(np.mean(np.abs(s) ** 2))

    a_pb = passband_A(centres - edges_m[0], widths, length, f0, flo, fc, model.eps_load)
    grid = np.arange(0.0, a_pb + KLOP_A_STEP, KLOP_A_STEP)
    table = np.array([cost(a) for a in grid])
    i = int(np.argmin(table))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(cost, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
    A = float(res.x) if res.fun <= table[i] else float(grid[i])
    eps, prof = klopfenstein_sections(A, centres - edges_m[0], length, f0, fc, model.eps_load)
    s_casc = cascade_s11(eps, widths, model.freqs, fc, model.eps_load)
    eps_pb, _ = klopfenstein_sections(a_pb, centres - edges_m[0], length, f0, fc, model.eps_load)
    s_pb = cascade_s11(eps_pb, widths, model.freqs, fc, model.eps_load)
    return {
        "A": A, "gamma_m": float(0.5 * abs(np.log(te10_z(model.eps_load, f0, fc)
                                                   / te10_z(1.0, f0, fc))) / np.cosh(A)),
        "A_passband_rule": a_pb, "A_grid": grid, "cascade_cost_by_A": table,
        "eps_sec": eps, "cascade_s11": s_casc, "cascade_cost": float(np.mean(np.abs(s_casc) ** 2)),
        "eps_sec_passband_rule": eps_pb,
        "cascade_cost_passband_rule": float(np.mean(np.abs(s_pb) ** 2)),
        "profile_x_m": prof["x_m"] + edges_m[0], "profile_eps_r": prof["eps_r"],
        "f0_hz": f0, "flo_hz": flo, "fc_te10_hz": fc, "length_m": length,
        "centres_m": centres, "widths_m": widths,
        "how": ("ln Z_TE10(f0) along normalized electrical length by Klopfenstein (Pozar 5.9); "
                "x(s) from beta(f0) = k0/Z; eps_r sampled at section centres; A minimizes the "
                "closed-form cascade's mean |S11|^2 over the 21 bins, grid step "
                f"{KLOP_A_STEP} on [0, A_passband] then bounded refinement"),
    }


# ------------------------------------------------------------------ stages
def _objectives(model: TaperModel, n_steps: int, segments: int | None):
    import jax
    import jax.numpy as jnp
    eps_load = model.eps_load

    def j_eps(eps_sec):
        s = model.s11(eps_sec, n_steps, segments)
        return jnp.mean(jnp.abs(s) ** 2), s

    def j_theta(theta):
        return j_eps(1.0 + (eps_load - 1.0) * jax.nn.sigmoid(theta))

    return j_eps, j_theta


def _db20(x):
    return float(20.0 * np.log10(max(float(x), 1e-30)))


def _summary(s11) -> dict:
    a = np.abs(np.asarray(s11))
    return {"mean_abs_s11_db": _db20(a.mean()), "max_abs_s11_db": _db20(a.max()),
            "mean_abs_s11_sq": float(np.mean(a ** 2)),
            "mean_abs_s11_sq_db": float(10.0 * np.log10(np.mean(a ** 2)))}


def stage_timing(out: Path, a) -> int:
    import jax
    import jax.numpy as jnp
    import optax
    mod = load_module()
    t = {}
    t0 = time.perf_counter()
    model = TaperModel(mod, 1)
    t["build_s"] = time.perf_counter() - t0
    n_steps, k = model.n_steps(a.periods or PERIODS[1], mod.CHECKPOINT_SEGMENTS)
    pre = model.preflight()
    dc.save_json(out / "timing_preflight_a36.json", pre)
    j_eps, j_theta = _objectives(model, n_steps, k)
    eps0 = np.full(30, START_EPS, dtype=np.float32)
    fwd = jax.jit(lambda e: j_eps(e)[0])
    for tag in ("forward_first_s", "forward_second_s"):
        t0 = time.perf_counter()
        jax.block_until_ready(fwd(eps0))
        t[tag] = time.perf_counter() - t0
    vg = jax.jit(jax.value_and_grad(j_theta, has_aux=True))
    theta = jnp.zeros(30, dtype=jnp.float32)
    opt = optax.adam(mod.LR)
    state = opt.init(theta)
    t["grad_s"] = []
    for it in range(a.iters):
        t0 = time.perf_counter()
        (v, s), g = vg(theta)
        jax.block_until_ready(g)
        t["grad_s"].append(time.perf_counter() - t0)
        updates, state = opt.update(g, state)
        theta = optax.apply_updates(theta, updates)
        dc.log(f"timing iter {it}: J {float(v):.4e} ({t['grad_s'][-1]:.1f} s)")
    t["memory_after_grads"] = dc.device_memory()
    fine = TaperModel(mod, FINE_REFINE)
    n72, _ = fine.n_steps(a.periods or PERIODS[FINE_REFINE], None)
    fwd72 = jax.jit(lambda e: jnp.mean(jnp.abs(fine.s11(e, n72, None)) ** 2))
    t0 = time.perf_counter()
    jax.block_until_ready(fwd72(eps0))
    t["forward_a72_first_s"] = time.perf_counter() - t0
    t["memory_end"] = dc.device_memory()
    grad_steady = t["grad_s"][-1]
    fwd_steady = t["forward_second_s"]
    # main: FD ladder (3 x 3 x 2 forwards), Klopfenstein/bare/flat forwards, the
    # witness (a 1.0x and a 1.5x gradient), N Adam gradients and the final forward;
    # resolve: two a/36 forwards (fresh process: first-call cost) and two a/72.
    proj = {"main_s": (18 + 4) * fwd_steady + (mod.N_ADAM + 1 + 1.5) * grad_steady
            + t["forward_first_s"] + t["grad_s"][0],
            "resolve_s": 5 * t["forward_first_s"] + 3 * t["forward_a72_first_s"]}
    proj["total_gpu_h"] = (proj["main_s"] + proj["resolve_s"]) / 3600.0
    proj["x64fd_note"] = "not included: runs only if the round-off rule fires"
    dc.save_json(out / "timing.json", {
        "grid_a36": model.grid.shape, "grid_a72": fine.grid.shape, "n_steps_a36": n_steps,
        "checkpoint_segments": k, "n_steps_a72": n72, "iters_timed": a.iters,
        "timings": t, "projection": proj, "device": str(jax.devices()[0]),
        "formula": "main = 22 forwards + (N_ADAM + 2.5) gradients at steady state + first-call "
                   "extras; resolve = 5 first-call a/36 forwards + 3 first-call a/72 forwards "
                   "(the module-setting a/36 forwards are shorter; counted as full)"})
    dc.log(f"projection: main {proj['main_s'] / 60:.1f} min, resolve {proj['resolve_s'] / 60:.1f} min, "
           f"total {proj['total_gpu_h']:.2f} GPU-h")
    return 0


def stage_main(out: Path, a) -> int:
    import jax
    import jax.numpy as jnp
    import optax
    from rfx import gradient_record_length_witness
    mod = load_module()
    wall = {}
    model = TaperModel(mod, 1)
    n_steps, k = model.n_steps(a.periods or PERIODS[1], mod.CHECKPOINT_SEGMENTS)
    n_iters = a.iters if a.iters is not None else mod.N_ADAM
    edges_mm = model.section_edges_mm()
    dc.save_json(out / "preflight_a36.json", model.preflight())
    dc.save_json(out / "model.json", {
        "structure": "WR-90 two-port, PEC y/z walls, CPML along x; vacuum | 30 sections | "
                     "eps_r 9 filling to the right absorber (validation/tmtt_paper/"
                     "waveguide_dielectric_taper.py build_sim, SMOKE=0 constants)",
        "module": str(MODULE.relative_to(REPO)), "dx_m": model.dx, "a_over_dx": A_OVER_DX,
        "walls_realized_m": model.walls_m, "cells_a36": CELLS_A36, "cpml_layers": model.cpml_layers,
        "module_cpml_layers": CPML_MODULE, "module_num_periods": PERIODS_MODULE,
        "record_s": n_steps * float(model.grid.dt), "dt_s": float(model.grid.dt),
        "far_path_left_port_m": (CELLS_A36["domain"] - CELLS_A36["port_offset"] + model.cpml_layers)
        * model.dx,
        "tau_far_vacuum_estimate_s": 2.0 * (CELLS_A36["domain"] - CELLS_A36["port_offset"]
                                            + model.cpml_layers) * model.dx
        / (C0 * np.sqrt(1.0 - (C0 / (2.0 * model.walls_m[0]) / model.freqs.min()) ** 2)),
        "grid_shape": model.grid.shape, "pad_x_lo": model.pad,
        "section_edges_index": model.edges, "section_edges_mm": edges_mm,
        "section_widths_cells": np.diff(model.edges), "fill_index": model.i_fill,
        "fill_mm": (model.i_fill - model.pad) * model.dx * 1e3,
        "port_left_mm": CELLS_A36["port_offset"] * model.dx * 1e3,
        "ref_left_mm": CELLS_A36["ref_offset"] * model.dx * 1e3,
        "port_right_mm": (CELLS_A36["domain"] - CELLS_A36["port_offset"]) * model.dx * 1e3,
        "domain_mm": CELLS_A36["domain"] * model.dx * 1e3,
        "freqs_hz": model.freqs, "eps_load": model.eps_load, "num_periods": a.periods or PERIODS[1],
        "n_steps": n_steps, "checkpoint_segments": k,
        "objective": "J = mean over the 21 bins of |S11|^2 (the module's loss)",
        "adam": {"lr": mod.LR, "iters": n_iters, "optax": "optax.adam(lr) defaults",
                 "start": "theta = 0 (eps_r = 5 in every section)",
                 "map": "eps_r = 1 + 8 sigmoid(theta)"},
        "jit": "jax.jit(jax.value_and_grad(J(theta), has_aux=True)); the module does not jit"})

    # ---- Klopfenstein (closed form, no FDTD) ------------------------------
    klop = design_klopfenstein(model)
    dc.save_json(out / "klopfenstein.json", {k_: v for k_, v in klop.items()
                                             if k_ not in ("profile_x_m", "profile_eps_r",
                                                           "cascade_s11")})
    dc.save_npz(out / "klopfenstein_profile.npz", x_m=klop["profile_x_m"],
                eps_r=klop["profile_eps_r"], eps_sec=klop["eps_sec"],
                cascade_s11=klop["cascade_s11"], freqs_hz=model.freqs)
    dc.log(f"Klopfenstein: A {klop['A']:.3f} (passband rule {klop['A_passband_rule']:.2f}), "
           f"cascade J {klop['cascade_cost']:.4e}")

    j_eps, j_theta = _objectives(model, n_steps, k)
    fwd = jax.jit(j_eps)

    # ---- baselines at a/36 --------------------------------------------------
    base = {}
    for name, eps in (("klopfenstein", klop["eps_sec"]), ("bare_step", np.ones(30)),
                      ("flat_start", np.full(30, START_EPS))):
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            J, s = fwd(jnp.asarray(eps, dtype=jnp.float32))
            s = np.asarray(s)
        wall[f"baseline_{name}_s"] = time.perf_counter() - t0
        base[name] = {"eps_sec": np.asarray(eps, dtype=float), "J": float(J), **_summary(s),
                      "run_time_advisories": sorted({str(w.message) for w in caught})}
        base[f"_s11_{name}"] = s
        dc.log(f"baseline {name}: J {float(J):.4e}, mean|S11| {_summary(s)['mean_abs_s11_db']:.2f} dB")
    dc.save_npz(out / "baselines_a36.npz", freqs_hz=model.freqs,
                **{f"s11_{n}": base.pop(f"_s11_{n}") for n in ("klopfenstein", "bare_step", "flat_start")},
                **{f"eps_{n}": base[n]["eps_sec"] for n in ("klopfenstein", "bare_step", "flat_start")})
    dc.save_json(out / "baselines_a36.json", base)

    # ---- AD against central differences at the start ----------------------
    eps0 = np.full(30, START_EPS, dtype=np.float32)
    J_only = jax.jit(lambda e: j_eps(e)[0])
    grad_eps = jax.jit(jax.grad(lambda e: j_eps(e)[0]))
    t0 = time.perf_counter()
    g0 = np.asarray(grad_eps(jnp.asarray(eps0)), dtype=float)
    wall["grad_eps_start_s"] = time.perf_counter() - t0
    ladder = dc.fd_ladder(lambda e: J_only(jnp.asarray(e, dtype=jnp.float32)), eps0,
                          FD_SECTIONS, FD_STEPS)
    ad = {n: float(g0[i]) for n, i in FD_SECTIONS.items()}
    verdict = dc.judge_fd(ad, ladder, FD_STEPS, FD_JUDGED_STEP, FD_REL_BAR, FD_JUDGE_FRAC)
    dc.save_json(out / "fd_start.json", {"precision": "float32", "eps0": eps0, "J0": float(J_only(eps0)),
                                         "variables": FD_SECTIONS, "grad_eps_all": g0,
                                         "ladder": ladder, "judgement": verdict})
    dc.log("AD vs FD at the start: " + "; ".join(
        f"{n}: AD {r['ad']:+.5e} FD {r['fd']:+.5e} rel {r['rel']:.2e}{' judged' if r['judged'] else ''}"
        for n, r in verdict["rows"].items()))
    if verdict["roundoff"]:
        dc.save_json(out / "x64_needed.json", {"rule": "FD moves more between h = 0.05 and 0.025 "
                                                       "than between 0.1 and 0.05",
                                               "variables": verdict["roundoff"]})
        dc.log(f"round-off rule met at {verdict['roundoff']}: x64 stage required")

    # ---- record-length witness at the start ---------------------------------
    def witness_objective(e, n):
        return jnp.mean(jnp.abs(model.s11(e, n, _segments_for(n, mod.CHECKPOINT_SEGMENTS))) ** 2)

    t0 = time.perf_counter()
    w = gradient_record_length_witness(witness_objective, jnp.asarray(eps0), n_steps,
                                       tol=WITNESS_TOL, factor=WITNESS_FACTOR)
    wall["witness_s"] = time.perf_counter() - t0
    g1 = np.asarray(next(iter(w.grad.values())))[0].astype(float)
    g15 = np.asarray(next(iter(w.grad_long.values())))[0].astype(float)
    per_var = {n: (float(abs(g15[i] - g1[i]) / abs(g15[i])) if g15[i] else None)
               for n, i in FD_SECTIONS.items()}
    judged_vars = verdict["judged"]
    wit = {"helper": "rfx.gradient_record_length_witness", "factor": WITNESS_FACTOR, "tol": WITNESS_TOL,
           "n_steps": {"1.0x": w.n_steps, "1.5x": w.n_steps_long},
           "segments": {"1.0x": _segments_for(w.n_steps, mod.CHECKPOINT_SEGMENTS),
                        "1.5x": _segments_for(w.n_steps_long, mod.CHECKPOINT_SEGMENTS)},
           "grad": {"1.0x": g1, "1.5x": g15}, "norm_rel_change": float(w.worst),
           "cosine": float(w.cosine_by_bin[0]), "helper_passed": bool(w.passed),
           "per_variable_rel_change": per_var, "judged_variables": judged_vars,
           "J": {"1.0x": float(np.real(w.value[0])), "1.5x": float(np.real(w.value_long[0]))}}
    wit["passed"] = bool(w.worst <= WITNESS_TOL and all(per_var[n] is not None and per_var[n] <= WITNESS_TOL for n in judged_vars))
    dc.save_json(out / "record_length_witness.json", wit)
    dc.log(f"record-length witness: norm {w.worst:.3e}, per variable "
           + ", ".join(f"{n} {v if v is None else f'{v:.3e}'}" for n, v in per_var.items()) + f" -> passed {wit['passed']}")
    if not wit["passed"]:
        dc.save_json(out / "witness_failed.json", wit)
        if not a.smoke:
            dc.log("record-length witness FAILED: the case stops before the descent")
            return 3
        dc.log("record-length witness FAILED; --smoke: continuing to exercise the descent")

    # ---- Adam, every iterate ------------------------------------------------
    store = dc.IterateStore(out / "iterations.npz", static={"freqs_hz": model.freqs,
                                                            "section_edges_mm": edges_mm})
    vg = jax.jit(jax.value_and_grad(j_theta, has_aux=True))
    opt = optax.adam(mod.LR)
    theta = jnp.zeros(30, dtype=jnp.float32)
    state = opt.init(theta)

    def eps_of(th):
        return 1.0 + (model.eps_load - 1.0) / (1.0 + np.exp(-np.asarray(th, dtype=float)))

    def deps_dtheta(th):
        sg = 1.0 / (1.0 + np.exp(-np.asarray(th, dtype=float)))
        return (model.eps_load - 1.0) * sg * (1.0 - sg)

    for it in range(n_iters):
        t0 = time.perf_counter()
        (J, s), g = vg(theta)
        jax.block_until_ready(g)
        dt = time.perf_counter() - t0
        s = np.asarray(s)
        g = np.asarray(g, dtype=float)
        store.append(theta=np.asarray(theta, dtype=float), eps_sec=eps_of(theta),
                     s11_re=s.real, s11_im=s.imag, abs_s11=np.abs(s), J=float(J),
                     J_db=10.0 * np.log10(float(J)), mean_abs_s11_db=_db20(np.abs(s).mean()),
                     grad_theta=g, grad_eps=g / deps_dtheta(theta), wall_s=dt)
        updates, state = opt.update(jnp.asarray(g, dtype=jnp.float32), state)
        theta = optax.apply_updates(theta, updates)
        store.persist()
        dc.save_npz(out / "adam_state.npz", iterate_next=it + 1, theta_next=np.asarray(theta),
                    **dc.adam_state_arrays(state))
        if it % 5 == 0 or it == n_iters - 1:
            dc.log(f"iter {it:3d} J {float(J):.4e} ({10 * np.log10(float(J)):.2f} dB) ({dt:.1f} s)")
    t0 = time.perf_counter()
    J, s = fwd(jnp.asarray(eps_of(theta), dtype=jnp.float32))
    s = np.asarray(s)
    store.append(theta=np.asarray(theta, dtype=float), eps_sec=eps_of(theta), s11_re=s.real,
                 s11_im=s.imag, abs_s11=np.abs(s), J=float(J), J_db=10.0 * np.log10(float(J)),
                 mean_abs_s11_db=_db20(np.abs(s).mean()), wall_s=time.perf_counter() - t0)
    store.persist()
    wall["adam_total_s"] = float(np.nansum(np.stack(store.rows["wall_s"])))
    dc.save_json(out / "main_wall.json", {**wall, "memory": dc.device_memory()})
    dc.log(f"final iterate {n_iters}: J {float(J):.4e} ({10 * np.log10(float(J)):.2f} dB)")
    return 0


def stage_x64fd(out: Path, a) -> int:
    import jax
    import jax.numpy as jnp
    if not jax.config.read("jax_enable_x64"):
        raise SystemExit("the x64 stage needs JAX_ENABLE_X64=1 in its environment")
    import rfx.simulation as rsim
    mod = load_module()
    model = TaperModel(mod, 1, precision="float64")
    n_steps, k = model.n_steps(a.periods or PERIODS[1], mod.CHECKPOINT_SEGMENTS)
    j_eps, _ = _objectives(model, n_steps, k)
    # compute_waveguide_s_matrix calls rfx.simulation.run without field_dtype
    # (rfx/sources/waveguide_port.py, extract_waveguide_s_matrix), so on this
    # lane Simulation(precision="float64") leaves the fields float32. The run
    # entry is wrapped for this process only: it asks for float64 fields and
    # records the dtype of every field state it returns.
    run_orig, seen = rsim.run, []

    def run_float64(*args, **kw):
        kw["field_dtype"] = jnp.float64
        r = run_orig(*args, **kw)
        seen.append(str(r.state.ex.dtype) if r.state is not None else "no state returned")
        return r

    rsim.run = run_float64
    J_only = jax.jit(lambda e: j_eps(e)[0])
    eps0 = np.full(30, START_EPS, dtype=np.float64)
    g0 = np.asarray(jax.jit(jax.grad(lambda e: j_eps(e)[0]))(jnp.asarray(eps0)), dtype=float)
    ladder = dc.fd_ladder(lambda e: J_only(jnp.asarray(e, dtype=jnp.float64)), eps0,
                          FD_SECTIONS, FD_STEPS)
    ad = {n: float(g0[i]) for n, i in FD_SECTIONS.items()}
    verdict = dc.judge_fd(ad, ladder, FD_STEPS, FD_JUDGED_STEP, FD_REL_BAR, FD_JUDGE_FRAC)
    J0 = float(J_only(eps0))
    rsim.run = run_orig
    dtypes = sorted(set(seen))
    if dtypes != ["float64"]:
        raise SystemExit(f"the float64 repeat saw field dtypes {dtypes}; it is not a float64 record")
    dc.save_json(out / "fd_float64.json", {"precision": "float64", "eps0": eps0,
                                           "J0": J0, "grad_eps_all": g0,
                                           "field_dtypes_seen": dtypes, "n_run_traces_seen": len(seen),
                                           "dtype_note": "recorded when rfx.simulation.run is traced; "
                                                         "the jitted objectives reuse their trace",
                                           "how": "rfx.simulation.run wrapped in this process to pass "
                                                  "field_dtype=float64; JAX_ENABLE_X64=1",
                                           "ladder": ladder, "judgement": verdict})
    dc.log("float64 AD vs FD: " + "; ".join(f"{n}: rel {r['rel']:.2e}" for n, r in verdict["rows"].items()))
    return 0


def stage_resolve(out: Path, a) -> int:
    """The final design, the Klopfenstein taper and the start, read back from the
    record and solved without the optimiser: the same lane at a/36, at a/72, and
    at a/36 with the module's own absorber and record length."""
    import jax.numpy as jnp
    mod = load_module()
    it = np.load(out / "iterations.npz")
    designs = {"final": np.asarray(it["eps_sec"][-1], dtype=float),
               "klopfenstein": np.asarray(dc.load_json(out / "klopfenstein.json")["eps_sec"], dtype=float),
               "flat_start": np.full(30, START_EPS)}
    meshes = (("a36", 1, CPML_A36, PERIODS[1], ("final", "klopfenstein", "flat_start")),
              (f"a{A_OVER_DX * FINE_REFINE}", FINE_REFINE, CPML_A36, PERIODS[FINE_REFINE],
               ("final", "klopfenstein", "flat_start")),
              ("a36_module_absorber_and_record", 1, CPML_MODULE, PERIODS_MODULE,
               ("final", "klopfenstein")))
    res, arrays = {}, {}
    for tag, refine, cpml, periods, names in meshes:
        model = TaperModel(mod, refine, cpml_a36=cpml)
        # the module's record rule (num_timesteps rounded up to a multiple of its
        # checkpoint_segments), so the a/36 solve has the loop's own step count
        n_steps, _ = model.n_steps(a.periods or periods, mod.CHECKPOINT_SEGMENTS)
        dc.save_json(out / f"preflight_{tag}.json", model.preflight())
        res[tag] = {"dx_m": model.dx, "grid_shape": model.grid.shape, "n_steps": n_steps,
                    "num_periods": a.periods or periods, "record_s": n_steps * float(model.grid.dt),
                    "cpml_layers": model.cpml_layers,
                    "section_edges_mm": model.section_edges_mm(), "designs": {}}
        for name in names:
            t0 = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                s = np.asarray(model.s11(jnp.asarray(designs[name], dtype=jnp.float32), n_steps, None))
            res[tag]["designs"][name] = {**_summary(s), "wall_s": time.perf_counter() - t0,
                                         "run_time_advisories": sorted({str(w.message) for w in caught})}
            arrays[f"s11_{tag}_{name}"] = s
            dc.log(f"resolve {tag} {name}: mean|S11| {_summary(s)['mean_abs_s11_db']:.2f} dB, "
                   f"max {_summary(s)['max_abs_s11_db']:.2f} dB")
    dc.save_npz(out / "resolve.npz", freqs_hz=np.asarray(mod.FREQS_HZ, dtype=float), **arrays,
                **{f"eps_{n}": v for n, v in designs.items()})
    dc.save_json(out / "resolve.json", res)
    return 0


def stage_finalize(out: Path, a) -> int:
    x64 = (out / "fd_float64.json").is_file()
    source = _record.source_block(a.repo_dir, precision="float32" + (
        " (the start-point FD check repeated in float64)" if x64 else ""))
    model = dc.load_json(out / "model.json")
    claims, derived = [], []
    wit = dc.load_json(out / "record_length_witness.json")
    fd32 = dc.load_json(out / "fd_start.json")
    if (out / "x64_needed.json").is_file() and not x64:
        raise SystemExit("x64_needed.json is present but fd_float64.json is absent: "
                         "the float64 judgement is not a record, finalize refuses")
    fd = dc.load_json(out / "fd_float64.json") if x64 else fd32
    fd_file = "fd_float64.json" if x64 else "fd_start.json"
    for n, r in fd["judgement"]["rows"].items():
        q = (f"|AD - FD| / |FD| at h = {FD_JUDGED_STEP} in eps_r, dJ/d eps_r of {n}, start "
             f"({fd['precision']})")
        if r["judged"]:
            claims.append(_record.claim(q, r["rel"], "1", fd_file, threshold=FD_REL_BAR,
                                        rule="pre-declared: judged when |FD| >= 0.1 max |FD|"))
        else:
            claims.append(_record.claim(q, r["rel"], "1", fd_file,
                                        note="reported: |FD| < 0.1 of the largest"))
    if x64:
        for n, r in fd32["judgement"]["rows"].items():
            claims.append(_record.claim(f"|AD - FD| / |FD| at h = {FD_JUDGED_STEP}, {n}, start "
                                        "(float32, reported)", r["rel"], "1", "fd_start.json"))
    claims.append(_record.claim("record-length witness ||g_1.5x - g_1.0x|| / ||g_1.5x|| over the "
                                "30 sections, start", wit["norm_rel_change"], "1",
                                "record_length_witness.json", threshold=WITNESS_TOL,
                                rule="pre-declared: factor 1.5, tol 0.05"))
    for n, v in wit["per_variable_rel_change"].items():
        if n in wit["judged_variables"] and v is not None:
            claims.append(_record.claim(f"record-length witness |g_1.5x - g_1.0x| / |g_1.5x|, {n}",
                                        v, "1", "record_length_witness.json", threshold=WITNESS_TOL,
                                        rule="pre-declared: per FD-judged variable"))
        else:
            claims.append(_record.claim(f"record-length witness |g_1.5x - g_1.0x| / |g_1.5x|, {n}",
                                        v, "1", "record_length_witness.json"))
    klop = dc.load_json(out / "klopfenstein.json")
    claims += [
        _record.claim("Klopfenstein A (cascade-model choice)", klop["A"], "1", "klopfenstein.json"),
        _record.claim("Klopfenstein closed-form cascade J", klop["cascade_cost"], "1", "klopfenstein.json"),
    ]
    base = dc.load_json(out / "baselines_a36.json")
    for name in ("klopfenstein", "bare_step", "flat_start"):
        claims.append(_record.claim(f"{name} at a/36: J = mean |S11|^2", base[name]["J"], "1",
                                    "baselines_a36.json"))
        claims.append(_record.claim(f"{name} at a/36: 20 log10 mean |S11|",
                                    base[name]["mean_abs_s11_db"], "dB", "baselines_a36.json"))
    if (out / "iterations.npz").is_file():
        it = np.load(out / "iterations.npz")
        J = np.asarray(it["J"])
        n = len(J) - 1
        claims += [
            _record.claim("J at iterate 0", float(J[0]), "1", "iterations.npz"),
            _record.claim(f"J at iterate {n} (final design, no step)", float(J[-1]), "1", "iterations.npz"),
            _record.claim("20 log10 mean |S11| at iterate 0", float(it["mean_abs_s11_db"][0]), "dB",
                          "iterations.npz"),
            _record.claim(f"20 log10 mean |S11| at iterate {n}", float(it["mean_abs_s11_db"][-1]), "dB",
                          "iterations.npz"),
            _record.claim("median wall time per Adam iteration (jitted value_and_grad)",
                          float(np.median(it["wall_s"][1:-1])), "s", "iterations.npz"),
        ]
        derived.append(_record.derived("reflected power drop, iterate 0 to final",
                                       float(10 * np.log10(J[0] / J[-1])), "dB",
                                       "10 log10(J[0] / J[final])", {"J[0]": float(J[0]),
                                                                    "J[final]": float(J[-1])}))
    if (out / "resolve.json").is_file():
        rs = dc.load_json(out / "resolve.json")
        for tag, r in rs.items():
            for name, d in r["designs"].items():
                claims.append(_record.claim(f"{name}, re-solved at {tag}: 20 log10 mean |S11|",
                                            d["mean_abs_s11_db"], "dB", "resolve.json"))
                claims.append(_record.claim(f"{name}, re-solved at {tag}: max |S11|",
                                            d["max_abs_s11_db"], "dB", "resolve.json"))
                claims.append(_record.claim(f"{name}, re-solved at {tag}: J = mean |S11|^2",
                                            d["mean_abs_s11_sq"], "1", "resolve.json"))
    run = {"platform": "VESSL", "preset": os.environ.get("RFX_SHOWCASE_PRESET"), "run_id": None}
    if (out / "main_wall.json").is_file():
        run["wall_s"] = dc.load_json(out / "main_wall.json")
    result = {"schema": _record.SCHEMA, "id": CASE_ID, "question": QUESTION, "source": source,
              "run": run, "model": model, "claims": claims, "derived": derived,
              "out_of_scope": ["the paper's 0.5 mm and 0.25 mm meshes and their numbers",
                               "other optimisers, other starts, other section counts",
                               "a Klopfenstein taper tuned in the FDTD (its A comes from the "
                               "closed-form cascade)",
                               "higher-order modes: the structure is transversely uniform"]}
    path = _record.write_result(out, result, _record.data_files(out))
    dc.log(f"wrote {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=("timing", "main", "x64fd", "resolve", "finalize"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo-dir", type=Path, default=REPO)
    ap.add_argument("--iters", type=int, default=None,
                    help="Adam iterations (default: the module's N_ADAM; timing default 2)")
    ap.add_argument("--periods", type=float, default=None, help="local smoke runs only")
    ap.add_argument("--smoke", action="store_true",
                    help="local smoke runs only: continue past a failed witness")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "timing" and a.iters is None:
        a.iters = 2
    stage = {"timing": stage_timing, "main": stage_main, "x64fd": stage_x64fd,
             "resolve": stage_resolve, "finalize": stage_finalize}[a.stage]
    if a.stage in ("main", "timing"):
        dc.save_json(a.out / f"source_{a.stage}.json",
                     _record.source_block(a.repo_dir, precision="float32"))
    return stage(a.out, a)


if __name__ == "__main__":
    raise SystemExit(main())
