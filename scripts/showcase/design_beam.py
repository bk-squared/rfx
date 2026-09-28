"""The beam-steering dielectric-cover design loop at lambda/20, every iterate recorded (issue 1359).

An x-directed dipole sits a quarter wavelength above a finite PEC reflector at
3 GHz, under a dielectric cover 1.5 lambda square and lambda/10 thick.  The
cover permittivity is the design: every one of the cover's 31 x 31 x 3 = 2883
cells, eps_r = 1 + 9 sigmoid(psi), as the module optimizes it.  Adam lowers the
module's steering objective (raise the directivity toward theta = 30 deg in the
E-plane while holding the broadside and back lobes down) through the FDTD solve
and the near-to-far-field transform.  The structure, the objective, the record
length, the start (the eps_r 2 -> 9 ramp along x) and the optimiser (Adam lr
0.08, 140 steps) are those of ``validation/tmtt_paper/beam_steering_superstrate.py``
with ``SMOKE=0``, imported by path and never edited.

Rules: ``docs/design_notes/20260928_design_films_predeclaration.md``, section
"Beam" and Amendment 3.  Stages, run in this order by
``scripts/vessl_showcase_design_beam.yaml``:

    python scripts/showcase/design_beam.py --stage timing    --out DIR
    python scripts/showcase/design_beam.py --stage main      --out DIR
    python scripts/showcase/design_beam.py --stage equivalence --out DIR2   # its own job (63 GB)
    JAX_ENABLE_X64=1 python scripts/showcase/design_beam.py --stage x64fd --out DIR  # if DIR/x64_needed.json
    python scripts/showcase/design_beam.py --stage resolve   --out DIR
    python scripts/showcase/design_beam.py --stage keyframes --out DIR
    python scripts/showcase/design_beam.py --stage finalize  --out DIR --equivalence DIR2
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

CASE_ID = "design-beam"
QUESTION = ("How do the cover permittivity, the E-plane directivity pattern and the "
            "gradient move together, iterate by iterate, when Adam steers a reflector-backed "
            "dipole's beam toward 30 degrees through the FDTD solve at lambda/20?")
MODULE = REPO / "validation" / "tmtt_paper" / "beam_steering_superstrate.py"
C0 = 299_792_458.0

# ------------------------------------------------------------ pre-declared
# cover cells (i, j, k) on the centre line y = 0, middle layer: x = -40, 0, +40 mm
FD_CELLS = {"cell_x7_y15_z1": (7, 15, 1), "cell_x15_y15_z1": (15, 15, 1),
            "cell_x23_y15_z1": (23, 15, 1)}
FD_STEPS = (0.2, 0.1, 0.05)              # in eps_r
FD_JUDGED_STEP = 0.1
FD_REL_BAR = 0.05
FD_JUDGE_FRAC = 0.1
WITNESS_FACTOR = 1.5
WITNESS_TOL = 0.05
EQUIV_TOL = 1e-4                         # Amendment 2 condition: module path vs this script's
UNIFORM_EPS = tuple(float(e) for e in range(1, 11))   # 1 = no cover
FINE_REFINE = 2                          # the re-solve mesh: lambda/40
SETTLING_BAR_DB = -40.0                  # the module's format_witness verdict
MATMUL_PRECISION = "highest"             # Amendment 3: no TF32 in float32 matmuls
KEYFRAMES = (0, 7, 21, 49, 84, 140)      # 0, 5, 15, 35, 60 and 100 % of 140
SPHERE_STEP_DEG = 2.0                    # the far-field sphere for the 3-D pattern


def load_module(name: str = "_showcase_beam_module"):
    """The T-MTT beam-steering module with its paper constants (``SMOKE=0``)."""
    import jax
    jax.config.update("jax_default_matmul_precision", MATMUL_PRECISION)
    if name in sys.modules:
        return sys.modules[name]
    os.environ["SMOKE"] = "0"
    spec = importlib.util.spec_from_file_location(name, MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if mod.SMOKE:
        raise RuntimeError("the beam module loaded in SMOKE mode")
    return mod


def module_start(shape) -> tuple[np.ndarray, np.ndarray]:
    """The module's start, its ``main()`` lines: an eps_r 2 -> 9 ramp along x
    over the cover cells, and its latent psi = logit((eps - 1) / 9)."""
    nx = shape[0]
    ramp = np.linspace(2.0, 9.0, nx, dtype=np.float32)
    eps0 = np.broadcast_to(ramp[:, None, None], shape).astype(np.float32)
    frac = np.clip((eps0 - 1.0) / 9.0, 1e-4, 1 - 1e-4)
    return eps0, np.log(frac / (1.0 - frac)).astype(np.float32)


# ------------------------------------------------------------------ problem
class BeamModel:
    """The module's problem.  At refine 1 it is ``build_problem()`` itself; at
    refine r the mesh is dx/r and every position, the domain and the CPML
    thickness are the refine-1 ones (all of them sit on refine-1 nodes, so the
    nodes nest)."""

    def __init__(self, mod, refine: int = 1, precision: str = "float32"):
        self.mod, self.refine, self.precision = mod, refine, precision
        saved = mod.Simulation
        try:
            if precision == "float64":
                from rfx.api import Simulation
                mod.Simulation = functools.partial(Simulation, precision="float64")
            if refine == 1:
                sim, region, grid, plate, lam, f0 = mod.build_problem()
            else:
                sim, region, grid, plate, lam, f0 = build_problem_refined(mod, refine)
        finally:
            mod.Simulation = saved
        self.sim, self.region, self.grid, self.plate, self.lam, self.f0 = sim, region, grid, plate, lam, f0
        self.lo, self.hi, self.shape = mod.resolve_design_indices(region, grid)
        self.dx = float(grid.dx)
        if refine == 1:
            self.start, self.block_shape = tuple(self.lo), tuple(self.shape)
        else:
            # the lambda/20 cover's cells, each written onto the refine^3 fine
            # cells nested in it (the coarse eps_r index i covers [x_i, x_i+1))
            _, c_region, c_grid, *_ = mod.build_problem()
            c_lo, _, c_shape = mod.resolve_design_indices(c_region, c_grid)
            self.start = nested_start(c_lo, c_grid.axis_pads, grid.axis_pads, refine)
            self.block_shape = tuple(refine * n for n in c_shape)

    def n_steps(self, num_periods: float | None = None) -> int:
        return int(self.grid.num_timesteps(num_periods=num_periods or self.mod.NUM_PERIODS))

    def pattern_fn(self, n_steps: int, segments: int | None = "auto"):
        """``pattern(eps_block) -> |E|^2 (THETA, PHI)``: the module's
        ``make_pattern_fn`` line for line, with the forward's segmented
        checkpointing switched on (``segments``, default the largest divisor
        of ``n_steps`` not above sqrt(n_steps)), the JAX far-field transform
        and the float64 arrays of the x64 stage.  ``eps_block`` has
        ``block_shape`` and is written at ``start``."""
        if segments == "auto":
            segments = sqrt_segments(n_steps)
        return _pattern_fn(self.mod, self.sim, self.grid, self.start, n_steps,
                           self.precision, segments)

    def module_pattern_fn(self, n_steps: int):
        """The module's own ``make_pattern_fn`` (float32, per-step checkpoint)."""
        p, _ = self.mod.make_pattern_fn(self.sim, self.grid, self.plate, self.lo, self.hi, n_steps)
        return p

    def preflight(self) -> list[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            issues = self.sim.preflight()
        return [str(s) for s in issues] + [f"warning: {w.message}" for w in caught]

    def describe(self) -> dict:
        g = self.grid
        return {"dx_m": self.dx, "lambda_over_dx": self.lam / self.dx, "grid_shape": g.shape,
                "cpml_layers": g.pad_x_lo, "design_lo": self.lo, "design_hi": self.hi,
                "design_shape": self.shape, "eps_block_start": self.start,
                "eps_block_shape": self.block_shape,
                "plate": {k: v for k, v in self.plate.items()},
                "region_corner_lo_m": self.region.corner_lo, "region_corner_hi_m": self.region.corner_hi,
                "f0_hz": self.f0, "lambda_m": self.lam}


def build_problem_refined(mod, refine: int):
    """``mod.build_problem`` on a mesh ``refine`` times finer, with the refine-1
    positions: the same arithmetic as the module at its own dx, then a
    Simulation at dx/refine with ``10 refine`` CPML layers (the same thickness)."""
    from rfx.geometry.csg import Box
    from rfx.optimize import DesignRegion
    RC = mod.RC
    f0 = 3.0e9
    lam = mod.C0 / f0
    half_lam = lam / 2.0
    freq_max = 4.0e9
    dx1 = lam / mod.DX_FRAC
    cpml1 = 10
    half = float(round(mod.HALF_FRAC * lam / dx1)) * dx1
    slab_thick = dx1 * mod.SLAB_CELLS_Z
    cpml_t = cpml1 * dx1
    m = dx1
    box_z_lo = cpml_t + m
    plate_z = box_z_lo + half_lam + m
    src_z = plate_z + lam / 4.0
    slab_z = src_z + max(0.55 * lam, half_lam)
    box_z_hi = slab_z + slab_thick + half_lam + m
    Lz = float(np.ceil((box_z_hi + m + cpml_t) / dx1) * dx1)
    cx_min = cpml_t + m + half + half_lam
    Lx = float(2 * int(np.ceil(cx_min / dx1)) * dx1)
    Ly = Lx
    cx, cy = Lx / 2.0, Ly / 2.0

    dx = dx1 / refine
    sim = mod.Simulation(freq_max=freq_max, domain=(Lx, Ly, Lz),
                         cpml_layers=cpml1 * refine, dx=dx)
    sim.add_source((cx, cy, float(src_z)), "ex")
    slab_mid_z = float(slab_z + 0.5 * slab_thick)
    sim.add_probe((cx, cy, float(src_z)), "ex")
    sim.add_probe((cx, cy, slab_mid_z), "ex")
    sim.add_probe((cx + 0.8 * half, cy, slab_mid_z), "ex")
    sim.add_probe((cx, cy, float(slab_z + slab_thick + 0.25 * lam)), "ex")
    region = DesignRegion(corner_lo=(cx - half, cy - half, float(slab_z)),
                          corner_hi=(cx + half, cy + half, float(slab_z + slab_thick)),
                          eps_range=(1.0, 10.0))
    box_lo = (float(cpml_t + m), float(cpml_t + m), float(box_z_lo))
    box_hi = (float(Lx - cpml_t - m), float(Ly - cpml_t - m), float(box_z_hi))
    sim.add_ntff_box(corner_lo=box_lo, corner_hi=box_hi, freqs=[f0])
    sim.add(Box((cx - half, cy - half, float(plate_z)),
                (cx + half, cy + half, float(plate_z))), material="pec")
    grid = sim._build_grid()
    plate = dict(z=float(plate_z), half=float(half), cx=cx, cy=cy)
    plate["realized"] = RC.assert_wall_planes(sim, 2, [float(plate_z)], at=(cx, cy),
                                              label="reflector plate", grid=grid, tol_m=1e-12)
    return sim, region, grid, plate, lam, f0


def sqrt_segments(n_steps: int) -> int:
    """The largest divisor of ``n_steps`` that is <= sqrt(n_steps)."""
    for k in range(int(np.sqrt(n_steps)), 0, -1):
        if n_steps % k == 0:
            return k
    return 1


def nested_start(coarse_lo, coarse_pads, fine_pads, refine: int) -> tuple:
    """Fine index of the first fine cell nested in coarse cell ``coarse_lo``."""
    return tuple(int(fp + refine * (c - cp)) for c, cp, fp in zip(coarse_lo, coarse_pads, fine_pads))


def nested_block(coarse, refine: int) -> np.ndarray:
    """Each coarse cell's eps_r on the ``refine``^3 fine cells nested in it."""
    b = np.asarray(coarse)
    for ax in range(3):
        b = np.repeat(b, refine, axis=ax)
    return b


def _pattern_fn(mod, sim, grid, start, n_steps, precision, segments):
    """``mod.make_pattern_fn`` with a dtype, ``checkpoint_segments`` and the JAX
    far-field transform on a box built from the declaration.

    Segmented checkpointing changes what reverse mode stores, not what it
    computes: the module's per-step ``checkpoint=True`` keeps the six field
    arrays of every step (1575 x 17.4 MB at lambda/20), the segmented path
    about 2 sqrt(n_steps) of them.  ``compute_far_field`` in the module picks
    the numpy transform on an eager call and the JAX one under a gradient;
    here the JAX one (``compute_far_field_jax``) is used on every call, with
    the NTFF box built once by ``make_ntff_box`` from ``sim._ntff`` (the call
    ``forward`` makes) and its frequencies held as numpy, so the gradient step
    can be jitted: under ``jax.jit`` the forward's own box carries traced
    frequencies that the transform reads with numpy."""
    import jax.numpy as jnp
    from rfx.core.jax_utils import is_tracer
    from rfx.farfield import compute_far_field_jax, make_ntff_box
    dtype = jnp.float64 if precision == "float64" else jnp.float32
    _sheets: list = []
    _wires: list = []
    base_materials, *_ = sim._assemble_materials(grid, pec_sheets=_sheets, pec_wires=_wires)
    base_eps_r = jnp.asarray(base_materials.eps_r, dtype=dtype)
    corner_lo, corner_hi, freqs = sim._ntff
    box = make_ntff_box(grid, corner_lo, corner_hi, freqs)
    box = box._replace(freqs=np.asarray(box.freqs))
    si, sj, sk = start
    witness: dict = {"settling": None}
    kw = {"checkpoint_segments": int(segments)} if segments else {}

    def pattern(eps_slab):
        nx, ny, nz = eps_slab.shape
        eps_override = base_eps_r.at[si:si + nx, sj:sj + ny, sk:sk + nz].set(
            jnp.clip(jnp.asarray(eps_slab, dtype=dtype), 1.0, 10.0))
        res = sim.forward(eps_override=eps_override, n_steps=n_steps, checkpoint=True,
                          skip_preflight=True, **kw)
        if not is_tracer(res.ntff_data.x_lo):
            witness["settling"] = res.settling_witness
        ff = compute_far_field_jax(res.ntff_data, box, grid, mod.THETA, mod.PHI)
        power = jnp.abs(ff.E_theta) ** 2 + jnp.abs(ff.E_phi) ** 2
        return power[0] * 1e27

    pattern.witness = witness
    pattern.segments = segments
    pattern.box = box
    return pattern


def steering_loss(mod, p):
    """The module's objective (``main()``'s ``loss``), on a sampled pattern."""
    import jax.numpy as jnp
    W = mod._W
    prad = jnp.sum(p * W)
    u_steer = p[mod.I_T0, mod.I_P0]
    u_broadside = p[0, :].mean()
    u_back = jnp.sum(p[mod.I_BACK:, :] * W[mod.I_BACK:, :]) / jnp.sum(W[mod.I_BACK:, :])
    eps = 1e-12
    return (-(jnp.log(u_steer + eps) - jnp.log(prad + eps))
            + mod.W_BROADSIDE * (jnp.log(u_broadside + eps) - jnp.log(prad + eps))
            + mod.W_BACK * (jnp.log(u_back + eps) - jnp.log(prad + eps)))


def directivity(mod, p) -> np.ndarray:
    """4 pi U / P_rad on the module's (THETA, PHI) grid, linear."""
    p = np.asarray(p, dtype=float)
    return 4.0 * np.pi * p / np.sum(p * np.asarray(mod._W))


def pattern_summary(mod, p) -> dict:
    d = directivity(mod, p)
    th = np.degrees(np.asarray(mod.THETA))
    i_pk = np.unravel_index(np.argmax(d), d.shape)
    e_plane = np.concatenate([d[::-1, 36], d[1:, 0]])   # phi = 180 deg then phi = 0
    th_e = np.concatenate([-th[::-1], th[1:]])
    i_e = int(np.argmax(e_plane))
    return {"D30_dbi": float(10 * np.log10(max(d[mod.I_T0, mod.I_P0], 1e-12))),
            "D_broadside_dbi": float(10 * np.log10(max(d[0, :].mean(), 1e-12))),
            "D_peak_dbi": float(10 * np.log10(d[i_pk])),
            "peak_theta_deg": float(th[i_pk[0]]),
            "peak_phi_deg": float(np.degrees(np.asarray(mod.PHI))[i_pk[1]]),
            "e_plane_peak_theta_deg": float(th_e[i_e]),
            "e_plane_peak_dbi": float(10 * np.log10(e_plane[i_e]))}


def settling_record(pattern) -> dict | None:
    w = getattr(pattern, "witness", {}).get("settling")
    if not w or w.get("status") != "measured":
        return {"status": None if not w else w.get("status")}
    worst = w["per_record_db"][w["worst_record"]]
    return {"status": "measured", "per_record_db": dict(w["per_record_db"]),
            "worst_record": w["worst_record"], "worst_db": float(worst),
            "passed": bool(worst <= SETTLING_BAR_DB)}




# ------------------------------------------------------------------ stages
def check_refined_matches_module(mod) -> dict:
    """``build_problem_refined(mod, 1)`` must build what ``mod.build_problem()``
    builds (grid, design indices, plate plane), or the fine re-solve is not the
    same structure."""
    a = mod.build_problem()
    b = build_problem_refined(mod, 1)
    la, ha, sa = mod.resolve_design_indices(a[1], a[2])
    lb, hb, sb = mod.resolve_design_indices(b[1], b[2])
    same = {"grid_shape": a[2].shape == b[2].shape, "design": (la, ha, sa) == (lb, hb, sb),
            "plate_planes": a[3]["realized"]["planes"] == b[3]["realized"]["planes"],
            "dx": a[2].dx == b[2].dx}
    if not all(same.values()):
        raise RuntimeError(f"build_problem_refined(mod, 1) differs from the module: {same}")
    return same


def stage_timing(out: Path, a) -> int:
    import jax
    import jax.numpy as jnp
    import optax
    mod = load_module()
    t = {}
    model = BeamModel(mod, 1)
    check = check_refined_matches_module(mod)
    n_steps = model.n_steps(a.periods)
    dc.save_json(out / "timing_preflight_l20.json", model.preflight())
    pattern = model.pattern_fn(n_steps)
    ones = np.ones(model.shape, dtype=np.float32)
    for tag in ("forward_first_s", "forward_second_s"):
        t0 = time.perf_counter()
        jax.block_until_ready(pattern(ones))
        t[tag] = time.perf_counter() - t0

    def loss(psi):
        p = pattern(mod.eps_of_psi(psi))
        return steering_loss(mod, p), p

    vg = jax.jit(jax.value_and_grad(loss, has_aux=True))
    psi = jnp.asarray(module_start(model.shape)[1])
    opt = optax.adam(mod.LR)
    state = opt.init(psi)
    t["grad_s"] = []
    for it in range(a.iters):
        t0 = time.perf_counter()
        (v, p), g = vg(psi)
        jax.block_until_ready(g)
        t["grad_s"].append(time.perf_counter() - t0)
        updates, state = opt.update(g, state)
        psi = optax.apply_updates(psi, updates)
        dc.log(f"timing iter {it}: L {float(v):+.4f} D30 {pattern_summary(mod, p)['D30_dbi']:+.2f} dBi "
               f"({t['grad_s'][-1]:.1f} s)")
    t["memory_after_grads"] = dc.device_memory()
    fine = BeamModel(mod, FINE_REFINE)
    n_fine = fine.n_steps(a.periods)
    pf = fine.pattern_fn(n_fine)
    t0 = time.perf_counter()
    jax.block_until_ready(pf(np.ones(fine.block_shape, dtype=np.float32)))
    t["forward_fine_first_s"] = time.perf_counter() - t0
    t["memory_end"] = dc.device_memory()
    g_s, f_s = t["grad_s"][-1], t["forward_second_s"]
    proj = {"main_s": (len(UNIFORM_EPS) + 18 + 4) * f_s + (mod.N_ITERS + 1 + 1.5 + 2) * g_s
            + t["forward_first_s"] + t["grad_s"][0],
            "resolve_s": 3 * f_s + t["forward_first_s"] + 3 * t["forward_fine_first_s"],
            "keyframes_s": len(KEYFRAMES) * (f_s + 30.0)}
    proj["total_gpu_h"] = sum(v for v in proj.values()) / 3600.0
    dc.save_json(out / "timing.json", {
        "grid_l20": model.grid.shape, "grid_l40": fine.grid.shape, "design_shape_l20": model.shape,
        "block_shape_l40": fine.block_shape, "n_steps_l20": n_steps, "n_steps_l40": n_fine,
        "iters_timed": a.iters, "timings": t, "projection": proj, "device": str(jax.devices()[0]),
        "refined_builder_check": check,
        "formula": "main = 32 forwards + (N_ITERS + 4.5) gradients at steady state + first-call "
                   "extras (the module-path equivalence gradient counted as two); resolve = 4 "
                   "lambda/20 forwards + 3 first-call lambda/40 forwards; keyframes = one forward "
                   "and a far-field sphere (30 s allowance) per keyframe"})
    dc.log(f"projection: {proj}")
    return 0


def _row(mod, psi, p, g=None):
    """One iterate: the cells, the pattern, and (when taken) the gradient."""
    psi = np.asarray(psi, dtype=float)
    sg = 1.0 / (1.0 + np.exp(-psi))
    eps = 1.0 + 9.0 * sg
    s = pattern_summary(mod, p)
    r = dict(psi=psi.astype(np.float32), eps=eps.astype(np.float32),
             D=directivity(mod, p).astype(np.float32), D30_dbi=s["D30_dbi"],
             D_broadside_dbi=s["D_broadside_dbi"], e_plane_peak_theta_deg=s["e_plane_peak_theta_deg"],
             e_plane_peak_dbi=s["e_plane_peak_dbi"])
    if g is not None:
        g = np.asarray(g, dtype=float)
        ge = g / (9.0 * sg * (1.0 - sg))
        r.update(grad_psi=g.astype(np.float32), grad_eps=ge.astype(np.float32),
                 grad_eps_map=ge.sum(axis=2).astype(np.float32))
    return r


def stage_main(out: Path, a) -> int:
    import jax
    import jax.numpy as jnp
    import optax
    from rfx import gradient_record_length_witness
    mod = load_module()
    wall = {}
    model = BeamModel(mod, 1)
    check = check_refined_matches_module(mod)
    n_steps = model.n_steps(a.periods)
    n_iters = a.iters if a.iters is not None else mod.N_ITERS
    shape = model.shape
    dc.save_json(out / "preflight_l20.json", model.preflight())
    dc.save_json(out / "model.json", {
        "structure": "x-directed dipole lambda/4 above a finite PEC plate (a sheet), a dielectric "
                     "cover above it, NTFF box, CPML (validation/tmtt_paper/"
                     "beam_steering_superstrate.py build_problem, SMOKE=0)",
        "module": str(MODULE.relative_to(REPO)), **model.describe(), "n_steps": n_steps,
        "num_periods": a.periods or mod.NUM_PERIODS,
        "theta_deg": np.degrees(np.asarray(mod.THETA)), "phi_deg": np.degrees(np.asarray(mod.PHI)),
        "i_theta0": mod.I_T0, "i_phi0": mod.I_P0, "i_back": mod.I_BACK,
        "objective": "the module's L = -log U(30,0)/P + 0.3 log mean U(0,phi)/P + 0.5 log "
                     "mean_{theta>90} U/P, on its 73 x 73 (theta, phi) grid",
        "parameterization": "the module's: every cover cell, eps_r = 1 + 9 sigmoid(psi), "
                            f"{shape[0]} x {shape[1]} x {shape[2]} = {int(np.prod(shape))} cells",
        "adam": {"lr": mod.LR, "iters": n_iters, "optax": "optax.adam(lr) defaults",
                 "start": "the module's eps_r 2 -> 9 ramp along x"},
        "refined_builder_check": check, "checkpoint_segments": sqrt_segments(n_steps),
        "matmul_precision": MATMUL_PRECISION, "keyframes": KEYFRAMES,
        "jit": "jax.jit(jax.value_and_grad(L(psi), has_aux=True)); the module does not jit",
        "far_field_path": "rfx.farfield.compute_far_field_jax on every call, on a box from "
                          "make_ntff_box(grid, *sim._ntff) with numpy frequencies (the module's "
                          "compute_far_field dispatches to numpy on eager calls)"})
    pattern = model.pattern_fn(n_steps)
    eps0, psi0 = module_start(shape)

    def L_cells(eps, pat=pattern):
        return steering_loss(mod, pat(eps))

    # the module-path comparison (A3.2) runs as its own job: stage "equivalence"
    g_scr = np.asarray(jax.jit(jax.grad(L_cells))(jnp.asarray(eps0)), dtype=float)

    # ---- baselines: no cover, and uniform covers ---------------------------
    base, maps = {}, {}
    for e in UNIFORM_EPS:
        name = f"uniform_eps{e:g}"
        t0 = time.perf_counter()
        p = np.asarray(pattern(np.full(shape, e, dtype=np.float32)))
        wall[f"baseline_{name}_s"] = time.perf_counter() - t0
        base[name] = {"eps_r": e, **pattern_summary(mod, p), "settling": settling_record(pattern)}
        maps[f"D_{name}"] = directivity(mod, p)
        dc.log(f"baseline {name}: D30 {base[name]['D30_dbi']:+.2f} dBi, settling "
               f"{base[name]['settling'].get('worst_db')}")
    best = max(base, key=lambda n: base[n]["D30_dbi"])
    base["best_uniform"] = best
    dc.save_json(out / "baselines_l20.json", base)
    dc.save_npz(out / "baselines_l20.npz", uniform_eps=np.asarray(UNIFORM_EPS), **maps)

    # ---- AD against central differences at the start ----------------------
    L_j = jax.jit(L_cells)
    g0 = g_scr                           # the jitted start gradient of the comparison above
    ladder = dc.fd_ladder(lambda e: L_j(jnp.asarray(e, dtype=jnp.float32)), eps0, FD_CELLS, FD_STEPS)
    ad = {n: float(g0[ijk]) for n, ijk in FD_CELLS.items()}
    verdict = dc.judge_fd(ad, ladder, FD_STEPS, FD_JUDGED_STEP, FD_REL_BAR, FD_JUDGE_FRAC)
    dc.save_json(out / "fd_start.json", {"precision": "float32", "eps0": eps0, "L0": float(L_j(eps0)),
                                         "variables": FD_CELLS, "grad_eps_all": g0,
                                         "ladder": ladder, "judgement": verdict})
    dc.log("AD vs FD at the start: " + "; ".join(
        f"{n}: AD {r['ad']:+.5e} FD {r['fd']:+.5e} rel {r['rel']:.2e}{' judged' if r['judged'] else ''}"
        for n, r in verdict["rows"].items()))
    if verdict["roundoff"]:
        dc.save_json(out / "x64_needed.json", {"rule": "FD moves more between h = 0.1 and 0.05 "
                                                       "than between 0.2 and 0.1",
                                               "variables": verdict["roundoff"]})
        dc.log(f"round-off rule met at {verdict['roundoff']}: x64 stage required")

    # ---- record-length witness at the start ---------------------------------
    pats = {n_steps: pattern}

    def witness_objective(e, n):
        if n not in pats:
            pats[n] = model.pattern_fn(n)
        return L_cells(e, pats[n])

    t0 = time.perf_counter()
    w = gradient_record_length_witness(witness_objective, jnp.asarray(eps0), n_steps,
                                       tol=WITNESS_TOL, factor=WITNESS_FACTOR)
    wall["witness_s"] = time.perf_counter() - t0
    g1 = np.asarray(next(iter(w.grad.values())))[0].astype(float)
    g15 = np.asarray(next(iter(w.grad_long.values())))[0].astype(float)
    per_var = {n: (float(abs(g15[ijk] - g1[ijk]) / abs(g15[ijk])) if g15[ijk] else None)
               for n, ijk in FD_CELLS.items()}
    wit = {"helper": "rfx.gradient_record_length_witness", "factor": WITNESS_FACTOR, "tol": WITNESS_TOL,
           "n_steps": {"1.0x": w.n_steps, "1.5x": w.n_steps_long},
           "grad": {"1.0x": g1, "1.5x": g15}, "norm_rel_change": float(w.worst),
           "cosine": float(w.cosine_by_bin[0]), "helper_passed": bool(w.passed),
           "per_variable_rel_change": per_var, "judged_variables": verdict["judged"],
           "L": {"1.0x": float(np.real(w.value[0])), "1.5x": float(np.real(w.value_long[0]))}}
    wit["passed"] = bool(w.worst <= WITNESS_TOL
                         and all(per_var[n] is not None and per_var[n] <= WITNESS_TOL
                                 for n in verdict["judged"]))
    dc.save_json(out / "record_length_witness.json", wit)
    dc.log(f"record-length witness: norm {w.worst:.3e}, per variable "
           + ", ".join(f"{n} {v if v is None else f'{v:.3e}'}" for n, v in per_var.items())
           + f" -> passed {wit['passed']}")
    pats.clear()
    if not wit["passed"]:
        dc.save_json(out / "witness_failed.json", wit)
        if not a.smoke:
            dc.log("record-length witness FAILED: the case stops before the descent")
            return 3
        dc.log("record-length witness FAILED; --smoke: continuing to exercise the descent")

    # ---- Adam, every iterate ------------------------------------------------
    def loss(psi):
        p = pattern(mod.eps_of_psi(psi))
        return steering_loss(mod, p), p

    vg = jax.jit(jax.value_and_grad(loss, has_aux=True))
    psi = jnp.asarray(psi0)
    opt = optax.adam(mod.LR)
    state = opt.init(psi)
    store = dc.IterateStore(out / "iterations.npz", static={
        "theta_deg": np.degrees(np.asarray(mod.THETA)), "phi_deg": np.degrees(np.asarray(mod.PHI))})
    for it in range(n_iters):
        t0 = time.perf_counter()
        (v, p), g = vg(psi)
        jax.block_until_ready(g)
        dt = time.perf_counter() - t0
        store.append(**_row(mod, psi, p, g), L=float(v), wall_s=dt)
        updates, state = opt.update(g, state)
        psi = optax.apply_updates(psi, updates)
        store.persist()
        dc.save_npz(out / "adam_state.npz", iterate_next=it + 1, psi_next=np.asarray(psi),
                    **dc.adam_state_arrays(state))
        if it % 5 == 0 or it == n_iters - 1:
            dc.log(f"iter {it:3d} L {float(v):+.4f} D30 {store.rows['D30_dbi'][-1]:+.2f} dBi ({dt:.1f} s)")
    t0 = time.perf_counter()
    p = np.asarray(pattern(mod.eps_of_psi(psi)))
    Lf = float(steering_loss(mod, jnp.asarray(p)))
    store.append(**_row(mod, psi, p), L=Lf, wall_s=time.perf_counter() - t0)
    store.persist()
    dc.save_json(out / "final_settling_l20.json", settling_record(pattern))
    wall["adam_total_s"] = float(np.nansum(np.stack(store.rows["wall_s"])))
    dc.save_json(out / "main_wall.json", {**wall, "memory": dc.device_memory()})
    dc.log(f"final iterate {n_iters}: L {Lf:+.4f}, D30 {store.rows['D30_dbi'][-1]:+.2f} dBi")
    return 0


def stage_equivalence(out: Path, a) -> int:
    """Judged (A3.2): at the start cover and the full record, the module's
    make_pattern_fn (per-step checkpoint, compute_far_field dispatch, not
    jitted) against this script's path (checkpoint_segments,
    compute_far_field_jax, jitted), pattern and gradient. Its own job: the
    module path's per-step tape measured 63 GB."""
    import jax
    import jax.numpy as jnp
    mod = load_module()
    model = BeamModel(mod, 1)
    n_steps = model.n_steps(a.periods)
    pattern = model.pattern_fn(n_steps)
    eps0, _ = module_start(model.shape)
    pm_fn = model.module_pattern_fn(n_steps)
    t0 = time.perf_counter()
    pm = np.asarray(pm_fn(eps0))
    ps = np.asarray(pattern(eps0))
    g_mod = np.asarray(jax.grad(lambda e: steering_loss(mod, pm_fn(e)))(jnp.asarray(eps0)), dtype=float)
    mem_module = dc.device_memory()
    g_scr = np.asarray(jax.jit(jax.grad(lambda e: steering_loss(mod, pattern(e))))(jnp.asarray(eps0)),
                       dtype=float)
    eq = {"what": "at the start cover, full record: the module's make_pattern_fn (per-step "
                  "checkpoint, compute_far_field dispatch, value_and_grad not jitted) against this "
                  "script's (checkpoint_segments, compute_far_field_jax, jitted)",
          "segments": pattern.segments, "n_steps": n_steps, "tol": EQUIV_TOL,
          "matmul_precision": MATMUL_PRECISION,
          "pattern_max_abs_diff_over_max": float(np.max(np.abs(pm - ps)) / np.max(np.abs(pm))),
          "grad_norm_rel_diff": float(np.linalg.norm(g_scr - g_mod) / np.linalg.norm(g_mod)),
          "D30_dbi_module": pattern_summary(mod, pm)["D30_dbi"],
          "D30_dbi_script": pattern_summary(mod, ps)["D30_dbi"],
          "wall_s": time.perf_counter() - t0, "memory_after_module_grad": mem_module}
    eq["passed"] = bool(eq["pattern_max_abs_diff_over_max"] <= EQUIV_TOL
                        and eq["grad_norm_rel_diff"] <= EQUIV_TOL)
    dc.save_json(out / "pattern_equivalence.json", eq)
    dc.save_npz(out / "equivalence_grads.npz", g_module=g_mod, g_script=g_scr)
    dc.log(f"module path vs script path: pattern {eq['pattern_max_abs_diff_over_max']:.2e}, gradient "
           f"{eq['grad_norm_rel_diff']:.2e} -> passed {eq['passed']}")
    return 0 if eq["passed"] else 4


def stage_x64fd(out: Path, a) -> int:
    import jax
    import jax.numpy as jnp
    import rfx.simulation as rsim
    if not jax.config.read("jax_enable_x64"):
        raise SystemExit("the x64 stage needs JAX_ENABLE_X64=1 in its environment")
    mod = load_module()
    model = BeamModel(mod, 1, precision="float64")
    pattern = model.pattern_fn(model.n_steps(a.periods))
    # forward() passes Simulation(precision="float64") on as field_dtype; the run
    # entry is wrapped for this process to record the dtype of every field state
    run_orig, seen = rsim.run, []

    def run_recording(*args, **kw):
        r = run_orig(*args, **kw)
        req = kw.get("field_dtype")
        if r.state is not None:
            obs = str(r.state.ex.dtype)
        elif getattr(r, "ntff_data", None) is not None:
            obs = str(r.ntff_data.x_lo.dtype)       # the NTFF accumulators follow the fields
        else:
            obs = "nothing observable"
        seen.append((str(jnp.dtype(req)) if req is not None else "None (float32)", obs))
        return r

    rsim.run = run_recording

    def L_cells(e):
        return steering_loss(mod, pattern(e))

    e0 = module_start(model.shape)[0].astype(np.float64)
    L_j = jax.jit(L_cells)
    g0 = np.asarray(jax.jit(jax.grad(L_cells))(jnp.asarray(e0)), dtype=float)
    ladder = dc.fd_ladder(lambda e: L_j(jnp.asarray(e, dtype=jnp.float64)), e0, FD_CELLS, FD_STEPS)
    ad = {n: float(g0[ijk]) for n, ijk in FD_CELLS.items()}
    verdict = dc.judge_fd(ad, ladder, FD_STEPS, FD_JUDGED_STEP, FD_REL_BAR, FD_JUDGE_FRAC)
    L0 = float(L_j(e0))
    rsim.run = run_orig
    dtypes = sorted(set(seen))
    if not dtypes or any(r != "float64" or o not in ("float64", "complex128") for r, o in dtypes):
        raise SystemExit(f"the float64 repeat saw (requested, observed) dtypes {dtypes}; "
                         "it is not a float64 record")
    dc.save_json(out / "fd_float64.json", {"precision": "float64", "eps0": e0, "L0": L0,
                                           "grad_eps_all": g0, "field_dtypes_seen": dtypes,
                                           "field_dtypes_seen_is": "(field_dtype requested of "
                                           "rfx.simulation.run, dtype of the returned fields or, "
                                           "when no state is returned, of the NTFF accumulators)",
                                           "n_run_traces_seen": len(seen), "ladder": ladder,
                                           "judgement": verdict})
    dc.log("float64 AD vs FD: " + "; ".join(f"{n}: rel {r['rel']:.2e}" for n, r in verdict["rows"].items()))
    return 0


def stage_resolve(out: Path, a) -> int:
    """The final cover, no cover and the best uniform cover, read back from the
    record and solved without the optimiser at lambda/20 and lambda/40 (each
    lambda/20 cell written onto its 2 x 2 x 2 nested lambda/40 cells)."""
    import jax.numpy as jnp
    mod = load_module()
    it = np.load(out / "iterations.npz")
    e_final = np.asarray(it["eps"][-1], dtype=np.float32)
    bj = dc.load_json(out / "baselines_l20.json")
    e_best = float(bj[bj["best_uniform"]]["eps_r"])
    res, maps = {}, {}
    for refine in (1, FINE_REFINE):
        tag = f"l{20 * refine}"
        model = BeamModel(mod, refine)
        n_steps = model.n_steps(a.periods)
        dc.save_json(out / f"preflight_{tag}.json", model.preflight())
        pattern = model.pattern_fn(n_steps)
        covers = {"final": nested_block(e_final, refine),
                  "no_cover": np.ones(model.block_shape, dtype=np.float32),
                  "best_uniform": np.full(model.block_shape, e_best, dtype=np.float32)}
        res[tag] = {**model.describe(), "n_steps": n_steps, "designs": {}}
        for name, cov in covers.items():
            t0 = time.perf_counter()
            p = np.asarray(pattern(np.asarray(cov, dtype=np.float32)))
            res[tag]["designs"][name] = {**pattern_summary(mod, p), "settling": settling_record(pattern),
                                         "L": float(steering_loss(mod, jnp.asarray(p))),
                                         "wall_s": time.perf_counter() - t0}
            maps[f"D_{tag}_{name}"] = directivity(mod, p)
            dc.log(f"resolve {tag} {name}: D30 {res[tag]['designs'][name]['D30_dbi']:+.2f} dBi, "
                   f"settling {res[tag]['designs'][name]['settling'].get('worst_db')}")
    res["best_uniform_eps_r"] = e_best
    dc.save_json(out / "resolve.json", res)
    dc.save_npz(out / "resolve.npz", **maps)
    return 0


def stage_keyframes(out: Path, a) -> int:
    """Recorded for visualization, not judged: at the declared keyframes, the
    E-plane near field (Ex and Ez phasors at f0 on the plane y = cy through
    the dipole) and D(theta, phi) over the whole sphere on a 2-degree grid,
    from one extra forward of each stored design. The optimization record is
    only read."""
    import jax.numpy as jnp
    from rfx.farfield import compute_far_field_jax
    mod = load_module()
    it = np.load(out / "iterations.npz")
    n_last = len(it["L"]) - 1
    keys = [k for k in KEYFRAMES if k <= n_last]
    model = BeamModel(mod, 1)
    sim, grid = model.sim, model.grid
    cy = model.plate["cy"]
    for comp in ("ex", "ez"):
        sim.add_dft_plane_probe(axis="y", coordinate=cy, component=comp,
                                freqs=jnp.asarray([model.f0], dtype=jnp.float32), name=f"{comp}_eplane")
    n_steps = model.n_steps(a.periods)
    _sheets: list = []
    _wires: list = []
    base = jnp.asarray(sim._assemble_materials(grid, pec_sheets=_sheets, pec_wires=_wires)[0].eps_r)
    corner_lo, corner_hi, freqs = sim._ntff
    from rfx.farfield import make_ntff_box
    box = make_ntff_box(grid, corner_lo, corner_hi, freqs)
    box = box._replace(freqs=np.asarray(box.freqs))
    th = np.radians(np.arange(0.0, 180.0 + 1e-9, SPHERE_STEP_DEG))
    ph = np.radians(np.arange(0.0, 360.0 - 1e-9, SPHERE_STEP_DEG))
    si, sj, sk = model.start
    nx, ny, nz = model.block_shape
    rec = {"keyframes": np.asarray(keys), "sphere_theta_deg": np.degrees(th),
           "sphere_phi_deg": np.degrees(ph)}
    meta = {"keyframes": keys, "f0_hz": model.f0, "plane": "y = cy (the E-plane through the dipole)",
            "components": ["ex", "ez"], "sphere": f"{SPHERE_STEP_DEG:g} deg in theta 0..180 and phi "
            "0..358: the whole sphere, because the plate is finite and the objective holds the back "
            "hemisphere down", "judged": False, "per_keyframe": {}}
    planes = {c: [] for c in ("ex", "ez")}
    sphere = []
    for k in keys:
        t0 = time.perf_counter()
        eps = jnp.asarray(it["eps"][k], dtype=jnp.float32)
        eo = base.at[si:si + nx, sj:sj + ny, sk:sk + nz].set(jnp.clip(eps, 1.0, 10.0))
        res = sim.forward(eps_override=eo, n_steps=n_steps, checkpoint=False, skip_preflight=True)
        by_name = {c: res.dft_planes[f"{c}_eplane"] for c in ("ex", "ez")}
        for c in ("ex", "ez"):
            planes[c].append(np.asarray(by_name[c].accumulator[0]).astype(np.complex64))
        ff = compute_far_field_jax(res.ntff_data, box, grid, jnp.asarray(th), jnp.asarray(ph))
        pw = np.asarray(jnp.abs(ff.E_theta[0]) ** 2 + jnp.abs(ff.E_phi[0]) ** 2, dtype=float)
        w = np.sin(th)[:, None] * np.radians(SPHERE_STEP_DEG) ** 2
        D = 4.0 * np.pi * pw / np.sum(pw * w)
        sphere.append(D.astype(np.float32))
        sw = res.settling_witness or {}
        meta["per_keyframe"][str(k)] = {
            "wall_s": time.perf_counter() - t0,
            "D30_dbi_sphere": float(10 * np.log10(D[int(round(30 / SPHERE_STEP_DEG)), 0])),
            "D30_dbi_recorded": float(it["D30_dbi"][k]),
            "plane_index_y": int(by_name["ex"].index),
            "settling_worst_db": (float(sw["per_record_db"][sw["worst_record"]])
                                  if sw.get("status") == "measured" else None)}
        dc.log(f"keyframe {k}: D30 sphere {meta['per_keyframe'][str(k)]['D30_dbi_sphere']:+.2f} dBi, "
               f"recorded {it['D30_dbi'][k]:+.2f} ({time.perf_counter() - t0:.1f} s)")
    pads = grid.axis_pads
    rec.update(ex_eplane=np.stack(planes["ex"]), ez_eplane=np.stack(planes["ez"]),
               D_sphere=np.stack(sphere),
               x_mm=(np.arange(planes["ex"][0].shape[0]) - pads[0]) * model.dx * 1e3,
               z_mm=(np.arange(planes["ex"][0].shape[1]) - pads[2]) * model.dx * 1e3)
    meta["geometry_mm"] = {"plate_z": model.plate["z"] * 1e3, "cx": model.plate["cx"] * 1e3,
                           "half": model.plate["half"] * 1e3,
                           "cover_z": [float(model.region.corner_lo[2] * 1e3),
                                       float(model.region.corner_hi[2] * 1e3)],
                           "dipole_z": (model.plate["z"] + model.lam / 4) * 1e3}
    meta["note"] = ("phasors are the raw DFT accumulators of the modulated-Gaussian drive; normalize "
                    "for display. Planes span the padded grid, x_mm/z_mm give the node coordinates.")
    dc.save_npz(out / "keyframes.npz", **rec)
    dc.save_json(out / "keyframes.json", meta)
    return 0


def stage_finalize(out: Path, a) -> int:
    x64 = (out / "fd_float64.json").is_file()
    precision = "float32" + (" (the start-point FD check repeated in float64)" if x64 else "")
    here = _record.source_block(a.repo_dir, precision=precision)
    if (out / "source_main.json").is_file():
        # the job that produced the numbers; finalize may run elsewhere (A3.2)
        source = {**dc.load_json(out / "source_main.json"), "precision": precision,
                  "finalize_source": here}
        if source["repo_sha"] != here["repo_sha"]:
            raise SystemExit(f"finalize at {here['repo_sha']} but the record was made at "
                             f"{source['repo_sha']}")
    else:
        source = here
    model = dc.load_json(out / "model.json")
    claims, derived = [], []
    if a.equivalence is not None:
        import shutil
        eq_src = dc.load_json(a.equivalence / "pattern_equivalence.json")
        rid = (a.equivalence / "run_id.txt").read_text().strip() \
            if (a.equivalence / "run_id.txt").is_file() else None
        eq_src["job"] = {"dir": str(a.equivalence), "vessl_run_id": rid,
                         "commit": (a.equivalence / "commit.txt").read_text().strip()}
        dc.save_json(out / "pattern_equivalence.json", eq_src)
        shutil.copy2(a.equivalence / "equivalence_grads.npz", out / "equivalence_grads.npz")
    if not (out / "pattern_equivalence.json").is_file():
        raise SystemExit("pattern_equivalence.json is absent: the judged module-path comparison "
                         "(A3.2) is part of this record; pass --equivalence DIR")
    eq = dc.load_json(out / "pattern_equivalence.json")
    claims += [
        _record.claim("module path vs this script, start cover: max abs pattern difference / max",
                      eq["pattern_max_abs_diff_over_max"], "1", "pattern_equivalence.json",
                      threshold=EQUIV_TOL, rule="Amendment 2 condition (lane leader)"),
        _record.claim("module path vs this script, start cover: ||g_script - g_module|| / ||g_module||",
                      eq["grad_norm_rel_diff"], "1", "pattern_equivalence.json",
                      threshold=EQUIV_TOL, rule="Amendment 2 condition (lane leader)")]
    if (out / "fd_start.json").is_file():
        fd32 = dc.load_json(out / "fd_start.json")
        if (out / "x64_needed.json").is_file() and not x64:
            raise SystemExit("x64_needed.json is present but fd_float64.json is absent: "
                             "the float64 judgement is not a record, finalize refuses")
        fd = dc.load_json(out / "fd_float64.json") if x64 else fd32
        fd_file = "fd_float64.json" if x64 else "fd_start.json"
        for n, r in fd["judgement"]["rows"].items():
            q = (f"|AD - FD| / |FD| at h = {FD_JUDGED_STEP} in eps_r, dL/d eps_r of {n}, start "
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
    if (out / "record_length_witness.json").is_file():
        wit = dc.load_json(out / "record_length_witness.json")
        claims.append(_record.claim("record-length witness ||g_1.5x - g_1.0x|| / ||g_1.5x|| over the "
                                    "2883 cover cells, start", wit["norm_rel_change"], "1",
                                    "record_length_witness.json", threshold=WITNESS_TOL,
                                    rule="pre-declared: factor 1.5, tol 0.05"))
        for n, v in wit["per_variable_rel_change"].items():
            kw = ({"threshold": WITNESS_TOL, "rule": "pre-declared: per FD-judged variable"}
                  if n in wit["judged_variables"] and v is not None else {})
            claims.append(_record.claim(f"record-length witness |g_1.5x - g_1.0x| / |g_1.5x|, {n}", v,
                                        "1", "record_length_witness.json", **kw))
    if (out / "baselines_l20.json").is_file():
        base = dc.load_json(out / "baselines_l20.json")
        for name in (f"uniform_eps{e:g}" for e in UNIFORM_EPS):
            claims.append(_record.claim(f"{name} at lambda/20: D(30 deg)", base[name]["D30_dbi"], "dBi",
                                        "baselines_l20.json"))
            s = base[name]["settling"]
            if s.get("status") == "measured":
                claims.append(_record.claim(f"{name} at lambda/20: worst probe settling", s["worst_db"],
                                            "dB", "baselines_l20.json"))
    if (out / "iterations.npz").is_file():
        it = np.load(out / "iterations.npz")
        n = len(it["L"]) - 1
        claims += [
            _record.claim("L at iterate 0", float(it["L"][0]), "1", "iterations.npz"),
            _record.claim(f"L at iterate {n} (final design, no step)", float(it["L"][-1]), "1",
                          "iterations.npz"),
            _record.claim("D(30 deg) at iterate 0", float(it["D30_dbi"][0]), "dBi", "iterations.npz"),
            _record.claim(f"D(30 deg) at iterate {n}", float(it["D30_dbi"][-1]), "dBi", "iterations.npz"),
            _record.claim(f"E-plane peak angle at iterate {n}", float(it["e_plane_peak_theta_deg"][-1]),
                          "deg", "iterations.npz"),
            _record.claim("median wall time per Adam iteration (jitted value_and_grad)",
                          float(np.median(it["wall_s"][1:-1])), "s", "iterations.npz"),
        ]
        derived.append(_record.derived("D(30 deg) gain, iterate 0 to final",
                                       float(it["D30_dbi"][-1] - it["D30_dbi"][0]), "dB",
                                       "D30_dbi[final] - D30_dbi[0]",
                                       {"D30_dbi[0]": float(it["D30_dbi"][0]),
                                        "D30_dbi[final]": float(it["D30_dbi"][-1])}))
    if (out / "final_settling_l20.json").is_file():
        s = dc.load_json(out / "final_settling_l20.json")
        if s.get("status") == "measured":
            claims.append(_record.claim("final design at lambda/20: worst probe settling", s["worst_db"],
                                        "dB", "final_settling_l20.json"))
    if (out / "resolve.json").is_file():
        rs = dc.load_json(out / "resolve.json")
        for tag in (k for k in rs if k.startswith("l")):
            for name, d in rs[tag]["designs"].items():
                claims.append(_record.claim(f"{name}, re-solved at {tag}: D(30 deg)", d["D30_dbi"], "dBi",
                                            "resolve.json"))
                claims.append(_record.claim(f"{name}, re-solved at {tag}: E-plane peak angle",
                                            d["e_plane_peak_theta_deg"], "deg", "resolve.json"))
                if d["settling"].get("status") == "measured":
                    claims.append(_record.claim(f"{name}, re-solved at {tag}: worst probe settling",
                                                d["settling"]["worst_db"], "dB", "resolve.json"))
    run = {"platform": "VESSL", "preset": os.environ.get("RFX_SHOWCASE_PRESET"), "run_id": None}
    if (out / "main_wall.json").is_file():
        run["wall_s"] = dc.load_json(out / "main_wall.json")
    result = {"schema": _record.SCHEMA, "id": CASE_ID, "question": QUESTION, "source": source,
              "run": run, "model": model, "claims": claims, "derived": derived,
              "out_of_scope": ["the paper's lambda/40 numbers and its 441-latent variant",
                               "realized gain and radiation efficiency (directivity only)",
                               "an external solver", "frequencies other than 3 GHz",
                               "keyframes.npz: recorded for visualization, not judged"]}
    path = _record.write_result(out, result, _record.data_files(out))
    dc.log(f"wrote {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=("timing", "main", "equivalence", "x64fd", "resolve", "keyframes",
                             "finalize"))
    ap.add_argument("--equivalence", type=Path, default=None,
                    help="finalize: the equivalence job's output directory")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo-dir", type=Path, default=REPO)
    ap.add_argument("--iters", type=int, default=None,
                    help="Adam iterations (default: the module's N_ITERS; timing default 2)")
    ap.add_argument("--periods", type=float, default=None, help="local smoke runs only")
    ap.add_argument("--smoke", action="store_true",
                    help="local smoke runs only: continue past a failed witness or comparison")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "timing" and a.iters is None:
        a.iters = 2
    stage = {"timing": stage_timing, "main": stage_main, "equivalence": stage_equivalence,
             "x64fd": stage_x64fd,
             "resolve": stage_resolve, "keyframes": stage_keyframes,
             "finalize": stage_finalize}[a.stage]
    if a.stage in ("main", "timing", "equivalence"):
        dc.save_json(a.out / f"source_{a.stage}.json",
                     _record.source_block(a.repo_dir, precision="float32"))
    return stage(a.out, a)


if __name__ == "__main__":
    raise SystemExit(main())
