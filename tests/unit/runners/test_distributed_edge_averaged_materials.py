"""Each E component on the multi-device lanes takes the mean eps and sigma of
the four cells around its edge, as on one device (#1303).

A Yee E component lies on an edge shared by four cells. Since #1213 the
single-device lanes give it the mean of their permittivity and conductivity,
which puts a dielectric's or a conductor's faces where they are drawn. The two
multi-device lanes -- ``sim.run(devices=...)`` and ``forward(distributed=True)``
-- took each edge's eps and sigma from the one cell that owns it, so the same
model solved a structure whose faces sat half a cell away: on a 12 mm PEC cube
(40 steps, 2 CPU devices, main 57ab73d3) the probe record differed from the
single-device lane by 6.0e-2 of its peak with an eps_r 4 block, 1.4e-2 with a
sigma 0.5 S/m block and 0.83 with a 10 ohm/sq sheet folded into sigma, and the
permittivity gradient of forward(distributed=True) by 2.8x its peak. Both lanes
now build each component's eps and sigma on every x slab through the
single-device helper (``slab_e_component_materials`` ->
``rfx.core.yee.component_e_materials``): a seam row reads the neighbour's last
cell from the static material halo, the x-lo face replicates the boundary
cell, and the CPML psi coefficient takes the same per-component permittivity.

What is checked, and what it measured on Mac arm64, JAX 0.10.2 (differences
as a fraction of the record's peak):

a. Lane parity. Each model against the same model with its material removed
   (the vacuum box, or the PEC volume alone): the multi-device record must
   agree with the single-device one as well as that box does, within
   ``FLOOR_FACTOR`` times its agreement (``CONDUCTOR_FLOOR_FACTOR`` for the
   sheet and the PEC volume) or ``ROUNDING``, whichever is larger. In the
   PEC cube every material case is bitwise equal to the single-device record
   on 2, 3 and 4 devices (uneven slabs: 13 cells pad to 14 / 15 / 16), both
   lanes; a PEC volume on a dielectric slab reads 1.0e-6 (run) / 1.6e-6
   (forward) against 6.9e-7 / 7.3e-7 for the volume alone. In a 25-cell CPML
   box the vacuum floor is 2.0e-6 / 1.8e-6 and every material case is under
   3.7e-6. The defect restored reads 6.6e-3 to 0.91.
b. Mutations with the helper calls kept: the cell-owned coefficient (every
   material case red), a seam row reading its own row (the case whose
   interface is on a slab cut red), the CPML psi coefficient on the cell
   permittivity while the update keeps the mean (red at 1.5e-3). The x-lo
   replicate is pinned at the coefficient level (the slab view against the
   single-device rule, bit for bit; a rank-0 ghost read as its vacuum sent
   red): on the records it stays inside the gate, because it only sets the
   Ey/Ez coefficient on the domain's x-lo plane, which the forward lane holds
   at zero (its faces are PEC-backed, as the single-device lanes' are) and
   which on the run lane's CPML body is the absorber's last and smallest
   field (measured 1.5e-6 -> 1.6e-6).
c. Vacuum and homogeneous models (vacuum, and eps_r 4.4 / sigma 0.02 filling
   a CPML box, both lanes): bitwise the records of the same kernels fed the
   cell values, i.e. main's rule. main itself, run out of process on the same
   host, gave the same bytes (measured when this change was made).
d. A Debye block: both lanes take per-component epsilon/sigma and pole-mask
   edge means. CPML reads the same component epsilon as the E update.
e. The permittivity gradient of forward(distributed=True) through a traced
   eps_override, eps_r 4 block: within ``GRAD_RTOL`` of the single-device
   gradient's peak (8.7e-7 PEC, 4.4e-7 CPML; 2.8 / 3.0 before).
f. Slow, Linux: two processes give the one-process records.

The x-hi ghost is never read by the mean (it reads the row before), and the
rows that are not cells of the model -- the ghosts and the alignment pad --
keep the value of the cell they hold, so the domain faces behave as before.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import jax

if (__name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "worker"
        and sys.argv[2] != "reference"):
    jax.distributed.initialize(sys.argv[2], 2, int(sys.argv[3]),
                               initialization_timeout=30)

import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from rfx import Box, DebyePole, GaussianPulse, LorentzPole, Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.core.yee import (  # noqa: E402
    MaterialArrays, cell_owned_component_materials, component_e_materials,
    map_lumped,
)
from rfx.runners import _distributed_common as common  # noqa: E402
from rfx.runners import distributed_nu, distributed_v2  # noqa: E402
from rfx.runners._distributed_common import (  # noqa: E402
    _split_materials, slab_e_component_materials,
)
from rfx.sources.sources import stamp_lumped_eps, stamp_lumped_sigma  # noqa: E402
from rfx.materials.lorentz import lorentz_pole  # noqa: E402

pytestmark = pytest.mark.distributed
SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[3]

# Pre-declared gates.
ROUNDING = 16 * 2.0 ** -23     # 1.9e-6 of the probe peak
FLOOR_FACTOR = 4               # times the same box's agreement without the material
#: A conductor -- the 10 ohm/sq sheet, a PEC volume -- makes the lanes' own
#: arithmetic differ more than vacuum does, whichever rule both lanes apply.
#: Measured with BOTH lanes on the cell-owned rule, then both on the edge
#: mean, as a multiple of the vacuum box's agreement: the sheet on JAX 0.4.33
#: 3.6x / 6.0x (0 / 0 on 0.6.2 and 0.10.2, bitwise), the PEC volume on its
#: dielectric slab on 0.6.2 3.5x / 6.9x. The restored defect reads 1.9e-1
#: and 1.1e-2 of the record peak on these two models.
CONDUCTOR_FLOOR_FACTOR = 16
CONDUCTOR_CASES = ("sheet", "pec_on_slab")
GRAD_RTOL = 1e-5               # of the single-device gradient's peak
MUTATION_MARGIN = 100          # a restored defect must exceed the gate this many times
CI_JAX = "0.6.2"               # the required fast-suite lane's JAX (Python 3.10)

WAVEFORM = GaussianPulse(f0=5e9, bandwidth=0.8)
STEPS = {"pec": 60, "cpml": 120}
#: lx (mm) per box; the CPML box's 25 cells keep the 4-layer absorber inside
#: the last slab's real cells on 2, 3 and 4 devices
LENGTH_MM = {"pec": 12, "cpml": 16}
CASES = ("eps4", "sigma", "sheet", "pec_on_slab", "cut", "faces")
#: the CPML box carries the cases the absorber is there for: a dielectric and a
#: conductor through its x faces, and an interface on a cut. The sheet and the
#: PEC volume are interior and bitwise in the PEC cube; in the CPML box the
#: lanes' own CPML arithmetic moves them by up to 1.8e-5 of the record peak,
#: rule or no rule (both lanes on the cell-owned rule read up to 8.7e-6).
# "yz": a lossy dielectric slab through the y and z absorbers (a substrate that runs
# into the side CPML): pins the per-component permittivity of Ex/Ey/Ez on the y/z faces
# (#1303 review: Ex there taking eps_z/eps_y moved it 2.5e-3 while every other case held).
CPML_CASES = ("eps4", "sigma", "cut", "faces", "yz")
FLOOR_CASE = {"pec_on_slab": "pec_in_vacuum"}   # the rest: "vacuum"


def _mm(*v):
    return tuple(x * 1e-3 for x in v)


def _devices(n):
    devices = jax.devices("cpu")
    if len(devices) < n:
        pytest.skip(f"needs {n} CPU devices")
    return devices[:n]


def _cut_x_mm(boundary, n_dev):
    """x (mm) of the first slab cut, from the grid the lane builds."""
    sim = _box(boundary, "run", "vacuum", n_dev)
    grid = sim._build_grid()
    assert grid.dx == 1e-3
    nx = grid.shape[0]
    nx_per = (nx + (-nx) % n_dev) // n_dev
    return float(nx_per - grid.pad_x_lo)      # whole millimetres, no rounding


def _box(boundary, lane, case, n_dev, waveform=WAVEFORM):
    lx = LENGTH_MM[boundary]
    kw = dict(freq_max=10e9, domain=(lx * 1e-3, 12e-3, 12e-3), dx=1e-3,
              boundary=boundary, cpml_layers=4 if boundary == "cpml" else 0)
    if lane == "fwd":
        kw["dx_profile"] = np.full(lx, 1e-3)     # the graded lane, uniform cells
    sim = Simulation(**kw)
    sim.add_source(_mm(4, 6, 6), "ez", waveform=waveform, amplitude_kind="current")
    sim.add_probe(_mm(8, 6, 6), "ez")
    sim.add_probe(_mm(6.5, 6, 5), "ey")
    if case == "eps4":
        sim.add_material("m", eps_r=4.0)
        sim.add(Box(_mm(5, 3, 3), _mm(7, 9, 9)), material="m")
    elif case == "sigma":
        sim.add_material("m", sigma=0.5)
        sim.add(Box(_mm(5, 3, 3), _mm(7, 9, 9)), material="m")
    elif case == "sheet":      # 10 ohm/sq, folded into sigma
        sim.add_thin_conductor(Box(_mm(6, 3, 3), _mm(6, 9, 9)),
                               sigma_bulk=1e3, thickness=1e-4)
    elif case in ("pec_on_slab", "pec_in_vacuum"):
        if case == "pec_on_slab":
            sim.add_material("m", eps_r=3.0, sigma=0.05)
            sim.add(Box(_mm(2, 2, 2), _mm(10, 10, 5)), material="m")
        sim.add(Box(_mm(9, 3, 5), _mm(11, 9, 7)), material="pec")
    elif case == "cut":        # the block's lower x face on the first slab cut
        x = _cut_x_mm(boundary, n_dev)
        sim.add_material("m", eps_r=4.0, sigma=0.05)
        sim.add(Box(_mm(x, 3, 3), _mm(x + 3, 9, 9)), material="m")
    elif case == "faces":      # through both x faces (and the CPML there)
        sim.add_material("m", eps_r=4.0, sigma=0.1)
        sim.add(Box((-1.0, 3e-3, 3e-3), (1.0, 9e-3, 7e-3)), material="m")
    elif case == "yz":         # through all four y/z faces (and their CPML)
        sim.add_material("m", eps_r=4.0, sigma=0.05)
        sim.add(Box((6e-3, -1.0, -1.0), (10e-3, 1.0, 1.0)), material="m")
    elif case != "vacuum":
        raise ValueError(case)
    return sim


def _record(boundary, lane, case, n_dev, devices=None):
    sim = _box(boundary, lane, case, n_dev)
    if lane == "run":
        out = sim.run(n_steps=STEPS[boundary], skip_preflight=True, devices=devices)
    elif devices is None:
        out = sim.forward(n_steps=STEPS[boundary], skip_preflight=True, checkpoint=False)
    else:
        out = sim.forward(n_steps=STEPS[boundary], skip_preflight=True, checkpoint=False,
                          distributed=True, devices=devices)
    return np.asarray(out.time_series, np.float64)


@functools.lru_cache(maxsize=None)
def _single(boundary, lane, case, n_dev):
    return _record(boundary, lane, case, n_dev)


def _rel(got, want):
    """max |difference| per probe, over the record's peak (all probes): a probe
    the model itself suppresses (Ey half a cell from the resistive sheet) is
    not measured against its own small peak."""
    peak = float(np.max(np.abs(want)))
    return [float(np.max(np.abs(got[:, p] - want[:, p]))) / peak
            for p in range(want.shape[1])]


@functools.lru_cache(maxsize=None)
def _floor(boundary, lane, floor_case, n_dev):
    """The floor box's agreement. It has no material for the helper to
    average (vacuum cells, PEC cells), so no mutation below can move it."""
    devices = jax.devices("cpu")[:n_dev]
    return _rel(_record(boundary, lane, floor_case, n_dev, devices),
                _single(boundary, lane, floor_case, n_dev))


def _parity(boundary, lane, case, n_dev):
    """(this model's agreement, the floor box's agreement), per probe."""
    devices = jax.devices("cpu")[:n_dev]
    got = _rel(_record(boundary, lane, case, n_dev, devices),
               _single(boundary, lane, case, n_dev))
    return got, _floor(boundary, lane, FLOOR_CASE.get(case, "vacuum"), n_dev)


def _gate(floor, case=None):
    factor = CONDUCTOR_FLOOR_FACTOR if case in CONDUCTOR_CASES else FLOOR_FACTOR
    return [max(factor * f, ROUNDING) for f in floor]


def _check(label, got, floor):
    gate = _gate(floor, label.split("/")[1])
    print(f"[{label}] agreement {['%.2e' % g for g in got]} vs gate "
          f"{['%.2e' % g for g in gate]} (floor {['%.2e' % f for f in floor]})")
    assert all(g <= t for g, t in zip(got, gate)), (label, got, gate)


def _realized_cut(boundary, n_dev):
    """The cut case's interface is the first real row of slab 1 (realized)."""
    sim = _box(boundary, "run", "cut", n_dev)
    grid = sim._build_grid()
    eps = np.asarray(sim._assemble_materials(grid)[0].eps_r)
    nx = grid.shape[0]
    nx_per = (nx + (-nx) % n_dev) // n_dev
    column = eps[:, grid.shape[1] // 2, grid.shape[2] // 2]
    assert column[nx_per - 1] == 1.0 and column[nx_per] == np.float32(4.0), column


# --------------------------------------------------------------------------
# the slab view is the single-device rule, coefficient by coefficient
# --------------------------------------------------------------------------

def _random_materials(shape, seed=1303):
    rng = np.random.default_rng(seed)
    mats = MaterialArrays(
        jnp.asarray(rng.uniform(1.0, 9.0, shape).astype(np.float32)),
        jnp.asarray(rng.uniform(0.0, 3.0, shape).astype(np.float32)),
        jnp.ones(shape, jnp.float32))
    # lumped elements on their own edges, one at a would-be seam cell
    mats = stamp_lumped_sigma(mats, (7, 2, 3), 1.0 / (50.0 * 1e-3), "ez")
    mats = stamp_lumped_eps(mats, (4, 3, 1), 2.5, "ey")
    return mats


def _staged(mats, n_dev):
    """The slabs the runners stage: vacuum alignment pad at high x, one ghost
    row a side, physical ghosts vacuum, seam ghosts the neighbour's cells
    (split_array_x, which shard_x_slabs and the forward staging reproduce --
    pinned by test_distributed_memory_staging.py / ..._nu_forward_staging.py)."""
    nx = mats.eps_r.shape[0]
    pad = (-nx) % n_dev
    widths = ((0, pad), (0, 0), (0, 0))
    padded = MaterialArrays(
        jnp.pad(mats.eps_r, widths, constant_values=1.0),
        jnp.pad(mats.sigma, widths, constant_values=0.0),
        jnp.pad(mats.mu_r, widths, constant_values=1.0),
        sigma_lumped=map_lumped(mats.sigma_lumped, lambda a: jnp.pad(a, widths)),
        eps_r_lumped=map_lumped(mats.eps_r_lumped, lambda a: jnp.pad(a, widths)))
    slabs = _split_materials(padded, n_dev)
    out = []
    for rank in range(n_dev):
        def pick(record):
            return map_lumped(record, lambda a: a[rank])
        out.append(MaterialArrays(slabs.eps_r[rank], slabs.sigma[rank], slabs.mu_r[rank],
                                  sigma_lumped=pick(slabs.sigma_lumped),
                                  eps_r_lumped=pick(slabs.eps_r_lumped)))
    return out, (nx + pad) // n_dev


def _slab_view_mismatches(helper, n_dev, shape=(13, 5, 6)):
    """Cells where the slab view differs from the single-device rule, and
    non-model rows that no longer hold their own cell's value."""
    mats = _random_materials(shape)
    want = component_e_materials(mats)
    slabs, nx_per = _staged(mats, n_dev)
    bad = {"model": 0, "other": 0}
    for rank, slab in enumerate(slabs):
        got = helper(slab, nx_per, shape[0], rank=rank)
        cell = cell_owned_component_materials(slab)
        local = np.arange(slab.eps_r.shape[0])
        rows = rank * nx_per - 1 + local
        # the slab's own real rows that are model cells; not ghosts, not pad
        model = (local >= 1) & (local < local.size - 1) & (rows < shape[0])
        for q in range(2):              # eps, sigma
            for c in range(3):          # x, y, z
                g = np.asarray(got[q][c])
                w = np.asarray(want[q][c])[rows[model]]
                bad["model"] += int(np.count_nonzero(g[model].view(np.uint32)
                                                     != w.view(np.uint32)))
                o = np.asarray(cell[q][c])[~model]
                bad["other"] += int(np.count_nonzero(g[~model].view(np.uint32)
                                                     != o.view(np.uint32)))
    return bad


@pytest.mark.parametrize("n_dev", [2, 3, 4])
def test_the_slab_view_is_the_single_device_rule(n_dev):
    """Random eps and sigma with lumped stamps, 13 cells over uneven slabs:
    every row a slab owns is component_e_materials bit for bit, and every
    ghost and pad row keeps its own cell's value."""
    bad = _slab_view_mismatches(slab_e_component_materials, n_dev)
    assert bad == {"model": 0, "other": 0}, bad


def _mutant(kind):
    """The helper with its defect restored and its calls kept."""
    def helper(materials, nx_per, nx, rank=None):
        real = slab_e_component_materials(materials, nx_per, nx, rank)
        if kind == "cell":
            del real
            return cell_owned_component_materials(materials)
        if kind == "x_lo_vacuum":     # rank 0's ghost row read as the vacuum it holds
            view = materials
        elif kind == "seam_own_row":  # every slab replicates its own first row
            def own(a):
                return a.at[0].set(a[1])
            view = MaterialArrays(own(materials.eps_r), own(materials.sigma), materials.mu_r,
                                  sigma_lumped=map_lumped(materials.sigma_lumped, own),
                                  eps_r_lumped=map_lumped(materials.eps_r_lumped, own))
        else:
            raise ValueError(kind)
        eps, sig = component_e_materials(view, (False, False, False))
        cell_eps, cell_sig = cell_owned_component_materials(materials)
        r = lax.axis_index("x") if rank is None else rank
        local = jnp.arange(materials.eps_r.shape[0])
        rows = r * nx_per - 1 + local
        model = ((local >= 1) & (local < local.size - 1) & (rows < nx))[:, None, None]
        return (tuple(jnp.where(model, e, c) for e, c in zip(eps, cell_eps)),
                tuple(jnp.where(model, s, c) for s, c in zip(sig, cell_sig)))
    return helper


@pytest.mark.parametrize("kind", ["cell", "x_lo_vacuum", "seam_own_row"])
def test_a_mutated_slab_view_is_caught(kind):
    """Each defect, the helper's calls kept, breaks the check above."""
    bad = _slab_view_mismatches(_mutant(kind), 3)
    print(f"[mutation {kind}] mismatching real cells {bad['model']}")
    assert bad["model"] > 0, bad


def _install(patch, kind):
    helper = _mutant(kind)
    for module in (common, distributed_v2, distributed_nu):
        patch.setattr(module, "slab_e_component_materials", helper)


# --------------------------------------------------------------------------
# a. lane parity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("lane", ["run", "fwd"])
@pytest.mark.parametrize("case", CASES)
def test_the_pec_cube_agrees_with_one_device(lane, case):
    _devices(2)
    if case == "cut":
        _realized_cut("pec", 2)
    got, floor = _parity("pec", lane, case, 2)
    _check(f"pec/{case}/{lane}/2", got, floor)


@pytest.mark.parametrize("lane", ["run", "fwd"])
@pytest.mark.parametrize("case", [
    c if c in ("faces", "cut", "yz") else pytest.param(c, marks=pytest.mark.slow)
    for c in CPML_CASES])
def test_the_cpml_box_agrees_with_one_device(lane, case):
    _devices(2)
    if case == "cut":
        _realized_cut("cpml", 2)
    got, floor = _parity("cpml", lane, case, 2)
    _check(f"cpml/{case}/{lane}/2", got, floor)


@pytest.mark.parametrize("lane", ["run", "fwd"])
def test_the_thin_xhi_absorber_reads_past_the_alignment_pad(lane):
    """An eps_r 4 fill through both x faces, 150 steps, two devices:
    23 x cells need one alignment row, and the x-hi absorber is two cells
    deep. With four cells and one pad row, a coefficient slice moved into
    the pad changes only the outermost absorber row and stays under the
    gate (8.2e-7 of the peak on the 25-cell box above); with two cells it
    reaches the probes. One cell is avoided: a one-cell hi-side absorber
    holding a conducting fill has a separate defect (ledger).

    Mac arm64, JAX 0.10.2, max |difference| / record peak, probe order:
    run: 1.43e-6 / 2.57e-6 / 3.60e-7, vacuum 3.21e-6 / 3.43e-6 / 1.01e-6,
    gates 1.29e-5 / 1.37e-5 / 4.02e-6;
    fwd: 7.33e-7 / 2.27e-6 / 8.25e-7, vacuum 2.52e-6 / 3.27e-6 / 7.09e-7,
    gates 1.01e-5 / 1.31e-5 / 2.84e-6 (max(FLOOR_FACTOR * floor, ROUNDING)).
    Replacing x_hi_edge by g in the uniform kernel's xhi_ slice gives
    1.61e-5 / 2.57e-6 / 4.25e-5: 1.249 / 0.187 / 10.566 times the gate.
    The fwd half is a parity check only. The same edit to
    _apply_cpml_e_local_nu's xhi leaves its records bit for bit unchanged:
    in this model the graded lane's x-hi E correction does nothing, since
    the inner row of a two-cell profile has c = 0 and the outer row is the
    face, which that lane zeroes every step.
    """
    devices = _devices(2)
    boundary = BoundarySpec(
        x=Boundary(lo="cpml", hi="cpml", lo_thickness=4, hi_thickness=2),
        y=Boundary(lo="cpml", hi="cpml", lo_thickness=4, hi_thickness=4),
        z=Boundary(lo="cpml", hi="cpml", lo_thickness=4, hi_thickness=4))
    kw = dict(freq_max=10e9, domain=_mm(16, 12, 12), dx=1e-3,
              boundary=boundary, cpml_layers=4)
    if lane == "fwd":
        kw["dx_profile"] = np.full(16, 1e-3)

    def record(material, dev):
        sim = Simulation(**kw)
        sim.add_source(_mm(4, 6, 6), "ez", waveform=WAVEFORM, amplitude_kind="current")
        sim.add_probe(_mm(8, 6, 6), "ez")
        sim.add_probe(_mm(6.5, 6, 5), "ey")
        sim.add_probe(_mm(15, 6, 6), "ez")
        if material:
            sim.add_material("m", eps_r=4.0)
            sim.add(Box((-1.0, 3e-3, 3e-3), (1.0, 9e-3, 7e-3)), material="m")
        grid = sim._build_grid() if lane == "run" else sim._build_nonuniform_grid()
        nx = grid.shape[0]
        pad_x = (-nx) % len(devices)
        assert nx % 2 == 1, nx
        assert pad_x == 1, (nx, pad_x)
        assert grid.pad_x_hi == grid.face_layers["x_hi"] == 2
        if lane == "run":
            out = sim.run(n_steps=150, skip_preflight=True, devices=dev)
        else:
            dist = {} if dev is None else dict(distributed=True, devices=dev)
            out = sim.forward(n_steps=150, skip_preflight=True, checkpoint=False, **dist)
        return np.asarray(out.time_series, np.float64)

    got = _rel(record(True, devices), record(True, None))
    floor = _rel(record(False, devices), record(False, None))
    gate = _gate(floor)
    assert all(g <= t for g, t in zip(got, gate)), (lane, got, gate, floor)


def _matrix(n_dev, boundaries, cases, mutation=None):
    """Run in a child with 4 host devices; RESULT lines, one per model."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1",
           "XLA_FLAGS": "--xla_force_host_platform_device_count=4"}
    path = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
            if p and not (Path(p) / "rfx").is_dir()]
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), *path])
    env = {k: v for k, v in env.items() if not k.lower().endswith("_proxy")}
    args = [sys.executable, "-W", "ignore", str(SCRIPT), "matrix", str(n_dev),
            ",".join(boundaries), ",".join(cases), mutation or "none"]
    run = subprocess.run(args, env=env, capture_output=True, text=True, timeout=1200)
    assert run.returncode == 0, run.stdout + run.stderr
    rows = [json.loads(line[7:]) for line in run.stdout.splitlines()
            if line.startswith("RESULT ")]
    assert rows, run.stdout + run.stderr
    return rows


def _matrix_main(n_dev, boundaries, cases, mutation):
    if mutation != "none":
        helper = _mutant(mutation)
        for module in (common, distributed_v2, distributed_nu):
            setattr(module, "slab_e_component_materials", helper)
    for boundary in boundaries:
        for case in cases:
            for lane in ("run", "fwd"):
                if case == "cut":
                    _realized_cut(boundary, n_dev)
                got, floor = _parity(boundary, lane, case, n_dev)
                print("RESULT " + json.dumps(dict(boundary=boundary, case=case, lane=lane,
                                                  n_dev=n_dev, got=got, floor=floor)),
                      flush=True)


@pytest.mark.parametrize("n_dev", [3, pytest.param(4, marks=pytest.mark.slow)])
def test_uneven_slabs_agree_with_one_device(n_dev):
    """13 cells over 3 slabs (pad 2) and 4 slabs (pad 3), PEC cube: the
    interface on a cut and the eps_r 4 block; the rest of the matrix is slow."""
    for row in _matrix(n_dev, ["pec"], ["cut", "eps4"]):
        _check(f"{row['boundary']}/{row['case']}/{row['lane']}/{row['n_dev']}",
               row["got"], row["floor"])


@pytest.mark.slow
@pytest.mark.parametrize("n_dev", [3, 4])
def test_the_whole_matrix_agrees_with_one_device(n_dev):
    rows = _matrix(n_dev, ["pec"], CASES) + _matrix(n_dev, ["cpml"], CPML_CASES)
    for row in rows:
        _check(f"{row['boundary']}/{row['case']}/{row['lane']}/{row['n_dev']}",
               row["got"], row["floor"])


# --------------------------------------------------------------------------
# b. mutations: the defect restored, the helper calls kept
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_mutation_the_cell_owned_coefficient_sends_every_material_case_red(monkeypatch):
    _devices(2)
    _install(monkeypatch, "cell")
    for boundary, cases in (("pec", CASES), ("cpml", CPML_CASES)):
        for case in cases:
            for lane in ("run", "fwd"):
                got, floor = _parity(boundary, lane, case, 2)
                gate = _gate(floor, case)
                print(f"[mutation cell {boundary}/{case}/{lane}] {['%.2e' % g for g in got]}")
                assert max(g / t for g, t in zip(got, gate)) > MUTATION_MARGIN, (case, got, gate)


@pytest.mark.slow
def test_mutation_a_seam_row_reading_its_own_row_sends_the_cut_case_red(monkeypatch):
    _devices(2)
    _install(monkeypatch, "seam_own_row")
    for boundary in ("pec", "cpml"):
        for lane in ("run", "fwd"):
            got, floor = _parity(boundary, lane, "cut", 2)
            gate = _gate(floor)
            print(f"[mutation seam {boundary}/cut/{lane}] {['%.2e' % g for g in got]}")
            assert max(g / t for g, t in zip(got, gate)) > MUTATION_MARGIN, (got, gate)
    # an interface off the cut does not see it
    got, floor = _parity("pec", "run", "faces", 2)
    assert all(g <= t for g, t in zip(got, _gate(floor))), got


@pytest.mark.slow
def test_mutation_the_psi_coefficient_on_the_cell_permittivity_sends_the_faces_case_red(
        monkeypatch):
    """The CPML E correction of both lanes takes the cell eps while the update
    keeps the four-cell mean (#1043's mismatch, restored): the dielectric that
    runs through the x-face absorber reads 1.5e-3 of the record peak."""
    _devices(2)
    # The low-level NU mutation below supplies this model's cell array.
    # Measure the vacuum reference before installing it, so the reference
    # cannot accidentally receive the dielectric model's permittivity.
    floors = {lane: _floor("cpml", lane, "vacuum", 2) for lane in ("run", "fwd")}

    def cell(materials, nx_per, nx, rank=None):
        slab_e_component_materials(materials, nx_per, nx, rank)   # the call is kept
        return cell_owned_component_materials(materials)

    # The uniform runner's CPML wrapper calls the name bound in its own module;
    # its E update reads _distributed_common's, which keeps the real helper.
    monkeypatch.setattr(distributed_v2, "slab_e_component_materials", cell)
    # The forward lane hands its CPML the means built before the loop; the
    # correction is given the cell array; all mean-builder calls stay in place.
    sim = _box("cpml", "fwd", "faces", 2)
    grid = sim._build_nonuniform_grid()
    slabs, _ = _staged(sim._assemble_materials_nu(grid)[0], 2)
    eps_cells = jnp.stack([m.eps_r for m in slabs])
    real_cpml = distributed_nu._apply_cpml_e_local_nu

    def cell_cpml(*args, **kwargs):
        # Keep every material helper call and replace only CPML's eps read.
        kwargs["eps_r"] = lax.dynamic_index_in_dim(
            eps_cells, lax.axis_index("x"), axis=0, keepdims=False)
        return real_cpml(*args, **kwargs)

    monkeypatch.setattr(distributed_nu, "_apply_cpml_e_local_nu", cell_cpml)
    for lane in ("run", "fwd"):
        got, _ = _parity("cpml", lane, "faces", 2)
        gate = _gate(floors[lane])
        print(f"[mutation psi cpml/faces/{lane}] {['%.2e' % g for g in got]}")
        assert max(g / t for g, t in zip(got, gate)) > MUTATION_MARGIN, (lane, got, gate)


# --------------------------------------------------------------------------
# c. vacuum and homogeneous models keep main's bits
# --------------------------------------------------------------------------

def _filled(lane, eps_r, sigma):
    kw = dict(freq_max=10e9, domain=(16e-3, 10e-3, 10e-3), dx=1e-3,
              boundary="cpml", cpml_layers=4)
    if lane == "fwd":
        kw["dz_profile"] = np.array([1.0, 0.8, 0.8, 1.0, 1.2, 1.0, 1.0, 0.8, 1.0, 1.0]) * 1e-3
    sim = Simulation(**kw)
    if eps_r != 1.0 or sigma:
        sim.add_material("d", eps_r=eps_r, sigma=sigma)
        sim.add(Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), material="d")
    sim.add_source((5e-3, 5e-3, 5e-3), "ez", waveform=WAVEFORM, amplitude_kind="current")
    sim.add_probe((11e-3, 5e-3, 5e-3), "ez")
    sim.add_probe((1.5e-3, 5e-3, 5e-3), "ey")
    return sim


def _filled_record(lane, eps_r, sigma):
    sim = _filled(lane, eps_r, sigma)
    devices = _devices(2)
    if lane == "run":
        out = sim.run(n_steps=80, skip_preflight=True, devices=devices)
        return [np.asarray(out.time_series), np.asarray(out.state.ez), np.asarray(out.state.hy)]
    out = sim.forward(n_steps=80, skip_preflight=True, checkpoint=False,
                      distributed=True, devices=devices)
    return [np.asarray(out.time_series)]


@pytest.mark.parametrize("lane", ["run", "fwd"])
@pytest.mark.parametrize("eps_r,sigma", [(1.0, 0.0), (4.4, 0.02)],
                         ids=["vacuum", "homogeneous"])
def test_vacuum_and_homogeneous_models_keep_mains_bits(monkeypatch, lane, eps_r, sigma):
    """The frozen in-process reference is main's rule: the same kernels fed
    the cell values (``cell_owned_component_materials``) for the E update and
    the CPML. The mean of four equal floats, pairwise, is that float, and the
    ghost and pad rows keep their cell's value, so the records must match
    byte for byte -- the fields too on the run lane."""
    new = _filled_record(lane, eps_r, sigma)

    def main_rule(materials, nx_per, nx, rank=None):
        slab_e_component_materials(materials, nx_per, nx, rank)   # traced, unused
        return cell_owned_component_materials(materials)

    with monkeypatch.context() as patch:
        for module in (common, distributed_v2, distributed_nu):
            patch.setattr(module, "slab_e_component_materials", main_rule)
        old = _filled_record(lane, eps_r, sigma)
    for a, b in zip(new, old):
        assert np.any(b), "a zero record cannot witness the coefficients"
        assert a.tobytes() == b.tobytes(), (lane, eps_r, int(np.count_nonzero(a != b)))


# --------------------------------------------------------------------------
# d. dispersive models share the component material rule
# --------------------------------------------------------------------------

def _debye_box(boundary, lane, kind="debye", waveform=WAVEFORM):
    sim = _box(boundary, lane, "vacuum", 2, waveform=waveform)
    poles = {}
    if kind in ("debye", "mixed"):
        poles["debye_poles"] = [DebyePole(delta_eps=1.0, tau=1e-11)]
    if kind in ("lorentz", "mixed"):
        poles["lorentz_poles"] = [lorentz_pole(1.0, 2 * np.pi * 3e9, 1e9)]
    sim.add_material("d", eps_r=3.0, **poles)
    # into the x-lo absorber on the CPML box, so the psi coefficient is read there
    lo = -1.0 if boundary == "cpml" else 5e-3
    sim.add(Box((lo, 3e-3, 3e-3), (7e-3, 9e-3, 9e-3)), material="d")
    sim.add_material("m", eps_r=4.0, sigma=0.05)
    sim.add(Box(_mm(9, 3, 3), _mm(11, 9, 9)), material="m")
    return sim


@pytest.mark.parametrize("boundary,lane", [
    ("pec", "run"), pytest.param("pec", "fwd", marks=pytest.mark.slow),
    ("cpml", "fwd"), ("cpml", "run")])
@pytest.mark.parametrize("kind", ["debye", "lorentz", "mixed"])
def test_a_dispersive_model_matches_one_device(boundary, lane, kind):
    """All ADE models share edge means and PEC-backed CPML faces with one device."""
    devices = _devices(2)
    n = STEPS[boundary]

    def record(dev):
        sim = _debye_box(boundary, lane, kind)
        if lane == "run":
            out = sim.run(n_steps=n, skip_preflight=True, devices=dev)
            if boundary == "cpml":
                # The old mixed slab body left a live outer plane (Ez read
                # 4.8e4 V/m at x-lo in the 80-step cross-lane measurement).
                for c, name in enumerate(("ex", "ey", "ez")):
                    field = np.asarray(getattr(out.state, name))
                    for axis in range(3):
                        if axis != c:
                            assert not np.any(np.take(field, [0, -1], axis=axis)), (kind, name, axis)
            return np.asarray(out.time_series, np.float64)
        kw = {} if dev is None else dict(distributed=True, devices=dev)
        return np.asarray(sim.forward(n_steps=n, skip_preflight=True, checkpoint=False,
                                      **kw).time_series, np.float64)

    actual, expected = record(devices), record(None)
    if lane == "run" and boundary == "cpml":
        peak_ulp = _peak_ulp(actual, expected)
        print(f"[{kind}/cpml/run] peak_ulp={peak_ulp:.3f}")
        assert peak_ulp <= 9, (kind, peak_ulp)
    got = _rel(actual, expected)
    floor = _rel(_record(boundary, lane, "vacuum", 2, devices), _single(boundary, lane, "vacuum", 2))
    _check(f"{kind}/{boundary}/{lane}", got, floor)


@pytest.mark.parametrize("kind", ["debye", "lorentz", "mixed"])
def test_a_large_dispersive_objective_has_finite_gradients(kind):
    """The raw squared-field objective exercises the coefficient reverse pass."""
    devices = _devices(2)
    sim = _debye_box("cpml", "fwd", kind, waveform=lambda t: 32.0 * WAVEFORM(t))
    grid = sim._build_nonuniform_grid()
    mats = sim._assemble_materials_nu(grid)[0]
    eps, sigma = jnp.asarray(mats.eps_r), jnp.asarray(mats.sigma)

    def evaluate(**kw):
        def objective(e, s):
            ts = sim.forward(eps_override=e, sigma_override=s, n_steps=60,
                             skip_preflight=True, checkpoint=False, **kw).time_series
            return jnp.sum(ts ** 2)
        return jax.value_and_grad(objective, argnums=(0, 1))(eps, sigma)

    single, dist = evaluate(), evaluate(distributed=True, devices=devices)
    assert float(single[0]) > 1e15, float(single[0])
    for name, actual, expected in zip(("loss", "eps", "sigma"),
                                      (dist[0], *dist[1]), (single[0], *single[1])):
        a, b = np.asarray(actual), np.asarray(expected)
        assert np.isfinite(a).all() and np.isfinite(b).all(), (kind, name)
        relative = float(np.max(np.abs(a.astype(np.float64) - b)) / np.max(np.abs(b)))
        print(f"[large-dispersive/{kind}/{name}] objective={float(single[0]):.9e} relative={relative:.9e}")
        assert relative <= 1e-4, (kind, name, relative)


#: a graded x profile for the #1302 box: uniform 1 mm next to both absorbers,
#: 0.9-1.1 mm cells between (24 mm in all)
GRADED_1302 = np.array([1.0] * 7 + [0.9, 1.1, 0.95, 1.05, 0.9, 1.1, 1.05, 0.95, 1.1, 0.9]
                       + [1.0] * 7) * 1e-3


def _cpml_1302_box(kind, lane):
    """The #1302 box: a pulse in a 24 x 12 x 12 mm CPML box (6 layers) on an
    eps_inf = 4 block carrying one Debye or Lorentz pole, 800 steps."""
    kw = {} if lane == "run" else {
        "dx_profile": np.full(24, 1e-3) if lane == "fwd" else GRADED_1302}
    sim = Simulation(freq_max=10e9, domain=(24e-3, 12e-3, 12e-3), dx=1e-3,
                     boundary="cpml", cpml_layers=6, **kw)
    sim.add_source(_mm(6, 6, 6), "ez", waveform=WAVEFORM, amplitude_kind="current")
    sim.add_probe(_mm(19, 6, 6), "ez")
    poles = {"vacuum": {},
             "debye": {"debye_poles": [DebyePole(delta_eps=1.0, tau=1e-11)]},
             "lorentz": {"lorentz_poles": [LorentzPole(
                 omega_0=2 * np.pi * 8e9, delta=2 * np.pi * 1e9,
                 kappa=(2 * np.pi * 8e9) ** 2)]}}[kind]
    sim.add_material("m", eps_r=4.0, **poles)
    sim.add(Box(_mm(10, 3, 3), _mm(16, 9, 9)), material="m")
    return sim


def _cpml_1302_record(kind, lane, devices=None):
    sim = _cpml_1302_box(kind, lane)
    if lane == "run":
        out = sim.run(n_steps=800, skip_preflight=True, devices=devices)
    else:
        kw = {} if devices is None else dict(distributed=True, devices=devices)
        out = sim.forward(n_steps=800, skip_preflight=True, checkpoint=False, **kw)
    return np.asarray(out.time_series, np.float64)


@pytest.mark.parametrize("lane", ["run", "fwd", "fwd_graded"])
def test_a_dispersive_cpml_box_stays_finite_and_matches_one_device(lane):
    """#1302: two devices, a Debye or Lorentz block in a CPML box. On main
    4725b748 the two-device run() grew without bound (probe non-finite from
    step 418; the one-device run is finite). Rows outside the domain now take
    the vacuum-cell ADE coefficients and dispersive runs keep the one-device
    PEC backing of the absorber's outer node planes. Bar: finite, and within
    FLOOR_FACTOR x the same box's vacuum two-vs-one-device agreement,
    measured here. (Graded run(devices=) refuses CPML with poles; the graded
    lane is forward(distributed=True).)"""
    devices = _devices(2)
    floor = _rel(_cpml_1302_record("vacuum", lane, devices),
                 _cpml_1302_record("vacuum", lane))
    for kind in ("debye", "lorentz"):
        two = _cpml_1302_record(kind, lane, devices)
        one = _cpml_1302_record(kind, lane)
        assert np.isfinite(one).all(), (kind, lane, "one device")
        bad = np.flatnonzero(~np.isfinite(two).all(axis=1))
        assert bad.size == 0, (kind, lane, f"two devices non-finite from step {bad[:1] + 1}")
        _check(f"cpml1302/{kind}/{lane}", _rel(two, one), floor)


# --------------------------------------------------------------------------
# e. the permittivity gradient
# --------------------------------------------------------------------------

@pytest.mark.parametrize("boundary", ["pec", pytest.param("cpml", marks=pytest.mark.slow)])
def test_the_permittivity_gradient_matches_one_device(boundary):
    devices = _devices(2)
    sim = _box(boundary, "fwd", "eps4", 2)
    grid = sim._build_nonuniform_grid()
    drawn = sim._assemble_materials_nu(grid)[0]
    eps = jnp.asarray(np.asarray(drawn.eps_r))
    sig = jnp.asarray(np.asarray(drawn.sigma))

    def loss(e, s, **kw):
        ts = sim.forward(eps_override=e, sigma_override=s, n_steps=STEPS[boundary],
                         skip_preflight=True, checkpoint=False, **kw).time_series
        return jnp.sum(ts ** 2)

    single = jax.grad(loss, argnums=(0, 1))(eps, sig)
    dist = jax.grad(lambda e, s: loss(e, s, distributed=True, devices=devices),
                    argnums=(0, 1))(eps, sig)
    # Both design variables: the four-cell mean carries eps and sigma alike.
    for name, g_s, g_d in zip(("eps", "sigma"), single, dist):
        g_s, g_d = np.asarray(g_s, np.float64), np.asarray(g_d, np.float64)
        rel = float(np.max(np.abs(g_d - g_s)) / np.max(np.abs(g_s)))
        ulp = float(np.max(np.abs(g_d - g_s)) / np.spacing(np.float32(np.max(np.abs(g_s)))))
        print(f"[gradient/{boundary}/{name}] |g_dist - g_single| / peak {rel:.2e} ({ulp:.0f} ULP)")
        assert np.isfinite(g_d).all() and np.any(g_s), name
        assert rel <= GRAD_RTOL, (name, rel)


# --------------------------------------------------------------------------
# cross-trace on a dielectric model (PI 2026-09-23: <= 9 ULP of each peak)
# --------------------------------------------------------------------------

def _peak_ulp(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.max(np.abs(got - want)) / np.spacing(np.float32(np.max(np.abs(want)))))


@pytest.mark.slow
@pytest.mark.parametrize("lane", ["run", "fwd"])
def test_a_dielectric_model_gives_the_plain_bits_in_every_trace_context(lane):
    devices = _devices(2)
    if lane == "run":
        def forward(a):
            sim = _box("pec", "run", "eps4", 2, waveform=lambda t: a * WAVEFORM(t))
            out = sim.run(n_steps=STEPS["pec"], devices=devices, skip_preflight=True)
            return out.time_series, out.state.ez, out.state.hy
        x0, tangent = 1.0, 1.0
        batch = jnp.array([1.0, 1.0], jnp.float32)
    else:
        sim = _box("pec", "fwd", "eps4", 2)
        grid = sim._build_nonuniform_grid()
        x0 = jnp.asarray(np.asarray(sim._assemble_materials_nu(grid)[0].eps_r))
        tangent = jnp.ones_like(x0)
        batch = jnp.stack([x0, x0])

        def forward(e):
            return (sim.forward(eps_override=e, n_steps=STEPS["pec"], skip_preflight=True,
                                checkpoint=False, distributed=True,
                                devices=devices).time_series,)

    plain = [np.asarray(o) for o in forward(x0)]

    def value_and_grad():
        def loss(x):
            out = forward(x)
            return jnp.sum(out[0] ** 2), out
        return jax.value_and_grad(loss, has_aux=True)(x0)[0][1]

    contexts = {
        "jvp": lambda: jax.jvp(forward, (x0,), (tangent,))[0],
        "vjp": lambda: jax.vjp(forward, x0)[0],
        "value_and_grad": value_and_grad,
        "vmap": lambda: tuple(o[0] for o in jax.vmap(forward)(batch)),
        "stop_gradient": lambda: jax.jvp(lambda x: forward(lax.stop_gradient(x)),
                                         (x0,), (tangent,))[0],
    }
    worst = {}
    for name, context in contexts.items():
        worst[name] = max(_peak_ulp(o, p) for o, p in zip(context(), plain))
    print(f"[cross-trace/{lane}] ULP of each peak, JAX {jax.__version__}: {worst}")
    # Binding on the CI build's JAX, as the PI scoped the rule (2026-09-23):
    # <= 9 ULP. There (arm64 0.6.2) the run lane is bitwise, main too, and
    # the forward lane reads 4 ULP under jvp/vjp/value_and_grad -- main the
    # same 4, and main's VACUUM box 13. Measured, not gated, elsewhere: on
    # 0.10.2 (Mac) the run lane reads 10 ULP under vjp/vmap, main 10.4, main's
    # vacuum 31. Compiler effects, not this lane's material rule.
    if jax.__version__ == CI_JAX:
        assert all(v <= 9 for v in worst.values()), worst


# --------------------------------------------------------------------------
# f. two processes (slow, Linux)
# --------------------------------------------------------------------------

def _worker(address, rank, output):
    rank = int(rank)
    reference = address == "reference"
    if not reference:
        assert jax.process_count() == 2 and jax.local_device_count() == 1
    run = _box("pec", "run", "cut", 2).run(n_steps=STEPS["pec"], skip_preflight=True,
                                           devices=jax.devices())
    sim = _box("cpml", "fwd", "faces", 2)
    grid = sim._build_nonuniform_grid()
    # an x-sharded design, the form a design spread over processes takes
    eps = sim.shard_distributed_override(np.asarray(sim._assemble_materials_nu(grid)[0].eps_r))

    def f(e):
        return sim.forward(eps_override=e, n_steps=STEPS["cpml"], skip_preflight=True,
                           checkpoint=False, distributed=True,
                           devices=jax.devices()).time_series

    trace = f(eps)
    grad = jax.grad(lambda e: jnp.log(jnp.sum(f(e) ** 2)))(eps)
    assert trace.is_fully_replicated and grad.sharding == eps.sharding
    np.save(Path(output) / f"run-{rank}.npy", np.asarray(run.time_series))
    np.save(Path(output) / f"fwd-{rank}.npy", np.asarray(trace))
    for shard in grad.addressable_shards:
        start = shard.index[0].start or 0
        np.save(Path(output) / f"grad-{start:06d}.npy", np.asarray(shard.data))
    print(f"rank={rank} ok", flush=True)
    if not reference:
        jax.distributed.shutdown()


@pytest.mark.slow
def test_two_processes_give_the_one_process_records(tmp_path):
    if sys.platform != "linux":
        pytest.skip("requires Linux: jax.distributed gRPC bind fails on macOS")
    if tuple(int(v) for v in jax.__version__.split(".")[:2]) < (0, 5):
        pytest.skip("JAX 0.4.x: 'Multiprocess computations aren't implemented on "
                    "the CPU backend' (test_distributed_multihost.py's two-process "
                    "test fails the same way there)")
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1",
           "PYTHONPATH": str(ROOT)}
    env = {k: v for k, v in env.items() if not k.lower().endswith("_proxy")}
    reference = tmp_path / "reference"
    reference.mkdir()
    one = subprocess.run(
        [sys.executable, "-W", "ignore", str(SCRIPT), "worker", "reference", "0", str(reference)],
        cwd=ROOT, env={**env, "XLA_FLAGS": "--xla_force_host_platform_device_count=2"},
        capture_output=True, text=True, timeout=600)
    assert one.returncode == 0, one.stdout + one.stderr
    with socket.socket() as listener:
        listener.bind(("localhost", 0))
        address = f"localhost:{listener.getsockname()[1]}"
    processes, handles = [], []
    deadline = time.monotonic() + 600
    try:
        for rank in range(2):
            handle = (tmp_path / f"worker-{rank}.log").open("w")
            handles.append(handle)
            processes.append(subprocess.Popen(
                [sys.executable, "-W", "ignore", str(SCRIPT), "worker", address, str(rank),
                 str(tmp_path)],
                cwd=ROOT, env={**env, "XLA_FLAGS": "--xla_force_host_platform_device_count=1"},
                stdout=handle, stderr=subprocess.STDOUT))
        for process in processes:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for handle in handles:
            handle.close()
    logs = "\n".join((tmp_path / f"worker-{r}.log").read_text() for r in range(2))
    assert all(p.returncode == 0 for p in processes), logs
    for name in ("run", "fwd"):
        want = np.load(reference / f"{name}-0.npy")
        for rank in range(2):
            got = np.load(tmp_path / f"{name}-{rank}.npy")
            assert got.shape == want.shape
            ulp = _peak_ulp(got, want)
            print(f"[two processes] {name} rank {rank}: {ulp:.2f} ULP of the peak "
                  f"(bitwise {got.tobytes() == want.tobytes()})")
            assert ulp <= 9, (name, rank, ulp)
    want = np.concatenate([np.load(s) for s in sorted(reference.glob("grad-*.npy"))])
    shards = sorted(tmp_path.glob("grad-*.npy"))
    assert len(shards) == 2, logs
    got = np.concatenate([np.load(s) for s in shards])
    assert got.shape == want.shape
    ulp = _peak_ulp(got, want)
    print(f"[two processes] eps gradient: {ulp:.2f} ULP of the peak")
    assert ulp <= 9, ulp


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "matrix":
        n_dev, boundaries, cases, mutation = args
        _matrix_main(int(n_dev), boundaries.split(","), cases.split(","), mutation)
    elif command == "worker":
        _worker(*args)
    else:
        raise AssertionError(command)
