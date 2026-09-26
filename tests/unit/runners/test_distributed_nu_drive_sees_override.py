"""A current source on the distributed graded forward is driven through the
permittivity the field is stepped with, an override included (#1279).

A soft current source adds ``Cb * I(t) / dV`` to its E edge every step, and
``Cb = dt / (eps + sigma*dt/2)`` carries the permittivity of that edge -- the
mean over the four cells the edge touches (#1210). The same model can be
declared two ways: its permittivity drawn as materials, or replaced through
``eps_override`` / ``sigma_override``, which is how ``forward()`` hands a design
variable to the solver. ``forward(distributed=True)`` stepped the field with the
override but built every source's ``Cb`` from the materials as drawn, so a
source standing in overridden material was driven ``Cb_drawn / Cb_override``
too hard: 3.02x for eps_r 3.38 -> 10.2 (distributed/single-device probe ratio
3.040 on two CPU devices), and a permittivity gradient taken through the
override missed the drive's own derivative. The single-device graded lane was
fixed in #1280; this lane now builds ``Cb`` inside its jitted program from the
slabs its E update receives, so an x-sharded override spread over several
processes, or a traced one, reaches the drive as well.

What is checked (two CPU devices; the x axis is 17 cells, split 9 + 9 after one
pad cell, so x = 7 mm is global i = 9, the first real cell of the second
device, whose i-1 cells belong to the first):

a. The drive each current source receives, read from the field after the first
   step (one ampere, a probe on the source edge), times the edge's control
   volume, equals the ``Cb`` the single-device E update multiplies that edge
   by under the same override -- ex, ey and ez, on the seam cell and inside a
   slab, for a local, an x-sharded and a vmapped override (random eps_r and
   sigma, so every one of the four cells counts). Measured <= 1.6e-7 relative
   on JAX 0.10.2 / 0.6.2 / 0.4.33; gate 1e-6. The reference is the
   single-device kernel because this lane's own E update still takes the
   owning cell's value, not the four-cell mean (the #1210 refusal,
   ``test_distributed_e_coefficients_are_still_cell_owned.py``); on these
   random cells the two differ by 0.41-1.35x, before and after this fix.
a2. The gradient with respect to an x-sharded override equals the one with
   respect to the same local array cell by cell, with the seam source's i-1
   cells on the other device: the drive's derivative there comes back through
   the halo transpose. Its size at those cells is recorded against the gate.
b. A one-material slab raised by e^0.1 through the override and the slab drawn
   that way give one probe record: time-domain least-squares ratio and 8 GHz
   ratio within 9 float32 ULP of 1 (measured <= 1.8e-7 and <= 3.5e-7 at d8629a2d,
   JAX 0.10.2; the gate is 1.07e-6). The per-sample difference is NOT gated at 9 ULP
   of the peak: the override's drive is computed in float32 inside the
   program, the drawn one in float64 on the host, 1 ULP apart here, and that
   rounding difference grows through the 200-step record to 18-40 ULP of the
   peak (forcing the host value into the program gives a bit-identical
   record).
c. The distributed lane against the single-device lane (#1280), a source in
   material overridden from eps_r 3.38 to 10.2: the probe record's
   least-squares ratio within 9 ULP of 1, and d ln|E(8 GHz)|^2 / d ln eps_r
   through a traced override within 1e-5 (measured <= 8.6e-7), also with 40
   warm-up steps and checkpoint_every=30.
d. Opt-in (``RFX_MAIN_REF=<commit>`` or ``RFX_MAIN_CHECKOUT=<dir>``): a 'field'
   source under an override, and a current source without one, are bitwise
   what main computes -- main's code run in a separate process.
e. Slow, Linux only: the x-sharded override across two processes gives the
   one-process record and gradients.

Mutations (slow): the drive handed the arrays as drawn, every builder call
kept, sends a, b and c red; the drive reading the seam edge's own row for its
i-1 cells sends a red.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import functools
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import time

import jax

# A two-process worker initializes the distributed runtime before anything
# touches JAX's backend (as in test_distributed_nu_forward_multiprocess.py).
if (__name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "worker"
        and sys.argv[2] != "reference"):
    jax.distributed.initialize(sys.argv[2], 2, int(sys.argv[3]),
                               initialization_timeout=30)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from rfx import Simulation  # noqa: E402
from rfx.core.yee import (  # noqa: E402
    FDTDState, MaterialArrays, curl_h_nu, update_e_nu,
)
from rfx.geometry import Box  # noqa: E402
from rfx.nonuniform import position_to_index  # noqa: E402
from rfx.runners import distributed_nu as nu  # noqa: E402
from rfx.sources import GaussianPulse  # noqa: E402

pytestmark = pytest.mark.distributed
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve()

DX = 1.0e-3
NX, NY = 12, 7                       # mm; the grid adds 2 CPML cells a face
#: graded z: two 0.8 mm cells, then 1 mm
DZ = np.array([0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0]) * DX
CPML = 2
N_DEVICES = 2
#: rank 1's first real cell (global i = 9); z = 1.6 mm lies between a 0.8 mm
#: and a 1.0 mm cell, so the transverse control-volume width is graded
SEAM = (7.0e-3, 3.0e-3, 1.6e-3)
INNER = (10.0e-3, 3.0e-3, 1.6e-3)    # global i = 12, inside rank 1
PROBE = (9.0e-3, 3.0e-3, 3.6e-3)
SLAB = ((3.0e-3, 1.0e-3, 0.8e-3), (11.0e-3, 6.0e-3, 5.6e-3))
PULSE = GaussianPulse(f0=8e9, bandwidth=0.8)
EPS_LO, EPS_HI = 3.38, 10.2
LN_STEP = 0.1                        # test b: eps_r x e^0.1
N_RECORD = 200                       # the whole pulse passes the probe
F_BIN = 8e9

# Pre-declared gates.
CB_RTOL = 1e-6                       # a: measured <= 1.9e-7
RATIO_TOL = 9 * 2.0 ** -23           # b, c: 9 float32 ULP of 1
GRAD_RTOL = 1e-5                     # c: measured <= 6.5e-7
PEAK_ULP = 9                         # a2, d, e: per array, against its peak

_EDGES = [(where, comp) for where in ("seam", "inner")
          for comp in ("ex", "ey", "ez")]
_AT = {"seam": SEAM, "inner": INNER}


def _one_ampere(t):
    """1 A at every step: after step 0 the source edge holds the drive
    coefficient itself (the fields start at zero)."""
    return jnp.ones_like(t)


def _board(eps_slab=EPS_LO, sigma_slab=0.0, *, sources=(("ey", SEAM, "current"),),
           probes=(("ey", PROBE),), waveform=PULSE):
    sim = Simulation(freq_max=15e9, domain=(NX * DX, NY * DX, float(DZ.sum())),
                     dx=DX, dz_profile=DZ, boundary="cpml", cpml_layers=CPML)
    sim.add_material("slab", eps_r=float(eps_slab), sigma=float(sigma_slab))
    sim.add(Box(*SLAB), material="slab")
    for comp, pos, kind in sources:
        sim.add_source(pos, comp, waveform=waveform, amplitude_kind=kind)
    for comp, pos in probes:
        sim.add_probe(pos, comp)
    return sim


def _edge_board():
    """A current source and a probe on each of the six edges of test a,
    over a drawn eps_r 4.4 / sigma 0.02 slab that the override replaces."""
    return _board(4.4, 0.02, waveform=_one_ampere,
                  sources=[(c, _AT[w], "current") for w, c in _EDGES],
                  probes=[(c, _AT[w]) for w, c in _EDGES])


def _drawn(sim):
    grid = sim._build_nonuniform_grid()
    return sim._assemble_materials_nu(grid)[0]


def _random_design(shape, seed=1279):
    """Random eps_r and sigma: the four cells around every edge differ."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(1.2, 4.8, shape).astype(np.float32),
            rng.uniform(0.0, 2.0, shape).astype(np.float32))


def _place(sim, host, form):
    return jnp.asarray(host) if form == "local" else sim.shard_distributed_override(host)


def _peak_ulp(want, got):
    want, got = np.asarray(want, np.float64), np.asarray(got, np.float64)
    assert want.shape == got.shape and np.isfinite(want).all() and np.isfinite(got).all()
    peak = np.float32(np.max(np.abs(want)))
    return float(np.max(np.abs(want - got)) / np.spacing(peak))


def _dft(ts, dt, f):
    t = jnp.arange(ts.shape[0]) * dt
    return jnp.sum(ts * jnp.exp(-2j * jnp.pi * f * t)) * dt


def _ls_ratio(num, den):
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    return float(np.dot(num, den) / np.dot(den, den))


def _dist(sim, **kw):
    return sim.forward(distributed=True, devices=jax.devices("cpu")[:N_DEVICES],
                       skip_preflight=True, **kw)


# --------------------------------------------------------------------------
# a. the drive coefficient each edge receives, against the stepper's Cb
# --------------------------------------------------------------------------

def _control_volume(grid, ijk, comp):
    """The E node's control volume from the realized cell sizes: the width
    along the component, the mean of the two adjacent widths across it
    (#672). Written out here, not taken from the source builder."""
    axis = "xyz".index(comp[1])
    widths = []
    for a, d in enumerate((grid.dx_arr, grid.dy_arr, grid.dz)):
        d = np.asarray(d, np.float64)
        n = int(ijk[a])
        assert n >= 1, "interior edges only"
        widths.append(d[n] if a == axis else 0.5 * (d[n - 1] + d[n]))
    return float(np.prod(widths))


def _stepper_cb(grid, eps, sigma):
    """The Cb the single-device E update multiplies each edge by: run
    ``update_e_nu`` once from E = 0 under a random H, divide by the curl."""
    rng = np.random.default_rng(7)
    h = [jnp.asarray(rng.standard_normal(grid.shape).astype(np.float32))
         for _ in range(3)]
    zero = jnp.zeros(grid.shape, jnp.float32)
    state = FDTDState(ex=zero, ey=zero, ez=zero, hx=h[0], hy=h[1], hz=h[2],
                      step=jnp.int32(0))
    mats = MaterialArrays(jnp.asarray(eps), jnp.asarray(sigma),
                          jnp.ones(grid.shape, jnp.float32))
    new = update_e_nu(state, mats, grid.dt, grid.inv_dx, grid.inv_dy, grid.inv_dz)
    curl = curl_h_nu(*h, grid.inv_dx, grid.inv_dy, grid.inv_dz)
    return {c: np.asarray(getattr(new, c), np.float64) / np.asarray(curl[a], np.float64)
            for a, c in enumerate(("ex", "ey", "ez"))}


def _drive_over_stepper(form):
    """``{(where, comp[, member]): Cb_drive / Cb_step - 1}`` under a random
    override (``vmap``: a batch of two designs, each against its own)."""
    sim = _edge_board()
    grid = sim._build_nonuniform_grid()
    sg = nu.build_sharded_nu_grid(grid, N_DEVICES)
    designs = [_random_design(grid.shape, seed) for seed in (1279, 1280)]
    idx = {w: position_to_index(grid, _AT[w]) for w in _AT}
    # realized, not declared: the seam edge is rank 1's first real cell, its
    # i-1 cells differ from its own, and the inner edge is inside the slab
    assert idx["seam"][0] == sg.nx_per_rank and idx["inner"][0] > sg.nx_per_rank
    i, j, k = idx["seam"]
    for eps, sigma in designs:
        assert eps[i - 1, j, k] != eps[i, j, k] and sigma[i - 1, j, k] != sigma[i, j, k]

    def first_step(e, s):
        return _dist(sim, eps_override=e, sigma_override=s, n_steps=1).time_series[0]

    if form == "vmap":
        # A batch of two: vmap over this lane needs a batch the device count
        # divides (a pre-existing staging limit, main included).
        firsts = jax.vmap(first_step)(*(jnp.stack([jnp.asarray(d[n]) for d in designs])
                                         for n in (0, 1)))
    else:
        designs = designs[:1]
        firsts = [first_step(_place(sim, designs[0][0], form),
                             _place(sim, designs[0][1], form))]
    out = {}
    for member, ((eps, sigma), first) in enumerate(zip(designs, firsts)):
        first = np.asarray(first, np.float64)
        cb_step = _stepper_cb(grid, eps, sigma)
        for n, (w, c) in enumerate(_EDGES):
            key = (w, c) if form != "vmap" else (w, c, member)
            out[key] = first[n] * _control_volume(grid, idx[w], c) / cb_step[c][idx[w]] - 1.0
    return out


@pytest.mark.parametrize("form", ["local", "sharded", "vmap"])
def test_every_edge_is_driven_through_the_steppers_cb_under_an_override(form):
    rel = _drive_over_stepper(form)
    print(f"[a/{form}] Cb_drive/Cb_step - 1: "
          + ", ".join(f"{' '.join(map(str, key))} {v:+.2e}" for key, v in rel.items()))
    assert max(abs(v) for v in rel.values()) <= CB_RTOL, rel


# --------------------------------------------------------------------------
# a2. the seam drive's gradient comes back to the device that owns its cells
# --------------------------------------------------------------------------

def _log_energy(ts):
    """ln of the probe records' summed squares (scale-free; the raw sum is
    ~1e16 (V/m)^2 on these boards -- see test_a_large_objective_...)."""
    return jnp.log(jnp.sum(ts ** 2))


def _seam_gradient(form, frozen_drive=False):
    """d ln(sum of squared probe samples)/d(eps_r, sigma), current source on
    the seam edge (Ey: its i-1 cells are the other device's)."""
    sim = _board(4.4, 0.02, probes=(("ey", PROBE), ("ey", SEAM)))
    grid = sim._build_nonuniform_grid()
    designs = [_place(sim, a, form) for a in _random_design(grid.shape)]

    def loss(e, s):
        ts = _dist(sim, eps_override=e, sigma_override=s, n_steps=40).time_series
        return _log_energy(ts)

    with _frozen_drive() if frozen_drive else nullcontext():
        g = jax.grad(loss, argnums=(0, 1))(*designs)
    if form == "sharded":
        assert all(got.sharding == d.sharding for got, d in zip(g, designs))
    return [np.asarray(a)[:grid.shape[0]] for a in g], position_to_index(grid, SEAM)


@contextmanager
def _frozen_drive():
    """The drive's Cb read through ``stop_gradient``: what the gradient is
    without the drive's own derivative (a witness of its size)."""
    real = nu.material_drive_scales

    def frozen(eps_r, sigma, mesh, drives, dt):
        return real(jax.lax.stop_gradient(eps_r), jax.lax.stop_gradient(sigma),
                    mesh, drives, dt)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nu, "material_drive_scales", frozen)
        yield


def test_the_seam_drives_gradient_reaches_the_cells_on_the_other_device():
    (g_eps, g_sig), (i, j, k) = _seam_gradient("local")
    (s_eps, s_sig), _ = _seam_gradient("sharded")
    (f_eps, f_sig), _ = _seam_gradient("local", frozen_drive=True)
    ulp = {"eps": _peak_ulp(g_eps, s_eps), "sigma": _peak_ulp(g_sig, s_sig)}
    # The drive's own share of the gradient at the seam edge's i-1 cells
    # (device 0's), in ULP of the gradient's peak: what the sharded gradient
    # would lose there if the halo transpose dropped the ghost-row cotangent.
    cells = (i - 1, j, slice(k - 1, k + 1))
    share = {name: float(np.max(np.abs(full[cells] - frozen[cells])))
             / float(np.spacing(np.float32(np.max(np.abs(full)))))
             for name, full, frozen in (("eps", g_eps, f_eps), ("sigma", g_sig, f_sig))}
    print(f"[a2] sharded vs local gradient, ULP of the peak: {ulp}; the drive's "
          f"share at the seam's i-1 cells, ULP of the peak: {share}")
    assert all(v <= PEAK_ULP for v in ulp.values()), ulp
    assert all(v > 100 * PEAK_ULP for v in share.values()), share


# --------------------------------------------------------------------------
# b. eps_r x e^0.1 by override and by materials: one record
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _raised_records(form, mutated=False):
    built = _board(EPS_LO * np.exp(LN_STEP))
    drawn_sim = _board(EPS_LO)
    ovr = np.asarray(_drawn(built).eps_r)
    grid = built._build_nonuniform_grid()
    i, j, k = position_to_index(grid, SEAM)
    # realized: the source's four cells are the raised slab in the override
    # and the plain one as drawn
    assert np.all(ovr[i - 1:i + 1, j, k - 1:k + 1] == np.float32(EPS_LO * np.exp(LN_STEP)))
    assert np.all(np.asarray(_drawn(drawn_sim).eps_r)[i - 1:i + 1, j, k - 1:k + 1]
                  == np.float32(EPS_LO))
    ts_b = np.asarray(_dist(built, n_steps=N_RECORD).time_series)[:, 0]
    ctx = _drive_from_drawn(drawn_sim) if mutated else nullcontext()
    with ctx:
        ts_o = np.asarray(_dist(drawn_sim, eps_override=_place(drawn_sim, ovr, form),
                                n_steps=N_RECORD).time_series)[:, 0]
    return ts_b, ts_o, float(grid.dt)


def _raised_ratios(form, mutated=False):
    ts_b, ts_o, dt = _raised_records(form, mutated)
    f_ratio = complex(_dft(jnp.asarray(ts_o), dt, F_BIN) / _dft(jnp.asarray(ts_b), dt, F_BIN))
    return _ls_ratio(ts_o, ts_b), f_ratio, _peak_ulp(ts_b, ts_o)


@pytest.mark.parametrize("form", ["local", "sharded"])
def test_raising_eps_r_by_override_changes_the_slab_not_the_drive(form):
    ls, f_ratio, ulp = _raised_ratios(form)
    print(f"[b/{form}] override/drawn probe record: least-squares ratio "
          f"{ls:.9f}, {F_BIN / 1e9:g} GHz ratio {f_ratio:.9f} "
          f"(|ratio - 1| {abs(f_ratio - 1.0):.2e}); per sample "
          f"{ulp:.2f} ULP of the peak (not gated); a drive from the drawn "
          f"arrays predicts e^0.1 = {np.exp(LN_STEP):.6f}")
    assert abs(ls - 1.0) <= RATIO_TOL, ls
    assert abs(f_ratio - 1.0) <= RATIO_TOL, f_ratio


# --------------------------------------------------------------------------
# c. the distributed lane against the single-device lane (#1280)
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _lane(lane, form=None, mutated=False, warmup=0, checkpoint=None):
    """``(probe record, d ln|E(8 GHz)|^2 / d ln eps_r)`` for a source in a
    drawn eps_r 3.38 slab that an override raises, with everything around it,
    to 10.2 -- one material, so the two lanes' E updates coincide.
    ``warmup`` / ``checkpoint`` are forward's ``n_warmup`` / ``checkpoint_every``:
    the warm-up steps take their own drive table, which the runner scales too."""
    kw = dict(n_warmup=warmup, checkpoint_every=checkpoint)
    sim = _board(EPS_LO)
    grid = sim._build_nonuniform_grid()
    full = np.full(grid.shape, EPS_HI, np.float32)
    i, j, k = position_to_index(grid, SEAM)
    assert np.all(np.asarray(_drawn(sim).eps_r)[i - 1:i + 1, j, k - 1:k + 1]
                  == np.float32(EPS_LO))

    def record(a):
        if lane == "single":
            return sim.forward(eps_override=jnp.asarray(full) * jnp.exp(a),
                               n_steps=N_RECORD, skip_preflight=True, **kw).time_series[:, 0]
        design = _place(sim, full, form)
        return _dist(sim, eps_override=design * jnp.exp(a), n_steps=N_RECORD,
                     **kw).time_series[:, 0]

    def objective(a):
        return jnp.log(jnp.abs(_dft(record(a), grid.dt, F_BIN)) ** 2)

    ctx = _drive_from_drawn(sim) if mutated else nullcontext()
    with ctx:
        ts = np.asarray(record(0.0))
        grad = float(jax.grad(objective)(0.0))
    return ts, grad


# (0, None): the plain run. (40, 30): 40 warm-up steps with their own drive
# table, and segmented remat; deleting the warm-up table's scaling moved the
# record 1.0096 and the gradient -12.3 % (#1279 review), unseen before.
@pytest.mark.parametrize("warmup,checkpoint", [(0, None), (40, 30)])
@pytest.mark.parametrize("form", ["local", "sharded"])
def test_the_distributed_lane_drives_the_overridden_source_as_the_single_device_lane(
        form, warmup, checkpoint):
    ts_s, g_s = _lane("single", warmup=warmup, checkpoint=checkpoint)
    ts_d, g_d = _lane("dist", form, warmup=warmup, checkpoint=checkpoint)
    ls = _ls_ratio(ts_d, ts_s)
    print(f"[c/{form}/w{warmup}/k{checkpoint}] distributed/single-device record: least-squares ratio "
          f"{ls:.9f} ({_peak_ulp(ts_s, ts_d):.2f} ULP of the peak per sample); "
          f"d ln|E({F_BIN / 1e9:g} GHz)|^2/d ln eps_r {g_d:.7f} vs {g_s:.7f} "
          f"(rel {g_d / g_s - 1:+.2e}); a drive from the drawn slab predicts "
          f"ratio {EPS_HI / EPS_LO:.4f}")
    assert abs(ls - 1.0) <= RATIO_TOL, ls
    assert abs(g_d / g_s - 1.0) <= GRAD_RTOL, (g_d, g_s)


def test_the_traced_drive_coefficient_has_a_finite_gradient_under_a_large_cotangent():
    """Cb = dt/(eps + sigma*dt/2) written with eps = eps_r*eps0 ~ 1e-11 squares
    eps in the reverse pass; a float32 cotangent above ~1e15 then overflowed to a
    NaN gradient at the four cells a current source's drive reads -- on this lane
    and on the single-device lane since #1280 (#1279 review). In eps_r units the
    same Cb keeps the reverse pass finite far past any objective a board makes."""
    from rfx.nonuniform import current_source_cb
    dt = float(_board(EPS_LO)._build_nonuniform_grid().dt)
    for cotangent in (1e18, 1e22, 1e26, 1e30):
        grads = jax.grad(lambda e, s: cotangent * current_source_cb(e, s, dt, traced=True),
                         argnums=(0, 1))(jnp.float32(2.0), jnp.float32(0.5))
        assert all(np.isfinite(float(g)) for g in grads), (cotangent, grads)


def test_a_large_objective_gives_a_finite_gradient_on_both_lanes():
    """The raw sum of squared probe samples (~1e16 (V/m)^2: two 1 A current
    sources in a lossy slab, 40 steps, eps and sigma overridden) as the objective,
    the reviewer's board: both lanes give a finite eps and sigma gradient."""
    dz = np.array([0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0]) * 1e-3
    sim = Simulation(freq_max=15e9, domain=(12e-3, 7e-3, float(dz.sum())), dx=1e-3,
                     dz_profile=dz, boundary="cpml", cpml_layers=2)
    sim.add_material("slab", eps_r=4.4, sigma=0.02)
    sim.add(Box((3e-3, 1e-3, 0.8e-3), (11e-3, 6e-3, 5.6e-3)), material="slab")
    for position, component in (((7e-3, 3e-3, 1.6e-3), "ey"), ((10e-3, 3e-3, 1.6e-3), "ez")):
        sim.add_source(position, component, waveform=GaussianPulse(f0=8e9, bandwidth=0.8),
                       amplitude_kind="current")
    sim.add_probe((9e-3, 3e-3, 3.6e-3), "ey")
    sim.add_probe((7e-3, 3e-3, 1.6e-3), "ey")
    shape = sim._build_nonuniform_grid().shape
    flat = np.arange(np.prod(shape))
    eps = jnp.asarray((1.2 + (flat % 71) * .05).astype(np.float32).reshape(shape))
    sig = jnp.asarray(((flat % 71) * .02).astype(np.float32).reshape(shape))
    lanes = {"single": {}, "dist": dict(distributed=True, devices=jax.devices("cpu")[:N_DEVICES])}
    for name, kw in lanes.items():
        def objective(e, s):
            ts = sim.forward(eps_override=e, sigma_override=s, n_steps=40,
                             skip_preflight=True, **kw).time_series
            return jnp.sum(ts ** 2)
        value = float(objective(eps, sig))
        ge, gs = (np.asarray(g) for g in jax.grad(objective, argnums=(0, 1))(eps, sig))
        print(f"[large/{name}] objective {value:.3e}; finite eps/sigma gradient "
              f"{bool(np.isfinite(ge).all())}/{bool(np.isfinite(gs).all())}")
        assert value > 1e15, value   # the regime that overflowed; else this says nothing
        assert np.isfinite(ge).all() and np.isfinite(gs).all(), name


# --------------------------------------------------------------------------
# mutations: the defect restored with every builder call kept
# --------------------------------------------------------------------------

@contextmanager
def _drive_from_drawn(sim):
    """Mutation (b): the in-program four-cell mean and Cb are still what
    builds the drive, but the arrays handed to them are the materials as
    DRAWN, staged the way the runner stages a geometry array."""
    grid = sim._build_nonuniform_grid()
    drawn = _drawn(sim)
    real = nu.material_drive_scales

    def from_drawn(eps_r, sigma, mesh, drives, dt):
        sg = nu.build_sharded_nu_grid(grid, mesh.devices.size)
        return real(nu.stage_concrete_forward_array(drawn.eps_r, sg, mesh, 1.0),
                    nu.stage_concrete_forward_array(drawn.sigma, sg, mesh, 0.0),
                    mesh, drives, dt)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nu, "material_drive_scales", from_drawn)
        yield


@contextmanager
def _seam_reads_its_own_row():
    """The drive reads the override, at the wrong cells: every slab is read
    from its first real row, so an Ey/Ez edge on a rank's first real cell
    replicates its own row where its i-1 cells (the neighbour's) belong."""
    real = nu.material_drive_scales

    def own_row(eps_r, sigma, mesh, drives, dt):
        ghost = 1
        drives = tuple((dev, ghost, (cell[0] + row0 - ghost, cell[1], cell[2]), comp, dV)
                       for dev, row0, cell, comp, dV in drives)
        return real(eps_r, sigma, mesh, drives, dt)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nu, "material_drive_scales", own_row)
        yield


@pytest.mark.slow
def test_mutation_the_drive_from_the_drawn_arrays_sends_a_b_c_red():
    grid = _edge_board()._build_nonuniform_grid()
    with _drive_from_drawn(_edge_board()):
        rel = _drive_over_stepper("sharded")
    ls_b, f_b, _ = _raised_ratios("sharded", mutated=True)
    ts_s, g_s = _lane("single")
    ts_d, g_d = _lane("dist", "sharded", mutated=True)
    ls_c = _ls_ratio(ts_d, ts_s)
    print(f"[mutation drawn] a: worst |Cb_drive/Cb_step - 1| "
          f"{max(abs(v) for v in rel.values()):.3f}; b: ratio {ls_b:.6f} "
          f"(e^0.1 = {np.exp(LN_STEP):.6f}), {F_BIN / 1e9:g} GHz {abs(f_b):.6f}; "
          f"c: ratio {ls_c:.4f} ({EPS_HI / EPS_LO:.4f} predicted), gradient "
          f"{g_d:.4f} vs {g_s:.4f}; grid {grid.shape}")
    assert min(abs(v) for v in rel.values()) > 1e3 * CB_RTOL
    assert abs(ls_b / np.exp(LN_STEP) - 1.0) < 1e-5
    assert abs(ls_c / (EPS_HI / EPS_LO) - 1.0) < 1e-4
    assert abs(g_d / g_s - 1.0) > 1e3 * GRAD_RTOL


@pytest.mark.slow
def test_mutation_the_seam_edge_reading_its_own_row_sends_a_red():
    with _seam_reads_its_own_row():
        rel = _drive_over_stepper("sharded")
    print("[mutation own row] a: "
          + ", ".join(f"{w} {c} {v:+.2e}" for (w, c), v in rel.items()))
    for (where, comp), v in rel.items():
        if where == "seam" and comp in ("ey", "ez"):
            assert abs(v) > 1e3 * CB_RTOL, (where, comp, v)
        else:  # Ex never reads i-1; an inner edge's i-1 row is its own slab
            assert abs(v) <= CB_RTOL, (where, comp, v)


# --------------------------------------------------------------------------
# d. opt-in: a 'field' source, and no override, bitwise what main computes
# --------------------------------------------------------------------------

def _bit_cases():
    """Records and gradients the fix must not move, computed by whichever
    ``rfx`` the process imports."""
    out = {}
    seam_field = (("ey", SEAM, "field"),)
    both = (("ey", SEAM, "current"), ("ez", INNER, "field"))
    sim = _board(4.4, 0.02, sources=both)
    grid = sim._build_nonuniform_grid()
    eps, sigma = _random_design(grid.shape)
    out["no_override"] = np.asarray(_dist(sim, n_steps=60).time_series)
    occ = jnp.asarray(np.random.default_rng(3).uniform(0, .2, grid.shape).astype(np.float32))
    out["no_override_occupancy_grad"] = np.asarray(jax.grad(
        lambda o: jnp.sum(_dist(sim, pec_occupancy_override=o, n_steps=60).time_series ** 2))(occ))
    sim = _board(4.4, 0.02, sources=seam_field)
    for form in ("local", "sharded"):
        def loss(e, s):
            ts = _dist(sim, eps_override=e, sigma_override=s, n_steps=60).time_series
            return jnp.sum(ts ** 2), ts
        (_, ts), g = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(
            _place(sim, eps, form), _place(sim, sigma, form))
        out[f"field_{form}_record"] = np.asarray(ts)
        out[f"field_{form}_grad_eps"] = np.asarray(g[0])[:grid.shape[0]]
        out[f"field_{form}_grad_sigma"] = np.asarray(g[1])[:grid.shape[0]]
    return out


def _env(devices, pythonpath):
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1",
           "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
           "PYTHONPATH": str(pythonpath)}
    return {k: v for k, v in env.items() if not k.lower().endswith("_proxy")}


def _main_tree(tmp_path_factory):
    checkout = os.environ.get("RFX_MAIN_CHECKOUT")
    if checkout:
        return Path(checkout)
    ref = os.environ.get("RFX_MAIN_REF")
    if not ref:
        pytest.skip("opt-in: set RFX_MAIN_REF=<main commit> or "
                    "RFX_MAIN_CHECKOUT=<dir holding main's rfx/> to compare "
                    "with main bit for bit on this host")
    run = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", ref, "rfx"],
                         capture_output=True)
    if run.returncode:
        pytest.skip(f"git archive {ref}: {run.stderr.decode(errors='replace')[:300]}")
    tree = tmp_path_factory.mktemp("main-rfx")
    with tarfile.open(fileobj=io.BytesIO(run.stdout)) as archive:
        try:
            archive.extractall(tree, filter="data")
        except TypeError:  # Python without extraction filters
            archive.extractall(tree)
    return tree


def test_field_sources_and_runs_without_an_override_are_main_bit_for_bit(tmp_path_factory):
    main = _main_tree(tmp_path_factory)
    results = {}
    for name, tree in (("main", main), ("here", ROOT)):
        out = tmp_path_factory.mktemp(f"bits-{name}")
        run = subprocess.run([sys.executable, str(SCRIPT), "bits", str(out), str(tree)],
                             env=_env(N_DEVICES, tree), capture_output=True, text=True,
                             timeout=600)
        assert run.returncode == 0, run.stdout + run.stderr
        results[name] = dict(np.load(out / "bits.npz"))
    assert results["main"].keys() == results["here"].keys()
    report = {}
    for key, want in results["main"].items():
        got = results["here"][key]
        assert np.any(want), f"vacuous {key}"
        report[key] = {"bitwise": want.tobytes() == got.tobytes(),
                       "differing": int(np.count_nonzero(want != got))}
    print("[d] " + json.dumps(report))
    assert all(r["bitwise"] for r in report.values()), report


# --------------------------------------------------------------------------
# e. slow, Linux: the x-sharded override across two processes
# --------------------------------------------------------------------------

def _worker(address, rank, output):
    rank = int(rank)
    if address != "reference":
        assert jax.process_count() == 2 and jax.local_device_count() == 1
    sim = _board(4.4, 0.02, sources=(("ey", SEAM, "current"), ("ez", INNER, "current")),
                 probes=(("ey", PROBE), ("ey", SEAM)))
    shape, sharding = sim.distributed_override_layout()

    def design(offset, step):
        # built from global indices on each owning device; no process holds
        # the whole design
        def local(index):
            lo, hi, _ = index[0].indices(shape[0])
            flat = np.arange(lo * shape[1] * shape[2], hi * shape[1] * shape[2])
            return (offset + (flat % 71) * step).astype(np.float32).reshape(
                hi - lo, *shape[1:])
        return jax.make_array_from_callback(shape, sharding, local)

    eps, sigma = design(1.2, .05), design(0., .02)
    f = lambda e, s: sim.forward(eps_override=e, sigma_override=s, distributed=True,
                                 n_steps=40, skip_preflight=True).time_series
    trace = f(eps, sigma)
    grads = jax.grad(lambda e, s: _log_energy(f(e, s)), argnums=(0, 1))(eps, sigma)
    assert trace.is_fully_replicated
    np.save(Path(output) / f"trace-{rank}.npy", np.asarray(trace))
    for name, grad, d in zip(("eps", "sigma"), grads, (eps, sigma)):
        assert grad.sharding == d.sharding, name
        for shard in grad.addressable_shards:
            np.save(Path(output) / f"grad-{name}-{shard.device.id}.npy", np.asarray(shard.data))
    print(f"rank={rank} ok", flush=True)
    if address != "reference":
        jax.distributed.shutdown()


@pytest.mark.slow
def test_two_processes_drive_the_seam_source_as_one_process(tmp_path):
    if sys.platform != "linux":
        pytest.skip("requires Linux: jax.distributed gRPC bind fails on macOS")
    reference = tmp_path / "reference"
    reference.mkdir()
    run = subprocess.run([sys.executable, str(SCRIPT), "worker", "reference", "0",
                          str(reference)], cwd=ROOT, env=_env(2, ROOT),
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    with socket.socket() as listener:
        listener.bind(("localhost", 0))
        address = f"localhost:{listener.getsockname()[1]}"
    processes, handles = [], []
    deadline = time.monotonic() + 300
    try:
        for rank in range(2):
            handle = (tmp_path / f"worker-{rank}.log").open("w")
            handles.append(handle)
            processes.append(subprocess.Popen(
                [sys.executable, str(SCRIPT), "worker", address, str(rank), str(tmp_path)],
                cwd=ROOT, env=_env(1, ROOT), stdout=handle, stderr=subprocess.STDOUT))
        for process in processes:
            process.wait(timeout=max(.01, deadline - time.monotonic()))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        for handle in handles:
            handle.close()
    logs = "\n".join((tmp_path / f"worker-{r}.log").read_text() for r in range(2))
    assert all(p.returncode == 0 for p in processes), logs

    def gathered(directory, name):
        return np.concatenate([np.load(p) for p in sorted(
            directory.glob(f"grad-{name}-*.npy"), key=lambda p: int(p.stem.split("-")[2]))])

    want = np.load(reference / "trace-0.npy")
    assert np.any(want)
    report = {f"trace-{r}": _peak_ulp(want, np.load(tmp_path / f"trace-{r}.npy"))
              for r in range(2)}
    for name in ("eps", "sigma"):
        g = gathered(reference, name)
        assert np.any(g), name
        report[f"grad-{name}"] = _peak_ulp(g, gathered(tmp_path, name))
    print("[e] two processes vs one, ULP of the peak: " + json.dumps(report))
    assert all(v <= PEAK_ULP for v in report.values()), report


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "bits":
        out, tree = Path(args[0]), Path(args[1]).resolve()
        import rfx
        assert Path(rfx.__file__).resolve().is_relative_to(tree), rfx.__file__
        np.savez(out / "bits.npz", **_bit_cases())
        print(f"bits from {rfx.__file__} jax {jax.__version__}")
    elif mode == "worker":
        _worker(*args)
    else:
        raise AssertionError(mode)
