"""The measurement DFT with a traced dt or traced bins (S2 M2 review P1).

A mesh that is a design variable makes dt a tracer; a jitted or vmapped bin
list makes the frequencies tracers. Both worked before the one-kernel DFT
and must keep working: with a concrete dt and bins the kernel reads host
tables, with a traced one it evaluates the phase in the trace. Nothing on
either path may convert a tracer on the host, and nothing may be cached per
(bins, dt).
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.measurement import dft
from rfx.measurement.dft import accumulate, phase, transform

MM = 1e-3
DZ = [0.5e-3] * 6 + [0.3e-3] * 5
BINS = np.array([5e9, 10e9])


def _graded(dz, kind):
    sim = Simulation(freq_max=20e9, domain=(6 * MM, 6 * MM, float(np.sum(DZ))),
                     dx=0.5 * MM, boundary="pec", dz_profile=dz)
    if kind == "wire":
        sim.add_port(position=(3 * MM, 3 * MM, 0.0), component="ez", impedance=50.0,
                     extent=1.0 * MM)
    else:
        sim.add_source((3 * MM, 3 * MM, 1.5 * MM), "ez")
    sim.add_probe((3 * MM, 3 * MM, 1.0 * MM), "ez")
    if kind in ("ez", "hx"):
        sim.add_dft_plane_probe(axis="z", coordinate=1.0 * MM, component=kind,
                                freqs=BINS, name="p")
    return sim


def test_gradient_through_a_traced_mesh_with_an_e_plane():
    """d/d(dz) of the plane spectrum at the probe's cell equals d/d(dz) of
    the same spectrum summed in the test from the probe's time record with
    the arithmetic rfx used before the tables, exp(-j 2 pi f (n+1) dt) dt
    with the traced dt. Bar: 1e-4 of the largest gradient entry (cross-trace
    bar for a quantity summed over the record). Until this fix the left side
    raised ConcretizationTypeError (float(dt) on a tracer)."""
    steps = 24

    i = j = 6  # the probe's cell (3 mm at 0.5 mm cells); the value check below confirms it

    def through_kernel(dz):
        r = _graded(dz, "ez").forward(n_steps=steps, skip_preflight=True)
        return jnp.sum(jnp.abs(r.dft_planes["p"].accumulator[:, i, j]) ** 2)

    def by_hand(dz):
        r = _graded(dz, "ez").forward(n_steps=steps, skip_preflight=True)
        n = jnp.arange(steps, dtype=jnp.float32)
        weight = jnp.exp(-2j * jnp.pi * jnp.asarray(BINS, jnp.float32)[:, None]
                         * ((n + 1.0) * r.dt)[None, :]) * r.dt
        return jnp.sum(jnp.abs(weight @ r.time_series[:, 0].astype(jnp.complex64)) ** 2)

    dz0 = jnp.asarray(DZ)
    value, g = jax.value_and_grad(through_kernel)(dz0)
    reference_value, reference = jax.value_and_grad(by_hand)(dz0)
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.max(jnp.abs(g))) > 0
    np.testing.assert_allclose(float(value), float(reference_value), rtol=1e-4)
    scale = float(jnp.max(jnp.abs(reference)))
    worst = float(jnp.max(jnp.abs(g - reference))) / scale
    print(f"\n[traced dz, ez plane] |g - g_hand| / max|g| = {worst:.3g}")
    assert worst <= 1e-4, worst


@pytest.mark.parametrize("kind", ["hx", "wire"])
def test_an_h_plane_and_a_wire_port_differentiate_through_a_traced_mesh(kind):
    def loss(dz):
        sim = _graded(dz, kind)
        if kind == "wire":
            r = sim.forward(n_steps=20, skip_preflight=True, port_s11_freqs=BINS)
            return jnp.sum(jnp.abs(r.s_params) ** 2)
        r = sim.forward(n_steps=20, skip_preflight=True)
        return jnp.sum(jnp.abs(r.dft_planes["p"].accumulator) ** 2)

    g = np.asarray(jax.grad(loss)(jnp.asarray(DZ)))
    assert np.all(np.isfinite(g)) and np.max(np.abs(g)) > 0


def test_jit_and_vmap_over_port_bins_equal_the_concrete_bins():
    """Uniform wire port: S with the bins traced by jit, and one bin at a
    time under vmap, against the concrete-bin S (host tables). Bar 1e-4 of
    peak at bins whose incident wave is within 40 dB of the band peak."""
    from tests.unit.sparams.test_ringdown_run import FREQS, _box
    bins = np.asarray(FREQS)[::10]
    kw = dict(n_steps=200, skip_preflight=True)
    concrete = _box("uniform").forward(port_s11_freqs=bins, **kw)
    vp, ii = (np.asarray(concrete.wire_port_sparams[0][1][k]) for k in (3, 1))
    strong = np.abs(vp + 50.0 * ii) >= 1e-2 * np.max(np.abs(vp + 50.0 * ii))
    assert strong.sum() >= 5
    reference = np.asarray(concrete.s_params).ravel()
    peak = np.max(np.abs(reference))
    sim = _box("uniform")
    jitted = np.asarray(jax.jit(lambda f: sim.forward(port_s11_freqs=f, **kw).s_params)(
        jnp.asarray(bins))).ravel()
    mapped = np.asarray(jax.vmap(lambda f: sim.forward(port_s11_freqs=f[None], **kw).s_params)(
        jnp.asarray(bins))).ravel()
    for name, got in (("jit", jitted), ("vmap", mapped)):
        worst = float(np.max(np.abs(got - reference)[strong]) / peak)
        print(f"\n[{name} over bins] |S - S_concrete| / peak = {worst:.3g}")
        assert worst <= 1e-4, (name, worst)


@pytest.mark.parametrize("kind", ["E", "H"])
def test_every_kernel_entry_takes_a_traced_dt_and_traced_bins(kind):
    """phase, accumulate, transform, the port wrapper, the flux adapter and
    the NTFF call (phase with the accumulator dtype) under grad in dt and
    under jit and vmap in the bins; values equal the concrete tables'."""
    from rfx.core.dft_utils import half_step_current_phase, port_dft_phase
    from rfx.measurement.accumulators import flux
    from rfx.measurement.plan import field_channel
    dt, bins = 1.9e-12, np.array([4e9, 9e9, 17e9])
    record = np.random.default_rng(3).normal(size=37).astype(np.float32)
    table = np.asarray(transform(record, bins, dt, kind))
    for got in (jax.jit(lambda f, d: transform(record, f, d, kind))(jnp.asarray(bins), dt),
                jax.vmap(lambda f: transform(record, f[None], dt, kind)[0])(jnp.asarray(bins))):
        np.testing.assert_allclose(np.asarray(got), table, rtol=2e-5, atol=2e-5 * np.abs(table).max())

    def objective(d):
        acc = accumulate(jnp.zeros(3, jnp.complex64), 0.7, 5, bins, d, kind)
        ntff = phase(5, bins, d, kind, dtype=jnp.complex64)
        port = port_dft_phase(jnp.int32(5), jnp.asarray(bins), d, kind)
        state = SimpleNamespace(**{c: jnp.ones((3, 3, 3), jnp.float32)
                                   for c in ("ex", "ey", "ez", "hx", "hy", "hz")})
        meta = ((0, 1, bins, tuple(field_channel(c) for c in ("ey", "ez", "hy", "hz")), 0, 3, 0, 3),)
        accs = (tuple(jnp.zeros((3, 3, 3), jnp.complex64) for _ in range(4)),)
        flux_sum = sum(jnp.sum(a) for a in flux(state, accs, meta, d, 5)[0])
        tail = jnp.sum(transform(record, bins, d, kind)) + jnp.sum(half_step_current_phase(bins, d))
        return jnp.real(jnp.sum(acc) + jnp.sum(ntff) + jnp.sum(port) + flux_sum + tail)

    slope = float(jax.grad(objective)(jnp.float32(dt)))
    step = dt * 1e-3
    finite = (float(objective(dt + step)) - float(objective(dt - step))) / (2 * step)
    assert np.isfinite(slope) and slope != 0.0
    np.testing.assert_allclose(slope, finite, rtol=2e-2)


def test_a_sweep_over_dt_keeps_no_table_and_one_compiled_replay():
    """Nothing per (bins, dt): the kernel holds no table and no cache, its
    replay compiles once for a record shape, and what a compiled scan holds
    of the bins is four uint32 words each. (First M2 form: an lru_cache of
    32 table sets of 8320 rows per bin and one compiled replay per dt,
    +15.9 GB over 60 dt at 1001 bins.)"""
    import inspect
    bins, record = np.linspace(1e9, 10e9, 11), jnp.ones(100)
    transform(record, bins, 1.86e-12).block_until_ready()
    compiled = dft._replay._cache_size()
    for k in range(1, 21):
        transform(record, bins, 1.86e-12 * (1 + 1e-3 * k)).block_until_ready()
    assert dft._replay._cache_size() == compiled
    static = {n for n, p in inspect.signature(dft._replay.__wrapped__).parameters.items()
              if p.kind is p.KEYWORD_ONLY}
    assert static == {"traced", "offset", "window", "alpha"}
    assert not [n for n, v in vars(dft).items()
                if not n.startswith("__")
                and (isinstance(v, (dict, list)) or hasattr(v, "cache_info"))]
    words = dft._words(bins, 1.86e-12, 0.5)
    assert words.shape == (4, 11) and words.dtype == np.uint32
    text = jax.jit(lambda n: phase(n, bins, 1.86e-12, "H")).lower(jnp.int32(3)).as_text()
    import re
    sizes = [int(np.prod([int(d) for d in m.split("x")]))
             for m in re.findall(r"dense<[^>]*> : tensor<((?:\d+x)*\d+)x[a-z]", text)]
    assert sizes and max(sizes) <= 4 * 11, sizes
