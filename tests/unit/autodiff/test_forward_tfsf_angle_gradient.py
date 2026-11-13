"""Predeclared P/P-ey angle witnesses (DESIGN Addenda 1–3, F1–F7)."""
import json
import sys
import time
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation

F0 = 5e9
N1, N2 = 1798, 3148
T_PROBE = 12


def emit(bar, **values):
    print("PWANGLE_IMPL " + json.dumps(dict(bar=bar, **values)), file=sys.stderr, flush=True)


def scene(pol="ez", theta=30., *, slab=True, h=.002, thickness=.025,
          boundary=None, layers=16, graded=None, method="bloch"):
    domain = (.30, .016, 4*h) if pol == "ez" else (.30, 4*h, .016)
    options = {}
    if graded == "x":
        options["dx_profile"] = np.r_[np.full(round(.1/h), h), np.full(round(.2/h), h/2), np.full(round(.1/h), h)]
    elif graded == "out":
        options["dz_profile" if pol == "ez" else "dy_profile"] = np.array([h, h/2, h/2, h, h])
    sim = Simulation(freq_max=10e9, domain=domain, dx=h, mode="3d", cpml_layers=layers,
                     boundary=boundary or {"x": "cpml", "y": "periodic", "z": "periodic"}, **options)
    sim.add_tfsf_source(f0=F0, bandwidth=.15, direction="+x", angle_deg=theta,
                        waveform="modulated_gaussian", polarization=pol, method=method)
    grid = sim.freeze_mesh()
    x_lo = grid.cpml_layers + sim._tfsf.margin
    x_hi = grid.shape[0] - x_lo - 1
    front = float(grid.node_of(0, x_lo)) + .040
    facts = dict(pol=pol, h=h, shape=list(grid.shape), dt=float(grid.dt),
                 tfsf_x=[int(x_lo), int(x_hi)], front=front, probes=[])
    if slab:
        sim.add_material("slab", eps_r=4., sigma=0.)
        sim.add(Box((front, 0., 0.), (front+thickness, domain[1], domain[2])), material="slab")
        materials, _, _ = sim._build_materials(grid)
        cells = np.flatnonzero(np.any(np.asarray(materials.eps_r) == 4., axis=(1, 2)))
        first = float(grid.node_of(0, int(cells[0])))
        back = float(grid.node_of(0, int(cells[-1]+1)))
        facts.update(slab_cells=[int(cells[0]), int(cells[-1]+1)],
                     realized_front=first, realized_back=back, thickness=back-first)
        if thickness == .024 and graded is None:
            assert abs(first-front) < 1e-12 and abs(back-front-thickness) < 1e-12
    for x in (front-.010, front+.012, front+.045):
        for fraction in (.3, .7):
            position = (x, fraction*domain[1], .5*domain[2]) if pol == "ez" else (
                x, .5*domain[1], fraction*domain[2])
            idx = [int(grid.index_of(a, v)) for a, v in enumerate(position)]
            assert x_lo < idx[0] < x_hi and idx[1 if pol == "ez" else 2] != 0
            for component in (("ez", "hx", "hy") if pol == "ez" else ("ey", "hx", "hz")):
                sim.add_probe(position, component=component)
                facts["probes"].append(dict(component=component, index=idx))
    return sim, facts


def run(sim, n, override=None):
    return sim.forward(n_steps=n, skip_preflight=False, tfsf_angle_override=override)


def dft(series, dt, n=None):
    s = series if n is None else series[:n]
    return jnp.sum(s[:, T_PROBE] * jnp.exp(-1j*2*jnp.pi*F0*jnp.arange(s.shape[0])*dt))


def transmission(pair, dt, n=None):
    return jnp.abs(dft(pair[0], dt, n) / dft(pair[1], dt, n))


@lru_cache(None)
def declared(pol, theta, n=N2):
    arrays = []
    for slab in (True, False):
        sim, facts = scene(pol, theta, slab=slab)
        start = time.perf_counter()
        arrays.append(np.asarray(run(sim, n).time_series))
        emit("declared", pol=pol, theta=theta, slab=slab, n=n,
             seconds=time.perf_counter()-start, facts=facts)
    return np.stack(arrays)


@lru_cache(None)
def functions(pol="ez", n=N2, h=.002, thickness=.025, layers=16):
    slab, facts = scene(pol, h=h, thickness=thickness, layers=layers)
    empty, _ = scene(pol, slab=False, h=h, layers=layers)
    def fields(theta):
        return jnp.stack((run(slab, n, theta).time_series, run(empty, n, theta).time_series))
    def observable(theta):
        return transmission(fields(theta), facts["dt"])
    return fields, observable, jax.jit(fields), jax.jit(observable), facts


@lru_cache(None)
def traced_fd(pol="ez", n=N2):
    _, _, _, fn, _ = functions(pol, n)
    return {s: float((fn(jnp.float32(30+s))-fn(jnp.float32(30-s)))/(2*s)) for s in (.5, .25)}


@lru_cache(None)
def gradient(pol="ez", n=N2, compiled=False):
    _, fn, _, _, _ = functions(pol, n)
    return float((jax.jit(jax.grad(fn)) if compiled else jax.grad(fn))(jnp.float32(30)))


@pytest.mark.slow
def test_00_first_job_stop_quantities():
    assert not jax.config.jax_enable_x64, "F5 requires the default complex64 DFT"
    facts = functions()[4]
    value = float(transmission(jnp.asarray(declared("ez", 30.)), facts["dt"]))
    emit("F7", shape=facts["shape"], n=N2, value=value, reference=.759503228792,
         difference=value-.759503228792)
    assert abs(value-.759503228792) <= 1e-5, "F7 STOP: declared P differs from reference"
    fd = traced_fd()
    scatter = abs(fd[.25]-fd[.5])/abs(fd[.25])
    emit("F5", fd025=fd[.25], fd05=fd[.5], relative_scatter=scatter)
    assert scatter <= 4e-4, "F5 STOP: traced FD scatter exceeds 4e-4"


@pytest.mark.parametrize("pol", ["ez", "ey"])
@pytest.mark.parametrize("theta", [30., 30.25], ids=["B1a", "B1b"])
def test_concrete_equal_requests(pol, theta):
    target = declared(pol, theta)
    for i, slab in enumerate((True, False)):
        sim, _ = scene(pol, slab=slab)
        actual = np.asarray(run(sim, N2, theta).time_series)
        emit("B1a" if theta == 30 else "B1b", pol=pol, slab=slab,
             max_difference=float(np.max(np.abs(actual-target[i]))))
        assert np.array_equal(actual, target[i]), f"B1 unequal probes: {pol}, {theta}, slab={slab}"


@pytest.mark.parametrize("pol", ["ez", "ey"])
@pytest.mark.parametrize("theta", [30., 30.25], ids=["B2a", "B2b"])
def test_traced_primal_growth(pol, theta):
    fields, _, _, _, facts = functions(pol)
    actual, _ = jax.jvp(fields, (jnp.float32(theta),), (jnp.float32(1),))
    actual, target = np.asarray(actual), declared(pol, theta)
    for n in (N1//2, N1, N2):
        for i, slab in enumerate((True, False)):
            delta = np.max(np.abs(actual[i, :n]-target[i, :n]), axis=0)
            scale = float(np.max(np.abs(target[i])))
            accumulated = float(np.sum(np.max(np.abs(target[i, :n]), axis=1), dtype=np.float64))
            emit("B2", pol=pol, theta=theta, slab=slab, n=n, per_probe=delta.tolist(),
                 peak=scale, accumulated=accumulated, law_bound=1.2e-7*accumulated)
            assert np.all(delta <= 1e-4*scale), "B2 fixed-peak bar"
            assert np.max(delta) <= 1.2e-7*accumulated, "B2 accumulated law"
        value = float(transmission(jnp.asarray(actual), facts["dt"], n))
        ref = float(transmission(jnp.asarray(target), facts["dt"], n))
        emit("B2_T", pol=pol, theta=theta, n=n, value=value, reference=ref)
        assert abs(value-ref) <= 1e-4*abs(ref), "B2 transmission bar"


@pytest.mark.slow
def test_cross_trace_grad_and_jit():
    eager, compiled = gradient(), gradient(compiled=True)
    emit("B3a", grad=eager, jit_grad=compiled, relative=abs(eager-compiled)/abs(eager))
    assert abs(eager-compiled) <= 1e-4*abs(eager), "B3 grad versus jit grad"


def test_cross_trace_vmap():
    _, _, fields, _, facts = functions()
    angles = jnp.array([29.75, 30.25], dtype=jnp.float32)
    batch = np.asarray(jax.vmap(fields)(angles))
    for i, theta in enumerate(angles):
        ref = np.asarray(fields(theta))
        per_step = np.max(np.abs(batch[i]-ref), axis=(0, 2))
        peak = np.max(np.abs(ref), axis=(0, 2))
        bound = 9*2**-23*peak
        value, target = [float(transmission(jnp.asarray(a), facts["dt"])) for a in (batch[i], ref)]
        emit("B3b", theta=float(theta), max_difference=float(np.max(per_step)), bound=bound.tolist(),
             value=value, reference=target)
        assert np.all(per_step <= bound), "B3 vmap probe series"
        assert abs(value-target) <= 1e-4*abs(target), "B3 vmap transmission"


@pytest.mark.parametrize("pol", ["ez", pytest.param("ey", marks=pytest.mark.slow)])
def test_angle_gradient_finite_difference(pol):
    ad = gradient(pol)
    for step, fd in traced_fd(pol).items():
        emit("B4", pol=pol, step=step, ad=ad, fd=fd, relative=abs(ad-fd)/abs(fd))
        assert abs(ad-fd) <= 1e-3*abs(fd), f"B4 derivative versus FD({step})"


@pytest.mark.parametrize("pol", ["ez", pytest.param("ey", marks=pytest.mark.slow)])
def test_complex_slab_dft_gradient(pol):
    fields, _, _, _, facts = functions(pol)
    fn = lambda theta: dft(fields(theta)[0], facts["dt"])
    compiled = jax.jit(fn)
    _, ad = jax.jvp(fn, (jnp.float32(30),), (jnp.float32(1),))
    for step in (.5, .25):
        fd = (compiled(jnp.float32(30+step))-compiled(jnp.float32(30-step)))/(2*step)
        ad_c, fd_c = complex(ad), complex(fd)
        emit("B4phi", pol=pol, step=step, ad=[ad_c.real, ad_c.imag],
             fd=[fd_c.real, fd_c.imag], relative=abs(ad_c-fd_c)/abs(fd_c))
        assert abs(ad_c-fd_c) <= 1e-3*abs(fd_c), f"B4phi complex derivative FD({step})"


@pytest.mark.slow
def test_gradient_record_length():
    short, long = gradient(n=N1), gradient(n=N2)
    emit("B5", N1=N1, N2=N2, short=short, long=long, relative=abs(short-long)/abs(long))
    assert abs(short-long) <= 1e-3*abs(long), "B5 gradient record length"


def test_source_end_concrete_override():
    sim, _ = scene()
    ref, _ = scene(theta=36.)
    actual, expected = run(sim, N2, 36.), run(ref, N2)
    emit("B8a", angle=actual.tfsf_angle_deg, actual=actual.settling_source_end_index,
         reference=expected.settling_source_end_index,
         actual_probe=actual.settling_probe_source_end_index,
         reference_probe=expected.settling_probe_source_end_index)
    assert actual.tfsf_angle_deg == 36.
    assert actual.settling_source_end_index == expected.settling_source_end_index, "B8 source end"
    assert actual.settling_probe_source_end_index == expected.settling_probe_source_end_index, "B8 probe source end"


def test_source_end_traced_override():
    sim, _ = scene()
    captured = []
    def fn(theta):
        result = run(sim, N1, theta)
        captured.append(result.settling_source_end_index)
        return result.time_series
    jax.jvp(fn, (jnp.float32(30),), (jnp.float32(1),))
    emit("B8b", captured=captured)
    assert captured and all(value is None for value in captured)
