"""Progress reporting on long solves (issue #667) — and its identity gates.

Issue #667: a multi-hour solve printed nothing between the call and its
return, so a slow run was indistinguishable from a hang. The measured case:
a 42.15 M-cell / 225,000-step ``compute_msl_s_matrix`` whose last log line
was written at second 0 and was still the last line 4 h 10 min later.

``report_every=N`` fixes that. It is only worth having if it CANNOT move a
number, so the identity gates below are the point of this file, not a
footnote to it:

* ``report_every=None`` runs the unchanged single-scan code path, so it is
  byte-identical by construction (verified against a pinned ``git archive``
  of the base commit during development; the in-repo gate is the pair below).
* ``report_every=N`` turns that one scan into host-driven chunks of the SAME
  compiled scan with ``carry`` threaded through — a continuation, not a
  re-solve, because the carry already holds the DFT accumulators and every
  port / flux / monitor state. :func:`test_chunking_is_bit_exact_*` locks
  that at SHA-256 over raw bytes, not at a tolerance.

``test_hash_harness_can_move`` is the negative control. Without it, an
identity test that passes proves nothing: a harness that hashes the wrong
thing, or a constant, would pass too.

Chunk sizes here deliberately do NOT divide the step count, so the ragged
final chunk (a second XLA compile, different scan length) is on the gate.
"""

from __future__ import annotations

import hashlib
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation, simulation as _sim
from rfx.core.yee import MaterialArrays
from rfx.grid import Grid
from rfx.progress import ProgressReporter, validate_report_every

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _sha(*arrays) -> str:
    """SHA-256 over the raw bytes of *arrays*, dtype and shape included."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _digest(res) -> dict:
    st = res.state
    return {
        "fields": _sha(*(getattr(st, c) for c in
                         ("ex", "ey", "ez", "hx", "hy", "hz"))),
        "time_series": _sha(res.time_series),
    }


_N_STEPS = 220


def _dc_free_pulse(t, dt):
    """Differentiated Gaussian — zero net DC.

    A plain Gaussian was the first choice here and it made the gate weaker
    in a way only the intermediates showed: its DC content deposits a static
    soft-source charge (the issue #388 class), whose self-energy neither
    radiates nor is absorbed by CPML. The measured effect on this fixture
    was ``max|Ez| = 6.16662`` at step 64 and ``6.16661`` at step 220 — a
    frozen electrostatic remnant dominating both states, with ``|Hx|`` seven
    orders below it. Byte-identity across a chunk boundary is a much weaker
    statement when the field being compared is mostly static, so the source
    is DC-free and the field genuinely rings down.
    """
    x = (t - 30 * dt) / (10 * dt)
    return -2.0 * x * jnp.exp(-(x ** 2))


def _low_level_run(report_every=None, *, perturb=False, n_steps=_N_STEPS):
    """A CPML box with a dielectric inclusion and two probes."""
    grid = Grid(freq_max=20e9, domain=(0.006, 0.006, 0.006), dx=5e-4,
                cpml_layers=6)
    shape = (grid.nx, grid.ny, grid.nz)
    eps = jnp.ones(shape, dtype=jnp.float32).at[10:14, 10:14, 10:14].set(4.0)
    mats = MaterialArrays(eps_r=eps,
                          sigma=jnp.zeros(shape, dtype=jnp.float32),
                          mu_r=jnp.ones(shape, dtype=jnp.float32))
    t = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    wf = _dc_free_pulse(t, grid.dt)
    if perturb:
        wf = wf.at[0].add(np.float32(1e-7))
    cx, cy, cz = grid.nx // 2, grid.ny // 2, grid.nz // 2
    src = [_sim.SourceSpec(i=cx, j=cy, k=cz, component="ez", waveform=wf)]
    probes = [_sim.ProbeSpec(i=cx - 4, j=cy, k=cz, component="ez"),
              _sim.ProbeSpec(i=cx + 4, j=cy, k=cz, component="ez")]
    kw = dict(boundary="cpml", sources=src, probes=probes, return_state=True)
    if report_every is not None:
        kw["report_every"] = report_every
        kw["report_label"] = "gate"
    return _sim.run(grid, mats, n_steps, **kw)


def _msl_thru():
    """The committed thru-line fixture (see tests/test_msl_settling_witness.py)."""
    domain_y, y_c = 0.008, 0.004
    sim = Simulation(freq_max=20e9, domain=(0.012, domain_y, 0.0032),
                     dx=2e-4, boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.012, domain_y, 0.0008)), material="sub")
    sim.add(Box((0.0, y_c - 0.0006, 0.0008),
                (0.012, y_c + 0.0006, 0.0010)), material="pec")
    sim.add_msl_port(position=(0.002, y_c, 0.0), width=0.0012, height=0.0008,
                     direction="+x", impedance=50.0, eps_r_sub=2.2, name="p1")
    sim.add_msl_port(position=(0.010, y_c, 0.0), width=0.0012, height=0.0008,
                     direction="-x", impedance=50.0, eps_r_sub=2.2, name="p2")
    return sim


_MSL_FREQS = jnp.linspace(2e9, 18e9, 12)


def _msl_digest(res) -> dict:
    return {
        "S": _sha(np.asarray(res.S)),
        "Z0": _sha(np.asarray(res.Z0)),
        "beta": _sha(np.asarray(res.beta)),
    }


# --------------------------------------------------------------------------
# identity gates — the reason this feature is allowed to exist
# --------------------------------------------------------------------------


@pytest.mark.parametrize("every", [64, 97, _N_STEPS, _N_STEPS + 500])
def test_chunking_is_bit_exact_low_level(every):
    """Host chunking must not move one bit of field state or time series.

    ``64`` and ``97`` both leave a ragged final chunk (a second compile at a
    different scan length); ``_N_STEPS`` is exactly one chunk; ``_N_STEPS +
    500`` exceeds the run and must still be legal (one report at the end).
    """
    ref = _digest(_low_level_run(None))
    got = _digest(_low_level_run(every))
    assert got == ref, (
        f"report_every={every} changed the result — chunk re-entry is NOT a "
        f"continuation.\n  reference {ref}\n  chunked   {got}"
    )


def _loaded_run(report_every=None, n_steps=_N_STEPS):
    """Same box, but with every carry slot #667 could plausibly disturb.

    Snapshots exercise the second (list-valued) leaf of the scan ``outputs``
    tuple, which the chunk concatenation has to rebuild; the flux monitor
    and the DFT plane exercise complex accumulators that integrate ACROSS
    chunk boundaries, i.e. the state whose continuity the whole "this is a
    continuation, not a re-solve" claim rests on.
    """
    from rfx.probes.probes import init_flux_monitor, init_dft_plane_probe

    grid = Grid(freq_max=20e9, domain=(0.006, 0.006, 0.006), dx=5e-4,
                cpml_layers=6)
    shape = (grid.nx, grid.ny, grid.nz)
    eps = jnp.ones(shape, dtype=jnp.float32).at[10:14, 10:14, 10:14].set(4.0)
    mats = MaterialArrays(eps_r=eps,
                          sigma=jnp.zeros(shape, dtype=jnp.float32),
                          mu_r=jnp.ones(shape, dtype=jnp.float32))
    t = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    wf = _dc_free_pulse(t, grid.dt)
    cx, cy, cz = grid.nx // 2, grid.ny // 2, grid.nz // 2
    src = [_sim.SourceSpec(i=cx, j=cy, k=cz, component="ez", waveform=wf)]
    probes = [_sim.ProbeSpec(i=cx - 4, j=cy, k=cz, component="ez")]
    freqs = jnp.array([8e9, 12e9])
    flux = [init_flux_monitor(0, cx + 5, freqs, shape, grid.dx, grid.dx,
                              dft_total_steps=n_steps)]
    planes = [init_dft_plane_probe(0, cx - 5, "ez", freqs, shape)]
    kw = dict(boundary="cpml", sources=src, probes=probes,
              flux_monitors=flux, dft_planes=planes,
              snapshot=_sim.SnapshotSpec(interval=1, components=("ez", "hx"),
                                         slice_axis=2, slice_index=cz),
              return_state=True)
    if report_every is not None:
        kw["report_every"] = report_every
    res = _sim.run(grid, mats, n_steps, **kw)
    named = (
        [(f"snapshot[{k}]", np.asarray(v))
         for k, v in sorted(res.snapshots.items())]
        + [(f"flux[{i}].{nm}", np.asarray(x))
           for i, fm in enumerate(res.flux_monitors)
           for nm, x in (("e1", fm.e1_dft), ("e2", fm.e2_dft),
                         ("h1", fm.h1_dft), ("h2", fm.h2_dft))]
        + [(f"dft_plane[{i}]", np.asarray(p.accumulator))
           for i, p in enumerate(res.dft_planes)]
        + [("time_series", np.asarray(res.time_series))]
        + [(f"field.{c}", np.asarray(getattr(res.state, c)))
           for c in ("ez", "hx")]
    )
    d = _digest(res)
    d["snapshots"] = _sha(*[a for k, a in named if k.startswith("snapshot")])
    d["flux"] = _sha(*[a for k, a in named if k.startswith("flux")])
    d["dft_planes"] = _sha(*[a for k, a in named if k.startswith("dft_plane")])
    return d, dict(named)


def _liveness(arrays: dict) -> dict:
    """Per-array evidence that the thing being hashed carries real signal."""
    out = {}
    for name, a in arrays.items():
        mag = np.abs(a.astype(np.complex128) if np.iscomplexobj(a)
                     else a.astype(np.float64))
        out[name] = {
            "max_abs": float(mag.max()),
            "l2": float(np.sqrt((mag ** 2).sum())),
            "std": float(mag.std()),
            "frac_nonzero": float((mag > 0).mean()),
        }
    return out


@pytest.mark.parametrize("every", [64, 97])
def test_chunking_is_bit_exact_with_loaded_carry(every):
    """Snapshots, a flux monitor and a DFT plane all survive chunk re-entry.

    The liveness block below is not decoration. A byte-identity gate over
    state that happens to be identically zero — a feed that never reached
    the structure, a monitor plane outside the field, an accumulator whose
    window never opened — passes for the wrong reason and proves nothing.
    So every array this gate hashes must first show a nonzero norm AND a
    non-degenerate spread (a constant array would also be uninformative).
    """
    ref, ref_arrays = _loaded_run(None)
    got, _ = _loaded_run(every)

    live = _liveness(ref_arrays)
    for name, st in live.items():
        assert st["l2"] > 0.0, (
            f"{name} is identically zero — this gate would pass for the "
            f"wrong reason. stats={st}"
        )
        assert st["std"] > 0.0, (
            f"{name} is a constant array (no spread), so a digest match "
            f"carries no information. stats={st}"
        )
        assert st["frac_nonzero"] > 0.01, (
            f"{name} is >99% zeros — the observable is not being driven. "
            f"stats={st}"
        )

    assert set(ref) >= {"snapshots", "flux", "dft_planes"}
    assert got == ref, (
        f"report_every={every} moved a loaded-carry digest.\n"
        f"  reference {ref}\n  chunked   {got}\n  liveness  {live}"
    )


def test_chunk_boundary_carry_is_a_live_intermediate():
    """The state threaded across a chunk boundary is a genuine mid-solve state.

    Byte-identity across chunking is only meaningful if the chunk boundary
    actually cuts the solve somewhere interesting. If the boundary state
    were still the zero initial condition (or already the final state), the
    continuation would be trivial and the identity gate would be testing
    nothing. Chunk k's boundary carry is by construction the state of a run
    of length ``k * report_every``, so run those lengths directly.
    """
    every = 64
    _, at_boundary = _loaded_run(None, n_steps=every)
    _, at_end = _loaded_run(None, n_steps=_N_STEPS)

    ez_b, ez_e = at_boundary["field.ez"], at_end["field.ez"]
    hx_b = at_boundary["field.hx"]

    # (a) not the zero initial condition (measured max|Ez| = 2.21e-2)
    assert np.abs(ez_b).max() > 1e-4, np.abs(ez_b).max()

    # (b) genuinely different from the final state. Normalise by the LARGER
    # of the two norms so the ratio stays bounded as the field rings down
    # (measured 1.0 -- the two states barely overlap).
    assert not np.array_equal(ez_b, ez_e)
    rel = (np.linalg.norm(ez_b - ez_e)
           / max(np.linalg.norm(ez_b), np.linalg.norm(ez_e),
                 np.finfo(float).tiny))
    assert rel > 0.1, (
        f"the chunk-boundary state is nearly the final state (relative "
        f"difference {rel:.3e}); the chunking gate would be cutting the "
        f"solve nowhere."
    )

    # (c) the field is ringing down, not sitting on a static remnant. A DC
    # soft-source charge would freeze max|Ez| (issue #388 class): measured
    # 2.21e-2 -> 3.14e-4, a factor ~70.
    decay = np.abs(ez_b).max() / max(np.abs(ez_e).max(), np.finfo(float).tiny)
    assert decay > 5.0, (
        f"max|Ez| barely moved between step {every} and step {_N_STEPS} "
        f"(factor {decay:.3g}) -- the field is static, so byte-identity "
        f"across a chunk boundary would be a much weaker statement than it "
        f"looks."
    )

    # (d) independent witness that this is a real EM field rather than an
    # electrostatic artifact: max|Ez|/max|Hx| should sit near the free-space
    # wave impedance Z0 = 377. Measured 789 at the boundary and 275 at the
    # end -- both within about 2x of Z0. The DC-carrying source this fixture
    # started with read 1.8e7 at step 220 instead, which is what an
    # electrostatic remnant looks like.
    eh = np.abs(ez_b).max() / max(np.abs(hx_b).max(), np.finfo(float).tiny)
    assert 37.7 < eh < 3770.0, (
        f"max|Ez|/max|Hx| = {eh:.3g} is more than a decade from Z0=377; the "
        f"fixture is not driving a propagating field."
    )

    # (e) the accumulators are mid-integration, not finished (measured 39%
    # relative change between the boundary and the end)
    acc_b, acc_e = at_boundary["dft_plane[0]"], at_end["dft_plane[0]"]
    assert np.abs(acc_b).max() > 0.0, "DFT accumulator empty at the boundary"
    arel = np.linalg.norm(acc_b - acc_e) / np.linalg.norm(acc_e)
    assert arel > 0.01, (
        f"the DFT accumulator barely advanced between the chunk boundary "
        f"and the end (relative change {arel:.3e}), so chunk re-entry never "
        f"had to carry a meaningful partial integral"
    )


def test_hash_harness_can_move():
    """Negative control: a 1e-7 nudge to the source must move both digests.

    Without this, the identity gates above are unfalsifiable.
    """
    ref = _digest(_low_level_run(None))
    moved = _digest(_low_level_run(None, perturb=True))
    assert moved["fields"] != ref["fields"]
    assert moved["time_series"] != ref["time_series"]


def test_chunking_is_bit_exact_msl_s_matrix():
    """The entry point issue #667 was filed against: S / Z0 / beta unmoved.

    Two full FDTD solves per call (one per driven port), so this is the
    slowest gate here; it is the one that covers the real extraction chain
    (DFT-plane accumulators -> N-probe wave split -> S assembly) rather than
    just the raw carry.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = _msl_digest(
            _msl_thru().compute_msl_s_matrix(freqs=_MSL_FREQS, num_periods=6.0))
        got = _msl_digest(
            _msl_thru().compute_msl_s_matrix(freqs=_MSL_FREQS, num_periods=6.0,
                                             report_every=500))
    assert got == ref, (
        f"report_every moved the extracted S-matrix.\n"
        f"  reference {ref}\n  reported  {got}"
    )


# --------------------------------------------------------------------------
# default-off contract
# --------------------------------------------------------------------------


def test_default_is_off_and_silent(capsys):
    _low_level_run(None)
    assert "[PROGRESS]" not in capsys.readouterr().out


def test_reports_are_emitted_with_the_documented_fields(capsys):
    _low_level_run(64)
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if "[PROGRESS]" in ln]
    # 220 steps / 64 -> 3 full chunks + a ragged 28-step chunk.
    assert len(lines) == 4, lines
    for ln in lines:
        assert "steps" in ln and "elapsed" in ln
        assert "steps/s" in ln and "ETA" in ln
    assert lines[0].startswith("  [PROGRESS] gate: 64/220 steps")
    assert lines[-1].startswith("  [PROGRESS] gate: 220/220 steps (100.0%)")


# --------------------------------------------------------------------------
# fences — a reporting request that cannot be honoured must say so
# --------------------------------------------------------------------------


def test_rejected_under_jit():
    """Wall-clock reporting under a trace would print once, at trace time."""
    grid = Grid(freq_max=20e9, domain=(0.004, 0.004, 0.004), dx=5e-4,
                cpml_layers=4)
    shape = (grid.nx, grid.ny, grid.nz)
    n_steps = 20
    t = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    wf = jnp.exp(-((t - 5 * grid.dt) / (3 * grid.dt)) ** 2)
    c = (grid.nx // 2, grid.ny // 2, grid.nz // 2)
    src = [_sim.SourceSpec(i=c[0], j=c[1], k=c[2], component="ez", waveform=wf)]

    @jax.jit
    def solve(eps):
        mats = MaterialArrays(eps_r=eps,
                              sigma=jnp.zeros(shape, dtype=jnp.float32),
                              mu_r=jnp.ones(shape, dtype=jnp.float32))
        return _sim.run(grid, mats, n_steps, boundary="cpml", sources=src,
                        report_every=5).state.ez

    with pytest.raises(ValueError, match="cannot run under jax.jit"):
        solve(jnp.ones(shape, dtype=jnp.float32))


def _traced_solve(transform):
    """Drive run(report_every=...) under *transform* via the MATERIALS route.

    This is the route that matters: under a bare ``grad``/``vmap`` the
    carry and ``xs`` stay concrete and the tracer reaches the scan body
    only through the material arrays, captured in the body's closure. A
    guard that inspected carry/xs alone let this through and printed
    progress lines at trace time.
    """
    grid = Grid(freq_max=20e9, domain=(0.004, 0.004, 0.004), dx=5e-4,
                cpml_layers=4)
    shape = (grid.nx, grid.ny, grid.nz)
    n_steps = 20
    t = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    x = (t - 5 * grid.dt) / (3 * grid.dt)
    wf = -2.0 * x * jnp.exp(-(x ** 2))
    c = (grid.nx // 2, grid.ny // 2, grid.nz // 2)
    src = [_sim.SourceSpec(i=c[0], j=c[1], k=c[2], component="ez",
                           waveform=wf)]
    probes = [_sim.ProbeSpec(i=c[0] - 2, j=c[1], k=c[2], component="ez")]

    def obj(eps):
        mats = MaterialArrays(eps_r=eps,
                              sigma=jnp.zeros(shape, dtype=jnp.float32),
                              mu_r=jnp.ones(shape, dtype=jnp.float32))
        res = _sim.run(grid, mats, n_steps, boundary="cpml", sources=src,
                       probes=probes, report_every=5)
        return jnp.sum(res.time_series ** 2)

    eps0 = jnp.ones(shape, dtype=jnp.float32)
    if transform == "grad":
        return jax.grad(obj)(eps0)
    if transform == "vmap":
        return jax.vmap(obj)(jnp.stack([eps0, eps0 * 1.01]))
    return jax.jit(obj)(eps0)


@pytest.mark.parametrize("transform", ["jit", "grad", "vmap"])
def test_rejected_under_every_transform_via_materials(transform, capsys):
    """grad and vmap must be caught too, not only jit.

    The tracer arrives through the materials closure under bare grad/vmap,
    so a guard limited to carry_init/xs stays silent and prints trace-time
    lines with a fabricated elapsed and rate.
    """
    with pytest.raises(ValueError, match="cannot run under jax.jit"):
        _traced_solve(transform)
    assert "[PROGRESS]" not in capsys.readouterr().out, (
        "a progress line was printed at trace time before the guard fired"
    )


def test_grouped_concat_matches_flat_concat():
    """The bounded-arity join must be byte-identical to the flat one.

    ``concat_chunks`` groups the concatenate because a flat
    ``jnp.concatenate`` takes one argument per chunk and XLA's compile cost
    grows superlinearly in argument count (measured 232 s at 22,500 chunks
    against 0.99 s grouped). Concatenation is associative so the bytes must
    not move -- that is what makes the optimisation admissible at all.
    """
    from rfx.progress import _CONCAT_GROUP, concat_chunks

    rng = np.random.default_rng(0)
    # More chunks than one group, so the grouping actually recurses.
    n_chunks = _CONCAT_GROUP * 2 + 3
    parts = [(jnp.asarray(rng.standard_normal((3, 2)), dtype=jnp.float32),)
             for _ in range(n_chunks)]

    flat = jax.tree_util.tree_map(
        lambda *ps: jnp.concatenate(ps, axis=0), *parts)
    grouped = concat_chunks(parts)

    assert np.asarray(grouped[0]).shape == (3 * n_chunks, 2)
    assert (np.asarray(flat[0]).tobytes()
            == np.asarray(grouped[0]).tobytes()), (
        "grouped concatenation moved bytes; it must be exactly associative"
    )


def test_rejected_with_checkpoint_segments():
    with pytest.raises(NotImplementedError, match="checkpoint_segments"):
        _low_level_run_with_segments()


def _low_level_run_with_segments():
    grid = Grid(freq_max=20e9, domain=(0.004, 0.004, 0.004), dx=5e-4,
                cpml_layers=4)
    shape = (grid.nx, grid.ny, grid.nz)
    mats = MaterialArrays(eps_r=jnp.ones(shape, dtype=jnp.float32),
                          sigma=jnp.zeros(shape, dtype=jnp.float32),
                          mu_r=jnp.ones(shape, dtype=jnp.float32))
    return _sim.run(grid, mats, 20, boundary="cpml",
                    checkpoint_segments=4, report_every=5)


@pytest.mark.parametrize("bad", [
    0, -1, 0.5, 1000.5,
    float("inf"),   # int(inf) raises OverflowError, not ValueError
    float("nan"),
    True,           # Python bools are ints; True would mean "every step"
    False,
    "100",
    None.__class__,
])
def test_rejects_nonsense_report_every(bad):
    with pytest.raises(ValueError, match="report_every"):
        validate_report_every(bad, n_steps=100)


def test_validate_accepts_int_like():
    assert validate_report_every(7, n_steps=100) == 7
    assert validate_report_every(np.int64(7), n_steps=100) == 7


# --------------------------------------------------------------------------
# until_decay lane — already a host loop, so the tick is a pure addition
# --------------------------------------------------------------------------


def test_until_decay_reports_and_is_bit_exact(capsys):
    """Reporting on the decay lane must not move the result or the stop step.

    ``decay_by=0.0`` is the documented forced-N escape (``U < 0`` never
    fires), so both runs execute exactly ``max_steps`` steps and any
    difference would be the reporting itself.
    """
    def go(report_every):
        grid = Grid(freq_max=20e9, domain=(0.005, 0.005, 0.005), dx=5e-4,
                    cpml_layers=5)
        shape = (grid.nx, grid.ny, grid.nz)
        mats = MaterialArrays(eps_r=jnp.ones(shape, dtype=jnp.float32),
                              sigma=jnp.zeros(shape, dtype=jnp.float32),
                              mu_r=jnp.ones(shape, dtype=jnp.float32))
        n = 120
        t = jnp.arange(n, dtype=jnp.float32) * grid.dt
        wf = jnp.exp(-((t - 20 * grid.dt) / (7 * grid.dt)) ** 2)
        c = (grid.nx // 2, grid.ny // 2, grid.nz // 2)
        src = [_sim.SourceSpec(i=c[0], j=c[1], k=c[2], component="ez",
                               waveform=wf)]
        probes = [_sim.ProbeSpec(i=c[0] - 3, j=c[1], k=c[2], component="ez")]
        kw = dict(boundary="cpml", sources=src, probes=probes,
                  decay_by=0.0, check_interval=25, min_steps=20, max_steps=n,
                  return_state=True)
        if report_every is not None:
            kw["report_every"] = report_every
            kw["report_label"] = "decay"
        return _sim.run_until_decay(grid, mats, **kw)

    ref = _digest(go(None))
    capsys.readouterr()
    got = _digest(go(50))
    out = capsys.readouterr().out
    assert got == ref, "reporting perturbed the until_decay result"

    lines = [ln for ln in out.splitlines() if "[PROGRESS]" in ln]
    # ticks at 50 and 100, then the final line at the true stop step 120.
    assert len(lines) == 3, lines
    assert "(cap)" in lines[0], "max_steps is a cap, and the line must say so"
    assert lines[-1].startswith("  [PROGRESS] decay: 120/120 (cap) steps")


# --------------------------------------------------------------------------
# unsupported lanes must warn rather than silently produce nothing
# --------------------------------------------------------------------------


def test_nonuniform_lane_warns_instead_of_going_quiet():
    dz = np.concatenate([np.full(8, 5e-4), np.full(16, 2.5e-4)])
    sim = Simulation(freq_max=20e9, domain=(0.004, 0.004, float(dz.sum())),
                     dx=5e-4, dz_profile=dz, boundary="cpml", cpml_layers=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(n_steps=10, report_every=5)
    msgs = [str(w.message) for w in caught if "report_every" in str(w.message)]
    assert msgs, "the non-uniform lane must not drop report_every in silence"
    assert "uniform lane only" in msgs[0]


# --------------------------------------------------------------------------
# reporter formatting unit
# --------------------------------------------------------------------------


def test_reporter_line_shape(capsys):
    r = ProgressReporter(1000, label="x")
    line = r.report(250)
    assert line.startswith("  [PROGRESS] x: 250/1000 steps (25.0%)")
    assert "| elapsed 0:00:00 |" in line
    assert "steps/s | ETA " in line
    assert r.last_reported == 250
    assert capsys.readouterr().out.strip() == line.strip()
