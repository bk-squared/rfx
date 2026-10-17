"""One until_decay energy stop on both meshes (design v4 and its float32 floor).

The stop reads the leapfrog-conserved interior energy
W_n = sum((eps E^n.E^n + (mu0/eps0) mu H^(n-1/2).H^(n+1/2)) dV) at chunk
boundaries k*I, against the running maximum of W over the boundaries from
the drive end on, and stops after two consecutive boundaries below
max(thr * reference, 1e-12 * largest W at any boundary so far). Before v4
the reference was taken during the drive: on a point current source it was
the source cell's quasi-static near field, and the run could end before the
radiated pulse reached a probe 4 cm away.

(a) the two large loop files hold no energy function or energy decision of
    their own; both import the rule from rfx.measurement.decay;
(b) a far probe in an open box sees its whole pulse before the stop;
(c) the loop's stop equals the declared rule applied in this file to W
    reduced from captured fields, including a graded periodic seam, with no rfx.measurement import;
(d) a stopped record is the forced-length record's prefix;
(e) fewer than two consecutive boundaries is refused;
(f) a driven waveguide port's drive end is read from the table it applies,
    and the run stops on both meshes;
(g) a drive longer than the run's n_steps is not cut at n_steps on the decay
    path: the record does not depend on n_steps;
(h) a cap with a quasi-static remnant and no DC drive names the lossy medium
    and its charge relaxation time;
(i) a flux monitor with a non-rectangular DFT window is refused with
    until_decay on both meshes;
(j) a 2-D TEz run (Ex, Ey, Hz) stops on the threshold: W reads the mode's own
    components;
(k) precision="mixed" (float16 fields): W is formed in float32, finite, and the
    run stops on the threshold after the probe's pulse;
(l) on a floor or cap end the record-length verdict (Result.settling_witness)
    leads with the stop line;
(m) the fused side-step + W program takes the per-cell arrays as arguments,
    not as compiled constants.
"""

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.sources.sources import GaussianPulse

ROOT = Path(__file__).resolve().parents[2]
PULSE = dict(f0=10e9, bandwidth=1.0, cutoff=5)


# --- (a) ---------------------------------------------------------------------

def test_large_loop_files_hold_no_energy_rule():
    for module in ("simulation", "nonuniform"):
        tree = ast.parse((ROOT / "rfx" / f"{module}.py").read_text())
        imported = {a.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                    and node.module in ("rfx.measurement", "rfx.measurement.decay")
                    for a in node.names}
        assert "make_decay_stop" in imported, module
        functions = [n.name for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        assert not [name for name in functions if "energy" in name.lower()], module
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        # The energy reference, its counter and the graded volume weights.
        assert not names & {"peak_U", "energy_below", "_dV", "decay_checks"}, module


# --- scenes -----------------------------------------------------------------

def _cube(mesh):
    """The design's dielectric cube (eps 6, 6 mm) in a CPML box, 1 mm cells
    ("uniform-upml": the same cube in a UPML box)."""
    graded = {} if mesh.startswith("uniform") else {
        "dz_profile": np.array([1.5e-3] * 8 + [1e-3] * 10 + [1.5e-3] * 8)}
    boundary = ({"x": "periodic", "y": "cpml", "z": "cpml"} if mesh == "graded-periodic"
                else "upml" if mesh == "uniform-upml" else "cpml")
    sim = Simulation(freq_max=14e9, domain=(24e-3, 22e-3, 34e-3),
                     boundary=boundary,
                     dx=1e-3, **graded)
    sigma = 0.0
    if mesh == "graded-periodic":
        from rfx.core.yee import EPS_0
        sigma = EPS_0 * 4e10
        sim.add_material("lossy_background", eps_r=1, sigma=sigma)
        sim.add(Box((0, 0, 0), (24e-3, 22e-3, 34e-3)), material="lossy_background")
    sim.add_material("cube_eps6", eps_r=6, sigma=6 * sigma)
    xlo, xhi = (18e-3, 24e-3) if mesh == "graded-periodic" else (8e-3, 14e-3)
    sim.add(Box((xlo, 7e-3, 14e-3), (xhi, 13e-3, 20e-3)), material="cube_eps6")
    sim.add_source((6.5e-3, 10.5e-3, 17.5e-3), "ez", waveform=GaussianPulse(**PULSE),
                   amplitude_kind="current")
    sim.add_probe((17.5e-3, 15.5e-3, 18.5e-3), "ez")
    return sim


def _open_box(mesh):
    """60 mm vacuum CPML box, 1 mm cells; source and probe 42 mm apart."""
    dz = np.array([1.25e-3] * 4 + [1e-3] * 50 + [1.25e-3] * 4)
    graded = {} if mesh == "uniform" else {"dz_profile": dz}
    z = .06 if mesh == "uniform" else float(dz.sum())
    sim = Simulation(freq_max=14e9, domain=(.06, .06, z), dx=1e-3, boundary="cpml",
                     **graded)
    sim.add_source((.009, .030, z / 2), "ez", waveform=GaussianPulse(**PULSE),
                   amplitude_kind="current")
    sim.add_probe((.051, .031, z / 2 + 1e-3), "ez")
    return sim


# --- (b) ---------------------------------------------------------------------

@pytest.mark.parametrize("mesh", ["uniform", "graded"])
def test_stop_waits_for_the_far_probe(mesh, record_property):
    interval = 10
    result = _open_box(mesh).run(until_decay=1e-3, decay_check_interval=interval,
                                 decay_min_steps=interval, decay_max_steps=3000,
                                 skip_preflight=True)
    probe = np.abs(np.asarray(result.time_series)[:, 0])
    peak_at = int(np.argmax(probe))
    tail = float(probe[-interval:].max() / probe.max())
    record_property("stop_peak_tail", (len(probe), peak_at, tail))
    assert len(probe) < 3000
    assert peak_at < len(probe) - interval, (len(probe), peak_at)
    assert tail <= 0.1, (len(probe), peak_at, tail)


# --- (c) ---------------------------------------------------------------------

def _inline_history(mesh, cap, monkeypatch):
    """W_n for n < cap from a forced-length run, and the drive end n_end.

    Fields are captured per step by wrapping jax.lax.scan: slot n holds
    E^n, H^(n-1/2) (before step n+1) and H^(n+1/2) (after it). eps and mu are
    read from the materials handed to this run's setup, n_end from the
    source tables it builds (last sample above 1e-6 of the peak, plus one).
    """
    from rfx import nonuniform, simulation
    from rfx.core.yee import EPS_0, MU_0, component_e_materials, component_h_materials

    uniform = mesh.startswith("uniform")
    owner, name = ((simulation, "_build_step_setup") if uniform
                   else (nonuniform, "_build_nu_scan"))
    original_setup, original_scan = getattr(owner, name), jax.lax.scan
    built, history = {}, []

    def setup(*args, **kwargs):
        grid = kwargs["grid"] if "grid" in kwargs else args[0]
        materials = kwargs["materials"] if "materials" in kwargs else args[1]
        widths = None if uniform else (grid.dx_arr, grid.dy_arr, grid.dz)
        kw = {} if widths is None else dict(cell_sizes=widths)
        flags = tuple(a in grid.periodic_axes for a in "xyz")
        built.update(grid=grid, widths=widths,
                     eps=component_e_materials(materials, flags, **kw)[0],
                     mu=component_h_materials(materials, flags, **kw))
        ends = []
        for src in kwargs["sources"]:
            table = np.abs(np.asarray(src.waveform if hasattr(src, "waveform") else src[4],
                                      dtype=np.float64))
            on = np.flatnonzero(table > 1e-6 * table.max())
            ends.append(int(on[-1]) + 1 if on.size else 0)
        built["n_end"] = max(ends)
        return original_setup(*args, **kwargs)

    def scan(body, init, xs=None, *args, **kwargs):
        if not (isinstance(init, dict) and "fdtd" in init):
            return original_scan(body, init, xs, *args, **kwargs)
        grid, eps, mu, widths = built["grid"], built["eps"], built["mu"], built["widths"]
        r = tuple(slice(getattr(grid, f"pad_{a}_lo"), getattr(grid, "n" + a)
                        - getattr(grid, f"pad_{a}_hi")) for a in "xyz")
        if widths is None:
            ve = vh = (grid.dx ** 3,) * 3
        else:
            # Each component's Yee control volume: E_a primal along a, dual (the
            # mean of the two adjacent cells; the first equals the first cell)
            # across; H_a the other way round.
            primal = [np.asarray(grid.cells(a), dtype=np.float64)[r[a]] for a in range(3)]
            dual = []
            for a in range(3):
                d = np.asarray(grid.cells(a), dtype=np.float64)
                dual.append(np.concatenate([d[:1], 0.5 * (d[:-1] + d[1:])])[r[a]])
            box = lambda w: jnp.asarray(w[0][:, None, None] * w[1][None, :, None]
                                        * w[2][None, None, :], jnp.float32)
            ve = [box([primal[b] if b == a else dual[b] for b in range(3)]) for a in range(3)]
            vh = [box([dual[b] if b == a else primal[b] for b in range(3)]) for a in range(3)]

        def observe(carry, row):
            new, out = body(carry, row)
            old, after = carry["fdtd"], new["fdtd"]
            e = sum(jnp.sum(eps[a][r] * getattr(old, "e" + c)[r] ** 2 * ve[a])
                    for a, c in enumerate("xyz"))
            h = sum(jnp.sum(mu[a][r] * getattr(old, "h" + c)[r] * getattr(after, "h" + c)[r] * vh[a])
                    for a, c in enumerate("xyz"))
            return new, (out, e + (MU_0 / EPS_0) * h)

        final, (out, w) = original_scan(observe, init, xs, *args, **kwargs)
        jax.debug.callback(lambda v: history.append(np.asarray(v)), w, ordered=True)
        return final, out

    with monkeypatch.context() as patch:
        patch.setattr(owner, name, setup)
        patch.setattr(jax.lax, "scan", scan)
        forced = _cube(mesh).run(until_decay=0, decay_max_steps=cap, skip_preflight=True)
        jax.block_until_ready(forced.time_series)
        jax.effects_barrier()
    w = np.concatenate(history).astype(np.float64)
    assert len(w) == cap
    return w, built["n_end"]


def _declared_stop(w, n_end, *, interval, thr, min_steps, consecutive=2):
    """The rule, written out: largest W over every boundary k*I (the floor
    1e-12 of it, counted only where W has stopped falling: >= 0.95 of the
    previous boundary's W per 50 steps), running-max reference over boundaries >= n_end."""
    ref, largest, below, rows, previous = 0.0, 0.0, 0, [], None
    for count in range(interval, len(w), interval):
        largest = max(largest, w[count])
        last, previous = previous, w[count]
        if count < n_end:
            continue
        ref = max(ref, w[count])
        flat = last is not None and w[count] >= 0.95 ** (interval / 50) * last
        below = below + 1 if (w[count] < thr * ref or (w[count] < 1e-12 * largest and flat)) else 0
        if count >= min_steps:
            rows.append((count, w[count], ref))
            if below >= consecutive:
                return count, rows
    return None, rows


@pytest.mark.parametrize("mesh", ["uniform", "graded", "uniform-upml", "graded-periodic"])
def test_stop_is_the_declared_rule_on_an_independent_history(mesh, monkeypatch,
                                                              record_property):
    from rfx.runners import nonuniform as nu_runner

    cap, thr, min_steps = 600, 1e-3, 100
    w, n_end = _inline_history(mesh, cap, monkeypatch)
    loop, returned = nu_runner.run_nonuniform_until_decay, {}

    def keep(*args, **kwargs):
        returned.clear()
        returned.update(loop(*args, **kwargs))
        return returned
    monkeypatch.setattr(nu_runner, "run_nonuniform_until_decay", keep)
    for interval in (50, 17):
        expected, rows = _declared_stop(w, n_end, interval=interval, thr=thr,
                                        min_steps=min_steps)
        result = _cube(mesh).run(until_decay=thr, decay_check_interval=interval,
                                 decay_min_steps=min_steps, decay_max_steps=cap - 1,
                                 skip_preflight=True)
        got = len(result.time_series)
        record_property(f"stop_I{interval}", (n_end, expected, got))
        assert expected is not None and got == expected, (interval, n_end, expected, got)
        if mesh.startswith("graded"):
            # The loop's own rows: same boundaries, W and reference to 1e-4.
            checks = returned["decay_checks"]
            assert [c for c, _, _ in checks] == [c for c, _, _ in rows]
            np.testing.assert_allclose(np.asarray(checks)[:, 1:], np.asarray(rows)[:, 1:],
                                       rtol=1e-4)


# --- (d) ---------------------------------------------------------------------

@pytest.mark.parametrize("mesh", ["uniform", "graded"])
def test_stopped_record_is_the_forced_prefix(mesh, record_property):
    stopped = np.asarray(_cube(mesh).run(until_decay=1e-3, decay_max_steps=1000,
                                         skip_preflight=True).time_series)
    m = len(stopped)
    assert m < 1000
    forced = np.asarray(_cube(mesh).run(until_decay=0, decay_max_steps=m + 60,
                                        skip_preflight=True).time_series)[:m]
    ulp = float(np.max(np.abs(stopped.astype(np.float64) - forced.astype(np.float64)))
                / np.spacing(np.float32(np.abs(forced).max())))
    record_property("stop_and_max_ulp_of_probe_peak", (m, ulp))
    assert stopped.dtype == forced.dtype
    if mesh == "graded":
        assert stopped.tobytes() == forced.tobytes()
    else:
        assert ulp <= 9


# --- (e) ---------------------------------------------------------------------

@pytest.mark.parametrize("mesh", ["uniform", "graded"])
@pytest.mark.parametrize("consecutive", [1, 0, -1])
def test_energy_criterion_refuses_fewer_than_two(mesh, consecutive):
    with pytest.raises(ValueError, match="decay_energy_consecutive must be >= 2"):
        _cube(mesh).run(until_decay=1e-3, decay_max_steps=300,
                        decay_energy_consecutive=consecutive, skip_preflight=True)


def test_graded_stop_fn_route_still_takes_one():
    from rfx.nonuniform import run_nonuniform_until_decay
    from tests.unit.nonuniform.test_nonuniform_until_decay import _build_nu_case

    grid, materials, sources, probes = _build_nu_case(60)
    result = run_nonuniform_until_decay(
        grid, materials, decay_by=0.0, decay_energy_consecutive=1, min_steps=61,
        max_steps=60, check_interval=20, sources=sources, probes=probes,
        stop_fn=lambda count, peek: count >= 40)
    assert len(result["time_series"]) == 40


# --- (f) ----------------------------------------------------------------------

@pytest.mark.parametrize("mesh", ["uniform", "graded"])
def test_driven_waveguide_port_runs_and_stops(mesh, record_property):
    """TE10 guide (23 mm x 10 mm, PEC walls) fed by a driven port, CPML at the x
    ends; the drive (10 GHz, 0.3 relative bandwidth) is above the 6.5 GHz cutoff."""
    graded = {} if mesh == "uniform" else {"dz_profile": np.full(10, 1e-3)}
    sim = Simulation(freq_max=12e9, domain=(.06, .023, .01), dx=1e-3, cpml_layers=6,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"}, **graded)
    sim.add_waveguide_port(.01, direction="+x", mode=(1, 0), mode_type="TE",
                           freqs=np.array([8e9, 10e9]), f0=10e9, bandwidth=0.3)
    sim.add_probe((.03, .01, .005), "ez")
    cap = 6000
    result = sim.run(until_decay=1e-3, decay_max_steps=cap, skip_preflight=True,
                     compute_s_params=False)
    stop = [d for d in result.diagnostics if d.code == "until_decay.stop"]
    assert len(stop) == 1
    values = dict(stop[0].values)
    record_property("stop", (len(result.time_series), values))
    assert values["drive_end_step"] < len(result.time_series) < cap, values
    assert values["criterion"] != "cap" and values["stop_step"] == len(result.time_series)


# --- (g) ----------------------------------------------------------------------

@pytest.mark.parametrize("mesh", ["uniform", "graded"])
def test_decay_record_does_not_depend_on_n_steps(mesh, record_property):
    """A 3 GHz pulse of 0.3 relative bandwidth lasts about 1300 steps, longer than the
    875 steps n_steps defaults to (20 periods of freq_max). Sized by n_steps, its drive
    table was cut at 875 on the uniform decay path: the cut left a static charge and the
    run went to its cap. The record must equal the one sized for decay_max_steps."""
    graded = {} if mesh == "uniform" else {"dz_profile": np.full(10, 1e-3)}

    def run(n_steps):
        sim = Simulation(freq_max=12e9, domain=(.012, .012, .01), dx=1e-3, cpml_layers=6,
                         boundary="cpml", **graded)
        sim.add_source((.006, .006, .005), "ez", waveform=GaussianPulse(f0=3e9, bandwidth=0.3))
        sim.add_probe((.008, .006, .005), "ez")
        sim.add_dft_plane_probe(axis="x", coordinate=.007, component="ez",
                                freqs=np.array([2e9, 3e9]), name="pz")
        return sim.run(until_decay=1e-3, decay_max_steps=6000, n_steps=n_steps,
                       skip_preflight=True)

    default, sized = run(None), run(6000)
    record_property("stops", (len(default.time_series), len(sized.time_series)))
    assert len(default.time_series) == len(sized.time_series) < 6000
    assert np.asarray(default.time_series).tobytes() == np.asarray(sized.time_series).tobytes()
    assert (np.asarray(default.dft_planes["pz"].accumulator).tobytes()
            == np.asarray(sized.dft_planes["pz"].accumulator).tobytes())


# --- (h) ----------------------------------------------------------------------

@pytest.mark.parametrize("mesh", ["uniform", "graded"])
def test_cap_advisory_names_charge_relaxing_in_a_lossy_medium(mesh):
    """The design cube with sigma = 0.05 S/m: conduction moves charge onto the cube's
    surfaces during the drive, and it relaxes with eps/sigma = 6 eps0 / 0.05 = 1.06 ns
    after it. At 1e-4 the run reaches its cap with an electrostatic remnant while no
    drive has DC (GaussianPulse deposits none); the advisory names the cube and its
    relaxation time in steps."""
    from rfx.core.yee import EPS_0

    sim = _cube(mesh)
    sim.add_material("cube_eps6", eps_r=6, sigma=0.05)
    with pytest.warns(UserWarning) as caught:
        result = sim.run(until_decay=1e-4, decay_max_steps=1500, skip_preflight=True)
    stop = [d for d in result.diagnostics if d.code == "until_decay.stop"][0]
    assert stop.values["criterion"] == "cap"
    message = [str(w.message) for w in caught if "H-share" in str(w.message)]
    assert len(message) == 1, [str(w.message) for w in caught]
    steps = 6 * EPS_0 / 0.05 / float(result.dt)
    assert "charge relaxing in a lossy medium (relaxation time eps/sigma)" in message[0]
    assert "'cube_eps6' (eps_r 6, sigma 0.05 S/m)" in message[0]
    assert f"= {steps:.0f} steps" in message[0], (steps, message[0])
    for remedy in ("fixed n_steps", "radiated_flux_box", "larger until_decay threshold"):
        assert remedy in message[0]


# --- (i) ----------------------------------------------------------------------

@pytest.mark.parametrize("mesh", ["uniform", "graded"])
@pytest.mark.parametrize("window", ["hann", "tukey"])
def test_windowed_flux_monitor_is_refused_with_until_decay(mesh, window):
    """The decay loops accumulate flux with rectangular weights; a declared window
    would need the run's length in advance. Refused before the first step."""
    sim = _cube(mesh)
    sim.add_flux_monitor(axis="x", coordinate=18e-3, freqs=np.array([8e9, 10e9]),
                         dft_window=window)
    with pytest.raises(ValueError, match=f"dft_window='{window}'.*until_decay"):
        sim.run(until_decay=1e-3, decay_max_steps=300, skip_preflight=True)


# --- (j) ----------------------------------------------------------------------

def test_two_d_tez_stops_on_the_threshold():
    """A 40 mm 2-D TEz CPML square with an Ey current source: W must read Ex, Ey and
    Hz. Read as TMz (Ez, Hx, Hy, all zero here) W stays 0 and the run went to its cap."""
    sim = Simulation(freq_max=14e9, domain=(.04, .04, .001), dx=1e-3, boundary="cpml",
                     mode="2d_tez")
    sim.add_source((.02, .02, 0.0), "ey", waveform=GaussianPulse(f0=10e9, bandwidth=1.0),
                   amplitude_kind="current")
    sim.add_probe((.025, .021, 0.0), "ey")
    result = sim.run(until_decay=1e-3, decay_max_steps=3000, skip_preflight=True)
    stop = [d for d in result.diagnostics if d.code == "until_decay.stop"][0]
    probe = np.abs(np.asarray(result.time_series)[:, 0])
    assert stop.values["criterion"] == "threshold", dict(stop.values)
    assert len(probe) < 3000 and probe[-50:].max() < 1e-2 * probe.max()


# --- (k) ----------------------------------------------------------------------

def test_mixed_precision_energy_is_finite_and_stops_on_the_threshold():
    """The open 60 mm box with a 2000 V/m field source at precision="mixed": squaring
    float16 fields overflowed W (its largest value and the floor read inf)."""
    import math

    sim = Simulation(freq_max=14e9, domain=(.06, .06, .06), dx=1e-3, boundary="cpml",
                     precision="mixed")
    sim.add_source((.009, .030, .03), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(**PULSE, amplitude=2000.0))
    sim.add_probe((.051, .031, .031), "ez")
    result = sim.run(until_decay=1e-3, decay_check_interval=10, decay_min_steps=10,
                     decay_max_steps=3000, skip_preflight=True)
    stop = dict([d for d in result.diagnostics if d.code == "until_decay.stop"][0].values)
    probe = np.abs(np.asarray(result.time_series, dtype=np.float64)[:, 0])
    assert math.isfinite(stop["floor_ratio"]) and stop["criterion"] == "threshold", stop
    assert stop["reached"] < 1e-3 and int(np.argmax(probe)) < len(probe) - 10, stop


# --- (l) ----------------------------------------------------------------------

def test_settling_verdict_leads_with_the_stop_line_on_a_cap():
    """A soft source with a ModulatedGaussian (relative DC 0.68) holds the energy up:
    the run ends on its cap, and the verdict a user reads about the record names that
    first. A threshold end leaves the verdict's reason as it was."""
    from rfx import ModulatedGaussian

    sim = Simulation(freq_max=4e9, domain=(0.02, 0.02, 0.02), dx=1e-3, cpml_layers=4,
                     boundary="cpml")
    sim.add_source((0.01, 0.01, 0.01), "ez", waveform=ModulatedGaussian(f0=2.2e9, bandwidth=1.2))
    sim.add_probe((0.006, 0.006, 0.006), "ez")
    with pytest.warns(UserWarning):
        capped = sim.run(until_decay=1e-3, decay_min_steps=200, decay_max_steps=600,
                         skip_preflight=True)
    assert capped.settling_witness["reason"].startswith("until_decay ended at step 600"), \
        capped.settling_witness["reason"]
    stopped = _cube("uniform").run(until_decay=1e-3, decay_max_steps=1000, skip_preflight=True)
    assert not stopped.settling_witness["reason"].startswith("until_decay ended")


# --- (m) ----------------------------------------------------------------------

@pytest.mark.parametrize("mesh", ["uniform", "graded"])
def test_side_step_program_holds_no_grid_size_constant(mesh, monkeypatch):
    """Lower the fused side step + W program at the first boundary of the dielectric
    cube and count the float32 constant elements in its HLO: per-cell arrays (eps, the
    update coefficients) compiled in as constants would be grid-size (about 3e4 cells
    here, several arrays)."""
    import re

    import rfx.measurement.decay as decay

    counts, original = [], decay.boundary_energy

    def spy(*args, **kwargs):
        measure = original(*args, **kwargs)

        def first(carry, count):
            if not counts:
                text = measure.lower(carry, count).as_text()
                counts.append(sum(int(np.prod([int(d) for d in m.group(1).split("x") if d]))
                                  for m in re.finditer(r"constant dense<[^>]*> : tensor<([0-9x]+)xf32>", text)))
            return measure(carry, count)
        return first
    monkeypatch.setattr(decay, "boundary_energy", spy)
    _cube(mesh).run(until_decay=1e-3, decay_max_steps=200, skip_preflight=True)
    assert counts and counts[0] < 10_000, counts
