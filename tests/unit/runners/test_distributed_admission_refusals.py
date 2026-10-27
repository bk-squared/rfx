"""Refuse the input classes the distributed lanes would drop or get wrong.

Periodic/Bloch boundaries, reference-plane and passive ports, surface monitors
(flux, DFT planes), Kerr materials, lumped RLC elements, subgridding.

Exercise the public dispatch and the shard_map runner entry with two CPU
devices. The pmap runner entry went with that runner (#1296).
PEC controls use the relative tolerance from
``test_distributed.TestDistributedRunner.test_distributed_matches_single_pec``.
"""

import warnings

import jax
import numpy as np
import pytest

from rfx import Box, DebyePole, GaussianPulse, Result, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.runners import distributed_v2


N_STEPS = 40
ENTRIES = ("api", "v2")
LANES = {
    "api": "distributed multi-device run()",
    "v2": "distributed (v2) runner",
}


def _mixed_lorentz_admission_scene(eps_r=2.0):
    """The mixed-dispersion corner-gradient scene, at its model timestep."""
    from rfx.materials.lorentz import lorentz_pole

    nx, ny, nz = 24, 24, 48
    dx = 1e-3
    sim = Simulation(freq_max=5e9, domain=(nx * dx, ny * dx, nz * dx),
                     dx=dx, boundary="cpml", cpml_layers=8,
                     dx_profile=np.full(nx, dx), dz_profile=np.full(nz, dx))
    sim.add_material("disp_slab", eps_r=eps_r,
                     debye_poles=[DebyePole(delta_eps=2.0, tau=5e-12)],
                     lorentz_poles=[lorentz_pole(delta_eps=1.5,
                                                omega_0=2.0 * np.pi * 30e9, delta=1e10)])
    sim.add(Box((10 * dx, 10 * dx, 6 * dx),
                (14 * dx, 14 * dx, 42 * dx)), material="disp_slab")
    sim.add_source(position=(10 * dx, 12 * dx, 24 * dx), component="ez")
    sim.add_probe(position=(13 * dx, 12 * dx, 24 * dx), component="ez")
    return sim


@pytest.mark.parametrize("entry", ("forward_override", "forward_sharded_override", "run_declared"))
def test_mixed_lorentz_refusal_matches_one_device(entry):
    """Distributed admission preserves the public one-device refusal text."""
    import jax.numpy as jnp

    sim = _mixed_lorentz_admission_scene(1.0 if entry == "run_declared" else 2.0)
    if entry.startswith("forward"):
        grid = sim._build_nonuniform_grid()
        eps = jnp.ones(tuple(int(n) for n in grid.shape), dtype=jnp.float32)
        sharded = (sim.shard_distributed_override(eps, _devices())
                   if entry == "forward_sharded_override" else eps)
        if entry == "forward_sharded_override":
            # 41 x cells on two devices: the forward layout carries a padding
            # row (value 0) after the grid's rows, which the judge must not read.
            assert sharded.shape[0] > grid.shape[0], (sharded.shape, grid.shape)
        def call(distributed):
            return sim.forward(eps_override=sharded if distributed else eps, n_steps=30,
                               emit_time_series=True,
                               **(dict(distributed=True, devices=_devices()) if distributed else {}))
    else:
        def call(distributed):
            return sim.run(n_steps=30, **(dict(devices=_devices()) if distributed else {}))
    with pytest.raises(ValueError) as native:
        call(False)
    with pytest.raises(ValueError) as distributed:
        call(True)
    assert str(distributed.value) == str(native.value)


def _devices():
    devices = jax.devices("cpu")
    assert len(devices) >= 2, "requires the root conftest's two CPU devices"
    return devices[:2]


def test_two_device_pole_override_is_judged_eagerly_but_not_under_outer_jit():
    import jax.numpy as jnp

    sim = _mixed_lorentz_admission_scene(2.0)
    grid = sim._build_nonuniform_grid()
    devices = _devices()

    def forward(eps):
        return sim.forward(eps_override=eps, n_steps=2, checkpoint=False,
                           emit_time_series=True, skip_preflight=True,
                           distributed=True, devices=devices).time_series

    # Literal overrides and expected shape; no admission helper supplies them.
    stable = jnp.full(grid.shape, 2., jnp.float32)
    unstable = jnp.ones(grid.shape, jnp.float32)
    with pytest.raises(ValueError, match=r"Lorentz/Drude poles \(omega_0=.*unstable at this model's time step"):
        forward(unstable)
    eager = np.asarray(forward(stable))
    assert eager.shape == (2, 1)
    assert np.all(np.isfinite(eager))

    # Documented limitation: outer-traced material values are not judged on
    # multiple devices. Pin completion/shape, not stability, for both inputs.
    compiled = jax.jit(forward)
    for eps in (stable, unstable):
        assert np.asarray(compiled(eps)).shape == (2, 1)


def test_two_device_traced_declared_pole_material_runs():
    import jax.numpy as jnp

    devices = _devices()

    def forward(eps):
        sim = _mixed_lorentz_admission_scene(eps)
        return sim.forward(n_steps=2, checkpoint=False, emit_time_series=True,
                           skip_preflight=True, distributed=True, devices=devices).time_series

    # Independent literal material values and observation dimensions.
    with pytest.raises(ValueError, match=r"Lorentz/Drude poles \(omega_0=.*unstable at this model's time step"):
        forward(jnp.float32(1.0))
    result = np.asarray(jax.jit(forward)(jnp.float32(2.0)))
    assert result.shape == (2, 1)
    assert np.all(np.isfinite(result))


def test_one_device_distributed_outer_jit_refuses_unstable_pole_override(monkeypatch):
    import jax.numpy as jnp
    from rfx.model import distributed_ports

    apply_override = distributed_ports.apply_material_overrides

    def one_device_override(*args, **kwargs):
        # This run uses one device even though the process has two available.
        # Pin admission here as well as the independent downstream slab check.
        assert kwargs["in_program"] is True, "one-device override must be judged in program"
        return apply_override(*args, **kwargs)

    monkeypatch.setattr(distributed_ports, "apply_material_overrides", one_device_override)

    sim = _mixed_lorentz_admission_scene()
    grid = sim._build_nonuniform_grid()
    devices = _devices()[:1]

    def forward(eps):
        return sim.forward(eps_override=eps, n_steps=2, checkpoint=False,
                           emit_time_series=True, skip_preflight=True,
                           distributed=True, devices=devices).time_series

    with pytest.raises(Exception, match=r"Lorentz/Drude poles \(omega_0=.*unstable at this model's time step"):
        np.asarray(jax.jit(forward)(jnp.ones(grid.shape, jnp.float32)))


def _build(*, entry="api", periodic="", boundary="pec", port=None,
           flux=False, ntff=False, kerr=False, debye=False, rlc=None,
           refine=False, dft=False, msl=False):
    """A PEC box has a source at x=6 mm and Ez probes at x=12 and 22 mm."""
    # v2 pads the 25-cell x grid.
    domain_x = 24e-3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        sim = Simulation(freq_max=15e9, domain=(domain_x, 12e-3, 12e-3),
                         dx=1e-3, boundary=(BoundarySpec(**{axis: "periodic" if axis in periodic else boundary
                                                   for axis in "xyz"}) if periodic else boundary))
        if port is None:
            sim.add_source(position=(6e-3, 6e-3, 6e-3), component="ez",
                           amplitude_kind="field")
        else:
            sim.add_port(position=(6e-3, 6e-3, 6e-3), component="ez", **port)
        for x in (12e-3, 22e-3):
            sim.add_probe(position=(x, 6e-3, 6e-3), component="ez")
        if flux:
            sim.add_flux_monitor(axis="x", coordinate=12e-3, n_freqs=3)
        if ntff:
            sim.add_ntff_box((4e-3, 4e-3, 4e-3), (20e-3, 8e-3, 8e-3),
                             n_freqs=3)
        if kerr:
            sim.add_material("kerr", eps_r=2.0, chi3=1e-2)
            sim.add(Box((8e-3, 3e-3, 3e-3), (16e-3, 9e-3, 9e-3)), material="kerr")
        if debye:
            sim.add_material("debye", eps_r=2.0,
                             debye_poles=[DebyePole(delta_eps=1.5, tau=1e-11)])
            sim.add(Box((8e-3, 3e-3, 3e-3), (16e-3, 9e-3, 9e-3)), material="debye")
        if rlc is not None:
            sim.add_lumped_rlc((9e-3, 6e-3, 6e-3), "ez", **rlc)
        if refine:
            sim.add_refinement(z_range=(4e-3, 8e-3), ratio=2)
        if dft:
            sim.add_dft_plane_probe(axis="x", coordinate=12e-3, n_freqs=3)
        if msl:
            # A 2 mm PEC trace on a 1 mm substrate over the z-lo PEC wall.
            sim.add_material("substrate", eps_r=3.66)
            sim.add(Box((0, 0, 0), (domain_x, 12e-3, 1e-3)), material="substrate")
            sim.add(Box((1e-3, 5e-3, 1e-3), (domain_x - 1e-3, 7e-3, 2e-3)),
                    material="pec")
            sim.add_msl_port(position=(2e-3, 6e-3, 0.0), width=2e-3, height=1e-3,
                             direction="+x", impedance=50.0)
    return sim


def _run(sim, entry, **kwargs):
    if entry == "api":
        return sim.run(n_steps=N_STEPS, devices=_devices(), **kwargs)
    return distributed_v2.run_distributed(sim, n_steps=N_STEPS, devices=_devices(),
                                          **kwargs)


def _assert_refused(sim, entry, *features, **kwargs):
    with pytest.raises(NotImplementedError) as exc:
        _run(sim, entry, **kwargs)
    message = str(exc.value)
    if entry == "api":
        remedy = "omit devices=... (use a single-device run() instead)"
    else:
        remedy = "call sim.run(...) without devices= instead of calling this runner"
        assert "omit devices=..." not in message, message
    for feature in (*features, LANES[entry], remedy, "rfx.runners."):
        assert feature in message, message


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("declaration", ("axes", "boundary_spec"))
def test_periodic_boundaries_are_refused(entry, declaration):
    """A box periodic in y must not run with the declared non-periodic wall."""
    if declaration == "axes":
        sim = _build(entry=entry, periodic="y")
    else:
        spec = BoundarySpec(
            x=Boundary(lo="pec", hi="pec"),
            y=Boundary(lo="periodic", hi="periodic"),
            z=Boundary(lo="pec", hi="pec"),
        )
        sim = _build(entry=entry, boundary=spec)
    assert sim._periodic_axes == "y"
    _assert_refused(sim, entry, "periodic", "'y'", "non-periodic wall")


@pytest.mark.parametrize("entry", ENTRIES)
def test_reference_plane_wire_port_is_refused(entry):
    sim = _build(entry=entry, port={"impedance": 50.0, "extent": 3e-3,
                                   "reference_plane_cells": 2, "direction": "+x"})
    with pytest.raises(NotImplementedError, match="reference_plane_cells.*V_ref.*one device"):
        _run(sim, entry)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("waveform", ("explicit", "default"))
def test_passive_port_is_refused(entry, waveform):
    """A passive 50 ohm port has no applied excitation, including waveform=None."""
    port = {"impedance": 50.0, "excite": False}
    if waveform == "explicit":
        port["waveform"] = GaussianPulse(f0=7.5e9, bandwidth=0.8)
    sim = _build(entry=entry, port=port)
    assert (sim._ports[0].waveform is None) == (waveform == "default")
    _assert_refused(sim, entry, "excite=False", "drive the port", "waveform=None")


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("monitor,feature", (
    ("flux", "add_flux_monitor()"),
))
def test_surface_monitor_is_carried(entry, monitor, feature):
    """Uniform flux records use the shared one-device arithmetic."""
    sim = _build(entry=entry, **{monitor: True})
    reference = sim.run(n_steps=N_STEPS, skip_preflight=True)
    actual = _run(sim, entry, skip_preflight=True)
    for name, expected in reference.flux_monitors.items():
        for component in ("e1_dft", "e2_dft", "h1_dft", "h2_dft"):
            np.testing.assert_array_equal(getattr(actual.flux_monitors[name], component),
                                          getattr(expected, component))


def test_explicit_bloch_phase_is_refused():
    """A Bloch boundary specifies a phase even when periodic axes are unset."""
    sim = _build()
    phase = (0.0, 0.1, 0.0)
    with pytest.raises(NotImplementedError, match="Bloch.*") as exc:
        distributed_v2.refuse_unsupported_distributed_features(
            sim, lane="direct helper", bloch=phase)
    assert ("call sim.run(...) without devices= instead of calling this runner"
            in str(exc.value))
    _assert_refused(_build(entry="v2"), "v2", "Bloch", bloch=phase)
    sim._bloch = phase
    _assert_refused(sim, "api", "Bloch")


@pytest.mark.parametrize("entry", ENTRIES)
def test_kerr_material_is_refused(entry):
    """A chi3 block between source and probe: this lane drops chi3 (linear physics)."""
    sim = _build(entry=entry, kerr=True)
    _assert_refused(sim, entry, "Kerr chi3", "'kerr'", "as linear")


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("element", ({"R": 10.0}, {"R": 10.0, "L": 1e-9}),
                         ids=("stamped_R", "series_RL"))
def test_lumped_rlc_element_is_refused(entry, element):
    """An element between source and probe: this lane never applies it (#1239).

    Measured before the refusal (#1239's 12 mm box, 10 ohm): the multi-device
    probe peak was bit-identical to the box without the element and 76 % above
    the single-device peak, where the element lowers it by 43 %. A lone R is
    stamped into the material; R+L has its own time update.
    """
    sim = _build(entry=entry, rlc=element)
    _assert_refused(sim, entry, "add_lumped_rlc()", "as if the elements were absent")


@pytest.mark.parametrize("entry", ENTRIES)
def test_subgrid_refinement_is_refused(entry):
    """The lanes have no subgrid coupling; before the refusal a refined box ran
    bit-identical to the unrefined one (review of #1239)."""
    sim = _build(entry=entry, refine=True)
    _assert_refused(sim, entry, "add_refinement()")


@pytest.mark.parametrize("entry", ENTRIES)
def test_msl_port_is_refused(entry):
    """Before the refusal a trace fed by an MSL port read exactly 0 under the
    trace on two devices (22.3 on one) - review of #1239."""
    sim = _build(entry=entry, msl=True)
    _assert_refused(sim, entry, "add_msl_port()")


# How the multi-device lane treats every Simulation attribute is the
# run_distributed column of tests/contracts/path_disposition.py, which covers
# every time-stepping path; its completeness test is
# tests/contracts/test_path_disposition.py.


@pytest.mark.parametrize("entry", ENTRIES)
def test_dft_plane_probe_is_carried(entry):
    """Public and direct uniform entries return the plane records."""
    sim = _build(entry=entry, dft=True)
    reference = sim.run(n_steps=N_STEPS, skip_preflight=True)
    actual = _run(sim, entry, skip_preflight=True)
    for name, expected in reference.dft_planes.items():
        np.testing.assert_array_equal(actual.dft_planes[name].accumulator, expected.accumulator)


@pytest.mark.parametrize("entry", ("api", "v2"))
def test_lumped_port_with_debye_block_matches_native(entry):
    """A 50 ohm port drives a box with a Debye block between port and probes.

    The staged dispersion coefficients must come from the materials the run
    actually uses (x-padded, with the port's resistive sigma folded in); a
    staging fed the unpadded or port-less materials moves this trace by more
    than the probe's own peak.
    """
    sim = _build(entry=entry, port={"impedance": 50.0}, debye=True)
    native = np.asarray(sim.run(n_steps=N_STEPS).time_series)
    multi = np.asarray(_run(
        sim, entry, **({"compute_s_params": False} if entry == "api" else {})
    ).time_series)
    assert np.max(np.abs(native)) > 0
    relative = np.max(np.abs(native - multi)) / np.max(np.abs(native))
    assert relative < 1e-4, f"{entry}: relative Ez difference {relative:.9e}"


@pytest.mark.parametrize("source", ("point", "lumped_port"))
def test_default_pec_model_matches_native(source):
    """A PEC box contains a point source or an excited single-cell 50 ohm port."""
    port = {"impedance": 50.0} if source == "lumped_port" else None
    for entry in ENTRIES:
        sim = _build(entry=entry, port=port)
        assert not sim._periodic_axes
        assert not sim._flux_monitors and sim._ntff is None
        assert all(p.extent is None and p.excite for p in sim._ports)
        native = np.asarray(sim.run(n_steps=N_STEPS).time_series)
        multi = np.asarray(_run(
            sim, entry, **({"compute_s_params": False} if entry == "api" else {})
        ).time_series)
        assert native.shape == multi.shape == (N_STEPS, 2)
        assert np.max(np.abs(native)) > 0
        relative = np.max(np.abs(native - multi)) / (np.max(np.abs(native)) + 1e-30)
        # Same PEC criterion as TestDistributedRunner.test_distributed_matches_single_pec.
        assert relative < 1e-4, f"{entry}: relative Ez difference {relative:.9e}"


@pytest.mark.parametrize("entry", ["api", "v2", "forward"])
@pytest.mark.parametrize("kind,match", [
    ("open", "open-slab.*periodic y/z"),
    ("methodB", "Method B oblique"),
    ("bloch", "2-D Bloch"),
    ("waveguide", "waveguide ports"),
])
def test_distributed_source_refusals(entry, kind, match, monkeypatch):
    sim = Simulation(freq_max=15e9, domain=(.024, .018, .016), dx=.001,
                     boundary="cpml", cpml_layers=4)
    if kind == "waveguide":
        sim.add_waveguide_port(x_position=.006, y_range=(.002, .010),
            z_range=(.002, .010), mode=(1, 0), mode_type="TE", direction="+x", f0=7.5e9)
    else:
        sim.add_tfsf_source(f0=7.5e9, angle_deg=20 if kind in ("methodB", "bloch") else 0,
                            method=kind if kind in ("methodB", "bloch") else "bloch")
    # Refusal precedes any compiled scan, never a nested native solve.
    monkeypatch.setattr(jax.lax, "scan", lambda *a, **k: pytest.fail("stepped refused source"))
    with pytest.raises(NotImplementedError, match=match):
        if entry == "forward":
            sim.forward(n_steps=4, distributed=True, devices=_devices(), checkpoint=False)
        else:
            _run(sim, entry)


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("kind", ["tfsf", "waveguide"])
def test_graded_sources_refused(entry, kind):
    from tests.unit.runners.test_distributed_forward_one_device_fallback import scene
    sim, _ = scene(kind)
    options = {"devices": _devices()}
    if entry == "forward":
        options.update(distributed=True, checkpoint=False)
    with pytest.raises(NotImplementedError, match="waveguide ports" if kind == "waveguide" else "TFSF.*graded multi-device"):
        getattr(sim, entry)(n_steps=4, **options)


def test_face_record_refused_before_split(monkeypatch):
    from dataclasses import replace
    from jax.sharding import Mesh
    from rfx.sources.tfsf import tfsf_from_declaration
    from rfx.stepping import slab_tfsf
    from rfx.stepping.slab import Slab
    sim = Simulation(freq_max=15e9, domain=(.024, .018, .016), dx=.001, cpml_layers=4)
    sim.add_tfsf_source(f0=7.5e9, closed_box=True)
    grid = sim._build_grid()
    cfg, line = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, n_steps=4)
    cfg = replace(cfg, face_record=(np.zeros((4, 1)), np.zeros((4, 1))))
    monkeypatch.setattr("rfx.sources.tfsf.tfsf_from_declaration", lambda *a, **k: (cfg, line))
    monkeypatch.setattr(slab_tfsf, "split_tables", lambda *a: pytest.fail("split a replay"))
    with pytest.raises(NotImplementedError, match="face_record replay"):
        slab_tfsf.prepare(sim, grid, 4, Slab(grid.nx, 2), Mesh(np.array(_devices()[:2]), ("x",)))


def test_waveguide_refuses_private_record_probes():
    sim = Simulation(freq_max=15e9, domain=(.024, .012, .012), dx=.001, boundary="cpml")
    sim.add_waveguide_port(x_position=.006, y_range=(.002, .010), z_range=(.002, .010),
        mode=(1, 0), mode_type="TE", direction="+x", f0=7.5e9)
    with pytest.raises(NotImplementedError, match="waveguide ports"):
        _run(sim, "v2", _record_probes=())


def test_closed_tfsf_accepts_inactive_conformal_pec_override():
    sim = Simulation(freq_max=15e9, domain=(.024, .018, .016), dx=.001, cpml_layers=4)
    sim.add_tfsf_source(f0=7.5e9, closed_box=True)
    result = _run(sim, "v2", conformal_pec=False)
    assert np.any(np.asarray(result.state.ez) != 0)
    assert all(d.code != "distributed.one_device_fallback" for d in result.diagnostics)


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_graded_passive_port_is_carried(entry):
    from tests.unit.runners.test_distributed_nu_ports import scene, evaluate
    devices = _devices()
    sim = scene('C3', len(devices))
    one = evaluate(sim, entry, 32)
    multi = evaluate(sim, entry, 32, devices)
    assert len(multi.sparam_time_records) == 2
    for a, b in zip(one.sparam_time_records, multi.sparam_time_records):
        np.testing.assert_array_equal(a.samples, b.samples)


@pytest.mark.parametrize('request_s', [{}, {'compute_s_params': True}, {'s_param_freqs': [3e9, 5e9, 7e9]}],
                         ids=['default', 'explicit', 'freqs'])
def test_uniform_user_jit_refuses_port_s(request_s):
    from tests.unit.runners.test_distributed_uniform_user_jit import scene

    if len(jax.devices('cpu')) < 2:
        pytest.skip('requires two CPU devices')
    sim = scene('pec', port=True)

    def call():
        return sim.run(n_steps=60, devices=jax.devices('cpu')[:2], skip_preflight=True, **request_s).s_params

    with pytest.raises(NotImplementedError, match="cannot compute port S-parameters inside jax.jit"):
        jax.jit(call)()
    if not request_s:
        # The refusal is for the staged request only: the same call made directly still measures S.
        assert call() is not None


def _closed_box_scene(**angles):
    sim = Simulation(freq_max=15e9, domain=(.024, .018, .016), dx=.001,
                     boundary="cpml", cpml_layers=4, snap="declared")
    sim.add_tfsf_source(f0=7.5e9, closed_box=True, margin=3, **angles)
    sim.add_probe((.015, .009, .008), "ez")
    return sim


def _closed_box_execute(sim, entry, count):
    options = dict(n_steps=4)
    if count > 1:
        options["devices"] = _devices()[:count]
    if entry == "forward":
        options.update(distributed=count > 1, checkpoint=False)
    return getattr(sim, entry)(**options)


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_closed_box_late_sheet_face_refusal_matches_one_device(entry):
    """A declaration-time cell check cannot see a later realized PEC sheet."""
    errors = []
    for count in (1, 2):
        sim = _closed_box_scene()
        # x=3 mm is the low TF face (CPML padding is outside physical coords).
        sim.add_thin_conductor(Box((.003, .006, .006), (.003, .010, .010)))
        with pytest.raises(ValueError, match="closed_box TFSF requires vacuum on and adjacent to all six faces") as exc:
            _closed_box_execute(sim, entry, count)
        errors.append(exc.value)
    assert type(errors[0]) is type(errors[1])
    assert str(errors[0]) == str(errors[1])


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("count", [1, 2])
def test_closed_box_late_sheet_inside_runs(entry, count):
    sim = _closed_box_scene()
    sim.add_thin_conductor(Box((.012, .006, .006), (.012, .010, .010)))
    result = _closed_box_execute(sim, entry, count)
    assert result.time_series.shape == (4, 1)
    assert np.isfinite(result.time_series).all()


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("angles", [dict(theta_deg=30.), dict(phi_deg=30.)])
def test_oblique_closed_box_refused_only_on_slabs(entry, angles):
    sim = _closed_box_scene(polarization="s", **angles)
    one = _closed_box_execute(sim, entry, 1)
    assert one.time_series.shape == (4, 1)
    with pytest.raises(NotImplementedError, match="oblique closed-box TFSF.*theta_deg or phi_deg"):
        _closed_box_execute(sim, entry, 2)


@pytest.mark.parametrize("angles", [dict(theta_deg=30.), dict(phi_deg=30.)])
def test_realized_oblique_closed_box_refused(angles):
    from rfx.runners._admission import refuse_slab_tfsf
    from rfx.sources.tfsf import tfsf_from_declaration
    sim = _closed_box_scene(polarization="s", **angles)
    cfg, _ = tfsf_from_declaration(sim._tfsf, sim._build_grid(), sim._freq_max, n_steps=4)
    with pytest.raises(NotImplementedError, match="oblique closed-box TFSF"):
        refuse_slab_tfsf(cfg)


def test_closed_box_device_limit_without_mesh():
    from types import SimpleNamespace
    from rfx.runners._admission import refuse_slab_tfsf_devices
    declaration = SimpleNamespace(closed_box=True)
    refuse_slab_tfsf_devices(declaration, 4)
    with pytest.raises(NotImplementedError, match="closed-box TFSF.*on 5 devices.*5 or more are refused"):
        refuse_slab_tfsf_devices(declaration, 5)


def test_uniform_forward_user_jit_refuses_port_s():
    from tests.unit.runners.test_distributed_uniform_user_jit import scene

    if len(jax.devices('cpu')) < 2:
        pytest.skip('requires two CPU devices')
    sim = scene('pec', port=True)
    with pytest.raises(NotImplementedError, match=(
            r"forward\(distributed=True\) on a uniform mesh cannot compute port S-parameters")) as failure:
        jax.jit(lambda: sim.forward(
            n_steps=60, devices=jax.devices('cpu')[:2], distributed=True,
            skip_preflight=True, port_s11_freqs=np.array([3e9, 5e9, 7e9])).s_params)()
    assert str(failure.value) == (
        "forward(distributed=True) on a uniform mesh cannot compute port S-parameters inside jax.jit "
        "(or another JAX trace that stages it): the S-parameters are measured from a second "
        "multi-device run outside the trace. Omit port_s11_freqs for the fields and probe records, "
        "call forward() directly, or run on one device (omit distributed=True).")
