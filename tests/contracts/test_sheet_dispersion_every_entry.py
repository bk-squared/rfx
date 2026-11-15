"""An f0 sheet with a Debye or Lorentz medium is refused at every entry.

The sheet operator replaces the E update at its edges, the pole (ADE)
update among it, so a run that holds both would return the fields of a
different model. One declaration-level check in ``admit()`` refuses the
pair before the first step; each row below enters through one public
entry with ``jax.lax.scan`` patched to fail, so a row that steps fails.

Rows whose entry refuses every f0 sheet before ``admit()`` (the mixed S
calculator, optimize, topology optimization, the material fit) accept
that earlier refusal. The public low-level loops (``rfx.run``,
``run_until_decay``, ``run_nonuniform``) take realized operands from the
caller, call no ``admit()`` and are not refused here.

Multi-device rows need two devices; run them with
XLA_FLAGS=--xla_force_host_platform_device_count=2 set before pytest.
"""
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, GaussianPulse, Simulation
from rfx.errors import RfxRefusal
from rfx.materials.lorentz import LorentzPole


SHEET = Box((4e-3, 4e-3, 12e-3), (16e-3, 16e-3, 12e-3))
# Below the sheet plane (z = 12 mm), off the source and probe columns.
POLE_BOX = Box((2e-3, 2e-3, 2e-3), (6e-3, 6e-3, 6e-3))
OMEGA_0 = 2 * np.pi * 8e9


def add_sheet(sim, box=SHEET):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_thin_conductor(box, sigma_bulk=1e4, surface_impedance_f0=10e9)
    return sim


def add_poles(sim, box=POLE_BOX, kind="debye"):
    if kind == "debye":
        sim.add_material("water", eps_r=4.0,
                         debye_poles=[DebyePole(delta_eps=2.0, tau=8.3e-12)])
    else:
        sim.add_material("water", eps_r=4.0, lorentz_poles=[
            LorentzPole(omega_0=OMEGA_0, delta=1e9, kappa=OMEGA_0 ** 2)])
    sim.add(box, material="water")
    return sim


def cube(graded, *, port=False, poles="debye"):
    """20 mm PEC cube, one z-normal f0 sheet, one pole box."""
    profile = {"dz_profile": [1e-3] * 20} if graded else {}
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.0 if graded else 0.02),
                     dx=1e-3, boundary="pec", **profile)
    add_sheet(sim)
    if port:
        sim.add_port((10e-3, 10e-3, 9e-3), "ez")
    else:
        sim.add_source((10e-3, 10e-3, 8e-3), "ex",
                       waveform=GaussianPulse(f0=5e9), amplitude_kind="field")
    sim.add_probe((10e-3, 10e-3, 14e-3), "ex")
    if poles is not None:
        add_poles(sim, kind=poles)
    return sim


def multidevice(graded):
    from tests.contracts.test_multidevice_f0_sheets import model
    sim = model(graded, "pec", "span", 2)
    # Below the sheet plane (z = 4 mm) and away from the sources.
    return add_poles(sim, Box((.002, .002, .001), (.006, .006, .003)))


def need_devices(count):
    if len(jax.devices()) < count:
        pytest.skip(f"requires {count} CPU devices")


def wr90(graded):
    """Coarse WR-90 two-port, an f0 sheet across the guide, a pole block."""
    from rfx.boundaries.spec import Boundary, BoundarySpec
    profile = {"dy_profile": [3e-3] * 8} if graded else {}
    sim = Simulation(
        freq_max=8e9, domain=(0.06, 0.0 if graded else 8 * 3e-3, 4 * 3e-3), dx=3e-3,
        boundary=BoundarySpec(x="cpml", y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=6, snap="declared", **profile)
    add_sheet(sim, Box((0.02, 0.0, 0.006), (0.03, 0.024, 0.006)))
    add_poles(sim, Box((0.036, 0.006, 0.0), (0.042, 0.018, 0.003)))
    freqs = jnp.linspace(6e9, 7e9, 3)
    for x, d, name in ((0.009, "+x", "l"), (0.051, "-x", "r")):
        sim.add_waveguide_port(x, direction=d, mode=(1, 0), mode_type="TE",
                               freqs=freqs, f0=6.5e9, bandwidth=0.4, name=name)
    return sim


def msl(graded):
    """The MSL thru of test_msl_sheet_threading with its f0 sheet and a
    pole block in the air above the sheet."""
    from tests.unit.sparams import test_msl_sheet_threading as t
    sim = t.build_msl_thru(sheet=("f0", t._sigma_bulk_for_rs0(1.0)),
                           dz_profile=[t.DX] * 21 if graded else None)
    y_c = t.LY / 2.0
    add_poles(sim, Box((5e-3, y_c - 3e-4, t.SHEET_Z + 2 * t.DX),
                       (7e-3, y_c + 3e-4, t.SHEET_Z + 4 * t.DX)))
    return sim


def scan(sim, **options):
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    return compute_lumped_wire_s_matrix_via_scan(sim, np.array([5e9]), n_steps=4,
                                                skip_preflight=True, **options)


def devices2():
    need_devices(2)
    return jax.devices()[:2]


ADMIT_ROWS = {
    "run-uniform": lambda: cube(False).run(n_steps=4, skip_preflight=True),
    "run-uniform-lorentz": lambda: cube(False, poles="lorentz").run(
        n_steps=4, skip_preflight=True),
    "run-graded": lambda: cube(True).run(n_steps=4, skip_preflight=True),
    "forward-uniform": lambda: cube(False).forward(n_steps=4, skip_preflight=True),
    "forward-graded": lambda: cube(True).forward(n_steps=4, skip_preflight=True),
    "forward-graded-lorentz": lambda: cube(True, poles="lorentz").forward(
        n_steps=4, skip_preflight=True),
    "run-devices-uniform": lambda: multidevice(False).run(
        n_steps=4, devices=devices2(), skip_preflight=True, compute_s_params=False),
    "run-devices-graded": lambda: multidevice(True).run(
        n_steps=4, devices=devices2(), skip_preflight=True, compute_s_params=False),
    "forward-distributed-uniform": lambda: multidevice(False).forward(
        n_steps=4, devices=devices2(), skip_preflight=True, distributed=True),
    "forward-distributed-graded": lambda: multidevice(True).forward(
        n_steps=4, devices=devices2(), skip_preflight=True, distributed=True),
    "lumped-wire-scan-uniform": lambda: scan(cube(False, port=True)),
    "lumped-wire-scan-graded": lambda: scan(cube(True, port=True)),
    "lumped-wire-scan-devices": lambda: scan(cube(False, port=True), devices=devices2()),
    "msl-uniform": lambda: msl(False).compute_msl_s_matrix(n_steps=4, skip_preflight=True),
    "msl-graded": lambda: msl(True).compute_msl_s_matrix(n_steps=4, skip_preflight=True),
    "waveguide-uniform": lambda: wr90(False).compute_waveguide_s_matrix(n_steps=4),
    "waveguide-graded": lambda: wr90(True).compute_waveguide_s_matrix(n_steps=4),
}


def _mixed():
    from tests.unit.materials.test_sheet_impedance import _mixed_probe_fed_msl
    sim = add_poles(_mixed_probe_fed_msl(), Box((6.2e-3, 1e-3, 3e-4), (6.8e-3, 2e-3, 4e-4)))
    sim.compute_mixed_s_matrix(freqs=np.linspace(1e9, 4e9, 3), num_periods=1.0,
                               skip_preflight=True)


def _design_cube():
    return cube(False, port=True)


def _optimize():
    from rfx.optimize import DesignRegion, optimize
    region = DesignRegion(corner_lo=(7e-3, 7e-3, 7e-3), corner_hi=(12e-3, 12e-3, 11e-3),
                          eps_range=(1.0, 4.4))
    optimize(_design_cube(), region, lambda r: -jnp.sum(r.time_series ** 2),
             n_iters=1, lr=0.01, verbose=False, n_steps=4, skip_preflight=True)


def _topology():
    from rfx.topology import TopologyDesignRegion, topology_optimize
    sim = _design_cube()
    region = TopologyDesignRegion(corner_lo=(7e-3, 7e-3, 7e-3), corner_hi=(12e-3, 12e-3, 11e-3),
                                  material_bg="air", material_fg="fr4", beta_projection=1.0)
    topology_optimize(sim, region, lambda r: -jnp.sum(r.time_series ** 2),
                      n_iterations=1, learning_rate=0.05, beta_schedule=[(0, 1.0)],
                      verbose=False, skip_preflight=True)


def _material_fit():
    from rfx.differentiable_material_fit import differentiable_material_fit

    def factory(eps_inf, debye_poles, lorentz_poles):
        sim = Simulation(freq_max=5e9, domain=(0.024, 0.009, 0.009))
        sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles)
        sim.add(Box((0.010, 0.0, 0.0), (0.016, 0.009, 0.009)), material="dut")
        add_sheet(sim, Box((0.004, 0.002, 0.004), (0.008, 0.007, 0.004)))
        sim.add_port((0.003, 0.0045, 0.0045), "ez",
                     waveform=GaussianPulse(f0=3e9, bandwidth=0.5))
        sim.add_probe((0.020, 0.0045, 0.0045), component="ez")
        sim.add_probe((0.007, 0.0045, 0.0045), component="ez")
        return sim

    differentiable_material_fit(factory, np.zeros((1, 1, 3), complex),
                                np.linspace(2.0e9, 4.0e9, 3), n_debye_poles=1,
                                n_iterations=1, learning_rate=0.0, verbose=False)


EARLIER_ROWS = {
    "mixed-s-matrix": _mixed,
    "optimize": _optimize,
    "topology-optimize": _topology,
    "material-fit": _material_fit,
}


def _no_scan(monkeypatch):
    monkeypatch.setattr(jax.lax, "scan",
                        lambda *a, **k: pytest.fail("stepped an f0 sheet with a pole medium"))


@pytest.mark.parametrize("row", sorted(ADMIT_ROWS))
def test_sheet_dispersion_refused_at_every_entry(row, monkeypatch):
    _no_scan(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="dispersive") as exc:
            ADMIT_ROWS[row]()
    assert isinstance(exc.value, RfxRefusal), type(exc.value)
    frames = [(Path(str(frame.path)).name, frame.name) for frame in exc.traceback]
    assert ("_admission.py", "admit") in frames, frames


@pytest.mark.parametrize("row", sorted(EARLIER_ROWS))
def test_sheet_dispersion_refused_by_earlier_sheet_refusal(row, monkeypatch):
    _no_scan(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises((RfxRefusal, NotImplementedError)):
            EARLIER_ROWS[row]()


def test_sheet_without_pole_medium_steps():
    """The same model without the pole box is not refused and steps."""
    result = cube(False, poles=None).run(n_steps=4, skip_preflight=True)
    series = np.asarray(result.time_series)
    assert series.shape[0] == 4
    assert np.all(np.isfinite(series))
