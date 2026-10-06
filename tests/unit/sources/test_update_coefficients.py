"""First E update at H=0, unit drive, against the kernel algebra (#1524)."""

import pytest
from unittest.mock import patch
import numpy as np
import jax.numpy as jnp
from rfx import Simulation, Box, DebyePole
from rfx.materials.lorentz import LorentzPole
from rfx.boundaries.spec import BoundarySpec
import rfx.simulation as low

EPS = 8.8541878128e-12


@pytest.mark.parametrize(
    "case", ("debye", "lorentz", "seam", "later_before", "later_after")
)
def test_source_first_e_update(case):
    boundary = (
        BoundarySpec(x="periodic", y="pec", z="pec")
        if case == "seam"
        else ("upml" if case.startswith("upml") else "cpml")
    )
    s = Simulation(
        freq_max=8e9,
        domain=(0.008,) * 3,
        dx=0.001,
        cpml_layers=2 if case.startswith("upml") else 0,
        boundary=boundary,
    )
    extra = (
        {"debye_poles": [DebyePole(2.0, 1 / (2 * np.pi * 9e9))]}
        if case == "debye"
        else {
            "lorentz_poles": [
                LorentzPole(2 * np.pi * 9e9, 2e9, 2 * (2 * np.pi * 9e9) ** 2)
            ]
        }
        if case == "lorentz"
        else {}
    )
    s.add_material("slab", eps_r=2.0 if extra else 4.0, **extra)
    s.add(
        Box(
            (0.0043 if case in ("smooth", "inverse") else 0.004, 0, 0),
            (0.008, 0.008, 0.008),
        ),
        material="slab",
    )
    pos = (
        0.0 if case == "seam" else -0.001 if case == "upml_pad" else 0.004,
        0.004,
        0.004,
    )
    soft = case in ("seam", "later_before", "later_after")

    def _declare_witness_source():
        if soft:
            s.add_source(pos, "ez", waveform=jnp.ones_like, amplitude_kind="current")
        else:
            s.add_port(pos, "ez", waveform=jnp.ones_like)

    if case == "later_after":
        s.add_port(pos, "ez", excite=False)
    _declare_witness_source()
    if case == "later_before":
        s.add_port(pos, "ez", excite=False)
    s.add_probe(pos, "ez")
    captured = {}
    original = low.run

    def _capture_coefficient_witness(*a, **kw):
        captured.update(kw)
        captured["args"] = a
        return original(*a, **kw)

    with patch.object(low, "run", _capture_coefficient_witness):
        r = s.run(
            n_steps=2,
            skip_preflight=True,
            compute_s_params=False,
            subpixel_smoothing={"smooth": True, "inverse": "kottke_pec"}.get(
                case, False
            ),
        )
    g = s._build_grid()
    idx = g.position_to_index(pos)
    dt = g.dt
    mats = captured.get(
        "materials", captured["args"][1] if len(captured["args"]) > 1 else None
    )
    c = mats.components
    eps = float(c.eps_update[2][idx])
    sig = float(c.sigma_update[2][idx])
    beta = 0
    if case == "debye":
        beta = float(jnp.sum(c.debye.coefficients[1][2], axis=0)[idx])
    cb = dt / (EPS * eps + beta + sig * dt / 2)
    if case == "smooth":
        cb = dt / (EPS * float(captured["aniso_eps"][2][idx]) + sig * dt / 2)
    if case == "inverse":
        inv = float(captured["aniso_inv_eps"][2][idx])
        cb = dt * inv / EPS / (1 + sig * dt * inv / (2 * EPS))
    if case.startswith("upml"):
        from rfx.boundaries.upml import _axis_sigma_E_H

        sx = _axis_sigma_E_H(g, "x")[0]
        sy = _axis_sigma_E_H(g, "y")[0]
        ep = float(c.upml_eps[2][idx])
        sg = float(c.upml_sigma[2][idx])
        sp = float((sx + sy)[idx])
        cb = (dt / (ep * EPS)) / (1 + sp * dt / (2 * EPS) + sg * dt / (2 * ep * EPS))
    j = 1 / 0.001**3 if soft else (1 / (50 * 0.001)) / 0.001
    measured = float(np.asarray(r.time_series).reshape(-1)[0])
    ref = cb * j
    np.testing.assert_allclose(measured, ref, rtol=2e-7)


@pytest.mark.parametrize("kind", ("plain", "debye", "lorentz", "mixed"))
@pytest.mark.parametrize("later_first", (False, True))
def test_graded_source_final_stamp(kind, later_first):
    """Both declaration orders see the same homogeneous ADE edge and load."""
    from rfx.runners import nonuniform as runner
    from rfx.nonuniform import current_source_volume

    s = Simulation(
        freq_max=8e9,
        domain=(0.008,) * 3,
        dx=0.001,
        dx_profile=np.array([0.001] * 8),
        cpml_layers=0,
        boundary="pec",
    )
    extra = {}
    if kind in ("debye", "mixed"):
        extra["debye_poles"] = [DebyePole(2.0, 1 / (2 * np.pi * 9e9))]
    if kind in ("lorentz", "mixed"):
        extra["lorentz_poles"] = [
            LorentzPole(2 * np.pi * 9e9, 2e9, 2 * (2 * np.pi * 9e9) ** 2)
        ]
    s.add_material("bulk", eps_r=2.0, **extra)
    s.add(Box((0, 0, 0), (0.008,) * 3), material="bulk")
    pos = (0.004,) * 3
    if later_first:
        s.add_port(pos, "ez", excite=False)
    s.add_source(pos, "ez", waveform=jnp.ones_like, amplitude_kind="current")
    if not later_first:
        s.add_port(pos, "ez", excite=False)
    s.add_probe(pos, "ez")
    captured = {}
    original = runner.run_nonuniform

    def _capture_coefficient_witness(*args, **kwargs):
        captured.update(kwargs)
        captured["args"] = args
        return original(*args, **kwargs)

    with patch.object(runner, "run_nonuniform", _capture_coefficient_witness):
        result = s.run(n_steps=2, skip_preflight=True, compute_s_params=False)
    grid = s._build_nonuniform_grid()
    idx = (4, 4, 4)
    dt = grid.dt
    # Homogeneous eps_inf=2; port stamps sigma=1/(50*dx)=20 S/m.
    beta = (
        EPS * 2 * dt / (2 / (2 * np.pi * 9e9) + dt) if kind in ("debye", "mixed") else 0
    )
    cb = dt / (EPS * 2 + beta + 20 * dt / 2)
    volume = current_source_volume(grid, idx, "ez")[0]
    np.testing.assert_allclose(
        np.asarray(result.time_series).reshape(-1)[0], cb / volume, rtol=3e-7
    )


@pytest.mark.parametrize("kind", ("plain", "debye", "lorentz", "mixed"))
def test_msl_jm_each_component(kind):
    """With H=0, each unit modal J correction is Cb(component)/dx."""
    from types import SimpleNamespace
    from rfx.model.materials import with_components
    from rfx.sources.msl_port import make_msl_port_sources_jm

    s = Simulation(
        freq_max=8e9, domain=(0.008,) * 3, dx=0.001, cpml_layers=0, boundary="pec"
    )
    extra = {}
    if kind in ("debye", "mixed"):
        extra["debye_poles"] = [DebyePole(2.0, 1 / (2 * np.pi * 9e9))]
    if kind in ("lorentz", "mixed"):
        extra["lorentz_poles"] = [
            LorentzPole(2 * np.pi * 9e9, 2e9, 2 * (2 * np.pi * 9e9) ** 2)
        ]
    s.add_material("slab", eps_r=2.0, **extra)
    s.add(Box((0, 0.004, 0), (0.008,) * 3), material="slab")
    grid = s._build_grid()
    materials, ds, ls, *_ = s._assemble_materials(grid, pec_sheets=[], pec_wires=[])
    materials = with_components(
        materials, grid, periodic=(False,) * 3, debye_spec=ds, lorentz_spec=ls
    )
    em = SimpleNamespace(
        j_grid_lo=4,
        k_grid_lo=4,
        n_y_grid=1,
        n_z_grid=1,
        cell_indices=[(4, 4, 4)],
        ey=np.zeros((1, 1)),
        ez=np.zeros((1, 1)),
        hy=np.ones((1, 1)),
        hz=np.ones((1, 1)),
    )
    electric, _ = make_msl_port_sources_jm(
        grid,
        SimpleNamespace(excitation=jnp.ones_like, direction="+x"),
        materials,
        2,
        em,
    )
    for spec, eps, fraction, sign in zip(electric, (1.5, 2.0), (0.5, 1.0), (1, -1)):
        beta = (
            EPS * 2 * grid.dt / (2 / (2 * np.pi * 9e9) + grid.dt) * fraction
            if ds
            else 0
        )
        reference = sign * grid.dt / (EPS * eps + beta) / grid.dx
        np.testing.assert_allclose(spec.waveform[0], reference, rtol=3e-7)


@pytest.mark.parametrize('kind', ['debye', 'lorentz'])
@pytest.mark.parametrize('passive_first', [False, True])
def test_ade_port_reads_later_port_stamp(kind, passive_first):
    sim = Simulation(freq_max=8e9, domain=(.008,) * 3, dx=.001,
                     cpml_layers=0, boundary='pec')
    poles = ({'debye_poles': [DebyePole(2., 1 / (2 * np.pi * 9e9))]}
             if kind == 'debye' else
             {'lorentz_poles': [LorentzPole(2 * np.pi * 9e9, 2e9, 2 * (2 * np.pi * 9e9)**2)]})
    sim.add_material('slab', eps_r=2., **poles)
    sim.add(Box((.004, 0, 0), (.008,) * 3), material='slab')
    pos = (.004,) * 3
    if passive_first:
        sim.add_port(pos, 'ez', impedance=75., excite=False)
    sim.add_port(pos, 'ez', waveform=jnp.ones_like)
    if not passive_first:
        sim.add_port(pos, 'ez', impedance=75., excite=False)
    sim.add_probe(pos, 'ez')
    result = sim.run(n_steps=2, skip_preflight=True, compute_s_params=False)
    dt = sim._build_grid().dt
    beta = EPS * dt / (2 / (2 * np.pi * 9e9) + dt) if kind == 'debye' else 0
    reference = dt / (1.5 * EPS + beta + (20 + 1 / .075) * dt / 2) * 20000
    np.testing.assert_allclose(np.asarray(result.time_series).reshape(-1)[0],
                               reference, rtol=3e-7)
