"""Mixed waveguide drives refuse before fields; an isolated guide still runs."""

import jax
import numpy as np
import pytest

from rfx import Simulation


MESSAGE = "Waveguide ports are not supported together with lumped ports or TFSF"
PATHS = ("uniform-run", "uniform-forward", "graded-run", "graded-forward")


def _guide(path):
    sim = Simulation(
        freq_max=20e9, domain=(.030, .020, .012), dx=.002,
        boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=2,
        **({"dz_profile": np.full(6, .002)} if path.startswith("graded") else {}),
    )
    sim.add_waveguide_port(
        .006, direction="+x", f0=10e9, freqs=np.array([10e9, 11e9]),
        probe_offset=2, ref_offset=1, name="guide",
    )
    return sim


def _public_call(sim, path):
    entry = path.split("-")[1]
    options = {"checkpoint": False} if entry == "forward" else {}
    return getattr(sim, entry)(n_steps=1, skip_preflight=True, **options)


def _stored_tfsf_call(sim, path):
    # Public TFSF boundary checks can reject this declaration before a runner.
    # Exercise the runner's own admission with a stored, valid source entry.
    source = Simulation(freq_max=20e9, domain=(.030, .020, .012), dx=.002,
                        boundary="cpml", cpml_layers=2)
    source.add_tfsf_source(f0=10e9, margin=1)
    if path.startswith("graded"):
        from rfx.runners.nonuniform import run_nonuniform_path
        sim._tfsf = source._tfsf
        return run_nonuniform_path(
            sim, n_steps=1,
            lane="fwd_nonuniform" if path.endswith("forward") else "run_nonuniform",
        )
    grid = sim._build_grid()
    materials, debye, lorentz, pec_mask, _, _, _ = sim._assemble_materials(grid)
    sim._tfsf = source._tfsf
    if path.endswith("run"):
        from rfx.runners.uniform import run_uniform
        return run_uniform(sim, n_steps=1, grid=grid, base_materials=materials,
                           debye_spec=debye, lorentz_spec=lorentz, pec_mask=pec_mask)
    from rfx.runners.uniform_forward import _forward_from_materials
    return _forward_from_materials(sim, grid, materials, debye, lorentz,
                                   n_steps=1, pec_mask=pec_mask, lane="fwd_uniform")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("companion", ["lumped", "wire", "tfsf"])
def test_mixed_refused_before_fields(monkeypatch, path, companion):
    from rfx.stepping import graded_setup, uniform_setup

    def forbidden(*args, **kwargs):
        pytest.fail("mixed waveguide declaration reached field construction or stepping")

    monkeypatch.setattr(uniform_setup, "init_state", forbidden)
    monkeypatch.setattr(graded_setup, "init_state", forbidden)
    monkeypatch.setattr(jax.lax, "scan", forbidden)
    sim = _guide(path)
    if companion != "tfsf":
        sim.add_port((.016, .010, .006), "ez", impedance=50.,
                     **({"extent": .002} if companion == "wire" else {}))
    with pytest.raises(ValueError) as caught:
        if companion == "tfsf":
            _stored_tfsf_call(sim, path)
        else:
            _public_call(sim, path)
    assert str(caught.value) == MESSAGE


@pytest.mark.parametrize("path", PATHS)
def test_waveguide_alone_runs(path):
    result = _public_call(_guide(path), path)
    jax.block_until_ready(result.time_series)
    assert result.time_series.shape[0] == 1
    assert "guide" in result.waveguide_sparams
