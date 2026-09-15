"""S-parameter drivers must not commit derived mesh profiles, even on failure."""
import numpy as np
import pytest

import rfx.runners.nonuniform  # noqa: F401 - monkeypatch below targets this
                              # submodule by path, and a string target only
                              # resolves if it is already imported. Without
                              # this the test passes or fails on whatever a
                              # sibling test happened to import first: it
                              # was green locally and red on the VESSL image
                              # with 'module rfx.runners has no attribute
                              # nonuniform'.
from rfx import Box, Simulation


class _StopBeforeSolve(RuntimeError):
    pass


def _board():
    sim = Simulation(freq_max=10e9, domain=(0.012, 0.006, 0.004),
                     boundary="cpml", cpml_layers=2)
    sim.add(Box((0, 0, 0), (0.012, 0.006, 0.001)), material="fr4")
    return sim


@pytest.mark.parametrize("auto_mesh", [False, True])
def test_waveguide_driver_preserves_declared_mesh_on_solve_failure(monkeypatch, auto_mesh):
    if auto_mesh:
        sim = _board()
    else:
        sim = Simulation(freq_max=40e9, domain=(0.012, 0.006, 0.004),
                         dx=0.0005, dx_profile=np.full(24, 0.0005),
                         boundary="cpml", cpml_layers=2)
    sim.add_waveguide_port(0.003, direction="+x", n_freqs=2,
                          probe_offset=1, ref_offset=1)
    resolved = sim._resolve_mesh()
    assert sim._uses_nonuniform_mesh
    assert sim.__dict__["_dz_profile"] is None

    def stop(*args, **kwargs):
        assert sim.__dict__["_dz_profile"] is None
        assert sim._resolve_mesh() == resolved
        raise _StopBeforeSolve("runner reached")

    monkeypatch.setattr("rfx.runners.nonuniform.run_nonuniform_path", stop)
    with pytest.raises(_StopBeforeSolve, match="runner reached"):
        sim._compute_waveguide_s_matrix_nu(
            n_steps=1, num_periods=1, normalize=True)
    assert sim.__dict__["_dz_profile"] is None
    assert sim._resolve_mesh() == resolved


def test_msl_driver_preserves_auto_mesh_on_solve_failure(monkeypatch):
    sim = _board()
    sim.add(Box((0, 0.002, 0.001), (0.012, 0.004, 0.001)), material="pec")
    sim.add_msl_port(position=(0.003, 0.003, 0), width=0.002, height=0.001,
                     direction="+x", impedance=50, n_probe_offset=3,
                     n_probe_spacing=2, n_probes=3, eps_r_sub=4.4)
    resolved = sim._resolve_mesh()
    assert sim._uses_nonuniform_mesh
    assert sim.__dict__["_dz_profile"] is None

    def stop(*args, **kwargs):
        assert sim.__dict__["_dz_profile"] is None
        assert sim._resolve_mesh() is resolved
        raise _StopBeforeSolve("runner reached")

    monkeypatch.setattr(sim, "run", stop)
    with pytest.raises(_StopBeforeSolve, match="runner reached"):
        sim.compute_msl_s_matrix(n_steps=1, n_freqs=2)
    assert sim.__dict__["_dz_profile"] is None
    assert sim._resolve_mesh() is resolved
