"""A3 admission and D3 physical measurements share the run/forward pipeline."""

from collections import namedtuple
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.measurement.oblique import reconstruct_oblique_physical


def _scene(*, graded=True, slab=False, angle=0):
    profile = 1e-3 * np.array([1.2, 1.1, 1., .9, .8, .7, .6,
                              .6, .7, .8, .9, 1., 1.1, .6])
    mesh = {"dz_profile": profile} if graded else {}
    sim = Simulation(freq_max=20e9, domain=(.030, .020, .012), dx=.001,
                     cpml_layers=4, **mesh)
    sim.add_tfsf_source(angle_deg=angle)
    sim.add_probe((.023, .012, .007), "ez")
    if slab:
        sim.add_material("slab", eps_r=4.)
        sim.add(Box((.013, 0., 0.), (.017, .020, .012)), material="slab")
    return sim


@pytest.mark.parametrize("slab", [False, True], ids=["vacuum", "eps4-slab"])
def test_graded_plane_wave_matches_run(slab):
    actual = np.asarray(_scene(slab=slab).forward(n_steps=240).time_series)
    expected = np.asarray(_scene(slab=slab).run(n_steps=240).time_series)
    assert np.isfinite(actual).all() and np.max(np.abs(actual)) > 0
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("angle,direction,body", [
    (20, "+x", "TFSF oblique incidence is not yet supported on a nonuniform mesh. "
     "Use angle_deg=0 with a uniform x axis."),
    (0, "+z", "TFSF z-directed incidence is not yet supported on a nonuniform mesh. "
     "Axis-aligned incidence along x (direction='+x' or '-x') is supported only "
     "when x is uniform."),
], ids=["oblique", "z-incidence"])
def test_graded_plane_wave_refusals_match_run(angle, direction, body):
    expected_body = (
        " preflight found 1 blocking error(s) (pass skip_preflight=True to bypass):"
        "\n  - ERROR: " + body)
    for entry in ("run", "forward"):
        sim = _scene(angle=angle)
        # Public registration already refuses z; exercise the preflight guard
        # on a stored z-directed declaration without changing that API fence.
        sim._tfsf = replace(sim._tfsf, direction=direction)
        with pytest.raises(ValueError) as caught:
            getattr(sim, entry)(n_steps=2)
        prefix = f"[{entry}]"
        message = str(caught.value)
        assert message.startswith(prefix)
        assert message[len(prefix):] == expected_body


def test_distributed_graded_plane_wave_refused_by_name():
    from tests._multi_device_scene import host_devices
    with pytest.raises(NotImplementedError, match="TFSF.*graded multi-device"):
        _scene().forward(n_steps=2, distributed=True, devices=host_devices(2))


def test_graded_plane_wave_gradient_matches_fd():
    """Checks differentiability through the graded plane wave only.

    This is not a record-length witness. FD uses two forward runs at eps±1e-3.
    """
    sim = _scene()
    grid = sim._build_nonuniform_grid()
    lo = sim._pos_to_nu_index(grid, (.013, 0., 0.))[0]
    hi = sim._pos_to_nu_index(grid, (.017, 0., 0.))[0]

    def energy(eps_r):
        eps = jnp.ones(grid.shape, dtype=jnp.float32).at[lo:hi, :, :].set(eps_r)
        return jnp.sum(sim.forward(eps_override=eps, n_steps=240).time_series ** 2)

    ad = float(jax.grad(energy)(jnp.float32(4.)))
    fd = (float(energy(4.001)) - float(energy(3.999))) / .002
    relative = abs(ad - fd) / abs(fd) if fd else float("inf")
    print(f"graded AD={ad!r}, FD={fd!r}, relative={relative!r}")
    assert np.isfinite(ad) and ad != 0 and fd != 0
    assert relative <= 1e-2


def test_oblique_plane_wave_matches_run():
    # Same default Bloch method and 20-degree angle as census PW-oblique.
    actual = np.asarray(_scene(graded=False, angle=20).forward(n_steps=240).time_series)
    expected = np.asarray(_scene(graded=False, angle=20).run(n_steps=240).time_series)
    assert not np.iscomplexobj(actual)
    assert np.max(np.abs(actual)) > 0
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def test_oblique_reconstruction_hand_computation():
    # k_t=pi/2, dy=1, probe j=(0, 1): exp(-j*k_t*j*dy)=(1, -j).
    # Re((a+jb)*1)=a; Re((a+jb)*(-j))=b. No helper-derived oracle.
    result = namedtuple("ToyResult", "state time_series")(state=None, time_series=jnp.array(
        [[1+2j, 3+4j], [-2+5j, 6-7j]], dtype=jnp.complex64))
    cfg = SimpleNamespace(k_transverse=np.pi / 2, transverse_axis="y")
    grid = SimpleNamespace(shape=(1, 2, 1), cells=lambda axis: np.ones(2))
    probes = [SimpleNamespace(j=0), SimpleNamespace(j=1)]
    actual = reconstruct_oblique_physical(result, cfg, grid, probes)
    np.testing.assert_allclose(actual.time_series, [[1., 4.], [-2., -7.]],
                               rtol=0., atol=1e-6)
