"""#2449: one record cannot report isolated S for two driven guide ports.

WR-90, a=22.86 mm, b=10.16 mm, dx=a/9 (9 by 4 whole cells),
48 cells long, eps_r=2.2 from cell 24. The graded z profile has the
same physical height with four half cells between two full cells.
These are admission/shape witnesses, not converged scattering measurements.
"""

import jax
import numpy as np
import pytest

from rfx import Box, Simulation


MESSAGE = (
    "One run() or forward() record supports at most one driven waveguide port; driven ports: p1, p2. Set amplitude 0.0 on all but one, or use compute_waveguide_s_matrix() for the full matrix."
)
PATHS = ("uniform-run", "uniform-forward", "graded-run", "graded-forward")
DX = .02286 / 9
FREQS = np.array([7e9, 7.9e9])


def _guide(path, *, amplitude=1., two_ports=True, step=True):
    profile = {"dz_profile": np.array([DX, DX / 2, DX / 2, DX / 2, DX / 2, DX])}
    sim = Simulation(
        freq_max=14e9, domain=(48 * DX, 9 * DX, 4 * DX), dx=DX,
        boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=4,
        snap="declared", **(profile if path.startswith("graded") else {}),
    )
    if step:
        sim.add_material("fill", eps_r=2.2)
        sim.add(Box((24 * DX, 0, 0), (52 * DX, 9 * DX, 4 * DX)), material="fill")
    options = dict(mode=(1, 0), mode_type="TE", freqs=FREQS,
                   f0=float(FREQS.mean()), bandwidth=.6, ref_offset=1, probe_offset=2)
    sim.add_waveguide_port(8 * DX, direction="+x", name="p1", **options)
    if two_ports:
        sim.add_waveguide_port(40 * DX, direction="-x", name="p2",
                               amplitude=amplitude, **options)
    return sim


def _call(sim, path, **options):
    entry = path.split("-")[-1]
    if entry == "forward":
        options["checkpoint"] = False
    return getattr(sim, entry)(n_steps=2, skip_preflight=True, **options)


def _patch_loop(monkeypatch, path, replacement):
    # Both uniform entries pass rfx.simulation.run to prepared.advance;
    # both graded entries pass the runner's imported run_nonuniform. These
    # are the fixed-n_steps time-loop entries, after all source assembly.
    if path.startswith("graded"):
        monkeypatch.setattr("rfx.runners.nonuniform.run_nonuniform", replacement)
    else:
        monkeypatch.setattr("rfx.simulation.run", replacement)


def _forbidden(*args, **kwargs):
    pytest.fail("two waveguide ports reached the time loop")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("amplitude", [1.], ids=["driven"])
def test_two_ports_refused_before_loop(monkeypatch, path, amplitude):
    _patch_loop(monkeypatch, path, _forbidden)
    with pytest.raises(ValueError) as caught:
        _call(_guide(path, amplitude=amplitude), path)
    assert str(caught.value) == MESSAGE


@pytest.mark.parametrize("path", ["uniform-run", "graded-run"])
def test_compute_s_params_false_still_refuses(monkeypatch, path):
    _patch_loop(monkeypatch, path, _forbidden)
    with pytest.raises(ValueError) as caught:
        _call(_guide(path), path, compute_s_params=False)
    assert str(caught.value) == MESSAGE


@pytest.mark.parametrize("path", ["uniform-forward", "graded-run"])
def test_single_port_reaches_loop(monkeypatch, path):
    class ReachedLoop(Exception):
        pass

    def reached(*args, **kwargs):
        raise ReachedLoop

    _patch_loop(monkeypatch, path, reached)
    with pytest.raises(ReachedLoop):
        _call(_guide(path, two_ports=False), path)


def test_two_port_step_calculator_returns_matrix():
    result = _guide("uniform-run").compute_waveguide_s_matrix(
        n_steps=2, normalize=False, skip_preflight=True)
    assert np.shape(result.s_params) == (2, 2, 2)


def test_graded_two_port_calculator_returns_matrix():
    # The graded normalized calculator already requires equal port media;
    # use the same two-port geometry without the dielectric step for this row.
    result = _guide("graded-run", step=False).compute_waveguide_s_matrix(
        n_steps=2, normalize=True, skip_preflight=True)
    assert np.shape(result.s_params) == (2, 2, 2)


@pytest.mark.parametrize("path", PATHS)
def test_distributed_existing_refusal(monkeypatch, path):
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("two CPU devices are unavailable")
    monkeypatch.setattr(jax.lax, "scan", _forbidden)
    options = dict(devices=devices[:2])
    if path.endswith("forward"):
        options["distributed"] = True
    with pytest.raises(NotImplementedError, match=r"waveguide ports \(add_waveguide_port\) are not supported on"):
        _call(_guide(path), path, **options)


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_subgrid_existing_refusal(monkeypatch, entry):
    sim = _guide("uniform-run")
    sim.add_refinement(z_range=(DX, 3 * DX), ratio=2, validation="research")
    monkeypatch.setattr(jax.lax, "scan", _forbidden)
    with pytest.raises((ValueError, NotImplementedError), match="waveguide|Waveguide|subgridded|refinement"):
        _call(sim, "uniform-" + entry)
