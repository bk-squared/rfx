"""The experimental metal smoothing advisory is shared by public solve entries."""

import warnings

import jax
import jax.numpy as jnp
import pytest

from rfx import Box, PolylineWire, Simulation
from rfx.errors import ConfigurationWarning


MESSAGE = (
    "subpixel_smoothing='kottke_pec' with a PEC volume is "
    "experimental, not recommended for metal; see known limitations, "
    '"A metal face resting on a dielectric between grid nodes"'
)


@pytest.mark.parametrize('smoothing, body, transform, entry', [
    (smoothing, body, transform, entry)
    for entry in ['run', 'forward']
    for smoothing,body,transform in [('kottke_pec', 'volume', 'eager'), (True, 'volume', 'eager'), (False, 'volume', 'eager'), ('kottke_pec', 'dielectric', 'eager'), ('kottke_pec', 'sheet', 'eager'), ('kottke_pec', 'box_sheet', 'eager'), ('kottke_pec', 'wire', 'eager'), ('kottke_pec', 'volume', 'jit'), ('kottke_pec', 'volume', 'grad')]
    if entry != 'run' or (smoothing, body, transform) == ('kottke_pec', 'volume', 'eager')
])
def test_kottke_pec_volume_warning(entry, smoothing, body, transform):
    """#2390: off-grid PEC/dielectric faces bias a closed-box resonance.

    A resolved PEC Box inside a tiny uniform PEC cavity anchors the warning
    to a metal volume, independently of the mesh-dependent resonance error.
    The cavity walls, a PEC sheet, and a filament alone are not PEC volumes.
    """
    sim = Simulation(freq_max=20e9, domain=(.008,) * 3, dx=.001,
                     boundary="pec", cpml_layers=0)
    if body == "sheet":
        sim.add_thin_conductor(Box(((4 - .35) * .001, (3 - .35) * .001, .004),
                                   ((6 + .35) * .001, (5 + .35) * .001, .004)))
    elif body == "box_sheet":
        sim.add(Box(((4 - .35) * .001, (3 - .35) * .001, .004),
                                   ((6 + .35) * .001, (5 + .35) * .001, .004)), material="pec")
    elif body == "wire":
        sim.add(PolylineWire(points=[(.004, .004, .003), (.004, .004, .005)],
                             radius=0), material="pec")
    else:
        sim.add_material("dielectric", eps_r=4.4)
        sim.add(Box((.004, .003, .003), (.006, .005, .005)),
                material="pec" if body == "volume" else "dielectric")
    sim.add_source((.002, .002, .002), "ez", amplitude_kind="field")
    sim.add_probe((.002, .003, .002), "ez")

    def solve(scale):
        result = getattr(sim, entry)(n_steps=2, skip_preflight=True,
                                     subpixel_smoothing=smoothing)
        return scale * jnp.sum(result.time_series)

    call = {"eager": lambda f: f, "jit": jax.jit, "grad": jax.grad}[transform](solve)
    expected = smoothing == "kottke_pec" and body == "volume"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jax.block_until_ready(call(1.0))
    notices = [w for w in caught if "not recommended for metal" in str(w.message)]
    assert len(notices) == int(expected), "first call must warn exactly for kottke_pec + PEC volume"
    if expected:
        assert str(notices[0].message) == MESSAGE
        assert notices[0].category is ConfigurationWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jax.block_until_ready(call(1.0))
        # The once-per-object state is shared across the public entries too.
        other = "forward" if entry == "run" else "run"
        getattr(sim, other)(n_steps=2, skip_preflight=True, subpixel_smoothing=smoothing)
    assert not [w for w in caught if "not recommended for metal" in str(w.message)]


@pytest.mark.parametrize("smoothing", ["kottke_pec", True])
def test_waveguide_kottke_pec_volume_warning(monkeypatch, smoothing):
    """A whole-cell PEC post warns at admission, before any FDTD stepping."""
    from rfx.boundaries.spec import BoundarySpec
    from rfx.sparams import _waveguide_stages

    sim = Simulation(freq_max=10e9, domain=(.012, .024, .010), dx=.002,
                     boundary=BoundarySpec(x="cpml", y="pec", z="pec"),
                     cpml_layers=2)
    sim.add(Box((.006, .010, 0), (.008, .012, .010)), material="pec")
    for position, direction in [(.002, "+x"), (.010, "-x")]:
        sim.add_waveguide_port(position, direction=direction, mode=(1, 0),
                               mode_type="TE", freqs=jnp.array([10e9]),
                               f0=10e9, bandwidth=.5)

    class AdmissionComplete(Exception):
        pass

    validate_ports = _waveguide_stages._validate_waveguide_ports

    def stop_after_admission(*args, **kwargs):
        validate_ports(*args, **kwargs)
        raise AdmissionComplete

    monkeypatch.setattr(_waveguide_stages, "_validate_waveguide_ports",
                        stop_after_admission)
    for call_index in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(AdmissionComplete):
                sim.compute_waveguide_s_matrix(n_steps=1, normalize=True,
                                               subpixel_smoothing=smoothing)
        notices = [w for w in caught if "not recommended for metal" in str(w.message)]
        expected = smoothing == "kottke_pec" and call_index == 0
        assert len(notices) == int(expected)
        if expected:
            assert str(notices[0].message) == MESSAGE
            assert notices[0].category is ConfigurationWarning
