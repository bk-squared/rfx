"""#2217: only conductive E components retain E_prev on the box tape."""

import re

import jax
from jax.ad_checkpoint import print_saved_residuals
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.core.yee import MaterialArrays
import rfx.model.materials as material_model


BOX = ((.010, .008, .006), (.014, .012, .010))
STEPS = 80


def _scene(kind):
    sim = Simulation(freq_max=16e9, domain=(.024, .020, .016), dx=.002,
                     boundary="cpml", cpml_layers=5)
    sim.add_source((.004, .010, .008), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=8e9, bandwidth=.8))
    sim.add_probe((.020, .010, .008), "ez")
    if kind == "lossy":
        sim.add_material("lossy", eps_r=1., sigma=.01)
        sim.add(Box((0., 0., 0.), (.024, .020, .016)), material="lossy")
    if kind.startswith("held"):
        sim.add_port((.012, .010, .008), "ez", impedance=50., excite=False)
    return sim


def _loss(sim, kind, *, checkpoint=False):
    # These entries must stay concrete when only epsilon is differentiated.
    edge_zeros = tuple(jnp.zeros((3, 3, 3), jnp.float32) for _ in range(3))

    def loss(eps):
        options = {}
        if kind == "designed-sigma":
            options["design_sigma_override"] = eps * .005
        elif kind == "held-edge-tuple":
            # Replaces the load in sig_c; the held background restores it.
            options["design_sigma_override"] = edge_zeros
        result = sim.forward(
            design_box=BOX, design_eps_override=eps,
            design_box_holds_ports=kind.startswith("held"),
            n_steps=STEPS, checkpoint=checkpoint, skip_preflight=True,
            **options)
        return jnp.sum(result.time_series ** 2)

    return loss


@pytest.mark.parametrize("kind,arrays", [
    ("lossless", 3), ("lossy", 6), ("designed-sigma", 6),
    ("held", 4), ("held-edge-tuple", 4),
])
def test_saved_window_residuals(kind, arrays, capsys):
    sim = _scene(kind)
    eps = jnp.full((3, 3, 3), 2., jnp.float32)
    # print_saved_residuals linearizes the objective as reverse-mode grad
    # does. jit also pins the concrete-background proof under an outer trace.
    print_saved_residuals(jax.jit(_loss(sim, kind)), eps)
    saved = capsys.readouterr().out
    # Three declared cells write four nodes on each axis; the minus-side
    # computation context is not saved. Counts come from the design table.
    windows = re.findall(r"f32\[80,4,4,4\]", saved)
    assert len(windows) == arrays, saved


@pytest.mark.parametrize("kind", ["lossless", "lossy", "held-edge-tuple"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_value_and_gradient_bits_against_original_builder(kind, checkpoint, monkeypatch):
    sim = _scene(kind)
    loss = _loss(sim, kind, checkpoint=checkpoint)
    eps = jnp.linspace(1.5, 3., 27, dtype=jnp.float32).reshape((3, 3, 3))
    actual = jax.jit(jax.value_and_grad(lambda e: loss(e)))(eps)
    # All-False restores main's coefficient graph: no stop_gradient on Ca.
    # This also catches unconditional stopping in the lossy mutation (m2).
    monkeypatch.setattr(material_model, "_design_box_constant_ca",
                        lambda *args, **kwargs: (False, False, False))
    reference = jax.jit(jax.value_and_grad(lambda e: loss(e)))(eps)
    assert np.any(np.asarray(reference[1]) != 0), "fixture gradient vanished"
    for got, expected in zip(actual, reference):
        assert np.all(np.isfinite(np.asarray(got)))
        assert np.asarray(got).tobytes() == np.asarray(expected).tobytes()


@pytest.mark.parametrize("design,step_bytes", [
    ("permittivity", 1536),  # 4 B * 4^3 write cells * 6, designed sigma.
    ("occupancy", 768),     # 4 B * 4^3 write cells * 3.
])
@pytest.mark.parametrize("checkpoint,segments", [(False, None), (False, 2), (True, 2)])
def test_estimate_write_window(design, step_bytes, checkpoint, segments):
    estimate = _scene("lossless").estimate_ad_memory(
        STEPS, design_box=BOX, design=design, checkpoint=checkpoint,
        checkpoint_segments=segments)
    assert estimate.tape_window_cells == 64
    assert estimate.tape_step_bytes == step_bytes


def test_unsegmented_checkpoint_bound_is_unchanged():
    estimate = _scene("lossless").estimate_ad_memory(
        STEPS, design_box=BOX, checkpoint=True)
    assert estimate.tape_is_bound
    # Amendment 1 leaves the old six-array computation-window bound alone.
    assert estimate.tape_window_cells == 125
    assert estimate.tape_step_bytes == estimate.carry_bytes + 3000


@pytest.mark.parametrize("carrier", ["background", "edge", "film", "lumped"])
def test_zero_traced_conductivity_keeps_its_ca_derivative(carrier):
    """A zero primal is insufficient when conductivity itself is designed."""
    shape = (7, 7, 7)
    zero = jnp.zeros(shape, jnp.float32)
    mats = MaterialArrays(jnp.ones(shape), zero, jnp.ones(shape))
    eps = jnp.full((3, 3, 3), 2., jnp.float32)
    bounds = (2, 5, 2, 5, 2, 5)

    def ca_sum(sigma):
        background, edge = mats, None
        if carrier == "background":
            background = mats._replace(sigma=sigma)
        elif carrier == "edge":
            edge = (eps * 0., eps * 0., sigma[2:5, 2:5, 2:5])
        elif carrier == "film":
            background = mats._replace(sigma_film=(None, None, sigma))
        else:
            background = mats._replace(sigma=sigma, sigma_lumped=(None, None, sigma))
        _, ca, _ = material_model._design_box_edge_coeffs(
            bounds, eps, edge, background, 1e-12, shape)
        return jnp.sum(ca[2])

    gradient = np.asarray(jax.jit(jax.grad(ca_sum))(zero))
    assert np.all(np.isfinite(gradient))
    assert np.any(gradient != 0)
