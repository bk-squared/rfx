"""Passive half-space crossing every CPML pad: eager extraction stays bounded."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import init_materials
from rfx.grid import Grid
from rfx.probes.probes import SParamProbe, extract_s_matrix, extract_s_matrix_wire
from rfx.sources.sources import GaussianPulse, LumpedPort, WirePort


def _assert_passive(s11):
    assert np.max(np.abs(s11)) <= 1 + 1e-3, np.abs(s11)


@jax.tree_util.register_pytree_node_class
class _CompiledProbe(SParamProbe):
    """Keep the eager probe's string/configuration fields static under JIT."""

    def tree_flatten(self):
        return ((self.v_dft, self.i_dft, self.v_inc_dft, self.freqs,
                 self.v_port_dft, self.v_ref_dft),
                (self.port_index, self.component, self.total_steps,
                 self.window, self.window_alpha))

    @classmethod
    def tree_unflatten(cls, metadata, arrays):
        return cls(*arrays[:4], *metadata, *arrays[4:])


def test_eager_extractor_dielectric_through_pad(monkeypatch):
    # Keep the public extractor's Python loop, drive/probe plumbing and
    # material preparation. Compile its unchanged step kernel
    # so this 400-step regression remains suitable for the default suite.
    from rfx.stepping import probe_loop
    make_step = probe_loop.make_probe_step
    def compiled_step(*args, **kwargs):
        step, source = make_step(*args, **kwargs)
        compiled = jax.jit(step)
        def call(carry, *values):
            carry = {**carry, 'sprobes': tuple(_CompiledProbe(*probe)
                                             for probe in carry['sprobes'])}
            return compiled(carry, *values)
        # Keep the supplied waveform's Python-float evaluation unchanged.
        return call, source
    monkeypatch.setattr(probe_loop, 'make_probe_step', compiled_step)

    # A deliberate gain must be rejected by the SAME assertion as the scene.
    with pytest.raises(AssertionError):
        _assert_passive(np.full(7, 1.01))

    waveform = GaussianPulse(f0=5.5e9, bandwidth=4e9)
    freqs = np.linspace(2e9, 9e9, 7)
    grid = Grid(freq_max=10e9, domain=(.024, .012, .012), dx=1e-3, cpml_layers=8)
    eps_r = np.ones(grid.shape, np.float32)
    eps_r[:, :, :grid.shape[2] // 2] = 4.
    materials = init_materials(grid.shape)._replace(eps_r=jnp.asarray(eps_r))
    reflections = []
    for wire in (False, True):
        ports = [
            WirePort(start=(x, .006, .006), end=(x, .006, .007),
                     component='ez', impedance=50., excitation=waveform)
            if wire else
            LumpedPort(position=(x, .006, .006), component='ez',
                       impedance=50., excitation=waveform)
            for x in (.012, .02)
        ]
        extractor = extract_s_matrix_wire if wire else extract_s_matrix
        s = np.asarray(extractor(grid, materials, ports, freqs, 400,
                                 boundary='cpml', cpml_axes='xyz'))
        reflections.append(s[0, 0])
    _assert_passive(np.asarray(reflections))
