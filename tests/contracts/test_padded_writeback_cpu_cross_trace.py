"""Padded CPU write-back keeps the probe spectra within 1e-4 of the plain peak.

Both forms use identical port coefficient bits, within 3.5 float32 ULP of the
float64 coefficient for the loaded edge.
"""

import jax
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries import _cpml_slab
from rfx.core.yee import EPS_0
from rfx.sources import port_drive, sources
from tests.contracts.test_soft_source_one_drive import relative_probe_spectrum

MM = 1e-3


def port_material_samples(materials, cell):
    """Gather the raw operands at the four cells incident to the interior Ez edge."""
    i, j, k = cell
    cells = np.s_[i-1:i+1, j-1:j+1, k]

    def operands(values, stamps):
        parts = stamps or (None,) * 3
        bulk_stamps = tuple(part[cells] for part in parts if part is not None)
        own = np.float32(0) if parts[2] is None else parts[2][cell]
        return values[cells], bulk_stamps, own

    film = materials.sigma_film
    film_sigma = np.float32(0) if film is None or film[2] is None else film[2][cell]
    return (operands(materials.eps_r, materials.eps_r_lumped),
            operands(materials.sigma, materials.sigma_lumped), film_sigma)


def loaded_edge_mean(values, stamps, own):
    """Remove all cell stamps before averaging and restore the edge's own stamp in full."""
    bulk = np.asarray(values, np.float64).reshape(4).copy()
    for stamp in stamps:
        bulk -= np.asarray(stamp, np.float64).reshape(4)
    return bulk.mean() + np.float64(own)


@pytest.mark.parametrize("n_steps", (120, 240))
def test_padded_writeback_cpu_plain_vs_jit(monkeypatch, n_steps, record_property):
    """The probe DFT admits 1e-4 relative error and the port Cb bits agree.

    In _e_update_coeffs_si the two rounded inputs (vacuum permittivity and dt)
    and five operations each contribute half a float32 ULP. Their 3.5 ULP
    allowance has no amplification from subtraction.
    """
    monkeypatch.setattr(_cpml_slab, "writeback_form", lambda: "padded")

    def record():
        sim = Simulation(freq_max=16e9, domain=(16 * MM, 12 * MM, 8 * MM),
                         dx=MM, boundary="cpml", cpml_layers=4)
        sim.add_material("sub", eps_r=3.0)
        sim.add(Box((2 * MM, 2 * MM, 1 * MM), (14 * MM, 10 * MM, 3 * MM)),
                material="sub")
        sim.add(Box((2 * MM, 2 * MM, 0), (14 * MM, 10 * MM, 1 * MM)),
                material="pec")
        sim.add_port(position=(6 * MM, 6 * MM, 2 * MM), component="ez",
                     impedance=50.0, waveform=GaussianPulse(f0=8e9, bandwidth=0.8))
        sim.add_probe((10 * MM, 6 * MM, 2 * MM), "ez")
        return sim.forward(
            n_steps=n_steps, skip_preflight=True, checkpoint=False,
            port_s11_freqs=np.array([6e9, 8e9], np.float32)).time_series

    port_edges = set()
    observations = []
    setup_port = sources.setup_lumped_port
    original = port_drive.e_update_coefficient_at

    def capture_port(grid, port, materials):
        stamped = setup_port(grid, port, materials)
        cell, = port._drive_stamps
        port_edges.add((tuple(map(int, cell)), port.component, float(grid.dt)))
        return stamped

    def observe(materials, cell, component, dt, **kwargs):
        cb = original(materials, cell, component, dt, **kwargs)
        cell = tuple(map(int, cell))
        if (cell, component, float(kwargs['grid'].dt)) in port_edges:
            assert component == 'ez'
            samples = port_material_samples(materials, cell)

            def save(value, operands):
                observations.append((np.array(value), jax.tree.map(np.array, operands)))

            jax.debug.callback(save, cb, samples)
        return cb

    monkeypatch.setattr(sources, 'setup_lumped_port', capture_port)
    monkeypatch.setattr(port_drive, 'e_update_coefficient_at', observe)
    records, bits = {}, {}
    try:
        with jax.default_device(jax.devices("cpu")[0]):
            for form, forward in (('plain', record), ('jit', jax.jit(record))):
                jax.clear_caches()
                port_edges.clear()
                observations.clear()
                records[form] = np.asarray(forward(), dtype=np.float64)
                jax.effects_barrier()
                (_, _, dt), = port_edges
                record_property(f'{form}_calls', len(observations))
                assert observations, f'{form} observed no coefficient at the port edge'
                bits[form] = []
                for index, (cb, (eps, sigma, film)) in enumerate(observations):
                    eps_e = loaded_edge_mean(*eps)
                    sigma_e = loaded_edge_mean(*sigma) + np.float64(film)
                    c64 = (dt / EPS_0) / (eps_e + sigma_e * dt / (2 * EPS_0))
                    assert cb.shape == () and cb.dtype == np.float32
                    bits[form].append(int(cb.view(np.uint32)))
                    ulps = abs(np.float64(cb) - c64) / np.spacing(np.float32(c64))
                    record_property(f'{form}_cb_bits_{index}', bits[form][-1])
                    record_property(f'{form}_cb_ulps_{index}', float(ulps))
                    assert ulps <= 3.5, (form, index, ulps)

        assert set(bits['plain']) == set(bits['jit']) == {bits['plain'][0]}, bits
        plain, compiled = records['plain'], records['jit']
        assert plain.shape == compiled.shape == (n_steps, 1)
        spectral = relative_probe_spectrum(plain, compiled, dt)
        record_property('spectral', spectral)
        assert spectral <= 1e-4, (n_steps, spectral)
    finally:
        jax.clear_caches()
