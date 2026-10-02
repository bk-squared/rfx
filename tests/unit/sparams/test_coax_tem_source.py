"""The thru's drive must stay in the realized transverse TEM subspace."""

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import MU_0, curl_h, init_materials, init_state, update_h
from tests.unit.sparams.test_coax_two_port_smatrix import _sim


@pytest.mark.parametrize("geometry", ["centered", "off_center", "narrow_annulus"])
def test_thru_source_does_not_drive_longitudinal_fields(monkeypatch, geometry):
    """A cheaper TEM invariant than fitting a time-domain thru at three planes.

    In this homogeneous PEC guide, a TEM transverse profile produces neither
    Hz through Faraday's law nor Ez through Ampere's law. Check the actual
    E/H source tables and the actual shorted edges handed to the runner,
    using the runner's Yee curl operators. Sampling analytic 1/r on cell
    indices fails this invariant at the staircased walls (#1356).
    """
    class SourceChecked(Exception):
        pass

    def inspect_source(grid, materials, n_steps, **kw):
        shape = (grid.nx, grid.ny, 1)
        fields = {c: np.zeros(shape, dtype=np.float32)
                  for c in ("ex", "ey", "hx", "hy")}
        for source in kw["sources"] + kw["mag_sources"]:
            fields[source.component][source.i, source.j, 0] += float(source.waveform[0])
        k = kw["sources"][0].k
        masks = [np.asarray(m)[:, :, k:k + 1] for m in kw["pec_edge_masks"]]
        for comp, mask in zip(("ex", "ey"), masks):
            # The runner clamps these edges after injecting the source.
            fields[comp][mask] = 0.0
        for pair in (("ex", "ey"), ("hx", "hy")):
            scale = max(float(np.max(np.abs(fields[c]))) for c in pair)
            assert scale > 0.0
            for c in pair:
                fields[c] = jnp.asarray(fields[c] / scale)

        # dx=1, dt=mu0 remove dimensions; only transverse cancellation is
        # tested. The z profile cannot contribute to either longitudinal curl.
        state = init_state(shape)._replace(ex=fields["ex"], ey=fields["ey"])
        h = update_h(state, init_materials(shape), dt=MU_0, dx=1.0)
        _, _, ez_curl = curl_h(fields["hx"], fields["hy"], state.hz,
                               dx=1.0, periodic=(False, False, False))
        hz_error = float(jnp.max(jnp.abs(h.hz)))
        ez_error = float(np.max(np.abs(np.asarray(ez_curl)[~masks[2]])))
        assert hz_error < 2e-6, f"TEM source creates longitudinal H: {hz_error}"
        assert ez_error < 2e-6, f"TEM source creates longitudinal E: {ez_error}"
        raise SourceChecked

    sim = _sim()
    port = sim._coaxial_ports[0]
    dx = float(sim._build_grid().dx)
    if geometry == "off_center":
        # Move the pin/shell centre by a fraction of a cell in each direction.
        sim._coaxial_ports[0] = port._replace(
            position=(port.position[0] + 0.3 * dx,
                      port.position[1] - 0.2 * dx, port.position[2]),
        )
    elif geometry == "narrow_annulus":
        # A covered but coarsely resolved annulus, 2.5 cells across radially.
        sim._coaxial_ports[0] = port._replace(outer_radius=port.pin_radius + 2.5 * dx)

    monkeypatch.setattr("rfx.simulation.run", inspect_source)
    with pytest.raises(SourceChecked):
        sim.compute_coaxial_two_port(n_steps=1, freqs=np.array([8e9]))
