"""The two tangential SAT pairs consume distinct magnetic face materials."""
import jax.numpy as jnp
import numpy as np
import pytest
from rfx.core.yee import MU_0
from rfx.core.yee import init_materials
from rfx.subgridding import jit_runner as runner
from rfx.subgridding.sbp_sat_3d import SubgridConfig3D


@pytest.mark.parametrize('half', ['h', 'e'])
@pytest.mark.parametrize('projection', ['dual', 'coarse', 'fine', 'average'])
def test_sat_pairs_read_their_own_mu(monkeypatch, half, projection):
    config = SubgridConfig3D(5, 5, 7, .002, 0, 5, 0, 5, 2, 5,
                             9, 9, 5, .001, 1e-12, 2, 1.)
    def mats(shape):
        i, j, _ = np.indices(shape)
        return init_materials(shape)._replace(mu_r=jnp.asarray(1+i+3*j, dtype=jnp.float32))
    coarse, fine = mats((5,5,7)), mats((9,9,5))
    fields_c = (jnp.ones(coarse.mu_r.shape),) * 6
    fields_f = (jnp.ones(fine.mu_r.shape),) * 6
    calls = []
    original = runner.interface_pair_deltas
    def capture(*args, **kwargs):
        calls.append({key: np.asarray(kwargs[key]) / MU_0
                      for key in ('mu_lower', 'mu_upper')})
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, 'interface_pair_deltas', capture)
    opts = runner.SubgridRunOptions(material_sat_face_projection='sample',
        material_sat_zlo_common_trace_projection=projection,
        material_sat_zhi_common_trace_projection=projection)
    getattr(runner, f'_z_slab_material_coupling_{half}_3d')(
        fields_c, fields_f, coarse, fine, config, opts=opts)
    # z-low, coarse side: Hy crosses y (cells 6 and 9), Hx crosses x
    # (cells 8 and 9). Test the arguments reaching the physical SAT solver.
    assert calls[0]['mu_lower'][2,2] == pytest.approx(2/(1/6+1/9), rel=2e-7)
    assert calls[1]['mu_lower'][2,2] == pytest.approx(2/(1/8+1/9), rel=2e-7)
    # Restricted fine side samples fine index (4,4): cells 14/17 vs 16/17.
    assert calls[0]['mu_upper'][2,2] == pytest.approx(2/(1/14+1/17), rel=2e-7)
    assert calls[1]['mu_upper'][2,2] == pytest.approx(2/(1/16+1/17), rel=2e-7)
    # Every pair, including the opposite face and fine-side solves, differs.
    for a, b in zip(calls[::2], calls[1::2]):
        assert not np.array_equal(a['mu_lower'], b['mu_lower'])
        assert not np.array_equal(a['mu_upper'], b['mu_upper'])
