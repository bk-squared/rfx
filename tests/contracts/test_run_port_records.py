"""The main run record exposes the same port observations as forward."""
import jax
import numpy as np
import pytest

from tests.contracts.run_forward_census import base_scene, SOURCE


@pytest.mark.parametrize('kind,path', [
    ('lumped', 'uniform'), ('wire', 'uniform'),
    ('wire', 'graded'), ('lumped', 'graded'),
])
def test_main_port_records_match_forward(kind, path):
    def declaration():
        sim = base_scene(path, source=False)
        options = {'extent': .003} if kind == 'wire' else {}
        return sim.add_port(SOURCE, 'ez', impedance=50., **options)

    bins = np.linspace(2e9, 20e9, 50, dtype=np.float32)
    # Graded lumped S requests are refused; both entries leave lumped DFTs None.
    spectra = not (kind == 'lumped' and path == 'graded')
    actual = declaration().run(n_steps=150, skip_preflight=True,
                               **({'s_param_freqs': bins} if spectra else {}))
    expected = declaration().forward(n_steps=150, skip_preflight=True, checkpoint=False,
                                     **({'port_s11_freqs': bins} if spectra else {}))
    field = f'{kind}_port_sparams'
    got, want = getattr(actual, field), getattr(expected, field)
    if not spectra:
        assert got is want is None
    else:
        assert got is not None, field
        assert want is not None
        assert len(got) == len(want) == 1
        for (meta, accs), (ref_meta, ref_accs) in zip(got, want, strict=True):
            assert jax.tree.structure(meta) == jax.tree.structure(ref_meta)
            for a, b in zip(jax.tree.leaves(meta), jax.tree.leaves(ref_meta), strict=True):
                np.testing.assert_array_equal(a, b)
            assert len(accs) == len(ref_accs)
            for a, b in zip(accs, ref_accs, strict=True):
                np.testing.assert_array_equal(a, b)
    assert len(actual.sparam_time_records) == len(expected.sparam_time_records) == 1
    for got_record, want_record in zip(actual.sparam_time_records, expected.sparam_time_records, strict=True):
        assert (got_record.id, got_record.kind, got_record.channels) == (
            want_record.id, want_record.kind, want_record.channels)
        np.testing.assert_array_equal(got_record.samples, want_record.samples)
