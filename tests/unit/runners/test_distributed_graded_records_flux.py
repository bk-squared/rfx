"""Graded normal-H interpolation and post-assembly dual-area quadrature."""
import pytest
from rfx.probes.probes import flux_spectrum
from tests.unit.runners._graded_records_scene import model, coordinate, pair, measure, record, mutation_result, with_bars


def add_flux(sim, grid, seam):
    for axis, index in ((0, seam - 1), (0, seam), (1, grid.shape[1] // 2)):
        sim.add_flux_monitor(axis='xyz'[axis], coordinate=coordinate(grid, axis, index),
            freqs=[3e9, 7e9], name=f'{axis}/{index}', dft_window='hann')


@pytest.mark.parametrize('n_devices', [2, 3])
@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_flux(boundary, n_devices, entry):
    sim, grid, seam = model(boundary, n_devices)
    add_flux(sim, grid, seam)
    one, many = pair(sim, n_devices, entry)
    values = {}
    for name, expected in one.flux_monitors.items():
        actual = many.flux_monitors[name]
        for component in ('e1_dft', 'e2_dft', 'h1_dft', 'h2_dft', 'dA', 'dA2'):
            values[f'{name}/{component}'] = measure(getattr(actual, component), getattr(expected, component))
        values[f'{name}/power'] = measure(flux_spectrum(actual), flux_spectrum(expected))
    # CPML wrapped step: compile-form rounding, operands byte-equal (rfx-archive DESIGN.md 4g amendment 1).
    with_bars(values, 'relative_peak' if boundary == 'cpml' else 'array_equal')
    with_bars({name: value for name, value in values.items()
               if name.endswith(('/dA', '/dA2'))}, 'array_equal')
    record('flux', dict(boundary=boundary, devices=n_devices, entry=entry, shape=grid.shape), values)
    assert max(row['peak'] for name, row in values.items() if name.endswith('/power')) > 0
    assert all(row['passed'] for row in values.values()), ('flux bar', values)


@pytest.mark.parametrize('mutation', ['ghost_exchange', 'flux_owned_rows'])
@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
def test_flux_mutations(mutation, boundary, monkeypatch):
    measured, report = {}, record
    def capture(kind, metadata, values):
        measured.update(values)
        report(kind, metadata, values)
    monkeypatch.setitem(globals(), 'record', capture)
    if mutation == 'ghost_exchange':
        from rfx.runners import distributed_nu
        monkeypatch.setattr(distributed_nu, '_exchange_h_ghosts_nu', lambda state, *a, **kw: state)
    else:
        from rfx.runners import _distributed_records as records
        original, cut = records.PlaneRecords.__init__, records._cut
        def wrong(self, *args, **kwargs):
            if not kwargs.get('flux'):
                return original(self, *args, **kwargs)
            # Shift the flux cut's owned start, before local regions and
            # packed capacities are built; the y-normal seam row is omitted.
            with monkeypatch.context() as patch:
                patch.setattr(records, '_cut', lambda axis, index, region, start, end, nx_per:
                    cut(axis, index, region, start + (axis != 0), end, nx_per))
                original(self, *args, **kwargs)
        monkeypatch.setattr(records.PlaneRecords, '__init__', wrong)
    with pytest.raises(AssertionError, match='flux bar') as failure:
        test_flux(boundary, 3, 'run')
    if mutation == 'flux_owned_rows':
        assert any(not row['passed'] for name, row in measured.items() if name.startswith('1/'))
    mutation_result('flux', f'{mutation}/{boundary}', failure.value)
