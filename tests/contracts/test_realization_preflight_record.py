"""Record construction participates in preflight's assembly diagnostics."""
import pytest

from tests.unit.boundaries.test_periodic_extents import model


def test_periodic_record_build_reports_assembly_refusal(monkeypatch):
    from rfx.model import realization_table

    sim = model()
    sim.add_port((.009, .003, .002), 'ex', extent=.002, impedance=50., excite=False)
    original = realization_table.bind_table
    failures = []

    def observe(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except ValueError as exc:
            failures.append(str(exc))
            raise

    monkeypatch.setattr(realization_table, 'bind_table', observe)
    escaped = None
    try:
        report = sim.preflight()
    except ValueError as exc:
        escaped = exc
    assert escaped is None, f"record assembly escaped preflight: {escaped}"
    text = ("periodic axis 'x': interval [0.009, 0.011] m crosses the seam of "
            "period L=0.01 m; split the interval into declarations inside [0, L].")
    assert failures == [text]
    assert [(i.code, i.source, str(i)) for i in report.errors] == [
        ('assembly_refusal', '_validate_cfg_assembly', f'The model cannot be assembled: {text}')]
    with pytest.raises(ValueError) as strict:
        sim.preflight(strict=True)
    assert str(strict.value) == f'preflight (strict) found 1 issue(s):\n  - The model cannot be assembled: {text}'
