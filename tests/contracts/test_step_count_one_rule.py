"""Every period-based request uses one floating-point step-count rule."""

import ast
from pathlib import Path

import numpy as np

from rfx import Simulation
from rfx.stepping.step_count import steps_for_periods

ROOT = Path(__file__).resolve().parents[2]
RULE = "rfx/stepping/step_count.py"


def scan_step_counts(source: str, path: str) -> list[str]:
    """Find ceil calls whose argument names a period quantity and a dt quantity."""
    hits = []
    for node in ast.walk(ast.parse(source, filename=path)):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", getattr(node.func, "attr", None))
        if name != "ceil":
            continue
        identifiers = {
            child.id if isinstance(child, ast.Name) else child.attr
            for argument in [*node.args, *(kw.value for kw in node.keywords)]
            for child in ast.walk(argument)
            if isinstance(child, (ast.Name, ast.Attribute))
        }
        if any("period" in identifier for identifier in identifiers) and any(
            "dt" in identifier for identifier in identifiers
        ):
            hits.append(f"{path}:{node.lineno}")
    return hits


def test_no_inline_step_count_outside_the_rule():
    hits = []
    for path in sorted((ROOT / "rfx").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative != RULE:
            hits.extend(scan_step_counts(path.read_text(encoding="utf-8"), relative))
    assert not hits, "Inline step-count rules:\n" + "\n".join(hits)


def _simulation(**kwargs):
    sim = Simulation(freq_max=10e9, domain=(0.012,) * 3, dx=0.003,
                     boundary="pec", cpml_layers=0, **kwargs)
    sim.add_source((0.006,) * 3, "ez", amplitude_kind="current")
    sim.add_probe((0.006,) * 3, "ez")
    return sim


def test_graded_run_forward_same_step_count_at_boundary():
    sim = _simulation(dz_profile=np.full(4, 0.003), dt=4e-12)
    assert sim._build_nonuniform_grid().dt == 4e-12
    # This point-source request's preflight carries no record length or count;
    # the source scan covers the waveguide preflight conversions.
    sim.preflight()
    run_rows = sim.run(num_periods=20).time_series.shape[0]
    forward_rows = sim.forward(num_periods=20).time_series.shape[0]
    assert (run_rows, forward_rows) == (501, 501)
    assert steps_for_periods(4e-12, 10e9, 20) == 501


def test_uniform_run_forward_same_step_count():
    sim = _simulation()
    dt = sim._build_grid().dt
    num_periods = 40 * 10e9 * dt
    expected = int(np.ceil(float(num_periods) * (1.0 / float(10e9)) / float(dt)))
    run_rows = sim.run(num_periods=num_periods).time_series.shape[0]
    forward_rows = sim.forward(num_periods=num_periods).time_series.shape[0]
    assert run_rows == forward_rows == expected
    assert steps_for_periods(dt, 10e9, num_periods) == expected
