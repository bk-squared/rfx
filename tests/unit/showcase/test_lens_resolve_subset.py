from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/showcase"))
import _lens_filter_common as common
import design_lens as lens


@pytest.mark.parametrize("stage, options", [
    ("main", ["--designs", "best_uniform", "--duration-factor", "2"]),
    ("main", ["--designs", "best_uniform"]),
    ("main", ["--duration-factor", "1"]),
    ("resolve", ["--duration-factor", "0"]),
    ("resolve", ["--duration-factor", "nan"]),
    ("resolve", ["--duration-factor", "-1"]),
    ("resolve", ["--duration-factor", "inf"]),
    ("resolve", ["--designs", "foo"]),
    ("resolve", ["--designs", "grin,grin"]),
    ("resolve", ["--designs", ""]),
    ("resolve", ["--designs", "grin,"]),
])
def test_resolve_option_errors(tmp_path, stage, options):
    with pytest.raises(SystemExit) as exc:
        common.main(lens, ["--stage", stage, "--out", str(tmp_path),
                           "--boundary", "nearest", *options])
    assert exc.value.code == 2


@pytest.fixture
def resolve_standin(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "reported_record_length.json").write_text('{"n_steps": 3024, "dt_s": 1.6e-12}')
    (source / "baselines.json").write_text('{"best_uniform": "uniform_2.7"}')
    baselines = {"no_lens": object(), "grin": object(), "uniform_2.7": object()}
    reported = object()
    case = SimpleNamespace(CASE="lens", OBJECTIVE="boresight",
                           MESHES=(0.001, 0.00075, 0.0005), baselines=lambda: baselines)
    args = SimpleNamespace(out=tmp_path / "out", precision="float32", band=[9e9, 10e9, 11e9])
    model = SimpleNamespace(grid=SimpleNamespace(dt=0.8e-12))
    calls = SimpleNamespace(setup=[], evaluate=[], record=[], context=[])

    def setup(case, directory, dx, precision, smoke, **kwargs):
        calls.setup.append(dx)
        return model

    def evaluate(case, model, designs, n, directory, label):
        calls.evaluate.append((designs, n, directory, label))
        return {name: {} for name in designs}

    def record(directory, case, model, stage, details):
        calls.record.append(details)

    def context(case, model, source, best, n, directory, **kwargs):
        calls.context.append((n, kwargs))

    monkeypatch.setattr(common, "setup", setup)
    monkeypatch.setattr(common, "evaluate", evaluate)
    monkeypatch.setattr(common, "record", record)
    monkeypatch.setattr(common, "lens_resolve_context", context)
    monkeypatch.setattr(common, "selected_design", lambda source: (reported, 7))
    return case, args, source, baselines, reported, calls


@pytest.mark.parametrize("options, names, n, factor", [
    ({"designs": ["best_uniform"], "duration_factor": 2.0}, ["best_uniform"], 12096, 2.0),
    ({}, ["reported", "no_lens", "grin", "best_uniform"], 6048, 1.0),
    ({"designs": ["grin", "reported"], "duration_factor": 1.0}, ["grin", "reported"], 6048, 1.0),
])
def test_resolved_subset_and_duration(resolve_standin, options, names, n, factor):
    case, args, source, baselines, reported, calls = resolve_standin
    vars(args).update(options)
    expected_designs = {"reported": reported, "no_lens": baselines["no_lens"],
                        "grin": baselines["grin"], "best_uniform": baselines["uniform_2.7"]}
    expected = {name: expected_designs[name] for name in names}
    common.resolved(case, args, source)
    assert calls.setup == [0.001, 0.00075, 0.0005]
    assert len(calls.evaluate) == 3
    for (designs, steps, directory, label), mesh in zip(calls.evaluate, ["mesh_0.001", "mesh_0.00075", "mesh_0.0005"]):
        assert designs == expected
        assert list(designs) == names
        assert steps == n
        assert directory == args.out / mesh
        assert label == "resolve"
    assert calls.context == [(n, {"duration_factor": factor, "designs": names})] * 3
    assert len(calls.record) == 3
    for details in calls.record:
        assert details["n_steps"] == n
        assert details["duration_factor"] == factor
        assert details["designs"] == names


def test_unavailable_resolve_design(resolve_standin):
    case, args, source, _, _, calls = resolve_standin
    case.OBJECTIVE = "fan"
    args.designs = ["no_lens"]
    args.duration_factor = 1.0
    with pytest.raises(SystemExit, match="no_lens"):
        common.resolved(case, args, source)
    assert calls.evaluate == []
