"""Refused mixed entries, with same-host main values from a separate process."""
from pathlib import Path
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

from rfx.probes.probes import PreDecisionLumpedDiagonalWarning
from tests.unit.sparams.test_mixed_port_sparam import _base_sim, _add_msl, _H_SUB

_MAIN = "2c32b7cf79b796fae825bfa08fc73d9914e61723"
_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT = _ROOT / "output/mixeddiag"
_FIELDS = ("S", "S_raw", "S_wave", "passivity_correction", "s21_power_witness", "reliable")


def _run(wire, channel):
    # Existing one-cell lumped branch fixture in test_mixed_port_sparam.py;
    # only extent changes for its wire control. No new geometry or tolerances.
    sim, y_c = _base_sim()
    sim.add_port(position=(2e-3, y_c, 0.0), component="ez", impedance=50.0,
                 **({"extent": _H_SUB} if wire else {}))
    _add_msl(sim, y_c, n_probe_offset=10, n_probe_spacing=4)
    return sim.compute_mixed_s_matrix(
        freqs=np.linspace(1e9, 2e9, 3), num_periods=2.0,
        skip_preflight=True, magnitude_channel=channel,
    )


def _capture(path):
    # Executed with the archive as cwd and PYTHONPATH, never imports new rfx.
    import rfx
    assert Path(rfx.__file__).resolve().is_relative_to(Path.cwd())
    arrays = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for wire in (False, True):
            for channel in ("flux", "wave"):
                result = _run(wire, channel)
                prefix = f"{wire}_{channel}_"
                for field in _FIELDS:
                    value = getattr(result, field)
                    if value is not None:
                        arrays[prefix + field] = np.asarray(value)
                arrays[prefix + "unprojected"] = np.asarray(
                    result.S if result.S_raw is None else result.S_raw)
    np.savez(path, **arrays)


@pytest.fixture(scope="module")
def main_values():
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    source = _OUTPUT / "main-src"
    if not (source / "rfx/sparams/mixed.py").exists():
        source.mkdir(exist_ok=True)
        archive = subprocess.run(["git", "archive", _MAIN], cwd=_ROOT,
                                 check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(source)], input=archive,
                       check=True, capture_output=True)
    baseline = _OUTPUT / "main-values.npz"
    # An explicit cache is only for local mutation runs against the already
    # measured main artifact. Normal test runs always rebuild the baseline.
    if not (os.environ.get("RFX_MIXEDDIAG_REUSE_MAIN") == "1" and baseline.exists()):
        code = ("import runpy; m = runpy.run_path(" + repr(str(Path(__file__).resolve()))
                + "); m['_capture'](" + repr(str(baseline)) + ")")
        proc = subprocess.run([sys.executable, "-c", code], cwd=source,
                              env={**os.environ, "PYTHONPATH": "."}, capture_output=True)
        (_OUTPUT / "main-capture.log").write_bytes(proc.stdout + proc.stderr)
        assert proc.returncode == 0, (_OUTPUT / "main-capture.log").read_text()
    with np.load(baseline) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module", params=[(False, "flux"), (False, "wave"),
                                       (True, "flux"), (True, "wave")],
                ids=["lumped-flux", "lumped-wave", "wire-flux", "wire-wave"])
def measured(request, main_values):
    import rfx.sparams.mixed as mixed
    wire, channel = request.param
    calls = []
    project = mixed._project_passive

    def recording_projection(s):
        calls.append(True)
        return project(s)

    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings(record=True) as seen:
        patch.setattr(mixed, "_project_passive", recording_projection)
        warnings.simplefilter("always")
        result = _run(wire, channel)
    np.savez(_OUTPUT / f"current-{wire}-{channel}.npz", **{
        field: np.asarray(getattr(result, field)) for field in _FIELDS
        if getattr(result, field) is not None
    })
    return wire, channel, result, seen, calls, main_values


def _bits(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def test_returned_entries_and_witnesses(measured):
    wire, channel, result, _, _, main = measured
    prefix = f"{wire}_{channel}_"
    if wire:
        for field in _FIELDS:
            expected = main.get(prefix + field)
            actual = getattr(result, field)
            if expected is None:
                assert actual is None
            else:
                _bits(actual, expected)
        return
    s = np.asarray(result.S)
    assert np.isnan(s[0, 0]).all()
    assert np.isnan(result.s21_power_witness[:, 0]).all()
    _bits(s[:, 1], main[prefix + "unprojected"][:, 1])
    _bits(result.reliable, main[prefix + "reliable"])
    if channel == "flux":
        assert np.isnan(s[1, 0]).all()
        assert np.isnan(result.S_wave[0, 0]).all()
        _bits(result.S_wave[1, 0], main[prefix + "S_wave"][1, 0])
        _bits(result.S_wave[:, 1], main[prefix + "S_wave"][:, 1])
    else:
        _bits(s[1, 0], main[prefix + "unprojected"][1, 0])
        assert result.S_wave is None


def test_projection_is_not_formed_and_warning_names_refusal(measured):
    wire, channel, result, seen, calls, _ = measured
    refusals = [w for w in seen if "refused driven lumped entries" in str(w.message)]
    assert not any(issubclass(w.category, PreDecisionLumpedDiagonalWarning) for w in seen)
    if wire:
        assert calls == [True]
        assert not refusals
        return
    assert not calls
    assert result.S_raw is None
    assert result.passivity_correction is None
    assert len(refusals) == 1
    assert refusals[0].category is UserWarning
    message = str(refusals[0].message)
    assert "S[0,0]" in message and "'lw0'" in message and "index 0" in message
    assert ("flux off-diagonals S[1,0]" in message) == (channel == "flux")
    assert "joint passivity projection is not formed" in message
    assert ("the driven reflection of a lumped port comes from run(compute_s_params=True) "
            "or compute_lumped_wire_s_matrix_via_scan on a lumped-only model") in message
    assert not any("nonfinite_sparams" in str(w.message) for w in seen)


def test_partial_checks_use_only_finite_entries():
    from types import SimpleNamespace
    from rfx.sparams._common import _mixed_reciprocity_deviation, _warn_if_nonpassive_smatrix
    s = np.full((3, 3, 2), np.nan + 0j)
    s[0, 1, 0], s[1, 0, 0] = 0.2, 0.4
    s[:, 2, 1] = [0.8, 0.8, 0.0]
    assert _mixed_reciprocity_deviation(s) == ((0, 1), 0.5)
    result = SimpleNamespace(S=s, freqs=[1e9, 2e9], port_names=("a", "b", "c"))
    with pytest.warns(UserWarning, match="driven port 2, frequency index 1") as seen:
        _warn_if_nonpassive_smatrix(result, extractor="mixed", skip_nan=True)
    assert len(seen) == 1
    s[:, 2, 1] = 0.0
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        _warn_if_nonpassive_smatrix(result, extractor="mixed", skip_nan=True, strict=True)
    assert not seen
    # Refusal suppresses NaN findings, not an unrelated infinite measurement.
    s[2, 2, 1] = np.inf
    with pytest.warns(UserWarning, match="contains 1 Inf"):
        _warn_if_nonpassive_smatrix(result, extractor="mixed", skip_nan=True)
    with pytest.raises(ValueError, match="contains 1 Inf"):
        _warn_if_nonpassive_smatrix(result, extractor="mixed", skip_nan=True, strict=True)


def test_every_driven_lumped_column_is_refused():
    from rfx.sparams._common import (
        _assemble_mixed_power_wave_s, _mixed_flux_magnitude_override,
    )
    # With Z0=4, (-V+Z0*I)/(2*sqrt(Z0))=1 at each drive;
    # (V-Z0*I)/(2*sqrt(Z0))=1 at every passive receive.
    v = np.full((3, 2, 2), 2.0 + 0j)
    current = np.full_like(v, -0.5)
    for k in range(2):
        v[k, k], current[k, k] = -2.0, 0.5
    vm = np.full((3, 1, 2), 2.0 + 0j)
    im = np.full_like(vm, -0.5)
    im[2] = 0.5
    plan = [("lw", 0), ("lw", 1), ("msl", 0)]
    wave, witness = _assemble_mixed_power_wave_s(
        v, current, vm, im, [4.0, 4.0], [1, 1], [4.0], False, plan,
    )
    expected = np.ones((3, 3, 2), dtype=np.complex64)
    expected[0, 0] = expected[1, 1] = np.nan
    expected[2, 2] = 0.0
    np.testing.assert_array_equal(wave, expected)
    assert np.isnan(witness).all()
    box = np.full((3, 2, 2), -1.0)
    box[0, 0] = box[1, 1] = 1.0
    plane = np.ones((3, 1, 2))
    plane[2] = -1.0
    flux, ill, _ = _mixed_flux_magnitude_override(
        wave, box, plane, plan, [-1], 2, refused_columns=(0, 1),
    )
    expected[:, :2] = np.nan
    np.testing.assert_array_equal(flux, expected)
    assert not ill.any()


def test_flux_refusal_does_not_change_other_nan_diagonals():
    from rfx.sparams._common import _mixed_flux_magnitude_override
    s = np.ones((2, 2, 1), dtype=np.complex64)
    s[0, 0] = s[1, 1] = np.nan
    box = np.array([[[1.0]], [[-1.0]]])
    plane = np.array([[[1.0]], [[-1.0]]])
    args = (s, box, plane, [("lw", 0), ("msl", 0)], [-1], 1)
    # Main uses infinity as the incident-power fallback for these unrelated
    # NaNs, yielding zero off-diagonal magnitude. Wire mode keeps that rule.
    legacy, _, _ = _mixed_flux_magnitude_override(*args)
    np.testing.assert_array_equal(np.asarray(legacy)[[0, 1], [1, 0]], 0.0)
    refused, _, _ = _mixed_flux_magnitude_override(*args, refused_columns=(0,))
    assert np.isnan(refused[1, 0]).all()
    np.testing.assert_array_equal(refused[0, 1], 0.0)
    assert np.isnan(refused[1, 1]).all()
