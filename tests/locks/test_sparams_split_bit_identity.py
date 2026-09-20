"""A/B bit-identity harness for the #980 split of ``rfx/api/_sparams.py``.

Issue #980 Phase 2 breaks the 8 647-line ``rfx/api/_sparams.py`` into
``rfx/sparams/`` a piece at a time. Every step is PURE CODE MOTION, and
``docs/agent-memory/development_methodology.md`` §2.2 gates a code-motion
refactor on BIT IDENTITY -- ``np.array_equal`` on the raw arrays, never a
tolerance -- not on "the physics still looks right".

This module is that gate. It drives each public ``compute_*`` S-matrix leg
end to end on the smallest existing fixture that reaches it, snapshots every
numeric field of the returned result, and compares the snapshot byte-exactly
against a baseline captured on the SAME host before the move.

Why the baseline is NOT committed
---------------------------------
The same FDTD code is measured to differ by up to 1.3e-3 relative between
macOS and the Linux cluster, and XLA is free to re-fuse a graph between
versions. Bit identity is therefore promised only for A/B on ONE host with
ONE toolchain -- which is exactly the question a code-motion refactor asks.
A committed baseline would be a cross-platform pin the code never made, so
the arrays live outside the repo and the test skips when they are absent.

Use
---
Capture, on the tree BEFORE the move::

    RFX_SPARAMS_BASELINE_DIR=/path/to/baseline \\
    RFX_SPARAMS_BASELINE_CAPTURE=1 \\
    pytest tests/locks/test_sparams_split_bit_identity.py

Compare, on the tree AFTER the move (same host, same interpreter, same
XLA_FLAGS)::

    RFX_SPARAMS_BASELINE_DIR=/path/to/baseline \\
    pytest tests/locks/test_sparams_split_bit_identity.py

Each leg writes ``<leg>.npz`` plus a ``<leg>.sha256`` sidecar over the
canonical byte serialisation of the snapshot, so the two sides can also be
diffed by hand from a PR body.

``test_sparams_module_namespace_is_the_declared_re_export_surface`` needs no
baseline and never skips: it pins the 66 module-level names ``rfx.api._sparams``
exported at ``060cb916`` (the pre-split main). ``from rfx.api._sparams import
<helper>`` appears in 47 files, and two tests monkeypatch
``"rfx.api._sparams.<name>"`` by string, so the re-export block the split adds
has to keep that namespace exactly whole -- no name added, none dropped.

Fixture provenance, per leg (smallest existing setup that reaches the leg;
step counts are truncated on purpose -- an A/B identity check does not need a
settled record, it needs a deterministic one that walks the whole path):

* waveguide      -- ``tests/_pec_short_advisory_fixture.build``, the fixture
  ``tests/unit/sparams/test_sparam_passivity_guard.py`` drives at
  ``num_periods=1``; run here on the default lane AND on ``normalize="flux"``
  because the two take different extractors.
* msl            -- ``_thru()`` of
  ``tests/unit/sparams/test_msl_passivity_enforcement.py`` (12 x 8 x 3.2 mm at
  dx = 200 um), ``num_periods=2``.
* mixed          -- the Layer-1c plumbing smoke of
  ``tests/unit/sparams/test_mixed_port_sparam.py``, ``num_periods=4``.
* coaxial        -- ``_make_one_port_sim()`` of
  ``tests/unit/sparams/test_coaxial_s_matrix.py``, ``n_steps=200`` (the
  DEPRECATED single-plane lane; it is still shipped, so it is still gated).
* coax line      -- geometry of ``_run()`` in
  ``tests/unit/sparams/test_coaxial_line_reflection.py``. Every end-to-end
  setup there is ``slow_physics`` at ``n_steps=5000``; this lock reuses the
  geometry verbatim at ``n_steps=400`` so the leg is covered in the fast lane.
  NOTE, honestly: that record is far from settled and its |Gamma| is not the
  calibrated number -- the calibration gates stay in that file, this is an
  identity witness only.
* coax two-port  -- ``_coax_two_port_sim()`` of
  ``tests/unit/sparams/test_settling_witness.py`` (same geometry as ``_sim()``
  in ``test_coax_two_port_smatrix.py``), ``n_steps=400``, the step count that
  file's own ``slow_physics`` truncation test uses.
* coax<->msl     -- ``tests/_coax_msl_instrument_fixture`` with its own
  ``instrument_kwargs(n_steps=200)``, as
  ``tests/unit/sparams/test_coax_msl_transition_ladder_dump.py`` drives it.

No ``jax.config.update('jax_enable_x64', True)`` here: it is process-global and
would red every same-process pytest-split shard.
"""

from __future__ import annotations

LOCK_PROVENANCE = {
    "fixture": "tests/_pec_short_advisory_fixture.py,tests/_coax_msl_instrument_fixture.py",
    "generator": "tests/locks/test_sparams_split_bit_identity.py (RFX_SPARAMS_BASELINE_CAPTURE=1)",
    "commit": "060cb916",
    "date": "2026-09-13",
    "run_id": "local",
    "host": "remilab pod linux x86_64, CPU, python 3.11.16, jax 0.10.2",
    "pinned_until": "2027-03-13",
}

import hashlib
import os
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse

_BASELINE_ENV = "RFX_SPARAMS_BASELINE_DIR"
_CAPTURE_ENV = "RFX_SPARAMS_BASELINE_CAPTURE"

_SKIP_REASON = (
    f"set {_BASELINE_ENV} to run the #980 A/B bit-identity gate. It compares "
    "raw S arrays against a baseline captured on the SAME host with the SAME "
    "interpreter and XLA flags, before the code motion -- bit identity across "
    "CPUs / XLA versions is not promised, so the baseline is deliberately not "
    f"committed. Capture it with {_BASELINE_ENV}=<dir> {_CAPTURE_ENV}=1 pytest "
    "tests/locks/test_sparams_split_bit_identity.py on the pre-move tree."
)


# ---------------------------------------------------------------------------
# The re-export surface. Every module-level name bound by rfx/api/_sparams.py
# at 060cb916, i.e. the pre-split main. Regenerate with:
#   python -c "import rfx.api._sparams as m; print(sorted(n for n in vars(m) \
#              if not (n.startswith('__') and n.endswith('__'))))"
# ---------------------------------------------------------------------------
_SPARAMS_NAMESPACE_AT_060CB916 = (
    "CoaxMSLTransitionResult", "CoaxialLineReflectionResult", "CoaxialPort",
    "CoaxialSMatrixResult", "CoaxialTwoPortResult", "GaussianPulse",
    "MSLSMatrixResult", "MixedSMatrixResult", "NonUniformGrid", "TYPE_CHECKING",
    "WAVEGUIDE_PHASE_BETA_CONVENTION", "WAVEGUIDE_PHASE_MAG_FLOOR",
    "WAVEGUIDE_RECIPROCITY_ADVISORY_TOL", "WaveguideSMatrixResult",
    "_C0_SPARAMS", "_FAR_PORT_LAMBDA_G_FRACTION", "_MSLPortEntry",
    "_SETTLING_WITNESS_DB", "_SparamMixin", "_WaveguidePortEntry",
    "_assemble_coax_msl_transition_from_voltages",
    "_assemble_coaxial_two_port_from_voltages", "_assemble_mixed_power_wave_s",
    "_assert_nu_shift_span_in_one_grading_zone", "_collocated_msl_h",
    "_enable_x64", "_finalize_sparam_result", "_ladder_split_witness",
    "_mixed_flux_magnitude_override", "_mixed_reciprocity_deviation",
    "_msl_axis_spacing", "_msl_cell_profile", "_msl_power_wave_scales",
    "_msl_wave_split_reliability", "_nu_shift_span_cells", "_project_passive",
    "_reciprocity_advisory_message", "_register_msl_h_planes",
    "_resolve_msl_auto_offsets", "_validate_extra_flux_monitor_entries",
    "_warn_if_nonpassive_smatrix", "_warn_if_passivity_projected",
    "_warn_if_ringdown_truncated", "_warn_junction_cpml_thickness",
    "_warn_junction_probe_clearance", "_warn_msl_beta_scan_railed",
    "_warn_msl_wave_split_unreliable", "_warn_ntff_box_dropped",
    "_warn_thin_absorber_vs_guide_wavelength", "_waveguide_s21_phase_residual",
    "annotations", "extract_multimode_s_matrix", "extract_multimode_s_matrix_flux",
    "extract_waveguide_s_matrix", "extract_waveguide_s_matrix_flux",
    "extract_waveguide_s_params_normalized", "interior_cells", "is_tracer",
    "jax", "jnp", "msl_modal_voltage", "msl_solve_s_from_waves", "np",
    "s21_phase_residual_deg_rms", "settling_verdict", "waveguide_plane_positions",
)


def test_sparams_module_namespace_is_the_declared_re_export_surface():
    """``rfx.api._sparams``'s module namespace must stay exactly whole.

    47 files do ``from rfx.api._sparams import <helper>``; two more patch
    ``"rfx.api._sparams.<name>"`` as a STRING, which binds nothing and fails
    silently the moment the name stops living there. When #980 moves a helper
    body out, the explicit re-export block has to put the name back -- and a
    name the split ACCIDENTALLY adds is just as much a surface change, so this
    is set equality, not a subset check.
    """
    import rfx.api._sparams as mod

    live = {n for n in vars(mod) if not (n.startswith("__") and n.endswith("__"))}
    declared = set(_SPARAMS_NAMESPACE_AT_060CB916)
    assert live - declared == set(), (
        "rfx.api._sparams gained module-level names not in the pre-split "
        f"surface: {sorted(live - declared)}"
    )
    assert declared - live == set(), (
        "rfx.api._sparams lost module-level names the pre-split surface had "
        f"(the re-export block is incomplete): {sorted(declared - live)}"
    )


# ---------------------------------------------------------------------------
# Fixtures. Copied verbatim from the test modules named in the docstring so
# this lock is frozen against later edits to those files; the baseline is
# meaningless if the geometry it was captured on can drift underneath it.
# ---------------------------------------------------------------------------

_WG_FREQS = np.linspace(4e9, 6e9, 6)


def _waveguide_result(normalize):
    """``tests/_pec_short_advisory_fixture`` at dx = 2 mm, cpml 8."""
    from tests._pec_short_advisory_fixture import build

    sim = build(_WG_FREQS, dx=2e-3, cpml=8)
    return sim.compute_waveguide_s_matrix(normalize=normalize, num_periods=1)


# --- msl: _thru() of tests/unit/sparams/test_msl_passivity_enforcement.py ---
_MSL_FREQS = jnp.linspace(2e9, 18e9, 16)


def _msl_result():
    sim = Simulation(freq_max=20e9, domain=(0.012, 0.008, 0.0032),
                     dx=2e-4, boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.012, 0.008, 0.0008)), material="sub")
    sim.add(Box((0., 0., 0.), (0.012, 0.008, 0.)), material="pec")
    sim.add(Box((0.0, 0.0034, 0.0008), (0.012, 0.0046, 0.0008)), material="pec")
    sim.add_msl_port(position=(0.002, 0.004, 0.0), width=0.0012, height=0.0008,
                     direction="+x", impedance=50.0, eps_r_sub=2.2, name="p1")
    sim.add_msl_port(position=(0.010, 0.004, 0.0), width=0.0012, height=0.0008,
                     direction="-x", impedance=50.0, eps_r_sub=2.2, name="p2")
    return sim.compute_msl_s_matrix(freqs=_MSL_FREQS, num_periods=2.0)


# --- mixed: Layer-1c smoke of tests/unit/sparams/test_mixed_port_sparam.py ---
_MIXED_EPS_R = 3.66
_MIXED_H_SUB = 254e-6
_MIXED_W_TRACE = 600e-6
_MIXED_DX = _MIXED_H_SUB / 3.0


def _mixed_result():
    lx, ly, lz = 8e-3, 3e-3, 754e-6
    sim = Simulation(
        freq_max=5e9, domain=(lx, ly, lz), dx=_MIXED_DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("sub", eps_r=_MIXED_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _MIXED_H_SUB)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0.0, y_c - _MIXED_W_TRACE / 2, _MIXED_H_SUB),
                (lx, y_c + _MIXED_W_TRACE / 2, _MIXED_H_SUB)), material="pec")
    sim.add_port(position=(2e-3, y_c, 0.0), component="ez",
                 impedance=50.0, extent=_MIXED_H_SUB)
    sim.add_msl_port(position=(5.5e-3, y_c, 0.0), width=_MIXED_W_TRACE,
                     height=_MIXED_H_SUB, direction="-x", impedance=50.0,
                     waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5),
                     n_probe_offset=10, n_probe_spacing=4)
    return sim.compute_mixed_s_matrix(
        freqs=np.linspace(1e9, 4e9, 5), num_periods=4.0, skip_preflight=True,
    )


# --- coaxial (DEPRECATED lane): tests/unit/sparams/test_coaxial_s_matrix.py --
def _coaxial_result():
    sim = Simulation(freq_max=10.0e9, domain=(0.020, 0.020, 0.020),
                     boundary="pec")
    sim.add_coaxial_port((0.010, 0.010, 0.015), face="top")
    return sim.compute_coaxial_s_matrix(n_steps=200, n_freqs=3)


# --- coax line: geometry of _run() in test_coaxial_line_reflection.py --------
_COAX_BAND = jnp.asarray([4.0e9, 6.0e9, 8.0e9, 10.0e9, 12.0e9])


def _coax_line_result():
    sim = Simulation(domain=(0.008, 0.008, 0.040), freq_max=40.0e9,
                     boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    return sim.compute_coaxial_line_reflection(
        termination="short", n_steps=400, freqs=_COAX_BAND)


# --- coax two-port: _coax_two_port_sim() of test_settling_witness.py ---------
def _coax_two_port_result():
    sim = Simulation(domain=(0.008, 0.008, 0.060), freq_max=40.0e9,
                     boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    return sim.compute_coaxial_two_port(n_steps=400, freqs=_COAX_BAND)


# --- coax <-> msl transition: tests/_coax_msl_instrument_fixture -------------
def _coax_msl_result():
    from tests._coax_msl_instrument_fixture import (
        build_instrument_junction, instrument_kwargs,
    )

    sim = build_instrument_junction()
    return sim.compute_coax_msl_transition(**instrument_kwargs(200))


# ---------------------------------------------------------------------------
# (leg id, builder, numeric fields to pin). The field lists enumerate every
# numeric field the result dataclass carries, so a field cannot escape the
# witness by being forgotten here.
# ---------------------------------------------------------------------------
_LEGS = (
    ("waveguide", lambda: _waveguide_result(False),
     ("s_params", "freqs", "reference_planes", "settling_db",
      "s21_phase_residual_deg_rms")),
    ("waveguide_flux", lambda: _waveguide_result("flux"),
     ("s_params", "freqs", "reference_planes", "settling_db",
      "s21_phase_residual_deg_rms")),
    ("msl", _msl_result,
     ("S", "freqs", "Z0", "beta", "reliable", "settling_db", "S_raw",
      "passivity_correction", "cond_a", "reference_impedances")),
    ("mixed", _mixed_result,
     ("S", "freqs", "z0_ref", "settling_db", "s21_power_witness", "reliable",
      "S_raw", "passivity_correction", "S_wave")),
    ("coaxial", _coaxial_result,
     ("s_params", "freqs", "reference_planes", "z_tem_ohm", "voltages",
      "currents")),
    ("coaxial_line_reflection", _coax_line_result,
     ("s11", "freqs", "gamma", "recurrence_residual", "fit_residual",
      "annulus_cells", "z0_numerical_ohm")),
    ("coaxial_two_port", _coax_two_port_result,
     ("s_params", "freqs", "reference_planes", "cond_a", "recurrence_residual",
      "fit_residual", "gamma", "annulus_cells", "settling_db")),
    ("coax_msl_transition", _coax_msl_result,
     ("s_params", "freqs", "reference_planes", "z0_ref", "cond_a",
      "cond_a_equilibrated", "recurrence_residual", "fit_residual", "gamma",
      "a_inc", "b_out", "settling_db")),
)

_NONE_KEY = "__none_fields__"


def _snapshot(result, fields) -> dict[str, np.ndarray]:
    """Every named field as a raw numpy array; ``None``s recorded by name.

    ``np.asarray`` on a jax array is a device-to-host copy, not a cast: the
    dtype and every bit of the mantissa survive, which is what the gate reads.
    """
    out: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for name in fields:
        assert hasattr(result, name), (
            f"{type(result).__name__} has no field {name!r}; the field list in "
            "this lock has drifted from the result dataclass"
        )
        value = getattr(result, name)
        if value is None:
            missing.append(name)
            continue
        out[name] = np.asarray(value)
    out[_NONE_KEY] = np.asarray(sorted(missing), dtype="<U64")
    return out


def _digest(snapshot: dict[str, np.ndarray]) -> str:
    """SHA256 over name + dtype + shape + raw bytes, in sorted key order."""
    h = hashlib.sha256()
    for key in sorted(snapshot):
        arr = np.ascontiguousarray(snapshot[key])
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _baseline_dir() -> Path | None:
    raw = os.environ.get(_BASELINE_ENV, "").strip()
    return Path(raw) if raw else None


def _capturing() -> bool:
    return os.environ.get(_CAPTURE_ENV, "").strip() not in ("", "0", "false")


@pytest.mark.parametrize("leg,build,fields", _LEGS, ids=[leg for leg, _, _ in _LEGS])
def test_sparam_leg_is_bit_identical_to_the_pre_split_baseline(leg, build, fields):
    base = _baseline_dir()
    if base is None:
        pytest.skip(_SKIP_REASON)

    with warnings.catch_warnings():
        # These records are deliberately truncated, and the deprecated coax
        # lane warns by design. The warnings are not what this gate measures;
        # the advisory behaviour itself is pinned in tests/unit/sparams.
        warnings.simplefilter("ignore")
        result = build()
    got = _snapshot(result, fields)
    got_sha = _digest(got)

    npz = base / f"{leg}.npz"
    sidecar = base / f"{leg}.sha256"

    if _capturing():
        base.mkdir(parents=True, exist_ok=True)
        np.savez(npz, **got)
        sidecar.write_text(got_sha + "\n", encoding="utf-8")
        pytest.skip(f"captured baseline for {leg}: {npz} sha256={got_sha}")

    assert npz.exists(), (
        f"no baseline for leg {leg!r} at {npz}. Capture it on the pre-move "
        f"tree with {_CAPTURE_ENV}=1."
    )
    with np.load(npz, allow_pickle=False) as data:
        ref = {k: data[k] for k in data.files}

    assert sorted(ref) == sorted(got), (
        f"{leg}: field set changed -- baseline has {sorted(ref)}, this tree "
        f"produced {sorted(got)}. A code-motion refactor may not add or drop "
        "a result field."
    )
    for key in sorted(got):
        a, b = got[key], ref[key]
        assert a.dtype == b.dtype, f"{leg}.{key}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{leg}.{key}: shape {a.shape} != {b.shape}"
        # np.array_equal, no tolerance: methodology 2.2 gates code motion on
        # bit identity. equal_nan=True because a NaN witness slot (e.g. the
        # differentiable coax path's settling_db) is a legitimate value here
        # and must compare equal to the same NaN slot in the baseline.
        if np.issubdtype(a.dtype, np.floating) or np.issubdtype(a.dtype, np.complexfloating):
            same = np.array_equal(a, b, equal_nan=True)
        else:
            same = np.array_equal(a, b)
        assert same, (
            f"{leg}.{key} is NOT bit-identical to the pre-split baseline.\n"
            f"  baseline sha256 : {sidecar.read_text().strip() if sidecar.exists() else '(none)'}\n"
            f"  this tree sha256: {got_sha}\n"
            f"  first mismatch  : {_first_mismatch(a, b)}"
        )

    if sidecar.exists():
        assert sidecar.read_text().strip() == got_sha, (
            f"{leg}: every array compared equal but the snapshot digest moved "
            f"({sidecar.read_text().strip()} -> {got_sha}); the serialisation "
            "itself changed"
        )


def _first_mismatch(a: np.ndarray, b: np.ndarray) -> str:
    flat_a, flat_b = np.ravel(a), np.ravel(b)
    for i in range(flat_a.size):
        x, y = flat_a[i], flat_b[i]
        if x != y and not (x != x and y != y):        # NaN == NaN for this report
            return f"index {np.unravel_index(i, a.shape)}: {x!r} != {y!r}"
    return "(none found — arrays differ only in a way this scan cannot see)"
