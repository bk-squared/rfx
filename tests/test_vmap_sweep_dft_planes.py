"""Tests for DFT plane accumulators on the vmap fast path (#578).

CPU-runnable on tiny grids (intentionally not gpu-marked, matching
``test_vmap_sweep_eligibility.py``, so the PR gate exercises this file).

Comparator-class work (rfx-known-issues.md case ledger; workspace R5): the
equivalence tests below carry a predeclared tolerance derived from an
inspected per-bin dump (not asserted blind), and a one-sided falsifier
ritual (deliberate sign flip in the vmap DFT kernel, which must turn the
equivalence check red) was run before this tolerance was trusted.

R5 per-bin dump (batched vmap vs. sequential ``sim.run()``, plane p1 =
axis=x, coordinate=0.010 m, component=ez, n_freqs=4, eps_r=2.0, n_steps=60
-- the eps_r=6.0 case measured tighter, 2.6e-4 to 2.8e-4 uniformly, and is
omitted for brevity):

    freq (Hz)     |run acc|      |vmap acc|     complex maxdiff   rel diff
    5.0000e+08    4.857830e-12   4.857516e-12   2.500504e-15      5.147e-04
    2.0000e+09    4.366568e-12   4.366276e-12   2.269723e-15      5.198e-04
    3.5000e+09    3.554534e-12   3.554285e-12   1.870204e-15      5.261e-04
    5.0000e+09    2.804546e-12   2.804345e-12   1.480970e-15      5.281e-04

Magnitudes agree to ~5 significant figures at every bin (rel diff 2.6e-4 to
5.3e-4, uniform across frequency and eps_r) -- consistent with ordinary
float32 accumulation roundoff between the two independently-coded scan
bodies, not a systematic phase/sign/off-by-one defect. Falsifier (temporary
sign flip of the vmap DFT kernel to ``exp(+1j...)``, reverted via ``git
checkout`` immediately after each run): relative error jumped to 0.11-1.79
(O(1)) uniformly across every bin, both when run as a standalone script and
when run against this committed pytest file -- and ONLY the
``TestVmapDftPlaneFastPath`` tests went red, confirming the falsifier is
localized to the code path it is meant to catch. So the gates below use
rtol with a comfortable margin above the observed ~5e-4 floor, not a
tight-as-possible bound.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.testing as npt
import pytest

try:  # modern JAX: scoped x64 promoted to top-level (experimental removed v0.8.0)
    from jax import enable_x64 as _enable_x64
except ImportError:  # older JAX (< ~0.4.31)
    from tests._x64_compat import enable_x64 as _enable_x64

from rfx import Simulation, GaussianPulse, Box
from rfx.vmap_sweep import vmap_material_sweep, VmapSweepResult

_FALLBACK_MATCH = "Falling back to sequential"


def _dft_sim(boundary: str = "cpml", eps_r: float = 4.0,
             amplitude_kind: str | None = None) -> Simulation:
    """CPML/PEC dielectric slab with one DFT plane MID-domain (not near a
    boundary): moving the plane away from the domain edge matters — near
    the edge the accumulated signal is dominated by roundoff noise (near
    zero at these short step counts) and a relative-tolerance check would
    pass vacuously without exercising anything (discovered while building
    this file: a plane at x=0.016 gave accumulator magnitudes ~1e-15 with
    no real signal; x=0.010 gives ~1e-12, a genuine non-trivial spectrum).
    """
    kwargs = {"cpml_layers": 6} if boundary == "cpml" else {}
    sim = Simulation(
        freq_max=5e9, domain=(0.02, 0.02, 0.02), boundary=boundary,
        dx=0.002, **kwargs,
    )
    sim.add_material("substrate", eps_r=eps_r)
    sim.add(Box((0.005, 0, 0), (0.015, 0.02, 0.02)), material="substrate")
    sim.add_source((0.01, 0.01, 0.01), "ez", waveform=GaussianPulse(f0=3e9),
                    amplitude_kind=amplitude_kind)
    sim.add_probe((0.005, 0.01, 0.01), "ez")
    sim.add_dft_plane_probe(axis="x", coordinate=0.010, component="ez",
                             n_freqs=4, name="p1")
    return sim


def _assert_dft_planes_match(vmap_planes: dict, ref_planes: dict, *,
                              rtol: float, ctx: str) -> None:
    """Compare dict[str, DFTPlaneProbe] (a Result) against a single batch
    slice of VmapSweepResult.dft_planes, comparison arithmetic in f64
    (#527 style: the accumulators are complex64-scale; casting up before
    diffing avoids a spurious float32-comparator artefact).

    Per-FREQUENCY-BIN relative error, normalized by that bin's own peak
    magnitude over the plane -- NOT a blind elementwise
    ``assert_allclose``. A DFT plane has near-zero cells far from the
    source/trace where the true signal is dominated by roundoff-level
    cancellation noise; an elementwise rtol check flags those as huge
    relative mismatches even though both paths agree to within float32
    precision on the actual signal (discovered building this test: doing
    the elementwise check first gave 27% "mismatched" elements, all at
    |value| ~ 1e-16 next to a ~1e-12 peak). This is the same max-abs-diff
    / max-abs-reference ratio manually verified and inlined into this
    module's docstring (the R5 per-bin dump) before this gate was written.
    """
    for name, probe in ref_planes.items():
        ref_acc = np.asarray(probe.accumulator, dtype=np.complex128)
        got_acc = np.asarray(vmap_planes[name], dtype=np.complex128)
        assert ref_acc.shape == got_acc.shape, (
            f"DFT plane {name!r} shape mismatch ({ctx}): "
            f"{got_acc.shape} vs {ref_acc.shape}"
        )
        for fi in range(ref_acc.shape[0]):
            mag_ref = np.max(np.abs(ref_acc[fi]))
            diff = np.max(np.abs(got_acc[fi] - ref_acc[fi]))
            assert mag_ref > 0.0, (
                f"DFT plane {name!r} freq bin {fi} is all-zero in the "
                f"reference ({ctx}) -- fixture does not exercise signal "
                "at this bin, tolerance check is meaningless"
            )
            rel = diff / mag_ref
            assert rel < rtol, (
                f"DFT plane {name!r} freq bin {fi} mismatch ({ctx}): "
                f"rel={rel:.3e} >= rtol={rtol:.3e} "
                f"(|ref|_max={mag_ref:.3e}, maxdiff={diff:.3e})"
            )


class TestVmapDftPlaneFastPath:
    """Fast-path (vmap-batched) DFT accumulator vs. direct sim.run(),
    per swept element -- the comparator-class equivalence gate."""

    def test_dft_plane_matches_run_cpml(self):
        eps_values = np.array([2.0, 6.0])
        n_steps = 60

        vmap_res = vmap_material_sweep(
            _dft_sim("cpml", eps_r=4.0), "substrate.eps_r", eps_values,
            n_steps=n_steps,
        )
        assert vmap_res.dft_planes is not None
        assert set(vmap_res.dft_planes.keys()) == {"p1"}
        # axis="x" plane -> (n1, n2) = (Ny, Nz) INCLUDING CPML padding;
        # not hand-computed here (grid padding is an implementation detail)
        # -- cross-checked against the shape run() itself produces below.
        ref0 = _dft_sim("cpml", eps_r=float(eps_values[0])).run(n_steps=n_steps)
        expected_shape = (len(eps_values),) + tuple(
            ref0.dft_planes["p1"].accumulator.shape)
        assert vmap_res.dft_planes["p1"].shape == expected_shape

        for idx, ev in enumerate(eps_values):
            ref = _dft_sim("cpml", eps_r=float(ev)).run(n_steps=n_steps)
            _assert_dft_planes_match(
                {"p1": vmap_res.dft_planes["p1"][idx]},
                ref.dft_planes, rtol=2e-3, ctx=f"cpml eps_r={ev}",
            )

    def test_dft_plane_matches_run_pec(self):
        """Non-CPML (plain PEC-wall) boundary takes the OTHER
        ``_build_vmap_scan_fn`` branch (``use_cpml=False``) -- pin it too,
        it is a structurally different scan body."""
        eps_values = np.array([2.0, 6.0])
        n_steps = 60

        vmap_res = vmap_material_sweep(
            _dft_sim("pec", eps_r=4.0), "substrate.eps_r", eps_values,
            n_steps=n_steps,
        )
        for idx, ev in enumerate(eps_values):
            ref = _dft_sim("pec", eps_r=float(ev)).run(n_steps=n_steps)
            _assert_dft_planes_match(
                {"p1": vmap_res.dft_planes["p1"][idx]},
                ref.dft_planes, rtol=2e-3, ctx=f"pec eps_r={ev}",
            )

    def test_dft_plane_dtype_matches_run(self):
        """#477/#484 x64-contract pin: the vmap accumulator dtype must be
        the SAME complex dtype run() would produce (complex64 with x64
        off, the default here)."""
        vmap_res = vmap_material_sweep(
            _dft_sim("cpml"), "substrate.eps_r", np.array([2.0, 4.0]),
            n_steps=20,
        )
        ref = _dft_sim("cpml").run(n_steps=20)
        assert vmap_res.dft_planes["p1"].dtype == ref.dft_planes["p1"].accumulator.dtype

    def test_no_dft_plane_registered_leaves_dft_planes_none(self):
        """Control: a sim with no add_dft_plane_probe call must NOT
        fabricate a dft_planes dict."""
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.02),
                          boundary="cpml", cpml_layers=6, dx=0.002)
        sim.add_material("substrate", eps_r=4.0)
        sim.add(Box((0.005, 0, 0), (0.015, 0.02, 0.02)), material="substrate")
        sim.add_source((0.01, 0.01, 0.01), "ez", waveform=GaussianPulse(f0=3e9))
        sim.add_probe((0.005, 0.01, 0.01), "ez")

        res = vmap_material_sweep(sim, "substrate.eps_r", np.array([2.0, 4.0]),
                                   n_steps=20)
        assert res.dft_planes is None


class TestVmapDftPlaneX64:
    """#477/#484 x64-contract pin, scoped (never module-level, per repo
    rule): under ``jax_enable_x64``, ``init_dft_plane_probe`` (called
    identically by both the vmap fast path and ``run()``) selects
    ``complex128`` instead of ``complex64`` (rfx/probes/probes.py:441).
    This is a separate axis from the default-precision equivalence tests
    above -- confirm the promoted dtype AND that equivalence still holds
    (same ~5e-4 relative floor observed at default precision; x64 widens
    the accumulator's storage precision but the underlying field state is
    still produced by the same float32-pinned Yee kernels, so the
    numerical agreement floor does not tighten)."""

    def test_dft_plane_matches_run_cpml_x64(self):
        eps_values = np.array([2.0, 6.0])
        n_steps = 60

        with _enable_x64(True):
            vmap_res = vmap_material_sweep(
                _dft_sim("cpml", eps_r=4.0), "substrate.eps_r", eps_values,
                n_steps=n_steps,
            )
            assert vmap_res.dft_planes["p1"].dtype == np.complex128

            for idx, ev in enumerate(eps_values):
                ref = _dft_sim("cpml", eps_r=float(ev)).run(n_steps=n_steps)
                assert ref.dft_planes["p1"].accumulator.dtype == np.complex128
                _assert_dft_planes_match(
                    {"p1": vmap_res.dft_planes["p1"][idx]},
                    ref.dft_planes, rtol=2e-3, ctx=f"x64 cpml eps_r={ev}",
                )


class TestVmapDftPlaneFallbackCarries:
    """#578: the sequential fallback must ALSO populate ``.dft_planes`` —
    no path asymmetry (design v2 PR B.1)."""

    def test_upml_fallback_carries_dft_planes(self):
        """UPML (not fast-path eligible) + a registered DFT plane: the
        fallback must return the SAME accumulator run() would (exact
        match expected -- the fallback literally calls sim.run() per
        swept value, so this is not a numerical-tolerance comparator, it
        is the same call twice)."""
        eps_values = np.array([2.0, 6.0])
        n_steps = 20

        with pytest.warns(UserWarning, match=_FALLBACK_MATCH):
            res = vmap_material_sweep(
                _dft_sim("upml", eps_r=4.0), "substrate.eps_r", eps_values,
                n_steps=n_steps,
            )
        assert res.dft_planes is not None
        assert res.dft_planes["p1"].shape[0] == len(eps_values)

        for idx, ev in enumerate(eps_values):
            ref = _dft_sim("upml", eps_r=float(ev)).run(n_steps=n_steps)
            npt.assert_allclose(
                res.dft_planes["p1"][idx],
                np.asarray(ref.dft_planes["p1"].accumulator),
                rtol=1e-10, atol=1e-20,
                err_msg=f"fallback dft_planes should exactly reproduce "
                        f"run() at eps_r={ev} (same call)",
            )


class TestVmapPortFamilyEligibility:
    """MSL/floquet/coaxial ports are separate registries from
    ``sim._ports`` (lumped/soft-source) and were never consulted by the
    fast-path eligibility guard before #578 -- a genuine silent-drop gap
    for MSL/floquet (run()-consumed) and an honesty gap for coaxial (not
    consumed by plain run() at all). These pins belong here rather than
    ``test_vmap_sweep_eligibility.py`` because they combine port families
    with a registered DFT plane to also exercise the fallback-carries-DFT
    path for a non-UPML fallback reason."""

    def test_floquet_port_takes_sequential_fallback(self):
        Lx, Ly, Lz = 0.015, 0.015, 0.03
        sim = Simulation(freq_max=15e9, domain=(Lx, Ly, Lz), boundary="cpml",
                          cpml_layers=8, dx=0.001)
        sim.add_material("substrate", eps_r=2.2)
        sim.add(Box((0, 0, Lz / 2 - 0.001), (Lx, Ly, Lz / 2)), material="substrate")
        patch_w = 0.008
        x0 = (Lx - patch_w) / 2
        y0 = (Ly - patch_w) / 2
        sim.add(Box((x0, y0, Lz / 2), (x0 + patch_w, y0 + patch_w, Lz / 2 + 0.001)),
                material="pec")
        sim.add_floquet_port(Lz * 0.25, axis="z", scan_theta=0.0, scan_phi=0.0,
                              polarization="te", n_freqs=10)
        sim.add_probe((Lx / 2, Ly / 2, Lz / 2 + 0.002), component="ex")

        with pytest.warns(UserWarning, match=_FALLBACK_MATCH):
            res = vmap_material_sweep(
                sim, "substrate.eps_r", np.array([2.0, 3.0]), n_steps=20,
            )
        assert res.time_series.shape[0] == 2

    def test_msl_port_takes_sequential_fallback(self):
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.01, 0.006),
                          boundary="cpml", cpml_layers=6, dx=0.001)
        sim.add_material("sub", eps_r=4.0)
        sim.add(Box((0, 0, 0), (0.02, 0.01, 0.002)), material="sub")
        sim.add(Box((0, 0.004, 0.002), (0.02, 0.006, 0.003)), material="pec")
        sim.add_msl_port(position=(0.003, 0.005, 0.0), width=0.002,
                          height=0.002, direction="+x", impedance=50.0)

        with pytest.warns(UserWarning, match=_FALLBACK_MATCH):
            res = vmap_material_sweep(
                sim, "sub.eps_r", np.array([2.0, 3.0]), n_steps=15,
            )
        assert res.time_series.shape[0] == 2

    def test_coaxial_port_takes_sequential_fallback_and_fails_loud(self):
        """Coaxial ports are not consumed by plain ``sim.run()`` at all
        (pre-existing NotImplementedError there); before #578 the vmap
        fast-path eligibility guard did not check ``_coaxial_ports``, so
        a coax-port sim silently took the fast path and just ignored the
        port (no error, wrong/incomplete physics). Post-#578 it routes to
        the sequential fallback, which surfaces run()'s existing honest
        NotImplementedError instead of a silent no-op -- this is the
        'honesty claim distinguishes the families' guard from the design
        doc, verified by an actual raise, not just a routing check."""
        sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02),
                          boundary="pec", dx=0.002)
        sim.add_material("sub", eps_r=4.0)
        sim.add(Box((0.005, 0, 0), (0.015, 0.02, 0.02)), material="sub")
        sim.add_coaxial_port((0.010, 0.010, 0.015), face="top")
        sim.add_source((0.01, 0.01, 0.005), "ez", waveform=GaussianPulse(f0=3e9))
        sim.add_probe((0.005, 0.01, 0.01), "ez")

        with pytest.warns(UserWarning, match=_FALLBACK_MATCH):
            with pytest.raises(NotImplementedError, match="add_coaxial_port"):
                vmap_material_sweep(
                    sim, "sub.eps_r", np.array([2.0, 3.0]), n_steps=15,
                )


class TestVmapAmplitudeKindCurrent:
    """PR #617 (issue #571) added ``amplitude_kind='current'`` soft
    sources with a dynamic per-batch Cb-normalization path in
    ``_build_vmap_scan_fn`` (rfx/vmap_sweep.py:450-468) -- until this
    file, no test exercised it at the ``vmap_material_sweep`` level (the
    #617 coverage gap named in the #578 design doc). Tolerance is looser
    than the DFT-plane gate above: at n_steps=20 the observed relative
    floor between the vmap dynamic-Cb path and run()'s equivalent is
    ~5e-5 (grows with step count as the dielectric-filled small CPML
    domain accumulates reflections -- ordinary float32 FDTD chaos, not a
    phase/sign defect; see the module docstring for the falsifier
    argument)."""

    def test_amplitude_kind_current_matches_run_cpml(self):
        eps_values = np.array([2.0, 6.0])
        n_steps = 20

        vmap_res = vmap_material_sweep(
            _dft_sim("cpml", eps_r=4.0, amplitude_kind="current"),
            "substrate.eps_r", eps_values, n_steps=n_steps,
        )
        for idx, ev in enumerate(eps_values):
            ref = _dft_sim(
                "cpml", eps_r=float(ev), amplitude_kind="current",
            ).run(n_steps=n_steps)
            npt.assert_allclose(
                vmap_res.time_series[idx],
                np.asarray(ref.time_series, dtype=np.float64),
                rtol=1e-3, atol=1e-6,
                err_msg=f"amplitude_kind='current' vmap-vs-run at eps_r={ev}",
            )

    def test_amplitude_kind_current_matches_run_pec(self):
        eps_values = np.array([2.0, 6.0])
        n_steps = 20

        vmap_res = vmap_material_sweep(
            _dft_sim("pec", eps_r=4.0, amplitude_kind="current"),
            "substrate.eps_r", eps_values, n_steps=n_steps,
        )
        for idx, ev in enumerate(eps_values):
            ref = _dft_sim(
                "pec", eps_r=float(ev), amplitude_kind="current",
            ).run(n_steps=n_steps)
            npt.assert_allclose(
                vmap_res.time_series[idx],
                np.asarray(ref.time_series, dtype=np.float64),
                rtol=1e-3, atol=1e-6,
                err_msg=f"amplitude_kind='current' vmap-vs-run (pec) at eps_r={ev}",
            )


class TestVmapReturnFieldsRaise:
    """#578: return_fields=True was documented-but-never-implemented
    (VmapSweepResult.final_fields was never populated) -- it now raises
    ValueError instead of silently returning final_fields=None."""

    def test_return_fields_true_raises(self):
        sim = _dft_sim("cpml")
        with pytest.raises(ValueError, match="return_fields=True"):
            vmap_material_sweep(
                sim, "substrate.eps_r", np.array([2.0, 4.0]), n_steps=10,
                return_fields=True,
            )

    def test_return_fields_false_is_unaffected(self):
        """Control: the default (False) must not raise and must behave
        exactly as before."""
        sim = _dft_sim("cpml")
        res = vmap_material_sweep(
            sim, "substrate.eps_r", np.array([2.0, 4.0]), n_steps=10,
            return_fields=False,
        )
        assert isinstance(res, VmapSweepResult)
        assert res.final_fields is None
