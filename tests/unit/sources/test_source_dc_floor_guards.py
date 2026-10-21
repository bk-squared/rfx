"""Soft-source deposited-DC guards (issues #386 / #388).

Measured mechanism (issue #388, 2026-07-19, both uniform and NU lanes):
a soft current source deposits its waveform's discrete time integral as a
net static end-charge ``q = S/((1+loss)*dz_src)``, ``S = sum s(t_n)*dt``.
The remnant's discrete-Poisson self-energy neither decays nor is absorbed
by CPML, so it floors the ``until_decay`` interior-energy criterion.

Locks, in order:
1-2. ``GaussianPulse`` (issue 1668) starts at zero, leaves no DC on the
   float32 table at cutoff 3, 4.5 and 6, differs from the pulse before
   issue 1668 by at most 1.7e-3 of its amplitude, and its float32 table
   matches a float64 evaluation of the closed form within 4 c^2 eps32 relative
   over the first half tau. (Until issue 1668: byte identity with the
   pre-cutoff formula, and the e^-(c^2) law of its DC.)
3. The run()-path ``until_decay`` DC-floor warning fires exactly per the
   predeclared four-case matrix (issue #388 measured rel_DC values at the
   demo dt = 1.2 ps: 6.0e-5 / ~e-10 / 0.68 / 4.0e-4 vs threshold 1e-3).
4. A pulse given its bandwidth in Hz samples to zero at every step and is
   refused (issue 1668); the ``unresolved_pulse`` advisory (issue #386)
   fires on the TF/SF absolute-Hz footgun and stays silent on the
   demo-class fractional-bandwidth configs.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx import Simulation, GaussianPulse, ModulatedGaussian
from rfx.measurement.decay import warn_dc_floor

DEMO_DT = 1.2e-12  # the issue-#388 measurement dt (patch_antenna_demo scale)


# ---------------------------------------------------------------------------
# 1-2. GaussianPulse properties (issue 1668)
#
# The expectations come from the closed form evaluated here in float64 NumPy,
# never from GaussianPulse itself:
#   s(t) = A [-2x e^(-x^2) - e^(-c^2) e^(-u^2) (2c + (4c^2-2)u - 4cu^2 - 4c^2u^3)]
#   for t >= 0 and 0 before, x = (t - c tau)/tau, u = t/tau, tau = 1/(pi f0 B).
# The pulse before issue 1668 was the first term alone, A (-2x) e^(-x^2).
# ---------------------------------------------------------------------------

_F0, _BW = 2.2e9, 1.2
_TAU = 1.0 / (math.pi * _F0 * _BW)
_CUTOFFS = (3.0, 4.5, 6.0)


def _pulse64(t, cutoff, amplitude=1.0):
    t = np.asarray(t, dtype=np.float64)
    x, u, c = t / _TAU - cutoff, t / _TAU, cutoff
    onset = np.exp(-c * c - u * u) * (
        2 * c + (4 * c * c - 2) * u - 4 * c * u * u - 4 * c * c * u ** 3)
    return np.where(t >= 0, amplitude * (-2 * x * np.exp(-x * x) - onset), 0.0)


def _old_pulse64(t, cutoff, amplitude=1.0):
    x = np.asarray(t, dtype=np.float64) / _TAU - cutoff
    return amplitude * (-2 * x * np.exp(-x * x))


def _table(pulse, n):
    """The float32 table a drive samples: ``pulse(float32(n) * dt)``."""
    times = jnp.arange(n, dtype=jnp.float32) * DEMO_DT
    values = np.asarray(jax.vmap(pulse)(times))
    assert values.dtype == np.float32
    return values.astype(np.float64), np.asarray(times, dtype=np.float64)


def test_gaussian_pulse_has_no_nan_far_from_its_pulse():
    """A tau of 5e-20 s evaluated up to 1e-6 s (u ~ 2e13; preflight would
    refuse such a drive, the class itself must not return NaN): every table
    value is finite, and the gradient is finite far before and after."""
    tiny = GaussianPulse(f0=5e9, bandwidth=1.0 / (math.pi * 5e9 * 5e-20))
    times = jnp.linspace(0.0, 1e-6, 10_001, dtype=jnp.float32)
    assert np.all(np.isfinite(np.asarray(jax.vmap(tiny)(times))))
    for pulse in (tiny, GaussianPulse(f0=5e9, bandwidth=0.8)):
        for t in (1e-6, -1e-6, 1e-3, -1e-3):
            assert np.isfinite(float(jax.grad(pulse)(jnp.float32(t)))), (pulse, t)


def test_gaussian_pulse_refuses_a_cutoff_below_2p5():
    """Below c = 2.5 the charge-free start changes the pulse by more than 2 %
    (largest added term 1.84e-2 at c = 2.5, 2.8e-2 at 2.4; closed form below)."""
    u = np.linspace(0.0, 8.0, 400_001)
    def added(c):
        return np.max(np.abs(np.exp(-c * c - u * u) * (
            2 * c + (4 * c * c - 2) * u - 4 * c * u * u - 4 * c * c * u ** 3)))
    assert added(2.5) < 2e-2 < added(2.4)
    GaussianPulse(f0=_F0, bandwidth=_BW, cutoff=2.5)
    for bad in (2.49, 1.0, 0.0, -3.0, float("nan")):
        with pytest.raises(ValueError, match="cutoff"):
            GaussianPulse(f0=_F0, bandwidth=_BW, cutoff=bad)


@pytest.mark.parametrize("cutoff", _CUTOFFS)
def test_gaussian_pulse_starts_at_zero(cutoff):
    """s(0) = 0 exactly, evaluated as a scalar and as a table."""
    pulse = GaussianPulse(f0=_F0, bandwidth=_BW, cutoff=cutoff)
    assert float(pulse(0.0)) == 0.0
    for n in (1, 4):
        assert _table(pulse, n)[0][0] == 0.0


@pytest.mark.parametrize("cutoff", _CUTOFFS)
def test_gaussian_pulse_float32_table_has_no_dc(cutoff):
    """|sum s(t_n) dt| / tau <= 1e-7 on the float32 table over the whole pulse.

    The old pulse left e^(-c^2) tau there (1.2e-4 at c = 3), which a soft
    source deposits as a static charge.
    """
    pulse = GaussianPulse(f0=_F0, bandwidth=_BW, cutoff=cutoff)
    n = int(math.ceil((cutoff + 6.0) * _TAU / DEMO_DT)) + 1
    values, _ = _table(pulse, n)
    dc = abs(float(np.sum(values))) * DEMO_DT / _TAU
    assert dc <= 1e-7, f"cutoff {cutoff}: |sum s dt|/tau = {dc:.3e}"


@pytest.mark.parametrize("cutoff", _CUTOFFS)
def test_gaussian_pulse_changes_the_old_pulse_by_at_most_1p7e_3(cutoff):
    """max |s - s_old| <= 1.7e-3 A over the whole pulse (A = 2.5 here)."""
    amplitude = 2.5
    pulse = GaussianPulse(f0=_F0, bandwidth=_BW, cutoff=cutoff,
                          amplitude=amplitude)
    n = int(math.ceil((cutoff + 6.0) * _TAU / DEMO_DT)) + 1
    values, times = _table(pulse, n)
    change = np.max(np.abs(values - _old_pulse64(times, cutoff, amplitude)))
    assert change <= 1.7e-3 * amplitude, (
        f"cutoff {cutoff}: max |s - s_old| = {change / amplitude:.3e} A")


@pytest.mark.parametrize("cutoff", _CUTOFFS)
def test_gaussian_pulse_float32_table_matches_float64_near_the_start(cutoff):
    """Over u < 1/2 the float32 table is within 4 c^2 eps32 of the float64
    closed form at the same times, relative to it, sample by sample (t = 0 is
    the zero of the previous test)."""
    pulse = GaussianPulse(f0=_F0, bandwidth=_BW, cutoff=cutoff)
    n = int(math.ceil(0.5 * _TAU / DEMO_DT))
    values, times = _table(pulse, n)
    want = _pulse64(times, cutoff)
    rel = np.abs(values[1:] - want[1:]) / np.abs(want[1:])
    worst = int(np.argmax(rel)) + 1
    # exp(-x^2) multiplies the argument's rounding by x^2 (the old formula reads 4.35e-6 at c = 4.5).
    bar = 4 * cutoff ** 2 * float(np.finfo(np.float32).eps)
    assert rel.max() <= bar, (
        f"cutoff {cutoff}: relative error {rel.max():.3e} (bar {bar:.2e}) at u = "
        f"{times[worst] / _TAU:.3f} (float32 {values[worst]:.6e}, "
        f"float64 {want[worst]:.6e})")


# ---------------------------------------------------------------------------
# 3. until_decay DC-floor warning — predeclared four-case matrix (#388)
# ---------------------------------------------------------------------------

def _dc_floor_fires(waveform, n_table=50_000) -> list[str]:
    sim = Simulation(freq_max=4e9, domain=(0.02, 0.02, 0.02), dx=2e-3,
                     cpml_layers=4)
    sim.add_source((0.01, 0.01, 0.01), "ez", waveform=waveform)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_dc_floor(sim, dt=DEMO_DT, n_table=n_table)
    return [str(w.message) for w in caught if "issue #388" in str(w.message)]


@pytest.mark.parametrize(
    "label, waveform, expect_fire",
    [
        # measured rel_DC 6.0e-5 < 1e-3
        ("gaussian_cutoff3",
         GaussianPulse(f0=2.2e9, bandwidth=1.2), False),
        # ~5 decades below the cutoff=3 value
        ("gaussian_cutoff45",
         GaussianPulse(f0=2.2e9, bandwidth=1.2, cutoff=4.5), False),
        # measured rel_DC 0.68 (DC suppression e^-1/bw^2 ~ 0.5 at bw=1.2)
        ("modgaussian_bw12",
         ModulatedGaussian(f0=2.2e9, bandwidth=1.2), True),
        # measured rel_DC 4.0e-4: above the 1e-6 threshold (it was 1e-3 until the
        # after-drive reference, under which 6e-5 already blocked the stop)
        ("modgaussian_bw04",
         ModulatedGaussian(f0=2.2e9, bandwidth=0.4), True),
    ],
)
def test_until_decay_dc_floor_matrix(label, waveform, expect_fire):
    fired = _dc_floor_fires(waveform)
    if expect_fire:
        assert fired, f"{label}: expected the #388 DC-floor warning"
        # Quantitative: the message must carry the rel_DC number.
        assert "relative DC" in fired[0] and "cap-hit" in fired[0]
        # All matrix fire-cases run a 50k-step table that covers the
        # full pulse — they must take the INTRINSIC-DC branch (with its
        # cutoff/bandwidth remedies AND the honesty note about
        # thin-source-cell-amplified floors this advisory cannot see),
        # not the truncation branch.
        assert "truncation-dominated" not in fired[0]
        assert "thin source cells" in fired[0]
        # #388: the intrinsic-DC advisory must route to the purpose-built
        # remedy (the radiated-flux stop criterion), not just the weaker
        # reduce-DC / fixed-n_steps options.
        assert "radiated_flux_box" in fired[0], (
            f"{label}: DC-floor warning should point to the flux-stop remedy")
    else:
        assert not fired, f"{label}: unexpected #388 warning: {fired}"


def test_static_remnant_cap_hit_points_to_flux_stop():
    """#388 post-run advisory: on a cap-hit with an electrostatic remnant (E-only, H≈0),
    the warning must route to the purpose-built flux-stop remedy (radiated_flux_box=)."""
    from rfx.measurement.decay import static_remnant_advisory as _warn_static_remnant_cap_hit
    from rfx.grid import Grid
    from rfx.core.yee import init_state, init_materials

    grid = Grid(freq_max=4e9, domain=(0.02, 0.02, 0.02), dx=2e-3, cpml_layers=4)
    materials = init_materials(grid.shape)
    # electrostatic remnant: an interior E-only field, H identically zero (H-share → 0)
    state = init_state(grid.shape)
    cx, cy, cz = grid.nx // 2, grid.ny // 2, grid.nz // 2
    state = state._replace(ez=state.ez.at[cx, cy, cz].set(1.0))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_static_remnant_cap_hit(state, materials, grid)
    msgs = [str(w.message) for w in caught if "issue #388" in str(w.message)]
    assert msgs, "electrostatic-remnant cap-hit should warn"
    assert "radiated_flux_box" in msgs[0], (
        "the cap-hit advisory should point to the flux-stop remedy")


def test_until_decay_dc_floor_wired_into_run():
    """run(until_decay=...) emits the #388 warning; a fixed-n_steps run
    of the same sim does not (the check binds to until_decay only)."""
    def _sim():
        sim = Simulation(freq_max=36e9, domain=(6e-3, 6e-3, 6e-3),
                         dx=1e-3, cpml_layers=4, boundary="cpml")
        sim.add_source((3e-3, 3e-3, 3e-3), "ez",
                       waveform=ModulatedGaussian(f0=30e9, bandwidth=1.2))
        return sim

    with pytest.warns(UserWarning, match="issue #388"):
        _sim().run(until_decay=1e-3, decay_min_steps=50,
                   decay_max_steps=300)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _sim().run(n_steps=20)
    assert not [w for w in caught if "issue #388" in str(w.message)], (
        "#388 DC-floor warning must fire only when until_decay is set"
    )


# ---------------------------------------------------------------------------
# 3b. truncation-aware branch (post-#392 review): rel_DC over the planned
#     table conflates intrinsic waveform DC with table-truncation DC.
# ---------------------------------------------------------------------------

def test_dc_floor_truncation_branch_when_table_ends_inside_pulse():
    """A 200-step table at the demo dt (0.24 ns) ends inside the default
    pulse (completion t0+3*tau ~ 0.48 ns at f0=5 GHz, bw=0.8): the high
    rel_DC (~1.0) is pure truncation artifact. The message must say so,
    recommend a longer decay_max_steps / fixed n_steps, and must NOT
    recommend raising the cutoff (that lengthens the onset and makes
    truncation worse)."""
    fired = _dc_floor_fires(GaussianPulse(f0=5e9, bandwidth=0.8),
                            n_table=200)
    assert fired, "truncated-table high rel_DC must still warn"
    msg = fired[0]
    assert "truncation-dominated" in msg
    assert "decay_max_steps" in msg and "n_steps" in msg
    assert "WORSE" in msg
    assert "higher GaussianPulse cutoff" not in msg


def test_dc_floor_table_ending_before_t0_plus_4p5_tau_is_truncation():
    """A 400-step table at the demo dt (0.48 ns, 6.03*tau at f0=5 GHz,
    bw=0.8) ends before t0 + 4.5*tau = 7.5*tau, where the differentiated
    Gaussian is still above 1.4e-8 of its amplitude: its relative DC (5.3e-5)
    is the cut tail, so the message is the truncation one, not the static
    charge one."""
    fired = _dc_floor_fires(GaussianPulse(f0=5e9, bandwidth=0.8), n_table=400)
    assert fired and "truncation-dominated" in fired[0], fired
    assert "static charge" not in fired[0]


def test_dc_floor_truncation_worsens_with_higher_cutoff():
    """Raising cutoff makes TRUNCATION worse: at n_table=480 (0.576 ns
    table, 7.2*tau, f0=5 GHz, bw=0.8) cutoff=3 is silent at the 1e-6
    threshold (the table runs 4.2*tau past t0; at 400 steps, 3.03*tau past
    t0, the cut tail alone read 5.3e-5) while cutoff=4.5 pushes completion
    to 7.5*tau ~ 0.60 ns past the table end and fires the truncation
    branch."""
    silent = _dc_floor_fires(GaussianPulse(f0=5e9, bandwidth=0.8),
                             n_table=480)
    assert not silent, f"cutoff=3 must be silent at n_table=480: {silent}"
    fired = _dc_floor_fires(
        GaussianPulse(f0=5e9, bandwidth=0.8, cutoff=4.5), n_table=480)
    assert fired, "cutoff=4.5 must fire at n_table=480 (longer onset)"
    assert "truncation-dominated" in fired[0]


def test_dc_floor_covers_msl_ports():
    """MSL ports launch via the same soft-Ez deposit mechanism; a
    high-DC waveform on an MSL port must fire the #388 advisory
    (post-#392 review; entry access mirrors the #386 validator)."""
    sim = Simulation(freq_max=4e9, domain=(0.02, 0.02, 0.02), dx=2e-3,
                     cpml_layers=4)
    sim.add_msl_port((0.004, 0.01, 0.002), width=4e-3, height=2e-3,
                     waveform=ModulatedGaussian(f0=2.2e9, bandwidth=1.2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_dc_floor(sim, dt=DEMO_DT, n_table=50_000)
    fired = [str(w.message) for w in caught
             if "issue #388" in str(w.message)]
    assert fired, "MSL-port high-DC waveform must fire the #388 warning"
    assert "relative DC" in fired[0]


def test_dc_floor_silent_on_passive_msl_port():
    """A passive (excite=False) MSL port is a matched termination — it
    deposits no source waveform, so the #388 advisory must stay silent
    even with a high-DC waveform registered on the entry."""
    sim = Simulation(freq_max=4e9, domain=(0.02, 0.02, 0.02), dx=2e-3,
                     cpml_layers=4)
    sim.add_msl_port((0.004, 0.01, 0.002), width=4e-3, height=2e-3,
                     excite=False,
                     waveform=ModulatedGaussian(f0=2.2e9, bandwidth=1.2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_dc_floor(sim, dt=DEMO_DT, n_table=50_000)
    assert not [w for w in caught if "issue #388" in str(w.message)], (
        "excite=False MSL port must not fire the #388 DC-floor warning")


# ---------------------------------------------------------------------------
# 3c. add_polarized_source must thread the waveform cutoff (post-#392
#     review): the real-Jones reconstruction rebuilds GaussianPulse and
#     previously dropped cutoff — silently reverting the #392 remedy.
# ---------------------------------------------------------------------------

def test_polarized_source_threads_cutoff():
    sim = Simulation(freq_max=4e9, domain=(0.02, 0.02, 0.02), dx=2e-3,
                     cpml_layers=4)
    wf = GaussianPulse(f0=2.2e9, bandwidth=1.2, cutoff=4.5)
    sim.add_polarized_source((0.01, 0.01, 0.01), polarization="slant45",
                             waveform=wf)
    comps = {pe.component: pe.waveform for pe in sim._ports}
    assert set(comps) == {"ex", "ey"}
    for comp, w in comps.items():
        assert w.cutoff == 4.5, (
            f"{comp} component waveform lost cutoff (got {w.cutoff}) — "
            f"the #392 remedy is silently ineffective on this path")
        assert w.f0 == wf.f0 and w.bandwidth == wf.bandwidth


# ---------------------------------------------------------------------------
# 3d. decay-parameter sanity advisories (post-#392 review; behaviour-
#     neutral, single run()-level site covering both honoured lanes).
# ---------------------------------------------------------------------------

def _tiny_decay_sim():
    sim = Simulation(freq_max=36e9, domain=(6e-3, 6e-3, 6e-3),
                     dx=1e-3, cpml_layers=4, boundary="cpml")
    sim.add_source((3e-3, 3e-3, 3e-3), "ez")
    return sim


def test_decay_min_steps_above_max_steps_warns():
    """decay_min_steps > decay_max_steps: every check index lies past
    the table end — zero energy checks, forced decay_max_steps run."""
    with pytest.warns(UserWarning, match="zero energy checks"):
        _tiny_decay_sim().run(until_decay=1e-3, decay_min_steps=80,
                              decay_max_steps=40)


def test_until_decay_at_or_above_one_warns():
    """until_decay >= 1: U < until_decay*peak_U is met while the energy
    is still rising — the stop can fire immediately."""
    with pytest.warns(UserWarning, match="still rising"):
        _tiny_decay_sim().run(until_decay=1.5, decay_min_steps=20,
                              decay_max_steps=60)


def test_sane_decay_params_do_not_warn():
    """The advisories are precise: a sane parameter set emits neither."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _tiny_decay_sim().run(until_decay=1e-3, decay_min_steps=20,
                              decay_max_steps=60)
    msgs = [str(w.message) for w in caught]
    assert not any("zero energy checks" in m or "still rising" in m
                   for m in msgs), msgs


def test_until_decay_zero_forced_n_escape_stays_silent():
    """until_decay=0.0 is the documented forced-N escape (U < 0 never
    fires; the run executes exactly decay_max_steps): the run()-level
    sanity advisories must stay silent on it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _tiny_decay_sim().run(until_decay=0.0, decay_min_steps=20,
                              decay_max_steps=60)
    msgs = [str(w.message) for w in caught]
    assert not any("zero energy checks" in m or "still rising" in m
                   for m in msgs), msgs


# ---------------------------------------------------------------------------
# 3e. MEASURED static-remnant cap-hit advisory (#388 candidate 3). Complements the
#     pre-run waveform-DC advisory: it fires on the ACTUAL run outcome (final-state
#     physical E/H energy share), catching thin-source-cell floors the pre-run
#     advisory cannot predict. Physical energy (½εE² vs ½μH²) separates a static
#     electrostatic remnant (H-share ~1e-10) from a still-ringing field (H-share ~0.2+).
# ---------------------------------------------------------------------------

def _run_capture(sim, **run_kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.run(**run_kw)
    fired = [str(w.message) for w in caught
             if "issue #388" in str(w.message) and "H-share" in str(w.message)]
    return len(res.time_series), fired


def _soft_ez_sim(waveform):
    sim = Simulation(freq_max=4e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                     cpml_layers=4, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.01), "ez", waveform=waveform)
    sim.add_probe((0.006, 0.006, 0.006), "ez")
    return sim


def test_static_remnant_cap_hit_advisory_fires():
    """A high-DC soft source cap-hits (energy floors); the measured advisory fires."""
    n, fired = _run_capture(_soft_ez_sim(ModulatedGaussian(f0=2.2e9, bandwidth=1.2)),
                            until_decay=1e-3, decay_min_steps=200, decay_max_steps=1500)
    assert n == 1500, "high-DC soft source must cap-hit (criterion cannot self-terminate)"
    assert fired, "measured #388 static-remnant advisory must fire on the cap-hit"
    assert "electrostatic" in fired[0] and "H-share" in fired[0]
    # The drive that blocks the stop is named with its class and relative DC.
    assert "ModulatedGaussian, relative DC" in fired[0], fired[0]


def test_static_remnant_advisory_silent_on_settled_run():
    """A low-DC source settles (the criterion fires) — no cap-hit, no advisory."""
    n, fired = _run_capture(_soft_ez_sim(GaussianPulse(f0=2.2e9, bandwidth=0.8)),
                            until_decay=1e-3, decay_min_steps=200, decay_max_steps=1500)
    assert n < 1500, "low-DC source should settle before the cap"
    assert not fired, f"advisory must stay silent on a settled run: {fired}"


def test_static_remnant_advisory_fires_on_nonuniform_lane():
    """The NU (dz_profile) lane floors the same way (#388 measured both lanes)."""
    dz = np.array([0.4e-3] * 4 + [0.5e-3] * 6)
    sim = Simulation(freq_max=4e9, domain=(0.02, 0.02, 0.01), dx=1e-3, dz_profile=dz,
                     cpml_layers=4, boundary="cpml")
    sim.add_source((0.01, 0.01, 0.003), "ez",
                   waveform=ModulatedGaussian(f0=2.2e9, bandwidth=1.2))
    sim.add_probe((0.006, 0.006, 0.003), "ez")
    n, fired = _run_capture(sim, until_decay=1e-3, decay_min_steps=200, decay_max_steps=1500)
    assert n == 1500 and fired, f"NU cap-hit static-remnant advisory must fire (steps={n})"


def test_static_remnant_advisory_silent_when_field_balanced():
    """False-fire guard (the 2f-slosh concern): a genuine still-ringing cap-hit has a
    BALANCED physical E/H share (H-share ~0.5), NOT a static remnant. Directly exercise
    the detector on synthetic states so the threshold logic is pinned without a fixture.
    """
    from rfx.measurement.decay import static_remnant_advisory as _warn_static_remnant_cap_hit
    from rfx.core.yee import init_state, init_materials, EPS_0, MU_0
    from rfx.grid import Grid

    grid = Grid(freq_max=4e9, domain=(0.02, 0.02, 0.02), dx=2e-3, cpml_layers=2)
    mats = init_materials(grid.shape)
    st = init_state(grid.shape)
    eta = math.sqrt(MU_0 / EPS_0)  # |E| = eta*|H| -> ½εE² == ½μH² (H-share 0.5)

    # (a) balanced propagating-like field: E=eta, H=1 -> H-share ~0.5 -> SILENT
    bal = st._replace(ex=st.ex + eta, hy=st.hy + 1.0)
    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter("always")
        _warn_static_remnant_cap_hit(bal, mats, grid)
    assert not [w for w in c if "issue #388" in str(w.message)], \
        "balanced E/H (still-ringing) must NOT trigger the static-remnant advisory"

    # (b) pure electrostatic remnant: E only, H=0 -> H-share 0 -> FIRES
    stat = st._replace(ex=st.ex + eta)
    with warnings.catch_warnings(record=True) as c:
        warnings.simplefilter("always")
        _warn_static_remnant_cap_hit(stat, mats, grid)
    assert [w for w in c if "issue #388" in str(w.message)], \
        "an E-only (electrostatic) remnant must trigger the advisory"


# ---------------------------------------------------------------------------
# 4. unresolved-pulse preflight advisory (#386)
# ---------------------------------------------------------------------------

def _demo_class_sim(bandwidth):
    """patch_antenna_demo scale: dx=2mm, freq_max=4GHz, f0=2.2GHz."""
    sim = Simulation(freq_max=4e9, domain=(0.06, 0.06, 0.03), dx=2e-3,
                     cpml_layers=4)
    sim.add_source((0.03, 0.03, 0.015), "ez",
                   waveform=GaussianPulse(f0=2.2e9, bandwidth=bandwidth))
    return sim


def test_undriven_pulse_is_refused_on_absolute_hz_bandwidth():
    """bandwidth=2.64e9 (absolute Hz for a 1.2 fractional intent) makes
    tau ~ 5e-20 s: every sample of the drive table is zero, so nothing would
    be driven. Preflight records a refusal and run() raises (issue 1668;
    before it this was the #386 advisory, driven only by the old pulse's
    jump at t=0)."""
    sim = _demo_class_sim(2.64e9)
    errors = [i for i in sim.preflight()
              if getattr(i, "severity", None) == "error"]
    assert [getattr(i, "code", None) for i in errors] == ["undriven_pulse"]
    assert "FRACTIONAL" in str(errors[0])
    with pytest.raises(ValueError, match="not resolved by dt"):
        sim.run(n_steps=20)


def test_unresolved_pulse_silent_on_demo_class_config():
    """The committed demo's config (fractional bandwidth=1.2; tau ~ 0.12ns
    vs dt ~ ps, ratio >> 3) must NOT fire."""
    codes = {getattr(i, "code", None)
             for i in _demo_class_sim(1.2).preflight()}
    assert "unresolved_pulse" not in codes


def _tfsf_sim(bandwidth):
    """TFSF carries a string waveform name; tau comes from f0/bandwidth."""
    sim = Simulation(freq_max=4e9, domain=(0.06, 0.06, 0.03), dx=2e-3,
                     cpml_layers=4, boundary="cpml")
    sim.add_tfsf_source(f0=2.2e9, bandwidth=bandwidth)
    return sim


def _hz_drive_sim(kind):
    """One drive of ``kind`` whose bandwidth is given in Hz (issue 1668)."""
    from rfx import Box
    hz = GaussianPulse(f0=5e9, bandwidth=4e9)
    if kind in ("source", "lumped", "wire", "passive"):
        sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                         boundary="cpml", cpml_layers=4)
        if kind == "source":
            sim.add_source((0.01, 0.01, 0.01), "ez", waveform=hz)
        elif kind == "lumped":
            sim.add_port((0.01, 0.01, 0.01), "ez", impedance=50., waveform=hz)
        elif kind == "wire":
            sim.add_port((0.01, 0.01, 0.009), "ez", impedance=50., extent=0.002,
                         waveform=hz)
        else:  # a terminated, not excited, port is not a drive
            sim.add_port((0.01, 0.01, 0.01), "ez", impedance=50., waveform=hz,
                         excite=False)
            sim.add_source((0.006, 0.01, 0.01), "ez",
                           waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
    elif kind == "msl":
        sim = Simulation(freq_max=3e9, domain=(.024, .024, .024), dx=.001,
                         boundary="pec", cpml_layers=0)
        sim.add_material("sub", eps_r=2.2)
        sim.add(Box((0., 0., 0.), (.024, .024, .003)), material="sub")
        sim.add(Box((0., .010, .003), (.024, .014, .003)), material="pec")
        sim.add_msl_port(position=(.006, .012, 0.), width=.004, height=.003,
                         direction="+x", impedance=50., waveform=hz,
                         eps_r_sub=2.2, n_probe_offset=3, n_probe_spacing=2,
                         n_probes=3)
    elif kind == "coax":
        sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=0.5e-3,
                         boundary="pec")
        sim.add_coaxial_port((0.01, 0.01, 0.02), face="top", waveform=hz)
    elif kind.startswith("waveguide"):
        sim = Simulation(freq_max=12e9, domain=(0.06, 0.02286, 0.01016),
                         dx=1e-3, boundary="cpml", cpml_layers=8)
        sim.add_waveguide_port(0.012, f0=10e9, bandwidth=5e9, direction="+x",
                               waveform=kind.split(":")[1])
    elif kind == "floquet":
        sim = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.004),
                         boundary="cpml", cpml_layers=6)
        sim.add_floquet_port(0.003, axis="z", scan_theta=0.0, f0=5e9,
                             bandwidth=4e9)
    elif kind == "tfsf":
        sim = _tfsf_sim(2.64e9)
    return sim


@pytest.mark.parametrize("kind", [
    "source", "lumped", "wire", "msl", "coax",
    "waveguide:differentiated_gaussian", "waveguide:modulated_gaussian",
    "floquet", "tfsf"])
def test_undriven_pulse_is_refused_for_every_drive_kind(kind):
    """A drive whose bandwidth is given in Hz is refused whatever carries it
    (issue 1668). A TF/SF source or waveguide port is judged by the waveform
    class of its waveform name, at its own f0 and bandwidth."""
    if kind == "waveguide:differentiated_gaussian":
        from rfx.model.waveguide_waveform import DG_REFUSAL

        with pytest.raises(ValueError) as caught:
            _hz_drive_sim(kind)
        assert str(caught.value) == DG_REFUSAL
        assert "#1942" in str(caught.value)
        return
    codes = [getattr(i, "code", None) for i in _hz_drive_sim(kind).preflight()
             if getattr(i, "severity", None) == "error"]
    assert "undriven_pulse" in codes, codes


def test_a_terminated_port_is_not_judged_as_a_drive():
    codes = [getattr(i, "code", None) for i in _hz_drive_sim("passive").preflight()]
    assert "undriven_pulse" not in codes and "unresolved_pulse" not in codes


def test_unresolved_pulse_silent_on_tfsf_fractional_bandwidth():
    codes = {getattr(i, "code", None)
             for i in _tfsf_sim(1.2).preflight()}
    assert "unresolved_pulse" not in codes


def test_cap_advisory_lists_lossy_media_but_not_conductors():
    """With no DC drive the remnant is named as charge relaxing in a lossy medium; a
    material whose eps/sigma is shorter than one step (a conductor) is not listed."""
    from rfx.measurement.decay import static_remnant_advisory
    from rfx.grid import Grid
    from rfx.core.yee import init_state, init_materials

    grid = Grid(freq_max=4e9, domain=(0.02, 0.02, 0.02), dx=2e-3, cpml_layers=4)
    state = init_state(grid.shape)
    state = state._replace(ez=state.ez.at[grid.nx // 2, grid.ny // 2, grid.nz // 2].set(1.0))
    lossy = [("'copper'", 1.0, 5.8e7), ("'cube'", 6.0, 0.05)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        static_remnant_advisory(state, init_materials(grid.shape), grid, (), lossy)
    msg = [str(w.message) for w in caught if "H-share" in str(w.message)][0]
    assert "'cube'" in msg and "'copper'" not in msg, msg
