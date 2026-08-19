"""#679 — surface-impedance (f0) sheet threading through the MSL S-parameter lane.

``compute_msl_s_matrix`` used to refuse ``surface_impedance_f0`` thin
conductors (the "MSL S-parameter" fence, #677 G9). Every FDTD dispatch of
the lane goes through the public ``run()``/``forward()``, which already
assemble their own materials with ``sheet_specs`` and build the
sheet-impedance ctx against their own final pec_mask — so the fence removal
alone makes the lane sheet-capable, with NO lane-level ctx plumbing. These
tests pin that with pre-declared oracles (issue #679; the
completing-is-not-the-property lesson of
``test_vmap_sweep_fallback_still_applies_the_sheet``):

O1  no-sheet identity — with no f0 sheets registered the lane's output is
    byte-identical to a golden captured BEFORE the fence removal.
O2  Rs->0 limit — a tiny-Rs0 f0 sheet reproduces the PEC-sheet realization
    of the same mask (the #677 footprint-identity tooth, through the full
    MSL extraction).
O3  loss monotonic on the low-Rs branch; the f0-vs-PEC dispersion static
    (the sheet realization changes LOSS, never the structure).

Plus the dispatch smokes the scout's plan asked for: fast threading witness
(uniform run()), NU (``dz_profile`` -> run_nonuniform_path) and AD
(``eps_override`` -> forward()).

Fixture: the calibrated dx=80um RO4350B laplace thru of
``test_msl_port_integration.py``, shortened to L_LINE=8mm. The sheet is
z-normal, floating in the air 2 cells above the 1-cell trace, x in
[4.5, 7.5] mm — clear of both feed planes (x=2/10 mm) and both N-probe
spans (offset 5, spacing 3, n=5 -> x in [2.4, 3.36] and [8.64, 9.6] mm),
so the lossless-line N-probe fit hazard (per-length loss inside a probed
span) is avoided by construction; see compute_msl_s_matrix's docstring.

Preflight context (R5 / preflight-quoting rule) — every settled run below
emits the same 7 ADVISORY preflight findings, re-printed verbatim by
``_settled``: 1-cell PEC trace advisory, pec_faces-with-finite-PEC
advisory, lossless-dielectric advisory, and the per-port
3-substrate-cell + mixed-cell-danger-zone advisories (h_sub/dx = 3.175).
They describe known Z0-bias envelopes of this deliberately coarse CPU
fixture, not defects; the oracles below compare runs of the SAME fixture
against each other (identity, limit, monotonicity), which is insensitive
to the shared discretization bias. The settled sheet-free extraction also
warns: standing-wave nulls at 4.4-5.0 GHz (3 bins, ``reliable`` False
there — outside the 3.0-4.5 GHz gate band used below), a ~20% Z0
honesty-guard deviation vs Hammerstad-Jensen (the documented coarse-mesh
staircase envelope), and the passivity projection clipping a worst
sigma_max of 1.001 (extraction ripple; the LOSSY sheet runs rs1/rs5/rs25
measured raw-passive, no projection warning).

Heavy tests are ``@pytest.mark.slow`` (weekly lane); the fast lane keeps
the threading witness and the NU smoke.
"""

import warnings
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

# --------------------------------------------------------------------------
# Fixture geometry (mirrors test_msl_port_integration.py, L_LINE shortened)
# --------------------------------------------------------------------------

EPS_R = 3.66          # RO4350B
H_SUB = 254e-6        # substrate thickness (m)
W_TRACE = 600e-6      # trace width (m)
L_LINE = 8e-3         # thru-line length (m)
PORT_MARGIN = 2e-3    # feed -> domain edge clearance (m)
DX = 80e-6
F_MAX = 5e9
LX = L_LINE + 2 * PORT_MARGIN
LY = W_TRACE + 2 * (2 * H_SUB + 8 * DX)
LZ = H_SUB + 1.5e-3
SHEET_F0 = 3.5e9              # band-centre for Rs0 (in the gate band)
SHEET_Z = H_SUB + 3.5 * DX    # 2 cells above the 1-cell trace, in air
SHEET_X = (4.5e-3, 7.5e-3)    # clear of feeds and probe spans (see docstring)
SHEET_HALF_W = 0.7e-3

#: The full-settle frequency grid (matches the golden capture exactly).
FREQS = np.linspace(0.5e9, 5e9, 16)
#: Quasi-TEM gate band at dx=80um (same window as the integration gate).
GATE = (FREQS >= 3.0e9) & (FREQS <= 4.5e9)

MU_0 = 4e-7 * np.pi

_FIXTURES = Path(__file__).parent / "fixtures"

#: O2 gate: measured max complex deviation 3.07e-6 (see test docstring),
#: gate ~30x above it and still ~1e-2 of the smallest physics signal here.
GATE_O2 = 1e-4


def _sigma_bulk_for_rs0(rs0, f0=SHEET_F0):
    """Invert Rs0 = sqrt(pi*f0*mu0/sigma_bulk)."""
    return float(np.pi * f0 * MU_0 / rs0 ** 2)


def build_msl_thru(sheet=None, dz_profile=None):
    """Two-port MSL thru.  ``sheet``: None | ("f0", sigma_bulk) | ("pec",).

    ``dz_profile`` switches the z axis to the non-uniform lane (the domain
    z extent is then ``sum(dz_profile)``).
    """
    if dz_profile is None:
        sim = Simulation(
            freq_max=F_MAX, domain=(LX, LY, LZ), dx=DX, cpml_layers=8,
            boundary=BoundarySpec(x="cpml", y="cpml",
                                  z=Boundary(lo="pec", hi="cpml")),
        )
    else:
        lz = float(np.sum(dz_profile))
        sim = Simulation(
            freq_max=F_MAX, domain=(LX, LY, lz), dx=DX,
            dz_profile=list(dz_profile), cpml_layers=8,
            boundary=BoundarySpec(x="cpml", y="cpml",
                                  z=Boundary(lo="pec", hi="cpml")),
        )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (LX, LY, H_SUB)), material="ro4350b")
    y_c = LY / 2.0
    sim.add(Box((0.0, y_c - W_TRACE / 2, H_SUB),
                (LX, y_c + W_TRACE / 2, H_SUB + DX)), material="pec")
    if sheet is not None:
        box = Box((SHEET_X[0], y_c - SHEET_HALF_W, SHEET_Z),
                  (SHEET_X[1], y_c + SHEET_HALF_W, SHEET_Z))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if sheet[0] == "f0":
                sim.add_thin_conductor(box, sigma_bulk=sheet[1],
                                       surface_impedance_f0=SHEET_F0)
            else:  # "pec": legacy sigma>=1e6 routing -> PEC mask sheet
                sim.add_thin_conductor(box, sigma_bulk=5.8e7)
    for x, d in ((PORT_MARGIN, "+x"), (PORT_MARGIN + L_LINE, "-x")):
        sim.add_msl_port(position=(x, y_c, 0.0), width=W_TRACE,
                         height=H_SUB, direction=d, impedance=50.0)
    return sim


# --------------------------------------------------------------------------
# Settled-run cache: each tag is ONE ~2.5-3 min CPU run, shared across tests
# --------------------------------------------------------------------------

_CACHE: dict = {}

_SHEETS = {
    "off": None,
    "pec": ("pec",),
    "rs_tiny": ("f0", _sigma_bulk_for_rs0(1e-6)),
    "rs1": ("f0", _sigma_bulk_for_rs0(1.0)),
    "rs5": ("f0", _sigma_bulk_for_rs0(5.0)),
    "rs25": ("f0", _sigma_bulk_for_rs0(25.0)),
}


def _settled(tag):
    """Full-settle run (num_periods=12, the integration fixture's setting),
    memoized per sheet config. Captured warnings are re-printed verbatim
    (preflight-quoting rule) and returned for assertions."""
    if tag not in _CACHE:
        sim = build_msl_thru(sheet=_SHEETS[tag])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = sim.compute_msl_s_matrix(freqs=FREQS, num_periods=12.0)
        msgs = [str(w.message) for w in caught]
        print(f"\n[msl-sheet {tag}] captured warnings ({len(msgs)}):")
        for m in msgs:
            print(f"[msl-sheet {tag}]  {m}")
        print(f"[msl-sheet {tag}] settling_db = {np.asarray(result.settling_db)}")
        _CACHE[tag] = (result, msgs)
    return _CACHE[tag]


# --------------------------------------------------------------------------
# O1 — no-sheet identity against the pre-change golden
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_o1_no_sheet_identity_vs_13de212_golden():
    """With NO f0 sheet registered, the lane's S is byte-identical to the
    golden captured at commit 13de212 (BEFORE the fence removal).

    Provenance: the golden pair
    ``tests/fixtures/golden_msl_sheet_thread_{s,freqs}_13de212.npy`` was
    produced on 2026-08-19 from a pristine detached worktree of commit
    13de212 (the #677/#678 merge, the parent of the #679 change), running
    THIS module's ``build_msl_thru(sheet=None)`` fixture with
    ``compute_msl_s_matrix(freqs=FREQS, num_periods=12.0)`` on CPU
    (JAX_PLATFORMS=cpu, float32 default precision, jax 0.8.x, linux
    x86-64). Two consecutive captures were np.array_equal (deterministic),
    and the post-change worktree reproduced the capture byte-exactly
    (max dev 0.0) before this file was committed. The removed fence ran
    BEFORE any physics on the no-sheet path and was a no-op there, so any
    drift here means the #679 edit touched the sheet-free lane — exactly
    what this oracle forbids. Byte identity is the gate on the capture
    platform (same BLAS/JAX build); cross-platform float drift would show
    up here as a tiny nonzero max-dev — investigate before touching the
    gate (no-silent-gate-loosening rule).
    """
    golden_s = np.load(_FIXTURES / "golden_msl_sheet_thread_s_13de212.npy")
    golden_f = np.load(_FIXTURES / "golden_msl_sheet_thread_freqs_13de212.npy")
    result, _ = _settled("off")
    S = np.asarray(result.S)
    np.testing.assert_array_equal(np.asarray(result.freqs), golden_f)
    # Diagnostic print before the byte gate (R5: show the curves, not a bool)
    print("[O1] |S11| golden:", np.abs(golden_s[0, 0]).round(5))
    print("[O1] |S11| now   :", np.abs(S[0, 0]).round(5))
    max_dev = float(np.max(np.abs(S - golden_s)))
    print(f"[O1] max |S - golden| = {max_dev:.3e}")
    assert S.dtype == golden_s.dtype, (S.dtype, golden_s.dtype)
    assert np.array_equal(S, golden_s), (
        f"no-sheet MSL S drifted from the 13de212 golden (max dev "
        f"{max_dev:.3e}) — the #679 change must be a no-op without sheets")


# --------------------------------------------------------------------------
# Fast threading witness (uniform run() dispatch) — fast lane
# --------------------------------------------------------------------------

def test_threading_witness_sheet_changes_s():
    """Completing is not the property — DIFFERING from sheet-free is.

    Short (n_steps=600) unsettled runs: S must be finite and the Rs0=5
    sheet must move it by a resolvable amount. Measured on the change
    commit: max rel |S| difference 1.89e-2 (gate >= 1e-3, ~19x margin).
    All warnings are quoted verbatim and allowed — the run is deliberately
    unsettled, so the settling witness and the Z0 honesty guard both fire
    (measured: 'ring-down settling witness FAILED', Z0 ~ 8.6 ohm at 394%
    deviation); the witness compares the two equally-unsettled runs, not
    their absolute physics.
    """
    F5 = np.linspace(2e9, 4.5e9, 5)

    def short(sheet):
        sim = build_msl_thru(sheet=sheet)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = sim.compute_msl_s_matrix(freqs=F5, n_steps=600)
        for w in caught:
            print(f"[witness sheet={sheet is not None}] WARN: {w.message}")
        return np.asarray(r.S)

    s_on = short(_SHEETS["rs5"])
    s_off = short(None)
    assert np.all(np.isfinite(s_on)), "sheet-bearing MSL run went non-finite"
    assert np.all(np.isfinite(s_off))
    scale = float(np.abs(s_off).max())
    assert scale > 0.0, "sheet-free control recorded no signal — bad fixture"
    rel = float(np.abs(s_on - s_off).max() / scale)
    print(f"[witness] max rel |S_on - S_off| = {rel:.4e}")
    assert rel > 1e-3, (
        f"sheet-on and sheet-free MSL runs agree to {rel:.2e} — the lane "
        f"appears to have simulated a sheet-FREE model (#369 class)")


# --------------------------------------------------------------------------
# O2 — Rs0 -> 0 reproduces the PEC-sheet realization of the same mask
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_o2_rs_to_zero_matches_pec_sheet():
    """An f0 sheet at Rs0 = 1e-6 ohm/sq must reproduce the legacy PEC-sheet
    realization of the SAME Box (sigma_bulk=5.8e7, no f0 -> pec_mask): the
    #677 operator acts on exactly the tangential edge set apply_pec_mask
    would zero, and at sigma_sheet*dt/eps >> 1 the exponential update
    drives those edges to ~0 like PEC does.

    Measured on the change commit (settled num_periods=12 runs):
    max|S_f0 - S_pec| = 3.07e-6 (complex), max||S_f0|-|S_pec|| = 1.40e-6,
    max||S11|| deviation = 6.00e-7; the 1e-4 gate is ~30x the measured
    envelope and ~450x below the smallest sheet-physics signal in this
    module (the 4.5e-4 O3 monotonicity margin at float32 |S|~1 scale).
    (Sibling unit-level pin: test_g3b_tiny_rs0_matches_pec_resonance,
    1e-4 relative on a cavity resonance.) Against sheet-FREE the same
    settled comparison measures 2.4e-2 — three orders larger — so this
    limit cannot pass by the sheet silently not being applied.
    """
    r_pec, _ = _settled("pec")
    r_tiny, _ = _settled("rs_tiny")
    s_pec = np.asarray(r_pec.S)
    s_tiny = np.asarray(r_tiny.S)
    assert np.all(np.isfinite(s_pec)) and np.all(np.isfinite(s_tiny))
    dev_cplx = float(np.abs(s_tiny - s_pec).max())
    dev_mag = float(np.abs(np.abs(s_tiny) - np.abs(s_pec)).max())
    print(f"[O2] max|S_f0(Rs=1e-6) - S_pec| = {dev_cplx:.3e} "
          f"(mag {dev_mag:.3e})")
    print("[O2] |S11| pec:", np.abs(s_pec[0, 0]).round(5))
    print("[O2] |S11| f0 :", np.abs(s_tiny[0, 0]).round(5))
    # Negative control: the PEC-limit pair must NOT equal the sheet-free run
    s_off = np.asarray(_settled("off")[0].S)
    rel_off = float(np.abs(s_tiny - s_off).max() / np.abs(s_off).max())
    print(f"[O2] control: max rel |S_f0 - S_off| = {rel_off:.3e}")
    assert rel_off > 1e-3, "tiny-Rs sheet is indistinguishable from NO sheet"
    assert dev_cplx < GATE_O2, (
        f"Rs0->0 f0 sheet deviates {dev_cplx:.3e} from the PEC-sheet "
        f"realization (gate {GATE_O2:g}) — the operator is not converging "
        f"to the PEC footprint on the MSL lane")


# --------------------------------------------------------------------------
# O3 — loss monotonic (low-Rs branch); f0-vs-PEC dispersion static
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_o3_loss_monotonic_dispersion_static_passive():
    """Loss ladder Rs0 = 1e-6, 1, 5 ohm/sq: in-band mean |S21| STRICTLY
    decreasing; the f0 realization's dispersion pinned to the PEC
    realization's; every sheet-on S passive.

    Measured on the change commit (settled runs, 3.0-4.5 GHz gate band):

        mean|S21|: off 0.997046 | Rs 1e-6 0.998284 | Rs 1 0.994244 |
                   Rs 5 0.992916 | Rs 25 0.996560
        absorbed 1-|S11|^2-|S21|^2: 2.2e-4 | 2.2e-4 | 8.1e-3 | 9.7e-3 |
                   2.1e-3

    The ladder is gated on the LOW-Rs branch {1e-6, 1, 5} only: sheet
    absorption is NON-monotonic in Rs0 over the full range — a shunt
    resistive sheet dissipates maximally at an intermediate Rs (the
    Salisbury-screen shape: Rs->0 is a lossless PEC reflector, Rs->inf is
    transparent), and Rs0=25 measured PAST the peak (absorbed 2.1e-3 <
    Rs5's 9.7e-3). Rs25 is still run here for the passivity witness and
    to keep that measured peak documented; gating it into a monotonic
    chain would be wrong physics, not a tolerance issue. Monotonicity
    margin: measured gaps 4.04e-3 (1e-6 -> 1) and 1.33e-3 (1 -> 5); the
    2e-4 required margin is ~6x under the smallest.

    Dispersion gate — this thru fixture has no in-band resonance (any
    half-wave feature of the 12 mm board sits far above the 5 GHz
    quasi-TEM ceiling of dx=80um), so 'resonance-position motion' is
    measured on the extracted dispersion of the FIXED footprint pair
    instead: the beta-implied equivalent frequency shift
    |f * (beta_f0(1e-6) - beta_pec)/beta_pec| must stay under 50 MHz per
    gate bin — the same 'the realization did not move the structure'
    claim the #677 cavity gate makes (its measured envelope was
    0.58/0.63 MHz; 50 MHz is the generous #679 gate). Measured:
    0.0499 MHz gate-band max, 0.145 MHz full-band max. The Rs LADDER
    itself is deliberately NOT beta-gated: a lossy near-field sheet loads
    the line reactively as well, and the ladder's extracted-beta spread
    measured 47-91 MHz equivalent between rungs (quoted below per run) —
    that is loading physics plus N-probe fit sensitivity on this coarse
    open line, not a realization artifact.
    """
    ladder = ["rs_tiny", "rs1", "rs5"]
    s21 = {}
    for tag in ladder + ["rs25"]:
        r, msgs = _settled(tag)
        S = np.asarray(r.S)
        assert np.all(np.isfinite(S)), f"{tag}: non-finite S"
        s21[tag] = float(np.mean(np.abs(S[1, 0, GATE])))
        sv = np.linalg.svd(S.transpose(2, 0, 1), compute_uv=False)
        sv_max = float(sv.max())
        absorbed = float(np.mean(
            1 - np.abs(S[0, 0, GATE]) ** 2 - np.abs(S[1, 0, GATE]) ** 2))
        print(f"[O3 {tag}] mean|S21|(gate) = {s21[tag]:.6f}  "
              f"sigma_max = {sv_max:.6f}  absorbed = {absorbed:.3e}")
        print(f"[O3 {tag}] |S21|(f) = "
              f"{np.abs(S[1, 0]).round(4)}")
        print(f"[O3 {tag}] beta.real(gate) = "
              f"{np.real(np.asarray(r.beta))[GATE].round(2)}")
        assert sv_max <= 1.0 + 1e-3, (
            f"{tag}: returned S non-passive (sigma_max {sv_max:.4f})")
    off_s21 = float(np.mean(np.abs(np.asarray(_settled('off')[0].S)[1, 0, GATE])))
    print(f"[O3] mean|S21|(gate): off={off_s21:.6f} " +
          " ".join(f"{t}={s21[t]:.6f}" for t in ladder + ["rs25"]))
    for a, b in zip(ladder, ladder[1:]):
        assert s21[a] - s21[b] > 2e-4, (
            f"in-band |S21| not decreasing from Rs0[{a}] to Rs0[{b}]: "
            f"{s21[a]:.6f} -> {s21[b]:.6f} (need > 2e-4 drop)")
    # Fixed-footprint dispersion gate: f0(Rs->0) vs PEC realization
    beta_f0 = np.real(np.asarray(_settled("rs_tiny")[0].beta))
    beta_pec = np.real(np.asarray(_settled("pec")[0].beta))
    df = np.abs(FREQS * (beta_f0 - beta_pec) / beta_pec)
    worst = float(df[GATE].max())
    print(f"[O3] f0-vs-PEC beta-implied shift: gate-band max "
          f"{worst/1e6:.4f} MHz, full-band max {float(df.max())/1e6:.4f} MHz")
    assert worst < 50e6, (
        f"f0 realization's dispersion moved {worst/1e6:.2f} MHz vs the PEC "
        f"realization of the same mask — the sheet operator must change "
        f"loss, not the structure (#677 tooth; measured envelope 0.05 MHz)")


# --------------------------------------------------------------------------
# NU smoke — pins the run_nonuniform_path dispatch
# --------------------------------------------------------------------------

def test_nu_smoke_sheet_applies_on_dz_profile_lane():
    """Same fixture on the graded-mesh lane (uniform dz_profile — the
    dispatch under test does not depend on the grading ratio): sheet-on and
    sheet-off short runs must differ, pinning that run_nonuniform_path
    receives and applies the ctx (nonuniform.py sheet_specs emission +
    per-node dual spacing #671). Measured on the change commit: max rel
    diff 1.89e-2 (gate >= 1e-3); warnings quoted and allowed as in the
    uniform witness (deliberately unsettled run)."""
    F5 = np.linspace(2e9, 4.5e9, 5)
    prof = [DX] * 22

    def short(sheet):
        sim = build_msl_thru(sheet=sheet, dz_profile=prof)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = sim.compute_msl_s_matrix(freqs=F5, n_steps=600)
        for w in caught:
            print(f"[NU sheet={sheet is not None}] WARN: {w.message}")
        return np.asarray(r.S)

    s_on = short(_SHEETS["rs5"])
    s_off = short(None)
    assert np.all(np.isfinite(s_on)), "NU sheet-bearing run went non-finite"
    scale = float(np.abs(s_off).max())
    assert scale > 0.0
    rel = float(np.abs(s_on - s_off).max() / scale)
    print(f"[NU witness] max rel |S_on - S_off| = {rel:.4e}")
    assert rel > 1e-3, (
        f"NU sheet-on/off agree to {rel:.2e} — run_nonuniform_path appears "
        f"to have dropped the sheet ctx")


# --------------------------------------------------------------------------
# AD smoke — pins the forward() (eps_override) dispatch
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_ad_smoke_eps_override_grad_finite_with_sheet():
    """jax.grad through compute_msl_s_matrix(eps_override=...) with an f0
    sheet registered: the traced forward() dispatch must accept the sheet
    (it used to be fenced) and the gradient must be finite and non-zero
    (a severed tape reads exactly 0.0 — issue #515 class). Measured on the
    change commit: grad = -1.912060e-01.

    ``enforce_passivity=False`` explicitly: mode-independent raw reduction
    (PR #468 defect class — do not rely on the is_tracer default branch).
    """
    sim = build_msl_thru(sheet=_SHEETS["rs5"])
    grid = sim._build_grid()
    eps_base = jnp.ones(grid.shape, dtype=jnp.float32)
    freqs_ad = np.linspace(2e9, 4.5e9, 4)

    def objective(alpha):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sim.compute_msl_s_matrix(
                freqs=freqs_ad, num_periods=3, eps_override=eps_base * alpha,
                checkpoint_segments=14, enforce_passivity=False)
        return jnp.real(jnp.sum(jnp.abs(r.S) ** 2))

    grad = jax.grad(objective)(jnp.float32(1.0))
    print(f"[AD smoke] grad = {float(grad):.6e}")
    assert bool(jnp.isfinite(grad)), f"gradient not finite: {grad}"
    assert float(grad) != 0.0, (
        "gradient is exactly 0.0 — the eps_override tape may be severed "
        "(#515 class) on the sheet-bearing forward() dispatch")
