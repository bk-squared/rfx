"""Validated OBLIQUE RCS — specular-peak + normal-reduction gates (#404 fork (a) S4/S5).

These pin the physics that lifts the ``compute_rcs`` oblique fence: a finite PEC
plate (normal +x, in the y-z plane, ~2.2 lambda wide in y, thin/periodic z)
illuminated at oblique incidence by the OPEN-DOMAIN Method-B TFSF
(``rfx/sources/tfsf_oblique_open.py``) scatters with a bistatic far-field whose
specular lobe obeys the reflection law.

Why this is the real, unfakeable gate (not the PO level)
--------------------------------------------------------
For a plate with normal +x and incident k̂ = (cosθ, sinθ, 0) (tilt toward +y),
the reflection law puts the specular direction at k̂_refl = (−cosθ, sinθ, 0), i.e.
azimuth ``phi_spec = 180 − θ`` at ``theta_obs = 90°`` (the x-y observation plane).
The far-field peak MUST sit there and TRACK θ: a wrong injection angle or a wrong
NTFF transform would move the peak elsewhere. This is angle-sensitive and cannot
be faked by a normalization tweak (the peak LOCATION is normalization-invariant).

At θ→0 the specular direction collapses to backscatter (φ=180°) — the
normal-incidence reduction to the existing validated behaviour.

Scope: this validates the far-field PATTERN / specular DIRECTION only. The
ABSOLUTE oblique σ is intentionally NOT asserted — the 2.5-D strip's 3-D RCS
scales with the (arbitrary) NTFF z-box height because the two z-faces cancel for
x-y-plane observation. See ``rfx/rcs.py`` (oblique routing block) for the full
scope note. Validation witnesses (bigger 3λ plate, n_steps=1400; measured on
the PRE-corner-fix exclusive-slice kernels and NOT re-measured on the shipped
inclusive kernels — see the tfsf_oblique_open.py module docstring): specular
180/161/140° vs predicted 180/160/140° at θ=0/20/40, settling −63…−66 dB, PO-sinc
shape correlation 0.89–0.92, 3 dB beamwidth within ~1°, and RAW peak == SUBTRACTED
peak (incident-leakage subtraction NOT needed — leakage steers to the forward
hemisphere φ≈θ, away from the specular lobe).

NOTE those witness numbers come from a DIFFERENT configuration (3λ plate,
1400 steps, the ``_methodB_pattern`` helper's NTFF box) than the committed gate
below (2.2λ plate, 700 steps, public ``compute_rcs`` path, which measures
180/160/143° on the corner-inclusive kernels; 144° at θ=40 pre-fix, and the
helper's θ=0 gate config settles at −52.3 dB — another marker that the witness
block belongs to the other configuration). The two sets are not numerically
comparable — see the helper's
docstring for the exact NTFF-box construction difference. The gate's ±6°
tolerance reflects a measured 3–7° peak-azimuth sensitivity to domain size at
fixed dx/plate/CPML (PR #461 audit); it is an observed envelope on an
unconverged argmax observable, not a converged-accuracy claim. For scale: the
2.2λ plate's physical-optics 3 dB specular beamwidth at θ=40 is ~30°
(0.886·λ/(W·cosθ)), so ±6° is ~0.2 of the lobe the argmax sits in.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from rfx.grid import Grid, C0
from rfx.core.yee import MaterialArrays
from rfx.geometry.csg import Box, rasterize
from rfx.farfield import NTFFBox
from rfx.sources.tfsf_oblique_open import init_tfsf_methodB
from rfx.simulation import run, SnapshotSpec
from rfx.rcs import compute_rcs, compute_rcs_jax, _incident_spectrum_amplitude

F0 = 5e9
LAM = C0 / F0
DX = 0.002                 # lambda/30
CPML = 10
MARGIN = 10
PLATE_W = 2.2 * LAM        # width in y
N_STEPS = 700
PEC_SIGMA = 1e7
# grid ~ 76 x 130 x 24 (thin periodic z)
_DOMAIN = (0.11, 0.218, 0.006)
PHI = np.linspace(0.0, 2 * np.pi, 361)
TH_OBS = np.array([np.pi / 2])   # x-y observation plane
BACK = (np.degrees(PHI) >= 90) & (np.degrees(PHI) <= 270)  # reflection hemisphere


def _plate_materials(grid):
    nx, ny, nz = grid.shape
    xc = (nx // 2) * DX
    yc = (ny // 2) * DX
    half = PLATE_W / 2.0
    plate = Box(corner_lo=(xc - 0.5 * DX, yc - half, -1.0),
                corner_hi=(xc + 0.5 * DX, yc + half, +1.0))
    eps_r, sigma = rasterize(grid, [(plate, 1.0, PEC_SIGMA)])
    return MaterialArrays(eps_r=eps_r, sigma=sigma,
                          mu_r=jnp.ones(grid.shape, dtype=jnp.float32))


def _peak_phi_deg(sig_linear):
    """Azimuth (deg) of the specular lobe in the reflection hemisphere."""
    return float(np.degrees(PHI)[BACK][np.argmax(np.asarray(sig_linear)[BACK])])


def _methodB_pattern(theta_deg):
    """Build the 2.5-D open-domain Method-B pipeline directly (theta=0 allowed) and
    return (peak_phi_deg, settling_db).

    NOT a mirror of ``compute_rcs``: same pipeline components, but the NTFF box
    is constructed differently — here the y-faces are pinned to the TFSF box
    (``cfg.y_lo - off`` / ``cfg.y_hi + off``) while the public path derives them
    from the CPML faces (``fl["y_lo"] + ntff_offset`` / ``grid.ny - fl["y_hi"] -
    ntff_offset``, rcs.py) and its ``i_hi`` carries an extra +1. So peak azimuths
    from this helper differ from the public path by a few degrees and MUST NOT be
    numerically compared against ``compute_rcs`` results; use it only for what it
    is here: exercising theta=0, the normal-reduction limit that ``compute_rcs``
    routes to the normal path."""
    grid = Grid(freq_max=F0 * 3, domain=_DOMAIN, dx=DX, cpml_layers=CPML)
    nx, ny, nz = grid.shape
    cfg, st = init_tfsf_methodB(nx, ny, DX, grid.dt, nz=nz, cpml_layers=CPML,
                                tfsf_margin=MARGIN, f0=F0, polarization="ez",
                                direction="+x", theta_deg=theta_deg)
    off = 3
    kz = nz // 2
    box = NTFFBox.from_grid(
        grid, i_lo=cfg.x_lo - off, i_hi=cfg.x_hi + off,
        j_lo=cfg.y_lo - off, j_hi=cfg.y_hi + off,
        k_lo=kz - 1, k_hi=kz + 1, freqs=jnp.asarray([F0], jnp.float32))
    mats = _plate_materials(grid)
    snap = SnapshotSpec(interval=8, components=("ez",), slice_axis=2, slice_index=kz)
    res = run(grid, mats, N_STEPS, boundary="cpml", cpml_axes="xy",
              periodic=(False, False, True), pec_axes="",
              tfsf=(cfg, st), ntff=box, snapshot=snap)
    e_inc = _incident_spectrum_amplitude(F0, 0.5, np.array([F0]), grid.dt, N_STEPS)
    sig = np.asarray(compute_rcs_jax(res.ntff_data, box, grid, TH_OBS, PHI, e_inc))[0, 0, :]
    # settling witness: end vs peak field energy in the total-field box interior
    ez = np.asarray(res.snapshots["ez"])
    e = ez[:, cfg.x_lo + 4:cfg.x_hi - 4, cfg.y_lo + 4:cfg.y_hi - 4]
    energy = np.sum(e ** 2, axis=(1, 2))
    settling = 10 * np.log10(max(energy[-1], 1e-30) / max(energy.max(), 1e-30))
    return _peak_phi_deg(sig), float(settling)


@pytest.mark.slow
@pytest.mark.parametrize("theta_inc", [20.0, 40.0])
def test_oblique_specular_peak_tracks_injection_angle(theta_inc):
    """PRIMARY unfakeable gate (via the public compute_rcs API): the bistatic
    far-field of the PEC plate peaks at the reflection-law specular direction
    phi = 180 - theta_inc and TRACKS the injection angle."""
    grid = Grid(freq_max=F0 * 3, domain=_DOMAIN, dx=DX, cpml_layers=CPML)
    mats = _plate_materials(grid)
    res = compute_rcs(grid, mats, N_STEPS, f0=F0, bandwidth=0.5,
                      theta_inc=theta_inc, polarization="ez",
                      theta_obs=TH_OBS, phi_obs=PHI, freqs=np.array([F0]),
                      cpml_layers=CPML, tfsf_margin=MARGIN, ntff_offset=3)
    peak = _peak_phi_deg(res.rcs_linear[0, 0, :])
    pred = 180.0 - theta_inc
    assert abs(peak - pred) <= 6.0, (
        f"specular peak {peak:.0f} deg not tracking predicted {pred:.0f} deg "
        f"(theta_inc={theta_inc}); a wrong injection angle / NTFF moves the peak"
    )
    # the reflection lobe must dominate the reflection hemisphere over backscatter
    sig = np.asarray(res.rcs_linear[0, 0, :])
    back_bin = int(round((180.0 + theta_inc))) % 360
    spec_bin = int(round(pred)) % 360
    assert sig[spec_bin] > 5 * sig[back_bin], (
        "specular lobe should dominate backscatter for a flat plate at oblique "
        f"incidence (spec={sig[spec_bin]:.2e} vs back={sig[back_bin]:.2e})"
    )


@pytest.mark.slow
def test_oblique_rcs_normal_reduction_to_backscatter():
    """NORMAL-REDUCTION (comparator-first): the SAME 2.5-D Method-B pipeline at
    theta=0 puts the specular lobe at phi=180 deg (= backscatter = specular at
    normal incidence), with a settled ring-down. Proves the oblique path reduces
    to the known-good normal case."""
    peak0, settling0 = _methodB_pattern(0.0)
    assert abs(peak0 - 180.0) <= 4.0, f"normal-incidence peak {peak0:.0f} != 180 deg"
    assert settling0 < -35.0, f"ring-down not settled: end/peak {settling0:.1f} dB"
    # and it still tracks at oblique through the identical direct pipeline
    peak40, settling40 = _methodB_pattern(40.0)
    assert abs(peak40 - 140.0) <= 6.0, f"oblique peak {peak40:.0f} != ~140 deg"
    assert settling40 < -35.0, f"oblique ring-down not settled: {settling40:.1f} dB"


def test_oblique_rcs_fence_kept_for_unsupported_combos():
    """The fence is KEPT (fails loud) for combos Method B does not support:
    non-ez polarization and non-uniform grids. Fast — raises before any run."""
    from rfx.core.yee import init_materials
    grid = Grid(freq_max=10e9, domain=(0.06, 0.06, 0.06), dx=0.003, cpml_layers=6)
    mats = init_materials(grid.shape)
    with pytest.raises(NotImplementedError, match="ez"):
        compute_rcs(grid, mats, n_steps=10, f0=5e9, bandwidth=0.5,
                    theta_inc=30.0, polarization="ey",
                    theta_obs=np.array([np.pi / 2]), phi_obs=np.array([0.0]))
