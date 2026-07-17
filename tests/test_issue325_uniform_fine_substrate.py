"""#325 grid-build lock: the patch substrate must rasterize to a UNIFORM-FINE
band with the coarse<->fine grading transition held clear of the resonator.

Background: cv05 (05_patch_antenna.py) placed the ground/substrate/patch stack at
a FIXED pre-smoothing z (`air_below = 12mm`) while `smooth_grading` inserts
transition cells that shift the fine band up — so the 1.5mm FR4 substrate built as
2 coarse cells / 1.361mm, not 6 fine cells (#325). Re-registering the stack onto a
6-cell band that sits ADJACENT to a grading transition was a FIRED STOP (research
note 20260711): the transition-adjacent substrate splits the mode (2.14/2.65/3.45)
and makes the openEMS agreement WORSE (2.65% -> 6.45%).

The correct build (verified in scripts/diagnostics/patch_tutorial_rfx.py::build):
DERIVE the stack z from where `smooth_grading` actually places the fine band, and
insert uniform-fine BUFFER cells so the transition sits away from the resonator.
This test locks that: substrate == n_sub fine cells AND no transition cell within
CLEARANCE_MIN of the stack. It FAILS on the committed cv05 geometry (2 cells) and
on the STOP-1 zero-buffer re-registration (transition adjacent) — a fails-closed
guard so neither broken geometry can be committed silently. No FDTD / no openEMS.
"""
import numpy as np
import pytest

from rfx.auto_config import smooth_grading

DX = 1.0e-3
H_SUB = 1.5e-3
N_SUB = 6
DZ_SUB = H_SUB / N_SUB          # 0.25 mm
N_BELOW, N_ABOVE = 12, 25
CLEARANCE_MIN = 2.0e-3          # transition must sit >= 2mm from the stack


def build_uniform_fine_z(n_buf, dx=DX, dz_sub=DZ_SUB, n_sub=N_SUB,
                         n_below=N_BELOW, n_above=N_ABOVE):
    """Uniform-fine substrate z-mesh + stack coordinates DERIVED from the built
    grid (imitates scripts/diagnostics/patch_tutorial_rfx.py::build, the verified
    #325 fix). Returns (dz_profile, z_gnd_lo, z_sub_lo, z_sub_hi, z_patch_hi,
    sub_cells).

    n_buf uniform-fine buffer cells on each side of the (1 GP + n_sub + 1 patch)
    stack push the coarse<->fine grading transition n_buf cells away.
    """
    raw = np.concatenate([
        np.full(n_below, dx),
        np.full(n_buf + 1 + n_sub + 1 + n_buf, dz_sub),
        np.full(n_above, dx),
    ])
    dz = smooth_grading(raw, max_ratio=1.3)
    edges = np.insert(np.cumsum(dz), 0, 0.0)
    fi = np.where(np.isclose(dz, dz_sub, rtol=1e-6))[0]
    assert len(fi) >= 2 + n_sub + 2 * n_buf, \
        f"expected >= {2 + n_sub + 2 * n_buf} fine cells, got {len(fi)}"
    f0 = int(fi[0]) + n_buf
    z_gnd_lo = float(edges[f0])
    z_sub_lo = float(edges[f0 + 1])
    z_sub_hi = float(edges[f0 + 1 + n_sub])
    z_patch_hi = float(edges[f0 + 1 + n_sub + 1])
    centers = 0.5 * (edges[:-1] + edges[1:])
    sub_cells = int(np.sum((centers >= z_sub_lo) & (centers < z_sub_hi)))
    return dz, z_gnd_lo, z_sub_lo, z_sub_hi, z_patch_hi, sub_cells


def _transition_clearance(dz, z_lo, z_hi, dx=DX, dz_sub=DZ_SUB):
    """Nearest grading-transition cell (size not in {dz_sub, dx}) to [z_lo, z_hi]."""
    edges = np.insert(np.cumsum(dz), 0, 0.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    is_trans = ~(np.isclose(dz, dz_sub, rtol=1e-6) | np.isclose(dz, dx, rtol=1e-6))
    tc = centers[is_trans]
    if not len(tc):
        return np.inf
    return float(min(np.min(np.abs(tc - z_lo)), np.min(np.abs(tc - z_hi))))


@pytest.mark.parametrize("n_buf", [8, 12, 16])
def test_uniform_fine_substrate_builds_n_sub_cells_with_clearance(n_buf):
    """The uniform-fine build rasterizes the substrate to exactly N_SUB fine
    cells AND keeps every grading transition >= CLEARANCE_MIN from the stack."""
    dz, z_gnd_lo, z_sub_lo, z_sub_hi, z_patch_hi, sub_cells = \
        build_uniform_fine_z(n_buf)
    assert sub_cells == N_SUB, \
        f"substrate must build {N_SUB} fine cells, got {sub_cells} (N_BUF={n_buf})"
    assert abs((z_sub_hi - z_sub_lo) - H_SUB) < 1e-9, \
        f"substrate thickness {(z_sub_hi - z_sub_lo)*1e3:.4f}mm != {H_SUB*1e3}mm"
    clr = _transition_clearance(dz, z_gnd_lo, z_patch_hi)
    assert clr >= CLEARANCE_MIN, \
        f"grading transition too close to the resonator: {clr*1e3:.2f}mm < {CLEARANCE_MIN*1e3}mm (N_BUF={n_buf})"


def test_committed_cv05_geometry_fails_the_lock():
    """Fails-closed guard: the committed cv05 geometry (fixed air_below=12mm, no
    buffer) rasterizes the substrate to 2 coarse cells — the lock must reject it.

    NOTE (drift risk, PR #379 review): cv05's z-mesh is inline module-level code
    (`05_patch_antenna.py`), not an importable function, so this test HAND-COPIES
    cv05's raw_dz construction (N_BELOW=12, N_ABOVE=25, DX=1mm, fixed substrate
    z=[12,13.5]mm — verified matching today). If cv05's air_below/dx/n_sub ever
    change, this frozen copy would silently stop reflecting cv05. When cv05 adopts
    build_uniform_fine_z (pending the §1-B physics witness), refactor this to
    import cv05's real construction so the guard tracks the live geometry.
    """
    raw = np.concatenate([
        np.full(N_BELOW, DX), np.full(1, DZ_SUB), np.full(N_SUB, DZ_SUB),
        np.full(N_ABOVE, DX),
    ])
    dz = smooth_grading(raw, max_ratio=1.3)
    edges = np.insert(np.cumsum(dz), 0, 0.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # cv05 places the substrate at the FIXED intended z = [12, 13.5] mm
    sub_cells = int(np.sum((centers >= 12e-3) & (centers < 13.5e-3)))
    assert sub_cells != N_SUB, \
        "committed cv05 geometry unexpectedly builds N_SUB cells — the #325 bug " \
        "would be gone and this guard is stale"
    assert sub_cells == 2, \
        f"the #325 bug should build exactly 2 coarse cells, got {sub_cells}"


def test_zero_buffer_reregistration_has_no_clearance():
    """Fails-closed guard against STOP-1: registering 6 fine cells with NO buffer
    (transition adjacent to the resonator) — the split-mode geometry — must be
    rejected by the clearance check even though its cell COUNT is correct."""
    dz, z_gnd_lo, z_sub_lo, z_sub_hi, z_patch_hi, sub_cells = \
        build_uniform_fine_z(n_buf=0)
    assert sub_cells == N_SUB   # count is right...
    clr = _transition_clearance(dz, z_gnd_lo, z_patch_hi)
    assert clr < CLEARANCE_MIN, \
        f"zero-buffer re-registration should have ~0 clearance (STOP-1 geometry), " \
        f"got {clr*1e3:.2f}mm — the clearance guard is not discriminating"
