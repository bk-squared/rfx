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

#931. The stack lost its two reserved metal cells. Ground and patch are
FOILS and therefore sheets on the substrate faces (design note §1.3), and a
sheet owns no cell, so the fine band is ``n_buf + n_sub + n_buf`` — not
``n_buf + 1 + n_sub + 1 + n_buf``. The reserved cells were the pre-#931
"give the metal its own cell" compensation: the old rule put one wall per
masked NODE plane, so a foil needed a masked cell to hang its plane on, and
the cavity the mesh actually built was 2.0 mm where the board is 1.5 mm.
Nothing about the #325 property this file locks depends on those cells —
the substrate still has to build N_SUB fine cells and the transition still
has to stay CLEARANCE_MIN away — and the live cv05 migration is
validation/crossval/05_patch_antenna.py's own (owner X-A).
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
    #325 fix). Returns (dz_profile, z_gnd, z_sub_lo, z_sub_hi, z_patch,
    sub_cells), where ``z_gnd == z_sub_lo`` and ``z_patch == z_sub_hi``.

    n_buf uniform-fine buffer cells on each side of the n_sub-cell substrate
    push the coarse<->fine grading transition n_buf cells away.

    #931: the ground and the patch are FOILS, so they are sheets on the two
    substrate faces (§1.3) — a sheet owns no cell. The mesh used to reserve
    ONE fine cell below the substrate for the ground and ONE above it for
    the patch (``n_buf + 1 + n_sub + 1 + n_buf``): the "give the metal its
    own cell" compensation, which existed because the pre-#931 rule
    realized a conductor as one wall per masked NODE plane and therefore
    needed a masked cell to hang that plane on. Under the contract the
    ground sheet sits ON ``z_sub_lo`` and the patch sheet ON ``z_sub_hi``,
    both exact nodes of the fine band, and the two reserved cells are gone.
    The realized cavity is then the drawn 1.5 mm substrate, not 2.0 mm of
    mesh with metal cells at each end.
    """
    raw = np.concatenate([
        np.full(n_below, dx),
        np.full(n_buf + n_sub + n_buf, dz_sub),
        np.full(n_above, dx),
    ])
    dz = smooth_grading(raw, max_ratio=1.3)
    edges = np.insert(np.cumsum(dz), 0, 0.0)
    fi = np.where(np.isclose(dz, dz_sub, rtol=1e-6))[0]
    assert len(fi) >= n_sub + 2 * n_buf, \
        f"expected >= {n_sub + 2 * n_buf} fine cells, got {len(fi)}"
    f0 = int(fi[0]) + n_buf
    z_sub_lo = float(edges[f0])
    z_sub_hi = float(edges[f0 + n_sub])
    z_gnd, z_patch = z_sub_lo, z_sub_hi      # the two sheet planes
    centers = 0.5 * (edges[:-1] + edges[1:])
    sub_cells = int(np.sum((centers >= z_sub_lo) & (centers < z_sub_hi)))
    return dz, z_gnd, z_sub_lo, z_sub_hi, z_patch, sub_cells


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
    dz, z_gnd, z_sub_lo, z_sub_hi, z_patch, sub_cells = \
        build_uniform_fine_z(n_buf)
    assert sub_cells == N_SUB, \
        f"substrate must build {N_SUB} fine cells, got {sub_cells} (N_BUF={n_buf})"
    assert abs((z_sub_hi - z_sub_lo) - H_SUB) < 1e-9, \
        f"substrate thickness {(z_sub_hi - z_sub_lo)*1e3:.4f}mm != {H_SUB*1e3}mm"
    # #931: the sheet planes ARE the substrate faces — no reserved metal cell
    # between the foil and the laminate it sits on.
    assert z_gnd == z_sub_lo and z_patch == z_sub_hi
    clr = _transition_clearance(dz, z_gnd, z_patch)
    assert clr >= CLEARANCE_MIN, \
        f"grading transition too close to the resonator: {clr*1e3:.2f}mm < {CLEARANCE_MIN*1e3}mm (N_BUF={n_buf})"


def test_committed_patch_crossval_geometry_fails_the_lock():
    """Fails-closed guard: the committed cv05 geometry (fixed air_below=12mm, no
    buffer) rasterizes the substrate to 2 coarse cells — the lock must reject it.

    NOTE (drift risk, PR #379 review): cv05's z-mesh is inline module-level code
    (`05_patch_antenna.py`), not an importable function, so this test HAND-COPIES
    cv05's raw_dz construction (N_BELOW=12, N_ABOVE=25, DX=1mm, fixed substrate
    z=[12,13.5]mm — verified matching today). If cv05's air_below/dx/n_sub ever
    change, this frozen copy would silently stop reflecting cv05. When cv05 adopts
    build_uniform_fine_z (pending the §1-B physics witness), refactor this to
    import cv05's real construction so the guard tracks the live geometry.

    #931: the ground's reserved fine cell is dropped from this copy too (a
    foil owns no cell). The number it locks does not move — measured, the
    fixed-z placement builds 2 coarse cells with or without that cell — and
    the property under test is the FIXED pre-smoothing z, not the stack's
    cell budget.
    """
    raw = np.concatenate([
        np.full(N_BELOW, DX), np.full(N_SUB, DZ_SUB), np.full(N_ABOVE, DX),
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
    dz, z_gnd, z_sub_lo, z_sub_hi, z_patch, sub_cells = \
        build_uniform_fine_z(n_buf=0)
    assert sub_cells == N_SUB   # count is right...
    clr = _transition_clearance(dz, z_gnd, z_patch)
    assert clr < CLEARANCE_MIN, \
        f"zero-buffer re-registration should have ~0 clearance (STOP-1 geometry), " \
        f"got {clr*1e3:.2f}mm — the clearance guard is not discriminating"
