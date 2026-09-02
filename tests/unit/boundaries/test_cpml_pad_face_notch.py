"""Locks for the one-cell vacuum notch at a domain face (issue #655).

Sibling of ``test_cpml_pad_material_extension.py``, which locks #627a's repair
of the CPML **pad**. #655 is the other half of the same node: #627a taught the
hi-face pad to source its material from one column further in when the
outermost interior column reads vacuum, but nothing wrote that material into
the outermost interior column itself. The assembled column for a slab flush
with the domain face therefore read

    [pad = material][... interior = material ...][ONE VACUUM CELL][pad = material]

with the vacuum cell sitting at the last interior node — a spurious thin film
sandwiched between the structure and its own impedance-matched absorber.

**Why the repair is on the pad-extension side and not in the rasterizer.**
``rfx.geometry.csg.Box``'s half-open ``[lo, hi)`` volume rule is deliberate and
load-bearing (see that class's docstring). Flipping ``coords < hi`` to
``coords <= hi`` was measured against the committed suite and moves geometry
across the package; it is not the fix. #627a made the adjudication already:
when the outermost interior column is vacuum and the one inside it is not,
that vacuum column IS the dropped node, and the material continues through it.
Having decided that for the pad, the only self-consistent thing to do is write
the same value into the boundary column — which is exactly what makes the hi
face behave like the lo face, where ``_extend_lo`` replicates the boundary node
itself and so pad and boundary node can never disagree.

**The defect is not Box-specific.** The repair sits on the assembled array and
fires on the pattern (outer column vacuum, inner column not), so it covers any
primitive that leaves exactly one vacuum node at a hi face. ``Sphere`` and
``Cylinder`` reach that state by a different route than ``Box`` — their axis
predicates are closed (``r2 <= radius**2``), so what drops their boundary node
is the float32 knife edge documented as consequence (2) of the ``Box``
docstring, not half-openness. Same symptom, same repair; locked below.

**Measured impact** (1-D plane-wave FDTD, eps_r=4 filling the domain, periodic
transverse, CPML on z; reflection isolated by field-level DFT subtraction
against the same fixture with the box extended half a cell so that only the one
node differs, the pad being identical in both):

    dx        cells/lambda0   |r| notch (flux)   |r| (probe FFT)   thin-film theory
    0.25 mm       120             0.0321            0.0385            0.0393
    0.50 mm        60             0.0830            0.0792            0.0786
    1.00 mm        30             0.1914            0.1570            0.1572
    1.50 mm        20             0.2377            0.2644            0.2358   <- rfx default

``dx = c0 / freq_max / 20`` is rfx's own default (``rfx/api/__init__.py``), so
the WORST row is the one a user gets by not passing ``dx`` at all. The error
grows as the mesh gets COARSER, which is the opposite of the direction a user
checking convergence would look, and it is why the gates here are pinned at the
default mesh rather than at a fine one: a fine-mesh-only test would understate
the defect by ~6x and could pass on a broken tree.

Thin-film theory for a one-cell vacuum film in a medium eps_m,
``|r| = 2*pi*(dx/lambda0)*(eps_m - 1)/(2*sqrt(eps_m))``, is the independent
oracle the flux number is checked against above.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import Simulation, GaussianPulse
from rfx.geometry.csg import Box, Sphere, Cylinder
from rfx.boundaries.spec import BoundarySpec
from rfx.probes.probes import flux_spectrum

C0 = 299792458.0
FREQ_MAX = 10e9
#: rfx's own default when ``dx`` is not passed — see ``Simulation`` (``C0 /
#: freq_max / 20``). Asserted, not assumed, in every test below.
DX_DEFAULT = C0 / FREQ_MAX / 20.0          # 1.4990 mm
N_CELLS = 30
DOMAIN = (N_CELLS * DX_DEFAULT,) * 3
EPS_SLAB = 4.0


def _default_mesh_sim(**kw):
    """A CPML sim at rfx's DEFAULT mesh — ``dx`` deliberately not passed."""
    sim = Simulation(freq_max=FREQ_MAX, domain=DOMAIN, boundary="cpml",
                     cpml_layers=8, **kw)
    grid = sim._build_grid()
    assert grid.dx == pytest.approx(DX_DEFAULT, rel=1e-12), (
        f"fixture is not at the default mesh: grid.dx={grid.dx} != "
        f"{DX_DEFAULT} (= c0/freq_max/20). These gates are pinned at the "
        f"default mesh on purpose — the #655 notch reflection grows as dx "
        f"gets coarser (0.032 at 120 cells/lambda vs 0.238 at the default "
        f"20), so re-pinning them at a finer mesh would understate the "
        f"defect ~6x and could pass on a broken tree.")
    return sim


def _interior(eps, grid, axis):
    """1-D cut through the assembled eps_r along ``axis``, interior only,
    taken at a transverse position inside the structure."""
    pads = ((grid.pad_x_lo, grid.pad_x_hi), (grid.pad_y_lo, grid.pad_y_hi),
            (grid.pad_z_lo, grid.pad_z_hi))
    lo, hi = pads[axis]
    n = eps.shape[axis]
    mid = [eps.shape[0] // 2, eps.shape[1] // 2, eps.shape[2] // 2]
    mid[axis] = slice(lo, n - hi)
    return eps[tuple(mid)], (lo, hi)


# --------------------------------------------------------------------------
# 1. The array-level lock, at the DEFAULT mesh, on all three hi faces.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("axis,name", [(0, "x"), (1, "y"), (2, "z")])
def test_flush_box_reads_material_through_the_face_default_mesh(axis, name):
    """(#655) Every interior node up to AND INCLUDING the face must read the
    slab's material, with the pad unchanged from #627a."""
    lo = [4 * DX_DEFAULT] * 3
    hi = [26 * DX_DEFAULT] * 3
    lo[axis] = 0.0
    hi[axis] = DOMAIN[axis]                      # flush with the hi face
    sim = _default_mesh_sim()
    sim.add_material("slab", eps_r=EPS_SLAB)
    sim.add(Box(tuple(lo), tuple(hi)), material="slab")

    grid = sim._build_grid()
    eps = np.asarray(sim._assemble_materials(grid)[0].eps_r)
    cut, (pad_lo, pad_hi) = _interior(eps, grid, axis)

    # R5: the whole interior trace, not a single sampled node.
    assert not np.any(cut == 1.0), (
        f"{name}-hi: vacuum node(s) at interior index "
        f"{np.where(cut == 1.0)[0].tolist()} of {len(cut)} for a box flush "
        f"with the {name}-hi face. Full interior eps_r cut: {cut.tolist()}")
    assert float(cut[-1]) == EPS_SLAB, (
        f"{name}-hi: last interior node reads {float(cut[-1])}, expected "
        f"{EPS_SLAB}. Full interior cut: {cut.tolist()}")

    # #627a must be untouched: the pad still carries the material.
    pad_axis_hi = np.take(eps, eps.shape[axis] - 1, axis=axis)
    assert float(pad_axis_hi.reshape(-1)[pad_axis_hi.size // 2]) == EPS_SLAB, (
        f"{name}-hi pad lost its material — this is #627a's lock, not #655's")


def test_flush_box_lossy_and_magnetic_materials_reach_the_face():
    """(#655) sigma and mu_r travel with eps_r — the notch was never
    eps_r-only, and a lossy slab's conductivity dropping out at the last
    interior node is the same defect."""
    sim = _default_mesh_sim()
    sim.add_material("slab", eps_r=EPS_SLAB, sigma=0.05, mu_r=2.0)
    sim.add(Box((0.0, 0.0, 4 * DX_DEFAULT),
                (DOMAIN[0], DOMAIN[1], 26 * DX_DEFAULT)), material="slab")
    grid = sim._build_grid()
    mats = sim._assemble_materials(grid)[0]
    j, k = grid.pad_y_lo + 5, grid.pad_z_lo + 10
    last = grid.shape[0] - grid.pad_x_hi - 1
    assert float(np.asarray(mats.eps_r)[last, j, k]) == EPS_SLAB
    assert float(np.asarray(mats.sigma)[last, j, k]) == pytest.approx(0.05), (
        f"sigma at the last interior node is "
        f"{float(np.asarray(mats.sigma)[last, j, k])}, expected 0.05")
    assert float(np.asarray(mats.mu_r)[last, j, k]) == pytest.approx(2.0)


# --------------------------------------------------------------------------
# 2. The paired MUST-STILL-BE-VACUUM cases.
#
# Test 1 asserts "no vacuum at the face". The wrong way to satisfy it is to
# over-fire the repair and fill things that should be air. Every assertion
# above is therefore paired with a case on the SAME code path that must come
# out vacuum, so a blanket fill reds here.
# --------------------------------------------------------------------------

def test_inset_box_keeps_its_vacuum_gap_including_the_last_interior_node():
    """(#655 over-fire guard) A box that stops short of the face must keep
    EVERY vacuum node it drew, the last interior node included. #627a's
    fallback is bounded to one column inward for exactly this reason; the
    #655 repair inherits that bound and must not walk further."""
    sim = _default_mesh_sim()
    sim.add_material("slab", eps_r=EPS_SLAB)
    # 10 cells of air before the x-hi face, on every axis.
    sim.add(Box((4 * DX_DEFAULT,) * 3, (20 * DX_DEFAULT,) * 3), material="slab")
    grid = sim._build_grid()
    eps = np.asarray(sim._assemble_materials(grid)[0].eps_r)
    cut, _ = _interior(eps, grid, 0)
    assert float(cut[-1]) == 1.0, (
        f"last interior node was filled to {float(cut[-1])} across a genuine "
        f"multi-cell air gap. Interior cut: {cut.tolist()}")
    assert float(np.asarray(eps)[-1, eps.shape[1] // 2, eps.shape[2] // 2]) == 1.0, (
        "x-hi pad picked up material across a genuine air gap (#627a's bound)")


def test_two_cell_gap_at_the_face_is_not_bridged():
    """(#655 over-fire guard, the knife-edge neighbour of the real case) A box
    ending TWO cells short of the face leaves the outer column AND the column
    inside it vacuum, so the fallback condition is False and nothing is
    repaired — the bound that separates 'the rasterizer dropped one node' from
    'the user drew an air gap' is exactly one column wide."""
    sim = _default_mesh_sim()
    sim.add_material("slab", eps_r=EPS_SLAB)
    sim.add(Box((0.0, 0.0, 0.0),
                (DOMAIN[0] - 2 * DX_DEFAULT, DOMAIN[1], DOMAIN[2])),
            material="slab")
    grid = sim._build_grid()
    eps = np.asarray(sim._assemble_materials(grid)[0].eps_r)
    cut, _ = _interior(eps, grid, 0)
    assert float(cut[-1]) == 1.0 and float(cut[-2]) == 1.0, (
        f"a two-cell air gap at the x-hi face was bridged: last two interior "
        f"nodes read {float(cut[-2])}, {float(cut[-1])}. Cut: {cut.tolist()}")


# --------------------------------------------------------------------------
# 3. Primitive-agnostic: Sphere / Cylinder reach the same state by the
#    float32 knife edge rather than by half-openness.
# --------------------------------------------------------------------------

def test_repair_keys_off_the_assembled_array_not_off_box():
    """(#655) Primitive-agnosticism, locked deterministically.

    Calls the shared helper directly on a hand-built array carrying the notch
    pattern, so nothing here depends on which primitive drew it or on how any
    float comparison rounded. All three hi faces; the lo faces are included as
    controls because ``_extend_lo`` replicates the boundary node itself and so
    has no analogous failure mode.
    """
    import jax.numpy as jnp

    from rfx.geometry.rasterize_grid import extend_cpml_pad_materials

    pad = 4
    n = 2 * pad + 6                    # 6 interior nodes per axis
    for axis in (0, 1, 2):
        eps = np.ones((n, n, n), dtype=np.float32)
        sigma = np.zeros((n, n, n), dtype=np.float32)
        mu = np.ones((n, n, n), dtype=np.float32)
        # Material over every interior node EXCEPT the outermost one on
        # ``axis`` — exactly the state a dropped boundary node leaves.
        sl = [slice(pad, n - pad)] * 3
        sl[axis] = slice(pad, n - pad - 1)
        eps[tuple(sl)] = EPS_SLAB
        mu[tuple(sl)] = 2.0
        sigma[tuple(sl)] = 0.05

        out_eps, out_sigma, out_mu = extend_cpml_pad_materials(
            jnp.asarray(eps), jnp.asarray(sigma), jnp.asarray(mu),
            *([pad, pad] * 3))
        out_eps = np.asarray(out_eps)

        probe = [n // 2] * 3
        probe[axis] = n - pad - 1                      # last interior node
        assert float(out_eps[tuple(probe)]) == EPS_SLAB, (
            f"axis {'xyz'[axis]}: the dropped boundary node was not repaired "
            f"(reads {float(out_eps[tuple(probe)])}); the repair must key off "
            f"the array pattern, not off any particular primitive")
        assert float(np.asarray(out_sigma)[tuple(probe)]) == pytest.approx(0.05)
        assert float(np.asarray(out_mu)[tuple(probe)]) == pytest.approx(2.0)
        probe[axis] = n - pad                          # first pad node
        assert float(out_eps[tuple(probe)]) == EPS_SLAB, "#627a's pad lock"


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_face_touching_sphere_and_cylinder_reach_the_face(kind):
    """(#655) Integration cover for the closed-predicate primitives.

    ``Sphere`` and ``Cylinder`` use closed axis predicates (``r2 <=
    radius**2``), so unlike ``Box`` they do NOT drop their boundary node by
    rule — they drop it when the float32 knife edge documented as consequence
    (2) of the ``Box`` docstring rounds against them, which is mesh-dependent
    and not predictable from the nominal dimensions. Measured: at ``dx`` =
    1.000 mm both shapes below lose their boundary node; at rfx's default
    1.4990 mm neither does. This test is therefore pinned to the mesh where
    the drop was measured, and it VERIFIES the precondition from the shape's
    own raw mask (independent of the assembler) rather than assuming it — a
    version of this test written at the default mesh passed on the unfixed
    tree, i.e. it was binding nothing.

    The deterministic, platform-independent lock on primitive-agnosticism is
    ``test_repair_keys_off_the_assembled_array_not_off_box`` above; this one
    is the end-to-end confirmation.
    """
    dx = 1e-3
    n = 20
    dom = (n * dx,) * 3
    r = 6 * dx
    shape = (Sphere((dom[0] - r, dom[1] / 2, dom[2] / 2), r) if kind == "sphere"
             else Cylinder((dom[0] / 2, dom[1] / 2, dom[2] / 2), r, dom[0],
                           axis="x"))
    sim = Simulation(freq_max=FREQ_MAX, domain=dom, dx=dx, boundary="cpml",
                     cpml_layers=8)
    sim.add_material("slab", eps_r=EPS_SLAB)
    sim.add(shape, material="slab")
    grid = sim._build_grid()

    j = k = grid.shape[1] // 2
    last = grid.shape[0] - grid.pad_x_hi - 1
    raw = np.asarray(shape.mask(grid))
    if bool(raw[last, j, k]) or not bool(raw[last - 1, j, k]):
        pytest.skip(
            f"{kind}: the float32 knife edge did not drop the boundary node "
            f"on this platform/mesh (raw mask at the last interior node = "
            f"{bool(raw[last, j, k])}, one inside = "
            f"{bool(raw[last - 1, j, k])}), so this fixture cannot exercise "
            f"the #655 path here. The deterministic lock is "
            f"test_repair_keys_off_the_assembled_array_not_off_box.")

    eps = np.asarray(sim._assemble_materials(grid)[0].eps_r)
    col = eps[:, j, k]
    assert float(col[last]) == EPS_SLAB, (
        f"{kind}: raw mask dropped the boundary node and the assembler left "
        f"it vacuum ({float(col[last])}) while the x-hi pad reads "
        f"{float(col[-1])} — the #655 sandwich, reached without Box. "
        f"x-column: {col.tolist()}")


# --------------------------------------------------------------------------
# 4. The non-uniform lane carries the same defect through the same shared
#    helper, so it must carry the same repair.
# --------------------------------------------------------------------------

def test_flush_box_reads_material_through_the_face_nonuniform():
    """(#655) NU mirror. ``rfx/runners/nonuniform.py`` calls the same
    ``extend_cpml_pad_materials``; the fix lives once, so this must follow."""
    sim = _default_mesh_sim(dz_profile=[DX_DEFAULT] * N_CELLS)
    sim.add_material("slab", eps_r=EPS_SLAB)
    sim.add(Box((0.0, 0.0, 0.0), DOMAIN), material="slab")
    grid = sim._build_nonuniform_grid()
    eps = np.asarray(sim._assemble_materials_nu(grid)[0].eps_r)
    for axis, name in ((0, "x"), (1, "y"), (2, "z")):
        cut, _ = _interior(eps, grid, axis)
        assert not np.any(cut == 1.0), (
            f"NU {name}-hi: vacuum at interior index "
            f"{np.where(cut == 1.0)[0].tolist()}. Cut: {cut.tolist()}")


# --------------------------------------------------------------------------
# 5. The physics gate: the notch's reflection, at the default mesh.
#
# An array-shape assertion alone can be satisfied by an artifact. This one
# binds the quantity the issue is actually about — how much power a slab
# touching an absorbing face throws back — and it is pinned at the default
# mesh where that quantity is largest.
# --------------------------------------------------------------------------

def _reflection_1d(*, extend_half_cell, dx, n_steps, lz, f0):
    """1-D plane-wave FDTD: eps_r=4 fills the domain, periodic transverse.

    ``extend_half_cell`` draws the box half a cell PAST the domain face, which
    occupies the last interior node without changing the pad — the reference
    the notch is measured against.
    """
    sim = Simulation(freq_max=2.0 * f0, domain=(dx, dx, lz), dx=dx,
                     boundary=BoundarySpec(x="periodic", y="periodic",
                                           z="cpml"),
                     cpml_layers=10)
    sim.add_material("diel", eps_r=EPS_SLAB)
    sim.add(Box((-1.0, -1.0, 0.0),
                (1.0, 1.0, lz + (0.5 * dx if extend_half_cell else 0.0))),
            material="diel")
    sim.add_source((0.0, 0.0, 0.25 * lz), "ex",
                   waveform=GaussianPulse(f0=f0, bandwidth=0.8),
                   amplitude_kind="field")
    sim.add_flux_monitor(axis="z", coordinate=0.50 * lz,
                         freqs=np.array([f0]), name="refl")
    sim.add_probe((0.0, 0.0, 0.50 * lz), "ex")

    grid = sim._build_grid()
    eps = np.asarray(sim._assemble_materials(grid)[0].eps_r)[0, 0, :]
    last_interior = float(eps[len(eps) - grid.pad_z_hi - 1])
    pad = float(eps[len(eps) - grid.pad_z_hi])
    res = sim.run(n_steps=n_steps, compute_s_params=False,
                  skip_preflight=True, subpixel_smoothing=False)
    ts = np.asarray(res.time_series, dtype=float).ravel()
    return res.flux_monitors["refl"], ts, last_interior, pad


def test_face_touching_slab_does_not_reflect_off_a_phantom_film_default_mesh():
    """(#655) At rfx's DEFAULT mesh a slab flush with a CPML face reflected
    |r| = 0.238 (5.6 % of power, -12.5 dB) off the phantom one-cell film —
    matching thin-film theory for a vacuum layer one cell thick to 0.8 %.
    After the repair the same fixture must agree with the half-cell-extended
    reference, i.e. the reflection must collapse into the CPML floor.
    """
    f0 = FREQ_MAX
    dx = DX_DEFAULT
    lam0 = C0 / f0
    lz = 80 * dx
    n_steps = 6000

    fm_notch, ts_notch, li_n, pad_n = _reflection_1d(
        extend_half_cell=False, dx=dx, n_steps=n_steps, lz=lz, f0=f0)
    fm_ref, ts_ref, li_r, pad_r = _reflection_1d(
        extend_half_cell=True, dx=dx, n_steps=n_steps, lz=lz, f0=f0)

    # R5 witness: prove each variant is the configuration its label claims,
    # by printing the STATE, before trusting any number derived from it.
    witness = (f"notch-run last-interior-eps={li_n} pad={pad_n} | "
               f"reference-run last-interior-eps={li_r} pad={pad_r}")
    assert pad_n == EPS_SLAB and pad_r == EPS_SLAB, (
        f"pads are not both material, so the two runs differ by more than the "
        f"one interior node and the comparison is meaningless ({witness})")
    assert li_r == EPS_SLAB, (
        f"the half-cell-extended REFERENCE run did not occupy the last "
        f"interior node, so it is not a notch-free reference ({witness})")
    # NOTE: deliberately NO assert on ``li_n`` here. Asserting the production
    # fixture's own array state would fire FIRST on a broken tree and the |r|
    # gate below would never be reached — a gate that reds for the wrong
    # reason proves nothing about the quantity it claims to bind. ``li_n``
    # travels in the witness string instead, so the failure that does fire
    # carries the array state with it.

    # Ring-down settling witness (mandatory for CPML claims-bearing numbers).
    for tag, ts in (("notch", ts_notch), ("ref", ts_ref)):
        db = 20 * np.log10(max(np.abs(ts[-len(ts) // 20:]).max(), 1e-300)
                           / np.abs(ts).max())
        assert db < -40.0, (
            f"{tag} run has not settled: tail/peak {db:.1f} dB at "
            f"n_steps={n_steps}; the DFT is truncation-contaminated")

    # Field-level subtraction (flux is bilinear in E,H — subtract the E/H
    # accumulators, then form the flux; the repo's canonical R/T recipe).
    diff = fm_notch._replace(
        e1_dft=fm_notch.e1_dft - fm_ref.e1_dft,
        e2_dft=fm_notch.e2_dft - fm_ref.e2_dft,
        h1_dft=fm_notch.h1_dft - fm_ref.h1_dft,
        h2_dft=fm_notch.h2_dft - fm_ref.h2_dft)
    p_diff = float(np.asarray(flux_spectrum(diff, exact_f64=True))[0])
    p_inc = float(np.asarray(flux_spectrum(fm_ref, exact_f64=True))[0])
    r_amp = float(np.sqrt(abs(p_diff) / abs(p_inc)))

    thin_film = np.pi * dx * (EPS_SLAB - 1.0) / (lam0 * np.sqrt(EPS_SLAB))
    assert r_amp < 0.02, (
        f"|r| = {r_amp:.4f} off a slab flush with a CPML face at rfx's "
        f"DEFAULT mesh ({lam0/dx:.0f} cells/lambda0). Pre-#655 this measured "
        f"0.2377, against thin-film theory {thin_film:.4f} for a one-cell "
        f"vacuum film — i.e. the phantom film is back. {witness} "
        f"(p_diff={p_diff:.4e}, p_inc={p_inc:.4e})")
