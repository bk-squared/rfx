"""Locks for CPML pad material extension (issue #627).

``rfx/api/_compile.py``'s ``_assemble_materials`` (mirrored on the
non-uniform mesh by ``rfx/runners/nonuniform.py``'s ``assemble_materials_nu``,
#582) extends the interior-edge ``eps_r``/``sigma``/``mu_r`` slice outward
into the CPML padding so guided modes see an impedance-matched absorber.
#627's review of that mirror found two gaps both assemblers inherited from
the (pre-existing) uniform path:

(a) For a ``Box`` whose hi face lands on (or past) the domain's last
    interior node, ``rfx.geometry.csg.Box``'s deliberately half-open
    ``[lo, hi)`` volume rasterization drops exactly that node from the
    box's own mask, so the naive interior-edge column for a hi-face pad
    read vacuum even though the structure's real material sits one column
    further in. Measured pre-fix on the #582 fixture: x-lo pad eps_r=4.0,
    x-hi pad eps_r=1.0, for a slab spanning the full x extent. **FIXED**
    here, in the shared ``rfx.geometry.rasterize_grid.extend_cpml_pad_materials``.
(b) Debye/Lorentz dispersion-pole masks are never extended into the pad
    at all (only the static eps_r/sigma/mu_r are), so a dispersive
    edge-touching material is impedance-matched at DC but not across the
    band. **NOT FIXED.** An earlier revision of this change extended pole
    masks the same way as (a); review's controlled discriminator found
    that turns a stable high-Q (Q~60) edge-touching Lorentz-slab
    simulation into a divergent one (20,000-step last/mid energy ratio
    649, vs 0.1557 for the static extension alone with the same pole
    left un-extended — no NaN, no exception, values just grow). Reverted
    in full (no flag, no partial path); tracked with the full factorial,
    mechanism hypothesis, and a separate pole-only-material coverage hole
    the attempted design also had, in issue #636.
    ``test_pole_extension_divergence_repro_636`` below (slow lane; the
    instability onset moved past 8,000 steps after #655 — see its
    docstring for the 2026-08-29 re-baseline) is the physics-level guard
    against silently reintroducing it; the mask-level tests red
    instantly in the fast lane.

The (a) fix is bounded to exactly one column inward on the hi-face
fallback (the rasterizer's per-box shortfall there is deterministically
one node) so a genuine multi-cell vacuum buffer between an interior
structure and the CPML pad — the overwhelmingly common case — is
untouched; that invariant is locked here too (test 4), since an earlier,
rejected design (an unbounded backward scan for "the last non-vacuum
column") would have silently bridged it.

(c) #808 (2026-09): for a POLE-CARRYING column, the hi-face fallback (a)
    promoted the statics anyway, so the pad — and after #655 the dropped
    boundary node itself — carried the material's eps_inf WITHOUT its
    poles: a material no declared model has. That surround moved the
    committed Debye-recovery observable past its gate (pinned 11% error
    -> 32%). The fallback and the #655 write are now gated on the
    combined dispersion-pole mask; lo-face statics still extend (the
    pinned envelope). Locked by the two
    ``test_pole_carrying_column_gets_no_hi_face_static_promotion_*``
    tests; pre-declaration and falsifiers in
    docs/design_notes/issue808_debye_pad_predeclaration.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import Simulation, GaussianPulse
from rfx.geometry.csg import Box
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole
from rfx.runners.nonuniform import build_nonuniform_grid, assemble_materials_nu

NA, NB, NZ = 45, 39, 4
DX = 1e-3
F0 = 3e9


def _build_uniform(*, dispersive: bool):
    sim = Simulation(freq_max=2.5 * F0, domain=(NA * DX, NB * DX, NZ * DX),
                      dx=DX, boundary="cpml", cpml_layers=8)
    if dispersive:
        sim.add_material("slab", eps_r=4.0,
                          debye_poles=[DebyePole(delta_eps=1.0, tau=1e-10)])
    else:
        sim.add_material("slab", eps_r=4.0)
    # Domain-face-touching in x AND y — the #627a trigger.
    sim.add(Box((0.0, 0.0, 0.3 * DX), (NA * DX, NB * DX, 2.0 * DX)), material="slab")
    return sim


def _assemble_uniform(sim):
    grid = sim._build_grid()
    materials, debye_spec, lorentz_spec, pec_mask, *_ = sim._assemble_materials(grid)
    return materials, debye_spec, grid.pad_x_lo, grid.pad_x_hi, grid.pad_y_lo, grid.pad_z_lo


def _assemble_nu(sim):
    grid = build_nonuniform_grid(
        sim._freq_max, sim._domain, sim._dx, sim._cpml_layers, None,
        dx_profile=np.full(NA, DX), dy_profile=np.full(NB, DX),
    )
    materials, debye_spec, lorentz_spec, pec_mask = assemble_materials_nu(sim, grid)
    return materials, debye_spec, grid.pad_x_lo, grid.pad_x_hi, grid.pad_y_lo, grid.pad_z_lo


def test_hi_face_pad_matches_lo_face_for_domain_touching_box_uniform():
    """(#627a) x-hi pad must carry the slab's eps_r, not vacuum."""
    sim = _build_uniform(dispersive=False)
    materials, _, plx, phx, ply, plz = _assemble_uniform(sim)
    eps = np.asarray(materials.eps_r)
    j, k = ply + 1, plz + 1  # interior cell, well away from the y-edge artifact
    x_lo = float(eps[plx - 1, j, k])
    x_hi = float(eps[-phx, j, k])
    assert x_lo == 4.0, x_lo
    assert x_hi == x_lo, (
        f"x-hi pad eps_r={x_hi} != x-lo pad eps_r={x_lo}: hi-face pad is not "
        f"impedance-matched (vacuum copied instead of the slab's material)")


def test_hi_face_pad_matches_lo_face_for_domain_touching_box_nu():
    """(#627a) NU mirror of the uniform-path lock above."""
    sim = _build_uniform(dispersive=False)
    materials, _, plx, phx, ply, plz = _assemble_nu(sim)
    eps = np.asarray(materials.eps_r)
    j, k = ply + 1, plz + 1
    x_lo = float(eps[plx - 1, j, k])
    x_hi = float(eps[-phx, j, k])
    assert x_lo == 4.0, x_lo
    assert x_hi == x_lo, (
        f"NU x-hi pad eps_r={x_hi} != x-lo pad eps_r={x_lo}")


def test_pole_carrying_column_gets_no_hi_face_static_promotion_uniform():
    """(#808) The dispersive twin of the lock above: for a POLE-CARRYING
    face-touching box, the hi-face fallback must NOT promote the statics.
    The promoted material would carry eps_inf without its poles — a
    material that exists in no declared model — and #808 measured that
    surround moving the committed Debye-recovery observable from its
    pinned 11% error to 32% (past its 20% gate), with a controlled
    pad-rule swap toggling the two states digit-for-digit. So the hi pad
    takes the background instead and the rasterizer's dropped boundary
    node stays unrepaired: the pre-#638 hi-face state the pin was
    measured with.

    The lo pad deliberately KEEPS the static copy. That behaviour
    predates #638, every committed dispersive envelope is pinned against
    it, and the #808 discriminator's identity arm (all pads background)
    measured the same recovery's tau at 64% error — "less pad material"
    is not automatically better.

    This is the F4 revert-probe guard of
    docs/design_notes/issue808_debye_pad_predeclaration.md: it reds if
    the pole gate in extend_cpml_pad_materials is reverted."""
    sim = _build_uniform(dispersive=True)
    materials, _, plx, phx, ply, plz = _assemble_uniform(sim)
    eps = np.asarray(materials.eps_r)
    nx = eps.shape[0]
    j, k = ply + 1, plz + 1
    assert float(eps[plx - 1, j, k]) == 4.0, (
        "x-lo pad must still carry the statics — the lo copy is the "
        "pinned envelope, not part of the #808 gate")
    assert float(eps[-phx, j, k]) == 1.0, (
        f"x-hi pad eps_r={float(eps[-phx, j, k])}: the hi-face fallback "
        f"promoted a pole-carrying column's statics into the pad — the "
        f"#808 eps_inf-without-pole surround is back")
    assert float(eps[nx - phx - 1, j, k]) == 1.0, (
        "the dropped hi-face boundary node was repaired with a pole "
        "material's statics — the #655 write must be gated for pole "
        "columns (#808)")


def test_pole_carrying_column_gets_no_hi_face_static_promotion_nu():
    """(#808) NU mirror of the lock above."""
    sim = _build_uniform(dispersive=True)
    materials, _, plx, phx, ply, plz = _assemble_nu(sim)
    eps = np.asarray(materials.eps_r)
    nx = eps.shape[0]
    j, k = ply + 1, plz + 1
    assert float(eps[plx - 1, j, k]) == 4.0
    assert float(eps[-phx, j, k]) == 1.0, (
        f"NU x-hi pad eps_r={float(eps[-phx, j, k])}: #808 gate missing "
        f"on the non-uniform lane")
    assert float(eps[nx - phx - 1, j, k]) == 1.0


def test_dispersion_poles_are_not_extended_into_the_pad_uniform():
    """(#627b, deliberately NOT fixed) Lock the reverted state: NEITHER
    pad gets the slab's Debye pole (the box's own rasterized mask never
    covers pad indices — they are outside [0, domain_extent) — and no
    extension step runs for pole masks, on either face, matching the
    original pre-#582 behaviour). If either count goes nonzero, someone
    re-added pole-mask extension — read the module docstring and
    ``test_pole_extension_stability_lock`` before doing that; the
    extension was reverted because it diverges for a high-Q pole (see
    that test and the CHANGELOG entry for #627). Contrast with the
    static eps_r, which DOES reach both pads (test 1/2 above) — that
    asymmetry is exactly what this test locks."""
    sim = _build_uniform(dispersive=True)
    materials, debye_spec, plx, phx, ply, plz = _assemble_uniform(sim)
    assert debye_spec is not None
    poles, masks = debye_spec
    mask = np.asarray(masks[0])
    n_pad_x_hi = int(mask[-phx:, :, :].sum())
    n_pad_x_lo = int(mask[:plx, :, :].sum())
    assert n_pad_x_lo == 0 and n_pad_x_hi == 0, (
        f"pole cells reached a CPML pad (x-lo={n_pad_x_lo}, "
        f"x-hi={n_pad_x_hi}) — pole-mask extension appears to have been "
        f"reintroduced without the stability question (see "
        f"test_pole_extension_stability_lock) being resolved first")


def test_dispersion_poles_are_not_extended_into_the_pad_nu():
    """(#627b) NU mirror of the uniform-path lock above."""
    sim = _build_uniform(dispersive=True)
    materials, debye_spec, plx, phx, ply, plz = _assemble_nu(sim)
    assert debye_spec is not None
    poles, masks = debye_spec
    mask = np.asarray(masks[0])
    n_pad_x_hi = int(mask[-phx:, :, :].sum())
    n_pad_x_lo = int(mask[:plx, :, :].sum())
    assert n_pad_x_lo == 0 and n_pad_x_hi == 0, (
        f"NU pole cells reached a CPML pad (x-lo={n_pad_x_lo}, "
        f"x-hi={n_pad_x_hi}) — see the uniform-path test's docstring")


def test_genuine_vacuum_buffer_before_cpml_is_not_bridged():
    """A structure that does NOT reach the domain edge (an ordinary interior
    box with several cells of air before the CPML pad — the overwhelmingly
    common simulation layout) must still see a plain-vacuum pad. This is the
    regression guard for the rejected "unbounded scan for the last
    non-vacuum column" design: that alternative would have smeared the
    interior structure's material across the intentional air gap into the
    absorber.
    """
    sim = Simulation(freq_max=2.5 * F0, domain=(NA * DX, NB * DX, NZ * DX),
                      dx=DX, boundary="cpml", cpml_layers=8)
    sim.add_material("slab", eps_r=4.0)
    # well inside the domain on every axis — at least 5 cells of vacuum
    # before any CPML pad on x/y, and centred on z
    sim.add(Box((10 * DX, 10 * DX, 0.3 * DX), (20 * DX, 20 * DX, 2.0 * DX)),
            material="slab")
    grid = sim._build_grid()
    materials, *_ = sim._assemble_materials(grid)
    eps = np.asarray(materials.eps_r)
    plx, phx = grid.pad_x_lo, grid.pad_x_hi
    ply, phy = grid.pad_y_lo, grid.pad_y_hi
    plz = grid.pad_z_lo
    k = plz + 1
    # Sample transverse positions INSIDE the box's own extent (box spans
    # interior indices 10..19 on both x and y), not the domain midpoint —
    # a transverse position outside the box is vacuum under every design,
    # including the rejected unbounded-scan one, so it cannot distinguish
    # them. plx+15/ply+15 sit inside [10,20) and are the positions an
    # unbounded backward scan (from the OPPOSITE face's pad) would walk
    # through and incorrectly find the box's material.
    assert float(eps[-phx, ply + 15, k]) == 1.0, (
        "x-hi pad picked up material across a genuine multi-cell vacuum "
        "gap — the hi-face fallback must be bounded to the rasterizer's "
        "documented one-column shortfall, not an unbounded scan")
    assert float(eps[plx + 15, -phy, k]) == 1.0, (
        "y-hi pad picked up material across a genuine multi-cell vacuum gap")


def test_uniform_and_nu_assemblers_stay_byte_identical_after_the_fix():
    """Extends #582's verified byte-identity property to the #627(a) fix:
    both assemblers must still agree exactly on eps_r/sigma/mu_r, and the
    (un-extended, per #627b's revert) pole masks must still agree with
    each other too.
    """
    for dispersive in (False, True):
        sim_u = _build_uniform(dispersive=dispersive)
        mat_u, debye_u, *_ = _assemble_uniform(sim_u)
        sim_n = _build_uniform(dispersive=dispersive)
        mat_n, debye_n, *_ = _assemble_nu(sim_n)

        assert np.array_equal(np.asarray(mat_u.eps_r), np.asarray(mat_n.eps_r))
        assert np.array_equal(np.asarray(mat_u.sigma), np.asarray(mat_n.sigma))
        assert np.array_equal(np.asarray(mat_u.mu_r), np.asarray(mat_n.mu_r))
        if debye_u is not None:
            _, masks_u = debye_u
            _, masks_n = debye_n
            assert np.array_equal(np.asarray(masks_u[0]), np.asarray(masks_n[0]))


def test_pole_extension_stability_lock():
    """Fast-lane canary for issue #636's fixture: the SHIPPED (statics-only)
    pad fill must decay on the high-Q edge-touching Lorentz slab.

    HISTORY / RE-BASELINE (2026-08-29, on b29f9de). The original #627b
    review measured, at 20,000 steps, last/mid-decile 649 for the
    pole-extended variant vs 0.1557 shipped, and at this test's 8,000
    steps 2.546 vs 0.4281 — so ratio < 1 at 8,000 steps then separated
    the variants and this test doubled as the physics-level guard against
    naive re-addition of pole-mask extension. Since c9c1864 (#655
    boundary-node fix) the instability ONSET HAS MOVED PAST 8,000 STEPS:
    re-measured on b29f9de, the pole-extended variant reads 0.4499 at
    8,000 steps (below 1 — this criterion no longer reds on naive
    re-addition) and 5.032 at 20,000 steps (still divergent), vs 0.2145
    shipped at 20,000 steps. The physics-level guard therefore lives in
    ``test_pole_extension_divergence_repro_636`` below (slow lane,
    20,000 steps, runs BOTH variants); the mask-level locks above red
    instantly on any naive re-addition in the fast lane.

    What this 8,000-step test still guards, cheaply: the shipped pad fill
    decaying on a resonant edge-touching interior — a ratio >= 1 here
    means the shipped extension has become unsafe for a resonant
    interior material.

    RE-BASELINE (2026-09-01, #808): the shipped rule now refuses the
    hi-face statics promotion for pole-carrying columns, so this
    fixture's x/y hi pads went from eps 4.0 to background and the
    measured envelope moved again: last/mid 0.4281 (pre-#655) -> 0.4499
    (b29f9de) -> 0.3979 (this change, x-hi pad eps=1.00 witnessed in the
    same run). Still decaying, as the #808 pre-declaration predicted —
    the gate only removes material from pads.
    """
    DX = 1e-3
    NA, NB, NZ = 45, 39, 12
    F0 = 3e9
    w0 = 2 * np.pi * F0
    STEPS = 8000

    sim = Simulation(freq_max=2.5 * F0, domain=(NA * DX, NB * DX, NZ * DX),
                      dx=DX, boundary="cpml", cpml_layers=8)
    sim.add_material("slab", eps_r=4.0,
                      lorentz_poles=[LorentzPole(omega_0=w0, delta=w0 / 120.0,
                                                  kappa=3.0 * w0 ** 2)])
    # edge-touching in x AND y (the #627a trigger); z is interior (3*DX to
    # 7*DX inside a 12*DX domain) so the resonance has real vacuum above
    # and below it to live in -- a thin-in-z fixture (as this author first
    # tried, and failed to reproduce the divergence with) starves the
    # resonance of an interior to couple through.
    sim.add(Box((0.0, 0.0, 3 * DX), (NA * DX, NB * DX, 7 * DX)), material="slab")
    sim.add_source((NA * DX / 3, NB * DX / 3, 5.0 * DX), "ez",
                    waveform=GaussianPulse(f0=F0, bandwidth=0.8),
                    amplitude_kind="field")
    sim.add_probe(((NA - 3) * DX, NB * DX / 2, 5.0 * DX), "ez")

    # R5 witness -- what actually landed in the pad for THIS run. A
    # mislabelled variant (pad contents not matching what the code claims
    # to do) is exactly what the #636 investigation caught once; printed
    # here (in the assertion messages) so a future failure shows it too.
    grid = sim._build_grid()
    materials, debye_spec, lorentz_spec, pec_mask, *_ = sim._assemble_materials(grid)
    eps = np.asarray(materials.eps_r)
    plx, phx = grid.pad_x_lo, grid.pad_x_hi
    ply, plz = grid.pad_y_lo, grid.pad_z_lo
    pole_pad_hi = pole_pad_lo = None
    if lorentz_spec is not None:
        _, masks = lorentz_spec
        pole = np.asarray(masks[0])
        pole_pad_hi = int(pole[-phx:, :, :].sum())
        pole_pad_lo = int(pole[:plx, :, :].sum())
    pad_witness = (
        f"x-hi pad eps={float(eps[-phx, ply + 5, plz + 5]):.2f} "
        f"x-hi pad pole cells={pole_pad_hi} x-lo pad pole cells={pole_pad_lo}")

    # Sanity check first: the shipped code must not extend pole masks at
    # all (issue #636). If this fails, the divergence assertion below is
    # not trustworthy either way -- fix this first.
    assert pole_pad_hi == 0 and pole_pad_lo == 0, (
        f"pole cells reached a CPML pad ({pad_witness}) -- pole-mask "
        f"extension appears to have been reintroduced; see issue #636 "
        f"before touching this")

    result = sim.run(n_steps=STEPS, compute_s_params=False,
                      skip_preflight=True, subpixel_smoothing=False)
    ts = np.asarray(result.time_series, dtype=float).ravel()
    n = len(ts)
    deciles = [float(np.abs(ts[i * n // 10:(i + 1) * n // 10]).max())
               for i in range(10)]
    last, mid = deciles[-1], deciles[3]
    ratio = last / max(mid, 1e-300)

    # Witness printed on pass too (R5): the measured envelope, so a
    # docstring re-baseline never needs a second instrumented run.
    print(f"\n#636 canary witness: last/mid={ratio:.4f} "
          f"deciles={[f'{d:.3e}' for d in deciles]} ({pad_witness})")

    assert np.isfinite(ts).all(), (
        f"non-finite field values ({pad_witness}) -- this fixture should "
        f"decay cleanly on shipped code")
    assert ratio < 1.0, (
        f"last-decile/mid-decile ratio {ratio:.4g} ({pad_witness}) -- the "
        f"shipped (statics-only) pad fill should have its last decile "
        f"below its mid-run decile on this high-Q edge-touching Lorentz "
        f"fixture (measured 0.3979 at this step count on the #808 change, "
        f"2026-09-01; 0.4281 in the original #636 discriminator; 0.1204 "
        f"at 20,000 steps on the #808 change). A ratio at or above 1 "
        f"means the shipped pad fill has become unsafe for a resonant "
        f"interior material -- see issues #636/#808 and "
        f"test_pole_extension_divergence_repro_636 before changing this "
        f"gate. deciles={deciles}")


class _PoleExtendedSim(Simulation):
    """Test-local harness for the #636 repro: replicate Lorentz/Debye pole
    masks into the CPML pads exactly the way the statics are replicated
    (including the #627a hi-face fallback), by piggybacking on
    ``extend_cpml_pad_materials`` with ``mask + 1`` as a fake eps array.
    This is the naive re-addition #627b tried and reverted; the shipped
    ``Simulation`` never does this. Mirrors
    ``validation/research/cpml_pole_pad/factorial_636.py``.
    """

    def _assemble_materials(self, grid, **kw):
        import jax.numpy as jnp
        from rfx.core.yee import MaterialArrays
        from rfx.geometry.rasterize_grid import extend_cpml_pad_materials

        out = super()._assemble_materials(grid, **kw)
        materials, debye_spec, lorentz_spec, *rest = out

        # #808: the shipped assembler now refuses the hi-face statics
        # promotion for pole-carrying columns, but the #636 factorial row
        # this harness measures is "statics AND poles both extended" —
        # without re-running the UNGATED statics extension here, the
        # variant would drift to the factorial's poles-over-vacuum-eps
        # row (measured catastrophically divergent to NaN), a different
        # documented state. Re-extending the finished arrays reproduces
        # the pre-#808 statics extension value-for-value: the gated cells
        # are exactly the ones still background, and the ungated rule
        # promotes them from the same inner columns it always used.
        if kw.get("include_cpml_pad_extension", True):
            _e, _s, _m = extend_cpml_pad_materials(
                materials.eps_r, materials.sigma, materials.mu_r,
                grid.pad_x_lo, grid.pad_x_hi,
                grid.pad_y_lo, grid.pad_y_hi,
                grid.pad_z_lo, grid.pad_z_hi)
            materials = MaterialArrays(eps_r=_e, sigma=_s, mu_r=_m)

        def ext_masks(spec):
            if spec is None:
                return None
            poles, masks = spec
            plx, phx = grid.pad_x_lo, grid.pad_x_hi
            ply, phy = grid.pad_y_lo, grid.pad_y_hi
            plz, phz = grid.pad_z_lo, grid.pad_z_hi
            new_masks = []
            for m in masks:
                fake_eps = m.astype(jnp.float32) + 1.0
                z = jnp.zeros_like(fake_eps)
                o = jnp.ones_like(fake_eps)
                e, _, _ = extend_cpml_pad_materials(
                    fake_eps, z, o, plx, phx, ply, phy, plz, phz)
                new_masks.append(e > 1.5)
            return (poles, new_masks)

        return (materials, ext_masks(debye_spec), ext_masks(lorentz_spec),
                *rest)


@pytest.mark.slow
def test_pole_extension_divergence_repro_636():
    """Minimal committed repro AND physics-level lock for issue #636:
    extending dispersion-pole masks into the CPML pad turns a stable
    high-Q edge-touching Lorentz-slab simulation into a divergent one,
    while the shipped statics-only pad fill decays — same fixture, same
    harness, only the pole-pad contents differ.

    Re-baselined measurements (2026-09-01, on the #808 change, this exact
    fixture, 20,000 steps, last/mid-decile of |ez|):

      shipped (poles NOT extended)      : 0.1204   decays
      poles + statics extended into pad : 5.032    grows (finite, no NaN)

    The shipped number moved 0.2145 (b29f9de) -> 0.1204 because #808
    gates the hi-face statics promotion for pole-carrying columns (this
    fixture's hi pads are background now); the extended variant is
    UNCHANGED at 5.032 because ``_PoleExtendedSim`` re-extends the
    statics ungated, reproducing the documented b29f9de "statics+poles"
    state value-for-value — which is itself the witness that the harness
    still measures the same factorial row. The original #627b-era margins
    (0.1557 vs 649 at 20,000 steps; 0.4281 vs 2.546 at 8,000) are
    historical: since c9c1864 (#655) the onset sits past 8,000 steps,
    which is why this lock runs 20,000 steps in the slow lane and why
    ``test_pole_extension_stability_lock`` (8,000 steps) no longer
    separates the variants. Root-cause envelope and the measured
    factorial live in docs/design_notes/i636_cpml_pole_pad_predeclaration.md
    and validation/research/cpml_pole_pad/.

    If the EXTENDED variant ever stops growing here, that is not noise:
    either the CPML/ADE composition changed (re-measure the #636 factorial
    before relying on it) or a deliberate fix landed — update this lock
    with the new measured margins in the same change.
    """
    DX = 1e-3
    NA, NB, NZ = 45, 39, 12
    F0 = 3e9
    w0 = 2 * np.pi * F0
    STEPS = 20000

    def build(cls):
        from rfx.geometry.csg import Box as _Box
        sim = cls(freq_max=2.5 * F0, domain=(NA * DX, NB * DX, NZ * DX),
                  dx=DX, boundary="cpml", cpml_layers=8)
        sim.add_material("slab", eps_r=4.0,
                         lorentz_poles=[LorentzPole(omega_0=w0,
                                                    delta=w0 / 120.0,
                                                    kappa=3.0 * w0 ** 2)])
        sim.add(_Box((0.0, 0.0, 3 * DX), (NA * DX, NB * DX, 7 * DX)),
                material="slab")
        sim.add_source((NA * DX / 3, NB * DX / 3, 5.0 * DX), "ez",
                       waveform=GaussianPulse(f0=F0, bandwidth=0.8),
                       amplitude_kind="field")
        sim.add_probe(((NA - 3) * DX, NB * DX / 2, 5.0 * DX), "ez")
        return sim

    def run_ratio(cls, expect_pad_poles):
        sim = build(cls)
        grid = sim._build_grid()
        _, _, lorentz_spec, *_ = sim._assemble_materials(grid)
        _, masks = lorentz_spec
        pole = np.asarray(masks[0])
        phx, plx = grid.pad_x_hi, grid.pad_x_lo
        n_pad = int(pole[:plx].sum() + pole[-phx:].sum())
        if expect_pad_poles:
            assert n_pad > 0, "harness failed to place poles in the pad"
        else:
            assert n_pad == 0, (
                f"shipped code put {n_pad} pole cells in the x pads -- "
                f"pole-mask extension appears to have been reintroduced")
        result = sim.run(n_steps=STEPS, compute_s_params=False,
                         skip_preflight=True, subpixel_smoothing=False)
        ts = np.asarray(result.time_series, dtype=float).ravel()
        assert np.isfinite(ts).all()
        n = len(ts)
        deciles = [float(np.abs(ts[i * n // 10:(i + 1) * n // 10]).max())
                   for i in range(10)]
        return deciles[-1] / max(deciles[3], 1e-300), deciles

    ratio_shipped, dec_s = run_ratio(Simulation, expect_pad_poles=False)
    ratio_extended, dec_e = run_ratio(_PoleExtendedSim, expect_pad_poles=True)

    print(f"\n#636 repro witness: shipped last/mid={ratio_shipped:.4g}, "
          f"extended last/mid={ratio_extended:.4g}")

    assert ratio_shipped < 1.0, (
        f"shipped variant no longer decays at 20,000 steps: last/mid = "
        f"{ratio_shipped:.4g} (measured 0.1204 on the #808 change, "
        f"2026-09-01; 0.2145 on b29f9de before the #808 gate); "
        f"deciles={dec_s}")
    assert ratio_extended > 1.0, (
        f"pole-extended variant no longer grows at 20,000 steps: last/mid "
        f"= {ratio_extended:.4g} (measured 5.032 on b29f9de). If a fix for "
        f"#636 landed deliberately, update this lock with new margins; "
        f"otherwise re-run the #636 factorial "
        f"(validation/research/cpml_pole_pad/) before trusting this. "
        f"deciles={dec_e}")
