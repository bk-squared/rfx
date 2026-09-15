"""Absorber and boundary-configuration preflight, moved verbatim out of
``rfx.api._preflight``.

Issue #980 Phase 3, leg 4. The absorber family: what the absorber is made of
(the #636 dispersive-pole advisory and the lossless-Q anti-pattern), how
thick it is (the per-face CPML thickness and the #647/#742 allocation-budget
advisory, over the shared ``_preflight_face_layers``), which boundary
combinations are refused (``pec_faces`` against finite PEC, UPML against
refinement, UPML against a mesh profile), and what is standing inside it
(probe/source placement and geometry extending into the pad). Everything
here was relocated byte for byte out of ``rfx/api/_preflight.py`` -- same
text, same order, same indentation, same docstrings, nothing renamed,
reordered, tidied or rewritten.

The move is gated on the committed advisory-text snapshot every
``sim.preflight()`` fixture renders
(``tests/locks/test_preflight_split_snapshot.py``), whose corpus was
extended first. The extension was chosen by a CALL CENSUS, as leg 3's was,
and it came back the mirror image of leg 3's answer. Leg 3 found a body no
fixture had ever entered. Here all eleven bodies were entered -- the family
hangs off ``_validate_simulation_config``'s unconditional spine, so 47 or 48
of the 48 fixtures called each one -- and two of them emitted nothing:
``_validate_cfg_absorber_budget_vs_grid``, whose advisory is reachable only
through a per-face ``BoundarySpec``, and
``_validate_cfg_upml_nonuniform_lane``, which needs ``upml`` together with a
mesh profile and a positive layer count. Both now have a fixture. The lock
module's own docstring carries the measurement.

Import contract, inherited from ``rfx.api._preflight``: import ONLY external
``rfx.*`` / stdlib / jax / numpy, never ``rfx.api`` -- that keeps
``rfx/api/__init__.py`` the sole composition point and the import graph
acyclic.

Unlike legs 1, 2 and 3 this module adds NOTHING to
``rfx/api/_preflight.py``'s re-export surface, because no module-level name
moves with it. The family's own leaves -- ``_absorber_boundary_for_axis``,
``_coord_in_absorber``, ``_coord_near_absorber``, ``_axis_pad_thickness_m``
and ``_ABSORBER_PROXIMITY_CELLS`` -- went to ``rfx/preflight/_common.py`` in
leg 0, precisely because their readers were never confined to this family:
``_check_msl_port_geometry`` reads three of them and
``_validate_cfg_ntff_absorber_overlap``, which leg 6 moved on to
``rfx/preflight/ntff.py``, reads a fourth. Measured over the eleven bodies below, every free module-level name
they hold is already in ``_common``: ``PreflightWarning``,
``PreflightConfigError``, ``_fmt_len``, ``_ABSORBER_PROXIMITY_CELLS``,
``_coord_in_absorber``, ``_coord_near_absorber`` and
``_absorber_boundary_for_axis``, plus ``math``, ``numpy`` and
``is_tracer``. So the facade's module namespace is untouched by this leg
rather than restored to 55 by a new import block, and the only surface that
changes is the class-scoped rebind.

One member of the family is shared and STAYS shared.
``_preflight_face_layers`` computes the allocated absorbing layers per face,
and outside the two absorber checks that own it three other bodies call it
by ``self.``: ``_validate_cfg_multiband_grading`` and
``_validate_cfg_nonuniform_limitations``, which left with leg 5 and live in
``rfx/preflight/mesh.py``, and ``_validate_cfg_pec_face_short_of_domain_wall``,
which left with leg 2 and lives in ``rfx/preflight/pec_geometry.py``. Every one of those is an
attribute lookup on the composed ``Simulation``, never a module-global read,
so moving the body here changes nothing for them: it is bound back onto
``_PreflightMixin`` and the MRO resolves it exactly as before. That is also
why it did NOT go to ``_common`` the way the leaf helpers did -- it is a
method that reads ``self._boundary_spec``, not a free function, and
``rfx/preflight/pec_geometry.py`` reaching it through ``self`` is what keeps
this module and that one from having to import each other.

``_validate_cfg_pec_boundary_open_structure`` is the one the split inventory
flagged as ambiguous, and it is here. Decided by reading it rather than by
its name: the body tests ``self._boundary == "pec"`` against ``self._ntff is
not None`` and advises ``boundary='cpml'``/``'upml'``. That is a statement
about which absorber the domain has, not about how a conductor rasterizes,
and it reads no ``pec_geometry`` module global -- it holds only
``PreflightWarning``. The NTFF condition is its trigger, not its subject.
"""

from __future__ import annotations

import math

import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.preflight._common import (
    _ABSORBER_PROXIMITY_CELLS,
    _absorber_boundary_for_axis,
    _coord_in_absorber,
    _coord_near_absorber,
    _fmt_len,
    PreflightConfigError,
    PreflightWarning,
)


def _validate_cfg_lossless_resonator_in_absorber(self, _w) -> None:
    """Warn when EVERY dielectric is perfectly lossless in an open (CPML/
    UPML) domain — design-guide Anti-Pattern #1: a lossless substrate in an
    open boundary yields an artificially infinite Q that reads as a
    plausible-but-wrong resonance (an R5 surface-metric trap detectable
    purely from setup). Deliberately narrow + single-shot to avoid noise:
    it fires only when no dielectric carries any loss, and is hedged
    because it is harmless if you are not measuring Q.
    """
    if self._boundary not in ("cpml", "upml"):
        return
    try:
        from rfx.api._spec import MATERIAL_LIBRARY
    except Exception:
        MATERIAL_LIBRARY = {}

    def _resolve(name):
        mspec = self._materials.get(name) if self._materials else None
        if mspec is not None:
            return (
                float(getattr(mspec, "eps_r", 1.0)),
                float(getattr(mspec, "sigma", 0.0)),
                bool(getattr(mspec, "debye_poles", None)
                     or getattr(mspec, "lorentz_poles", None)),
            )
        lib = MATERIAL_LIBRARY.get(name)
        if isinstance(lib, dict):
            return (
                float(lib.get("eps_r", 1.0)),
                float(lib.get("sigma", 0.0)),
                bool(lib.get("debye_poles") or lib.get("lorentz_poles")),
            )
        return None

    lossless_names: list[str] = []
    any_lossy_dielectric = False
    for entry in self._geometry:
        resolved = _resolve(entry.material_name)
        if resolved is None:
            continue
        eps_r, sigma, has_poles = resolved
        # Dielectric (not vacuum/air, not a conductor/PEC).
        if not (eps_r > 1.05 and sigma < 1.0):
            continue
        if sigma <= 0.0 and not has_poles:
            lossless_names.append(entry.material_name)
        else:
            any_lossy_dielectric = True

    if lossless_names and not any_lossy_dielectric:
        uniq = sorted(set(lossless_names))
        _w.warn(
            PreflightWarning(
                f"all dielectric(s) {uniq} are perfectly lossless in an open "
                f"({self._boundary.upper()}) domain. If you are measuring "
                f"Q / resonance, this gives an ARTIFICIALLY infinite Q "
                f"(design-guide Anti-Pattern #1, an R5 surface-metric trap) — "
                f"add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. "
                f"(Harmless if you are not measuring Q.)",
                code="lossless_q",
                source="_validate_cfg_lossless_resonator_in_absorber",
            ),
            stacklevel=2,
        )

def _validate_cfg_dispersive_pole_at_absorber_face(
    self, _w, dx: float,
    cpml_thick_lo: list[float], cpml_thick_hi: list[float],
) -> None:
    """Issue #636 advisory: a resonance-risk dispersive material
    touching a CPML/UPML face.

    The pad extension replicates only the STATIC eps_r/sigma/mu_r into
    the absorber (#627a) — dispersion-pole masks are deliberately not
    replicated (#627b, reverted) — so a dispersive structure that
    touches an absorbing face sees a pad matched at eps_inf only:
    band-limited impedance mismatch (in-band reflections) near the
    pole resonance. That is a fidelity limitation, not an
    instability; the shipped configuration is stable.

    Do NOT fix it by extending the pole masks into the pad. The #636
    one-attempt factorial (2026-08-29, b29f9de; predeclaration and
    results in docs/design_notes/i636_cpml_pole_pad_predeclaration.md,
    scripts in validation/research/cpml_pole_pad/) measured: naive
    extension diverges on a high-Q edge-touching Lorentz slab
    (last/mid-decile 5.03 at 20k steps vs 0.21 shipped, growing mode
    at 3.83 GHz inside the pole's eps<0 polariton gap with ~58% of
    |E| in the pads, peaked at the slab's z-interface), and a Drude
    (eps_inf=1) slab diverges even under the CFS-corner alpha rule
    with 12 layers (+5.2e-4/step at 60k steps, 2.64 GHz, inside its
    eps<0 band). The growing modes are interface (surface-polariton)
    waves of the eps(omega)<0 band living on the extended structure's
    boundaries inside the pad — a regime where stretched-coordinate
    PMLs violate the geometric stability condition — so no CPML
    parameter choice covered the factorial. Guarded by
    tests/unit/boundaries/test_cpml_pad_material_extension.py.

    Trigger (broadened by #808): ANY pole family — Debye, Lorentz of
    any Q, Drude — whose geometry touches a face that carries an
    absorber. The original filter warned only on the divergence-risk
    families (high-Q in-band Lorentz, Drude) and deliberately kept
    Debye quiet as noise; #808 then measured exactly that quiet
    configuration silently moving a committed Debye-recovery
    observable past its gate when the pad's statics-without-pole
    surround changed (bisect to #638; controlled pad-rule swap
    toggled the pinned and failing states digit-for-digit — see
    docs/design_notes/issue808_debye_pad_predeclaration.md). The pad
    state is an input-fidelity fact for every dispersive
    face-toucher, so every family now gets the advisory; the
    resonance-risk families additionally keep the #636 divergence
    wording. One aggregated advisory per simulation.
    """
    if self._boundary not in ("cpml", "upml"):
        return
    if not any(t > 0 for t in list(cpml_thick_lo) + list(cpml_thick_hi)):
        return

    w_band = 1.5 * 2.0 * np.pi * self._freq_max

    def _classify_poles(mat):
        """Every declared pole with a short label, plus whether any
        of them is in the #636 divergence-risk class (high-Q in-band
        Lorentz, or Drude). #808 broadened this from a filter that
        RETURNED only the risk class to a classifier over all
        families — the statics-without-pole pad state is a property
        of every dispersive face-toucher."""
        texts: list[str] = []
        risk = False
        for pole in (getattr(mat, "lorentz_poles", None) or ()):
            w0 = float(pole.omega_0)
            delta = float(pole.delta)
            if w0 == 0.0:
                texts.append("Drude")
                risk = True
                continue
            q_txt = (f"Q={w0 / (2.0 * delta):.0f}" if delta > 0
                     else "Q=inf")
            in_band = w0 <= w_band
            high_q = (delta == 0.0) or (w0 / (2.0 * delta) >= 10.0)
            if high_q and in_band:
                texts.append(q_txt)
                risk = True
            else:
                texts.append(q_txt + ("" if in_band else " out-of-band"))
        for pole in (getattr(mat, "debye_poles", None) or ()):
            texts.append(f"Debye tau={float(pole.tau):.3g}s")
        return texts, risk

    hits: list[tuple[int, str, str, str, bool]] = []
    for idx, entry in enumerate(self._geometry):
        try:
            mat = self._resolve_material(entry.material_name)
        except Exception:
            continue
        pole_txts, risk = _classify_poles(mat)
        if not pole_txts:
            continue
        if not hasattr(entry.shape, "bounding_box"):
            continue
        try:
            c1, c2 = entry.shape.bounding_box()
        except (NotImplementedError, TypeError):
            continue
        faces = []
        hi_exact = hi_over = False
        for ax in range(min(3, len(self._domain))):
            d = self._domain[ax]
            if cpml_thick_lo[ax] > 0 and c1[ax] <= 0.5 * dx:
                faces.append(f"{'xyz'[ax]}-lo")
            if cpml_thick_hi[ax] > 0 and c2[ax] >= d - 0.5 * dx:
                faces.append(f"{'xyz'[ax]}-hi")
                # Overdraw discriminates the realized hi-face state
                # (measured 2026-09-01): drawn PAST the face the shape
                # rasterizes its own cells into the absorber, so the
                # "pad stays background" wording is false there — the
                # message branches on this flag below.
                if c2[ax] > d:
                    hi_over = True
                else:
                    hi_exact = True
        if not faces:
            continue
        hits.append((idx, entry.material_name, "+".join(faces),
                     ", ".join(pole_txts), risk, hi_exact, hi_over))

    if not hits:
        return
    worst = hits[0]
    any_risk = any(h[4] for h in hits)
    # Each hi-face branch states the REALIZED state truthfully: a shape
    # ending AT the face loses its boundary node ([lo, hi) drop) and the
    # #808 gate keeps the pad background; a shape drawn PAST the face
    # rasterizes its own cells into the absorber, so the pad carries the
    # statics — with pole cells only up to the overdraw depth.
    any_hi_over = any(h[6] for h in hits)
    any_hi_exact = any(h[5] for h in hits)
    hi_clauses = []
    if any_hi_over:
        hi_clauses.append(
            "at a hi face drawn PAST the domain the shape rasterizes "
            "its own cells into the absorber, so the pad carries the "
            "material's statics (with pole cells only up to the "
            "overdraw depth; the deeper pad is statics-without-pole)"
        )
    if any_hi_exact or not any_hi_over:
        hi_clauses.append(
            "at a hi face ending at the domain face the pad stays "
            "background and the rasterizer's dropped boundary node is "
            "left unrepaired (#808 gate)"
        )
    msg = (
        f"Dispersive material '{worst[1]}' (geometry entry #{worst[0]}, "
        f"{worst[3]}) touches absorbing face(s) {worst[2]}. The realized "
        f"absorber there matches no declared material: a lo-face pad "
        f"replicates the material's STATIC eps/sigma/mu without its "
        f"dispersion poles (#627a; poles are deliberately not extended, "
        f"#627b), and " + ", and ".join(hi_clauses) + " — a band-limited "
        "impedance step instead of a matched continuation. "
    )
    if len(hits) > 1:
        msg += (
            f"{len(hits)} geometry entries are affected (first shown; "
            f"all in this finding's loc). "
        )
    msg += (
        "This is a fidelity limitation, not an instability. Do NOT "
        "extend pole masks into the pad to fix it: measured divergent "
        "(issue #636 factorial, "
        "docs/design_notes/i636_cpml_pole_pad_predeclaration.md — "
        "surface-polariton modes of the eps(omega)<0 band inside the "
        "absorber grow without bound; the Drude cell diverges even "
        "under the CFS alpha rule)"
    )
    if any_risk:
        msg += (
            "; this material's pole family is in that divergence-risk "
            "class, so the in-band mismatch is also resonance-sharp"
        )
    msg += (
        ". And do not extend the statics for it either: the "
        "eps_inf-without-pole surround silently moved a committed "
        "Debye recovery past its gate (issue #808, "
        "docs/design_notes/issue808_debye_pad_predeclaration.md). "
        "Mitigations: leave >= 1 cell of background material between "
        "the dispersive structure and the domain face, add pole "
        "damping (lower Q), or move the resonance out of band."
    )
    _w.warn(
        PreflightWarning(
            msg,
            code="dispersive_pole_at_absorber_face",
            loc="geometry[" + ",".join(
                f"#{h[0]} {h[2]} {h[3]}" for h in hits) + "]",
            source="_validate_cfg_dispersive_pole_at_absorber_face",
        ),
        stacklevel=2,
    )

def _validate_cfg_compute_cpml_thickness(
    self, cpml_thickness: float
) -> tuple[list[float], list[float], set]:
    """Per-face CPML thickness (2026-04). Mirrors Grid._face_pad:
    pec_faces / pmc_faces / periodic-axis faces consume 0 cells;
    remaining faces get the axis CPML thickness (non-uniform z
    aggregates the leading dz_profile entries). Under asymmetric
    composition (half-symmetric PMC + CPML, one-sided reflector)
    the lo and hi sides of a single axis can differ — the legacy
    symmetric scalar forced both sides to the max and produced
    false positives on the reflector face.

    Issue #647: the per-face LAYER COUNT now comes from
    :meth:`_preflight_face_layers`, which reads
    ``Boundary.lo_thickness`` / ``hi_thickness`` off the normalized
    ``_boundary_spec``. Before that, every absorbing face was reported
    at the global ``cpml_layers`` budget, so
    ``z=Boundary(lo='pec', hi='cpml', hi_thickness=2)`` with
    ``cpml_layers=16`` told every consumer the z_hi absorber was
    ``16*dx`` when the grid allocates ``2*dx`` — an 8x over-report that
    biases the MSL / waveguide clearance advisories (which use the
    magnitude as a calibrated buffer, not just as a boolean).

    Returns ``(cpml_thick_lo, cpml_thick_hi, _pmc_faces_set)``.
    """
    _pmc_faces_set = set(self._boundary_spec.pmc_faces())
    _face_layers = self._preflight_face_layers()
    # ``cpml_thickness`` is the BUDGET thickness (cpml_layers * dx, or 0
    # on a non-absorbing boundary); per-cell thickness is that divided
    # by the budget layer count.
    _per_layer = (cpml_thickness / self._cpml_layers
                  if self._cpml_layers else 0.0)

    def _face_thickness(ax_idx: int, side: str) -> float:
        ax_name = "xyz"[ax_idx]
        n_face = _face_layers[f"{ax_name}_{side}"]
        if n_face <= 0:
            return 0.0
        if (ax_name == "z"
                and self._dz_profile is not None
                and not is_tracer(self._dz_profile)):
            # Non-uniform z aggregates real cell sizes rather than
            # n*dx. The LEADING entries are used on both sides, as
            # before -- a hi-face trailing aggregation would be more
            # faithful on a graded profile but is a separate change
            # with its own regression surface.
            n = min(n_face, len(self._dz_profile))
            return float(sum(self._dz_profile[:n]))
        return n_face * _per_layer

    cpml_thick_lo = [_face_thickness(ax, "lo") for ax in range(3)]
    cpml_thick_hi = [_face_thickness(ax, "hi") for ax in range(3)]
    return cpml_thick_lo, cpml_thick_hi, _pmc_faces_set

def _validate_cfg_absorber_budget_vs_grid(self, _w, dx: float) -> None:
    """``cpml_layers`` wider than an axis of the grid it is applied to.

    Issue #647. ``cpml_layers`` is an ALLOCATION BUDGET shared by all
    six faces, and the CPML scratch buffers are cut from it on every
    axis — including axes that allocate no absorber at all. When the
    budget exceeds an axis's own cell count the face slices ``[:n]`` /
    ``[-n:]`` shrink to the axis while the length-``n`` coefficient
    profile does not, and the run used to die inside the scan with a
    broadcasting ``TypeError`` and NOTHING from preflight
    ("All checks passed" was the measured output on the reported
    fixture). ``rfx.boundaries.cpml`` now clamps the buffer per axis,
    which is exact — the clamped region carries no active absorber
    layers — so this is an advisory about a mis-sized number, not a
    rejection: the requested absorber is realized in full.

    Only reachable with a per-face ``BoundarySpec``. With a scalar
    ``boundary='cpml'`` every axis is padded by ``2*cpml_layers``, so
    the budget can never exceed an axis extent.

    The extent arithmetic mirrors ``rfx.grid.Grid.__init__``
    (``ceil(domain/dx) + 1 + pad_lo + pad_hi``) through
    :meth:`_preflight_face_layers`; it inherits that helper's
    waveguide-axis divergence, which makes this check UNDER-fire (never
    over-fire) on waveguide-port simulations.

    Issue #737/#742: skips any axis whose ``pad_lo`` and ``pad_hi`` are
    both 0 -- a PEC-closed or periodic-closed axis allocates no absorber
    on either face, so there is nothing for ``cpml_layers`` to exceed
    and the advisory was firing on a boundary condition it should never
    have been conditioned on. This adopts the allocation>0 convention
    already used by every OTHER consumer of :meth:`_preflight_face_layers`:
    :meth:`_validate_cfg_compute_cpml_thickness` (``if n_face <= 0:
    return 0.0``), the nonuniform z-thickness check below (``_z_layers
    > 0``, whose #647 comment states this identical rationale), and
    ``cpml_axes_eff`` in ``rfx.nonuniform`` (``if (lo + hi) > 0``) --
    this was the sole consumer that had not adopted it.
    """
    n_budget = int(self._cpml_layers or 0)
    if n_budget <= 0 or dx <= 0:
        return
    face_layers = self._preflight_face_layers()
    for ax_idx, ax_name in enumerate("xyz"):
        if ax_name == "z" and self._mode.startswith("2d"):
            continue
        extent_m = (self._domain[ax_idx] if ax_idx < len(self._domain)
                    else self._domain[-1])
        pad_lo = face_layers[f"{ax_name}_lo"]
        pad_hi = face_layers[f"{ax_name}_hi"]
        if pad_lo <= 0 and pad_hi <= 0:
            # Issue #737/#742: no allocation on either face of this
            # axis (PEC/PMC-closed or periodic-closed) -- nothing to
            # budget. See docstring for the allocation>0 precedent.
            continue
        n_cells = int(math.ceil(extent_m / dx)) + 1 + pad_lo + pad_hi
        if n_budget <= n_cells:
            continue
        axis_boundary = getattr(self._boundary_spec, ax_name)
        _w.warn(
            PreflightWarning(
                f"cpml_layers={n_budget} exceeds the {ax_name}-axis grid "
                f"extent ({n_cells} cells: "
                f"ceil({_fmt_len(extent_m)}/{_fmt_len(dx)})+1 interior "
                f"nodes + {pad_lo}/{pad_hi} absorber cells, from "
                f"{ax_name}=Boundary(lo={axis_boundary.lo!r}, "
                f"hi={axis_boundary.hi!r})). The absorber you asked for "
                f"is unaffected — the layer budget is clamped to the "
                f"axis for the CPML scratch buffers only, and the "
                f"clamped region carries no active absorber layers "
                f"(issue #647). It does mean the layer count was sized "
                f"for a coarser mesh or a larger domain than this one; "
                f"set that face's lo_thickness/hi_thickness explicitly "
                f"if you meant a thinner absorber.",
                code="absorber_budget_exceeds_axis",
                source="_validate_cfg_absorber_budget_vs_grid",
            ),
            stacklevel=3,
        )

def _preflight_face_layers(self) -> dict[str, int]:
    """Allocated absorbing layers per face — preflight's mirror of
    ``rfx.grid.Grid._face_pad`` (issue #647).

    Keyed off the normalized ``_boundary_spec``, which is correct for
    BOTH legacy scalar construction (``boundary='cpml'`` +
    ``pec_faces=`` + ``set_periodic_axes()`` are all folded into it by
    ``_build_spec_from_legacy``) and per-face ``BoundarySpec``
    construction. A face is allocated
    ``Boundary.resolved_{lo,hi}_thickness(cpml_layers)`` cells when its
    own token absorbs, and 0 otherwise — which is what makes PEC / PMC
    / periodic faces fall out without a separate rule.

    Known divergence from ``Grid._face_pad``, deliberately not
    mirrored: the grid drops non-port axes from ``cpml_axes`` when
    waveguide ports are present, so on such a simulation this
    over-reports the absorber on the non-port axes. That is the
    pre-existing behaviour every current warning is calibrated
    against; changing it belongs to a waveguide-lane change, not here.
    """
    n_default = int(self._cpml_layers or 0)
    out: dict[str, int] = {}
    spec = self._boundary_spec
    for ax_name, boundary in (("x", spec.x), ("y", spec.y),
                              ("z", spec.z)):
        for side in ("lo", "hi"):
            face = f"{ax_name}_{side}"
            token = getattr(boundary, side)
            if token not in ("cpml", "upml"):
                out[face] = 0
            elif ax_name == "z" and self._mode.startswith("2d"):
                # 2D modes collapse z to a single cell with NO absorber
                # (Grid sets pad_z_lo = pad_z_hi = 0 and strips z from
                # cpml_axes — rfx/grid.py). Without this mirror rule the
                # z thickness is the full cpml budget and every 2D
                # source/probe at z=0 false-trips absorber_overlap
                # (issue #166).
                out[face] = 0
            else:
                resolve = getattr(boundary, f"resolved_{side}_thickness")
                out[face] = int(resolve(n_default))
    return out

def _validate_cfg_pec_faces_with_finite_pec(self, _w) -> None:
    """Warn about pec_faces + finite PEC objects co-existing.

    pec_faces creates an INFINITE PEC boundary face across the whole
    domain side. Users building antennas or finite-GP structures
    often use pec_faces thinking it's a "ground plane" — but it's
    a full-domain boundary condition, not a finite structure.
    """
    if self._pec_faces and self._geometry:
        has_finite_pec = any(
            entry.material_name == "pec"
            for entry in self._geometry
        )
        if has_finite_pec:
            pec_face_list = ", ".join(sorted(self._pec_faces))
            _w.warn(
                PreflightWarning(
                    f"pec_faces={{{pec_face_list}}} creates an INFINITE PEC "
                    f"boundary AND the geometry contains finite PEC objects. "
                    f"For antennas or finite-GP structures, the pec_faces "
                    f"boundary makes the ground plane cover the entire domain "
                    f"face, which changes the physics (cavity vs radiating "
                    f"antenna). If you need a finite ground plane, remove "
                    f"pec_faces and declare it: a foil is a SHEET "
                    f"(add_thin_conductor, or a zero-thickness PEC Box — "
                    f"one node plane, normal E through it live); a thick "
                    f"plate is a PEC Box VOLUME (walls on both faces, "
                    f"lattice ownership contract #931).",
                    code="pec_faces_finite_pec",
                    source="_validate_cfg_pec_faces_with_finite_pec",
                ),
                stacklevel=3,
            )

def _validate_cfg_upml_refinement(self) -> None:
    """UPML boundary does not support subgridding/refinement."""
    if self._boundary == "upml" and self._refinement is not None:
        raise PreflightConfigError(
            "boundary='upml' does not support subgridding/refinement",
            code="upml_refinement",
            source="_validate_cfg_upml_refinement",
        )

def _validate_cfg_upml_nonuniform_lane(self, _w) -> None:
    """Advisory: ``boundary='upml'`` + a mesh profile is refused at run.

    Same shape as ``_validate_cfg_precision_x64``'s non-uniform arm:
    the enforcement point is ``_reject_upml_on_nonuniform`` at lane
    entry (``rfx/api/_execute.py``), which ``skip_preflight=True``
    does NOT bypass. This warning exists so the coming ``ValueError``
    is explained before the run reaches it, and so a
    ``skip_preflight=True`` caller still sees the reason in the one
    place they do look. Without it ``preflight()`` printed "All checks
    passed" and ``run()`` then refused, which reads as a preflight
    that does not know what the runner will do.

    The advisory carries its BASIS, not just its verdict, so a reader
    can tell a live guard from a stale one:

    * observed — on a 4x4x3 mm ez-dipole domain, two configs identical
      apart from the mesh profile: ``apply_upml_e`` ran 1x on the
      uniform lane and 0x on the non-uniform lane, while
      ``sim._boundary`` still read ``'upml'`` afterwards;
    * mechanism — ``rfx/nonuniform.py`` picks its absorber with
      ``use_cpml = grid.cpml_layers > 0`` (line 1051) and never reads
      the boundary type; every ``apply_upml_e``/``apply_upml_h`` call
      site is in the uniform scan body in ``rfx/simulation.py``. There
      is no UPML code on that lane to reach;
    * cost — CPML and UPML differ in reflection and in how material
      inside the pad is handled, so the run was a different absorber's
      result, and any post-hoc audit of ``sim._boundary`` reported the
      absorber that never ran;
    * alternative — ``boundary='cpml'`` runs what IS implemented here;
      dropping the mesh profile(s) reaches the uniform lane, which
      does implement UPML;
    * falsifier — this guard is stale the moment ``rfx/nonuniform.py``
      dispatches its absorber on the boundary type instead of on
      ``cpml_layers``, or an ``apply_upml_*`` call site appears
      outside ``rfx/simulation.py``. Both are one grep.

    Fires on exactly the condition ``_reject_upml_on_nonuniform``
    raises on (``boundary == 'upml'`` and ``cpml_layers > 0``), plus
    the mesh-profile test that its call sites supply — so the advisory
    and the error cannot disagree about which configs are refused.
    """
    is_nonuniform = (
        self._dx_profile is not None
        or self._dy_profile is not None
        or self._dz_profile is not None
    )
    if not is_nonuniform or self._boundary != "upml":
        return
    if self._cpml_layers <= 0:
        return
    _w.warn(PreflightWarning(
        "boundary='upml' was requested with a non-uniform mesh "
        "(dx/dy/dz profile set), but the non-uniform runner has no "
        "UPML code at all: rfx/nonuniform.py selects its absorber "
        "with `use_cpml = grid.cpml_layers > 0` and never reads the "
        "boundary type, and every apply_upml_e/apply_upml_h call site "
        "is in the uniform scan body (rfx/simulation.py). Measured "
        "before this was guarded (4x4x3 mm ez-dipole, configs "
        "identical apart from the mesh profile): apply_upml_e ran 1x "
        "on the uniform lane and 0x on the non-uniform lane while "
        "sim._boundary still read 'upml', so the run was CPML and "
        "even a post-hoc audit reported the absorber that never ran "
        "(issue #680). That is not a slightly worse UPML — CPML and "
        "UPML differ in reflection and in how material inside the pad "
        "is treated. Use boundary='cpml' to run the absorber that is "
        "implemented on this lane, or drop the mesh profile(s) to "
        "reach the uniform lane, which does implement UPML. "
        "run()/forward() raise ValueError at lane entry rather than "
        "proceed; this advisory explains that error in advance and "
        "covers skip_preflight=True, which does NOT bypass the lane "
        "guard. This guard is stale if rfx/nonuniform.py ever "
        "dispatches its absorber on the boundary type, or if an "
        "apply_upml_* call site appears outside rfx/simulation.py.",
        code="upml_nonuniform_lane_unsupported",
        source="_validate_cfg_upml_nonuniform_lane",
    ))


def _validate_cfg_absorber_placement(
    self,
    _w,
    dx: float,
    cpml_thickness: float,
    cpml_thick_lo: list[float],
    cpml_thick_hi: list[float],
    absorber_label: str,
) -> None:
    """P1.2/P1.3: Probe or source inside, or suspiciously close to, the
    absorber region.

    Issue #500: membership goes through :func:`_coord_in_absorber` —
    the requested domain ``[0, domain_extent]`` is absorber-free by
    construction (exterior padding), so a probe/port is only ever
    "inside" the absorber when its coordinate is genuinely outside
    that interval (previously this compared against an interior
    reading of the CPML thickness and false-fired on geometry
    anywhere within roughly the outer half of the thickness from an
    edge, e.g. a probe at the domain centre — verified false positive
    in #500 repro 2).

    Review finding M3 / H1: dropping the interior-frame comparison
    also removed the only proximity coverage the pre-#500 code
    happened to provide — a probe genuinely INSIDE the domain but
    right at its edge used to warn (for the wrong reason) and, after
    the #500 fix alone, went silent. Two regressions surfaced this as
    load-bearing: ``tests/unit/preflight/test_run_preflight_parity.py`` (a probe one
    cell inside a 0.02m domain was the run()-parity fixture's only
    warning trigger) and ``tests/unit/sparams/test_msl_internal_probe_advisories.py::
    test_user_probe_advisories_and_332_still_fire`` (the #470
    regression lock's "user probe near the x-CPML" case, node 9 of a
    pad=8 grid — one cell inside). :func:`_coord_near_absorber`
    restores this honestly: a coordinate that is interior but within
    ``_ABSORBER_PROXIMITY_CELLS`` cells of an active absorber boundary
    gets a distinct, lower-severity ``absorber_proximity`` advisory
    instead of being silently indistinguishable from a comfortably
    interior placement — fields that close to the boundary still
    carry CPML fringe/reflection error even though they are not
    literally inside the absorbing medium.
    """
    if cpml_thickness > 0:
        _internal = getattr(self, "_internal_probe_indices", frozenset())
        for _pi, pe in enumerate(self._probes):
            if _pi in _internal:
                # Library-registered diagnostic probes (e.g. the MSL
                # settling-witness probes, issue #470): the library
                # placed them deliberately and must not warn about
                # itself — that self-noise buried the genuine MSL
                # port-clearance advisories. User probes are
                # unaffected (precedent: passive DFT probes are
                # excluded from the geometry-extent check, #303).
                continue
            pos = pe.position
            for ax, coord in enumerate(pos):
                domain_extent = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                ax_i = min(ax, 2)
                ct_lo = cpml_thick_lo[ax_i]
                ct_hi = cpml_thick_hi[ax_i]
                if _coord_in_absorber(coord, domain_extent, ct_lo, ct_hi):
                    _w.warn(
                        PreflightWarning(
                            f"Probe at {pos} is near/inside {absorber_label} region "
                            f"({absorber_label} {'xyz'[ax]}-thickness: "
                            f"lo={_fmt_len(ct_lo)}, hi={_fmt_len(ct_hi)}). "
                            f"Signal will be attenuated. Move probe to interior.",
                            code="absorber_overlap",
                            source="_validate_cfg_absorber_placement",
                        ),
                        stacklevel=3,
                    )
                    break
                if _coord_near_absorber(coord, domain_extent, ct_lo, ct_hi, dx):
                    # Issue #510 nit 1: worded off the domain edge, not
                    # off "where the absorber begins" — the edge is
                    # exactly what _coord_near_absorber measures from
                    # (_absorber_boundary_for_axis's lo_b/hi_b), so this
                    # framing is unconditionally true. The prior wording
                    # ("within N cells of the {label} absorber") could
                    # overstate proximity to the absorbing MEDIUM itself,
                    # which (hi side only) can start up to one cell
                    # further out than the boundary this margin is
                    # measured from (see _absorber_boundary_for_axis's
                    # docstring, and nit 3 below). Review follow-up:
                    # "just past which" (not "where") the absorber is
                    # active, because _coord_in_absorber's own
                    # membership predicate is a STRICT less-than
                    # (coord < lo_b) -- the boundary coordinate itself
                    # reads as interior, so the absorber is active
                    # strictly beyond it, not exactly at it.
                    _w.warn(
                        PreflightWarning(
                            f"Probe at {pos} is within "
                            f"{_ABSORBER_PROXIMITY_CELLS} cells "
                            f"({_fmt_len(_ABSORBER_PROXIMITY_CELLS * dx)}) of the "
                            f"domain edge on the {'xyz'[ax]}-axis, just past which "
                            f"the {absorber_label} absorber is active. "
                            f"Fields there carry CPML fringe/reflection error; move "
                            f"inward for claims-bearing measurement.",
                            code="absorber_proximity",
                            source="_validate_cfg_absorber_placement",
                        ),
                        stacklevel=3,
                    )
                    break

        for pe in self._ports:
            pos = pe.position
            for ax, coord in enumerate(pos):
                domain_extent = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                ax_i = min(ax, 2)
                ct_lo = cpml_thick_lo[ax_i]
                ct_hi = cpml_thick_hi[ax_i]
                if _coord_in_absorber(coord, domain_extent, ct_lo, ct_hi):
                    _w.warn(
                        PreflightWarning(
                            f"Source/port at {pos} is near/inside {absorber_label} region "
                            f"({absorber_label} {'xyz'[ax]}-thickness: "
                            f"lo={_fmt_len(ct_lo)}, hi={_fmt_len(ct_hi)}). "
                            f"Energy will be absorbed. Move source to interior.",
                            code="absorber_overlap",
                            source="_validate_cfg_absorber_placement",
                        ),
                        stacklevel=3,
                    )
                    break
                if _coord_near_absorber(coord, domain_extent, ct_lo, ct_hi, dx):
                    # Issue #510 nit 1 — see the matching probe-loop
                    # comment above; same unconditionally-true
                    # domain-edge framing.
                    _w.warn(
                        PreflightWarning(
                            f"Source/port at {pos} is within "
                            f"{_ABSORBER_PROXIMITY_CELLS} cells "
                            f"({_fmt_len(_ABSORBER_PROXIMITY_CELLS * dx)}) of the "
                            f"domain edge on the {'xyz'[ax]}-axis, just past which "
                            f"the {absorber_label} absorber is active. "
                            f"Fields there carry CPML fringe/reflection error; move "
                            f"inward for claims-bearing measurement.",
                            code="absorber_proximity",
                            source="_validate_cfg_absorber_placement",
                        ),
                        stacklevel=3,
                    )
                    break


def _validate_cfg_geometry_in_cpml(
    self,
    _w,
    cpml_thickness: float,
    cpml_thick_lo: list[float],
    cpml_thick_hi: list[float],
    absorber_label: str,
) -> None:
    """P1.9: Geometry (dielectric OR PEC) extending into CPML region.

    CPML modifies field-update equations with absorbing coefficients;
    any structure placed there is effectively eaten by the absorber
    and produces physically meaningless results (issue #61).
    Periodic axes have no CPML (see _build_grid — issue #68), so
    the per-axis thresholds above already carry `cpml_thick_xyz[ax]
    == 0` on those axes and the check naturally skips.

    Issue #500 rewrite: the pre-#500 version treated the CPML as
    occupying ``[0, thick_lo]`` / ``[d - thick_hi, d]`` *inside* the
    requested domain (an "intentional full-domain edge" heuristic
    existed specifically to claw back the resulting false positives
    on the canonical transmission-line/MSL-substrate pattern —
    ``Box((0,0,0), (LX,LY,H_SUB))``). That frame was wrong: every rfx
    grid builder pads CPML EXTERIOR to the requested domain (see
    :func:`_absorber_boundary_for_axis`), so a Box entirely within
    ``[0, d]`` can never touch the absorber regardless of how close it
    sits to an edge — the heuristic is now unnecessary rather than
    merely refined. The only genuine issue-#61 case left is a Box
    whose bounding box extends to a NEGATIVE coordinate or past
    ``domain_extent`` — i.e. literally drawn into the exterior pad.

    Issue #660 reporting change: one warning per crossed AXIS naming the
    entry count, the worst offender and its overshoot distance, instead of
    one message per geometry entry that named neither.
    """
    if cpml_thickness > 0 and self._boundary == "cpml":
        # Issue #660: collect every crossing first, warn ONCE per axis.
        # The pre-#660 loop warned inside the entry loop with a message
        # carrying only the material name and the axis, so a 61-solid CAD
        # import emitted 61 lines of which 56 were byte-identical
        # (measured), and the overshoot distance — the one number that
        # separates a one-cell rounding artefact from an 11mm
        # coordinate-origin error — was computed and thrown away. The
        # crossed boundary and the offending bbox face are printed for the
        # same reason; ``_GeometryEntry`` carries no id/name field (only
        # ``shape`` + ``material_name``, verified), so the entry is
        # identified by its index and shape type, and the full per-entry
        # index list rides in the structured finding's ``loc``.
        per_axis: dict[int, list[tuple]] = {}
        for idx, entry in enumerate(self._geometry):
            if hasattr(entry.shape, "bounding_box"):
                try:
                    c1, c2 = entry.shape.bounding_box()
                    for ax in range(min(3, len(self._domain))):
                        thick_lo = cpml_thick_lo[ax]
                        thick_hi = cpml_thick_hi[ax]
                        if thick_lo <= 0 and thick_hi <= 0:
                            continue
                        d = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                        lo_b, hi_b = _absorber_boundary_for_axis(d, thick_lo, thick_hi)
                        over_lo = (
                            lo_b - c1[ax]
                            if (lo_b is not None and c1[ax] < lo_b) else None
                        )
                        over_hi = (
                            c2[ax] - hi_b
                            if (hi_b is not None and c2[ax] > hi_b) else None
                        )
                        if over_lo is None and over_hi is None:
                            continue
                        # A bbox can cross both faces (a shape wider than
                        # the domain); report the deeper crossing.
                        if over_hi is not None and (over_lo is None or over_hi >= over_lo):
                            over, side, coord, bound = over_hi, "hi", c2[ax], hi_b
                        else:
                            over, side, coord, bound = over_lo, "lo", c1[ax], lo_b
                        per_axis.setdefault(ax, []).append((
                            over, idx, entry.material_name,
                            type(entry.shape).__name__, side, coord, bound,
                        ))
                        # One finding per entry, first crossing axis —
                        # unchanged from pre-#660.
                        break
                except (NotImplementedError, TypeError):
                    pass

        for ax in sorted(per_axis):
            recs = per_axis[ax]
            axis = "xyz"[ax]
            over, idx, mat, kind, side, coord, bound = max(recs, key=lambda r: r[0])
            msg = (
                f"Material '{mat}' (geometry entry #{idx}, {kind}) extends "
                f"into CPML region along {axis}-axis: bbox {side} face at "
                f"{_fmt_len(coord)} is {_fmt_len(over)} past the "
                f"{axis}-{side} absorber boundary at {_fmt_len(bound)}."
            )
            if len(recs) > 1:
                msg += (
                    f" {len(recs)} geometry entries cross the {axis}-axis "
                    f"absorber (worst shown; overshoot "
                    f"{_fmt_len(min(r[0] for r in recs))} to "
                    f"{_fmt_len(over)}); per-entry index, face and "
                    f"overshoot in this finding's loc."
                )
            msg += (
                f" {absorber_label} modifies field updates — geometry "
                f"inside the absorber is physically meaningless (issue #61)."
            )
            # Per-entry detail lives here rather than in N warning lines:
            # every crossing entry's index, crossed face and overshoot.
            _w.warn(
                PreflightWarning(
                    msg,
                    code="geometry_in_absorber",
                    loc="geometry[" + ",".join(
                        f"#{r[1]} {r[4]} {_fmt_len(r[0])}" for r in recs
                    ) + "]",
                    source="_validate_cfg_geometry_in_cpml",
                ),
                stacklevel=3,
            )


def _validate_cfg_dielectric_at_absorber_seam(self, _w) -> None:
    """A dielectric reaches an absorber seam in a shape nothing can continue.

    The complement of ``_validate_cfg_geometry_in_cpml``. That check reports
    geometry standing INSIDE the absorber; this one reports geometry the pad
    material extension cannot carry INTO it, under a feature that says it
    puts it there. The extension continues a boundary-reaching structure
    outward -- the staircase lane by replicating the material arrays, the
    smoothed lane by continuing the SHAPE. A shape with no continuation
    across the reached face (a sphere, a cylinder reached across its axis, an
    imported mesh) gets neither on the smoothed lane and is then solved with
    vacuum in its own pad: the end facet #831 measured as ``|B/A| ~ 0.53`` on
    a straight guide, WORSENING with absorber depth.

    **It asks the continuation itself**, rather than re-deriving "reaches a
    padded face" from ``self._domain``. The first draft did re-derive it, and
    the two rules disagreed on a case neither author had in mind: a shape
    that CROSSES the boundary is unextendable per the builder (its bounding
    box reaches the face) and was invisible to a domain-window test, which
    asked whether the face landed inside ``[0, d]``. Two hand-written copies
    of one predicate is the #627 defect, not an implementation detail, so
    there is now one: :func:`rfx.geometry.smoothing.smoothed_shape_pairs`,
    the same call the three runner sites make, with the same PEC and
    dispersion filtering.

    Silent where nothing is lost or another check owns it: a Box and an
    axis-aligned Cylinder ARE continued; a PEC volume is continued by neither
    lane; a dispersive material at an absorber face is
    ``_validate_cfg_dispersive_pole_at_absorber_face``'s subject (pole masks
    are deliberately never continued -- #627b divergence, #808 promoted
    statics); and a traced mesh axis has no concrete node positions to test,
    the same precedent ``_validate_cfg_graded_box_rasterization`` sets.

    Declared vs solved, in input units: which entry reaches which face, and
    what permittivity the pad there will hold. It does not predict what that
    costs the result -- how much a facet reflects depends on the mode.
    """
    if self._boundary not in ("cpml", "upml") or not self._geometry:
        return
    if int(getattr(self, "_cpml_layers", 0)) <= 0:
        return

    profiles = (self._dx_profile, self._dy_profile, self._dz_profile)
    if is_tracer(self._dx) or any(
            p is not None and is_tracer(p) for p in profiles):
        return

    try:
        grid = (self._build_nonuniform_grid()
                if any(p is not None for p in profiles)
                else self._build_grid())
    except Exception:
        # Preflight never turns a configuration problem into a crash here:
        # whatever makes the grid unbuildable has its own check, and this
        # advisory has nothing to say without realized pads.
        return

    from rfx.geometry.smoothing import smoothed_shape_pairs
    try:
        _pairs, unextendable = smoothed_shape_pairs(self, grid)
    except Exception:
        return
    if not unextendable:
        return

    # The entry index and material name ride ON the finding. They used to be
    # recovered here by ``id(u.shape)`` against ``self._geometry``, which
    # misses exactly the cases worth naming: the continuation rewrites the
    # shape as it walks the axes, so a cylinder continued along x and
    # unextendable on y reports the CONTINUED object and the lookup fell
    # through to ``Material '?' (geometry entry #-1, ...)``.
    for u in unextendable:
        idx, mat_name = u.entry_index, u.material_name
        axis_name = "xyz"[u.axis]
        _w.warn(
            PreflightWarning(
                f"Material '{mat_name}' (geometry entry #{idx}, "
                f"{type(u.shape).__name__}) reaches the {axis_name}-{u.side} "
                f"absorber seam, and {u.reason}. With subpixel smoothing the "
                f"pad there is solved at eps_r = 1.0, not the declared "
                f"{u.eps_r:g}, so the structure is terminated by an end facet "
                f"at the interior/pad boundary (issue #1043). Move it clear "
                f"of the face, or declare it as a Box / axis-aligned "
                f"Cylinder, which are continued.",
                code="dielectric_at_absorber_seam",
                loc=f"geometry[#{idx}] {axis_name}-{u.side}",
                source="_validate_cfg_dielectric_at_absorber_seam",
            ),
            stacklevel=3,
        )


def _validate_cfg_pec_boundary_open_structure(self, _w) -> None:
    """P0.4: PEC boundary on likely open structure."""
    if self._boundary == "pec" and self._ntff is not None:
        _w.warn(
            PreflightWarning(
                "PEC boundary with NTFF far-field: PEC reflects all energy "
                "back into domain. Use boundary='cpml' or boundary='upml' for open structures "
                "(antennas, scatterers).",
                code="pec_boundary_open",
                source="_validate_cfg_pec_boundary_open_structure",
            ),
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly.
#
# Each of the eleven functions above was a ``def`` in the ``_PreflightMixin``
# class body, so its ``__qualname__`` read ``_PreflightMixin.<name>``; a
# module-level ``def`` gets the bare name instead. ``rfx/api/__init__.py``
# rewrites exactly ``<mixin>.<name>`` -> ``Simulation.<name>`` at
# class-composition time and SKIPS any function whose qualname does not match
# that pattern, so leaving the bare name here would change what a TypeError
# reports -- a user-visible behaviour change inside a pure code-motion step.
# ``tests/unit/autodiff/test_design_mask_removed.py
# ::test_no_public_simulation_method_leaks_a_mixin_class_name`` states the
# rule but only walks PUBLIC members, and all eleven names here are private,
# so ``tests/locks/test_preflight_split_snapshot.py`` pins these eleven
# directly.
#
# None of the eleven was a ``@staticmethod``, so like leg 3 this module has
# no decorator the facade has to re-apply and every restored qualname below
# becomes ``Simulation.<name>`` after composition.
# ---------------------------------------------------------------------------
_validate_cfg_lossless_resonator_in_absorber.__qualname__ = (
    "_PreflightMixin._validate_cfg_lossless_resonator_in_absorber"
)
_validate_cfg_dispersive_pole_at_absorber_face.__qualname__ = (
    "_PreflightMixin._validate_cfg_dispersive_pole_at_absorber_face"
)
_validate_cfg_compute_cpml_thickness.__qualname__ = (
    "_PreflightMixin._validate_cfg_compute_cpml_thickness"
)
_validate_cfg_absorber_budget_vs_grid.__qualname__ = (
    "_PreflightMixin._validate_cfg_absorber_budget_vs_grid"
)
_preflight_face_layers.__qualname__ = "_PreflightMixin._preflight_face_layers"
_validate_cfg_pec_faces_with_finite_pec.__qualname__ = (
    "_PreflightMixin._validate_cfg_pec_faces_with_finite_pec"
)
_validate_cfg_upml_refinement.__qualname__ = (
    "_PreflightMixin._validate_cfg_upml_refinement"
)
_validate_cfg_upml_nonuniform_lane.__qualname__ = (
    "_PreflightMixin._validate_cfg_upml_nonuniform_lane"
)
_validate_cfg_absorber_placement.__qualname__ = (
    "_PreflightMixin._validate_cfg_absorber_placement"
)
_validate_cfg_geometry_in_cpml.__qualname__ = (
    "_PreflightMixin._validate_cfg_geometry_in_cpml"
)
_validate_cfg_pec_boundary_open_structure.__qualname__ = (
    "_PreflightMixin._validate_cfg_pec_boundary_open_structure"
)
# Twelfth body, added here rather than moved (#1043 stage B): it was never in
# the class, so its qualname is set for the same reason -- the composition-time
# rewrite in rfx/api/__init__.py only promotes ``_PreflightMixin.<name>``.
_validate_cfg_dielectric_at_absorber_seam.__qualname__ = (
    "_PreflightMixin._validate_cfg_dielectric_at_absorber_seam"
)


#: #801's two measured points, and they are MEASUREMENTS, not a derived bound.
#: At n = 3 (dx = h/3) on the isolated-patch rig with a +10h laterally padded
#: domain: 6 absorber layers with the ground flush against the face GREW
#: (settling 0.00 dB, +8.5e-4 per step); 8 and 12 layers at the same dx and the
#: same geometry SETTLED (-44.2 / -46.0 dB); and 6 layers with every conductor
#: pulled 2 cells clear of the pads SETTLED (-42.8 dB). At n = 2, 4 layers grew
#: and 8 and 16 settled. Nobody has derived a stability boundary -- the
#: mechanism is not established -- so these are the edges of the measured
#: region and the advisory says so in its own text.
_THIN_ABSORBER_LAYER_FLOOR = 6
_THIN_ABSORBER_CLEARANCE_CELLS = 2


def _realized_clearance_cells(distance: float, dx: float) -> int:
    """Whole cells between a declared conductor face and the absorber boundary.

    Deliberately NOT :func:`_coord_near_absorber`, and the difference is the
    whole point. That helper answers a DECLARED-coordinate question -- is this
    coordinate within ``n_cells * dx`` of the boundary -- on the raw float. This
    one asks where the face RASTERIZES: a PEC volume's face realizes on the
    nearest node plane (lattice ownership contract #931 §1.1), so the clearance
    the solver sees is the rounded cell count.

    They disagree on exactly the arm this check was calibrated against. #801's
    rig carries a uniform -0.1-cell registration nudge, so a conductor inset by
    two cells declares its face at ``1.9 * dx``: ``_coord_near_absorber`` reads
    that as within two cells (it is, by 0.1 of one), while the solver rasterizes
    it onto node 2 and the arm SETTLES. Scoring the declared float would fire on
    the one geometry measured to be stable, which is the failure mode an
    advisory can least afford.
    """
    return int(np.rint(distance / dx))


def _validate_cfg_conductor_in_thin_absorber(self, _w, dx, absorber_label) -> None:
    """A conductor reaches an absorber face that has few layers (#801).

    Fires on a CONJUNCTION, because that is what was measured: a conductor
    whose realized face lands within
    ``_THIN_ABSORBER_CLEARANCE_CELLS`` cells of an absorbing face, AND that
    face carrying no more than ``_THIN_ABSORBER_LAYER_FLOOR`` layers. Either
    condition alone was stable in every arm measured; together they produced a
    ring-down that decayed to about -67 dB and then climbed monotonically --
    an eigenvalue of the update operator, not a transient.

    **What this is not.** It is not a stability bound. The mechanism behind the
    growth is NOT established: the measurements say which two conditions must
    hold together, not why the operator amplifies, and a 1-D CPML slice model
    does not reproduce it (it misses the arm that grows hardest and is ~190x
    too small where it fires). So the two constants are the edges of a measured
    region, and this is an ADVISORY rather than a refusal.

    **It will over-warn, knowingly.** The measured co-factor was a laterally
    PADDED domain; the one arm measured with no padding (4 layers, conductor at
    the face) settled. Padding is not part of the fire condition, because one
    stable arm does not bound a region either, and a check that silently
    assumed it would miss the padded cases this exists for. The message says
    so, and the remedy it names is the one that was measured to work.

    Silent where another check owns it or the input cannot carry the question:
    a graded mesh (``dz_profile`` and friends) has no single cell size at a
    face, so "clearance in cells" is not one number there -- the same precedent
    ``_validate_cfg_graded_box_rasterization`` sets for a traced axis; a
    non-absorbing face is already 0 layers out of :func:`_preflight_face_layers`
    and falls out without a rule; and geometry standing INSIDE the absorber is
    :func:`_validate_cfg_geometry_in_cpml`'s subject, which this does not
    duplicate -- a conductor that crosses the face reads as clearance 0 here and
    is reported by both, from the two sides it is wrong from.

    Declared vs solved, in input units: which conductor reaches which face, how
    many whole cells of clearance it realizes, and how many layers that face has.

    Related: #801.
    """
    if self._boundary not in ("cpml", "upml") or not self._geometry:
        return
    if dx is None or dx <= 0 or is_tracer(dx) or is_tracer(self._dx):
        return
    if any(getattr(self, name, None) is not None
           for name in ("_dx_profile", "_dy_profile", "_dz_profile")):
        return

    try:
        face_layers = self._preflight_face_layers()
    except Exception:
        # Whatever makes the boundary spec unreadable has its own check; this
        # advisory has nothing to say without per-face layer counts.
        return

    findings: list[tuple] = []
    for idx, entry in enumerate(self._geometry):
        mat_name = getattr(entry, "material_name", None)
        if mat_name != "pec":
            try:
                if self._resolve_material(mat_name).sigma < self._PEC_SIGMA_THRESHOLD:
                    continue
            except Exception:
                continue
        if not hasattr(entry.shape, "bounding_box"):
            continue
        try:
            c1, c2 = entry.shape.bounding_box()
        except Exception:
            continue
        for ax in range(min(3, len(self._domain))):
            axis_name = "xyz"[ax]
            extent = self._domain[ax]
            for side, distance in (("lo", float(c1[ax])),
                                   ("hi", float(extent) - float(c2[ax]))):
                layers = int(face_layers.get(f"{axis_name}_{side}", 0) or 0)
                if layers <= 0 or layers > _THIN_ABSORBER_LAYER_FLOOR:
                    continue
                cells = _realized_clearance_cells(distance, dx)
                if cells >= _THIN_ABSORBER_CLEARANCE_CELLS:
                    continue
                findings.append((idx, mat_name, type(entry.shape).__name__,
                                 axis_name, side, layers, cells, distance))

    for idx, mat_name, shape_name, axis_name, side, layers, cells, distance in findings:
        _w.warn(
            PreflightWarning(
                f"Conductor '{mat_name}' (geometry entry #{idx}, {shape_name}) "
                f"realizes {cells} cell(s) of clearance from the {axis_name}-{side} "
                f"{absorber_label} face (declared {_fmt_len(abs(distance))} "
                f"{'past' if distance < 0 else 'from'} it), and that face has "
                f"{layers} absorbing layer(s). Both at once is a measured growth "
                f"class (issue #801): on the isolated-patch rig at dx = h/3 with a "
                f"laterally padded domain, 6 layers with the ground flush against "
                f"the face grew (ring-down 0.00 dB, +8.5e-4 per step), while 8 and "
                f"12 layers at the same mesh and geometry settled (-44.2 / -46.0 dB) "
                f"and 6 layers with the conductors pulled 2 cells clear settled "
                f"(-42.8 dB). Either remedy removed it in every arm measured. These "
                f"are the EDGES OF THE MEASURED REGION, not a stability bound: the "
                f"mechanism is not established, which is why this advises rather "
                f"than refuses. The measured co-factor was the lateral padding, and "
                f"the one unpadded arm (4 layers, conductor at the face) settled, so "
                f"this may over-warn on an unpadded domain.",
                code="conductor_in_thin_absorber",
                loc=f"geometry[#{idx}] {axis_name}-{side}",
                source="_validate_cfg_conductor_in_thin_absorber",
            ),
            stacklevel=3,
        )


# Thirteenth body, added here rather than moved (#801): like the twelfth it was
# never in the class, so its qualname is set for the same reason -- the
# composition-time rewrite in rfx/api/__init__.py only promotes
# ``_PreflightMixin.<name>``.
_validate_cfg_conductor_in_thin_absorber.__qualname__ = (
    "_PreflightMixin._validate_cfg_conductor_in_thin_absorber"
)
