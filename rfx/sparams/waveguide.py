"""Waveguide S-matrix calculators, moved verbatim out of ``rfx.api._sparams``.

Issue #980 Phase 2. ``compute_waveguide_s_matrix`` and its non-uniform-mesh
sibling ``_compute_waveguide_s_matrix_nu`` were two methods of the
``_SparamMixin`` class body in ``rfx/api/_sparams.py``; they are relocated here
byte for byte, dedented by exactly four spaces and otherwise untouched — same
signatures, same docstrings, same logic, nothing renamed, reordered or cleaned
up. The move is gated on bit identity of the extracted S arrays
(``tests/locks/test_sparams_split_bit_identity.py``).

They are MODULE-LEVEL FUNCTIONS whose first parameter is still ``self``: it is
the ``Simulation`` instance, exactly as before. ``rfx.api._sparams`` binds them
back onto ``_SparamMixin`` as class attributes at the same place in the class
body, so ``sim.compute_waveguide_s_matrix(...)`` keeps its name, signature,
docstring and bound-method behaviour, and ``_compute_waveguide_s_matrix_nu``
keeps writing ``self._waveguide_ports`` on that same instance. (``__qualname__``
is restored at the foot of this module; ``inspect.getdoc`` is byte-identical to
the pre-move value, while the raw ``__doc__`` differs by exactly the four-space
dedent, which is what dedenting a class body costs.) A class wrapper may follow
in a later step.

Import contract, inherited from ``rfx.api._sparams``: import ONLY
``rfx.api._spec`` plus external ``rfx.*`` / stdlib / jax / numpy, never the
``rfx.api`` package itself, so ``rfx/api/__init__.py`` stays the sole
composition point and the import graph stays acyclic.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.sources.waveguide_port import (
    extract_waveguide_s_matrix,
    extract_waveguide_s_matrix_flux,
    extract_waveguide_s_params_normalized,
    extract_multimode_s_matrix,
    extract_multimode_s_matrix_flux,
    waveguide_plane_positions,
)

from rfx.api._spec import (
    WaveguideSMatrixResult,
    _WaveguidePortEntry,
)

from rfx.sparams._common import (
    _assert_nu_shift_span_in_one_grading_zone,
    _finalize_sparam_result,
    _warn_if_ringdown_truncated,
    _warn_junction_probe_clearance,
    _warn_junction_cpml_thickness,
    _warn_ntff_box_dropped,
    _warn_thin_absorber_vs_guide_wavelength,
    _waveguide_s21_phase_residual,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    # Type-only forward reference for the
    # ``port_reference_sims: "list[Simulation] | None"`` annotation, carried
    # over unchanged from ``rfx.api._sparams``. NOT a runtime import: the
    # module-import contract above forbids a runtime ``from rfx.api import
    # ...``, and a TYPE_CHECKING guard never executes.
    from rfx.api import Simulation


def compute_waveguide_s_matrix(
    self,
    *,
    n_steps: int | None = None,
    num_periods: float = 20.0,
    normalize: bool | str = False,
    subpixel_smoothing: bool | str = False,
    eps_override: "jnp.ndarray | None" = None,
    sigma_override: "jnp.ndarray | None" = None,
    checkpoint_segments: int | None = None,
    strict_passivity: bool = False,
    port_reference_sims: "list[Simulation] | None" = None,
) -> WaveguideSMatrixResult:
    """Compute a theoretically clean axis-normal boundary-aperture waveguide S-matrix.

    Parameters
    ----------
    num_periods : float
        Length of the FDTD run (in source-period multiples) used to
        derive ``n_steps`` when ``n_steps`` is not given. The
        spectra are computed POST-SCAN from the recorded modal V/I
        time series via a rectangular full-record DFT (matching
        OpenEMS's ``utilities.DFT_time2freq``); ``num_periods``
        therefore governs both the CPML drain horizon AND the DFT
        integration window. Phase 2 cleanup (2026-04-25) removed
        the legacy ``num_periods_dft`` early-gate knob — the rect
        full-record DFT is finite-energy on the recorded transient
        so no gating is needed even on strong reflectors.
    normalize : bool or "flux"
        Controls the S-parameter extraction algorithm:

        ``False`` (default) — modal V/I decomposition, no reference
        run.  Magnitude includes Yee impedance mismatch
        (Z_TE_num/Z_TE_exact ≈ 3 % at dx/λ = 0.07).  Use for
        |S11| of strong reflectors (PEC short, high-Q resonators)
        where this error is smaller than the ±10–20 % round-trip
        dispersion error introduced by ``normalize=True``.  On the
        differentiable chain: a traced ``eps_override`` /
        ``sigma_override`` flows through it.

        ``True`` — two-run modal normalization.  Cancels one-way
        Yee dispersion for **transmission** (off-diagonal) by
        dividing device outgoing waves by reference outgoing waves
        at the same port.  **Does not** cancel dispersion for
        reflection (round-trip vs one-way path mismatch); use
        ``normalize=False`` or ``normalize="flux"`` for S11 of
        strong reflectors.  Host-assembled and **outside the
        differentiable chain**: a traced ``eps_override`` /
        ``sigma_override`` (``jax.grad``, ``jax.jit``) raises
        ``NotImplementedError`` at this call, before any FDTD run;
        a concrete override still runs forward.

        ``"flux"`` — hybrid power-flux extraction.  Magnitude from
        Poynting-vector DFT (|S|² = P_flux / P_inc), phase from
        modal V/I.  Corrects both the Z_TE impedance-mismatch error
        in S11 and the round-trip dispersion error in the
        ``normalize=True`` diagonal formula.  Costs 2 × N_ports
        FDTD runs (same as ``normalize=True``).  On the
        differentiable chain like ``False``.  The result dtype
        follows the ``freqs`` precision — complex64 by default,
        complex128 under ``JAX_ENABLE_X64`` — the same rule as
        ``False`` (a hard complex64 cast on this lane was removed
        in v1.8).

        **Reference impedance.**  On ``False`` and ``True`` each
        S_ij is the modal voltage-wave ratio ``b_i / a_j``, where
        the waves at a port are decomposed with that port's OWN
        discrete modal impedance (Yee-discrete Z_TE or Z_TM from
        the port's cutoff and cell size); no ``sqrt(Z_i / Z_j)``
        renormalization is applied.  ``"flux"`` takes |S_ij| from
        the power ratio and its phase from the same modal waves.
        The returned matrix therefore equals a power-wave S only
        when every port shares one cross-section and mode — the
        validated scope (straight guides and same-guide junctions).
        Ports with dissimilar cross-sections are outside it.

        **Multimode.**  ``n_modes > 1`` on any port is assembled on
        the host for every ``normalize`` value and is outside the
        differentiable chain; ``eps_override`` / ``sigma_override``
        differentiation is single-mode only.
    checkpoint_segments : int or None
        Segmented gradient checkpointing for the **uniform** waveguide
        AD path (issue #73 / PR #125).  Splits the ``n_steps`` scan
        into ``K`` segments that are rematerialised via
        ``jax.checkpoint`` during the backward pass, reducing peak
        reverse-mode memory from O(n_steps·|carry|) to
        O((K + n_steps/K)·|carry|) (≈ O(√n_steps·|carry|) at the
        optimal K ≈ √n_steps, at ≈ 2× backward compute cost).
        ``K`` is forwarded to ``rfx.simulation.run`` with
        ``checkpoint=True``; ``K`` MUST exactly divide the
        auto-computed ``n_steps`` (the runner rejects non-divisors —
        choose the nearest divisor of √n_steps; padding is rejected
        because it would shift the V/I DFT windows).  Default
        ``None`` is byte-identical to the pre-checkpoint scan.

        On a NON-uniform mesh (``dx_profile`` / ``dy_profile`` /
        ``dz_profile``, issue #73) ``checkpoint_segments=K`` is now supported: ``K`` is
        translated to the NU runner's ``checkpoint_every`` chunk size — the
        divisor of ``n_steps`` nearest to ``n_steps/K`` — and applied to the
        *device* run only (the vacuum reference is constant in the design
        variable). The chunk MUST divide ``n_steps`` (same as the uniform
        path): a non-divisor chunk would let the NU runner's zero-padding add
        spurious ring-down steps to the carry-accumulated flux DFT and shift
        the S-matrix. With an exact divisor the result is forward-IDENTICAL
        and the ≈O(√n_steps) tape reduction is realised under ``jax.grad``
        with ``normalize='flux'`` + an ``eps_override`` / ``sigma_override``
        design variable.
    port_reference_sims : list[Simulation] or None
        Per-driven-port matched-straight-guide reference simulations for
        **interior-PEC multi-port** structures (T-junctions, branches,
        septa). ``port_reference_sims[i]`` is a ``Simulation`` describing
        the STRAIGHT continuation of driven port ``i``'s guide with **no
        junction** — same domain / ``dx`` / boundary, geometry = port
        ``i``'s guide walls extended straight through. Only valid with
        ``normalize='flux'`` (raises otherwise); single-mode ports only;
        uniform mesh only; not combinable with ``eps_override`` /
        ``sigma_override``.

        **Multi-port junctions (port_reference_sims).** The default flux
        path references incident power ``P_inc`` to a single shared VACUUM
        run (empty domain, no interior PEC). For a straight guide the walls
        come from ``pec_axes`` so that vacuum reference already carries the
        guided ``P_inc`` correctly. For a junction the interior PEC septum /
        branch is stripped from the vacuum reference, which then radiates
        into free space: ``P_inc`` is mis-normalized and every ``|S|``
        inflates hard (non-passive; ``normalize=True`` gave max|S|~230,
        ``normalize='flux'`` gave max|S|~9.8, |S11|~1.9 on the verified
        compact 3-port T-junction). Passing ``port_reference_sims[i]`` = the
        matched straight guide for driven arm ``i`` moves ``P_inc`` toward
        the true single-mode guided incident power (PEC-folded identically
        to the device run).

        **Far-port discipline (required for a physical S-matrix).** This
        plumbing is NECESSARY but NOT SUFFICIENT. A physical junction
        S-matrix additionally requires: (1) each port's probe plane placed
        >= 5 evanescent decay lengths of the next higher mode away from the
        junction; (2) CPML thickness >= ~0.5 guide wavelengths at band
        centre; (3) a converged mesh. On far-port geometry (arms 90/90/70 mm,
        48 mm CPML, dx 1.0/0.667 mm) the matched-reference flux path reaches
        passivity 1.006/1.002, reciprocity 0.001, mesh-convergence 0.0297
        and 0.087 vs MEEP (r2000, cross-device). On COMPACT geometry the
        matched reference fixes |S11| (1.86 → 0.49, physical) but the overall
        matrix stays non-physical (residual max|S|~3.9); the two in-method
        advisories (probe clearance, CPML thickness) warn when the far-port
        discipline is not met.

        The absorber half of that discipline is checked on **every**
        uniform path, not just this one: an in-method advisory (issue
        #494) fires whenever the absorber on a port's propagation axis is
        thinner than ``0.5 * lambda_g`` at the **lowest** measured
        frequency, which is where ``lambda_g`` is longest and the
        ``cpml_layers=16`` default weakest. It is checked in-method
        because the functional entry points run no ``sim.preflight()``.
        Treat ``0.5 * lambda_g`` as a floor, not a target: at the WR-90
        band edge the measured residual ``|S11|`` ripple was 0.0706 at
        0.30 ``lambda_g``, 0.0366 at 0.50 and 0.0093 at 0.75, so a
        0.5-``lambda_g`` absorber can still set the accuracy envelope
        instead of discretization. The advisory is silent on a
        non-uniform mesh (``cpml_layers * dx`` is ambiguous under a
        graded profile) and on a band that starts at or below cutoff
        (``lambda_g`` is undefined there; the ``port_freqs_below_cutoff``
        preflight owns that case).

        This enables junction measurements UNDER the documented
        discipline; it does NOT make arbitrary compact junctions
        valid. See the skipped ``test_api.py`` T-junction reciprocity test
        and the companion evidence gate test
        ``tests/crossval/test_waveguide_tjunction_e4e5_gates.py``.
    """
    if not normalize:
        import warnings
        warnings.warn(
            "compute_waveguide_s_matrix(normalize=False): S21 and "
            "S-parameter phase include Yee numerical dispersion. "
            "For S21 accuracy and reciprocity use normalize=True. "
            "For |S11| of strong reflectors (PEC short, resonators) "
            "normalize=False is more accurate — see the normalize "
            "parameter docstring.",
            stacklevel=2,
        )
    if self._ports or self._tfsf:
        raise ValueError(
            "compute_waveguide_s_matrix() is not supported together with lumped ports or TFSF"
        )
    if self._periodic_axes:
        raise ValueError(
            "compute_waveguide_s_matrix() is not supported with manual periodic-axis overrides"
        )
    if len(self._waveguide_ports) < 2:
        raise ValueError(
            "compute_waveguide_s_matrix() requires at least two waveguide ports"
        )

    entries = list(self._waveguide_ports)
    if any(entry.probe_plane is not None for entry in entries):
        raise ValueError(
            "compute_waveguide_s_matrix() does not use per-port probe_plane; use reference_plane only or leave probe_plane unset"
        )
    if any(entry.calibration_preset not in (None, "measured") for entry in entries):
        raise ValueError(
            "compute_waveguide_s_matrix() currently supports only measured/default reference planes or explicit reference_plane overrides"
        )

    # Per-port straight-guide references for interior-PEC junctions.
    # Cheap guards first (raise BEFORE any FDTD); the grid-match check
    # and PEC-fold happen after the device grid is assembled below.
    if port_reference_sims is not None:
        if normalize != "flux":
            raise ValueError(
                "port_reference_sims requires normalize='flux' — the "
                "per-port reference feeds the flux P_inc normalization"
            )
        if any(entry.n_modes > 1 for entry in entries):
            raise NotImplementedError(
                "port_reference_sims is not supported with multimode ports "
                "(n_modes>1): the multimode extractor has no per-port-"
                "reference support"
            )
        if eps_override is not None or sigma_override is not None:
            raise NotImplementedError(
                "port_reference_sims combined with eps_override / "
                "sigma_override is an unvalidated combination and is not "
                "supported"
            )
        if len(port_reference_sims) != len(entries):
            raise ValueError(
                "port_reference_sims must supply one Simulation per "
                f"waveguide port ({len(entries)}), got "
                f"{len(port_reference_sims)}"
            )

    # Issue #704 audit: same silent NTFF drop class as the MSL path.
    _warn_ntff_box_dropped(self, "compute_waveguide_s_matrix()")

    # Non-uniform-mesh dispatch. Earlier the uniform scan ran with
    # the coarse boundary dx and silently ignored ``dx_profile`` /
    # ``dy_profile`` (handover v2 experiment 12) — and until #811 a
    # dz_profile-ONLY mesh still fell through to the uniform lane,
    # because this gate tested only the transverse profiles while
    # preflight described the graded mesh. The dedicated NU
    # two-run extractor below is enabled when its supported scope
    # is met (``normalize=True``, single-mode ports); otherwise
    # raise so the user is not given silently-wrong numbers.
    # Frequency grid resolved HERE rather than after the non-uniform
    # dispatch below: the NU lane's absorber advisory needs it, and
    # duplicating the resolver into that branch would be one more hand copy
    # of production logic (the #576-review class of defect). Pure function of
    # the entries and freq_max, so moving it earlier only makes the
    # matching-grid check fail sooner.
    from rfx.api._preflight import resolve_waveguide_port_freqs

    def _resolve_freqs(entry: _WaveguidePortEntry) -> jnp.ndarray:
        # ONE definition, shared with preflight_sparameters(calculator=
        # "waveguide") so the setup audits read the run's own band.
        return resolve_waveguide_port_freqs(self, entry)

    freqs = _resolve_freqs(entries[0])
    for entry in entries[1:]:
        entry_freqs = _resolve_freqs(entry)
        if entry_freqs.shape != freqs.shape or not np.allclose(np.asarray(entry_freqs), np.asarray(freqs)):
            raise ValueError("waveguide S-matrix requires matching frequency grids on all ports")

    if (
        self._dz_profile is not None
        or self._dx_profile is not None
        or self._dy_profile is not None
    ):
        if self._interface_eps == "dual_average":
            raise ValueError("interface_eps='dual_average' is not supported on the S-parameter NU lane")
        if checkpoint_segments is not None and checkpoint_segments < 1:
            raise ValueError(
                f"checkpoint_segments must be >= 1, got {checkpoint_segments}"
            )
        unsupported = []
        if normalize is not True and normalize != "flux":
            unsupported.append("normalize=True or normalize='flux' is required")
        if any(entry.n_modes > 1 for entry in entries):
            unsupported.append("multi-mode ports (n_modes>1) are not supported")
        # The differentiable eps/sigma AD channel is wired on the NU
        # path only for normalize='flux' (mirrors the uniform PR #172
        # flux-AD fix): the flux extractor is now jnp-native end-to-end
        # so a traced eps_override flows into the device Yee update and
        # back through the S-matrix. normalize=True is kept out of scope
        # — its diagonal a_inc_ref denominator carries the #88 band-edge
        # fragility, so accepting eps_override there could yield
        # silently-wrong gradients.
        if eps_override is not None and normalize != "flux":
            unsupported.append(
                "eps_override (differentiable AD channel) on the NU path "
                "requires normalize='flux'"
            )
        if sigma_override is not None and normalize != "flux":
            unsupported.append(
                "sigma_override (differentiable AD channel) on the NU path "
                "requires normalize='flux'"
            )
        if subpixel_smoothing:
            unsupported.append("subpixel_smoothing is not supported")
        if port_reference_sims is not None:
            unsupported.append(
                "port_reference_sims (per-port straight-guide junction "
                "references) is not supported on the non-uniform lane"
            )
        if unsupported:
            raise NotImplementedError(
                "compute_waveguide_s_matrix() on a non-uniform mesh "
                "(dx_profile / dy_profile / dz_profile) supports "
                "normalize=True or "
                "normalize='flux' and single-mode ports. "
                + "; ".join(unsupported)
                + ". Drop the dx/dy/dz profile to use the uniform lane."
            )
        # Far-port absorber advisory, on the NU lane too (#576 review F3).
        # It used to sit ~110 lines below this branch's return, i.e. it was
        # UNREACHABLE from here: the guard existed, its text named the exact
        # remedy, and no NU caller ever asked it. That silence is why both NU
        # fixture producers shipped 0.33 and 0.099 lambda_g stacks — patching
        # the two producers would have left the next one to repeat it. Pure
        # NumPy, no FDTD, so it costs a grid build and a few array ops.
        # No bare `except: pass` around it: an advisory that can fail
        # silently is the failure mode this finding IS. If the plumbing ever
        # breaks, the caller hears about it as a warning rather than getting
        # the same quiet all-clear that let 0.33 lambda_g ship.
        from rfx.runners.nonuniform import _build_waveguide_port_config_nu
        try:
            _nu_grid = self._build_nonuniform_grid()
            _nu_cfgs = [
                # n_steps is still None here when the caller passed
                # num_periods; the advisory reads only the config's
                # propagation axis and modal cutoff, never anything derived
                # from the step count, so a placeholder is honest rather
                # than a lie about the run length.
                _build_waveguide_port_config_nu(
                    self, _e, _nu_grid, jnp.asarray(freqs),
                    int(n_steps) if n_steps else 1)
                for _e in entries
            ]
        except Exception as _exc:  # noqa: BLE001 - reported, not swallowed
            import warnings as _w
            _w.warn(
                "compute_waveguide_s_matrix: could not evaluate the far-port "
                f"absorber advisory on the non-uniform lane ({_exc!r}). The "
                "0.5 guide-wavelength discipline is therefore UNCHECKED for "
                "this run — verify the absorber depth by hand (#576).",
                stacklevel=2,
            )
        else:
            _warn_thin_absorber_vs_guide_wavelength(
                _nu_grid, _nu_cfgs, freqs, self._cpml_layers,
                self._boundary_spec,
            )
            # Record-vs-far-boundary and port-index mirror audits
            # (post-v1.8 plan item 2). Same shared hook the uniform lane
            # below calls and preflight_sparameters(calculator=
            # "waveguide") calls; the grid and configs this lane already
            # built are passed in so no second mode solve is paid for.
            import warnings as _wg_warnings_nu
            self._preflight_waveguide_setup(
                _wg_warnings_nu, freqs=freqs, num_periods=num_periods,
                grid=_nu_grid, cfgs=_nu_cfgs,
                n_steps=int(n_steps) if n_steps else None,
            )

        _res_nu = self._compute_waveguide_s_matrix_nu(
            n_steps=n_steps,
            num_periods=num_periods,
            normalize=normalize,
            eps_override=eps_override,
            sigma_override=sigma_override,
            checkpoint_segments=checkpoint_segments,
        )
        return _finalize_sparam_result(
            _res_nu,
            extractor="compute_waveguide_s_matrix",
            strict=strict_passivity,
            check_reciprocity=True,
            # normalize=False carries documented Yee-dispersion + band-edge
            # |S11| overshoot (validated paths reach ~1.4-1.7), so use a
            # loose bound there that still catches gross extractor bugs
            # (|S11|>>1); normalize=True/"flux" correct dispersion -> tight.
            passivity_tol=2.0 if normalize is False else 0.10,
        )

    # Uniform-lane honesty guard (v1.8 WP1), the mirror of the
    # non-uniform guard above. The two-run normalized lane
    # (normalize=True) assembles S on the host:
    # extract_waveguide_s_params_normalized converts the device-run
    # outgoing wave with np.array, so a TRACED eps_override /
    # sigma_override (jax.grad, jax.jit) cannot flow through it.
    # Measured 2026-09-02 (tests/unit/autodiff/test_waveguide_two_run_lane_traced_override.py):
    # only the device-run site fires — the vacuum reference run carries
    # no design variable. Raise here, naming the lane, instead of a
    # TracerArrayConversionError deep in the extractor. A concrete
    # override still runs forward on this lane; the differentiable
    # lanes are normalize=False and normalize="flux".
    if normalize and normalize != "flux":
        for _ov_name, _ov_val in (
            ("eps_override", eps_override),
            ("sigma_override", sigma_override),
        ):
            if _ov_val is not None and is_tracer(_ov_val):
                raise NotImplementedError(
                    "compute_waveguide_s_matrix(normalize=True) is the "
                    "host-assembled two-run lane and is outside the "
                    f"differentiable chain: a traced {_ov_name} (under "
                    "jax.grad / jax.jit) cannot pass its np.array sites. "
                    "Use normalize=False or normalize='flux' to "
                    f"differentiate with respect to {_ov_name}."
                )

    grid = self._build_grid()
    _wg_sheet_specs: list = []
    _wg_pec_sheets: list = []
    _wg_pec_wires: list = []
    base_materials, debye_spec, lorentz_spec, pec_mask_wg, pec_shapes, boundary_pec_shapes, _ = self._assemble_materials(
        grid, sheet_specs=_wg_sheet_specs, pec_sheets=_wg_pec_sheets,
        pec_wires=_wg_pec_wires)
    _wg_pec_sheets = tuple(_wg_pec_sheets)
    _wg_pec_wires = tuple(_wg_pec_wires)
    # #931 §1.7: the realized PEC edges of this device — volumes,
    # sheets and wires — built ONCE and handed to every device run of
    # the extractors below.  This replaces the sigma=1e10 CELL fold
    # this lane used to do (see the note at the old fold site).
    from rfx.boundaries.pec import realized_pec_edge_masks as _rpem
    _wg_pec_edge_masks = None
    if pec_mask_wg is not None or _wg_pec_sheets or _wg_pec_wires:
        _wg_pec_edge_masks = _rpem(
            pec_mask_wg, sheets=_wg_pec_sheets, wires=_wg_pec_wires,
            periodic=self._periodic_flags())
    # The junction geometry census also needs these edges when Kottke
    # below encodes PEC in inverse permittivity and clears solver masks.
    _wg_geometry_pec_edges = _wg_pec_edge_masks
    # #677: node-thin sheet ctx for the DEVICE runs of this lane; the
    # edge exclusion uses the same realized edges so sheet and PEC
    # never contend for one edge.  The vacuum REFERENCE runs never
    # receive the ctx (explicit strip at the extractor call sites).
    from rfx.materials.thin_conductor import build_sheet_impedance_ctx as _build_sheet_ctx
    _wg_sheet_ctx = _build_sheet_ctx(
        _wg_sheet_specs, pec_edge_masks=_wg_pec_edge_masks)
    if _wg_sheet_ctx is not None and subpixel_smoothing:
        raise ValueError(
            "surface-impedance (surface_impedance_f0) sheets are not "
            "supported with subpixel_smoothing / conformal on the "
            "waveguide S-matrix lane (#677 v1): the sheet operator "
            "assumes the plain isotropic E update at its edges.")
    # #931 §1.7: the interior PEC of this lane is the realized edge
    # set, applied per step by the shared ``apply_pec_edges``.  The
    # sigma=1e10 CELL fold that stood here was a fourth realization of
    # the same geometry and damped only the components indexed by the
    # occupied cell, so a one-cell iris or wall got its lower face and
    # never its far one.
    # **Stage 2 caveat, unchanged**: under ``subpixel_smoothing=
    # "kottke_pec"`` the inverse-eps tensor already encodes the PEC
    # zero (inv = 0 freezes the field) and is the single source of
    # truth; that lane takes no edge masks.
    _use_kottke_pec_early = (subpixel_smoothing == "kottke_pec")
    if _use_kottke_pec_early:
        _wg_pec_edge_masks = None
    materials = base_materials
    # G-AD-WIRE-WG2: public eps_override / sigma_override channel.
    # Mirror the MSL pattern: replace eps_r / sigma on the assembled
    # materials *after* the PEC fold so PEC boundaries are untouched.
    if eps_override is not None:
        materials = materials._replace(eps_r=eps_override)
    if sigma_override is not None:
        materials = materials._replace(sigma=sigma_override)

    # Per-port straight-guide reference materials for interior-PEC
    # junctions. Each reference sim is a geometry carrier: assemble its
    # materials on a grid that must match the device grid, and realize
    # its PEC edges identically to the device path above (only the
    # plain path — no subpixel/conformal handling for references).
    ref_materials_per_port = None
    ref_pec_edge_masks_per_port = None
    if port_reference_sims is not None:
        ref_materials_per_port = []
        ref_pec_edge_masks_per_port = []
        for _i, _ref_sim in enumerate(port_reference_sims):
            _ref_grid = _ref_sim._build_grid()
            if _ref_grid.shape != grid.shape or float(_ref_grid.dx) != float(grid.dx):
                raise ValueError(
                    f"port_reference_sims[{_i}] grid "
                    f"(shape={_ref_grid.shape}, dx={_ref_grid.dx}) must "
                    f"match the device grid (shape={grid.shape}, "
                    f"dx={grid.dx})"
                )
            _ref_pec_sheets: list = []
            _ref_pec_wires: list = []
            _ref_base, _, _, _ref_pec_mask, _, _, _ = _ref_sim._assemble_materials(
                _ref_grid, pec_sheets=_ref_pec_sheets,
                pec_wires=_ref_pec_wires)
            ref_materials_per_port.append(_ref_base)
            _ref_edges_i = None
            if (_ref_pec_mask is not None or _ref_pec_sheets
                    or _ref_pec_wires):
                _ref_edges_i = _rpem(
                    _ref_pec_mask, sheets=tuple(_ref_pec_sheets),
                    wires=tuple(_ref_pec_wires))
            ref_pec_edge_masks_per_port.append(_ref_edges_i)

    if n_steps is None:
        n_steps = grid.num_timesteps(num_periods=num_periods)
    _, debye, lorentz = self._init_dispersion(materials, grid.dt, debye_spec, lorentz_spec)

    # Build configs — may be a single config or a list of configs per port
    has_multimode = any(entry.n_modes > 1 for entry in entries)
    raw_cfgs = [self._build_waveguide_port_config(entry, grid, freqs, n_steps) for entry in entries]

    # Unify source waveform across all ports so that the S-matrix
    # extraction uses identical excitation.  Different source spectra
    # (from mismatched f0/bandwidth) cause S11 ≠ S22 artifacts in the
    # unnormalized path because V/I decomposition error varies with
    # frequency.  Use port 0's waveform as the canonical source.
    def _flatten_cfgs(cfgs):
        out = []
        for c in cfgs:
            if isinstance(c, list):
                out.extend(c)
            else:
                out.append(c)
        return out

    flat0 = _flatten_cfgs(raw_cfgs)
    ref_t0 = flat0[0].src_t0
    ref_tau = flat0[0].src_tau
    need_unify = any(
        c.src_t0 != ref_t0 or c.src_tau != ref_tau for c in flat0[1:]
    )
    if need_unify:
        raw_cfgs = [
            cfg._replace(src_t0=ref_t0, src_tau=ref_tau)
            if not isinstance(cfg, list)
            else [c._replace(src_t0=ref_t0, src_tau=ref_tau) for c in cfg]
            for cfg in raw_cfgs
        ]

    # Far-port absorber advisory for EVERY uniform two-port path (issue
    # #494). Emitted here, before the FDTD runs and before the
    # single-mode / multimode split, because the functional entry points
    # run no sim.preflight() and the port_reference_sims advisory below
    # covers only the junction path.
    _warn_thin_absorber_vs_guide_wavelength(
        grid, raw_cfgs, freqs, self._cpml_layers, self._boundary_spec,
    )

    # Record-vs-far-boundary and port-index mirror audits (post-v1.8 plan
    # item 2). The same shared hook the non-uniform lane above and
    # preflight_sparameters(calculator="waveguide") call, given this
    # lane's already-built grid, configs and step count.
    import warnings as _wg_warnings
    self._preflight_waveguide_setup(
        _wg_warnings, freqs=freqs, num_periods=num_periods,
        grid=grid, cfgs=raw_cfgs, n_steps=int(n_steps),
    )

    # Compute Kottke per-component smoothed permittivity if requested.
    # Shared by both single-mode and multi-mode paths.
    # Mirrors rfx/runners/uniform.py: shape_eps_pairs from sim geometry,
    # then compute_smoothed_eps. The reference run is vacuum and has no
    # ε interfaces, so it always passes aniso_eps=None inside the
    # extractor.
    #
    # THE MIRROR IS BROKEN AS OF #1043 STAGE B, on purpose, and here is the
    # note that says so rather than leaving it to be rediscovered. The two
    # runner sites now build their pairs through
    # ``rfx.geometry.smoothing.smoothed_shape_pairs``, which continues a
    # dielectric that reaches a CPML/UPML face out through that pad — without
    # it, such a structure is solved with eps_r = 1 in its own absorber and
    # ends in a facet (#831: |B/A| 0.53 on a straight guide, worse as the
    # absorber deepens). The two blocks below still build the raw list, so a
    # waveguide fixture whose dielectric reaches a port's absorber carries
    # that facet.
    #
    # Not changed with the runners because this lane is v1.8 chain-closed:
    # 185 verdicts replay against a frozen artifact, so moving its numbers is
    # a measurement change that wants its own pre-declaration and its own
    # re-measurement, not a ride on the runner fix ("refactoring and
    # measurement changes do not travel together", #928). Whether any
    # committed fixture is actually affected was NOT measured — the site was
    # found by grep during stage B and is recorded in section 8a of
    # docs/design_notes/issue1043_pad_continuation_results.md. Anyone closing
    # it: the reference run passes dielectric_shapes=[] and so cannot carry a
    # facet; only the device run can.
    # Stage 2 unified path: subpixel_smoothing="kottke_pec" routes
    # through compute_inv_eps_tensor_diag and skips the Stage 1
    # eps_correction + apply_conformal_pec chain entirely. Both
    # device and reference (vacuum) runs see the same boundary-
    # face PEC walls, so the inverse-permittivity tensor is
    # computed twice (once per material context).
    use_kottke_pec = (subpixel_smoothing == "kottke_pec")
    aniso_eps = None
    aniso_inv_eps = None
    ref_aniso_inv_eps = None
    if use_kottke_pec:
        from rfx.geometry.smoothing import compute_inv_eps_tensor_diag
        shape_eps_pairs = [
            (entry.shape, self._resolve_material(entry.material_name).eps_r)
            for entry in self._geometry
        ]
        aniso_inv_eps = compute_inv_eps_tensor_diag(
            grid,
            dielectric_shapes=shape_eps_pairs,
            pec_shapes=pec_shapes or [],
            background_eps=1.0,
        )
        # Reference run is empty guide with same boundary walls only.
        # Must NOT include interior PEC geometry (e.g. PEC short box):
        # if device and reference share the same obstacle, both DFTs
        # are identical and (device - reference) / incident = 0.
        ref_aniso_inv_eps = compute_inv_eps_tensor_diag(
            grid,
            dielectric_shapes=[],
            pec_shapes=boundary_pec_shapes,
            background_eps=1.0,
        )
        # Yee-stagger correction: the Kottke union reaches inv=0
        # on Yee-staggered components only when the cell-center
        # AND the offset position are both inside the PEC shape.
        # For thin PEC obstacles (e.g. a 1-cell-wide PEC short),
        # cell-center is inside but Ey/Ez Yee positions are at
        # cell-corner offsets that fall *outside* the box → inv
        # remains 1 (vacuum). That leaves the H field free to
        # propagate inside the PEC region and seeds late-time
        # exponential growth.
        #
        # Fix: where ``pec_mask`` (cell-center binary) is True,
        # force all three inv components to zero. This is the
        # cell-center analogue of Stage 1's sigma=1e10 fold,
        # without the Ca→-1 instability that the sigma fold has
        # at Yee-staggered cells where inv > 0.
        if pec_mask_wg is not None:
            inv_xx, inv_yy, inv_zz = aniso_inv_eps
            inv_xx = jnp.where(pec_mask_wg, 0.0, inv_xx)
            inv_yy = jnp.where(pec_mask_wg, 0.0, inv_yy)
            inv_zz = jnp.where(pec_mask_wg, 0.0, inv_zz)
            aniso_inv_eps = (inv_xx, inv_yy, inv_zz)
            # pec_mask_wg marks interior PEC geometry (e.g. the PEC
            # short). The reference run has no interior PEC — do NOT
            # apply pec_mask_wg to ref_aniso_inv_eps, or the reference
            # becomes identical to the device and S11 = 0.
    elif subpixel_smoothing:
        from rfx.geometry.smoothing import compute_smoothed_eps
        shape_eps_pairs = [
            (entry.shape, self._resolve_material(entry.material_name).eps_r)
            for entry in self._geometry
        ]
        if shape_eps_pairs:
            aniso_eps = compute_smoothed_eps(
                grid, shape_eps_pairs, background_eps=1.0,
            )

    # Stage 1 conformal PEC: when BoundarySpec declares conformal
    # faces and pec_shapes was populated (boundary half-space +
    # any user PEC), compute Dey-Mittra weights and apply
    # eps_correction. Mirrors runners/uniform.py:96-124.
    # ``conformal_weights`` flows through extract_waveguide_*
    # into rfx.simulation.run, which already calls
    # ``apply_conformal_pec`` per step in its scan body.
    # Suppressed when use_kottke_pec — Stage 2 owns the PEC
    # tensor encoding and the eps_correction would double-correct.
    conformal_weights = None
    ref_aniso_eps = None
    if (self._boundary_spec.conformal_faces() and pec_shapes
            and not use_kottke_pec):
        from rfx.geometry.conformal import (
            compute_conformal_weights_sdf,
            clamp_conformal_weights,
            conformal_eps_correction,
        )
        w_ex, w_ey, w_ez = compute_conformal_weights_sdf(grid, pec_shapes)
        w_ex, w_ey, w_ez = clamp_conformal_weights(w_ex, w_ey, w_ez, 0.1)
        conformal_weights = (w_ex, w_ey, w_ez)
        # Per-component conformal-corrected eps. Merge with the
        # smoothed eps (if any): conformal overrides at boundary
        # cells, smoothed survives in the interior.
        eps_base = materials.eps_r
        eps_ex_c, eps_ey_c, eps_ez_c = conformal_eps_correction(
            eps_base, w_ex, w_ey, w_ez,
        )
        if aniso_eps is not None:
            s_ex, s_ey, s_ez = aniso_eps
            boundary_ex = w_ex < 1.0
            boundary_ey = w_ey < 1.0
            boundary_ez = w_ez < 1.0
            eps_ex_c = jnp.where(boundary_ex, eps_ex_c, s_ex)
            eps_ey_c = jnp.where(boundary_ey, eps_ey_c, s_ey)
            eps_ez_c = jnp.where(boundary_ez, eps_ez_c, s_ez)
        aniso_eps = (eps_ex_c, eps_ey_c, eps_ez_c)
        # The reference run (vacuum) shares the same boundary
        # walls, so the conformal eps correction applies equally.
        # Build it from the ref vacuum eps so the only difference
        # ref-vs-device is the obstacle in ``materials.eps_r``.
        ref_eps_base = jnp.ones_like(eps_base)
        ref_ex, ref_ey, ref_ez = conformal_eps_correction(
            ref_eps_base, w_ex, w_ey, w_ez,
        )
        ref_aniso_eps = (ref_ex, ref_ey, ref_ez)

    if has_multimode and _wg_sheet_ctx is not None:
        raise ValueError(
            "surface-impedance (surface_impedance_f0) sheets are not "
            "supported on the multimode waveguide S-matrix path "
            "(#677 v1): the multimode extractors do not thread the "
            "sheet operator ctx, so the runs would silently simulate "
            "NO sheet. Use n_modes=1 ports or drop the f0 sheet.")
    if has_multimode:
        # Multi-mode path: each raw_cfg is a list of WaveguidePortConfig
        port_mode_cfgs: list[list] = []
        for entry, raw in zip(entries, raw_cfgs):
            if isinstance(raw, list):
                port_mode_cfgs.append(raw)
            else:
                port_mode_cfgs.append([raw])

        ref_shifts_mm = []
        desired_refs_mm = []
        for entry, mode_cfgs in zip(entries, port_mode_cfgs):
            first_cfg = mode_cfgs[0]
            planes = waveguide_plane_positions(first_cfg)
            desired_ref = (
                entry.reference_plane
                if entry.reference_plane is not None
                else planes["source"]
            )
            ref_shifts_mm.append(desired_ref - planes["reference"])
            desired_refs_mm.append(desired_ref)

        mm_pec_axes = "".join(axis for axis in "xyz" if axis not in grid.cpml_axes)
        if normalize == "flux":
            from rfx.core.yee import init_materials as _init_vacuum_materials
            ref_materials = _init_vacuum_materials(grid.shape)
            s_params, mode_map = extract_multimode_s_matrix_flux(
                grid,
                materials,
                ref_materials,
                port_mode_cfgs,
                n_steps,
                boundary="cpml",
                cpml_axes=grid.cpml_axes,
                pec_axes=mm_pec_axes,
                debye=debye,
                lorentz=lorentz,
                ref_shifts=ref_shifts_mm,
                aniso_eps=aniso_eps,
                conformal_weights=conformal_weights,
                aniso_inv_eps=aniso_inv_eps,
                pec_edge_masks=_wg_pec_edge_masks,
            )
        elif normalize:
            # The two-run normalized extractor divides each receiving
            # channel by its own empty-guide outgoing wave
            # (b_dev/b_ref). For cross-mode channels the empty-guide
            # reference is ~0 (orthogonal modes do not couple in a
            # uniform guide), so the ratio blows up (measured
            # cross-mode |S| ~ 4.7 on an over-moded WR-90 slab).
            # Use normalize="flux" instead — power ratios referenced
            # to the always-nonzero incident modal power avoid the
            # 0/0 and also fix the reflection noise floor.
            raise ValueError(
                "compute_waveguide_s_matrix(normalize=True) is not "
                "supported with n_modes > 1 (cross-mode channels hit a "
                "0/0 in the two-run normalization). Use "
                "normalize='flux' for multi-mode S-matrices."
            )
        else:
            s_params, mode_map = extract_multimode_s_matrix(
                grid,
                materials,
                port_mode_cfgs,
                n_steps,
                boundary="cpml",
                cpml_axes=grid.cpml_axes,
                pec_axes=mm_pec_axes,
                debye=debye,
                lorentz=lorentz,
                ref_shifts=ref_shifts_mm,
                aniso_eps=aniso_eps,
                conformal_weights=conformal_weights,
                aniso_inv_eps=aniso_inv_eps,
                pec_edge_masks=_wg_pec_edge_masks,
            )
        # Report the ABSOLUTE de-embed target plane (matches the single-mode + coax paths and
        # the WaveguideSMatrixResult schema), NOT the relative shift ref_shifts_mm — that is the
        # extractor's phase-shift input, not a plane coordinate (RF-audit 2026-07-23).
        reference_planes = np.array(desired_refs_mm, dtype=float)
        # Build port names including mode indices
        port_names_mm = []
        port_directions_mm = []
        for port_idx, mode_idx, mtype, m_n in mode_map:
            entry = entries[port_idx]
            port_names_mm.append(f"{entry.name}_mode{mode_idx}_{mtype}{m_n[0]}{m_n[1]}")
            port_directions_mm.append(entry.direction)
        _res_mm = WaveguideSMatrixResult(
            s_params=s_params,
            freqs=jnp.asarray(freqs),
            port_names=tuple(port_names_mm),
            port_directions=tuple(port_directions_mm),
            reference_planes=reference_planes,
        )
        return _finalize_sparam_result(
            _res_mm,
            extractor="compute_waveguide_s_matrix",
            strict=strict_passivity,
            check_reciprocity=True,
            passivity_tol=2.0 if normalize is False else 0.10,
        )

    # Single-mode path (original behavior)
    cfgs = raw_cfgs

    # Far-port discipline advisories for interior-PEC junction references.
    # Pure-numpy heuristics emitted BEFORE the FDTD runs; no simulation.
    if port_reference_sims is not None:
        _warn_junction_probe_clearance(
            grid, cfgs, materials.sigma,
            [m.sigma for m in ref_materials_per_port], freqs,
            device_pec_edges=_wg_geometry_pec_edges,
            ref_pec_edges=ref_pec_edge_masks_per_port,
        )
        _warn_junction_cpml_thickness(
            grid, cfgs, freqs, self._cpml_layers,
        )

    # ``cfg.u_lo/u_hi`` are CELL spans (the entries ``_plane_indexer``
    # slices), so this compares exactly the field-array ranges the two
    # ports would write to. Two apertures that meet at a shared wall node
    # abut without sharing a cell and are NOT an overlap; the builders used
    # to hand over node spans, one entry too long, so such a pair was
    # rejected. ``test_abutting_apertures_on_one_face_share_no_cell``
    # (tests/unit/ports/test_waveguide_port_aperture_cell_count.py) pins it.
    def _slices_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return max(a[0], b[0]) < min(a[1], b[1])

    by_direction = {}
    for entry, cfg in zip(entries, cfgs):
        by_direction.setdefault(entry.direction, []).append(cfg)

    for direction, side_cfgs in by_direction.items():
        plane_indices = {cfg.x_index for cfg in side_cfgs}
        if len(plane_indices) != 1:
            raise ValueError(
                f"waveguide ports on boundary {direction} must share one boundary plane"
            )
        for i in range(len(side_cfgs)):
            for j in range(i + 1, len(side_cfgs)):
                if _slices_overlap((side_cfgs[i].u_lo, side_cfgs[i].u_hi), (side_cfgs[j].u_lo, side_cfgs[j].u_hi)) and _slices_overlap((side_cfgs[i].v_lo, side_cfgs[i].v_hi), (side_cfgs[j].v_lo, side_cfgs[j].v_hi)):
                    raise ValueError(
                        f"waveguide ports on the same {direction} boundary must have disjoint apertures"
                    )

    ref_shifts = []
    for entry, cfg in zip(entries, cfgs):
        # Default reference plane = the user-facing port plane
        # (snapped x_position). Previously defaulted to the internal
        # ``reference_x_m`` (= source + ref_offset·dx) which left the
        # returned S-matrix phase-shifted by `exp(-jβ·ref_offset·dx)`
        # relative to the physical port — a silent convention mismatch
        # vs. Meep, OpenEMS, and any analytic formula the user would
        # compare against. Keep the ``entry.reference_plane`` override
        # for explicit user control.
        planes = waveguide_plane_positions(cfg)
        desired_ref = (
            entry.reference_plane
            if entry.reference_plane is not None
            else planes["source"]
        )
        ref_shifts.append(desired_ref - planes["reference"])

    _pec_axes = "".join(axis for axis in "xyz" if axis not in grid.cpml_axes)
    # The G-WI5 runtime guardrail that stood here -- "conformal=True is
    # KNOWN to produce NaN S-parameters at fine mesh" -- was DELETED
    # 2026-09-15 (#1043 / PR #1047). The NaN was the CPML psi coefficient
    # reading a different permittivity than the Yee half of the same
    # timestep: conformal_eps_correction sets aniso_eps to eps_eff = eps/w
    # at wall cells, which is HIGHER than materials.eps_r, and those cells
    # run through the x CPML pads. Measured on the tripwire's own rungs,
    # |S21| at dx 3 / 2 / 1.5 mm: 0.7564 / 0.6974 / nan before, and
    # 0.7796 / 0.6974 / 0.7274 after. Warning about a NaN that no longer
    # happens would send users to staircase for no reason.
    #
    # NOT claimed by this removal: that conformal PEC is ACCURATE. The
    # 2026-06-08 accuracy verdicts on the four conformal methods were taken
    # with the defect present and are still to be re-measured.

    if normalize == "flux":
        from rfx.core.yee import init_materials as _init_vacuum_materials
        ref_materials = _init_vacuum_materials(grid.shape)
        s_params = extract_waveguide_s_matrix_flux(
            grid,
            materials,
            ref_materials,
            cfgs,
            n_steps,
            boundary="cpml",
            cpml_axes=grid.cpml_axes,
            pec_axes=_pec_axes,
            debye=debye,
            lorentz=lorentz,
            ref_debye=None,
            ref_lorentz=None,
            ref_shifts=ref_shifts,
            aniso_eps=aniso_eps,
            ref_aniso_eps=ref_aniso_eps,
            conformal_weights=conformal_weights,
            aniso_inv_eps=aniso_inv_eps,
            ref_aniso_inv_eps=ref_aniso_inv_eps,
            ref_materials_per_port=ref_materials_per_port,
            pec_edge_masks=_wg_pec_edge_masks,
            ref_pec_edge_masks_per_port=ref_pec_edge_masks_per_port,
            checkpoint_segments=checkpoint_segments,
            return_settling=True,
            sheet_impedance=_wg_sheet_ctx,
        )
        s_params, settling_db = s_params
    elif normalize:
        from rfx.core.yee import init_materials as _init_vacuum_materials
        ref_materials = _init_vacuum_materials(grid.shape)
        s_params = extract_waveguide_s_params_normalized(
            grid,
            materials,
            ref_materials,
            cfgs,
            n_steps,
            boundary="cpml",
            cpml_axes=grid.cpml_axes,
            pec_axes=_pec_axes,
            debye=debye,
            lorentz=lorentz,
            ref_debye=None,
            ref_lorentz=None,
            ref_shifts=ref_shifts,
            aniso_eps=aniso_eps,
            ref_aniso_eps=ref_aniso_eps,
            conformal_weights=conformal_weights,
            aniso_inv_eps=aniso_inv_eps,
            ref_aniso_inv_eps=ref_aniso_inv_eps,
            pec_edge_masks=_wg_pec_edge_masks,
            checkpoint_segments=checkpoint_segments,
            return_settling=True,
            sheet_impedance=_wg_sheet_ctx,
        )
        s_params, settling_db = s_params
    else:
        s_params, settling_db = extract_waveguide_s_matrix(
            grid,
            materials,
            cfgs,
            n_steps,
            boundary="cpml",
            cpml_axes=grid.cpml_axes,
            pec_axes=_pec_axes,
            debye=debye,
            lorentz=lorentz,
            ref_shifts=ref_shifts,
            aniso_eps=aniso_eps,
            conformal_weights=conformal_weights,
            aniso_inv_eps=aniso_inv_eps,
            pec_edge_masks=_wg_pec_edge_masks,
            checkpoint_segments=checkpoint_segments,
            return_settling=True,
            sheet_impedance=_wg_sheet_ctx,
        )
    reference_planes = np.array(
        [
            entry.reference_plane
            if entry.reference_plane is not None
            # Report the de-embed TARGET (the physical port/source plane, line 1042), not the
            # internal raw-extraction plane. Previously reported ["reference"] = source +
            # ref_offset·dx, so the metadata claimed a plane ref_offset cells off from where the
            # S-params are actually referenced (RF-audit 2026-07-23; matches the NU sibling).
            else waveguide_plane_positions(cfg)["source"]
            for entry, cfg in zip(entries, cfgs)
        ],
        dtype=float,
    )
    _port_names = tuple(entry.name for entry in entries)
    # Issue #538: the energy ring-down witness now reaches the waveguide
    # path — same aggregate truncation warning as the lumped/MSL path.
    # NaN entries (traced AD runs) are skipped by the warner's finite
    # mask; the array itself is always attached for the record.
    settling_db = np.asarray(settling_db, dtype=float)
    _warn_if_ringdown_truncated(
        settling_db, _port_names, num_periods=float(num_periods),
    )
    # Post-solve discretization witness (post-v1.8 plan item 5): one
    # banner line after the settling line. Report-only.
    _ph_rms, _ph_meta = _waveguide_s21_phase_residual(
        s_params, freqs, reference_planes, cfgs, normalize=normalize,
    )
    _res_sm = WaveguideSMatrixResult(
        s_params=s_params,
        freqs=jnp.asarray(freqs),
        port_names=_port_names,
        port_directions=tuple(entry.direction for entry in entries),
        reference_planes=reference_planes,
        settling_db=settling_db,
        s21_phase_residual_deg_rms=_ph_rms,
        s21_phase_residual_meta=_ph_meta,
    )
    return _finalize_sparam_result(
        _res_sm,
        extractor="compute_waveguide_s_matrix",
        strict=strict_passivity,
        check_reciprocity=True,
        passivity_tol=2.0 if normalize is False else 0.10,
    )


def _compute_waveguide_s_matrix_nu(
    self,
    *,
    n_steps: int | None,
    num_periods: float,
    normalize: bool,
    eps_override=None,
    sigma_override=None,
    checkpoint_segments: int | None = None,
) -> WaveguideSMatrixResult:
    """Non-uniform-mesh two-run S-matrix extraction.

    Drives each port in turn, running device + vacuum-reference
    scans through ``run_nonuniform_path`` so ``dx_profile`` /
    ``dy_profile`` / ``dz_profile`` actually flow into the Yee update. The per-port
    drive is implemented by temporarily zeroing ``amplitude`` on
    non-driven entries; the original port list is restored in a
    ``finally`` block. Reference run uses ``eps_override`` /
    ``sigma_override`` to replace the assembled materials with
    vacuum before the scan launches.

    Current scope (matches the uniform path minus a few niceties):
      - ``normalize=True`` or ``normalize='flux'``.
      - Single-mode ports (``n_modes == 1``) only.
      - ``eps_override`` / ``sigma_override`` (the differentiable AD
        design variable) are wired only for ``normalize='flux'``: they
        are threaded into the *device* run so the traced eps flows
        through the jnp-native flux extraction and back to the
        S-matrix gradient. The *reference* run stays vacuum. They are
        rejected for ``normalize=True`` (its diagonal a_inc_ref
        denominator carries the #88 band-edge fragility).

    Extracts ``a_inc`` / ``b_out`` via the same
    ``extract_waveguide_port_waves`` helper as the uniform path and
    applies the same diagonal-subtraction + off-diagonal-division
    normalisation (see ``extract_waveguide_s_params_normalized``
    in ``rfx/sources/waveguide_port.py``).
    """
    if self._interface_eps == "dual_average":
        raise ValueError("interface_eps='dual_average' is not supported on the S-parameter NU lane")
    from dataclasses import replace as _dc_replace
    from rfx.runners.nonuniform import (
        run_nonuniform_path,
        assemble_materials_nu,
    )
    from rfx.sources.waveguide_port import (
        extract_waveguide_port_waves,
        settling_db_from_port_records,
        waveguide_plane_positions,
    )

    # ``normalize`` may be True (lumped V/I ratio) or "flux" (Poynting
    # power-ratio magnitude + modal phase). normalize=False is not
    # supported on the NU path.
    _flux_mode = (normalize == "flux")
    if not normalize:
        raise NotImplementedError(
            "compute_waveguide_s_matrix(normalize=False) is not yet "
            "supported on the non-uniform mesh path; use normalize=True, "
            "normalize='flux', or drop the dx/dy/dz profiles to stay on "
            "the uniform lane."
        )

    entries = list(self._waveguide_ports)
    if any(entry.n_modes > 1 for entry in entries):
        raise NotImplementedError(
            "Multi-mode waveguide ports are not yet supported on the "
            "non-uniform mesh path."
        )

    n_ports = len(entries)

    # Build the grid directly so we can restrict ``cpml_axes`` to
    # axes that are not fully PEC/PMC-bounded. The rasteriser (see
    # ``rfx/geometry/rasterize_grid.py::coords_from_nonuniform_grid``)
    # uses a single ``grid.cpml_layers`` offset for every axis;
    # when a fully PEC-bounded axis is shorter than
    # ``cpml_layers + 1`` cells the offset slice hits IndexError.
    # Dropping that axis from ``cpml_axes`` keeps the physical
    # grid identical (PEC faces already have pad=0) but zeroes the
    # offset so the rasteriser snaps cells to 0 cleanly.
    from rfx.runners.nonuniform import build_nonuniform_grid
    pec_set = (self._boundary_spec.pec_faces()
               if self._boundary_spec is not None else None) or set()
    pmc_set = (self._boundary_spec.pmc_faces()
               if self._boundary_spec is not None else None) or set()

    def _axis_fully_closed(ax: str) -> bool:
        return {f"{ax}_lo", f"{ax}_hi"}.issubset(pec_set | pmc_set)

    cpml_axes = "".join(
        ax for ax in "xyz"
        if ax not in (self._periodic_axes or "")
        and not _axis_fully_closed(ax)
    )
    # Missing z profiles are synthesized locally, preserving the declared
    # auto mesh and its cached resolution throughout device/reference runs.
    grid = build_nonuniform_grid(
        self._freq_max, self._domain, self._dx, self._cpml_layers,
        self._dz_profile,
        dx_profile=self._dx_profile,
        dy_profile=self._dy_profile,
        pec_faces=pec_set or None,
        pmc_faces=pmc_set or None,
        cpml_axes=cpml_axes,
    )
    if n_steps is None:
        # ``NonUniformGrid`` does not expose ``num_timesteps`` (known
        # asymmetry vs. ``Grid``); inline the same formula here.
        n_steps = int(np.ceil(num_periods / self._freq_max / float(grid.dt)))

    # Assemble device materials once to learn the full array shape;
    # vacuum reference is shape-matched onto that same array.
    # Shape probe only: the vacuum reference is ones_like/zeros_like of
    # these arrays. Every drive run below goes through
    # run_nonuniform_path, which assembles with its own collectors and
    # realizes the sheets — so the #931 collectors here are passed and
    # dropped (an explicit "cells only", not an omission).
    dev_materials_concrete, _, _, _ = assemble_materials_nu(
        self, grid, pec_sheets=[], pec_wires=[])
    vacuum_eps = jnp.ones_like(dev_materials_concrete.eps_r)
    vacuum_sigma = jnp.zeros_like(dev_materials_concrete.sigma)

    # Frequency grid must match across ports.
    port_freqs = entries[0].freqs
    if port_freqs is None:
        port_freqs = jnp.linspace(
            self._freq_max / 10, self._freq_max, entries[0].n_freqs,
        )
    for entry in entries[1:]:
        other = entry.freqs if entry.freqs is not None else jnp.linspace(
            self._freq_max / 10, self._freq_max, entry.n_freqs,
        )
        if other.shape != port_freqs.shape or not np.allclose(
            np.asarray(other), np.asarray(port_freqs)
        ):
            raise ValueError(
                "waveguide S-matrix requires matching frequency grids on all ports"
            )

    # jnp-functional: collect per-drive columns; stack after loop
    s_columns: list[list] = []  # s_columns[drive_idx] = list of (n_freqs,) jnp arrays over recv_idx
    settling_runs: list[float] = []  # per-drive ring-down witness (#827)
    ref_shifts: tuple[float, ...] | None = None
    reference_planes_out: np.ndarray | None = None
    final_cfgs: list | None = None

    original_entries = list(entries)
    try:
        for drive_idx in range(n_ports):
            self._waveguide_ports = [
                _dc_replace(
                    e,
                    amplitude=(e.amplitude if idx == drive_idx else 0.0),
                )
                for idx, e in enumerate(original_entries)
            ]

            # Device run: thread the public eps/sigma override (the
            # traced design variable) into the device Yee update so the
            # gradient flows from eps_override -> device fields -> flux
            # -> S-matrix (mirrors the uniform PR #172 flux-AD wiring).
            # When eps_override is None the assembled device materials
            # are used unchanged (device fields identical); the np->jnp
            # flux extraction below matches the prior np path to
            # rtol<=1e-5 (float reassociation only, per uniform PR #172).
            # issue #73: translate the uniform `checkpoint_segments` (K
            # segments) → the NU runner's `checkpoint_every` (chunk size).
            # The chunk MUST exactly divide n_steps (pad=0): the NU runner
            # zero-pads non-divisor chunks and those extra ring-down steps
            # would corrupt the carry-accumulated flux DFT (the time_series
            # is truncated to n_steps but the flux accumulator is NOT), so a
            # non-divisor chunk is NOT forward-identical for the flux
            # S-matrix — same divisor rule as the uniform V/I-DFT path. Pick
            # the divisor of n_steps nearest to n_steps/K. Checkpoint ONLY the
            # *device* run — the vacuum reference is constant in the design
            # variable so it carries no AD tape. The √N tape win is realised
            # under jax.grad (normalize='flux' + eps_override); plain forward
            # is identical.
            from rfx.simulation import _nearest_divisor
            _ckpt_every = None
            if checkpoint_segments is not None and checkpoint_segments > 1:
                _ck = _nearest_divisor(n_steps, max(1, n_steps // int(checkpoint_segments)))
                if 0 < _ck < n_steps:
                    _ckpt_every = _ck
            dev_result = run_nonuniform_path(
                self, n_steps=n_steps,
                eps_override=eps_override,
                sigma_override=sigma_override,
                attach_waveguide_flux=_flux_mode,
                checkpoint_every=_ckpt_every,
            )
            # Reference run stays vacuum (incident-power reference) and is
            # independent of the design variable. ``strip_interior_pec``
            # drops the rasterized interior PEC (iris / wall / post) from
            # the reference so it is a clean empty guide: the boundary y/z
            # guide walls survive (they are enforced via pec_faces, not
            # pec_mask). Without this the vacuum override replaces only
            # eps/sigma and the reference retains the device's interior
            # PEC mask → device and reference DFTs are bit-identical →
            # (device - reference) = 0 → S11 = 0 for any PEC reflector.
            # This mirrors the uniform reference run, which builds the
            # reference with dielectric_shapes=[] + boundary-only PEC.
            ref_result = run_nonuniform_path(
                self,
                n_steps=n_steps,
                eps_override=vacuum_eps,
                sigma_override=vacuum_sigma,
                attach_waveguide_flux=_flux_mode,
                strip_interior_pec=True,
                # #677: the surface-impedance sheet no longer rides
                # materials.sigma, so sigma_override=vacuum does NOT
                # strip it — the ctx must be stripped EXPLICITLY here,
                # beside strip_interior_pec, or the "empty guide"
                # reference would still carry the lossy sheet.
                strip_sheet_impedance=True,
            )

            dev_wg = dev_result.waveguide_ports or {}
            ref_wg = ref_result.waveguide_ports or {}
            if len(dev_wg) != n_ports or len(ref_wg) != n_ports:
                raise RuntimeError(
                    "NU waveguide S-matrix expected one final cfg per "
                    "port on both device and reference runs"
                )

            # Issue #827 (the waveguide instance of the NU witness gap):
            # energy ring-down witness per drive, worst over BOTH runs of
            # the pair -- the reference run feeds a_inc/P_inc, so its
            # truncation corrupts the same S values (same composition as
            # the uniform normalized/flux lanes, #538). NaN under tracing
            # propagates instead of being max()-eaten.
            _sd_dev = settling_db_from_port_records(
                [dev_wg[e.name] for e in original_entries])
            _sd_ref = settling_db_from_port_records(
                [ref_wg[e.name] for e in original_entries])
            settling_runs.append(
                float("nan") if (np.isnan(_sd_dev) or np.isnan(_sd_ref))
                else max(_sd_dev, _sd_ref))

            # Compute ref_shifts from the first drive's configs (same
            # measured planes for every drive / run).
            if ref_shifts is None:
                shifts = []
                planes_out = []
                for entry in original_entries:
                    cfg = dev_wg[entry.name]
                    planes = waveguide_plane_positions(cfg)
                    desired = (
                        entry.reference_plane
                        if entry.reference_plane is not None
                        else planes["source"]
                    )
                    # Grading-zone assertion: beta for this shift is the
                    # boundary-cell value (cfg.dx), so the span the
                    # shift covers must lie inside one uniform zone of
                    # the profile -- otherwise fail loudly here rather
                    # than apply one beta across cells of two sizes.
                    _assert_nu_shift_span_in_one_grading_zone(
                        grid, cfg, desired, entry.name,
                    )
                    shifts.append(desired - planes["reference"])
                    planes_out.append(desired)
                ref_shifts = tuple(shifts)
                reference_planes_out = np.asarray(planes_out, dtype=float)
                # The same final configs, kept for the post-solve S21
                # phase residual below (its beta ingredients are the
                # extractor's own f_cutoff / dt / dx, not re-derived).
                final_cfgs = [dev_wg[e.name] for e in original_entries]

            drive_name = original_entries[drive_idx].name
            a_inc_ref, _ = extract_waveguide_port_waves(
                ref_wg[drive_name], ref_shift=ref_shifts[drive_idx],
            )
            safe_a_inc = jnp.where(
                jnp.abs(a_inc_ref) > 1e-30,
                a_inc_ref,
                jnp.ones_like(a_inc_ref),
            )

            if _flux_mode:
                # Power-flux magnitude + modal phase (mirrors the
                # uniform extract_waveguide_s_matrix_flux). Immune to
                # the band-edge a_inc_ref denominator collapse that
                # makes the normalize=True diagonal blow up (issue #88):
                # P_inc = |F_ref[drive]| is large and well-conditioned
                # across the whole band, not source-spectrum-weighted.
                F_ref = ref_result.waveguide_port_flux
                F_dev = dev_result.waveguide_port_flux
                if F_ref is None or F_dev is None:
                    raise RuntimeError(
                        "normalize='flux' on the NU path requires "
                        "per-port flux spectra; run_nonuniform_path did "
                        "not return waveguide_port_flux."
                    )
                # jnp-native (mirrors the uniform PR #172 flux-AD fix):
                # no np.asarray() concretization — keeps the whole flux
                # extraction on the AD tape so an eps_override-traced
                # device run yields finite gradients through the
                # S-matrix. Uses the DOUBLE-WHERE trick at sqrt(0) /
                # angle(0) (guard the INPUT, not just the output): a
                # single jnp.where still leaks NaN grad through the dead
                # branch (#171/#172/#148). Forward values are identical
                # to the prior np version for P_inc / P > 0.
                P_inc = jnp.abs(F_ref[drive_idx])
                safe_P_inc = jnp.where(
                    P_inc > 1e-60, P_inc, jnp.ones_like(P_inc)
                )
                recv_col = []
                for recv_idx in range(n_ports):
                    recv_name = original_entries[recv_idx].name
                    _, b_recv_dev = extract_waveguide_port_waves(
                        dev_wg[recv_name], ref_shift=ref_shifts[recv_idx],
                    )
                    # AD-safe angle (double-where): angle() has an
                    # undefined gradient at 0; angle(1)=0 matches
                    # np.angle(0)=0 so the primal is unchanged.
                    ratio = b_recv_dev / safe_a_inc
                    ratio_ok = jnp.abs(ratio) > 0.0
                    phase = jnp.angle(
                        jnp.where(ratio_ok, ratio, jnp.ones_like(ratio))
                    )
                    if recv_idx == drive_idx:
                        P_num = jnp.abs(F_ref[drive_idx] - F_dev[drive_idx])
                    else:
                        P_num = jnp.abs(F_dev[recv_idx])
                    # AD-safe sqrt (double-where): a perfect match/null
                    # makes the power ratio exactly 0, where
                    # d(sqrt)/dx = inf would leak 0*inf=nan through the
                    # backward pass; primal stays exactly sqrt(x) for
                    # x>0 and exactly 0 at x=0.
                    p_ratio = P_num / safe_P_inc
                    p_ok = p_ratio > 0.0
                    mag = jnp.where(
                        p_ok,
                        jnp.sqrt(
                            jnp.where(p_ok, p_ratio, jnp.ones_like(p_ratio))
                        ),
                        0.0,
                    )
                    recv_col.append(mag * jnp.exp(1j * phase))
                s_columns.append(recv_col)
                continue

            recv_col: list = []
            for recv_idx in range(n_ports):
                recv_name = original_entries[recv_idx].name
                _, b_ref = extract_waveguide_port_waves(
                    ref_wg[recv_name], ref_shift=ref_shifts[recv_idx],
                )
                _, b_dev = extract_waveguide_port_waves(
                    dev_wg[recv_name], ref_shift=ref_shifts[recv_idx],
                )
                if recv_idx == drive_idx:
                    recv_col.append((b_dev - b_ref) / safe_a_inc)
                else:
                    # Use a tighter guard than the diagonal safe_a_inc
                    # (1e-30): the NU path operates at lower float32
                    # signal levels (~1e-31) because the TFSF table
                    # injection scales with dt/dx. The reference
                    # outgoing wave b_ref at non-driven ports is
                    # proportional to the driven-port incident wave and
                    # can fall to ~1e-31 in float32. A 1e-30 guard
                    # fires falsely and replaces b_ref with 1.0, giving
                    # S21 = b_dev * 1e-31 instead of b_dev/b_ref ≈ 1.
                    # 1e-60 is safely below float32 underflow (~1e-38)
                    # so it only fires when b_ref is genuinely zero.
                    safe_b = jnp.where(
                        jnp.abs(b_ref) > 1e-60,
                        b_ref,
                        jnp.ones_like(b_ref),
                    )
                    recv_col.append(b_dev / safe_b)
            s_columns.append(recv_col)
    finally:
        self._waveguide_ports = original_entries

    # Issue #827 (waveguide instance): the ring-down witness now reaches
    # the NU lane -- same aggregate truncation warning and settling_db
    # field as the uniform single-mode lanes (#538). NaN entries (traced
    # AD runs) are skipped by the warner's finite mask; the array itself
    # is always attached for the record.
    settling_db = np.asarray(settling_runs, dtype=float)
    _warn_if_ringdown_truncated(
        settling_db, tuple(e.name for e in original_entries),
        num_periods=float(num_periods),
    )
    _s_params_nu = jnp.stack(
        [jnp.stack(col) for col in s_columns], axis=1)
    _reference_planes_nu = (
        reference_planes_out
        if reference_planes_out is not None
        else np.array(
            [
                e.reference_plane if e.reference_plane is not None
                else 0.0
                for e in original_entries
            ],
            dtype=float,
        )
    )
    # Post-solve discretization witness (post-v1.8 plan item 5), same
    # banner and same field as the uniform lane. Report-only.
    _ph_rms_nu, _ph_meta_nu = _waveguide_s21_phase_residual(
        _s_params_nu, port_freqs, _reference_planes_nu, final_cfgs or [],
        normalize=normalize,
    )
    return WaveguideSMatrixResult(
        s_params=_s_params_nu,
        freqs=jnp.asarray(port_freqs),
        port_names=tuple(e.name for e in original_entries),
        port_directions=tuple(e.direction for e in original_entries),
        reference_planes=_reference_planes_nu,
        settling_db=settling_db,
        s21_phase_residual_deg_rms=_ph_rms_nu,
        s21_phase_residual_meta=_ph_meta_nu,
    )


# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly.
#
# Both functions above were ``def``s in the ``_SparamMixin`` class body, so
# their ``__qualname__`` read ``_SparamMixin.<name>``; a module-level ``def``
# gets the bare ``<name>`` instead. ``rfx/api/__init__.py`` rewrites exactly
# ``_SparamMixin.<name>`` -> ``Simulation.<name>`` at class-composition time so
# that a bad keyword argument reports ``Simulation.compute_waveguide_s_matrix()
# got an unexpected keyword argument``, and it SKIPS any function whose
# qualname does not match that pattern. Leaving the bare name here would
# therefore change those TypeError messages — a user-visible behaviour change
# in a pure code-motion step. ``tests/unit/autodiff/test_design_mask_removed.py
# ::test_no_public_simulation_method_leaks_a_mixin_class_name`` pins it.
# ---------------------------------------------------------------------------
compute_waveguide_s_matrix.__qualname__ = "_SparamMixin.compute_waveguide_s_matrix"
_compute_waveguide_s_matrix_nu.__qualname__ = (
    "_SparamMixin._compute_waveguide_s_matrix_nu"
)
