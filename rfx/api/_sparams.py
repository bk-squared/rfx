"""S-parameter extraction methods for :class:`Simulation`.

Import contract (Part B Stage 2 refactor):
  This module is a transitional mixin. It must import ONLY from
  ``rfx.api._spec`` plus external ``rfx.*`` / stdlib / jax / numpy.
  It must NEVER do ``from rfx.api import ...`` or ``from . import ...``
  the package, to keep ``rfx/api/__init__.py`` the sole composition point.

The methods here were moved verbatim out of ``rfx/api/__init__.py``'s
``class Simulation`` body. They are pure structural relocations — same
indentation, decorators, signatures, docstrings, and logic. ``Simulation``
inherits ``_SparamMixin`` so every method below remains a bound method on
``Simulation`` instances; all existing call-sites are unaffected.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rfx.sources.sources import GaussianPulse
from rfx.sources.coaxial_port import CoaxialPort
from rfx.sources.waveguide_port import (
    extract_waveguide_s_matrix,
    extract_waveguide_s_matrix_flux,
    extract_waveguide_s_params_normalized,
    extract_multimode_s_matrix,
    waveguide_plane_positions,
)

from rfx.api._spec import (
    WaveguideSMatrixResult,
    CoaxialSMatrixResult,
    MSLSMatrixResult,
    _WaveguidePortEntry,
    _MSLPortEntry,
)


class _SparamMixin:
    """S-parameter extraction methods mixed into :class:`Simulation`."""

    def compute_waveguide_s_matrix(
        self,
        *,
        n_steps: int | None = None,
        num_periods: float = 20.0,
        normalize: bool | str = False,
        subpixel_smoothing: bool | str = False,
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
            dispersion error introduced by ``normalize=True``.

            ``True`` — two-run modal normalization.  Cancels one-way
            Yee dispersion for **transmission** (off-diagonal) by
            dividing device outgoing waves by reference outgoing waves
            at the same port.  **Does not** cancel dispersion for
            reflection (round-trip vs one-way path mismatch); use
            ``normalize=False`` or ``normalize="flux"`` for S11 of
            strong reflectors.

            ``"flux"`` — hybrid power-flux extraction.  Magnitude from
            Poynting-vector DFT (|S|² = P_flux / P_inc), phase from
            modal V/I.  Corrects both the Z_TE impedance-mismatch error
            in S11 and the round-trip dispersion error in the
            ``normalize=True`` diagonal formula.  Costs 2 × N_ports
            FDTD runs (same as ``normalize=True``).
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

        # Non-uniform-mesh dispatch. Earlier the uniform scan ran with
        # the coarse boundary dx and silently ignored ``dx_profile`` /
        # ``dy_profile`` (handover v2 experiment 12). The dedicated NU
        # two-run extractor below is enabled when its supported scope
        # is met (``normalize=True``, single-mode ports); otherwise
        # raise so the user is not given silently-wrong numbers.
        if self._dx_profile is not None or self._dy_profile is not None:
            unsupported = []
            if normalize is not True:
                unsupported.append("normalize=True is required")
            if any(entry.n_modes > 1 for entry in entries):
                unsupported.append("multi-mode ports (n_modes>1) are not supported")
            if unsupported:
                raise NotImplementedError(
                    "compute_waveguide_s_matrix() on a non-uniform mesh "
                    "(dx_profile / dy_profile) supports normalize=True "
                    "and single-mode ports. "
                    + "; ".join(unsupported)
                    + ". Drop the dx/dy profile to use the uniform lane."
                )
            return self._compute_waveguide_s_matrix_nu(
                n_steps=n_steps,
                num_periods=num_periods,
                normalize=normalize,
            )

        grid = self._build_grid()
        base_materials, debye_spec, lorentz_spec, pec_mask_wg, pec_shapes, boundary_pec_shapes, _ = self._assemble_materials(grid)
        # Waveguide S-matrix runner doesn't support pec_mask yet.
        # Fold PEC mask back into high sigma for compatibility.
        # **Stage 2 caveat**: when ``subpixel_smoothing="kottke_pec"`` is
        # active (use_kottke_pec, computed below), the inverse-eps
        # tensor encodes the PEC zero directly (inv = 0 freezes the
        # field). Folding pec_mask to sigma=1e10 then would conflict
        # with the Yee-stagger offsets in inv_xx/yy/zz: pec_mask is
        # per-cell-center, but inv_xx is at Ex(i+0.5, j, k) offsets,
        # so PEC boundary cells can have sigma=1e10 AND a fractional
        # inv > 0 — that combo blows up Ca ≈ -1 and field NaNs.
        # Skipped for Stage 2; the Kottke union (inv=0 inside PEC,
        # fractional at boundary) is the single source of truth.
        _use_kottke_pec_early = (subpixel_smoothing == "kottke_pec")
        if pec_mask_wg is not None and not _use_kottke_pec_early:
            base_materials = base_materials._replace(
                sigma=jnp.where(pec_mask_wg, 1e10, base_materials.sigma))
        materials = base_materials
        if n_steps is None:
            n_steps = grid.num_timesteps(num_periods=num_periods)
        _, debye, lorentz = self._init_dispersion(materials, grid.dt, debye_spec, lorentz_spec)

        def _resolve_freqs(entry: _WaveguidePortEntry) -> jnp.ndarray:
            if entry.freqs is not None:
                return entry.freqs
            return jnp.linspace(self._freq_max / 10, self._freq_max, entry.n_freqs)

        freqs = _resolve_freqs(entries[0])
        for entry in entries[1:]:
            entry_freqs = _resolve_freqs(entry)
            if entry_freqs.shape != freqs.shape or not np.allclose(np.asarray(entry_freqs), np.asarray(freqs)):
                raise ValueError("waveguide S-matrix requires matching frequency grids on all ports")

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

        # Compute Kottke per-component smoothed permittivity if requested.
        # Shared by both single-mode and multi-mode paths.
        # Mirrors rfx/runners/uniform.py: shape_eps_pairs from sim geometry,
        # then compute_smoothed_eps. The reference run is vacuum and has no
        # ε interfaces, so it always passes aniso_eps=None inside the
        # extractor.
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

        if has_multimode:
            # Multi-mode path: each raw_cfg is a list of WaveguidePortConfig
            port_mode_cfgs: list[list] = []
            for entry, raw in zip(entries, raw_cfgs):
                if isinstance(raw, list):
                    port_mode_cfgs.append(raw)
                else:
                    port_mode_cfgs.append([raw])

            ref_shifts_mm = []
            for entry, mode_cfgs in zip(entries, port_mode_cfgs):
                first_cfg = mode_cfgs[0]
                planes = waveguide_plane_positions(first_cfg)
                desired_ref = (
                    entry.reference_plane
                    if entry.reference_plane is not None
                    else planes["source"]
                )
                ref_shifts_mm.append(desired_ref - planes["reference"])

            if normalize:
                raise ValueError(
                    "compute_waveguide_s_matrix(normalize=True) is not yet "
                    "supported with n_modes > 1"
                )

            s_params, mode_map = extract_multimode_s_matrix(
                grid,
                materials,
                port_mode_cfgs,
                n_steps,
                boundary="cpml",
                cpml_axes=grid.cpml_axes,
                pec_axes="".join(axis for axis in "xyz" if axis not in grid.cpml_axes),
                debye=debye,
                lorentz=lorentz,
                ref_shifts=ref_shifts_mm,
                aniso_eps=aniso_eps,
                conformal_weights=conformal_weights,
                aniso_inv_eps=aniso_inv_eps,
            )
            reference_planes = np.array(ref_shifts_mm, dtype=float)
            # Build port names including mode indices
            port_names_mm = []
            port_directions_mm = []
            for port_idx, mode_idx, mtype, m_n in mode_map:
                entry = entries[port_idx]
                port_names_mm.append(f"{entry.name}_mode{mode_idx}_{mtype}{m_n[0]}{m_n[1]}")
                port_directions_mm.append(entry.direction)
            return WaveguideSMatrixResult(
                s_params=np.array(s_params),
                freqs=np.array(freqs),
                port_names=tuple(port_names_mm),
                port_directions=tuple(port_directions_mm),
                reference_planes=reference_planes,
            )

        # Single-mode path (original behavior)
        cfgs = raw_cfgs

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
            )
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
            )
        else:
            s_params = extract_waveguide_s_matrix(
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
            )
        reference_planes = np.array(
            [
                entry.reference_plane
                if entry.reference_plane is not None
                else waveguide_plane_positions(cfg)["reference"]
                for entry, cfg in zip(entries, cfgs)
            ],
            dtype=float,
        )
        return WaveguideSMatrixResult(
            s_params=np.array(s_params),
            freqs=np.array(freqs),
            port_names=tuple(entry.name for entry in entries),
            port_directions=tuple(entry.direction for entry in entries),
            reference_planes=reference_planes,
        )

    def compute_msl_s_matrix(
        self,
        *,
        n_steps: int | None = None,
        num_periods: float = 40.0,
        freqs: jnp.ndarray | None = None,
        n_freqs: int = 100,
        raw_3probe_dump_path: str | None = None,
        strict_extractor: bool = False,
    ) -> "MSLSMatrixResult":
        """Compute the MSL S-matrix using 3-probe numerical de-embedding.

        For each registered MSL port, runs one FDTD simulation with that
        port driven and the others passive (matched termination). At
        each port three downstream DFT plane probes record Ez and the
        first probe also records Hy; β, Z0 and the wave amplitudes are
        extracted post-scan via the OpenEMS-style 3-probe recurrence and
        assembled into the full S-matrix.

        Parameters
        ----------
        n_steps : int or None
            Timesteps per FDTD run. ``None`` → auto from ``num_periods``.
        num_periods : float
            Source-period multiples used to derive ``n_steps`` when not
            provided. Default 40 (MSL transients are slow to drain).
        freqs : array, optional
            Frequency grid. Defaults to
            ``linspace(freq_max / 10, freq_max, n_freqs)``.
        n_freqs : int
            Number of frequencies if ``freqs`` is None.
        raw_3probe_dump_path : str or None
            Optional ``.npz`` path. When provided, write the real
            simulation-derived 3-probe voltage/current phasors used by the
            extractor, together with the production S-matrix, so
            ``scripts/diagnostics/replay_msl_3probe_dump.py`` can independently
            replay the de-embedding without rerunning FDTD. This is intended
            for E3 validation evidence.
        strict_extractor : bool
            Honesty guard for the 3-probe de-embedding (issue #80 Fix A).
            After extraction, the per-frequency ``|q|`` and extracted ``Z0``
            are validated against physical bounds (``|q| <= 1`` for a passive
            line; extracted ``Z0`` within 10 % of the analytic
            Hammerstad-Jensen value). When ``False`` (default) a violation
            raises a loud :func:`warnings.warn`; when ``True`` it raises
            :class:`ValueError` instead. The extracted ``|S11|`` is
            unreliable when either bound is violated.

        Returns
        -------
        MSLSMatrixResult
        """
        from rfx.sources.msl_port import (
            MSLPort,
            _msl_yz_cells,
            compute_s21,
            extract_msl_s_params,
            msl_forward_amplitude,
            msl_probe_x_coords,
        )

        if not self._msl_ports:
            raise ValueError("No MSL ports registered. Call add_msl_port() first.")
        if self._ports or self._waveguide_ports or self._floquet_ports:
            raise NotImplementedError(
                "compute_msl_s_matrix() is defined only for add_msl_port(...) "
                "families in the current simulation. Use separate "
                "simulations for add_port(...), add_waveguide_port(...), "
                "or add_floquet_port(...) S-parameter workflows."
            )
        if self._tfsf is not None:
            raise NotImplementedError(
                "compute_msl_s_matrix() is not supported together with TFSF; "
                "TFSF is a plane-wave source, not an MSL port."
            )
        if self._coaxial_ports:
            raise NotImplementedError(
                "compute_msl_s_matrix() does not include add_coaxial_port(...); "
                "coaxial-port S-parameters need a separate validated V/I "
                "extraction and calibration contract."
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ):
            raise NotImplementedError(
                "compute_msl_s_matrix() currently supports the uniform Yee "
                "lane only. Drop dx_profile/dy_profile/dz_profile or use a "
                "documented diagnostic path."
            )
        if self._refinement is not None:
            raise NotImplementedError(
                "compute_msl_s_matrix() is not supported with SBP-SAT "
                "subgridding."
            )
        if self._solver == "adi":
            raise NotImplementedError(
                "compute_msl_s_matrix() is not supported with solver='adi'; "
                "use the uniform Yee solver."
            )

        entries = list(self._msl_ports)
        n_ports = len(entries)

        grid = self._build_grid()

        if freqs is None:
            freqs_arr = np.asarray(jnp.linspace(self._freq_max / 10, self._freq_max, n_freqs))
        else:
            freqs_arr = np.asarray(freqs)
        n_freqs_used = int(freqs_arr.shape[0])

        # Build MSLPort descriptors and probe x-coords once (geometry shared).
        msl_ports: list[MSLPort] = []
        for pe in entries:
            x_feed, y_centre, z_lo = pe.position
            msl_ports.append(MSLPort(
                feed_x=float(x_feed),
                y_lo=float(y_centre - pe.width / 2),
                y_hi=float(y_centre + pe.width / 2),
                z_lo=float(z_lo),
                z_hi=float(z_lo + pe.height),
                direction=pe.direction,
                impedance=pe.impedance,
                excitation=pe.waveform,
            ))

        probe_xs = [
            msl_probe_x_coords(
                grid, mp,
                n_offset_cells=pe.n_probe_offset,
                n_spacing_cells=pe.n_probe_spacing,
            )
            for mp, pe in zip(msl_ports, entries)
        ]

        # Per-axis cell-size arrays for V/I integration. Both uniform
        # and non-uniform grids are supported.
        def _profile(axis: str, n: int) -> np.ndarray:
            attr = {"x": "dx_profile", "y": "dy_profile", "z": "dz_profile"}[axis]
            prof = getattr(grid, attr, None)
            if prof is not None:
                return np.asarray(prof, dtype=float)
            return np.full(n, float(grid.dx), dtype=float)

        dy_arr = _profile("y", grid.ny)
        dz_arr = _profile("z", grid.nz)

        # Fixed cross-section indices per port (same across all runs).
        port_idx_meta = []
        for mp in msl_ports:
            cells = _msl_yz_cells(grid, mp)
            j_set = sorted({c[1] for c in cells})
            k_set = sorted({c[2] for c in cells})
            j_lo, j_hi = j_set[0], j_set[-1]
            k_lo, k_hi = k_set[0], k_set[-1]
            j_centre = (j_lo + j_hi) // 2
            k_top = k_hi  # trace sits at the top of the substrate
            port_idx_meta.append(dict(
                j_lo=j_lo, j_hi=j_hi,
                k_lo=k_lo, k_hi=k_hi,
                j_centre=j_centre, k_top=k_top,
                height=mp.z_hi - mp.z_lo,
            ))

        # Stash existing add_dft_plane_probe registrations and restore on exit.
        saved_dft = list(self._dft_planes)
        saved_msl = list(self._msl_ports)
        saved_ports = list(self._ports)
        try:
            S = np.zeros((n_ports, n_ports, n_freqs_used), dtype=complex)
            Z0_per_run = np.zeros((n_ports, n_freqs_used), dtype=complex)
            beta_first = np.zeros(n_freqs_used, dtype=complex)
            raw_v123 = np.zeros((n_ports, n_ports, 3, n_freqs_used), dtype=complex)
            raw_i1 = np.zeros((n_ports, n_ports, n_freqs_used), dtype=complex)
            raw_z0 = np.zeros((n_ports, n_ports, n_freqs_used), dtype=complex)
            raw_q = np.zeros((n_ports, n_ports, n_freqs_used), dtype=complex)

            for driven in range(n_ports):
                # Re-instantiate a clean simulation by mutating in place:
                # use add_msl_port as the registration path, but here we
                # need finer control over excite=True/False per-run, so
                # rebuild ``self._msl_ports`` for this run.
                run_entries = []
                for idx, pe in enumerate(entries):
                    new_excite = (idx == driven) and pe.excite
                    if new_excite:
                        wf = pe.waveform if pe.waveform is not None else \
                            GaussianPulse(f0=self._freq_max / 2, bandwidth=0.8)
                    else:
                        wf = None
                    run_entries.append(_MSLPortEntry(
                        name=pe.name, position=pe.position,
                        width=pe.width, height=pe.height,
                        direction=pe.direction, impedance=pe.impedance,
                        waveform=wf, excite=new_excite,
                        n_probe_offset=pe.n_probe_offset,
                        n_probe_spacing=pe.n_probe_spacing,
                        mode=pe.mode,
                        eps_r_sub=pe.eps_r_sub,
                    ))
                self._msl_ports = run_entries

                # Register DFT plane probes for V (Ez) and I (Hy).
                self._dft_planes = list(saved_dft)
                ez_probe_names: list[list[str]] = [[] for _ in range(n_ports)]
                hy_probe_names: list[str] = [None] * n_ports  # type: ignore
                for p_idx, (mp, pxs) in enumerate(zip(msl_ports, probe_xs)):
                    for q_idx, x_coord in enumerate(pxs):
                        nm = f"_msl_run{driven}_p{p_idx}_ez{q_idx}"
                        self.add_dft_plane_probe(
                            axis="x", coordinate=float(x_coord),
                            component="ez", freqs=jnp.asarray(freqs_arr),
                            name=nm,
                        )
                        ez_probe_names[p_idx].append(nm)
                    nm_hy = f"_msl_run{driven}_p{p_idx}_hy"
                    self.add_dft_plane_probe(
                        axis="x", coordinate=float(pxs[0]),
                        component="hy", freqs=jnp.asarray(freqs_arr),
                        name=nm_hy,
                    )
                    hy_probe_names[p_idx] = nm_hy

                # Run; pass n_steps through (None → auto).
                result = self.run(
                    n_steps=n_steps,
                    num_periods=num_periods,
                    compute_s_params=False,
                )
                planes = result.dft_planes or {}

                # Helper: integrate V and I per port from the recorded planes.
                v_per_port: list[list[np.ndarray]] = []
                i_first_per_port: list[np.ndarray] = []
                for p_idx, meta in enumerate(port_idx_meta):
                    vs = []
                    for nm in ez_probe_names[p_idx]:
                        ez_plane = np.asarray(planes[nm].accumulator)
                        # ez_plane shape: (n_freqs, ny, nz)
                        v_f = np.zeros(n_freqs_used, dtype=complex)
                        for k in range(meta["k_lo"], meta["k_hi"] + 1):
                            v_f = v_f + ez_plane[:, meta["j_centre"], k] * float(dz_arr[k])
                        vs.append(v_f)
                    v_per_port.append(vs)
                    hy_plane = np.asarray(planes[hy_probe_names[p_idx]].accumulator)
                    ny_grid = hy_plane.shape[1]
                    # Yee: Hy[:,j,k] lives at z=(k+0.5)*dz; k_hi-1 is inside
                    # substrate just below trace surface where H is largest.
                    k_h = max(meta["k_lo"], meta["k_top"] - 1)
                    # fringing return current extends ~2*h laterally;
                    # widen y-integration to capture the full Ampere integral.
                    dy_local = float(dy_arr[meta["j_centre"]])
                    n_y_margin = max(2, int(round(2 * meta["height"] / dy_local)))
                    j_lo_ext = max(0, meta["j_lo"] - n_y_margin)
                    j_hi_ext = min(ny_grid - 1, meta["j_hi"] + n_y_margin)
                    i_f = np.zeros(n_freqs_used, dtype=complex)
                    for j in range(j_lo_ext, j_hi_ext + 1):
                        i_f = i_f + hy_plane[:, j, k_h] * float(dy_arr[j])
                    # Sign convention: for a +x propagating quasi-TEM wave
                    # with Ez>0 (ground→trace), Hy<0 (x̂×ẑ = −ŷ), so the
                    # raw Hy integral gives I<0 and Z0<0.  Negate for +x
                    # ports so that Z0 = V/I > 0 matches the physical
                    # current direction (current flows +x on trace).
                    # For -x ports the sign is already correct.
                    mp_p = msl_ports[p_idx]
                    if mp_p.direction == "+x":
                        i_f = -i_f
                    i_first_per_port.append(i_f)
                    raw_v123[driven, p_idx, 0, :] = v_per_port[p_idx][0]
                    raw_v123[driven, p_idx, 1, :] = v_per_port[p_idx][1]
                    raw_v123[driven, p_idx, 2, :] = v_per_port[p_idx][2]
                    raw_i1[driven, p_idx, :] = i_f
                    _, z0_p, q_p = extract_msl_s_params(
                        v_per_port[p_idx][0],
                        v_per_port[p_idx][1],
                        v_per_port[p_idx][2],
                        i_f,
                    )
                    raw_z0[driven, p_idx, :] = z0_p
                    raw_q[driven, p_idx, :] = q_p

                # Driven port: full 3-probe extraction → S[driven, driven].
                v1d, v2d, v3d = v_per_port[driven]
                i1d = i_first_per_port[driven]
                s11_d, z0_d, q_d = extract_msl_s_params(v1d, v2d, v3d, i1d)
                S[driven, driven, :] = s11_d
                Z0_per_run[driven, :] = z0_d
                if driven == 0:
                    # β = -ln(q) / Δ; Δ = n_probe_spacing * dx
                    spacing = entries[0].n_probe_spacing * float(grid.dx)
                    beta_first = -np.log(q_d + 0j) / (spacing + 1e-30)

                # Driven port forward amplitude (for S21-style off-diagonals).
                alpha_d, _ = msl_forward_amplitude(v1d, v2d, v3d)

                for j in range(n_ports):
                    if j == driven:
                        continue
                    v1p, v2p, v3p = v_per_port[j]
                    alpha_p, _ = msl_forward_amplitude(v1p, v2p, v3p)
                    S[j, driven, :] = compute_s21(alpha_p, alpha_d)

            # --- Honesty guard on the 3-probe extractor (issue #80 Fix A) ---
            # raw_q / raw_z0 are now fully populated for every driven run and
            # every port. Validate them against physical bounds and surface a
            # loud warning (or ValueError when strict_extractor=True) so the
            # caller does not silently optimize an unreliable |S11|.
            import warnings as _w

            from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

            _Q_EPS = 1e-6
            _Z0_TOL = 0.10
            for driven in range(n_ports):
                pe = entries[driven]
                eps_r_ref = pe.eps_r_sub if pe.eps_r_sub is not None else 1.0
                z0_hj, _ = hammerstad_jensen_z0_eps_eff(
                    pe.width, pe.height, eps_r_ref
                )
                q_abs = np.abs(raw_q[driven, driven, :])
                k_q = int(np.argmax(q_abs))
                q_max = float(q_abs[k_q])
                z0_dev = np.abs(raw_z0[driven, driven, :] - z0_hj) / z0_hj
                k_z = int(np.argmax(z0_dev))
                z0_dev_max = float(z0_dev[k_z])
                violations: list[str] = []
                if q_max > 1.0 + _Q_EPS:
                    violations.append(
                        f"|q| = {q_max:.4f} > 1 at f = "
                        f"{freqs_arr[k_q] / 1e9:.4f} GHz (non-physical for a "
                        f"passive line)"
                    )
                if z0_dev_max > _Z0_TOL:
                    violations.append(
                        f"extracted Z0 = "
                        f"{raw_z0[driven, driven, k_z].real:.2f} ohm deviates "
                        f"{z0_dev_max * 100:.1f}% from the analytic "
                        f"Hammerstad-Jensen Z0 = {z0_hj:.2f} ohm at f = "
                        f"{freqs_arr[k_z] / 1e9:.4f} GHz "
                        f"(> {_Z0_TOL * 100:.0f}% tolerance)"
                    )
                if violations:
                    msg = (
                        f"compute_msl_s_matrix: 3-probe extractor is unstable "
                        f"for MSL port {pe.name!r}: "
                        + "; ".join(violations)
                        + ". The extracted |S11| is UNRELIABLE — see issue "
                        "#80. Increase n_probe_offset / n_probe_spacing, or "
                        "use a field-based loss."
                    )
                    if strict_extractor:
                        raise ValueError(msg)
                    _w.warn(msg, stacklevel=2)

            if raw_3probe_dump_path is not None:
                import json
                from pathlib import Path

                path = Path(raw_3probe_dump_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                metadata = {
                    "schema": "rfx.msl_3probe_dump",
                    "schema_version": 1,
                    "production_smatrix_schema": "S[receiver_port, driven_port, frequency_index]",
                    "raw_v123_shape": "(n_driven, n_ports, 3, n_freqs)",
                    "raw_i1_shape": "(n_driven, n_ports, n_freqs)",
                    "phase_convention": "DFT accumulator convention from add_dft_plane_probe",
                    "current_convention": (
                        "line current sign normalized so +x and -x MSL ports "
                        "produce positive characteristic impedance on the "
                        "validated thru-line envelope"
                    ),
                    "deembedding": (
                        "three equally spaced voltage probes plus current at "
                        "probe 1; replay computes q, alpha, gamma, S11, Sij "
                        "from raw phasors without calling compute_msl_s_matrix"
                    ),
                    "grid": {
                        "dx_m": float(grid.dx),
                        "dt_s": float(grid.dt),
                        "nx": int(grid.nx),
                        "ny": int(grid.ny),
                        "nz": int(grid.nz),
                    },
                    "simulation": {
                        "freq_max_hz": float(self._freq_max),
                        "num_periods": float(num_periods),
                        "n_steps": None if n_steps is None else int(n_steps),
                    },
                    "port_definitions": [
                        {
                            "name": str(pe.name),
                            "position_m": [float(x) for x in pe.position],
                            "width_m": float(pe.width),
                            "height_m": float(pe.height),
                            "direction": pe.direction,
                            "impedance_ohm": float(pe.impedance),
                            "n_probe_offset": int(pe.n_probe_offset),
                            "n_probe_spacing": int(pe.n_probe_spacing),
                            "mode": pe.mode,
                        }
                        for pe in entries
                    ],
                }
                np.savez(
                    path,
                    metadata_json=np.asarray(json.dumps(metadata)),
                    freqs_hz=np.asarray(freqs_arr, dtype=np.float64),
                    raw_v123=raw_v123,
                    raw_i1=raw_i1,
                    raw_z0=raw_z0,
                    raw_q=raw_q,
                    production_smatrix=S,
                    production_z0=Z0_per_run,
                    production_beta=beta_first,
                    port_names=np.asarray(tuple(pe.name for pe in entries), dtype=object),
                    driven_port_indices=np.arange(n_ports, dtype=np.int64),
                )

            return MSLSMatrixResult(
                S=S,
                freqs=np.asarray(freqs_arr),
                Z0=Z0_per_run,
                beta=beta_first,
                port_names=tuple(pe.name for pe in entries),
            )
        finally:
            self._dft_planes = saved_dft
            self._msl_ports = saved_msl
            self._ports = saved_ports

    def compute_coaxial_s_matrix(
        self,
        *,
        n_steps: int = 320,
        freqs: jnp.ndarray | None = None,
        n_freqs: int = 21,
        field_scale: float = 1.0e4,
        magnetic_ratio: float = 1.0,
        signal_floor: float = 1.0e-12,
        reference_plane_axial_index_offset: int = 0,
    ) -> "CoaxialSMatrixResult":
        """Experimental coaxial S-matrix via distributed TEM plane sources.

        For each registered ``add_coaxial_port(...)`` port, runs one FDTD
        simulation with that port driven and all other coaxial ports passive.
        A distributed transverse E/M plane source is injected on the port's
        cross-section (the M67 prototype scaffold promoted to the public
        API); DFT plane probes capture the resulting Ex/Ey/Hx/Hy on every
        coaxial port's reference plane; the V/I extractor recovers ``V`` and
        ``I`` via the radial line / azimuthal loop integrals; and the
        standard power-wave decomposition assembles the full S-matrix.

        Status: **experimental**. The plane source can produce a residual
        forward wave and the extracted reference-plane V/I has known
        amplitude bias for coarse grids; ``status="degraded"`` is reported
        when any V/I sample falls below ``signal_floor``. Use this API for
        development; do not promote claims beyond E2/E3 without an external
        cross-solver fixture (see ``port_external_reference_requirements``).

        Parameters
        ----------
        n_steps:
            FDTD timesteps per driven-port run. Default 320.
        freqs:
            Frequency grid (Hz). Defaults to a uniform grid covering
            ``[freq_max / 10, freq_max]``.
        n_freqs:
            Number of frequencies if ``freqs`` is None. Default 21.
        field_scale:
            Linear scale on the radial E waveform. Increase to lift the
            plane signal above DFT noise (V/I extraction is amplitude-linear
            so the S-matrix is invariant under this scale).
        magnetic_ratio:
            Multiplier on the ``H`` waveform after the analytic ``1/Z_TEM``
            factor. ``1.0`` injects the lossless-TEM Poynting-balanced
            amplitude; smaller values bias toward an E-only injection.
        signal_floor:
            Absolute V or I phasor magnitude below which the result is
            flagged as ``"degraded"``.
        reference_plane_axial_index_offset:
            Axial-index offset for the source/probe plane relative to the
            port pin centre.

        Returns
        -------
        CoaxialSMatrixResult
        """

        from rfx.probes.probes import init_dft_plane_probe
        from rfx.simulation import run as _run
        from rfx.sources.coaxial_port import (
            build_coaxial_tem_plane_source_specs,
            extract_coaxial_plane_vi_from_dft,
        )

        if not self._coaxial_ports:
            raise ValueError(
                "No coaxial ports registered. Call add_coaxial_port() first."
            )
        if (
            self._ports
            or self._waveguide_ports
            or self._floquet_ports
            or self._msl_ports
        ):
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is defined only for "
                "add_coaxial_port(...) families in the current simulation."
            )
        if self._tfsf is not None:
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is not supported with TFSF; "
                "TFSF is a plane-wave source, not a coaxial port."
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ):
            raise NotImplementedError(
                "compute_coaxial_s_matrix() supports the uniform Yee lane only."
            )

        ports: list[CoaxialPort] = list(self._coaxial_ports)
        n_ports = len(ports)

        # Build the working grid + materials with all coaxial geometries
        # stamped (PEC center pin, PTFE dielectric fill, PEC outer shell from
        # M66). ``_build_materials`` only assembles bulk materials and shapes;
        # ``add_coaxial_port`` only registers the port descriptor, so without
        # this loop the FDTD would run with the source dropped into pure
        # vacuum and the wave would radiate bidirectionally with no coax
        # structure to confine it (this is the real source of the
        # calibration-blocked status documented in the handover).
        from rfx.sources.coaxial_port import (
            setup_coaxial_port,
            add_coaxial_matched_termination,
            add_coaxial_open_termination,
            add_coaxial_pec_end_cap,
        )
        grid = self._build_grid()
        materials, _, _ = self._build_materials(grid)
        for p in ports:
            materials = setup_coaxial_port(grid, p, materials)
        for term_port_idx, term_R, term_offset_cells in self._coaxial_terminations:
            materials = add_coaxial_matched_termination(
                grid,
                ports[term_port_idx],
                materials,
                target_impedance=term_R,
                axial_offset_cells=term_offset_cells,
            )
        for term_port_idx, retract_cells in self._coaxial_open_terminations:
            materials = add_coaxial_open_termination(
                grid,
                ports[term_port_idx],
                materials,
                pin_retract_cells=retract_cells,
            )
        for cap_port_idx, cap_offset_cells in self._coaxial_pec_end_caps:
            materials = add_coaxial_pec_end_cap(
                grid,
                ports[cap_port_idx],
                materials,
                axial_offset_cells=cap_offset_cells,
            )

        # Frequency grid.
        if freqs is None:
            freqs = jnp.linspace(
                self._freq_max / 10.0,
                self._freq_max,
                int(n_freqs),
                dtype=jnp.float32,
            )
        else:
            freqs = jnp.asarray(freqs, dtype=jnp.float32)

        # Reference-plane axial indices per port (cross-section z-plane).
        from rfx.sources.coaxial_port import _coaxial_port_geometry
        plane_indices: list[int] = []
        for p in ports:
            _, _, _, pin_center, _, _ = _coaxial_port_geometry(grid, p)
            plane_indices.append(
                int(grid.position_to_index(pin_center)[2])
                + int(reference_plane_axial_index_offset)
            )

        # Output buffers.
        n_freqs_used = int(freqs.shape[0])
        s = np.zeros((n_ports, n_ports, n_freqs_used), dtype=np.complex128)
        z_tem_arr = np.zeros((n_ports, n_freqs_used), dtype=np.complex128)
        v_dump = np.zeros((n_ports, n_ports, n_freqs_used), dtype=np.complex128)
        i_dump = np.zeros((n_ports, n_ports, n_freqs_used), dtype=np.complex128)

        status = "passed"

        for driven in range(n_ports):
            spec = build_coaxial_tem_plane_source_specs(
                grid=grid,
                port=ports[driven],
                n_steps=int(n_steps),
                field_scale=float(field_scale),
                magnetic_ratio=float(magnetic_ratio),
                reference_plane_axial_index_offset=int(
                    reference_plane_axial_index_offset
                ),
            )
            z_tem_arr[driven, :] = complex(spec.z_tem_ohm)

            # DFT plane probes on every port's cross-section.
            dft_planes = []
            for p_idx, p in enumerate(ports):
                for component in ("ex", "ey", "hx", "hy"):
                    dft_planes.append(
                        init_dft_plane_probe(
                            axis=2,
                            index=plane_indices[p_idx],
                            component=component,
                            freqs=freqs,
                            grid_shape=grid.shape,
                            dft_total_steps=int(n_steps),
                        )
                    )

            result = _run(
                grid,
                materials,
                int(n_steps),
                boundary="pec",
                sources=list(spec.electric_sources),
                mag_sources=list(spec.magnetic_sources),
                dft_planes=dft_planes,
                return_state=False,
            )
            if result.dft_planes is None:
                raise RuntimeError(
                    "compute_coaxial_s_matrix(): runner returned no DFT planes"
                )

            # Slice DFT planes back into per-port (ex, ey, hx, hy) groups.
            per_port: list[dict[str, np.ndarray]] = []
            for p_idx in range(n_ports):
                start = p_idx * 4
                group = result.dft_planes[start : start + 4]
                comp_map = {
                    probe.component: np.asarray(probe.accumulator, dtype=np.complex128)
                    for probe in group
                }
                per_port.append(comp_map)

            # Extract V/I at each port's reference plane.
            voltages = []
            currents = []
            for p_idx, p in enumerate(ports):
                vi = extract_coaxial_plane_vi_from_dft(
                    grid=grid,
                    port=p,
                    plane_axial_index=plane_indices[p_idx],
                    ex_dft=per_port[p_idx]["ex"],
                    ey_dft=per_port[p_idx]["ey"],
                    hx_dft=per_port[p_idx]["hx"],
                    hy_dft=per_port[p_idx]["hy"],
                )
                v = np.asarray(vi.vi.voltage, dtype=np.complex128)
                i = np.asarray(vi.vi.current, dtype=np.complex128)
                voltages.append(v)
                currents.append(i)
                v_dump[driven, p_idx, :] = v
                i_dump[driven, p_idx, :] = i
                if (
                    float(np.max(np.abs(v))) <= float(signal_floor)
                    or float(np.max(np.abs(i))) <= float(signal_floor)
                ):
                    status = "degraded"

            # Power-wave decomposition at each receive port (a_j at driven, b_i
            # at receiver) using the analytic Z_TEM as Z0.
            z0 = complex(spec.z_tem_ohm)
            a_j = (voltages[driven] + z0 * currents[driven]) / (2.0 * np.sqrt(z0))
            for receiver in range(n_ports):
                b_i = (voltages[receiver] - z0 * currents[receiver]) / (
                    2.0 * np.sqrt(z0)
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    s[receiver, driven, :] = np.where(
                        np.abs(a_j) > 0.0,
                        b_i / a_j,
                        np.nan + 1j * np.nan,
                    )

        reference_planes = np.asarray(
            [
                float(grid.position_to_index(p.position)[2] + reference_plane_axial_index_offset)
                * float(grid.dx)
                for p in ports
            ],
            dtype=float,
        )

        return CoaxialSMatrixResult(
            s_params=s,
            freqs=np.asarray(freqs, dtype=float),
            port_names=tuple(f"coax_{i}" for i in range(n_ports)),
            port_faces=tuple(p.face for p in ports),
            reference_planes=reference_planes,
            z_tem_ohm=z_tem_arr,
            voltages=v_dump,
            currents=i_dump,
            status=status,
        )

    def _compute_waveguide_s_matrix_nu(
        self,
        *,
        n_steps: int | None,
        num_periods: float,
        normalize: bool,
    ) -> WaveguideSMatrixResult:
        """Non-uniform-mesh two-run S-matrix extraction.

        Drives each port in turn, running device + vacuum-reference
        scans through ``run_nonuniform_path`` so ``dx_profile`` /
        ``dy_profile`` actually flow into the Yee update. The per-port
        drive is implemented by temporarily zeroing ``amplitude`` on
        non-driven entries; the original port list is restored in a
        ``finally`` block. Reference run uses ``eps_override`` /
        ``sigma_override`` to replace the assembled materials with
        vacuum before the scan launches.

        Current scope (matches the uniform path minus a few niceties):
          - ``normalize=True`` only.
          - Single-mode ports (``n_modes == 1``) only.

        Extracts ``a_inc`` / ``b_out`` via the same
        ``extract_waveguide_port_waves`` helper as the uniform path and
        applies the same diagonal-subtraction + off-diagonal-division
        normalisation (see ``extract_waveguide_s_params_normalized``
        in ``rfx/sources/waveguide_port.py``).
        """
        from dataclasses import replace as _dc_replace
        from rfx.runners.nonuniform import (
            run_nonuniform_path,
            assemble_materials_nu,
        )
        from rfx.sources.waveguide_port import (
            extract_waveguide_port_waves,
            waveguide_plane_positions,
        )

        if not normalize:
            raise NotImplementedError(
                "compute_waveguide_s_matrix(normalize=False) is not yet "
                "supported on the non-uniform mesh path; use normalize=True "
                "or drop dx/dy_profile to stay on the uniform lane."
            )

        entries = list(self._waveguide_ports)
        if any(entry.n_modes > 1 for entry in entries):
            raise NotImplementedError(
                "Multi-mode waveguide ports are not yet supported on the "
                "non-uniform mesh path."
            )

        n_ports = len(entries)

        # ``_build_nonuniform_grid`` requires a concrete dz_profile.
        # Synthesise one from the scalar dx when the user did not supply
        # a dz_profile (same semantics as the uniform lane's implicit
        # z-resolution). Restored in the ``finally`` below.
        _dz_profile_saved = self._dz_profile
        if self._dz_profile is None:
            _nz = int(round(float(self._domain[2]) / float(self._dx)))
            self._dz_profile = np.full(max(_nz, 1), float(self._dx))

        # Build the grid directly so we can restrict ``cpml_axes`` to
        # axes that are not fully PEC/PMC-bounded. The rasteriser (see
        # ``rfx/geometry/rasterize.py::coords_from_nonuniform_grid``)
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
        try:
            grid = build_nonuniform_grid(
                self._freq_max, self._domain, self._dx, self._cpml_layers,
                self._dz_profile,
                dx_profile=self._dx_profile,
                dy_profile=self._dy_profile,
                pec_faces=pec_set or None,
                pmc_faces=pmc_set or None,
                cpml_axes=cpml_axes,
            )
        except Exception:
            self._dz_profile = _dz_profile_saved
            raise
        if n_steps is None:
            # ``NonUniformGrid`` does not expose ``num_timesteps`` (known
            # asymmetry vs. ``Grid``); inline the same formula here.
            n_steps = int(np.ceil(num_periods / self._freq_max / float(grid.dt)))

        # Assemble device materials once to learn the full array shape;
        # vacuum reference is shape-matched onto that same array.
        dev_materials_concrete, _, _, _ = assemble_materials_nu(self, grid)
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
        n_freqs = int(port_freqs.shape[0])

        s_matrix = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex64)
        ref_shifts: tuple[float, ...] | None = None
        reference_planes_out: np.ndarray | None = None

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

                dev_result = run_nonuniform_path(self, n_steps=n_steps)
                ref_result = run_nonuniform_path(
                    self,
                    n_steps=n_steps,
                    eps_override=vacuum_eps,
                    sigma_override=vacuum_sigma,
                )

                dev_wg = dev_result.waveguide_ports or {}
                ref_wg = ref_result.waveguide_ports or {}
                if len(dev_wg) != n_ports or len(ref_wg) != n_ports:
                    raise RuntimeError(
                        "NU waveguide S-matrix expected one final cfg per "
                        "port on both device and reference runs"
                    )

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
                        shifts.append(desired - planes["reference"])
                        planes_out.append(desired)
                    ref_shifts = tuple(shifts)
                    reference_planes_out = np.asarray(planes_out, dtype=float)

                drive_name = original_entries[drive_idx].name
                a_inc_ref, _ = extract_waveguide_port_waves(
                    ref_wg[drive_name], ref_shift=ref_shifts[drive_idx],
                )
                a_inc_ref_np = np.asarray(a_inc_ref)
                safe_a_inc = np.where(
                    np.abs(a_inc_ref_np) > 1e-30,
                    a_inc_ref_np,
                    np.ones_like(a_inc_ref_np),
                )

                for recv_idx in range(n_ports):
                    recv_name = original_entries[recv_idx].name
                    _, b_ref = extract_waveguide_port_waves(
                        ref_wg[recv_name], ref_shift=ref_shifts[recv_idx],
                    )
                    _, b_dev = extract_waveguide_port_waves(
                        dev_wg[recv_name], ref_shift=ref_shifts[recv_idx],
                    )
                    b_ref_np = np.asarray(b_ref)
                    b_dev_np = np.asarray(b_dev)

                    if recv_idx == drive_idx:
                        s_matrix[recv_idx, drive_idx, :] = (
                            b_dev_np - b_ref_np
                        ) / safe_a_inc
                    else:
                        safe_b = np.where(
                            np.abs(b_ref_np) > 1e-30,
                            b_ref_np,
                            np.ones_like(b_ref_np),
                        )
                        s_matrix[recv_idx, drive_idx, :] = b_dev_np / safe_b
        finally:
            self._waveguide_ports = original_entries
            self._dz_profile = _dz_profile_saved

        return WaveguideSMatrixResult(
            s_params=np.asarray(s_matrix),
            freqs=np.asarray(port_freqs),
            port_names=tuple(e.name for e in original_entries),
            port_directions=tuple(e.direction for e in original_entries),
            reference_planes=reference_planes_out
            if reference_planes_out is not None
            else np.array(
                [
                    e.reference_plane if e.reference_plane is not None
                    else 0.0
                    for e in original_entries
                ],
                dtype=float,
            ),
        )

