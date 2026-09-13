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

import jax
import jax.numpy as jnp
import numpy as np

try:  # Public API on current JAX; the 0.4.x GPU image uses the old location.
    from jax import enable_x64 as _enable_x64
except ImportError:
    from jax.experimental import enable_x64 as _enable_x64

from rfx.core.jax_utils import is_tracer
from rfx.sources.sources import GaussianPulse
from rfx.sources.coaxial_port import CoaxialPort
from rfx.sources.waveguide_port import (
    extract_waveguide_s_matrix,
    extract_waveguide_s_matrix_flux,
    extract_waveguide_s_params_normalized,
    extract_multimode_s_matrix,
    extract_multimode_s_matrix_flux,
    waveguide_plane_positions,
)

from rfx.nonuniform import NonUniformGrid, interior_cells

from rfx.api._spec import (
    WaveguideSMatrixResult,
    CoaxialSMatrixResult,
    CoaxialLineReflectionResult,
    CoaxialTwoPortResult,
    MSLSMatrixResult,
    MixedSMatrixResult,
    CoaxMSLTransitionResult,
    _WaveguidePortEntry,
    _MSLPortEntry,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    # Type-only forward reference to the composed class for the
    # ``port_reference_sims: "list[Simulation] | None"`` annotation. This is
    # NOT a runtime import: the module-import contract above forbids a runtime
    # ``from rfx.api import ...`` (cycle / sole-composition-point), and a
    # TYPE_CHECKING guard never executes, so the contract is preserved.
    from rfx.api import Simulation


# ---------------------------------------------------------------------------
# #980 Phase 2 re-export surface.
#
# The helper bodies that used to sit here now live in ``rfx.sparams._common``,
# moved verbatim. They are re-bound as module globals of THIS module because
# two things depend on that namespace and neither would fail loudly if it
# quietly emptied out:
#
#   * 47 files do ``from rfx.api._sparams import <helper>``;
#   * ``tests/unit/sparams/test_sparam_passivity_guard.py`` and
#     ``tests/unit/sparams/test_waveguide_port_reference_sims.py`` patch
#     ``"rfx.api._sparams.<name>"`` by STRING, which binds nothing and passes
#     vacuously the moment the name stops living here;
#
# and because ``_SparamMixin`` below calls these helpers by bare name, so they
# have to resolve as globals of this module. Listed explicitly, not
# ``import *``: the surface is the contract.
# ---------------------------------------------------------------------------
from rfx.sparams._common import (
    _msl_cell_profile,
    msl_modal_voltage,
    msl_solve_s_from_waves,
    _msl_wave_split_reliability,
    _warn_msl_wave_split_unreliable,
    _warn_msl_beta_scan_railed,
    _SETTLING_WITNESS_DB,
    settling_verdict,
    _validate_extra_flux_monitor_entries,
    _warn_if_ringdown_truncated,
    _nu_shift_span_cells,
    _assert_nu_shift_span_in_one_grading_zone,
    _msl_axis_spacing,
    _resolve_msl_auto_offsets,
    _project_passive,
    _warn_if_passivity_projected,
    WAVEGUIDE_RECIPROCITY_ADVISORY_TOL,
    _reciprocity_advisory_message,
    _warn_if_nonpassive_smatrix,
    _finalize_sparam_result,
    _C0_SPARAMS,
    WAVEGUIDE_PHASE_MAG_FLOOR,
    WAVEGUIDE_PHASE_BETA_CONVENTION,
    s21_phase_residual_deg_rms,
    _waveguide_s21_phase_residual,
    _warn_junction_probe_clearance,
    _warn_junction_cpml_thickness,
    _warn_ntff_box_dropped,
    _assemble_mixed_power_wave_s,
    _mixed_reciprocity_deviation,
    _mixed_flux_magnitude_override,
    _FAR_PORT_LAMBDA_G_FRACTION,
    _warn_thin_absorber_vs_guide_wavelength,
    _assemble_coaxial_two_port_from_voltages,
    _ladder_split_witness,
    _assemble_coax_msl_transition_from_voltages,
    _register_msl_h_planes,
    _collocated_msl_h,
    _msl_power_wave_scales,
)


class _SparamMixin:
    """S-parameter extraction methods mixed into :class:`Simulation`."""

    def _resolve_msl_probe_entries(self, grid):
        """One resolved ladder for preflight and both MSL extraction lanes."""
        return _resolve_msl_auto_offsets(self, list(self._msl_ports), grid)

    # Runtime-only; set by compute_msl_s_matrix for the duration of a run.
    _dft_plane_regions: dict[str, tuple[int, int, int, int]]

    # #980 Phase 2: the body moved verbatim to ``rfx/sparams/waveguide.py``.
    # Bound here as a class attribute so ``sim.compute_waveguide_s_matrix``
    # keeps its name, signature, ``__doc__`` and bound-method behaviour.
    # The import sits in the CLASS body rather than the module import block
    # on purpose: a module-level ``from rfx.sparams import waveguide as
    # _waveguide`` would add ``_waveguide`` to ``rfx.api._sparams``'s module
    # namespace, and
    # ``tests/locks/test_sparams_split_bit_identity.py`` pins that namespace
    # by SET EQUALITY (no name added, none dropped).
    from rfx.sparams.waveguide import compute_waveguide_s_matrix

    # #980 Phase 2: body moved VERBATIM to ``rfx/sparams/msl.py``. The import
    # is class-scoped and deleted so ``rfx.api._sparams``'s module namespace
    # stays exactly the pre-split 66-name re-export surface the lock pins
    # (a module-level ``_msl`` reds
    # ``test_sparams_module_namespace_is_the_declared_re_export_surface``).
    from rfx.sparams import msl as _msl

    compute_msl_s_matrix = _msl.compute_msl_s_matrix

    del _msl

    # #980 Phase 2: the body moved VERBATIM to ``rfx/sparams/mixed.py``.
    # Imported into the class body, not aliased at module level, because
    # ``tests/locks/test_sparams_split_bit_identity.py`` pins this module's
    # namespace by SET EQUALITY against the pre-split main -- a module-level
    # ``_mixed`` alias is a surface change and reds that lock. Binding the
    # function itself here keeps ``Simulation.compute_mixed_s_matrix``'s name,
    # signature, ``__doc__`` and bound-method behaviour byte for byte.
    from rfx.sparams.mixed import compute_mixed_s_matrix

    from rfx.sparams import coax as _coax

    compute_coaxial_s_matrix = _coax.compute_coaxial_s_matrix

    compute_coaxial_line_reflection = _coax.compute_coaxial_line_reflection

    compute_coaxial_two_port = _coax.compute_coaxial_two_port

    compute_coax_msl_transition = _coax.compute_coax_msl_transition

    del _coax

    # #980 Phase 2: the body moved verbatim to ``rfx/sparams/waveguide.py``
    # together with ``compute_waveguide_s_matrix`` above, which calls it as
    # ``self._compute_waveguide_s_matrix_nu(...)``. Class-body import for the
    # same namespace-lock reason spelled out at that binding.
    from rfx.sparams.waveguide import _compute_waveguide_s_matrix_nu
