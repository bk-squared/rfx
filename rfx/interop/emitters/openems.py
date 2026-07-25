"""Project a ``rfx-design-ir/v1`` design document onto a runnable openEMS script.

``emit_openems_script(document)`` is text in, text out: it needs no solver, no
licence and no ``Simulation``.  The returned string is a self-contained Python
script that builds the structure with CSXCAD, runs openEMS once per driven
port, and writes the S-matrix to JSON.

Scope of v1 — read this before reading the code
-----------------------------------------------
The supported fence is **S-parameters from lumped / wire / MSL ports on a
uniform mesh**.  That is not an accident of effort: it is the intersection of
what this repository has actually exercised against openEMS (see the mapping
census in the branch's emitter notes — 10 scripts call the openEMS API, and
the port families they cover are lumped, wire and MSL) with what can be
translated without inventing a convention.  Everything else refuses.

Generable
    uniform ``dx`` mesh; ``pec`` / ``pmc`` / ``cpml`` / ``upml`` faces;
    ``add_material(eps_r=, sigma=)``; PEC-by-sigma materials; ``Box`` and
    ``Cylinder`` geometry in paint order; ``add_port`` (single-cell lumped and
    ``extent=`` wire); ``add_msl_port``; ``GaussianPulse`` excitation.

Refused, with the reason attached to each raise
    coaxial ports (no ``AddCoaxialPort`` exists — verified absent on the local
    v0.0.35 install — and the radial-``AddLumpedPort`` workaround is recorded
    as failed across three runs); every cell-relative coaxial termination;
    Floquet ports and ``periodic`` faces (``PBC`` raises "Unknown boundary
    condition"); ``add_waveguide_port`` (``AddRectWaveGuidePort`` exists but has
    zero in-repo precedent, and its rfx side carries cell-count extraction
    planes); ``add_tfsf_source``; ``add_lumped_rlc``; ``add_thin_conductor``;
    non-uniform mesh profiles; ``refinement``; rfx-only solver controls
    (``solver != "yee"``, ``precision != "float32"``, ``stencil_order != 2``);
    conformal PEC faces; ``mode != "3d"``; probes, DFT-plane probes, flux
    monitors and NTFF boxes; soft (impedance-0) sources; ``Sphere`` /
    ``PolylineWire`` / ``Via`` / ``CurvedPatch``; non-Gaussian waveforms;
    dispersive or magnetic dielectrics.

Several of those are refusals *for now* rather than verdicts — the reason
string says which.  The ones that are verdicts are the ones where a
translation would have to pick a convention the reader could not see.

What a generated script does NOT prove
--------------------------------------
**Structural equivalence of a generated setup is not evidence of physics
agreement.**  A generated script that runs and returns passive S-parameters
shows that the projection is executable and self-consistent.  It does not show
that openEMS and rfx agree, and it does not show that the projection is
faithful to the rfx design: the absorber formulations differ, the port models
differ (the repo's own wire-port envelope measures the lumped/wire port-
convention gap at ``≈0.20`` in ``|S|``), and the reference planes differ.  A
physics-agreement claim needs a matched-resolution run against a committed
reference, quoted with its preflight context — a separate exercise.

Divergences honoured
--------------------
Each is cited in the code at the point it is applied, using the numbering of
the branch's rfx→openEMS mapping report:

``D1``  CPML padding direction — rfx adds absorber cells **outside** the user
        domain (``rfx/grid.py``: ``nx = ceil(Lx/dx) + 1 + pad_lo + pad_hi``,
        and ``(idx - axis_pads[ax]) * dx`` recovers user coordinates), while
        openEMS ``PML_<N>`` consumes ``N`` cells **inside** the mesh extent it
        is handed.  The emitted mesh therefore spans the user domain *plus*
        the per-face pads, and ``PML_<N>`` eats exactly the added margin.
        Getting this wrong silently compares a different structure; the repo's
        own hand-written scripts get it wrong.
``D2``  ``ceil`` vs ``round`` on the cell count — rfx uses ``ceil``; the
        emitter follows rfx, so the emitted extent is ``n_cells * dx`` and may
        exceed ``domain``.
``D3``  ``PML_<N>`` is parameterised from ``cpml_layers``; the scripts'
        hard-coded ``PML_8`` is stale against the current rfx default of 16.
``D4``  ``MUR`` is never emitted (measured 8 % resonance error on a patch, and
        exponential blow-up on a dielectric).
``D5``  Off-grid ports are silently dropped by openEMS ("Unused primitive",
        ``uf_inc == 0``, all-NaN S).  Port coordinates are snapped exactly the
        way ``Grid.position_to_index`` snaps them (``round(pos/dx)``), so they
        land on mesh lines by construction; the emitter then *verifies* it and
        the generated script carries the runtime excitation guard.
``D6``  ``priority`` has no rfx counterpart and is synthesised from paint
        order — see ``_plan_geometry`` for the derivation, which is stronger
        than the scripts' de-facto ladder.
``D7``  Port-model span mismatch: rfx lumped ports are single-cell and rfx
        wire ports are a point-feed column, while openEMS lumped ports average
        over the span and ``MSLPort`` occupies a span along the propagation
        axis.  ``msl_port_w_cells`` has no rfx counterpart.
``D8``  Reference planes differ; only magnitudes are comparable without an
        explicit de-embedding contract, which no script in this repo has.
``D9``  Units — openEMS geometry is in mm via ``SetDeltaUnit(1e-3)``.
``D12`` ``EndCriteria`` vs a step count: a step count does not transfer
        between solvers because their timesteps differ, and ``EndCriteria``
        can fire early on a resonator.  Run control is an emitter parameter and
        is itemised in the header.
``D13`` ``CalcPort`` is called without a scalar ``ref_impedance`` (upstream bug
        when ``Z_ref`` is array-valued).
``D14`` ``FDTD.Run()`` needs an absolute path and leaves the process CWD inside
        the sim directory; the generated script passes an absolute path and
        restores the CWD in a ``finally``.
``D15`` The numpy-2.x alias shim must precede the openEMS import — the local
        numpy is 2.x and ``openEMS.ports.MSLPort`` itself uses ``np.int``.
``D16`` Metal thickness is taken verbatim from the rfx shape; the emitter does
        not choose between a sheet and a one-cell box.

Status: **provisional**.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, field
from typing import Any, Sequence

from rfx.interop._errors import UnsupportedDesignFeature

__all__ = [
    "OPENEMS_EMITTER_VERSION",
    "OpenEMSPlan",
    "emit_openems_script",
    "plan_openems_projection",
]

#: Emitter contract version.  Bump when the generated script's structure or the
#: supported fence changes in a way a reader of an old artifact would notice.
OPENEMS_EMITTER_VERSION = "rfx-openems-emitter/v1"

#: ``SetDeltaUnit`` value used by every openEMS script in this repository, and
#: therefore by the emitter: geometry coordinates are millimetres (D9).
UNIT_M = 1.0e-3

#: rfx treats a material with ``sigma >= 1e6`` S/m as a PEC mask rather than a
#: lossy dielectric (``rfx/api/_compile.py``: ``_PEC_SIGMA_THRESHOLD``, applied
#: in the geometry assembly loop).  The openEMS counterpart of that branch is
#: ``AddMetal``, not ``AddMaterial(kappa=...)``.
PEC_SIGMA_THRESHOLD = 1.0e6

#: Absorbing boundary tokens on the rfx side.
_ABSORBING = ("cpml", "upml")

#: Tolerance for "this coordinate is already a mesh line", relative to ``dx``.
#: Snapped coordinates are ``round(pos/dx) * dx``, so the residual is float
#: noise (~1e-16 relative); anything larger means a genuine off-grid feature.
_LINE_TOL_REL = 1.0e-9

_AXES = ("x", "y", "z")
_COMPONENT_AXIS = {"ex": 0, "ey": 1, "ez": 2}
_SUPPORTED_SHAPE_KINDS = ("box", "cylinder")


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------

def _refuse(construct: str, reason: str) -> UnsupportedDesignFeature:
    """Refusal naming the construct first, then why it cannot be projected.

    Per decision D4 of the design note the emitter refuses rather than
    approximating, and per the task's refusal contract the message must name
    the construct so a reader knows what to remove or which target to use.
    """
    return UnsupportedDesignFeature(
        f"openEMS emitter cannot project {construct}: {reason}"
    )


# ---------------------------------------------------------------------------
# Plan — the reviewable intermediate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _MaterialPlan:
    name: str
    ident: str
    is_metal: bool
    epsilon: float
    kappa: float


@dataclass(frozen=True)
class _GeometryPlan:
    index: int
    kind: str
    material_ident: str
    is_metal: bool
    priority: int
    start_mm: tuple[float, float, float]
    stop_mm: tuple[float, float, float]
    radius_mm: float | None = None


@dataclass(frozen=True)
class _PortPlan:
    number: int
    family: str  # "lumped" | "wire" | "msl"
    label: str
    impedance: float
    start_mm: tuple[float, float, float]
    stop_mm: tuple[float, float, float]
    excite: bool
    priority: int
    exc_dir: int
    prop_dir: int | None = None  # msl only
    f0_hz: float | None = None
    fc_hz: float | None = None


@dataclass(frozen=True)
class OpenEMSPlan:
    """Reviewable projection of a design document onto openEMS constructs.

    Exposed so tests (and humans) can check the arithmetic that matters —
    above all the CPML span (D1) — without parsing generated source text.
    All coordinates are millimetres, matching ``SetDeltaUnit(UNIT_M)``.
    """

    rfx_version: str
    schema: str
    dx_m: float
    n_cells: tuple[int, int, int]
    pad_lo: tuple[int, int, int]
    pad_hi: tuple[int, int, int]
    mesh_lines_mm: dict[str, tuple[float, ...]]
    boundary: tuple[str, str, str, str, str, str]
    materials: tuple[_MaterialPlan, ...]
    geometry: tuple[_GeometryPlan, ...]
    ports: tuple[_PortPlan, ...]
    driven_port_numbers: tuple[int, ...]
    freqs_hz: tuple[float, ...]
    n_timesteps: int
    end_criteria: float
    msl_port_w_cells: int
    approximations: tuple[str, ...] = field(default=())

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Mesh-line counts per axis — must equal the rfx ``Grid.shape`` (D1)."""
        return tuple(len(self.mesh_lines_mm[ax]) for ax in _AXES)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Document accessors (fail loudly rather than defaulting)
# ---------------------------------------------------------------------------

def _require_mapping(value: Any, what: str) -> dict:
    if not isinstance(value, dict):
        raise _refuse(
            what, f"expected a mapping in the design document, got {type(value).__name__}"
        )
    return value


def _require_list(value: Any, what: str) -> list:
    if not isinstance(value, list):
        raise _refuse(
            what, f"expected a list in the design document, got {type(value).__name__}"
        )
    return value


def _get(payload: dict, key: str, what: str) -> Any:
    if key not in payload:
        raise _refuse(
            f"{what}.{key}",
            "the key is absent from the design document; the document was not "
            "produced by rfx.interop.design_to_dict at this schema version",
        )
    return payload[key]


def _mm(value_m: float) -> float:
    return float(value_m) / UNIT_M


def _mm3(values_m: Sequence[float]) -> tuple[float, float, float]:
    return (_mm(values_m[0]), _mm(values_m[1]), _mm(values_m[2]))


def _ident(prefix: str, name: str, used: set[str]) -> str:
    """A CSXCAD property name that is unique and safe in generated source."""
    base = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(name))
    if not base:
        base = "unnamed"
    candidate = f"{prefix}_{base}"
    suffix = 1
    while candidate in used:
        suffix += 1
        candidate = f"{prefix}_{base}_{suffix}"
    used.add(candidate)
    return candidate


# ---------------------------------------------------------------------------
# Mesh (D1, D2, D5, D9)
# ---------------------------------------------------------------------------

def _face_plan(
    boundary_spec: dict, cpml_layers: int, periodic_axes: str
) -> tuple[dict[str, int], dict[str, str]]:
    """Per-face absorber pad (cells) and openEMS boundary string.

    Mirrors ``Grid.__init__``'s ``_face_pad``: a ``pec`` / ``pmc`` / ``periodic``
    face gets ``pad = 0`` even when the axis participates in CPML, and an
    absorbing face gets its per-face thickness (``lo_thickness`` /
    ``hi_thickness``) or the scalar ``cpml_layers``.  Note the comment in
    ``rfx/grid.py`` calls that per-face allocation "the Meep / OpenEMS / Tidy3D
    convention" — so the pad side of the translation is already aligned; only
    the *direction* of the absorber differs (D1).
    """
    pads: dict[str, int] = {}
    strings: dict[str, str] = {}
    for axis in _AXES:
        axis_payload = _require_mapping(
            _get(boundary_spec, axis, "boundary.spec"), f"boundary.spec.{axis}"
        )
        if axis_payload.get("conformal", False):
            raise _refuse(
                f"boundary.spec.{axis}.conformal",
                "conformal=True selects rfx's Dey-Mittra PEC face-shift, which "
                "changes how the boundary face rasterises; openEMS has no "
                "conformal PEC, so emitting a staircased face would compare a "
                "different structure",
            )
        for side in ("lo", "hi"):
            face = f"{axis}_{side}"
            token = _get(axis_payload, side, f"boundary.spec.{axis}")
            if not isinstance(token, str):
                raise _refuse(f"boundary.spec.{face}", f"expected a token string, got {token!r}")
            if token == "periodic":
                raise _refuse(
                    f"periodic boundary on face {face}",
                    "the openEMS v0.0.35 Python bindings expose no periodic "
                    "boundary through SetBoundaryCond — 'PBC' raises 'Unknown "
                    "boundary condition', and PEC/PMC each kill a field "
                    "component of a normally incident plane wave. Documented "
                    "workarounds (an oversized waveguide, or driving the solver "
                    "from hand-written XML) are not translations",
                )
            if token == "pec":
                pads[face] = 0
                strings[face] = "PEC"
            elif token == "pmc":
                pads[face] = 0
                strings[face] = "PMC"
            elif token in _ABSORBING:
                thickness = axis_payload.get(f"{side}_thickness")
                layers = int(cpml_layers if thickness is None else thickness)
                if layers <= 0:
                    raise _refuse(
                        f"absorbing face {face} with {layers} layers",
                        "an absorbing face with zero thickness has no openEMS "
                        "counterpart: 'PML_0' is not a boundary condition and "
                        "dropping to PEC would close an open domain",
                    )
                if axis in periodic_axes:
                    # Grid._face_pad zeroes the pad when the axis is not a CPML
                    # axis, and _build_grid strips periodic axes from
                    # cpml_axes.  A design in that state is already refused
                    # above by the periodic token, so reaching here means the
                    # legacy periodic_axes view disagrees with the spec.
                    raise _refuse(
                        f"absorbing face {face} on periodic axis {axis!r}",
                        "boundary.legacy.periodic_axes marks this axis periodic "
                        "while boundary.spec marks the face absorbing; rfx "
                        "strips CPML from periodic axes, so the two views "
                        "describe different absorbers",
                    )
                pads[face] = layers
                # D3: parameterised from the design, not the scripts' PML_8.
                # D4: MUR is never emitted.
                strings[face] = f"PML_{layers}"
            else:
                raise _refuse(f"boundary token {token!r} on face {face}", "unknown token")
    return pads, strings


def _axis_lines_mm(
    n_cells: int, pad_lo: int, pad_hi: int, dx_m: float
) -> tuple[float, ...]:
    """Mesh lines for one axis, in mm, spanning the pads as well as the domain.

    **D1 — the highest-severity divergence.**  rfx's grid is
    ``ceil(L/dx) + 1 + pad_lo + pad_hi`` nodes and user coordinates are
    ``(idx - pad_lo) * dx``, i.e. index 0 lies *outside* the user domain.
    openEMS's ``PML_<N>`` instead consumes ``N`` cells of whatever mesh it is
    given.  So the emitted mesh runs from ``-pad_lo * dx`` to
    ``(n_cells + pad_hi) * dx`` and the declared ``PML_<N>`` eats exactly the
    added margin, leaving ``[0, n_cells*dx]`` as the clear region — the same
    clear volume rfx simulates.  Meshing only ``[0, L]`` (what the repo's
    hand-written comparators do) buries ``pad * dx`` of the structure in the
    absorber at each absorbing face.
    """
    return tuple(
        _mm(index * dx_m) for index in range(-pad_lo, n_cells + pad_hi + 1)
    )


def _pin(
    lines_mm: tuple[float, ...], required_mm: float, dx_m: float
) -> tuple[tuple[float, ...], bool]:
    """Guarantee *required_mm* is exactly a mesh line (D5).

    openEMS drops an excitation whose edges fall between mesh lines — it logs
    "Unused primitive", ``uf_inc`` stays 0 and every S value comes back NaN.
    One committed fixture records a whole wasted solver run to that cause.
    Port coordinates are snapped the way ``Grid.position_to_index`` snaps them
    so this is normally a no-op assertion; the return flag says whether a line
    actually had to be inserted, which would mean the mesh is no longer the
    uniform mesh rfx used.
    """
    tol = _LINE_TOL_REL * _mm(dx_m)
    best = min(range(len(lines_mm)), key=lambda i: abs(lines_mm[i] - required_mm))
    if abs(lines_mm[best] - required_mm) <= tol:
        if lines_mm[best] != required_mm:
            patched = list(lines_mm)
            patched[best] = required_mm
            return tuple(patched), False
        return lines_mm, False
    merged = tuple(sorted(set(lines_mm) | {required_mm}))
    return merged, True


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def _plan_materials(document: dict) -> dict[str, _MaterialPlan]:
    registered = _require_mapping(_get(document, "materials", "document"), "materials")
    library = _require_mapping(
        _get(document, "material_library", "document"), "material_library"
    )
    overlap = set(registered) & set(library)
    if overlap:
        raise _refuse(
            f"materials {sorted(overlap)}",
            "the same name appears in both 'materials' and 'material_library'; "
            "the document should record a registered material in exactly one "
            "of them, so which value applies is ambiguous",
        )

    used: set[str] = set()
    plans: dict[str, _MaterialPlan] = {}
    for name in sorted({**registered, **library}):
        payload = _require_mapping(
            registered.get(name, library.get(name)), f"materials.{name}"
        )
        sigma = float(_get(payload, "sigma", f"materials.{name}"))
        eps_r = float(_get(payload, "eps_r", f"materials.{name}"))
        mu_r = float(_get(payload, "mu_r", f"materials.{name}"))
        chi3 = float(_get(payload, "chi3", f"materials.{name}"))
        debye = _get(payload, "debye_poles", f"materials.{name}")
        lorentz = _get(payload, "lorentz_poles", f"materials.{name}")

        # rfx's own assembly loop branches on this threshold and, for a PEC
        # material, ignores eps_r / mu_r / chi3 / poles entirely — so those
        # fields are checked only on the dielectric branch.
        is_metal = sigma >= PEC_SIGMA_THRESHOLD
        if not is_metal:
            if debye:
                raise _refuse(
                    f"material {name!r} with Debye poles",
                    "openEMS has no Debye/Lorentz dispersion primitive reachable "
                    "through the path this emitter targets, and no script in "
                    "this repository has ever exercised one; flattening the "
                    "poles to a single eps_r would silently change the material",
                )
            if lorentz:
                raise _refuse(
                    f"material {name!r} with Lorentz poles",
                    "same as Debye: no reachable openEMS counterpart, and "
                    "flattening to a single eps_r would change the material",
                )
            if chi3 != 0.0:
                raise _refuse(
                    f"material {name!r} with chi3={chi3!r}",
                    "the Kerr nonlinearity has no openEMS counterpart on this "
                    "path; dropping it would turn a nonlinear design into a "
                    "linear one under the same name",
                )
            if mu_r != 1.0:
                raise _refuse(
                    f"material {name!r} with mu_r={mu_r!r}",
                    "openEMS exposes SetMaterialProperty(mue=...), but no script "
                    "in this repository has ever exercised it, so the mapping is "
                    "unverified. An unproven primitive translation is exactly "
                    "the comparator-divergence failure class this project treats "
                    "as dominant — prove it with a reproduce-gated script first",
                )
        plans[name] = _MaterialPlan(
            name=name,
            ident=_ident("metal" if is_metal else "mat", name, used),
            is_metal=is_metal,
            epsilon=eps_r,
            kappa=sigma,
        )
    return plans


# ---------------------------------------------------------------------------
# Geometry (D6, D16)
# ---------------------------------------------------------------------------

def _plan_geometry(
    document: dict, materials: dict[str, _MaterialPlan]
) -> tuple[tuple[_GeometryPlan, ...], int, int]:
    """Translate the paint list, synthesising openEMS ``priority`` (D6).

    rfx geometry is an ordered last-write-wins paint list
    (``rfx/geometry/csg.py``: "Applied in order; later shapes overwrite earlier
    ones") — decision D4a.  openEMS resolves overlap by ``priority``, higher
    wins, and there is no rfx field to copy.

    The synthesis is not just "priority = paint index".  rfx's assembly loop
    (``rfx/api/_compile.py``) treats the two material classes differently:
    dielectrics are painted in order (``eps_r = where(mask, ...)``), while every
    PEC material accumulates into a **union mask** that is applied on top
    regardless of position in the paint list.  So in rfx a dielectric painted
    after a PEC box does *not* overwrite the PEC.  The faithful projection is
    therefore two bands — all metals above all dielectrics, paint order within
    each band — which happens to reproduce the de-facto ladder the repo's coax
    script arrived at by hand (dielectric low, metal high) but now with a
    derivation instead of a precedent.

    Ports sit between the bands: a dielectric must not bury a port (rfx folds
    the port conductance in on top of the material), while PEC must win over a
    port (rfx marks port cells inside PEC as *dead* and applies no port sigma
    there — issue #318).  Returns the two port/metal band bases.
    """
    entries = _require_list(_get(document, "geometry", "document"), "geometry")
    n = len(entries)
    port_priority = 1 + n
    metal_base = 2 + n

    plans: list[_GeometryPlan] = []
    for index, entry in enumerate(entries):
        payload = _require_mapping(entry, f"geometry[{index}]")
        material_name = str(_get(payload, "material_name", f"geometry[{index}]"))
        shape = _require_mapping(
            _get(payload, "shape", f"geometry[{index}]"), f"geometry[{index}].shape"
        )
        kind = str(_get(shape, "kind", f"geometry[{index}].shape"))
        params = _require_mapping(
            _get(shape, "params", f"geometry[{index}].shape"),
            f"geometry[{index}].shape.params",
        )
        if kind not in _SUPPORTED_SHAPE_KINDS:
            raise _refuse(
                f"geometry[{index}] shape kind {kind!r}",
                "openEMS has AddSphere / AddPolygon / AddLinPoly / "
                "AddCylindricalShell / AddCurve, but not one of them is called "
                "by any script in this repository, so no translation of this "
                "primitive has ever been checked against a known-good result. "
                f"Supported here: {', '.join(_SUPPORTED_SHAPE_KINDS)}",
            )
        if material_name not in materials:
            raise _refuse(
                f"geometry[{index}] material {material_name!r}",
                "the material is neither in 'materials' nor in "
                "'material_library'; the document is internally inconsistent",
            )
        material = materials[material_name]
        priority = (metal_base + index) if material.is_metal else (1 + index)

        if kind == "box":
            # D16: thickness verbatim from the rfx Box. A zero-thickness rfx
            # Box stays a zero-thickness openEMS sheet; the emitter does not
            # promote it to a one-cell slab or vice versa.
            start = _mm3(_get(params, "corner_lo", f"geometry[{index}].shape.params"))
            stop = _mm3(_get(params, "corner_hi", f"geometry[{index}].shape.params"))
            plans.append(
                _GeometryPlan(
                    index=index,
                    kind="box",
                    material_ident=material.ident,
                    is_metal=material.is_metal,
                    priority=priority,
                    start_mm=start,
                    stop_mm=stop,
                )
            )
            continue

        # Cylinder: rfx records the *centre* plus a height, openEMS wants the
        # two axis endpoints and a separate radius.
        centre = list(_get(params, "center", f"geometry[{index}].shape.params"))
        radius = float(_get(params, "radius", f"geometry[{index}].shape.params"))
        height = float(_get(params, "height", f"geometry[{index}].shape.params"))
        axis = str(_get(params, "axis", f"geometry[{index}].shape.params"))
        if axis not in _AXES:
            raise _refuse(
                f"geometry[{index}] cylinder axis {axis!r}",
                "openEMS AddCylinder takes free axis endpoints, but an rfx "
                "cylinder axis outside 'xyz' is not a state this emitter knows "
                "how to read",
            )
        ax = _AXES.index(axis)
        lo = list(centre)
        hi = list(centre)
        lo[ax] = centre[ax] - height / 2.0
        hi[ax] = centre[ax] + height / 2.0
        plans.append(
            _GeometryPlan(
                index=index,
                kind="cylinder",
                material_ident=material.ident,
                is_metal=material.is_metal,
                priority=priority,
                start_mm=_mm3(lo),
                stop_mm=_mm3(hi),
                radius_mm=_mm(radius),
            )
        )
    return tuple(plans), port_priority, metal_base


# ---------------------------------------------------------------------------
# Excitations
# ---------------------------------------------------------------------------

def _gauss_from_waveform(payload: Any, what: str) -> tuple[float, float, list[str]]:
    """``GaussianPulse`` → ``SetGaussExcite(f0, fc)`` with ``fc = bandwidth*f0``."""
    mapping = _require_mapping(payload, what)
    kind = str(_get(mapping, "kind", what))
    if kind != "gaussian_pulse":
        raise _refuse(
            f"{what} waveform kind {kind!r}",
            "openEMS's SetGaussExcite is the only excitation shape any script "
            "in this repository has driven a port with. SetSinusExcite exists "
            "for a CW source but is unexercised here, and a modulated Gaussian "
            "has no direct counterpart — approximating either by a Gaussian "
            "would change the spectrum the S-parameters are read from",
        )
    params = _require_mapping(_get(mapping, "params", what), f"{what}.params")
    f0 = float(_get(params, "f0", f"{what}.params"))
    bandwidth = float(_get(params, "bandwidth", f"{what}.params"))
    amplitude = float(_get(params, "amplitude", f"{what}.params"))
    cutoff = float(_get(params, "cutoff", f"{what}.params"))
    if not (f0 > 0.0 and bandwidth > 0.0):
        raise _refuse(
            f"{what} waveform",
            f"SetGaussExcite needs f0 > 0 and a positive fractional bandwidth; "
            f"got f0={f0!r}, bandwidth={bandwidth!r}",
        )
    notes: list[str] = []
    if amplitude != 1.0:
        notes.append(
            f"{what}: GaussianPulse amplitude={amplitude!r} is dropped — "
            "SetGaussExcite is amplitude-normalised. S-parameters are ratios so "
            "this does not affect them, but any absolute field or power number "
            "read out of this run is on openEMS's normalisation, not rfx's."
        )
    if cutoff != 3.0:
        notes.append(
            f"{what}: GaussianPulse cutoff={cutoff!r} (Gaussian truncation in "
            "sigma) has no SetGaussExcite counterpart; openEMS truncates on its "
            "own rule, so the two pulses differ in their far tails."
        )
    return f0, bandwidth * f0, notes


def _plan_ports(
    document: dict,
    *,
    dx_m: float,
    port_priority: int,
    msl_port_w_cells: int,
) -> tuple[tuple[_PortPlan, ...], list[str], list[tuple[str, float]]]:
    """Lumped / wire / MSL ports; everything else in ``excitations`` refuses.

    Returns the port plans, the approximation notes they generated, and the
    ``(axis, coordinate_m)`` pairs that must land on mesh lines (D5).
    """
    excitations = _require_mapping(
        _get(document, "excitations", "document"), "excitations"
    )

    for key, construct, reason in (
        (
            "coaxial_ports",
            "add_coaxial_port(...)",
            "openEMS v0.0.35 has no AddCoaxialPort — verified absent on the "
            "local install — and the documented radial-AddLumpedPort workaround "
            "failed three separate runs (mesh misalignment, MUR-on-PTFE "
            "instability, then uf_inc ~ 6e-14 with no usable coupling). There is "
            "no translation to emit, only a known-bad one",
        ),
        (
            "coaxial_matched_loads",
            "add_coaxial_matched_load(...)",
            "the axial offset is recorded in *cells*, not metres, so it has no "
            "meaning on another solver's mesh — it would move physically if dx "
            "changed. It is listed in the document's own non_portable annotation",
        ),
        (
            "coaxial_open_terminations",
            "add_coaxial_open_termination(...)",
            "pin_retract_cells is a cell-relative offset (non_portable), so the "
            "termination plane cannot be placed from this document",
        ),
        (
            "coaxial_pec_end_caps",
            "add_coaxial_pec_end_cap(...)",
            "axial_offset_cells is a cell-relative offset (non_portable), so the "
            "cap plane cannot be placed from this document",
        ),
        (
            "floquet_ports",
            "add_floquet_port(...)",
            "a Floquet port needs periodic boundaries, and the openEMS v0.0.35 "
            "Python bindings expose none — 'PBC' raises 'Unknown boundary "
            "condition'",
        ),
        (
            "waveguide_ports",
            "add_waveguide_port(...)",
            "AddRectWaveGuidePort exists in the local bindings but is called by "
            "zero scripts in this repository, so its port-plane and mode-"
            "normalisation conventions have never been checked against rfx's; "
            "the rfx side additionally carries probe_offset / ref_offset as cell "
            "counts (non_portable). This is the largest known gap — it needs a "
            "reproduce-gated openEMS waveguide script before it can be emitted",
        ),
        (
            "lumped_rlc",
            "add_lumped_rlc(...)",
            "openEMS AddLumpedElement is called by no script in this repository, "
            "and rfx's R/L/C is an ADE sub-cell model whose series/parallel "
            "topology mapping is unverified",
        ),
    ):
        if _require_list(_get(excitations, key, "excitations"), f"excitations.{key}"):
            raise _refuse(construct, reason)

    if _get(excitations, "tfsf", "excitations") is not None:
        raise _refuse(
            "add_tfsf_source(...)",
            "the only openEMS plane-wave precedent in this repository is a soft-E "
            "one-cell slab in a script that is DEFERRED and not wired up, with no "
            "absorber story; the design note calls a plane-wave projection 'an "
            "approximation ... not a translation'",
        )
    if _require_list(
        _get(excitations, "soft_sources", "excitations"), "excitations.soft_sources"
    ):
        raise _refuse(
            "add_source(...) (soft, impedance-0 source)",
            "a soft field source maps to AddExcitation, which no script in this "
            "repository uses for a port-driven S-parameter run; it has no "
            "reference impedance, so no S-parameter can be read from it. Use "
            "add_port() for a projected setup",
        )

    lumped = _require_list(
        _get(excitations, "lumped_ports", "excitations"), "excitations.lumped_ports"
    )
    msl = _require_list(
        _get(excitations, "msl_ports", "excitations"), "excitations.msl_ports"
    )
    if not lumped and not msl:
        raise _refuse(
            "a design with no lumped, wire or MSL port",
            "this emitter projects port-driven S-parameter runs; with no such "
            "port there is nothing to excite and nothing to extract",
        )
    if lumped and msl:
        raise _refuse(
            "a mixed lumped/wire + MSL port set",
            "rfx's own S-parameter driver refuses mixed port families (the "
            "wave-decomposition conventions differ), and openEMS would compose "
            "two different port models into one S-matrix",
        )

    notes: list[str] = []
    required_lines: list[tuple[str, float]] = []
    plans: list[_PortPlan] = []
    snap_shifts: list[str] = []

    def snap(value_m: float, label: str) -> float:
        cells = int(round(value_m / dx_m))
        snapped = cells * dx_m
        # Grid.position_to_index uses round(pos/dx); reporting the shift makes
        # the rasterisation visible rather than silent.
        if abs(snapped - value_m) > _LINE_TOL_REL * dx_m:
            snap_shifts.append(
                f"{label}: {value_m!r} m → {snapped!r} m "
                f"({(snapped - value_m) / dx_m:+.3f} cells)"
            )
        return snapped

    for index, entry in enumerate(lumped):
        payload = _require_mapping(entry, f"excitations.lumped_ports[{index}]")
        what = f"excitations.lumped_ports[{index}]"
        component = str(_get(payload, "component", what))
        if component not in _COMPONENT_AXIS:
            raise _refuse(
                f"{what} component {component!r}",
                "openEMS AddLumpedPort takes an excitation direction of x, y or "
                "z; an rfx port component outside ex/ey/ez has no counterpart",
            )
        axis_index = _COMPONENT_AXIS[component]
        position = list(_get(payload, "position", what))
        extent = _get(payload, "extent", what)
        impedance = float(_get(payload, "impedance", what))
        excite = bool(_get(payload, "excite", what))
        reference_plane_cells = _get(payload, "reference_plane_cells", what)
        if reference_plane_cells is not None:
            raise _refuse(
                f"{what} reference_plane_cells={reference_plane_cells!r}",
                "the de-embedding plane is recorded as a cell count "
                "(non_portable), and openEMS's lumped port measures at its own "
                "plane with no equivalent knob; emitting the port without the "
                "plane would compare two different measurement planes",
            )

        start = [snap(float(v), f"{what}.position[{i}]") for i, v in enumerate(position)]
        stop = list(start)
        if extent is None:
            # D7: rfx's single-cell lumped port; openEMS spans one cell and
            # averages over it. The repo's own wire envelope measures the size
            # of this convention gap at ~0.20 in |S|.
            stop[axis_index] = start[axis_index] + dx_m
            family = "lumped"
        else:
            end_m = float(position[axis_index]) + float(extent)
            stop[axis_index] = snap(end_m, f"{what}.position+extent")
            family = "wire"
            if stop[axis_index] == start[axis_index]:
                raise _refuse(
                    f"{what} extent={extent!r}",
                    "the wire port rasterises to zero cells at this dx, and "
                    "openEMS asserts start != stop in the excitation direction",
                )
        f0 = fc = None
        if excite:
            waveform = _get(payload, "waveform", what)
            if waveform is None:
                raise _refuse(
                    f"{what} with excite=True and no waveform",
                    "rfx defaults an absent waveform inside run(), which is "
                    "run-time state the design document deliberately does not "
                    "carry; there is no pulse to project. Pass an explicit "
                    "waveform= to the port",
                )
            f0, fc, wf_notes = _gauss_from_waveform(waveform, what)
            notes.extend(wf_notes)

        number = len(plans) + 1
        for axis_name, coordinate in zip(_AXES, start):
            required_lines.append((axis_name, coordinate))
        for axis_name, coordinate in zip(_AXES, stop):
            required_lines.append((axis_name, coordinate))
        plans.append(
            _PortPlan(
                number=number,
                family=family,
                label=f"{family}_port_{number}",
                impedance=impedance,
                start_mm=_mm3(start),
                stop_mm=_mm3(stop),
                excite=excite,
                priority=port_priority,
                exc_dir=axis_index,
                f0_hz=f0,
                fc_hz=fc,
            )
        )

    for index, entry in enumerate(msl):
        payload = _require_mapping(entry, f"excitations.msl_ports[{index}]")
        what = f"excitations.msl_ports[{index}]"
        name = str(_get(payload, "name", what))
        position = list(_get(payload, "position", what))
        width = float(_get(payload, "width", what))
        height = float(_get(payload, "height", what))
        direction = str(_get(payload, "direction", what))
        impedance = float(_get(payload, "impedance", what))
        excite = bool(_get(payload, "excite", what))
        mode = str(_get(payload, "mode", what))
        if direction not in ("+x", "-x"):
            raise _refuse(
                f"{what} direction {direction!r}",
                "rfx's MSL port is x-propagating only, and this emitter does not "
                "guess a rotation of the MSLPort axes",
            )
        if not (width > 0.0 and height > 0.0):
            raise _refuse(
                f"{what} width={width!r}, height={height!r}",
                "openEMS MSLPort asserts start != stop in every component, so a "
                "degenerate cross-section cannot be built",
            )
        if mode != "laplace":
            notes.append(
                f"{what}: rfx mode={mode!r} selects rfx's own transverse-field "
                "seed for the port; openEMS MSLPort launches its own line mode "
                "and has no equivalent knob, so the launched profiles differ."
            )
        # D7: MSLPort occupies a SPAN along the propagation axis; port_w_cells
        # (6, inherited from upstream MSL_NotchFilter.py) has no rfx
        # counterpart at all. rfx's add_msl_port documents `direction` as the
        # direction the launched wave propagates, so the span extends that way
        # — note this is the *opposite* reading from add_port()'s `direction`,
        # which names the outward normal. Nothing in this repository pins the
        # MSL case against a measurement, which is why it is itemised.
        sign = 1.0 if direction == "+x" else -1.0
        span_m = msl_port_w_cells * dx_m
        x_feed = snap(float(position[0]), f"{what}.position[0]")
        y_centre = float(position[1])
        z_lo = snap(float(position[2]), f"{what}.position[2]")
        y_lo = snap(y_centre - width / 2.0, f"{what}.trace_y_lo")
        y_hi = snap(y_centre + width / 2.0, f"{what}.trace_y_hi")
        z_hi = snap(z_lo + height, f"{what}.trace_plane")
        x_end = x_feed + sign * span_m
        if y_hi == y_lo or z_hi == z_lo:
            raise _refuse(
                f"{what} cross-section",
                "the trace width or substrate height rasterises to zero cells at "
                "this dx, so MSLPort's start != stop assertion cannot hold",
            )

        f0 = fc = None
        if excite:
            waveform = _get(payload, "waveform", what)
            if waveform is None:
                raise _refuse(
                    f"{what} with excite=True and no waveform",
                    "rfx defaults the MSL pulse inside run() from freq_max, which "
                    "is run-time state the design document does not carry. Pass "
                    "an explicit waveform= to the port",
                )
            f0, fc, wf_notes = _gauss_from_waveform(waveform, what)
            notes.extend(wf_notes)

        number = len(plans) + 1
        for coordinate in (x_feed, x_end):
            required_lines.append(("x", coordinate))
        for coordinate in (y_lo, y_hi):
            required_lines.append(("y", coordinate))
        for coordinate in (z_lo, z_hi):
            required_lines.append(("z", coordinate))
        plans.append(
            _PortPlan(
                number=number,
                family="msl",
                label=f"msl_port_{number}_{name}",
                impedance=impedance,
                # start at the trace plane, stop at ground: exc_dir=2 (Ez) then
                # points from trace down to ground, matching the repo's MSL
                # referee exactly.
                start_mm=(_mm(x_feed), _mm(y_lo), _mm(z_hi)),
                stop_mm=(_mm(x_end), _mm(y_hi), _mm(z_lo)),
                excite=excite,
                priority=port_priority,
                exc_dir=2,
                prop_dir=0,
                f0_hz=f0,
                fc_hz=fc,
            )
        )

    if any(p.family == "msl" for p in plans):
        notes.append(
            f"[D7] MSL ports: msl_port_w_cells={msl_port_w_cells} sets the "
            "MSLPort span along the propagation axis. That span has NO rfx "
            "counterpart — rfx extracts through an N-probe spatial fit "
            "downstream of a feed plane, while MSLPort launches and de-embeds "
            "inside its own span — so the two measure at different planes and "
            "no committed comparison in this repository pins the choice. The "
            "span direction follows add_msl_port's documented meaning of "
            "direction= (the direction the launched wave propagates), which is "
            "the OPPOSITE reading from add_port's direction= (the outward "
            "normal); if that reading is wrong the two ports swap ends. "
            "openEMS may additionally log 'Unused primitive ... msl_feed_N' "
            "when the design's own trace already covers the port span — benign, "
            "and distinct from the port-dropped failure mode."
        )

    if not any(p.excite for p in plans):
        raise _refuse(
            "a design whose every port has excite=False",
            "every port is a passive matched load, so no run can be driven and "
            "no S-parameter column exists",
        )

    if snap_shifts:
        notes.append(
            "port coordinates were snapped to the mesh the way "
            "Grid.position_to_index snaps them (round(pos/dx)), so the emitted "
            "port sits where rfx rasterises it, not at the metres the user "
            "typed: " + "; ".join(snap_shifts)
        )
    return tuple(plans), notes, required_lines


# ---------------------------------------------------------------------------
# Run control (D12) and frequencies
# ---------------------------------------------------------------------------

def _courant_dt(dx_m: float) -> float:
    """rfx's ``Grid.courant_dt`` for 3D — a pure function of ``dx``."""
    return dx_m / (299792458.0 * math.sqrt(3.0)) * 0.99


def _default_freqs(ports: Sequence[_PortPlan]) -> tuple[float, ...]:
    driven = [p for p in ports if p.excite and p.f0_hz and p.fc_hz]
    f0 = min(p.f0_hz for p in driven)  # type: ignore[misc]
    fc = max(p.fc_hz for p in driven)  # type: ignore[misc]
    lo = max(f0 - fc, 0.02 * f0)
    hi = f0 + fc
    n = 21
    return tuple(lo + (hi - lo) * i / (n - 1) for i in range(n))


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def plan_openems_projection(
    document: dict,
    *,
    freqs_hz: Sequence[float] | None = None,
    n_timesteps: int | None = None,
    num_periods: float | None = None,
    end_criteria: float | None = None,
    msl_port_w_cells: int = 6,
) -> OpenEMSPlan:
    """Reviewable projection plan for *document* — the emitter's real work.

    Separated from text generation so the arithmetic that matters (above all
    the D1 absorber span) is assertable directly.
    """
    document = _require_mapping(document, "document")
    schema = str(_get(document, "schema", "document"))
    if schema != "rfx-design-ir/v1":
        raise _refuse(
            f"design document schema {schema!r}",
            "this emitter reads rfx-design-ir/v1 only; a different schema may "
            "spell fields differently and silently mis-project them",
        )
    rfx_version = str(_get(document, "rfx_version", "document"))

    if _get(document, "refinement", "document") is not None:
        raise _refuse(
            "add_refinement(...) (SBP-SAT subgrid)",
            "the subgrid path is an rfx research prototype, falsified in 3D "
            "(PR #90) and outside the public support scope. The design note is "
            "explicit: round-trip it for rfx, never project it outward, because "
            "its presence in a foreign setup would read as a validated capability",
        )

    solver = _require_mapping(_get(document, "solver", "document"), "solver")
    if str(_get(solver, "solver", "solver")) != "yee":
        raise _refuse(
            f"solver={_get(solver, 'solver', 'solver')!r}",
            "rfx's ADI solver has no openEMS counterpart (openEMS is explicit "
            "Yee only); emitting the same structure under a different time "
            "integrator would compare two different numerical schemes",
        )
    if int(_get(solver, "stencil_order", "solver")) != 2:
        raise _refuse(
            f"stencil_order={_get(solver, 'stencil_order', 'solver')!r}",
            "the (2,4) spatial stencil is rfx-specific; openEMS is second order, "
            "so the emitted run would have a different dispersion relation",
        )
    precision = str(_get(solver, "precision", "solver"))
    if precision != "float32":
        raise _refuse(
            f"precision={precision!r}",
            "openEMS's engine precision is fixed and not selectable from the "
            "Python bindings, so an rfx design that asks for a different working "
            "precision cannot be reproduced",
        )

    domain = _require_mapping(_get(document, "domain", "document"), "domain")
    mode = str(_get(domain, "mode", "domain"))
    if mode != "3d":
        raise _refuse(
            f"mode={mode!r}",
            "rfx's 2-D mode is a single-cell-thick z slab with periodic z; "
            "openEMS has no 2-D mode, and emitting a one-cell 3-D slab would be "
            "a different problem",
        )
    extent = list(_get(domain, "extent", "domain"))

    mesh = _require_mapping(_get(document, "mesh", "document"), "mesh")
    dx = _get(mesh, "dx", "mesh")
    if dx is None:
        raise _refuse(
            "a design with dx=None (auto-mesh)",
            "the auto-mesh resolves inside run() and writes _dx back, so a "
            "document exported before the run carries no mesh at all. Export "
            "after a run, or pass dx= explicitly",
        )
    dx_m = float(dx)
    if dx_m <= 0.0:
        raise _refuse(f"dx={dx_m!r}", "the mesh step must be positive")
    for profile in ("dx_profile", "dy_profile", "dz_profile"):
        if _get(mesh, profile, "mesh") is not None:
            raise _refuse(
                f"non-uniform mesh profile {profile}",
                "openEMS takes explicit mesh lines, so a graded profile is "
                "expressible in principle — but rfx's absorber padding on a "
                "non-uniform axis, and the transition cells smooth_grading "
                "inserts, decide where the material stack actually rasterises "
                "(the #325 class of defect). Emitting a cumulative-sum guess "
                "would compare a differently-graded structure. v1 refuses; the "
                "fix is a pinned NonUniformGrid coordinate comparison, not a "
                "wider emitter",
            )

    boundary = _require_mapping(_get(document, "boundary", "document"), "boundary")
    spec = _require_mapping(_get(boundary, "spec", "boundary"), "boundary.spec")
    legacy = _require_mapping(_get(boundary, "legacy", "boundary"), "boundary.legacy")
    cpml_layers = int(_get(legacy, "cpml_layers", "boundary.legacy"))
    periodic_axes = str(_get(legacy, "periodic_axes", "boundary.legacy"))
    pads, strings = _face_plan(spec, cpml_layers, periodic_axes)

    if _require_list(
        _get(document, "thin_conductors", "document"), "thin_conductors"
    ):
        raise _refuse(
            "add_thin_conductor(...)",
            "openEMS's conducting-sheet property is called by no script in this "
            "repository, and rfx's sub-cell sheet model would otherwise have to "
            "be flattened to either a zero-thickness sheet or a one-cell slab — "
            "two different physical models (D16), neither of them the rfx one",
        )

    observables = _require_mapping(
        _get(document, "observables", "document"), "observables"
    )
    for key, construct, reason in (
        (
            "probes",
            "add_probe(...)",
            "openEMS's AddProbe is called by no script in this repository, so "
            "the field-component and weighting conventions of a translated point "
            "probe are unchecked. Dropping the probe instead would silently "
            "produce a script that does not measure what was asked for",
        ),
        (
            "dft_planes",
            "add_dft_plane_probe(...)",
            "the openEMS counterpart is AddDump plus an HDF5 read-back whose "
            "DumpType/FileType and post-processing are the emitter's choice, not "
            "the design's; no committed comparison pins that choice",
        ),
        (
            "flux_monitors",
            "add_flux_monitor(...)",
            "no script in this repository extracts a flux spectrum from openEMS, "
            "so the surface-integration and normalisation conventions are "
            "unverified",
        ),
    ):
        if _require_list(_get(observables, key, "observables"), f"observables.{key}"):
            raise _refuse(construct, reason)
    if _get(observables, "ntff", "observables") is not None:
        raise _refuse(
            "add_ntff_box(...)",
            "openEMS's CreateNF2FFBox is auto-sized and takes no corners, so the "
            "rfx box corners would be discarded — an approximation, not a "
            "translation. The far-field would then be computed on a different "
            "surface than the design specifies",
        )

    materials = _plan_materials(document)
    geometry, port_priority, _metal_base = _plan_geometry(document, materials)
    ports, port_notes, required_lines = _plan_ports(
        document,
        dx_m=dx_m,
        port_priority=port_priority,
        msl_port_w_cells=msl_port_w_cells,
    )

    # --- mesh lines (D1, D2) ------------------------------------------------
    n_cells = tuple(int(math.ceil(float(extent[i]) / dx_m)) for i in range(3))
    pad_lo = tuple(pads[f"{ax}_lo"] for ax in _AXES)
    pad_hi = tuple(pads[f"{ax}_hi"] for ax in _AXES)
    lines: dict[str, tuple[float, ...]] = {
        ax: _axis_lines_mm(n_cells[i], pad_lo[i], pad_hi[i], dx_m)
        for i, ax in enumerate(_AXES)
    }

    inserted: list[str] = []
    for axis_name, coordinate_m in required_lines:
        lines[axis_name], did_insert = _pin(
            lines[axis_name], _mm(coordinate_m), dx_m
        )
        if did_insert:
            inserted.append(f"{axis_name}={_mm(coordinate_m)!r} mm")

    notes: list[str] = []
    notes.append(
        "[D1] absorber span: the emitted mesh is the user domain PLUS the rfx "
        "absorber pads, because rfx adds CPML cells outside the domain while "
        "openEMS PML_<N> consumes cells inside the mesh it is given. Per axis "
        "(pad_lo, n_cells, pad_hi) = "
        + ", ".join(
            f"{ax}:({pad_lo[i]}, {n_cells[i]}, {pad_hi[i]})"
            for i, ax in enumerate(_AXES)
        )
        + f"; mesh-line counts {tuple(len(lines[ax]) for ax in _AXES)} match the "
        "rfx Grid.shape. rfx CPML and openEMS PML are different absorber "
        "formulations, so the same layer count does not mean the same absorber."
    )
    notes.append(
        f"[D2] cell counts use rfx's ceil(domain/dx), not the round() the "
        f"repository's hand-written openEMS scripts use, so the emitted clear "
        f"extent is {tuple(n_cells[i] * dx_m for i in range(3))} m and may exceed "
        f"the requested domain {tuple(float(v) for v in extent)} m."
    )
    notes.append(
        "[D3/D4] absorbing faces are emitted as PML_<cpml_layers> taken from the "
        "design (not the scripts' hard-coded PML_8), and MUR is never emitted: "
        "MUR on a dielectric is unstable and cost one patch run an 8 % resonance "
        "error."
    )
    notes.append(
        "[D6] openEMS priority is synthesised from the rfx paint order, in two "
        "bands: dielectrics 1..N in paint order, ports at "
        f"{port_priority}, PEC metals above that. rfx applies PEC as a union "
        "mask on top of the painted dielectrics regardless of paint position, "
        "and marks port cells inside PEC as dead, so metal>port>dielectric is "
        "the faithful ordering — not a flat priority=index."
    )
    notes.append(
        "[D7/D8] port models and reference planes differ. rfx's lumped port is "
        "single-cell and its wire port is a point-feed column, while openEMS "
        "averages over the port span; the repository's own wire-port envelope "
        "measures that convention gap at about 0.20 in |S|. Compare magnitudes "
        "only: phase, group delay and Z0 need a de-embedding contract that no "
        "script in this repository has."
    )
    notes.append(
        "[D9] all geometry coordinates in this script are millimetres "
        f"(SetDeltaUnit({UNIT_M!r}))."
    )
    notes.append(
        "[D5] port edge coordinates are snapped with round(pos/dx), the same "
        "rule Grid.position_to_index uses, so every port edge coincides with a "
        "mesh line — an off-grid port is silently dropped by openEMS ('Unused "
        "primitive', uf_inc=0, all-NaN S). Geometry faces are deliberately NOT "
        "pinned: rfx rasterises them against the uniform grid, so inserting "
        "lines at conductor edges would mesh a different problem than rfx solved."
    )
    if inserted:
        notes.append(
            "[D5] WARNING — mesh lines had to be INSERTED at "
            + ", ".join(inserted)
            + ". The mesh is therefore no longer the uniform mesh rfx used, and "
            "this run is not comparable to the rfx design cell-for-cell."
        )
    notes.extend(port_notes)

    # --- frequencies (not design state) ------------------------------------
    if freqs_hz is None:
        chosen = _default_freqs(ports)
        notes.append(
            "the S-parameter frequency list is NOT design state — in rfx it is "
            "an argument to run()/compute_*_s_matrix, which the design document "
            "deliberately excludes. The emitter chose "
            f"{len(chosen)} points from {chosen[0]:.6e} to {chosen[-1]:.6e} Hz "
            "spanning the excitation band. Pass freqs_hz= to pin it to the "
            "frequencies an rfx result was actually taken at."
        )
    else:
        chosen = tuple(float(f) for f in freqs_hz)
        if not chosen:
            raise _refuse("an empty freqs_hz list", "there is nothing to extract")
        if any(f <= 0.0 or not math.isfinite(f) for f in chosen):
            raise _refuse(
                "a non-positive or non-finite frequency in freqs_hz",
                "openEMS's port DFT needs finite positive frequencies",
            )

    # --- run control (D12) --------------------------------------------------
    absorbing = any(s.startswith("PML_") for s in strings.values())
    dt_rfx = _courant_dt(dx_m)
    freq_max = float(_get(domain, "freq_max", "domain"))
    if n_timesteps is not None and num_periods is not None:
        raise _refuse(
            "both n_timesteps= and num_periods=",
            "they are two spellings of the same run length; pass one",
        )
    if n_timesteps is not None:
        steps = int(n_timesteps)
        source = f"n_timesteps={steps} passed to the emitter"
    elif num_periods is not None:
        steps = int(math.ceil(float(num_periods) / freq_max / dt_rfx))
        source = (
            f"num_periods={num_periods!r} at freq_max={freq_max:.6e} Hz, "
            f"converted with rfx's Courant dt={dt_rfx:.6e} s"
        )
    elif absorbing:
        steps = 500_000
        source = (
            "the emitter default for an open domain: a 500000-step ceiling that "
            "EndCriteria is expected to cut short (the repository's MSL referee "
            "uses exactly this pair)"
        )
    else:
        periods = 60.0
        steps = int(math.ceil(periods / freq_max / dt_rfx))
        source = (
            f"the emitter default for a closed (non-absorbing) domain: "
            f"num_periods={periods} at freq_max={freq_max:.6e} Hz via rfx's "
            f"Courant dt={dt_rfx:.6e} s, because a closed lossless domain never "
            f"decays and EndCriteria would never fire"
        )
    if steps <= 0:
        raise _refuse(f"a run length of {steps} timesteps", "must be positive")

    if end_criteria is not None:
        criteria = float(end_criteria)
        criteria_source = f"end_criteria={criteria!r} passed to the emitter"
    elif absorbing:
        criteria = 1.0e-4
        criteria_source = (
            "the emitter default 1e-4 (-40 dB residual energy) for an open domain"
        )
    else:
        criteria = 0.0
        criteria_source = (
            "0 (disabled) because the domain has no absorbing face: energy is "
            "conserved, so any decay criterion is meaningless"
        )
    notes.append(
        "[D12] run control is NOT design state (n_steps / until_decay are "
        f"run() arguments). NrTS={steps} from {source}; EndCriteria={criteria!r} "
        f"from {criteria_source}. A step count does not transfer between "
        "solvers — openEMS computes its own Courant timestep from its own mesh, "
        "so an identical NrTS is a different physical duration. EndCriteria can "
        "also fire early on a high-Q resonator, cutting the signal before the "
        "DFT resolves the peak."
    )
    notes.append(
        "[D16] conductor thickness is taken verbatim from the rfx shape: a "
        "zero-thickness rfx Box stays a zero-thickness openEMS sheet and a "
        "one-cell Box stays a one-cell slab. The emitter does not convert "
        "between the two, because they are different physical models."
    )

    if msl_port_w_cells < 5 and any(p.family == "msl" for p in ports):
        raise _refuse(
            f"msl_port_w_cells={msl_port_w_cells}",
            "openEMS MSLPort asserts at least 5 mesh lines along the "
            "propagation direction of its span",
        )

    return OpenEMSPlan(
        rfx_version=rfx_version,
        schema=schema,
        dx_m=dx_m,
        n_cells=n_cells,  # type: ignore[arg-type]
        pad_lo=pad_lo,  # type: ignore[arg-type]
        pad_hi=pad_hi,  # type: ignore[arg-type]
        mesh_lines_mm=lines,
        boundary=(
            strings["x_lo"],
            strings["x_hi"],
            strings["y_lo"],
            strings["y_hi"],
            strings["z_lo"],
            strings["z_hi"],
        ),
        materials=tuple(materials[name] for name in sorted(materials)),
        geometry=geometry,
        ports=ports,
        driven_port_numbers=tuple(p.number for p in ports if p.excite),
        freqs_hz=chosen,
        n_timesteps=steps,
        end_criteria=criteria,
        msl_port_w_cells=msl_port_w_cells,
        approximations=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _floats(values: Sequence[float]) -> str:
    return ", ".join(repr(float(v)) for v in values)


def _wrap(text: str, prefix: str, width: int = 78) -> list[str]:
    """Wrap *text* into comment/docstring lines, each starting with *prefix*."""
    return textwrap.wrap(
        " ".join(text.split()),
        width=width,
        initial_indent=prefix,
        subsequent_indent=prefix,
    ) or [prefix.rstrip()]


def _header(plan: OpenEMSPlan) -> str:
    lines = [
        '"""GENERATED BY rfx — DO NOT EDIT BY HAND.',
        "",
        f"emitter          : {OPENEMS_EMITTER_VERSION}",
        f"source rfx       : {plan.rfx_version}",
        f"design IR schema : {plan.schema}",
        "target           : openEMS v0.0.35 Python bindings + CSXCAD",
        "profile          : external projection (design-note decision D3) —",
        "                   explicitly lossy, NOT the rfx round-trip document",
        "",
        "WHAT THIS SCRIPT DOES NOT PROVE",
    ]
    lines.extend(
        _wrap(
            "Structural equivalence of this generated setup to the rfx design is "
            "not evidence of physics agreement. If this script runs and returns "
            "passive S-parameters, that shows the projection is executable and "
            "self-consistent — nothing more. The absorber formulations differ, "
            "the port models differ, and the reference planes differ. A "
            "physics-agreement claim needs a matched-resolution comparison "
            "against a committed reference, quoted together with the rfx "
            "preflight output.",
            "  ",
        )
    )
    lines.extend(["", "APPROXIMATIONS APPLIED (itemised, per decision D3)"])
    for index, note in enumerate(plan.approximations, start=1):
        body = _wrap(note, "       ")
        first = body[0].lstrip()
        lines.append(f"  [{index:2d}] {first}")
        lines.extend(body[1:])
    lines.append('"""')
    return "\n".join(lines)


def _render_build(plan: OpenEMSPlan) -> list[str]:
    out: list[str] = []
    out.append("def _build(driven_number):")
    out.append('    """Assemble the structure with exactly one port driven."""')
    out.append("    f0, fc = _EXCITATION[driven_number]")
    out.append("    fdtd = openEMS(NrTS=NRTS, EndCriteria=END_CRITERIA)")
    out.append("    fdtd.SetGaussExcite(f0, fc)")
    out.append("    fdtd.SetBoundaryCond(list(BOUNDARY))")
    out.append("    csx = ContinuousStructure()")
    out.append("    fdtd.SetCSX(csx)")
    out.append("")
    out.append("    # Mesh first: openEMS MSLPort reads the grid lines in its")
    out.append("    # constructor, so properties and ports must come after this.")
    out.append("    mesh = csx.GetGrid()")
    out.append(f"    mesh.SetDeltaUnit({UNIT_M!r})")
    for axis in _AXES:
        out.append(f"    mesh.SetLines({axis!r}, {axis.upper()}_LINES_MM)")
    out.append("")
    if plan.materials:
        out.append("    # Materials. rfx treats sigma >= 1e6 S/m as a PEC mask")
        out.append("    # rather than a lossy dielectric, so those become AddMetal.")
        for material in plan.materials:
            if material.is_metal:
                out.append(
                    f"    {material.ident} = csx.AddMetal({material.ident!r})"
                    f"  # rfx {material.name!r}, sigma={material.kappa!r} S/m"
                )
            else:
                out.append(
                    f"    {material.ident} = csx.AddMaterial({material.ident!r}, "
                    f"epsilon={material.epsilon!r}, kappa={material.kappa!r})"
                    f"  # rfx {material.name!r}"
                )
        out.append("")
    if plan.geometry:
        out.append("    # Geometry in rfx paint order; priority synthesised per [D6].")
        for geo in plan.geometry:
            if geo.kind == "box":
                out.append(
                    f"    {geo.material_ident}.AddBox([{_floats(geo.start_mm)}], "
                    f"[{_floats(geo.stop_mm)}], priority={geo.priority})"
                    f"  # geometry[{geo.index}]"
                )
            else:
                out.append(
                    f"    {geo.material_ident}.AddCylinder([{_floats(geo.start_mm)}], "
                    f"[{_floats(geo.stop_mm)}], radius={geo.radius_mm!r}, "
                    f"priority={geo.priority})  # geometry[{geo.index}]"
                )
        out.append("")
    out.append("    ports = []")
    for port in plan.ports:
        out.append(f"    # {port.label}: rfx {port.family} port")
        excite_expr = f"(1.0 if driven_number == {port.number} else 0.0)"
        if port.family == "msl":
            out.append(
                "    # MSLPort builds its own launch-conductor sheet from this"
            )
            out.append(
                "    # property. When the design's own trace already covers the"
            )
            out.append(
                "    # port span with metal, that sheet is fully overridden and"
            )
            out.append(
                "    # openEMS logs 'Unused primitive ... msl_feed_N'. That is"
            )
            out.append(
                "    # benign (both are PEC) and is NOT the port-dropped failure"
            )
            out.append(
                "    # mode -- the uf_inc guard below is what discriminates."
            )
            out.append(
                f"    _metal_{port.number} = csx.AddMetal('msl_feed_{port.number}')"
            )
            out.append("    ports.append(MSLPort(")
            out.append(f"        csx, port_nr={port.number},")
            out.append(f"        metal_prop=_metal_{port.number},")
            out.append(f"        start=[{_floats(port.start_mm)}],")
            out.append(f"        stop=[{_floats(port.stop_mm)}],")
            out.append(f"        prop_dir={port.prop_dir}, exc_dir={port.exc_dir},")
            out.append(f"        excite={excite_expr},")
            out.append(f"        Feed_R={port.impedance!r},")
            out.append(f"        priority={port.priority},")
            out.append("    ))")
        else:
            out.append("    ports.append(fdtd.AddLumpedPort(")
            out.append(f"        {port.number}, {port.impedance!r},")
            out.append(f"        [{_floats(port.start_mm)}],")
            out.append(f"        [{_floats(port.stop_mm)}],")
            out.append(f"        {_AXES[port.exc_dir]!r},")
            out.append(f"        excite={excite_expr},")
            out.append(f"        priority={port.priority},")
            out.append("    ))")
    out.append("    return fdtd, csx, ports")
    return out


_RUNNER = '''
def _run_one(driven_number, sim_root, threads):
    sim_dir = os.path.abspath(os.path.join(sim_root, "drive_%d" % driven_number))
    if os.path.isdir(sim_dir):
        shutil.rmtree(sim_dir)
    os.makedirs(sim_dir, exist_ok=True)

    fdtd, csx, ports = _build(driven_number)
    csx.Write2XML(os.path.join(sim_dir, "geometry.xml"))

    # [D14] Run() asserts on a relative path (bare AssertionError from
    # openEMS.pyx) and leaves the process CWD inside the sim directory.
    original_cwd = os.getcwd()
    try:
        if threads:
            fdtd.Run(sim_dir, verbose=0, cleanup=True, numThreads=threads)
        else:
            fdtd.Run(sim_dir, verbose=0, cleanup=True)
    finally:
        try:
            os.chdir(original_cwd)
        except OSError:
            os.chdir("/tmp")

    # [D13] Never pass ref_impedance as a scalar float: that trips an upstream
    # bug in Port.CalcPort when Z_ref is array-valued.
    for port in ports:
        port.CalcPort(sim_dir, FREQS_HZ)

    driven = [p for p in ports if p.number == driven_number][0]
    incident = np.asarray(driven.uf_inc, dtype=complex)

    # [D5] Excitation guard, the hand-ported stand-in for rfx preflight (an
    # external script gets none). An off-grid port makes openEMS log "Unused
    # primitive", leaves uf_inc at exactly zero and turns every S value into
    # NaN. No absolute floor is used on purpose: a verified-good small run on
    # this install peaked at |uf_inc| ~ 3e-14, so the 1e-9 floor some
    # hand-written comparators use would false-fire here.
    peak = float(np.max(np.abs(incident))) if incident.size else 0.0
    if not np.isfinite(peak) or peak == 0.0:
        raise RuntimeError(
            "port %d injected no incident wave (peak |uf_inc| = %r). openEMS "
            "silently drops an excitation whose edges miss the mesh lines; "
            "check %s for 'Unused primitive'." % (driven_number, peak, sim_dir)
        )
    # Independent witness on the time domain. The voltage-probe filenames come
    # from the port object itself because they differ per family: a lumped port
    # writes "port_ut_<n>", an MSLPort writes "port_ut_<n>A/B/C".
    trace_peak = None
    for name in list(getattr(driven, "U_filenames", []) or []):
        trace = os.path.join(sim_dir, name)
        if not os.path.exists(trace):
            continue
        raw = np.loadtxt(trace, comments="%")
        if not raw.size:
            continue
        peak_here = float(np.max(np.abs(np.atleast_2d(raw)[:, 1])))
        trace_peak = peak_here if trace_peak is None else max(trace_peak, peak_here)
    if trace_peak == 0.0:
        raise RuntimeError(
            "port %d time trace is identically zero: the excitation never "
            "entered the grid." % driven_number
        )

    column = {}
    for port in ports:
        received = np.asarray(port.uf_ref, dtype=complex)
        column[port.number] = received / incident
    # A lumped port's Z_ref is the scalar feed resistance; MSLPort's is an
    # array over frequency. Normalise so the artifact shape does not depend on
    # the port family.
    z_ref = np.atleast_1d(np.asarray(driven.Z_ref, dtype=complex))
    return column, peak, trace_peak, z_ref


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sim-dir", default="openems_run",
                        help="scratch directory for the solver runs")
    parser.add_argument("--output", default="openems_sparams.json",
                        help="where to write the S-parameter JSON")
    parser.add_argument("--threads", type=int, default=0,
                        help="numThreads for FDTD.Run (0 = openEMS default)")
    args = parser.parse_args(argv)

    sim_root = os.path.abspath(args.sim_dir)
    columns = {}
    diagnostics = {}
    for driven in DRIVEN_PORTS:
        column, peak, trace_peak, z_ref = _run_one(driven, sim_root, args.threads)
        columns[driven] = column
        diagnostics[str(driven)] = {
            "incident_peak_abs": peak,
            "port_ut_peak_abs": trace_peak,
            "z_ref_real_ohm": [float(v.real) for v in z_ref],
            "z_ref_imag_ohm": [float(v.imag) for v in z_ref],
        }

    s_matrix = {}
    max_abs = 0.0
    for driven, column in columns.items():
        for received, values in column.items():
            key = "S%d%d" % (received, driven)
            s_matrix[key] = [[float(v.real), float(v.imag)] for v in values]
            finite = np.abs(values)[np.isfinite(np.abs(values))]
            if finite.size:
                max_abs = max(max_abs, float(np.max(finite)))

    payload = {
        "schema": "rfx-openems-emitted-sparams/v1",
        "emitter": EMITTER,
        "source_rfx_version": SOURCE_RFX_VERSION,
        "design_ir_schema": DESIGN_IR_SCHEMA,
        "approximations": APPROXIMATIONS,
        "grid": {
            "delta_unit_m": UNIT,
            "line_counts": [len(X_LINES_MM), len(Y_LINES_MM), len(Z_LINES_MM)],
            "boundary": list(BOUNDARY),
            "nrts": NRTS,
            "end_criteria": END_CRITERIA,
        },
        "freqs_hz": [float(f) for f in FREQS_HZ],
        "driven_ports": list(DRIVEN_PORTS),
        "s_matrix": s_matrix,
        "port_diagnostics": diagnostics,
        "passivity": {
            "max_abs_s": max_abs,
            "documented_envelope": 1.05,
            "within_envelope": bool(max_abs <= 1.05),
        },
    }

    # Passivity witness, not a gate: |S| above the documented 1.05 single-run
    # envelope on a passive structure is unphysical and must be reported with
    # the run, never quoted as physics. Above 2.0 it is a broken setup.
    if max_abs > 2.0:
        raise RuntimeError(
            "max |S| = %r on a passive structure: the setup is broken "
            "(check the excitation guard and the port geometry), not physics."
            % max_abs
        )
    if max_abs > 1.05:
        sys.stderr.write(
            "WARNING: max |S| = %r exceeds the documented 1.05 passivity "
            "envelope. Suspect the extraction/normalisation path first, then "
            "confirm with an energy witness before attributing anything to "
            "physics.\\n" % max_abs
        )

    output = os.path.abspath(args.output)
    with open(output, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.write("wrote %s (max |S| = %r)\\n" % (output, max_abs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def emit_openems_script(
    document: dict,
    *,
    freqs_hz: Sequence[float] | None = None,
    n_timesteps: int | None = None,
    num_periods: float | None = None,
    end_criteria: float | None = None,
    msl_port_w_cells: int = 6,
) -> str:
    """Project a ``rfx-design-ir/v1`` document onto a runnable openEMS script.

    Parameters
    ----------
    document
        A document from :func:`rfx.interop.design_to_dict`.
    freqs_hz
        Frequencies for the port DFT.  **Not design state** — in rfx this is an
        argument to ``run()`` / ``compute_*_s_matrix``, which the design
        document deliberately excludes.  When omitted the emitter spans the
        excitation band and says so in the generated header.
    n_timesteps, num_periods, end_criteria
        Run control, also not design state (D12).  Pass at most one of
        ``n_timesteps`` / ``num_periods``.  Both defaults are itemised in the
        generated header, along with the fact that a step count does not
        transfer between solvers whose timesteps differ.
    msl_port_w_cells
        Span of an ``MSLPort`` along the propagation axis, in cells.  This has
        **no rfx counterpart** (D7); 6 is the value inherited from upstream's
        ``MSL_NotchFilter.py`` and used by this repository's MSL referee.

    Returns
    -------
    str
        A self-contained Python script.  Generating it needs no solver and no
        licence; running it needs openEMS.

    Raises
    ------
    UnsupportedDesignFeature
        For every construct outside the fence documented in this module.  The
        message names the construct and why it is refused.
    """
    plan = plan_openems_projection(
        document,
        freqs_hz=freqs_hz,
        n_timesteps=n_timesteps,
        num_periods=num_periods,
        end_criteria=end_criteria,
        msl_port_w_cells=msl_port_w_cells,
    )

    body: list[str] = ["#!/usr/bin/env python3", _header(plan), ""]
    body.append("import argparse")
    body.append("import json")
    body.append("import os")
    body.append("import shutil")
    body.append("import sys")
    body.append("")
    body.append("import numpy as np")
    body.append("")
    body.append("# [D15] numpy 2.x removed np.float / np.int / np.complex / np.mat,")
    body.append("# and openEMS.ports.MSLPort still uses np.int. The shim MUST run")
    body.append("# before the openEMS import.")
    body.append("for _name, _value in (('float', float), ('int', int),")
    body.append("                      ('complex', complex), ('bool', bool),")
    body.append("                      ('object', object), ('str', str)):")
    body.append("    if not hasattr(np, _name):")
    body.append("        setattr(np, _name, _value)")
    body.append("if not hasattr(np, 'mat'):")
    body.append("    np.mat = np.asmatrix")
    body.append("")
    body.append("from CSXCAD.CSXCAD import ContinuousStructure  # noqa: E402")
    body.append("from openEMS.openEMS import openEMS  # noqa: E402")
    if any(p.family == "msl" for p in plan.ports):
        body.append("from openEMS.ports import MSLPort  # noqa: E402")
    body.append("")
    body.append(f"EMITTER = {OPENEMS_EMITTER_VERSION!r}")
    body.append(f"SOURCE_RFX_VERSION = {plan.rfx_version!r}")
    body.append(f"DESIGN_IR_SCHEMA = {plan.schema!r}")
    body.append(f"UNIT = {UNIT_M!r}")
    body.append(f"BOUNDARY = {list(plan.boundary)!r}")
    body.append(f"NRTS = {plan.n_timesteps!r}")
    body.append(f"END_CRITERIA = {plan.end_criteria!r}")
    body.append(f"DRIVEN_PORTS = {list(plan.driven_port_numbers)!r}")
    body.append("")
    for axis in _AXES:
        body.append(
            f"{axis.upper()}_LINES_MM = np.array([{_floats(plan.mesh_lines_mm[axis])}])"
        )
    body.append("")
    body.append(f"FREQS_HZ = np.array([{_floats(plan.freqs_hz)}])")
    body.append("")
    body.append("# Per-driven-port Gaussian excitation: SetGaussExcite(f0, fc) with")
    body.append("# fc = bandwidth * f0, the mapping every openEMS script here uses.")
    body.append("_EXCITATION = {")
    for port in plan.ports:
        if port.excite:
            body.append(f"    {port.number}: ({port.f0_hz!r}, {port.fc_hz!r}),")
    body.append("}")
    body.append("")
    body.append("APPROXIMATIONS = [")
    for note in plan.approximations:
        body.append(f"    {note!r},")
    body.append("]")
    body.append("")
    body.append("")
    body.extend(_render_build(plan))
    body.append("")
    body.append(_RUNNER.strip("\n"))
    body.append("")
    return "\n".join(body)
