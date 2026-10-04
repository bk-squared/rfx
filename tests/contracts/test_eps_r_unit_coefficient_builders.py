"""Every traced E-coefficient builder takes its reverse pass in eps_r units (#1357).

The E coefficients were written in SI units, ``eps = eps_r*EPS_0`` and a
division by it. The forward VALUE is fine. The reverse pass is not: the VJP of
``x/eps`` multiplies the cotangent by ``eps**-2`` (about 1.3e22), so a float32
cotangent above about 2.6e16 on Cb overflowed, and ``0*inf`` gave NaN on every
lossless cell. Whether it fired depended only on the objective's scale; the
#1357 fixture (a sum of squared probe fields on the graded-mesh lane, about
2e17) had 16 non-finite gradient cells. On the ADI lane the SI coupling
``half_dt**2/(MU_0*eps*dx*dx)`` squares a number near 1e-23, so its eps
gradient was non-finite at any scale.

The fix is one helper, ``rfx.core.yee.si_value_eps_r_grad(si_fn, eps_r_fn,
*args)``. It returns the SI spelling's value bit for bit and takes the
derivative from the same coefficients written in eps_r units.

Two tables, and a discovery step that holds them to the code:

* ``BUILDERS`` names each routed builder. Each must call the helper, or
  a shared coefficient builder, which calls it.
* ``NOT_A_TRACED_EPS_COEFFICIENT`` names every other function in ``rfx/`` that
  reads ``EPS_0``, with the reason no traced permittivity is divided through
  it there (host arithmetic, a vacuum constant, a profile, a pad value, a lane
  no traced permittivity reaches).

Discovery: every function in ``rfx/`` whose own body reads ``EPS_0`` (or an
import alias of it) must be a SPELLING -- a function handed to the helper, or
one such a function calls in its module -- or a
``NOT_A_TRACED_EPS_COEFFICIENT`` row. A new builder written in SI units fails
here until someone routes it or says why it is not one (UPML's ``_e_coeffs``
was missed this way before discovery existed). A routed builder may not read
``EPS_0`` in its own body, so a builder that routes one coefficient and
leaves another in SI is found too. Module-level code runs at import and cannot
see a traced value, so it is not counted.

The allow-list only shrinks: a stale row (a function that no longer reads
``EPS_0``) fails, and its length is capped at ``ALLOWLIST_CEILING``.

What discovery cannot see: a division by an SI permittivity that arrives
already scaled, with no ``EPS_0`` in the dividing function (the mixed
Debye+Lorentz update divides by ``gamma_total = 1/cc + beta``, built in SI by
``init_lorentz``). The behaviour rows in
``tests/unit/autodiff/test_gradient_scale_invariance.py`` gate that.
"""
from __future__ import annotations

import ast
import functools
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# (file, function) -> what it builds. Each must call one of ROUTES, and none
# may read EPS_0 in its own body.
BUILDERS = {
    ("rfx/core/yee.py", "e_update_coeffs"):
        "Ca/Cb of update_e, update_e_nu, update_e_box, the design box and every "
        "Cb-normalised source",
    ("rfx/core/yee.py", "precompute_coeffs"):
        "the GPU fast lane's baked Ca/Cb (_bake)",
    ("rfx/core/yee.py", "update_e_aniso"):
        "Stage-1 subpixel Ca/Cb (uniform)",
    ("rfx/core/yee.py", "update_e_nu_aniso"):
        "Stage-1 subpixel Ca/Cb (graded mesh)",
    ("rfx/core/yee.py", "update_e_aniso_inv"):
        "Stage-2 inverse-eps Ca/Cb (kottke_pec)",
    ("rfx/boundaries/cpml.py", "apply_cpml_e"):
        "the CPML psi coefficient dt/eps",
    ("rfx/boundaries/upml.py", "init_upml"):
        "UPML's Ca/Cb (_e_coeffs)",
    ("rfx/adi.py", "adi_step_2d"):
        "ADI 2-D damping, dt/eps and the Cx/Cy couplings",
    ("rfx/adi.py", "adi_step_3d"):
        "ADI 3-D damping, dt/eps, cc and the Cx/Cy/Cz couplings",
    ("rfx/adi.py", "apply_adi_cpml_2d"):
        "the ADI ADE-CPML 1/eps",
    ("rfx/materials/thin_conductor.py", "sheet_update_coeffs"):
        "the resistive sheet's exponential-stepping A/B",
    ("rfx/lumped.py", "edge_update_denominator"):
        "a lumped element's D0 = eps/dt + sigma/2 (traced branch)",
    ("rfx/runners/_distributed_common.py", "component_e_coeffs"):
        "both multi-device lanes' slab Ca/Cb (#1303: e_update_coeffs per component)",
    ("rfx/runners/_distributed_common.py", "_apply_cpml_e_distributed"):
        "sim.run(devices=...) CPML psi coefficient dt/eps (_ce)",
    ("rfx/runners/distributed_nu.py", "_apply_cpml_e_local_nu"):
        "forward(distributed=True) CPML psi coefficient dt/eps (_ce)",
    ("rfx/materials/debye.py", "init_debye"): "Debye initialization",
    ("rfx/materials/debye.py", "debye_e_coeffs"):
        "Debye ADE ca/cb/cc, including distributed in-loop builders",
    ("rfx/materials/lorentz.py", "init_lorentz"): "Lorentz initialization",
    ("rfx/materials/lorentz.py", "lorentz_e_coeffs"):
        "Lorentz ADE ca/cb/cc, including distributed in-loop builders",
    ("rfx/materials/lorentz.py", "mixed_e_component_coeffs"):
        "mixed Debye+Lorentz ca/cb/cc on every supported lane",
    ("rfx/simulation.py", "_update_e_with_optional_dispersion"):
        "uniform mixed Debye+Lorentz update",
    ("rfx/nonuniform.py", "_update_e_nu_dispersive"):
        "graded mixed Debye+Lorentz update",
}

_HOST = "host arithmetic on Python/NumPy values"
_VACUUM = "a vacuum constant: EPS_0 divides no permittivity"
_C0 = "host: c0 or the wave impedance from EPS_0 and MU_0"
_ENERGY = "a diagnostic energy: EPS_0 multiplies fields, divides nothing"
_SUBGRID = ("the SBP-SAT coupling takes SI eps, but no traced permittivity "
            "reaches the subgridded lane: forward()/optimize() refuse "
            "add_refinement (#1240) and run() takes no eps_override")

# (file, qualified function) -> why no traced permittivity is divided through
# EPS_0 here. May only shrink (ALLOWLIST_CEILING).
NOT_A_TRACED_EPS_COEFFICIENT = {
    ("rfx/materials/debye.py", "debye_pole_coeffs"):
        "material-independent ADE beta: EPS_0 multiplies the pole strength",
    ("rfx/materials/lorentz.py", "lorentz_pole_coeffs"):
        "material-independent ADE c: EPS_0 multiplies the pole strength",
    ("rfx/adi.py", "init_adi_cpml_2d"): _C0,
    ("rfx/adi.py", "make_adi_absorbing_sigma"): _C0,
    ("rfx/adi.py", "make_adi_absorbing_sigma_3d"): _C0,
    ("rfx/auto_config.py", "auto_configure"): _HOST,
    ("rfx/boundaries/cpml.py", "_cpml_profile"):
        "the CPML profile: sigma_max and b = exp(-(sigma/kappa + alpha)*dt/EPS_0)",
    ("rfx/boundaries/cpml.py", "apply_cpml_e._face"):
        "the vacuum dt/EPS_0 used when no materials are given",
    ("rfx/boundaries/upml.py", "_sigma_profile_1d"): _C0,
    ("rfx/boundaries/upml.py", "init_upml._h_coeffs"):
        "UPML's H coefficients: the PML loss sigma_perp*dt/(2*EPS_0)",
    ("rfx/current_moments.py", "accumulate_current_moments"):
        "EPS_0/dt multiplies a field difference (displacement current)",
    ("rfx/current_moments.py", "end_of_record_moments"): _HOST,
    ("rfx/floquet.py", "extract_floquet_modes"): _C0,
    ("rfx/lumped.py", "setup_rlc_materials"):
        "a capacitor's eps_r stamp C*d/(EPS_0*A): EPS_0 divides a capacitance",
    ("rfx/lumped.py", "setup_rlc_materials_traced"):
        "a capacitor's eps_r stamp C*d/(EPS_0*A): EPS_0 divides a capacitance",
    ("rfx/nonuniform.py", "current_source_cb"):
        "the drive Cb: its traced branch is already in eps_r units (#1317), "
        "its float branch is host arithmetic",
    ("rfx/ris.py", "RISUnitCell._build_sim"): _HOST,
    ("rfx/runners/_distributed_common.py", "cpml_coeff_e_vacuum"): _VACUUM,
    ("rfx/runners/subgridded.py", "_run_subgridded_once"): _C0,
    ("rfx/simulation.py", "_warn_static_remnant_cap_hit"): _ENERGY,
    ("rfx/sources/coaxial_port.py", "build_coaxial_tem_plane_source_specs"):
        _HOST + " (float(eps_r))",
    ("rfx/sources/coaxial_port.py", "coaxial_tem_capacitance_per_m"): _HOST,
    ("rfx/sources/msl_port.py", "compute_msl_mode_profile._cap_per_metre"): _HOST,
    ("rfx/sources/tfsf.py", "_apply_closed_box_e"): _VACUUM,
    ("rfx/sources/tfsf.py", "apply_tfsf_e"): _VACUUM,
    ("rfx/sources/tfsf.py", "update_tfsf_1d_e"): _VACUUM + " (the 1-D auxiliary line)",
    ("rfx/sources/tfsf_2d.py", "_update_e_tez"): _VACUUM + " (the 1-D auxiliary line)",
    ("rfx/sources/tfsf_2d.py", "_update_e_tmz"): _VACUUM + " (the 1-D auxiliary line)",
    ("rfx/sources/tfsf_2d.py", "apply_tfsf_2d_e"): _VACUUM,
    ("rfx/sources/tfsf_2d.py", "init_tfsf_2d"): _C0,
    ("rfx/sources/tfsf_oblique_open.py", "apply_methodB_e"): _VACUUM,
    ("rfx/sources/waveguide_port.py", "_compute_mode_impedance"):
        "a modal impedance beta/(omega*EPS_0): EPS_0 divides no permittivity",
    ("rfx/sources/waveguide_port.py", "apply_waveguide_port_e"): _VACUUM,
    ("rfx/sources/waveguide_port.py", "init_waveguide_port"): _HOST,
    ("rfx/sparams/_common.py", "_resolve_msl_auto_offsets"): _C0,
    ("rfx/sparams/mixed.py", "compute_mixed_s_matrix"): _C0,
    ("rfx/sparams/msl.py", "compute_msl_s_matrix"): _C0,
    ("rfx/subgridding/disjoint_3d.py", "compute_disjoint_energy_3d"): _ENERGY,
    ("rfx/subgridding/jit_runner.py", "_z_slab_material_coupling_e_3d.face_coeffs"):
        _SUBGRID,
    ("rfx/subgridding/jit_runner.py", "_z_slab_material_coupling_h_3d.face_coeffs"):
        _SUBGRID,
    ("rfx/subgridding/sbp_sat_1d.py", "_update_e_1d"): _VACUUM,
    ("rfx/subgridding/sbp_sat_1d.py", "compute_energy"): _ENERGY,
    ("rfx/subgridding/sbp_sat_1d.py", "step_subgrid_1d"): _VACUUM,
    ("rfx/subgridding/sbp_sat_2d.py", "_shared_node_update_2d"): _VACUUM,
    ("rfx/subgridding/sbp_sat_2d.py", "_update_ez_interior_2d"): _VACUUM,
    ("rfx/subgridding/sbp_sat_2d.py", "compute_energy_2d"): _ENERGY,
    ("rfx/subgridding/sbp_sat_3d.py", "compute_energy_3d"): _ENERGY,
}

# The allow-list's length when it was written. Lower it when a row goes; a PR
# that raises it is adding an EPS_0 reader it says is not a coefficient.
ALLOWLIST_CEILING = 47

HELPER = "si_value_eps_r_grad"
SHARED_ROUTES = {
    ("rfx/core/yee.py", "e_update_coeffs"),
    ("rfx/materials/debye.py", "debye_e_coeffs"),
    ("rfx/materials/lorentz.py", "lorentz_e_coeffs"),
    ("rfx/materials/lorentz.py", "mixed_e_component_coeffs"),
}
ROUTES = {HELPER} | {name for _, name in SHARED_ROUTES}
_EPS_0 = "EPS_0"


# --- the AST walk ---------------------------------------------------------

def _module_rel(module: str) -> str:
    return module.replace(".", "/") + ".py"


class _Module:
    """One file: its function defs by qualified name, the names that mean
    EPS_0, and what its ``from ... import`` names point at."""

    def __init__(self, rel: str, text: str):
        self.rel = rel
        self.tree = ast.parse(text)
        self.defs: dict[str, ast.AST] = {}
        self.imports: dict[str, tuple[str, str]] = {}   # local -> (rel, name)
        self.eps_names = {_EPS_0}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    local = alias.asname or alias.name
                    if alias.name == _EPS_0:
                        self.eps_names.add(local)
                    self.imports[local] = (_module_rel(node.module), alias.name)
        self._index(self.tree, [])

    def _index(self, node, stack):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defs[".".join(stack + [child.name])] = child
                self._index(child, stack + [child.name])
            elif isinstance(child, ast.ClassDef):
                self._index(child, stack + [child.name])
            else:
                self._index(child, stack)

    @staticmethod
    def own_nodes(fn):
        """The nodes of ``fn``'s body, not descending into nested defs
        (each nested def is its own row)."""
        todo = list(ast.iter_child_nodes(fn))
        while todo:
            node = todo.pop()
            yield node
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                todo.extend(ast.iter_child_nodes(node))

    def reads_eps_0(self, fn) -> bool:
        for node in self.own_nodes(fn):
            if isinstance(node, ast.Name) and node.id in self.eps_names:
                return True
            if isinstance(node, ast.Attribute) and node.attr == _EPS_0:
                return True
        return False

    def lookup(self, name: str, scope: str):
        """What a bare ``name`` means inside function ``scope``: a def nested
        in ``scope`` or an enclosing function, a module def, or an import."""
        parts = scope.split(".") if scope else []
        while True:
            qual = ".".join(parts + [name])
            if qual in self.defs:
                return (self.rel, qual)
            if not parts:
                break
            parts.pop()
        return self.imports.get(name)


@functools.lru_cache(maxsize=None)
def _parse_tree(root: Path) -> dict[str, _Module]:
    return {path.relative_to(root).as_posix():
            _Module(path.relative_to(root).as_posix(), path.read_text(encoding="utf-8"))
            for path in sorted((root / "rfx").rglob("*.py"))}


def _parse(root: Path, overrides=None) -> dict[str, _Module]:
    """Every ``rfx/**.py`` under ``root``; ``overrides`` maps a path to the
    text to parse instead (a new path adds a file)."""
    mods = dict(_parse_tree(root))
    for rel, text in (overrides or {}).items():
        mods[rel] = _Module(rel, text)
    return mods


def _calls(fn):
    return (n for n in _Module.own_nodes(fn) if isinstance(n, ast.Call))


def spellings(mods: dict[str, _Module]) -> set[tuple[str, str]]:
    """Every function handed to the helper as ``si_fn`` or ``eps_r_fn``, and
    every function those call by bare name, transitively."""
    found, todo = set(), []
    for mod in mods.values():
        for qual, fn in mod.defs.items():
            for call in _calls(fn):
                f = call.func
                name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
                if name != HELPER:
                    continue
                todo.extend(mod.lookup(a.id, qual) for a in call.args[:2]
                            if isinstance(a, ast.Name))
    while todo:
        key = todo.pop()
        if (key is None or key in found or key[0] not in mods
                or key[1] not in mods[key[0]].defs):
            continue
        found.add(key)
        mod = mods[key[0]]
        todo.extend(mod.lookup(c.func.id, key[1])
                    for c in _calls(mod.defs[key[1]]) if isinstance(c.func, ast.Name))
    return found


def eps_0_readers(mods: dict[str, _Module]) -> set[tuple[str, str]]:
    return {(rel, qual) for rel, mod in mods.items()
            for qual, fn in mod.defs.items() if mod.reads_eps_0(fn)}


def unclassified(mods: dict[str, _Module]) -> list[str]:
    """``file::function`` readers of EPS_0 that no table accounts for."""
    known = spellings(mods) | set(NOT_A_TRACED_EPS_COEFFICIENT)
    return sorted(f"{rel}::{qual}" for rel, qual in eps_0_readers(mods) - known)


def _called_names(fn) -> set[str]:
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return names


def unrouted(mods: dict[str, _Module], rows) -> list[str]:
    """The ``file::function`` rows whose body calls none of ``ROUTES``, or
    reads EPS_0 itself (a coefficient left in SI next to a routed one). A
    row whose function is gone raises, so a rename cannot hide it."""
    bad = []
    for rel, name in rows:
        if name not in mods[rel].defs:
            raise LookupError(f"{rel}: no def {name}")
        fn = mods[rel].defs[name]
        if not (_called_names(fn) & ROUTES) or mods[rel].reads_eps_0(fn):
            bad.append(f"{rel}::{name}")
    return bad


@pytest.fixture(scope="module")
def mods():
    return _parse(REPO)


# --- the gates -------------------------------------------------------------

@pytest.mark.parametrize("row", sorted(SHARED_ROUTES))
def test_shared_coefficient_route_is_itself_routed(row, mods):
    """A shared builder counts as a route only because it calls the helper."""
    rel, name = row
    assert HELPER in _called_names(mods[rel].defs[name])


@pytest.mark.parametrize("row", sorted(BUILDERS), ids=lambda r: f"{r[0]}::{r[1]}")
def test_builder_is_routed_through_the_eps_r_unit_helper(row, mods):
    assert unrouted(mods, [row]) == [], BUILDERS[row]


def test_every_eps_0_reader_is_accounted_for(mods):
    """Discovery: a function that reads EPS_0 is a spelling handed to the
    helper or an allow-listed non-coefficient."""
    missing = unclassified(mods)
    assert not missing, (
        "these functions read EPS_0 and no table accounts for them:\n  "
        + "\n  ".join(missing)
        + "\n\nIf one builds an E coefficient from a permittivity that can be "
        "traced, route it through rfx.core.yee.si_value_eps_r_grad (#1357) "
        "and add it to BUILDERS. Otherwise add it to "
        "NOT_A_TRACED_EPS_COEFFICIENT with the reason, and raise "
        "ALLOWLIST_CEILING in the same diff.")


def test_the_tables_are_not_stale(mods):
    readers = eps_0_readers(mods)
    stale = sorted(f"{rel}::{qual}" for rel, qual in NOT_A_TRACED_EPS_COEFFICIENT
                   if (rel, qual) not in readers)
    assert not stale, (
        "allow-listed but no longer reading EPS_0 (delete the row and lower "
        "ALLOWLIST_CEILING):\n  " + "\n  ".join(stale))
    assert len(NOT_A_TRACED_EPS_COEFFICIENT) <= ALLOWLIST_CEILING
    both = sorted(set(BUILDERS) & set(NOT_A_TRACED_EPS_COEFFICIENT))
    assert not both, f"a routed builder cannot also be allow-listed: {both}"


# --- falsifiers: the defects planted in the parsed text --------------------

def _plant(rel, old, new):
    text = (REPO / rel).read_text(encoding="utf-8")
    assert text.count(old) == 1, f"{rel}: the planting anchor moved"
    return {rel: text.replace(old, new)}


def test_discovery_finds_a_missed_builder():
    """UPML's ``_e_coeffs`` as it was before #1357 routed it (the builder the
    first version of this contract missed), and a new file with an SI Cb."""
    planted = _plant(
        "rfx/boundaries/upml.py",
        "        return si_value_eps_r_grad(_upml_e_coeffs_si, _upml_e_coeffs_eps_r,\n"
        "                                   sigma_perp, eps_r, sigma_mat, dt)\n",
        "        eps_abs = eps_r * jnp.float32(EPS_0)\n"
        "        return sigma_perp, (dt / eps_abs) / (1.0 + sigma_mat)\n")
    # The two spellings nothing hands to the helper any more surface too.
    assert unclassified(_parse(REPO, planted)) == [
        "rfx/boundaries/upml.py::_upml_e_coeffs_eps_r",
        "rfx/boundaries/upml.py::_upml_e_coeffs_si",
        "rfx/boundaries/upml.py::init_upml._e_coeffs"]
    new_file = ("from rfx.core.yee import EPS_0 as _E0\n\n\n"
                "def new_cb(eps_r, dt):\n    return dt / (eps_r * _E0)\n")
    assert unclassified(_parse(REPO, {"rfx/_planted_builder.py": new_file})) == [
        "rfx/_planted_builder.py::new_cb"]


def test_a_builder_with_one_coefficient_left_in_si_is_named():
    """One of update_e_aniso's three components back in SI next to two
    routed ones: the route check alone passes it, the EPS_0 read does not."""
    rel = "rfx/core/yee.py"
    fn = _parse(REPO)[rel].defs["update_e_aniso"]
    text = (REPO / rel).read_text(encoding="utf-8")
    body = "\n".join(text.splitlines()[fn.lineno - 1:fn.end_lineno])
    line = "    ca_ez, cb_ez = e_update_coeffs(eps_ez, sigma_ez, dt)"
    assert body.count(line) == 1
    mods = _parse(REPO, {rel: text.replace(body, body.replace(
        line, "    ca_ez, cb_ez = 1.0, dt / (eps_ez * EPS_0)"))})
    assert unrouted(mods, [(rel, "update_e_aniso")]) == [f"{rel}::update_e_aniso"]
    assert unclassified(mods) == [f"{rel}::update_e_aniso"]


def test_the_scan_sees_a_builder_put_back_in_si_units():
    """CPML's psi coefficient returned to its SI spelling through a lambda
    (no EPS_0 read left in apply_cpml_e) must be named by the route check."""
    rel = "rfx/boundaries/cpml.py"
    fn = _parse(REPO)[rel].defs["apply_cpml_e"]
    text = (REPO / rel).read_text(encoding="utf-8")
    body = "\n".join(text.splitlines()[fn.lineno - 1:fn.end_lineno])
    assert body.count(f"{HELPER}(") == 2
    planted = _parse(REPO, {rel: text.replace(body, body.replace(
        f"{HELPER}(", "(lambda si, r, *a: si(*a))("))})
    assert unrouted(planted, BUILDERS) == [f"{rel}::apply_cpml_e"]
