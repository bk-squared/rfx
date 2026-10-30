"""Compatibility imports retain identity and name their replacement path."""

from importlib import import_module
import warnings


def test_old_material_imports_warn_and_keep_identity():
    targets = {
        "MaterialArrays": "rfx.core.yee",
        "init_materials": "rfx.core.yee",
        "DebyePole": "rfx.materials.debye",
        "DebyeCoeffs": "rfx.materials.debye",
        "DebyeState": "rfx.materials.debye",
        "init_debye": "rfx.materials.debye",
        "update_e_debye": "rfx.stepping.poles",
    }
    for name, target in targets.items():
        expected = getattr(import_module(target), name)
        for _ in range(2):
            namespace = {}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                exec(f"from rfx.materials import {name}", namespace)
            assert namespace[name] is expected
            assert caught, f"old import {name} did not warn"
            assert all(w.category is DeprecationWarning for w in caught)
            assert all(f"rfx.materials.{name} moved to {target}.{name};" in str(w.message)
                       for w in caught)
