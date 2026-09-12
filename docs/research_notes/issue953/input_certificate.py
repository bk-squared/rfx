"""Build-only certificate for the current cv06b falsifier input contract."""
import hashlib
import importlib.util
import json
from pathlib import Path
import rfx

ROOT = Path(__file__).resolve().parents[3]
assert Path(rfx.__file__).resolve().parent == ROOT / "rfx"
def forbid_solve(*args, **kwargs):
    raise AssertionError("input certificate must not execute FDTD")
rfx.Simulation.run = forbid_solve
rfx.Simulation.compute_msl_s_matrix = forbid_solve
producer = ROOT / "scripts/diagnostics/cv06b_build_falsifiers.py"
spec = importlib.util.spec_from_file_location("cv06b_input_certificate", producer)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
records = {}
for label, cv, sim, geometry in module.prepare_inputs():
    records[label] = {
        "declared_dx_m": cv.DX,
        "declared_stub_width_m": cv.W_STUB,
        "declared_stub_length_m": cv.STUB_LEN,
        "realized_metal": geometry,
    }
print(json.dumps({
    "scope": "production build-only input certificate; not RF validation",
    "producer_sha256": hashlib.sha256(producer.read_bytes()).hexdigest(),
    "case_sha256": hashlib.sha256((ROOT / "validation/crossval/06b_msl_notch_filter_uniform.py").read_bytes()).hexdigest(),
    "inputs": records,
}, indent=2))
