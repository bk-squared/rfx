import hashlib, importlib.util, json, math, subprocess, sys
from pathlib import Path
import numpy as np
from rfx import Simulation

def no_solve(*args, **kwargs):
    raise AssertionError("build-only probe must not execute FDTD")
Simulation.run = no_solve
Simulation.compute_msl_s_matrix = no_solve
source = Path("validation/crossval/06b_msl_notch_filter_uniform.py")
base = subprocess.check_output(["git", "rev-parse", "origin/main"], text=True).strip()
base_source = subprocess.check_output(["git", "show", f"{base}:{source}"])
assert base_source == source.read_bytes(), "case differs from main"
spec = importlib.util.spec_from_file_location("cv06b_build_probe", source)
cv = importlib.util.module_from_spec(spec); spec.loader.exec_module(cv)
label = sys.argv[1]
if label == "stub_narrow":
    cv.W_STUB = 5 * cv.DX
elif label != "baseline":
    raise ValueError(label)
sim = cv._build_sim()
metal = cv.realized_metal(sim)
def anchor(width, length):
    u = width / cv.H_SUB
    effective = (cv.EPS_R + 1)/2 + (cv.EPS_R - 1)/2 / math.sqrt(1 + 12/u)
    return {"width_m": width, "length_m": length, "eps_eff": effective, "f_hz": cv.C0 / (4*length*math.sqrt(effective))}
lx = cv.L_LINE + 2*cv.PORT_MARGIN
faces = [(lx-cv.W_STUB)/2, (lx+cv.W_STUB)/2]
result = {
    "label":label, "base_main":base,
    "case_source_sha256":hashlib.sha256(base_source).hexdigest(),
    "scope":"actual geometry build and diagnostic analytic models only; no FDTD or electrical-width adjudication",
    "declared":{"dx_m":cv.DX,"trace_width_m":cv.W_TRACE,"stub_width_m":cv.W_STUB,"stub_length_m":cv.STUB_LEN,"substrate_height_m":cv.H_SUB},
    "realized_metal":metal,
    "declared_stub_faces_in_dx":[v/cv.DX for v in faces],
    "diagnostic_anchors":{
        "shipped_trace_row_pitch_declared_length":anchor(metal["trace_w_elec"], cv.STUB_LEN),
        "stub_row_pitch_declared_length":anchor(metal["stub_w_elec"],cv.STUB_LEN),
        "stub_node_span_declared_length":anchor(metal["stub_w"],cv.STUB_LEN),
        "stub_node_span_realized_length":anchor(metal["stub_w"],metal["stub_len"]),
    }
}
print(json.dumps(result, indent=2, default=lambda o: o.item() if isinstance(o,np.generic) else o.tolist()))
