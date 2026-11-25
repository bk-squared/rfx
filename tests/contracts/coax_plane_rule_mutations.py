"""Coaxial plane errors with altered higher modes and disabled physical bounds."""
import os
import subprocess
import sys

import pytest


CASES = {
    "baseline": ("baseline", "", ""),
    "m2": ("defect_restored", "test_default_band_rows", "rewrite(coax_planes, '_coax_plane_rule', '_search(one, every, t_c, n_hi)', '_search(one, every, tolerance, n_hi)')"),
    "m3": ("defect_restored", "test_default_band_rows", "coax_planes.TE21_RATE_FACTOR = 1.0"),
    "m4": ("defect_restored", "test_default_band_rows", "rewrite(coax_planes, '_closed_form_clearance', 'np.log(JUNCTION_AMPLITUDE/junction_threshold)', 'np.log(JUNCTION_AMPLITUDE/(junction_threshold/JUNCTION_TOLERANCE_FRACTION))')"),
    "m5": ("defect_restored", "test_default_band_rows", "rewrite(coax_planes, '_closed_form_clearance', 'SOURCE_TE21_AMPLITUDE*np.exp(-a2t*cc*dz)', '0.0*np.exp(-a2t*cc*dz)')"),
    "m6": ("defect_restored", "test_default_band_rows", "rewrite(coax_planes, '_coax_plane_rule', 'C_c = max(c_a, c_b)', 'C_c = c_a')"),
    "m7": ("defect_restored", "test_start_fallback_pin", "rewrite(coax_planes, '_search', 'if every(n1).max() <= threshold:', 'if True:')"),
    "m8": ("defect_restored", "test_all_bin_evaluation_counts", "rewrite(coax_planes, '_search', 'if every(n1).max() <= threshold:\\n        return n1, False', 'if every(n1).max() <= threshold:\\n        _bisect(lambda n: every(n).max(), threshold, bound)\\n        return n1, False')"),
    "m9": ("defect_restored", "test_one_bin_error_is_bin_zero", "rewrite(coax_planes, '_start_errors', '_start_error(z, b[:1], a1[:1], a2[:1])', '_start_error(z, b[-1:], a1[-1:], a2[-1:])')"),
    "m10": ("defect_restored", "test_eps_r_scaling", "rewrite(coax_planes, '_line_inputs', 'b = 2*np.pi*f*np.sqrt(eps_r)/C0', 'b = 2*np.pi*f*np.sqrt(2.1)/C0')"),
    "m11": ("defect_restored", "test_cutoff_wavenumbers", "rewrite(coax_planes, '_cutoff_wavenumber', 'brentq(fn, x[i], x[i + 1])/inner', 'brentq(fn, x[i], x[i + 1])/outer')"),
    "m12": ("defect_restored", "test_fit_matches_public_fit_two_wave", "rewrite(coax_planes, '_fit_reflection', 'B*np.exp(g*z0)/(A*np.exp(-g*z0))', 'A*np.exp(-g*z0)/(B*np.exp(g*z0))')"),
    "m13": ("defect_restored", "test_declared_band_rows", "rewrite(coax_planes, '_line_inputs', 'np.linspace(freq_min, freq_max, BIN_COUNT)', 'np.linspace(freq_max/10, freq_max, BIN_COUNT)')"),
    "m14": ("defect_restored", "test_clearance_fallback_row", "rewrite(coax_planes, '_coax_plane_rule', '_search(one, every, t_s, n_hi)', '(_bisect(one, t_s, n_hi), False)')"),
    "R1": ("refusal_removed", "test_refuses_traced_input", "rewrite(coax_planes, 'coax_plane_rule', 'except (TypeError, ValueError):', 'except ():')"),
    "R2": ("refusal_removed", "test_refuses_nonpositive_inputs", "rewrite(coax_planes, 'coax_plane_rule', 'if not np.isfinite(value) or value <= 0:', 'if False:')"),
    "R3": ("refusal_removed", "test_refuses_outer_not_above_inner", "rewrite(coax_planes, 'coax_plane_rule', 'if outer <= inner:', 'if False:')"),
    "R4": ("refusal_removed", "test_refuses_freq_min_outside_band", "rewrite(coax_planes, 'coax_plane_rule', 'if not np.isfinite(freq_min) or freq_min <= 0 or freq_min >= freq_max:', 'if False:')"),
    "R5": ("refusal_removed", "test_refuses_tolerance_outside_unit_interval", "rewrite(coax_planes, 'coax_plane_rule', 'if not np.isfinite(tolerance) or tolerance <= 0 or tolerance >= 1:', 'if False:')"),
    "R6": ("refusal_removed", "test_refuses_radius_ratio", "rewrite(coax_planes, 'coax_plane_rule', 'if outer/inner >= RADIUS_RATIO_LIMIT:', 'if False:')"),
    "R7": ("refusal_removed", "test_refuses_freq_max_at_cutoff", "rewrite(coax_planes, '_line_inputs', 'if freq_max >= fc1:', 'if False:')"),
    "R8": ("refusal_removed", "test_refuses_phase_step_at_pi", "rewrite(coax_planes, '_coax_plane_rule', 'if phase >= np.pi:', 'if False:')"),
    "R9": ("refusal_removed", "test_refuses_start_search_bound", "rewrite(coax_planes, '_coax_plane_rule', 'if s_c is None:', 'if False:')"),
    "R10": ("refusal_removed", "test_refuses_source_search_bound", "rewrite(coax_planes, '_coax_plane_rule', 'if c_b is None:', 'if False:')"),
    "R11": ("refusal_removed", "test_refuses_closed_form_clearance_bound", "rewrite(coax_planes, '_coax_plane_rule', 'if c_a is None:', 'if False:')"),
    "R12": ("refusal_removed", "test_refuses_cutoff_scan_without_root", "rewrite(coax_planes, '_cutoff_wavenumber', 'if not roots.size:', 'if False:')"),
    "R13": ("refusal_removed", "test_refuses_zero_spacing", "rewrite(coax_planes, '_coax_plane_rule', 'if D_c < 1:', 'if False:')"),
    "R14-start": ("refusal_removed", "test_refuses_nonfinite_fit_error", "rewrite(coax_planes, '_start_errors', 'if not np.all(np.isfinite(value)):', 'if False:')"),
    "R14-source": ("refusal_removed", "test_refuses_nonfinite_fit_error", "rewrite(coax_planes, '_source_errors', 'if not np.all(np.isfinite(value)):', 'if False:')"),
    "R12-control": ("rewrite_control", "test_refuses_cutoff_scan_without_root", "rewrite(coax_planes, '_cutoff_wavenumber', 'if not roots.size:', 'if not roots.size:')"),
    "R14-start-control": ("rewrite_control", "test_refuses_nonfinite_fit_error", "rewrite(coax_planes, '_start_errors', 'if not np.all(np.isfinite(value)):', 'if not np.all(np.isfinite(value)):')"),
    "R14-source-control": ("rewrite_control", "test_refuses_nonfinite_fit_error", "rewrite(coax_planes, '_source_errors', 'if not np.all(np.isfinite(value)):', 'if not np.all(np.isfinite(value)):')"),
}


@pytest.mark.parametrize("case", CASES)
def test_plane_rule_mutation(case, capsys):
    mode, witness, defect = CASES[case]
    script = '''
import inspect
import pytest
from rfx.model import coax_planes

def rewrite(owner, name, old, new):
    """Keep altered coaxial functions in their own globals, unlike a copied scope."""
    fn = getattr(owner, name)
    source = inspect.getsource(fn)
    assert source.count(old) == 1, (name, old, source.count(old))
    scope = inspect.unwrap(fn).__globals__
    exec(source.replace(old, new), scope)
    setattr(owner, name, scope[name])
'''
    if mode != "baseline":
        script += defect + "\n"
    path = "tests/unit/model/test_coax_plane_rule.py"
    script += f"raise SystemExit(pytest.main([{path!r}, '-q', '-o', 'addopts=', '--tb=short', '-k', {witness!r}]))\n"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            env=dict(os.environ, PYTEST_ADDOPTS=""))
    with capsys.disabled():
        print(f"{case} {mode}: exit={result.returncode}\n{result.stdout}\n{result.stderr}", file=sys.stderr)
    expected = 0 if mode in ("baseline", "rewrite_control") else 1
    assert result.returncode == expected
    if mode == "rewrite_control":
        assert ("1 passed" if case == "R12-control" else "2 passed") in result.stdout
    else:
        assert ("passed" if expected == 0 else "FAILED " + path + "::" + witness) in result.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s", "-o", "addopts="]))
