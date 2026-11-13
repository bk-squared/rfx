"""Record float32 NTFF reverse-mode finiteness; no numerical verdict."""
import gc
import json
import os
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import numpy as np

from rfx.measurement.farfield import compute_far_field_jax
from tests.unit.autodiff.test_adjoint_farfield import scene, loss_function, collocation_mode
from tests.unit.autodiff.test_design_box_tape import BOX_LO, BOX_HI


def array_summary(value):
    array = np.asarray(value)
    bad = np.argwhere(~np.isfinite(array))
    return dict(dtype=str(array.dtype), shape=list(array.shape),
                finite=bool(np.all(np.isfinite(array))),
                nan_count=int(np.count_nonzero(np.isnan(array))),
                inf_count=int(np.count_nonzero(np.isinf(array))),
                first_nonfinite_index=bad[0].tolist() if len(bad) else None)


def forward_fields(sim, eps, steps):
    result = sim.forward(design_box=(BOX_LO, BOX_HI), design_eps_override=eps,
                         n_steps=steps, checkpoint_segments=8,
                         gradient="autodiff", skip_preflight=True)
    theta = jnp.array([np.pi/6, np.pi/2, 5*np.pi/6])
    phi = jnp.arange(8) * (2*np.pi/8)
    ff = compute_far_field_jax(result.ntff_data, result.ntff_box,
                               result.grid, theta, phi)
    intensity = jnp.abs(ff.E_theta)**2 + jnp.abs(ff.E_phi)**2
    power = jnp.sum(intensity * jnp.sin(theta)[None, :, None], axis=(1, 2))
    power = power * (np.pi/3) * (2*np.pi/8)
    return ff.E_theta, ff.E_phi, power


def main():
    jax.config.update("jax_enable_x64", False)
    out = Path(os.environ["OUT"])
    out.mkdir(parents=True, exist_ok=True)
    report = dict(diagnostic="float32 NTFF autodiff; observations only", cases=[])
    started = time.monotonic()
    error = None
    try:
        with collocation_mode("face_centre"):
            for lossy in (False, True):
                sim, eps = scene("ntff", "float32", lossy)
                loss = loss_function(sim, "ntff", "autodiff")
                value_fn = jax.jit(loss, static_argnums=1)
                grad_fn = jax.jit(jax.grad(loss), static_argnums=1)
                ff_fn = jax.jit(lambda e, n: forward_fields(sim, e, n), static_argnums=1)
                for steps in (2400, 4800):
                    value = float(value_fn(eps, steps))
                    gradient = grad_fn(eps, steps)
                    leaves = [dict(path=jax.tree_util.keystr(path) or "design_eps_override",
                                   **array_summary(leaf))
                              for path, leaf in jax.tree_util.tree_flatten_with_path(gradient)[0]]
                    fields = ff_fn(eps, steps)
                    report["cases"].append(dict(
                        lossy=lossy, sigma=0.2 if lossy else 0., collocation="face_centre",
                        steps=steps, gradient_finite=all(leaf["finite"] for leaf in leaves),
                        gradient_leaves=leaves, objective_finite=bool(np.isfinite(value)),
                        objective_value=value if np.isfinite(value) else str(value),
                        forward={name: array_summary(field) for name, field in
                                 zip(("ff.E_theta", "ff.E_phi", "power"), fields)}))
                    (out / "diag.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
                jax.clear_caches()
                gc.collect()
    except Exception:
        error = traceback.format_exc()
        report["execution_error"] = error
        raise
    finally:
        elapsed = time.monotonic()-started
        report["seconds"] = elapsed
        (out / "diag.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        # JUnit describes script execution only; NaNs are observations, not failures.
        suite = ET.Element("testsuite", name="diagnostic_execution", tests="1",
                           failures="0", errors=str(int(error is not None)), time=str(elapsed))
        case = ET.SubElement(suite, "testcase", classname="diagnostic", name="diag", time=str(elapsed))
        if error is not None:
            ET.SubElement(case, "error", message="diagnostic execution error").text = error
        ET.ElementTree(suite).write(out / "diag.xml", encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
