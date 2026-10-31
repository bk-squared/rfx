"""The CPU CPML primal write-back adds no whole-field copy (#2125).

An optimization barrier after each dynamic_update_slice in _apply_ordered_slab
prevents the chained-update copies introduced by jaxlib 0.11.2. The custom JVP
calls that primal; its scatter-add tangent chain needs no barrier. Exercise the
forward compile and reverse gradient, including checkpoint recomputation.
Ported from fa9990ffd; census: rfx-archive/rfx/records/20261028-2125-copy-sites/.
"""

from collections import Counter
from pathlib import Path
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "perf"))
from _census_scenes import base  # noqa: E402
from step_cost_census import domain_size_threshold  # noqa: E402

WRITEBACK = "boundaries/_cpml_slab.py"

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason="the slab write-back with dynamic_update_slice is the CPU form; GPU takes the padded one")


def _innermost_rfx_frame(text):
    """stack_frame_id -> 'path/under/rfx.py' of its innermost rfx frame, from the module's frame tables."""
    section, files, locations, frames = None, {}, {}, {}
    for line in text.splitlines():
        if line in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
            section = line
            continue
        row = re.match(r"^(\d+) (.*)$", line)
        if not (section and row):
            continue
        key, body = int(row.group(1)), row.group(2)
        fields = dict(re.findall(r"(\w+)=(\d+)", body))
        if section == "FileNames":
            files[key] = body.strip('"')
        elif section == "FileLocations":
            locations[key] = int(fields["file_name_id"])
        elif section == "StackFrames":
            frames[key] = (int(fields["file_location_id"]), int(fields["parent_frame_id"]))

    def resolve(frame):
        seen = set()
        while frame in frames and frame not in seen:
            seen.add(frame)
            location, frame = frames[frame]
            name = files[locations[location]]
            if "/rfx/" in name:
                return name.split("/rfx/", 1)[1]
        return None
    return resolve


@pytest.mark.parametrize("path", ["uniform", "graded"])
@pytest.mark.parametrize("mode", ["forward", "gradient"])
def test_cpml_writeback_adds_no_field_copy(path, mode):
    sim = base(24, path).sim
    grid = sim._build_nonuniform_grid() if path == "graded" else sim._build_grid()
    def probes(eps):
        return jnp.asarray(sim.forward(n_steps=20, skip_preflight=True, eps_override=eps).time_series)

    def energy(eps):
        return jnp.sum(probes(eps) ** 2)

    step = jax.jit(probes if mode == "forward" else jax.grad(energy))
    text = step.lower(jnp.ones(grid.shape, jnp.float32)).compile().as_text()
    threshold = domain_size_threshold(text, int(np.prod(grid.shape)))
    source = _innermost_rfx_frame(text)
    definitions = dict(re.findall(r"^\s*(?:ROOT )?(%[\w.\-]+) = (.*)$", text, flags=re.M))

    def frame_of(name):
        found = re.search(r"stack_frame_id=(\d+)", definitions.get(name, ""))
        return source(int(found.group(1))) if found else None

    # The attribution must see the write-back at all, or the count below proves nothing.
    assert any(frame_of(name) == WRITEBACK for name in definitions), "no instruction attributed to the write-back"
    copies = Counter()
    for body in definitions.values():
        copy = re.match(r"\w+\[([\d,]*)\]\S* copy\((%[\w.\-]+)\)", body)
        if copy and np.prod([int(d) for d in copy.group(1).split(",") if d]) >= threshold:
            copies[frame_of(copy.group(2))] += 1
    assert copies[WRITEBACK] == 0, (jax.__version__, dict(copies))
