"""Device operands pinned to the pre-table stepping signatures and pad writes."""

from dataclasses import fields, is_dataclass
import numpy as np
import jax
import pytest
from rfx import Simulation, Box, Sphere

LANES = (
    "uniform_run",
    "uniform_forward",
    "graded_run",
    "graded_forward",
    "distributed_v2_run",
    "distributed_v2_forward",
    "distributed_nu_run",
    "distributed_nu_forward",
    "adi_run",
    "adi_forward",
    "subgridded",
    "conformal",
    "smoothed",
    "full_tensor",
    "cpml",
    "traced_mesh",
    "nominal_mesh",
    "traced_shape",
)


def arrays(value, path=""):
    if isinstance(value, (jax.Array, np.ndarray, jax.core.Tracer)):
        return {path: (tuple(value.shape), str(value.dtype))}
    if hasattr(value, "_fields"):
        children = ((n, getattr(value, n)) for n in value._fields)
    elif is_dataclass(value):
        children = ((f.name, getattr(value, f.name)) for f in fields(value))
    elif isinstance(value, (tuple, list)):
        children = enumerate(value)
    elif isinstance(value, dict):
        children = value.items()
    else:
        return {}
    return {
        k: v
        for n, x in children
        for k, v in arrays(x, f"{path}.{n}" if path else str(n)).items()
    }


@pytest.mark.parametrize("lane", LANES)
def test_device_material_arrays_match_base(lane, monkeypatch):
    from rfx.stepping import prepared

    # cb4cd5c16: uniform/graded take (grid, MaterialArrays, n_steps);
    # slab takes sharded_materials, graded slab takes it positionally;
    # ADI takes eps_r and sigma after six fields; subgrid takes coarse/fine.
    # MaterialArrays and ComponentMaterials below are written out from that
    # commit, never derived from the current product or an assembly helper.
    shape = (15, 15, 15) if lane == "cpml" else (11, 11, 11)
    if lane.startswith("distributed_v2"):
        shape = (16, 11, 11)  # two x slabs with two ghosts each
    elif lane.startswith("distributed_nu"):
        shape = (2, 9, 11, 11)  # rank, ceil(11/2) + three ghosts, y, z
    expected = {}
    if lane.startswith("adi"):
        expected = {n: (shape, "float32") for n in ("eps_r", "sigma")}
    else:
        for prefix, size in [("materials", shape)] + (
            [("fine", (21, 21, 13))] if lane == "subgridded" else []
        ):
            expected.update(
                {f"{prefix}.{n}": (size, "float32") for n in ("eps_r", "sigma", "mu_r")}
            )
            if not lane.startswith("distributed"):
                expected.update(
                    {
                        f"{prefix}.components.{n}.{a}": (size, "float32")
                        for n in (
                            "eps",
                            "sigma",
                            "mu",
                            "eps_update",
                            "sigma_update",
                            "mu_update",
                            "upml_eps",
                            "upml_sigma",
                        )
                        for a in range(3)
                    }
                )
                expected.update(
                    {
                        f"{prefix}.components.cells_view.{n}": (size, "float32")
                        for n in ("eps_r", "sigma", "mu_r")
                    }
                )
    if lane in ("conformal", "smoothed"):
        expected.update({f"aniso_eps.{a}": (shape, "float32") for a in range(3)})
    if lane == "conformal":
        expected.update(
            {f"conformal_weights.{a}": (shape, "float32") for a in range(3)}
        )
    if lane == "full_tensor":
        expected.update(
            {f"aniso_eps.diagonal.{a}": (shape, "float32") for a in range(3)}
        )
        expected.update(
            {f"aniso_eps.nodes.xi.{a}": (shape, "float32") for a in range(6)}
        )
        expected.update(
            {
                f"aniso_eps.nodes.{n}.{a}": (shape, "bool")
                for n in ("bulk", "active")
                for a in range(3)
            }
        )
        expected["aniso_eps.nodes.triplets"] = ((8, *shape, 3, 3), "float32")
        expected.update(
            {
                f"aniso_eps.nodes.cross.{a}": ((2, 2, *shape), "float32")
                for a in range(3)
            }
        )
        expected["aniso_eps.nodes.guard_count"] = ((), "int32")
    calls = []
    orig = prepared.advance

    def advance(root, stepper, *args, **kwargs):
        layout = kwargs.get("_operand_layout", "grid")

        def step(*args, **kwargs):
            if layout == "slab":
                mats = {"materials": kwargs["sharded_materials"]}
            elif layout == "adi3d":
                mats = {"eps_r": args[6], "sigma": args[7]}
            else:
                mats = {"materials": args[1]}
                if lane == "subgridded":
                    mats["fine"] = args[2]
            # The base stepping signatures pass these tensor operands separately.
            mats.update(
                {
                    name: kwargs[name]
                    for name in ("aniso_eps", "aniso_inv_eps", "conformal_weights")
                    if name in kwargs
                }
            )
            observed = arrays(mats)
            assert observed == expected, (lane, observed)
            calls.append(observed)
            return stepper(*args, **kwargs)

        return orig(root, step, *args, **kwargs)

    monkeypatch.setattr(prepared, "advance", advance)
    if lane in ("traced_mesh", "nominal_mesh", "traced_shape"):
        import jax.numpy as jnp

        def evaluate(delta):
            profile = jnp.full(10, 0.001).at[3].add(delta).at[4].add(-delta)
            kw = (
                {}
                if lane == "traced_shape"
                else dict(
                    dz_profile=profile,
                    dt=0.2 * 0.001 / 299792458.0,
                    dt_min_cell=0.0005,
                    nominal_profiles={"z": np.full(10, 0.001)}
                    if lane == "nominal_mesh"
                    else None,
                )
            )
            sim = Simulation(
                freq_max=1e9, domain=(0.01,) * 3, dx=0.001, boundary="pec", **kw
            )
            sim.add_material("glass", eps_r=2.0)
            sim.add(
                Sphere((0.005,) * 3, 0.002 + delta)
                if lane == "traced_shape"
                else Box((0.0023,) * 3, (0.0067,) * 3),
                material="glass",
            )
            return jnp.sum(sim.forward(n_steps=1, skip_preflight=True).time_series)

        jax.jit(evaluate)(0.0)
    else:
        graded = lane.startswith("graded") or "_nu_" in lane
        sim = Simulation(
            freq_max=1e9,
            domain=(0.01,) * 3,
            dx=0.001,
            boundary="cpml" if lane == "cpml" else "pec",
            cpml_layers=2,
            solver="adi" if lane.startswith("adi") else "yee",
            **({"dz_profile": np.full(10, 0.001)} if graded else {}),
        )
        sim.add_material("glass", eps_r=2.0)
        if not lane.startswith("adi"):
            sim.add(Box((0.0023,) * 3, (0.0067,) * 3), material="glass")
        if lane == "conformal":
            sim.add(Sphere((0.007,) * 3, 0.0013), material="pec")
        if lane == "subgridded":
            sim.add_refinement(z_range=(0.002, 0.008), ratio=2, validation="research")
        kw = dict(n_steps=1, skip_preflight=True)
        if lane.startswith("distributed"):
            kw["devices"] = jax.devices("cpu")[:2]
            if lane.endswith("forward"):
                kw["distributed"] = True
        if lane == "conformal":
            kw["conformal_pec"] = True
        if lane in ("smoothed", "full_tensor"):
            kw["subpixel_smoothing"] = True if lane == "smoothed" else "full_tensor"
        if lane.endswith("forward"):
            sim.forward(**kw)
        else:
            sim.run(compute_s_params=False, **kw)

    assert len(calls) == 1


def test_pad_device_array_updates_match_base(monkeypatch):
    """Also catch throwaway device arrays that never reach the stepping call."""
    from jax._src.numpy.array_methods import _IndexUpdateRef
    import rfx.model.materials as assembly

    original_pad = assembly.extend_cpml_pad_materials
    original_set = _IndexUpdateRef.set
    writes = []
    active = False

    def set_value(indexed, *args, **kwargs):
        if active:
            writes.append((tuple(indexed.array.shape), str(indexed.array.dtype)))
        return original_set(indexed, *args, **kwargs)

    def pad(*args, **kwargs):
        nonlocal active
        active = True
        try:
            return original_pad(*args, **kwargs)
        finally:
            active = False

    monkeypatch.setattr(_IndexUpdateRef, "set", set_value)
    monkeypatch.setattr(assembly, "extend_cpml_pad_materials", pad)
    sim = Simulation(
        freq_max=1e9, domain=(0.01,) * 3, dx=0.001, boundary="cpml", cpml_layers=2
    )
    sim.add_material("glass", eps_r=2.0)
    sim.add(Box((0.0, 0.002, 0.002), (0.01, 0.006, 0.006)), material="glass")
    sim.run(n_steps=1, skip_preflight=True, compute_s_params=False)
    # Base copies three float32 material arrays: for each axis, one low pad
    # write and two high writes (boundary + pad). No dispersion in this scene.
    assert writes == [((15, 15, 15), "float32")] * (3 * (1 + 2) * 3)
