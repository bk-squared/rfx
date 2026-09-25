"""Put the magnetic absorber grading back on the electric integer nodes."""

from functools import wraps


def restore_integer_cpml_profile(monkeypatch, axes=("x", "y", "z")):
    """Reuse the electric integer-node profiles on the selected magnetic faces.

    The original initializer and every profile call still run. Selecting axes
    supports the NU face-attribution measurements without duplicating grading code.
    """
    from rfx.boundaries import cpml

    if not set(axes) <= {"x", "y", "z"}:
        raise ValueError(f"invalid CPML axes: {axes}")
    original = cpml.init_cpml

    @wraps(original)
    def integer_profile(*args, **kwargs):
        params, state = original(*args, **kwargs)
        magnetic = getattr(params, "magnetic", None)
        if magnetic is not None:
            faces = {f"{axis}_{side}": getattr(params, f"{axis}_{side}")
                     for axis in axes for side in ("lo", "hi")}
            params = params._replace(magnetic=magnetic._replace(**faces))
        return params, state

    monkeypatch.setattr(cpml, "init_cpml", integer_profile)
