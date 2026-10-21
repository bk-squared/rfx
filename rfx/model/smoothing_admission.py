"""Admission for the single-device full permittivity tensor."""

FULL_TENSOR_MESSAGE = (
    "subpixel_smoothing='full_tensor' is supported on a single-device Yee mesh only; "
    "True gives the diagonal (first-order) smoothing on this path"
)


def admit_full_tensor(sim, value, *, distributed=False, devices=None, unsupported=False):
    """Refuse a consumer that cannot carry every node-tensor coupling."""
    if value != "full_tensor":
        return
    if (unsupported or distributed or (devices is not None and len(devices) > 1)
            or sim._solver != "yee"
            or sim._refinement is not None or sim._boundary == "upml"):
        raise NotImplementedError(FULL_TENSOR_MESSAGE)
