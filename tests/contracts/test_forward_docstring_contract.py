"""Star import binds the public names declared in rfx.__all__.
"""


import rfx


def test_star_import_binds_all_and_is_importable():
    ns: dict = {}
    exec("from rfx import *", ns)
    star_names = {n for n in ns if not n.startswith("__")}
    assert star_names == set(rfx.__all__), (
        "from rfx import * did not bind exactly __all__; "
        f"symmetric_difference={star_names ^ set(rfx.__all__)}"
    )
    for name in rfx.__all__:
        assert ns[name] is getattr(rfx, name)
