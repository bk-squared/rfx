"""Runner for the vector-pad diagnosis (``_vector_pad_diag``); never collected by CI.

Skipped unless ``RFX_VPAD_DIAG=1``. Writes one JSON document to stderr between
the markers ``VPAD_DIAG_BEGIN`` and ``VPAD_DIAG_END``. Asserts nothing.
"""

import json
import os
import sys

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('RFX_VPAD_DIAG') != '1',
                                reason='diagnosis only: set RFX_VPAD_DIAG=1')


def test_vector_pad_diagnosis():
    from tests.unit.runners._vector_pad_diag import diagnose
    report = diagnose()
    sys.stderr.write('VPAD_DIAG_BEGIN\n' + json.dumps(report) + '\nVPAD_DIAG_END\n')
