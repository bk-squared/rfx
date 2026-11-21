"""Opt-in per-file counts and unrun GPU node IDs for the round-2 named jobs."""

from collections import defaultdict
import json

import pytest


_counts = defaultdict(lambda: dict(passed=0, failed=0, skipped=0, deselected=0, errors=0, gpu_not_run=[]))
_outcomes = {}


def pytest_deselected(items):
    for item in items:
        row = _counts[item.nodeid.split('::')[0]]
        row['deselected'] += 1
        if item.get_closest_marker('gpu'):
            row['gpu_not_run'].append(item.nodeid)


def pytest_collection_finish(session):
    for item in session.items:
        _counts[item.nodeid.split('::')[0]]


def pytest_collectreport(report):
    if report.failed:
        _counts[report.nodeid]['errors'] += 1


def pytest_runtest_logreport(report):
    if report.failed:
        _outcomes[report.nodeid] = 'failed' if report.when == 'call' else 'errors'
    elif report.skipped and report.nodeid not in _outcomes:
        _outcomes[report.nodeid] = 'skipped'
    elif report.when == 'call' and report.passed:
        _outcomes.setdefault(report.nodeid, 'passed')


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    for node, outcome in _outcomes.items():
        _counts[node.split('::')[0]][outcome] += 1
    for path, counts in sorted(_counts.items()):
        print('ROUND2_FILE ' + json.dumps(dict(file=path, **counts)), flush=True)
