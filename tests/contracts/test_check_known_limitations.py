"""Exercise the real CLI and workflow boundary on offline synthetic inputs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/ci/check_known_limitations.py'


def run_check(tmp_path, body, closing='7', opened='', closed='', sweep=False):
    page = tmp_path / 'limitations.md'
    page.write_text('# Known limitations\n' + body)
    summary = tmp_path / 'summary.md'
    env = {**os.environ, 'CLOSING_ISSUES': closing, 'OPEN_ISSUES': opened,
           'CLOSED_ISSUES': closed, 'GITHUB_STEP_SUMMARY': str(summary)}
    result = subprocess.run([sys.executable, str(SCRIPT), '--path', str(page),
                             *(['--sweep'] if sweep else [])],
                            env=env, capture_output=True, text=True)
    assert summary.read_text().strip() == result.stdout.strip()
    return result


def test_closing_present_entry_fails(tmp_path):
    result = run_check(tmp_path, '## A\nTracker: #7\n')
    assert result.returncode == 1
    assert 'A: PR closes [7]' in result.stdout


@pytest.mark.parametrize('remaining', ['', '## Other\nTracker: #8\n'])
def test_deleted_entry_passes(tmp_path, remaining):
    assert run_check(tmp_path, remaining).returncode == 0


@pytest.mark.parametrize('refs', ['#7 #8', '#7, #8', '#8, #7'])
def test_another_explicitly_open_tracker_passes(tmp_path, refs):
    assert run_check(tmp_path, f'## A\nTracker: {refs}\n', opened='8').returncode == 0


@pytest.mark.parametrize('opened', ['', '9', '7'])
def test_unknown_or_closing_tracker_is_not_an_open_escape(tmp_path, opened):
    assert run_check(tmp_path, '## A\nTracker: #7 #8\n', opened=opened).returncode == 1


def test_multi_issue_line_second_closing_reference_fails(tmp_path):
    assert run_check(tmp_path, '## A\nTracker: #8, #7\n').returncode == 1


def test_all_trackers_closing_fails_even_if_currently_open(tmp_path):
    assert run_check(tmp_path, '## A\nTracker: #7 #8\n', closing='7\n8',
                     opened='7 8').returncode == 1


def test_accepted_entries_do_not_fail_on_historical_citations(tmp_path):
    body = '## A\nHistory #7\nTracker: none — accepted limitation (refused at default: tests/synthetic.py::test_refuses)\n'
    assert run_check(tmp_path, body).returncode == 0


def test_a_citation_in_the_body_is_not_a_tracker(tmp_path):
    assert run_check(tmp_path, '## A\nHistory #7\nTracker: #8\n').returncode == 0


def test_sweep_reports_every_closed_tracker_with_open_sibling(tmp_path):
    result = run_check(tmp_path, '## A\nTracker: #7 #8 #9\n', opened='8',
                       closed='7 9', sweep=True)
    assert result.returncode == 1
    assert 'closed trackers [7, 9]' in result.stdout


def test_sweep_with_only_open_trackers_passes(tmp_path):
    assert run_check(tmp_path, '## A\nTracker: #7\n', opened='7', sweep=True).returncode == 0


def test_missing_tracker_fails_cli(tmp_path):
    assert run_check(tmp_path, '## A\nMissing\n').returncode == 1


def workflow():
    return yaml.safe_load((ROOT / '.github/workflows/pr-body.yml').read_text())


@pytest.mark.parametrize('owner,name,expected', [
    ('elsewhere', 'rfx', 0), ('bk-squared', 'other', 0), ('bk-squared', 'rfx', 1),
])
def test_workflow_filters_closing_references_to_this_repository(tmp_path, owner, name, expected):
    """Run the actual workflow shell and its jq expression, with a fake gh read."""
    job = workflow()['jobs']['pr-body-contract']
    step = next(s for s in job['steps'] if s['name'] == 'Known limitations closed by this PR')
    assert step['if'] == "github.event_name != 'merge_group'"
    assert workflow()['permissions']['issues'] == 'read'
    assert step['env']['GH_TOKEN'] == '${{ github.token }}'
    reader = next(s for s in job['steps'] if 'closingIssuesReferences' in s.get('run', ''))
    assert reader['if'] == "github.event_name != 'merge_group'"
    assert job['steps'].index(reader) < job['steps'].index(step)
    assert sum('closingIssuesReferences' in s.get('run', '') for s in job['steps']) == 1
    assert not step.get('continue-on-error', False)
    assert not job.get('if')
    # PyYAML's YAML 1.1 loader spells GitHub's 'on' key as True.
    assert 'edited' in workflow()[True]['pull_request']['types']
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    gh = bindir / 'gh'
    gh.write_text('#!' + sys.executable + '\n' + '''import json, os, subprocess, sys
args = sys.argv[1:]
if args[:2] == ['label', 'list']:
    payload = [{'name': 'lane:ci-infra'}]
elif args[:3] == ['pr', 'view', '42']:
    payload = (json.loads(os.environ['FAKE_CLOSING_JSON'])
               if args[args.index('--json') + 1] == 'closingIssuesReferences'
               else {'labels': []})
elif args[:2] == ['issue', 'view']:
    payload = {'labels': []}
else:
    raise AssertionError(args)
if '--jq' in args:
    sys.exit(subprocess.run(['jq', '-r', args[args.index('--jq') + 1]],
        input=json.dumps(payload), text=True).returncode)
print(json.dumps(payload))
''')
    gh.chmod(0o755)
    # The workflow's python command executes the real script against a tiny page.
    wrapper = bindir / 'python'
    wrapper.write_text('#!/bin/sh\nexec "$KL_PYTHON" "$KL_SCRIPT" --path "$KL_PAGE"\n')
    wrapper.chmod(0o755)
    page = tmp_path / 'page.md'
    page.write_text('# Page\n## A\nTracker: #7\n')
    env = {**os.environ, 'PATH': str(bindir) + os.pathsep + os.environ['PATH'],
           'KL_PYTHON': sys.executable, 'KL_SCRIPT': str(SCRIPT), 'KL_PAGE': str(page),
           'OPEN_ISSUES': '', 'GITHUB_REPOSITORY': 'bk-squared/rfx', 'PR_NUMBER': '42',
           'GITHUB_ENV': str(tmp_path / 'github-env'),
           'GITHUB_STEP_SUMMARY': str(tmp_path / 'summary.md'),
           'FAKE_CLOSING_JSON': json.dumps({'closingIssuesReferences': [
               {'number': 7, 'repository': {'owner': {'login': owner}, 'name': name}}]})}
    read = subprocess.run(['bash', '-e', '-c', reader['run']], env=env,
                          capture_output=True, text=True)
    assert read.returncode == 0, read.stdout + read.stderr
    # Apply the real reader's GITHUB_ENV output as Actions does between steps.
    lines = iter(Path(env['GITHUB_ENV']).read_text().splitlines())
    for line in lines:
        key, delimiter = line.split('<<', 1)
        value = []
        for item in lines:
            if item == delimiter:
                break
            value.append(item)
        env[key] = '\n'.join(value)
    assert 'CLOSING_ISSUES' in env
    result = subprocess.run(['bash', '-e', '-c', step['run']], env=env,
                            capture_output=True, text=True)
    assert result.returncode == expected, result.stdout + result.stderr


def test_merge_group_skips_and_body_edits_do_not_restart_physics():
    steps = workflow()['jobs']['pr-body-contract']['steps']
    skip = next(s for s in steps if s['name'] == 'PR gate already passed before queue admission')
    assert skip['if'] == "github.event_name == 'merge_group'"
    assert 'PR-level' in skip['run']
    heavy_path = ROOT / '.github/workflows/pr-tests.yml'
    heavy = yaml.safe_load(heavy_path.read_text())
    assert heavy[True]['pull_request'] is None  # origin/main's default trigger list
    assert 'scripts/ci/check_known_limitations.py' not in heavy_path.read_text()
    assert 'edited' in workflow()[True]['pull_request']['types']
    weekly = yaml.safe_load((ROOT / '.github/workflows/governance-audit.yml').read_text())
    step = next(s for s in weekly['jobs']['audit']['steps'] if '--sweep' in s.get('run', ''))
    assert '!cancelled()' in step['if']
    assert 'continue-on-error' not in step


def test_sweep_missing_state_fails_closed(tmp_path):
    assert run_check(tmp_path, '## A\nTracker: #7\n', sweep=True).returncode == 1


@pytest.mark.parametrize('state,expected', [('OPEN', 0), ('CLOSED', 1), ('MERGED', 1), ('UNKNOWN', 1)])
def test_live_state_lookup_controls_the_open_escape(tmp_path, state, expected):
    page = tmp_path / 'page.md'
    page.write_text('## A\nTracker: #7 #8\n')
    gh = tmp_path / 'gh'
    gh.write_text('#!' + sys.executable + '\n' +
                  'import sys\nassert sys.argv[1:] == ["issue", "view", "8", "--repo", '
                  '"bk-squared/rfx", "--json", "state", "--jq", ".state"]\n'
                  + f'print({state!r})\n')
    gh.chmod(0o755)
    env = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
           'CLOSING_ISSUES': '7', 'GITHUB_REPOSITORY': 'bk-squared/rfx'}
    env.pop('OPEN_ISSUES', None)
    result = subprocess.run([sys.executable, str(SCRIPT), '--path', str(page)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == expected, result.stdout + result.stderr


def test_state_lookup_failure_is_not_a_green_sweep(tmp_path):
    page = tmp_path / 'page.md'
    page.write_text('## A\nTracker: #7\n')
    gh = tmp_path / 'gh'
    gh.write_text('#!/bin/sh\nexit 1\n')
    gh.chmod(0o755)
    env = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH']}
    env.pop('OPEN_ISSUES', None)
    result = subprocess.run([sys.executable, str(SCRIPT), '--path', str(page), '--sweep'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert 'Cannot complete tracker check' in result.stdout


def test_pending_new_issue_fails_cli(tmp_path):
    result = run_check(tmp_path, '## A\nTracker: TODO-NEW-ISSUE (silent gradient error)\n')
    assert result.returncode == 1
    assert 'leader must open a tracker' in result.stdout
