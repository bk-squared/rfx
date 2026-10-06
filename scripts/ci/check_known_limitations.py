#!/usr/bin/env python3
"""Keep known-limitations trackers live at PR closure and in the weekly audit.

Stdlib only. CLOSING_ISSUES contains whitespace-separated local issue numbers.
OPEN_ISSUES, when set (including empty), is an explicit offline state snapshot;
otherwise read the tracked issues/PRs with gh. Unknown states never count as open.
The weekly --sweep reports every closed/merged tracker, even in multi-issue entries.
Each subject is a leaf Markdown heading; parent headings group subjects. The
introductory H1 and the final 'What this page is not' section are not entries.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import NamedTuple

PAGE = Path(__file__).resolve().parents[2] / 'docs/guides/known_limitations.md'


class Entry(NamedTuple):
    title: str
    body: str


def entries(text: str) -> list[Entry]:
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    # Code examples cannot create subjects or satisfy the Tracker contract.
    text = re.sub(r'^(`{3,}|~{3,}).*?^\1\s*$', '', text, flags=re.M | re.S)
    headings = [(m.start(), m.end(), len(m[1]), m[2].strip()) for m in
                re.finditer(r'^(#{2,6})\s+(.+)$', text, re.M)]
    # Also recognize the page's former bold-subject format, so adding an entry
    # in that style cannot silently evade the contract.
    headings += [(m.start(), m.end(), 7, ' '.join(m[1].split())) for m in
                 re.finditer(r'^(?:- )?\*\*([^*]+?)\*\*', text, re.M)]
    headings.sort()
    result = []
    for i, (_, body_start, level, title) in enumerate(headings):
        if title == 'What this page is not':
            break
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        # A heading with child headings is a section, not a subject.
        if i + 1 < len(headings) and headings[i + 1][2] > level:
            continue
        result.append(Entry(title, text[body_start:end]))
    return result


def trackers(entry: Entry) -> set[int]:
    lines = re.findall(r'^Tracker:\s*([^\n]+)$', entry.body, re.M)
    if len(lines) != 1:
        raise ValueError(f'{entry.title}: expected exactly one Tracker line')
    value = lines[0].strip()
    if value.startswith('TODO-NEW-ISSUE'):
        raise ValueError(f'{entry.title}: leader must open a tracker: {value}')
    accepted = re.fullmatch(
        r'none — accepted limitation \((?:refused at default|warns at default|diagnostic, does not enter the result): '
        r'tests/[^\s():]+\.py::(?:[A-Za-z_]\w*::)*test_\w+\)', value,
    )
    if accepted:
        return set()
    if not re.fullmatch(r'#\d+(?:(?:\s*,\s*|\s+)#\d+)*', value):
        raise ValueError(f'{entry.title}: invalid Tracker: {value}')
    return {int(n) for n in re.findall(r'#(\d+)', value)}


def numbers(value: str) -> set[int]:
    return {int(n) for n in value.split()}


def state(number: int) -> str:
    args = ['gh', 'issue', 'view', str(number)]
    if os.environ.get('GITHUB_REPOSITORY'):
        args += ['--repo', os.environ['GITHUB_REPOSITORY']]
    return subprocess.run(
        args + ['--json', 'state', '--jq', '.state'],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()


def check(text: str, closing: set[int], *, sweep: bool = False) -> list[str]:
    subjects = entries(text)
    failures = []
    indexed = []
    for entry in subjects:
        try:
            indexed.append((entry, trackers(entry)))
        except ValueError as exc:
            failures.append(str(exc))
    wanted = {n for _, refs in indexed for n in refs} if sweep else {
        n for _, refs in indexed if refs & closing for n in refs - closing
    }
    if 'OPEN_ISSUES' in os.environ:
        opened = numbers(os.environ['OPEN_ISSUES'])
        # For offline sweeps, CLOSED_ISSUES is explicit; absent != closed.
        closed = numbers(os.environ.get('CLOSED_ISSUES', ''))
        if sweep:
            failures.extend(f'#{n}: missing offline state'
                            for n in sorted(wanted - opened - closed))
    else:
        states = {n: state(n) for n in sorted(wanted)}
        opened = {n for n, value in states.items() if value == 'OPEN'}
        closed = {n for n, value in states.items() if value in {'CLOSED', 'MERGED'}}
        for n, value in states.items():
            if value not in {'OPEN', 'CLOSED', 'MERGED'}:
                failures.append(f'#{n}: unknown state {value!r}')
    for entry, refs in indexed:
        if sweep:
            stale = refs & closed
            if stale:
                failures.append(f'{entry.title}: closed trackers {sorted(stale)}')
        elif refs & closing and not (refs - closing) & opened:
            failures.append(
                f'{entry.title}: PR closes {sorted(refs & closing)} but entry remains; '
                'delete/rewrite it or retain another verified OPEN tracker.'
            )
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, default=PAGE)
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args(argv)
    try:
        failures = check(args.path.read_text(encoding='utf-8'),
                         numbers(os.environ.get('CLOSING_ISSUES', '')), sweep=args.sweep)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        failures = [f'Cannot complete tracker check: {exc}']
    report = '## Known limitations tracker audit\n\n' + (
        '\n'.join(f'- {failure}' for failure in failures) if failures
        else 'No stale trackers found.'
    ) + '\n'
    print(report)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as handle:
            handle.write(report)
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
