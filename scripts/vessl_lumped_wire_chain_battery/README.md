# Job specifications for the lumped / wire chain battery

One file per stage group of
`scripts/diagnostics/lumped_wire_chain_battery_measure.py`. Every file pins
`RFX_SHA` to the commit it ran against with no fallback and aborts at job start
if the worktree has moved or is dirty, so a record can always name the tree that
produced it.

The whole campaign is small — the largest case is 1,230 grid nodes for 8,393
steps — so the cost is job startup, not the solve. That is why the stages are
grouped into a few jobs that loop inside one process rather than one job per
case, and why `cpu-32-mem-64` is enough.

The driver writes each stage JSON straight into the run directory on the shared
volume and persists after every case, so a job killed half way leaves the cases
it finished. The EXIT trap copies the job log beside them and opens the
permissions; it is the second line of defence, not the first.

`--kind lumped` is the only difference between these files and the lumped leg's;
no lumped job has been submitted.

## Check the run block parses before submitting

A job specification is a YAML file whose `run:` value is a shell script, and
only the YAML half gets checked by anything. Two of this battery's jobs died on
the other half: one on `cp -a` racing a `.pyc`, and one on an unterminated
quoted string left behind when an edit replaced the first line of a multi-line
construct and not the rest — the YAML parsed perfectly and the shell did not.

Before submitting, extract the block and check it with both shells:

    python -c "import yaml,sys; sys.stdout.write(yaml.safe_load(open(sys.argv[1]))['run'])" \
        scripts/vessl_lumped_wire_chain_battery/<file>.yaml > /tmp/runblock.sh
    sh -n /tmp/runblock.sh && bash -n /tmp/runblock.sh

## What the contract suite cannot check on this image

`ghcr.io/bk-squared/rfx-openems:5b423bdfe0c8` carries **git version 2.25.1**,
measured by the job itself on runs 369367262739 and 369367262746; an
`apt-get install git` in the job left it at 2.25.1. Forty-four contract tests
build a temporary repository with `git init -q -b main`, and `-b` arrived in
git 2.28, so on this image they stop at

    subprocess.CalledProcessError: Command '['git', 'init', '-q', '-b', 'main']'
    returned non-zero exit status 129

That is 2 failures and 42 errors in `tests/contracts`, all of them in
`test_changelog_fragments`, `test_prune_worktrees` and
`test_ci_workflows_contract`. The same 44 pass on a host with a newer git,
where the suite is 3473 passed and nothing fails. Until the image's git moves,
a contract run on this lane is 3429 passed with 44 tests that did not run —
which is not the same thing as a pass, and is why the job prints the version.
