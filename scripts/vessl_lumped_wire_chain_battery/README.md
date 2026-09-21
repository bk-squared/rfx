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
