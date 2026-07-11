# rfx Studio operations and lab-server guide

## Local release

The supported default is single-user and loopback-only:

```bash
python -m pip install "rfx-fdtd[studio]"
rfx studio --workspace .rfx-studio
```

The wheel contains the compiled frontend. No Node.js runtime is required after
installation. `RFX_TELEMETRY` defaults to `off`; `on` permits only a short
allowlist of operational fields and rejects spec, prompt, artifact, path,
secret, token, and key fields.

## Engineering review workflow

Use the Studio views as a review sequence before treating a run as evidence:

1. **Design** previews geometry and blocks invalid drafts at preflight.
2. **Setup** reviews the canonical domain and grid estimate, materials and
   geometry assignments, ports/excitations, boundary conditions, requested
   observations, frequency sweep, and CPU solve contract.
3. **Spec & Code** shows the editable canonical JSON, deterministic generated
   Python, and the semantic proposal against the immutable revision.
4. **Results** shows lifecycle events and the RF evidence summary before the
   individual S11, Smith, and field plots.

The RF evidence summary derives S11 minimum, VSWR, sampled -10 dB bandwidth,
and sweep resolution only from the checksummed run artifact. It also cites the
spec, compiled-model, runtime/package, and artifact identities. Solver-native
convergence history, mesh statistics, or port diagnostics are labeled **not
captured** when absent; a successful lifecycle does not imply those checks
passed. The bundled patch workflow remains a structural CPU smoke test, not a
calibrated quantitative RF reference.

## Schema, backup, restore, and replay compatibility

```bash
rfx workspace migrate --workspace .rfx-studio
rfx workspace backup --workspace .rfx-studio -o backup.rfx-backup.zip
rfx workspace restore backup.rfx-backup.zip --workspace .rfx-studio
```

Migration is additive and idempotent. Back up before upgrading. Restore verifies
every checksum and swaps the workspace atomically. Backups include the SQLite
online-backup image plus `runs/` and `artifacts/`; files with credential-like
names are excluded. Down-migrations are not attempted because dropping columns
or tables would make rollback lossy.

`rfx-replay-bundle/v1` readers must continue verifying v1 manifests for the
1.x release line. A future incompatible writer uses a new schema value and a
new reader; it must not silently reinterpret v1. Reproducibility means declared
metrics within the manifest tolerances, not bit-identical fields.

## Design Copilot provider

Studio selects the OpenAI Responses API when `OPENAI_API_KEY` is present. The
key stays in the process environment and is not written to SQLite, generated
code, audit events, artifacts, telemetry, or the browser. The default model is
`gpt-5.5` for broad API-project availability; deployments with access to a newer
model can select it explicitly:

```bash
export OPENAI_API_KEY=...
export RFX_OPENAI_MODEL=gpt-5.6
rfx studio --workspace .rfx-studio
```

The provider request uses the Responses API with a strict JSON schema and
`store=false`. Studio supplies only the exact current spec plus a bounded,
structured summary of the selected immutable run. Model output is limited to a
small JSON Patch; protected identity, schema, workflow, and CPU backend paths
cannot be changed. Studio applies the patch in memory, compiles it, runs
preflight, and estimates CPU cost before showing it. No revision or run is
created until the user approves the corresponding Studio action.

If no API key is present, Studio labels the provider as `Offline rules` and
supports only a narrow deterministic starter/parameter flow. To force that mode
for demos or air-gapped QA:

```bash
RFX_COPILOT_PROVIDER=local rfx studio --workspace .rfx-studio
```

`RFX_COPILOT_PROVIDER=openai` fails at startup without a key. A configured key
can still receive a provider error when its API project lacks model access or
quota; Studio reports that error and does not silently present a rules-based
result as an LLM answer.

## Optional authenticated lab server

Remote bind is refused unless all three gates are supplied:

1. a 32+ character token in a mode-0600 file;
2. one or more exact HTTPS Origins;
3. `--tls-terminated`, confirming an HTTPS reverse proxy is in front.

```bash
umask 077
openssl rand -hex 32 > /secure/rfx-studio.token

rfx studio \
  --host 0.0.0.0 \
  --port 8765 \
  --workspace /srv/rfx-studio \
  --auth-token-file /secure/rfx-studio.token \
  --allowed-origin https://rfx.lab.example \
  --tls-terminated \
  --no-browser
```

Terminate TLS in an established reverse proxy. Forward only to
`127.0.0.1:8765` or an isolated container network, preserve the original Host,
set `X-Forwarded-Proto: https`, cap request bodies, and apply network access
control. Example Nginx location:

```nginx
location / {
    proxy_pass http://127.0.0.1:8765;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto https;
    client_max_body_size 2m;
    proxy_read_timeout 3600s;
}
```

API and MCP requests accept `Authorization: Bearer <token>`. Browser users may
use HTTP Basic username `rfx` with the token as the password; the token is never
stored in an ExperimentSpec or artifact. State-changing requests reject Origins
outside the exact allowlist, and MCP separately validates Host and Origin to
prevent DNS rebinding.

## Multi-user adapter boundary

The optional lab adapter hashes user ids into separate workspace and secret
roots, applies per-user cell/run/artifact quotas, namespaces object storage, and
assigns scheduler resource names per user. SQLite/filesystem remain the shipped
single-node adapters. `PostgresMetadataConfig` validates and redacts a future
Postgres DSN; `ObjectStoreAdapter` and `SchedulerAdapter` define the replacement
boundaries without making an unavailable service a runtime dependency.

Do not treat this adapter as a public multi-tenant SaaS authorization system.
Deploy behind organization authentication, map the authenticated principal to
one tenant workspace, and test backup/restore and user isolation before adding
workers.
