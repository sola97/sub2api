# Native compaction routing isolation

Initial `/responses/compact` requests preserve the account's ordinary upstream
model mapping. Only an explicit account `compact_model_mapping` overrides it.
The global `gateway.openai_compact_model` (environment variable
`GATEWAY_OPENAI_COMPACT_MODEL`) is an error-triggered fallback, not an initial
model override. It remains available for recognized model-availability and
context-window errors, with at most one retry.

If the input already contains a `compaction` or `compaction_summary` with
nonempty `encrypted_content`, automatic cross-model retry is disabled. Native
encrypted checkpoints are opaque and cannot be assumed portable between models
or providers. Do not drop them to make the request appear successful. Explicit
account mappings remain administrator-controlled and must point to a compatible
upstream. This change does not provide cross-account checkpoint affinity.

## 2026-10-21 incident

The native GPT-6.1 Sol probe was routed through account 4 (`sora-cpa`), but the
global compact override changed its upstream model to GPT-5.5. The saved upstream
error was an EOF from the internal DeepSeek CPA `/v1/responses/compact` endpoint.
CPA returned 500 and sub2api exhausted eligible account failover, returning 502.
The global override existed before the image-compaction deployment.

Production was rolled back to immutable image
`ghcr.io/sola97/sub2api:deepseek-compact-88ebb4d1e0f5c707c71135eb76553cc13f81f890`.
This patch is not deployed. The old global compact setting remains on production,
so rollback alone does not fix native compaction routing. A direct native compact
diagnostic returned 404; upstream native endpoint availability still requires
separate verification before any new production deployment.

A separate GPT-5.6 Sol / DeepSeek continuation failed locally with
`invalid DeepSeek compact encrypted_content`. No request replay body is retained
in the production ops schema. The exact failed envelope is unavailable; do not
attribute it to key rotation, image validation, or corruption without evidence.
Image capsule v2 must retain authentication and v1 backward readability. The
rolled-back v1-only build cannot read v2 image capsules; rollback does not recover
conversations already holding those newer capsules. Such conversations require a
compatible build before continuation, not removal of their encrypted context.

Validation uses local stub upstreams, not production customer conversations.
