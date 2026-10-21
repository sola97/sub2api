# Native compaction routing isolation implementation plan

> Execution: main agent only, per user instruction. Production was rolled back to 88ebb4d1; do not deploy this candidate.

**Goal:** Preserve the requested native compact model on the initial request, while keeping explicit account compact mappings and bounded error-triggered global fallback.

**Architecture:** Separate explicit account compact mapping from global error fallback. Initial HTTP forward and passthrough consult only account mappings. Global fallback remains available after a recognized model/context failure, but cannot change models when an existing opaque encrypted compaction checkpoint would cross model boundaries.

**Tech Stack:** Existing Go service and test transport; no dependencies, migrations, or production changes.

**Spec:** User requests in this conversation: observe gpt-6.1-sol compact, roll back production, continue code fixes without deployment.

## Global constraints

- Keep original capsule authentication and v1/v2 compatibility; never discard a failed checkpoint to simulate success.
- Production remains on 88ebb4d1. No new image deployment, restart, or config modification after rollback.
- Preserve explicit account compact_model_mapping and normal model_mapping behavior.
- No sub-agents. Main-agent tests and self-review are not independent review.
- The gpt-5.6-sol invalid capsule has no retained request body (migration 136 removed request replay). Its exact cause remains unproven; do not change cryptographic rules on speculation.

## Task 1: Isolate initial mapping from global fallback

Files: backend/internal/service/openai_compact_fallback.go, openai_gateway_forward.go, openai_gateway_passthrough.go, openai_compact_fallback_test.go.

Interfaces: add resolveOpenAIInitialCompactModel(account *Account, requestedModel string) string, returning only an explicit eligible account mapping. Existing resolveOpenAICompactFallbackModel retains account-before-global retry semantics.

- [x] Add an API-key regression with requested gpt-6.1-sol and global gpt-5.5, covering both passthrough modes and asserting upstream model stays gpt-6.1-sol and native opaque capsule is unchanged.
- [x] Replace the prior eager-global test expectation with preservation of the initial requested model. Run tests to observe the regression failing on baseline.
- [x] Extract account-only resolution, call it on initial requests, and keep the existing retry resolver for recognized errors.
- [x] Block retry model switching when input includes an opaque encrypted compaction or compaction_summary; add tests asserting original request bytes and checkpoint remain unchanged.
- [x] Run compact/mapping/DeepSeek/image/retirement regressions and race checks; complete full service suite and vet. Commit/push status recorded in task delivery.

## Task 2: Document actual boundaries

Files: docs/operations/native-compact-routing.md.

- [x] Document initial account mapping versus error fallback, encrypted checkpoint restrictions, production rollback image, and the unresolved gpt-5.6-sol sample.
- [x] Verify public production health read-only after rollback. Do not send further billable production compaction diagnostics during local development.

Acceptance: gpt-6.1-sol remains native despite global gpt-5.5, explicit per-account mapping still works, error-triggered fallback works without encrypted checkpoints, and no ciphertext is dropped or exposed. Passing local tests does not certify upstream native compact availability.
