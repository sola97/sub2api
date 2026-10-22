# DeepSeek Remote Compaction Bridge

This fork adds a gateway-side implementation of Codex remote compaction v2 for
accounts that route GPT model names to DeepSeek. It applies to native DeepSeek
accounts, direct `api.deepseek.com` accounts, and OpenAI-platform accounts whose
configured model mapping targets `deepseek-*` through a compatible proxy.

The bridge is opt-in per model. It currently handles `gpt-5.5` and
`gpt-5.6-sol` only when the account maps that exact model name to
`deepseek-flash`. Other GPT-5.6 variants, including `gpt-5.6`, `terra`, and
`luna`, are not bridged. Keep them out of additional account allowlists if they
must not be served from those sources.

When a request ends with a final `compaction_trigger`, the gateway asks the
upstream for a plain summary without tools, then returns exactly one
`compaction` output item. The summary is authenticated-encrypted using the
deployment's persistent `JWT_SECRET`. On the next turn, the gateway restores
this checkpoint as a user message before forwarding to DeepSeek, so the upstream
never receives an unknown `type=compaction` item.

The workflow `.github/workflows/deepseek-compact-image.yml` runs the focused
regression tests and publishes an amd64 image:

```
ghcr.io/sola97/sub2api:deepseek-compact-<commit>
```

Keep `JWT_SECRET` stable across container restarts and deployments. If it
changes, checkpoints issued with the old secret become invalid.

## Image-containing histories

Image preprocessing is explicitly configured and disabled by default:

```yaml
gateway:
  deepseek_compact_vision_account_id: 0
  deepseek_compact_vision_model: ""
```

The corresponding environment variables are
`GATEWAY_DEEPSEEK_COMPACT_VISION_ACCOUNT_ID` and
`GATEWAY_DEEPSEEK_COMPACT_VISION_MODEL`. Choose an active, schedulable OpenAI
API-key account whose Responses model can actually interpret images. The route
must not map back to DeepSeek. No account or model is inferred from the client.

The gateway describes each distinct image using this account, then sends a
text-only history copy to DeepSeek. The authenticated encrypted capsule keeps
the stable image ID and its description, but not the original image payload or
URL. Continuation restores a deterministic image-description appendix even if
the summary omitted the image. Subsequent compaction carries the image records
forward without interpreting the same image again. Source requests and image
URLs are not overwritten, and a stored description cannot fetch or replay the
original image.

This does not add a client image retrieval API or an agent tool. Remote image
URLs retain their original lifetime; the gateway does not make them permanent.
Images are not silently discarded when unsupported, unreadable or oversized.
Only eight distinct images are accepted per checkpoint. Descriptions are bounded
to 8 KiB each, and an individual image source is limited to 32 MiB. The existing
256 KiB encoded capsule limit remains in force. Assets which cannot fit even in a
minimal checkpoint fail before model calls; the actual summary and descriptions
are checked again before issuing the capsule. Large inline screenshots are
described normally because their bytes are not persisted in the capsule.
File-ID-only inputs and unsupported image-generation records fail explicitly.

Vision requests have a 60-second timeout, a bounded response body and no automatic
retry. Refusal, incomplete responses and empty descriptions fail the compaction;
the gateway never issues a success capsule for those failures. The explicit
vision route can consume upstream quota. Auxiliary vision usage is not added to
DeepSeek tokens or billed as DeepSeek usage; separate end-user billing for this
auxiliary operation is not implemented by this change.

Text capsules remain payload version 1. New image capsules use payload version 3,
which stores bounded references. Version 2 capsules from the previous release
remain readable and continue through the legacy original-image path until they
are replaced. Old code rejects version 3 capsules, so rolling back the binary
does not allow newly compacted image sessions to continue until upgraded again.
Keep `JWT_SECRET` unchanged.

Deployment is deferred while the current user session depends on this gateway.
