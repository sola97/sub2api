# DeepSeek Remote Compaction Bridge

This fork adds a gateway-side implementation of Codex remote compaction v2 for
accounts that route GPT model names to DeepSeek. It applies to native DeepSeek
accounts, direct `api.deepseek.com` accounts, and OpenAI-platform accounts whose
configured model mapping targets `deepseek-*` through a compatible proxy.

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
