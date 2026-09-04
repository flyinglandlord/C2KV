# Legacy compatibility boundary

The original implementation remains under `python/` until the Megatron parity gates
are complete. New training work must not add features there. The only supported new
imports are through `c2kv/transformers/`, which keeps the Transformers evaluation
pipeline available without coupling it to Megatron training.

Removal criteria are documented in `docs/migration/C2KV_MEGATRON_HANDOFF_ZH.md`.
