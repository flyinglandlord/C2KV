# C2KV: Concatenable and Compressible KV Cache

C2KV compresses independently prefetched documents into concatenable memory KV
entries. The `megatron` branch uses Megatron-SWIFT for training and keeps
Transformers only for evaluation compatibility.

## Direct-file workflow

This repository is not installed as a Python package. There is no `pyproject.toml`,
editable install, Hydra, or YAML dependency. Run files from the repository root and
keep environment versions outside the source tree.

```bash
python check_environment.py --config configs/train/qwen3_0_6b_smoke.json
python run_tests.py --suite unit

torchrun --nproc_per_node=1 train_megatron.py \
  --config configs/train/qwen3_0_6b_smoke.json

torchrun --nproc_per_node=2 train_megatron.py \
  --config configs/train/qwen3_4b_mdoc.json
```

Export a native MCore checkpoint and evaluate it with the preserved Transformers
pipeline:

```bash
torchrun --nproc_per_node=2 export_megatron.py \
  --config configs/train/qwen3_4b_mdoc.json \
  --checkpoint outputs/qwen3-4b-c2kv/checkpoint-10000 \
  --output outputs/qwen3-4b-c2kv-hf

python evaluate.py --config configs/eval/qwen3_4b_hotpotqa.json
```

Use `--dry-run` on training, export, or evaluation to validate the resolved command
without importing a GPU runtime.

## Architecture

The framework-neutral core compiles each example into one structured forward pass:

1. System tokens attend causally within the system prompt.
2. Each document attends to the system and its own causal document history, never to
   another document.
3. Each C2KV memory slot attends to the system, the document sink, its source chunk,
   configured overlap, and prior memory slots from the same document.
4. Query and response tokens attend to the system, all memory slots, and their own
   causal history, but not the raw documents.
5. Reconstruction tokens attend to the matching document's memory slots and their
   own causal reconstruction history.

This removes the old Trainer-owned multi-forward `DynamicCache` path and makes the
training semantics compatible with Megatron pipeline scheduling. Logical RoPE
positions preserve independent document prefill while repositioning memory slots as
if the documents had been concatenated.

```text
train_megatron.py            stable training entry
export_megatron.py           MCore -> Transformers export
evaluate.py                  Transformers evaluation entry
check_environment.py         runtime audit
run_tests.py                 unittest runner

c2kv/core/                   paper semantics: layout, masks, positions, losses
c2kv/data/                   normalized schema, adapters, mixtures, collator
c2kv/megatron/               MCore Bridge plugin, attention, trainer, checkpoint
c2kv/transformers/           isolated legacy-evaluation compatibility
c2kv/eval/                   evaluation backend and output interfaces
configs/                     JSON-only experiment configuration
tests/                       unit, parity, and distributed integration gates
python/                      frozen legacy implementation during parity migration
```

## Data contract

JSON and JSONL files are read directly. The preferred record is:

```json
{
  "id": "sample-1",
  "system": "Answer from the documents.",
  "documents": ["document one", "document two"],
  "question": "the question",
  "answer": "the answer"
}
```

The normalizer also accepts `query`/`response`, `instruction`/`output`, standard
`messages`, and a `text` field in pretraining mode. New datasets are added through
`c2kv.data.registry.register_adapter`; trainers do not contain dataset names or
storage paths.

## Checkpoints and extension boundaries

Native MCore checkpoints are authoritative for resume. HF export retains paper-
compatible `gist_q_proj`, `gist_k_proj`, and `gist_v_proj` names and writes
`c2kv_config.json`. The base-model QKV and token table remain frozen; the additional
fused C2KV QKV projection plus the paper's memory/reconstruction embeddings are
optimized. `mean` and `embedding_mean` residuals use the exact compression-chunk
indices recorded by the layout compiler instead of inferring them from padding.

JSONL inputs are indexed by byte offset and decoded on demand. Large corpora therefore
do not need to be loaded into Python memory before Megatron workers start.

The first correctness backend intentionally materializes a dense arbitrary mask and
is capped by `dense_mask_max_tokens`. The sparse range API in
`c2kv/core/masking.py` is the stable boundary for a block-sparse H200 kernel. Do not
raise the dense cap for large-scale pretraining; implement and gate the sparse
backend instead.

Architecture-specific behavior is registered in
`c2kv/megatron/architectures.py`. Adding a larger dense or MoE family should provide
an architecture spec and, only when needed, a specialized MCore loader/bridge. Core
layout, data, and evaluation code should remain unchanged.

See `docs/migration/C2KV_MEGATRON_HANDOFF_ZH.md` for the migration gates, known
boundaries, server evidence, and continuation checklist.
