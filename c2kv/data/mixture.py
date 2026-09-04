"""Lazy, file-backed datasets with deterministic epoch-aware ratios."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import List, Sequence

from c2kv.config import ExperimentConfig
from c2kv.core.compression import CompressionSchedule
from c2kv.core.layout import LayoutCompiler
from c2kv.core.types import TokenizedExample

from .adapters import JsonRecordStore
from .registry import get_adapter


class C2KVFileDataset:
    def __init__(self, path: str, tokenizer, config: ExperimentConfig, adapter: str = "default"):
        self.path = str(Path(path).expanduser().resolve())
        self.tokenizer = tokenizer
        self.config = config
        self.records = JsonRecordStore(self.path)
        self.adapter = get_adapter(adapter)
        self.epoch = 0
        self.schedule = CompressionSchedule(
            config.c2kv.compression_ratios,
            config.c2kv.compression_schedule,
            config.training.seed,
        )
        memory_token_id = config.c2kv.memory_token_id
        reconstruction_token_id = config.c2kv.reconstruction_token_id
        if memory_token_id < 0:
            memory_token_id = tokenizer.eos_token_id
        if reconstruction_token_id < 0:
            reconstruction_token_id = memory_token_id
        self.compiler = LayoutCompiler(
            memory_token_id=memory_token_id,
            reconstruction_token_id=reconstruction_token_id,
            max_memory_slots=config.c2kv.max_memory_slots,
            sink_tokens=config.c2kv.sink_tokens,
            overlap_tokens=config.c2kv.overlap_tokens,
        )

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _encode(self, text: str, limit: int) -> tuple[int, ...]:
        if not text or limit == 0:
            return ()
        return tuple(self.tokenizer.encode(text, add_special_tokens=False)[:limit])

    def _encode_role(
        self,
        text: str,
        role: str,
        limit: int,
        add_generation_prompt: bool = False,
        keep_bos: bool = False,
    ) -> tuple[int, ...]:
        if not text or limit == 0:
            return ()
        try:
            encoded = self.tokenizer.apply_chat_template(
                [{"role": role, "content": text}],
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
            if hasattr(encoded, "input_ids"):
                encoded = encoded.input_ids
            elif isinstance(encoded, dict):
                encoded = encoded["input_ids"]
            tokens = list(encoded)
        except (AttributeError, TypeError):
            tokens = list(self.tokenizer.encode(text, add_special_tokens=False))
        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if not keep_bos and tokens and tokens[0] == bos_token_id:
            tokens = tokens[1:]
        return tuple(tokens[:limit])

    def _pretrain_example(self, record, index: int) -> TokenizedExample:
        text = str(record.get("text", ""))
        limit = self.config.data.max_document_tokens + self.config.data.max_query_tokens
        tokens = self._encode(text, limit)
        if len(tokens) < 2:
            raise ValueError(f"pretrain record {index} must contain at least two tokens")
        split = min(self.config.data.max_document_tokens, len(tokens) - 1)
        response = list(tokens[split:])
        if self.tokenizer.eos_token_id is not None and response[-1] != self.tokenizer.eos_token_id:
            response.append(self.tokenizer.eos_token_id)
        return TokenizedExample(
            sample_id=str(record.get("sample_id", record.get("id", f"{Path(self.path).name}:{index}"))),
            system_tokens=(),
            document_tokens=(tuple(tokens[:split]),),
            query_tokens=(),
            response_tokens=tuple(response),
            reconstruction_document_ids=(0,) if self.config.objectives.reconstruction else (),
        )

    def __getitem__(self, index: int):
        record = self.records[index]
        if self.config.data.mode == "pretrain":
            if "text" not in record:
                raise ValueError(f"pretrain record {index} requires a text field")
            tokenized = self._pretrain_example(record, index)
            ratio = self.schedule.ratio_for(tokenized.sample_id, self.epoch)
            return self.compiler.compile(tokenized, ratio)
        example = self.adapter(record, f"{Path(self.path).name}:{index}")
        data = self.config.data
        documents = tuple(
            self._encode_role(document, "user", data.max_document_tokens)
            for document in example.documents[: data.max_documents]
        )
        response = list(self._encode(example.response, data.max_query_tokens))
        if response and self.tokenizer.eos_token_id is not None and response[-1] != self.tokenizer.eos_token_id:
            response.append(self.tokenizer.eos_token_id)
        reconstruction_ids = tuple(range(len(documents))) if self.config.objectives.reconstruction else ()
        tokenized = TokenizedExample(
            sample_id=example.sample_id,
            system_tokens=self._encode_role(
                example.system, "system", data.max_system_tokens, keep_bos=True
            ),
            document_tokens=documents,
            query_tokens=self._encode_role(
                example.query,
                "user",
                data.max_query_tokens,
                add_generation_prompt=True,
            ),
            response_tokens=tuple(response),
            reconstruction_document_ids=reconstruction_ids,
        )
        ratio = self.schedule.ratio_for(example.sample_id, self.epoch)
        return self.compiler.compile(tokenized, ratio)


class MixedDataset:
    """Concatenate datasets while forwarding epoch changes to each source."""

    def __init__(self, datasets: Sequence[C2KVFileDataset]):
        if not datasets:
            raise ValueError("at least one dataset is required")
        self.datasets = list(datasets)
        self.ends: List[int] = []
        size = 0
        for dataset in datasets:
            size += len(dataset)
            self.ends.append(size)

    def __len__(self) -> int:
        return self.ends[-1]

    def set_epoch(self, epoch: int) -> None:
        for dataset in self.datasets:
            dataset.set_epoch(epoch)

    def __getitem__(self, index: int):
        dataset_index = bisect_right(self.ends, index)
        start = 0 if dataset_index == 0 else self.ends[dataset_index - 1]
        return self.datasets[dataset_index][index - start]
