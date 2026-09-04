"""Typed JSON configuration for direct-file C2KV runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str
    architecture: str = "auto"
    dtype: str = "bfloat16"
    trust_remote_code: bool = True


@dataclass(frozen=True)
class C2KVConfig:
    memory_token_id: int = -1
    reconstruction_token_id: int = -1
    compression_ratios: Tuple[int, ...] = (4, 8, 16)
    compression_schedule: str = "deterministic"
    max_memory_slots: int = 512
    sink_tokens: int = 0
    overlap_tokens: int = 0
    residual_type: str = "none"
    train_projections: str = "qkv"
    attention_backend: str = "dense_reference"
    dense_mask_max_tokens: int = 4096


@dataclass(frozen=True)
class DataConfig:
    mode: str = "multi_document"
    train: Tuple[str, ...] = ()
    validation: Tuple[str, ...] = ()
    max_document_tokens: int = 1024
    max_documents: int = 10
    max_query_tokens: int = 1024
    max_system_tokens: int = 256
    shuffle_seed: int = 42
    num_workers: int = 4


@dataclass(frozen=True)
class ObjectiveConfig:
    language_model: float = 1.0
    reconstruction: float = 0.0
    qkv_regularization: float = 0.0
    self_distillation: float = 0.0
    distillation_temperature: float = 1.0


@dataclass(frozen=True)
class ParallelConfig:
    tensor: int = 1
    pipeline: int = 1
    context: int = 1
    expert: int = 1
    sequence_parallel: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str = "outputs/c2kv-megatron"
    micro_batch_size: int = 1
    global_batch_size: int = 8
    max_steps: int = 1000
    sequence_length: int = 4096
    learning_rate: float = 2e-5
    min_learning_rate: float = 2e-6
    warmup_fraction: float = 0.05
    save_steps: int = 500
    eval_steps: int = 500
    log_steps: int = 10
    seed: int = 42
    gradient_checkpointing: bool = True
    wandb_project: str = "c2kv-megatron"
    wandb_run_name: str = ""
    extra_megatron_args: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    c2kv: C2KVConfig = field(default_factory=C2KVConfig)
    data: DataConfig = field(default_factory=DataConfig)
    objectives: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ExperimentConfig":
        sections = dict(raw)
        if "model" not in sections:
            raise ValueError("config section 'model' is required")
        c2kv = dict(sections.get("c2kv", {}))
        data = dict(sections.get("data", {}))
        c2kv["compression_ratios"] = tuple(c2kv.get("compression_ratios", (4, 8, 16)))
        data["train"] = tuple(data.get("train", ()))
        data["validation"] = tuple(data.get("validation", ()))
        config = cls(
            model=ModelConfig(**sections["model"]),
            c2kv=C2KVConfig(**c2kv),
            data=DataConfig(**data),
            objectives=ObjectiveConfig(**sections.get("objectives", {})),
            parallel=ParallelConfig(**sections.get("parallel", {})),
            training=TrainingConfig(**sections.get("training", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        errors: List[str] = []
        if not self.model.name_or_path:
            errors.append("model.name_or_path must not be empty")
        if self.model.architecture not in {"auto", "llama", "qwen2", "qwen3"}:
            errors.append("model.architecture must be auto, llama, qwen2, or qwen3")
        if not self.c2kv.compression_ratios or any(r <= 0 for r in self.c2kv.compression_ratios):
            errors.append("c2kv.compression_ratios must contain positive integers")
        if self.c2kv.compression_schedule not in {"fixed", "deterministic"}:
            errors.append("c2kv.compression_schedule must be fixed or deterministic")
        if self.c2kv.residual_type not in {"none", "mean", "embedding_mean"}:
            errors.append("c2kv.residual_type must be none, mean, or embedding_mean")
        if self.c2kv.residual_type != "none" and self.parallel.sequence_parallel:
            errors.append("C2KV residuals are not yet correctness-gated with sequence parallelism")
        if set(self.c2kv.train_projections.lower()) - set("qkv"):
            errors.append("c2kv.train_projections may contain only q, k, and v")
        if self.data.mode not in {"pretrain", "sft", "multi_document", "compress_history"}:
            errors.append("data.mode is not supported")
        if self.parallel.context != 1:
            errors.append("context parallelism is reserved but not yet correctness-gated")
        if self.training.global_batch_size < self.training.micro_batch_size:
            errors.append("training.global_batch_size must be >= micro_batch_size")
        if self.objectives.self_distillation and self.parallel.pipeline != 1:
            errors.append("self-distillation is not yet correctness-gated with pipeline parallelism")
        if self.objectives.self_distillation and self.objectives.language_model == 0:
            errors.append("self-distillation requires a positive language-model objective")
        if not 0.0 <= self.objectives.self_distillation <= 1.0:
            errors.append("objectives.self_distillation must be between 0 and 1")
        if self.objectives.distillation_temperature <= 0:
            errors.append("objectives.distillation_temperature must be positive")
        if self.objectives.language_model < 0 or self.objectives.reconstruction < 0:
            errors.append("language-model and reconstruction weights must be non-negative")
        if self.objectives.language_model == 0 and self.objectives.reconstruction == 0:
            errors.append("at least one token objective must have a positive weight")
        if self.objectives.qkv_regularization and self.parallel.pipeline != 1:
            errors.append("QKV regularization is not yet correctness-gated with pipeline parallelism")
        if self.training.extra_megatron_args.get("mtp_num_layers"):
            errors.append("MTP is not yet correctness-gated with the C2KV structured mask")
        if self.c2kv.attention_backend != "dense_reference":
            errors.append("only the dense_reference attention backend is currently correctness-gated")
        if self.training.sequence_length > self.c2kv.dense_mask_max_tokens:
            errors.append("training.sequence_length exceeds the dense reference mask limit")
        if errors:
            raise ValueError("invalid C2KV config:\n- " + "\n- ".join(errors))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"expected a JSON object in {config_path}")
    return ExperimentConfig.from_dict(raw)
