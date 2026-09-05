"""Translate the repository's stable JSON schema to ms-swift arguments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List

from c2kv.config import ExperimentConfig


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _append(arguments: List[str], name: str, value: Any) -> None:
    arguments.append(f"--{name}")
    if isinstance(value, (list, tuple)):
        arguments.extend(_stringify(item) for item in value)
    else:
        arguments.append(_stringify(value))


def build_ms_swift_arguments(config: ExperimentConfig) -> List[str]:
    training = config.training
    parallel = config.parallel
    plugin_path = Path(__file__).with_name("plugin.py").resolve()
    arguments: List[str] = []
    values = {
        "model": config.model.name_or_path,
        "external_plugins": str(plugin_path),
        "output_dir": training.output_dir,
        "torch_dtype": config.model.dtype,
        "tuner_type": "full",
        "finetune": True,
        "padding_free": False,
        "packing": False,
        "attention_backend": "unfused",
        "micro_batch_size": training.micro_batch_size,
        "global_batch_size": training.global_batch_size,
        "train_iters": training.max_steps,
        "max_length": training.sequence_length,
        "lr": training.learning_rate,
        "min_lr": training.min_learning_rate,
        "lr_warmup_fraction": training.warmup_fraction,
        "save_steps": training.save_steps,
        "eval_steps": training.eval_steps,
        "logging_steps": training.log_steps,
        "seed": training.seed,
        "data_seed": config.data.shuffle_seed,
        "dataloader_num_workers": config.data.num_workers,
        "tensor_model_parallel_size": parallel.tensor,
        "pipeline_model_parallel_size": parallel.pipeline,
        "context_parallel_size": parallel.context,
        "expert_model_parallel_size": parallel.expert,
        "sequence_parallel": parallel.sequence_parallel,
        "save_safetensors": True,
        "report_to": ["wandb", "tensorboard"],
        "wandb_project": training.wandb_project,
    }
    # MegatronSftArguments validates the framework-facing dataset fields during
    # construction, before C2KVMegatronSft can replace loading with our lazy
    # C2KVFileDataset implementation.  Pass the real source paths through so
    # that ms-swift's early validation and our data layer share one source of
    # truth.  Splitting stays owned by C2KV, hence the explicit zero ratio.
    if config.data.train:
        values["dataset"] = list(config.data.train)
    if config.data.validation:
        values["val_dataset"] = list(config.data.validation)
    values["split_dataset_ratio"] = 0.0
    if training.wandb_run_name:
        values["wandb_exp_name"] = training.wandb_run_name
    if training.gradient_checkpointing:
        values.update(
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
    for name, value in values.items():
        _append(arguments, name, value)
    for name, value in training.extra_megatron_args.items():
        _append(arguments, name, value)
    return arguments


def build_export_arguments(
    config: ExperimentConfig,
    checkpoint_dir: str,
    output_dir: str,
    test_precision: bool = False,
) -> List[str]:
    plugin_path = Path(__file__).with_name("plugin.py").resolve()
    values = {
        "model": config.model.name_or_path,
        "mcore_model": str(Path(checkpoint_dir).expanduser().resolve()),
        "output_dir": str(Path(output_dir).expanduser().resolve()),
        "external_plugins": str(plugin_path),
        "to_hf": True,
        "exist_ok": True,
        "test_convert_precision": test_precision,
        "tensor_model_parallel_size": config.parallel.tensor,
        "pipeline_model_parallel_size": config.parallel.pipeline,
        "context_parallel_size": config.parallel.context,
        "sequence_parallel": config.parallel.sequence_parallel,
    }
    arguments: List[str] = []
    for name, value in values.items():
        _append(arguments, name, value)
    return arguments
