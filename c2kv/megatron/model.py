"""ms-swift pipeline with C2KV-owned data and trainer hooks."""

from __future__ import annotations

from swift.megatron.pipelines.train.sft import MegatronSft

from c2kv.config import ExperimentConfig
from c2kv.data import C2KVCollator, C2KVFileDataset, MixedDataset

from .trainer import C2KVMegatronTrainer
from .runtime import configure_runtime


class C2KVMegatronSft(MegatronSft):
    def __init__(self, args, experiment_config: ExperimentConfig):
        self.experiment_config = experiment_config
        super().__init__(args)

    @property
    def tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    def _build_dataset(self, paths):
        if not paths:
            return None
        datasets = [C2KVFileDataset(path, self.tokenizer, self.experiment_config) for path in paths]
        return datasets[0] if len(datasets) == 1 else MixedDataset(datasets)

    def _prepare_dataset(self):
        data = self.experiment_config.data
        train_dataset = self._build_dataset(data.train)
        validation_dataset = self._build_dataset(data.validation)
        if train_dataset is None:
            raise ValueError("data.train must contain at least one JSON/JSONL path")
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        self.template.data_collator = C2KVCollator(
            pad_token_id,
            self.experiment_config.training.sequence_length,
            self.experiment_config.c2kv.dense_mask_max_tokens,
            include_teacher=bool(self.experiment_config.objectives.self_distillation),
        )
        return train_dataset, validation_dataset

    def prepare_trainer(self):
        return C2KVMegatronTrainer(self.args, self.template, self.experiment_config)


def run_c2kv_megatron(arguments, experiment_config: ExperimentConfig):
    configure_runtime(residual_type=experiment_config.c2kv.residual_type)
    return C2KVMegatronSft(arguments, experiment_config).main()
