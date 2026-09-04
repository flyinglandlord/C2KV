import json
import tempfile
from pathlib import Path
from unittest import TestCase

from c2kv.config import ExperimentConfig, load_experiment_config


class ConfigTest(TestCase):
    def test_minimal_config(self):
        config = ExperimentConfig.from_dict({"model": {"name_or_path": "Qwen/Qwen3-0.6B"}})
        self.assertEqual(config.c2kv.compression_ratios, (4, 8, 16))
        self.assertEqual(config.parallel.tensor, 1)

    def test_json_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"model": {"name_or_path": "model"}}), encoding="utf-8")
            self.assertEqual(load_experiment_config(path).model.name_or_path, "model")

    def test_context_parallel_is_explicitly_gated(self):
        with self.assertRaisesRegex(ValueError, "context parallelism"):
            ExperimentConfig.from_dict(
                {"model": {"name_or_path": "model"}, "parallel": {"context": 2}}
            )

    def test_dense_mask_limit_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "dense reference mask limit"):
            ExperimentConfig.from_dict(
                {
                    "model": {"name_or_path": "model"},
                    "c2kv": {"dense_mask_max_tokens": 128},
                    "training": {"sequence_length": 256},
                }
            )

    def test_distillation_requires_language_model_tokens(self):
        with self.assertRaisesRegex(ValueError, "self-distillation requires"):
            ExperimentConfig.from_dict(
                {
                    "model": {"name_or_path": "model"},
                    "objectives": {
                        "language_model": 0.0,
                        "reconstruction": 1.0,
                        "self_distillation": 0.5,
                    },
                }
            )
