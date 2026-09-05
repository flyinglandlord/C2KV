from unittest import TestCase

from c2kv.config import ExperimentConfig
from c2kv.megatron.arguments import build_ms_swift_arguments


def _values_after(arguments, flag, count):
    start = arguments.index(flag) + 1
    return arguments[start : start + count]


class MegatronArgumentsTest(TestCase):
    def test_dataset_paths_are_available_during_ms_swift_validation(self):
        config = ExperimentConfig.from_dict(
            {
                "model": {"name_or_path": "model"},
                "data": {
                    "train": ["train-a.jsonl", "train-b.jsonl"],
                    "validation": ["validation.jsonl"],
                },
            }
        )

        arguments = build_ms_swift_arguments(config)

        self.assertEqual(
            _values_after(arguments, "--dataset", 2),
            ["train-a.jsonl", "train-b.jsonl"],
        )
        self.assertEqual(
            _values_after(arguments, "--val_dataset", 1),
            ["validation.jsonl"],
        )
        self.assertEqual(
            _values_after(arguments, "--split_dataset_ratio", 1),
            ["0.0"],
        )

    def test_empty_validation_does_not_emit_a_bare_list_argument(self):
        config = ExperimentConfig.from_dict(
            {
                "model": {"name_or_path": "model"},
                "data": {"train": ["train.jsonl"]},
            }
        )

        arguments = build_ms_swift_arguments(config)

        self.assertNotIn("--val_dataset", arguments)
