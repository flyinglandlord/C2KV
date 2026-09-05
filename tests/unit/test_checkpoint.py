import json
import tempfile
from pathlib import Path
from unittest import TestCase

from c2kv.config import ExperimentConfig
from c2kv.megatron.checkpoint import write_c2kv_manifest


class CheckpointManifestTest(TestCase):
    def test_default_memory_token_uses_exported_eos_token(self):
        config = ExperimentConfig.from_dict({"model": {"name_or_path": "model"}})
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps({"eos_token_id": 42}), encoding="utf-8")

            write_c2kv_manifest(directory, config)

            exported_config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(exported_config["gist_token_id"], 42)
