import os
from pathlib import Path
from unittest import TestCase, skipUnless


SMOKE_OUTPUT = os.environ.get("C2KV_SMOKE_OUTPUT")


@skipUnless(SMOKE_OUTPUT, "set C2KV_SMOKE_OUTPUT after a distributed smoke run")
class SmokeArtifactTest(TestCase):
    def _checkpoint(self) -> Path:
        checkpoints = sorted(Path(SMOKE_OUTPUT).glob("checkpoint-*"))
        self.assertTrue(checkpoints)
        return checkpoints[-1]

    def test_training_artifacts_exist(self):
        output = Path(SMOKE_OUTPUT)
        self.assertTrue((output / "logging.jsonl").exists())
        checkpoint = self._checkpoint()
        self.assertTrue((checkpoint / "iter_0000002").is_dir())
        self.assertTrue((checkpoint / "c2kv_config.json").is_file())

    def test_transformers_checkpoint_contains_c2kv_weights(self):
        from safetensors import safe_open

        model_path = self._checkpoint() / "model.safetensors"
        self.assertTrue(model_path.is_file())
        with safe_open(model_path, framework="pt") as weights:
            names = set(weights.keys())
        self.assertIn("model.gist_embed_tokens.weight", names)
        self.assertTrue(any(name.endswith("gist_q_proj.weight") for name in names))
        self.assertTrue(any(name.endswith("gist_k_proj.weight") for name in names))
        self.assertTrue(any(name.endswith("gist_v_proj.weight") for name in names))
