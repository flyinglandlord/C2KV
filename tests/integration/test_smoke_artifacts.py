import os
from pathlib import Path
from unittest import TestCase, skipUnless


SMOKE_OUTPUT = os.environ.get("C2KV_SMOKE_OUTPUT")


@skipUnless(SMOKE_OUTPUT, "set C2KV_SMOKE_OUTPUT after a distributed smoke run")
class SmokeArtifactTest(TestCase):
    def test_training_artifacts_exist(self):
        output = Path(SMOKE_OUTPUT)
        self.assertTrue((output / "logging.jsonl").exists())
        self.assertTrue(any(output.glob("checkpoint-*")))
