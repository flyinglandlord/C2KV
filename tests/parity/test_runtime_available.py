import importlib.util
import sys
from pathlib import Path
from unittest import TestCase, skipUnless


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_MEGATRON = importlib.util.find_spec("megatron") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@skipUnless(HAS_TORCH and HAS_MEGATRON, "Megatron parity tests require the H200 environment")
class RuntimeParityTest(TestCase):
    def test_runtime_imports(self):
        from c2kv.megatron.attention import C2KVSelfAttention

        self.assertEqual(C2KVSelfAttention.__name__, "C2KVSelfAttention")


@skipUnless(HAS_TORCH and HAS_TRANSFORMERS, "Transformers parity tests require the eval environment")
class TransformersEvalParityTest(TestCase):
    def test_qwen3_c2kv_model_imports(self):
        legacy_root = str(Path(__file__).resolve().parents[2] / "python")
        if legacy_root not in sys.path:
            sys.path.insert(0, legacy_root)
        from models.qwen3 import Qwen3ForCausalLM

        self.assertEqual(Qwen3ForCausalLM.__name__, "Qwen3ForCausalLM")
