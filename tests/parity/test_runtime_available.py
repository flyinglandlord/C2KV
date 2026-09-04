import importlib.util
from unittest import TestCase, skipUnless


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_MEGATRON = importlib.util.find_spec("megatron") is not None


@skipUnless(HAS_TORCH and HAS_MEGATRON, "Megatron parity tests require the H200 environment")
class RuntimeParityTest(TestCase):
    def test_runtime_imports(self):
        from c2kv.megatron.attention import C2KVSelfAttention

        self.assertEqual(C2KVSelfAttention.__name__, "C2KVSelfAttention")
