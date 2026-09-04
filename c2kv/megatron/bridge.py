"""MCore Bridge conversion for the additional paper-named Q/K/V weights."""

from __future__ import annotations

from types import SimpleNamespace

from mcore_bridge.bridge import GPTBridge


class C2KVBridge(GPTBridge):
    """Add `gist_{q,k,v}_proj` keys without changing base-model conversion."""

    hf_special_embeddings_key = "model.gist_embed_tokens.weight"

    def _set_word_embeddings(self, mg_model, hf_state_dict, to_mcore):
        super()._set_word_embeddings(mg_model, hf_state_dict, to_mcore)
        language_model = getattr(mg_model, "language_model", mg_model)
        if not to_mcore or self.hf_special_embeddings_key in hf_state_dict:
            self._set_state_dict(
                language_model,
                "c2kv_special_embeddings",
                hf_state_dict,
                self.hf_special_embeddings_key,
                to_mcore,
            )

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        converted = super()._set_layer_attn(mg_layer, hf_state_dict, layer_idx, to_mcore)
        mg_attention = None if mg_layer is None else mg_layer.self_attention
        memory_projection = None if mg_attention is None else mg_attention.c2kv_linear_qkv
        proxy = None if memory_projection is None else SimpleNamespace(linear_qkv=memory_projection)
        prefix = f"{self.hf_attn_prefix}.gist_"

        if to_mcore:
            memory_state = {
                key[len(prefix) :]: value for key, value in hf_state_dict.items() if key.startswith(prefix)
            }
            if memory_state:
                self._set_qkv(proxy, memory_state, True, layer_idx=layer_idx)
            return converted

        memory_state = self._set_qkv(proxy, {}, False, layer_idx=layer_idx)
        converted.update({f"{prefix}{key}": value for key, value in memory_state.items()})
        return converted
