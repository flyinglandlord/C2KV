"""C2KV checkpoint metadata written beside native MCore and HF weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from c2kv.config import ExperimentConfig


FORMAT_VERSION = 1


def write_c2kv_manifest(output_dir: str, config: ExperimentConfig) -> Path:
    path = Path(output_dir) / "c2kv_config.json"
    payload: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "method": "C2KV",
        "projection_names": ["gist_q_proj", "gist_k_proj", "gist_v_proj"],
        "config": config.to_dict(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    hf_config_path = Path(output_dir) / "config.json"
    if hf_config_path.exists():
        with hf_config_path.open("r", encoding="utf-8") as handle:
            hf_config = json.load(handle)
        ratios = config.c2kv.compression_ratios
        gist_type = f"interleave-{ratios[0]}" if len(ratios) == 1 else "dynamic-interleave"
        hf_config.update(
            gist_type=gist_type,
            gist_param=config.c2kv.train_projections,
            gist_residual_type=config.c2kv.residual_type.replace("embedding_mean", "embed-mean"),
            gist_overlap=config.c2kv.overlap_tokens,
            gist_extra_embed_num=2,
        )
        if config.c2kv.memory_token_id >= 0:
            hf_config["gist_token_id"] = config.c2kv.memory_token_id
        with hf_config_path.open("w", encoding="utf-8") as handle:
            json.dump(hf_config, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return path
