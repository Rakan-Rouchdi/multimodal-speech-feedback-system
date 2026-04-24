from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


def infer_dataset_name(result: Dict) -> str:
    input_file = Path(result.get("meta", {}).get("input_file", ""))
    if input_file.parent.name:
        return input_file.parent.name
    return "unsorted"


def save_result_json(
    result: Dict,
    base_outputs_dir: str = "outputs",
    dataset: Optional[str] = None,
) -> str:
    session_id = result["meta"]["session_id"]
    variant = result["meta"].get("pipeline_variant", "unknown")
    dataset_name = dataset or infer_dataset_name(result)

    variant_map = {
        "multimodal": "M",
        "speech_only": "S",
        "text_only": "T",
    }
    variant_code = variant_map.get(variant, "U")  # U = unknown

    output_dir = Path(base_outputs_dir) / dataset_name / "raw" / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{session_id}_{variant_code}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return str(path)
