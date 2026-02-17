from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def save_result_json(result: Dict, outputs_dir: str = "outputs") -> str:
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)
    session_id = result["meta"]["session_id"]
    path = Path(outputs_dir) / f"{session_id}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return str(path)
