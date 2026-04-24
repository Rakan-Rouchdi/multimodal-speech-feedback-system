from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TranscriptionResult:
    transcript: str
    language: Optional[str]
    segments: List[dict]
    clean_text: str = ""
    words: List[Dict] = field(default_factory=list)
