from __future__ import annotations

import os
import tempfile
from pathlib import Path


_TMP_ROOT = Path(tempfile.gettempdir()) / "speech_feedback_dissertation"
_TMP_ROOT.mkdir(parents=True, exist_ok=True)

# Keep third-party caches out of read-only package locations. This avoids
# import-time failures from librosa/numba and matplotlib in local runs/tests.
os.environ.setdefault("NUMBA_CACHE_DIR", str(_TMP_ROOT / "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_TMP_ROOT / "matplotlib"))
