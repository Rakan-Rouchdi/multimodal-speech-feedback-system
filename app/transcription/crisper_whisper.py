from __future__ import annotations

import contextlib
import re
import wave
from typing import Dict, List, Optional

from faster_whisper import WhisperModel

from app.transcription.types import TranscriptionResult


# ── CrisperWhisper model config ──────────────────────────────────────────────
MODEL_ID = "nyrahealth/faster_CrisperWhisper"
COMPUTE_TYPE = "int8"
CPU_THREADS = 10


def clean_transcript(raw: str) -> str:
    """Normalise CrisperWhisper raw transcript for NLP metrics."""
    # CrisperWhisper often uses commas as word separators in disfluent speech.
    text = raw.replace(",", " ")
    # Lowercase bracket tokens [UH] -> uh, [UM] -> um
    text = re.sub(r'\[([A-Z]+)\]', lambda m: m.group(1).lower(), text)
    # Split compact sentence-like ASR output such as Word.Word.Word.
    text = re.sub(r"(?<=[A-Za-z])\.(?=[A-Z])", " ", text)
    # Collapse multiple spaces, strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─────────────────────────────────────────────────────────────────────────────


class CrisperWhisperTranscriber:
    """
    Local transcription using faster_CrisperWhisper (CTranslate2 int8).

    Key behaviour:
    - Uses only the CrisperWhisper model which transcribes disfluencies as [UH], [UM], etc.
    - Produces word-level timestamps for richer analysis.
    """

    def __init__(
        self,
        device: str = "cpu",
        compute_type: str = COMPUTE_TYPE,
        cpu_threads: int = CPU_THREADS,
    ):
        self.model = WhisperModel(
            MODEL_ID,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=2,
        )

    @staticmethod
    def _wav_duration(audio_path: str) -> Optional[float]:
        if not audio_path.lower().endswith(".wav"):
            return None
        try:
            with contextlib.closing(wave.open(audio_path)) as wav_file:
                return wav_file.getnframes() / wav_file.getframerate()
        except Exception:
            return None

    @staticmethod
    def _looks_incomplete(result: TranscriptionResult, duration_sec: Optional[float]) -> bool:
        if duration_sec is None or duration_sec < 20:
            return False
        if len(result.words) < 20:
            return True
        return False

    def _transcribe_once(
        self,
        audio_path: str,
        language: str,
        *,
        vad_filter: bool,
    ) -> TranscriptionResult:
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        seg_list = []
        transcript_parts = []
        word_list = []

        for seg in segments:
            seg_text = seg.text.strip()
            if seg_text:
                transcript_parts.append(seg_text)

            seg_list.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg_text,
            })

            if seg.words:
                for w in seg.words:
                    clean = w.word.lstrip(" ,")
                    if clean:
                        word_list.append({
                            "text": clean,
                            "start_s": round(w.start, 3),
                            "end_s": round(w.end, 3),
                        })

        transcript = " ".join(t for t in transcript_parts if t)
        language_detected = getattr(info, "language", None)

        return TranscriptionResult(
            transcript=transcript,
            language=language_detected,
            segments=seg_list,
            clean_text=clean_transcript(transcript),
            words=word_list,
        )

    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
    ) -> TranscriptionResult:
        """
        Transcribe an audio file and return a TranscriptionResult.

        CrisperWhisper captures disfluencies ([UH], [UM]) and provides
        word-level timestamps for downstream speech and text analysis.
        """
        result = self._transcribe_once(audio_path, language, vad_filter=True)
        duration_sec = self._wav_duration(audio_path)
        if self._looks_incomplete(result, duration_sec):
            result = self._transcribe_once(audio_path, language, vad_filter=False)
        return result
