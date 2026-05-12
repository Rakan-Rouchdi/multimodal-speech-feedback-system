# Cold-Start Latency Notes

The cold-start benchmark was run with transcription-result caching disabled by passing `use_transcription_cache=False` to the pipeline.
The existing `outputs/cache/crisperwhisper_transcriptions.json` cache was not used for transcription lookups and was not intentionally written by this generator.

## Runtime Scope

- Completed benchmark rows: 60.
- Attempted benchmark rows: 60.
- One-time transcriber model loading time: 4.0276 seconds.
- Transcriber model reused in-process after loading: yes.
- Existing transcription cache modification time unchanged: yes.

The main latency summary reports per-recording runtime after any separately measured one-time transcriber model load.
For text-enabled variants, this means transcription is uncached, but the loaded model object is reused in-process unless noted otherwise.

## Environment

- Platform: `macOS-26.3-arm64-arm-64bit-Mach-O`.
- Python: `3.13.0`.
- CPU count: `12`.
- faster-whisper: `1.2.1`.
- tensorflow: `2.21.0`.
- librosa: `0.11.0`.

## Coverage

- Variants completed: Multimodal, Speech-only, Text-only.
