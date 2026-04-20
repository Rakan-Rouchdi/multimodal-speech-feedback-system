#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_transcription.sh  –  Run CrisperWhisper batch transcription
#
# Uses faster_CrisperWhisper (CTranslate2/int8) — much faster than the
# HuggingFace transformers version on CPU.
#
# Usage:
#   ./run_transcription.sh <your_hf_token>
#
# Prerequisites:
#   1. Accept the model license at:
#      https://huggingface.co/nyrahealth/faster_CrisperWhisper
#   2. Get your HuggingFace token at:
#      https://huggingface.co/settings/tokens
# ─────────────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$1" ] && [ -z "$HF_TOKEN" ]; then
  echo "Usage: ./run_transcription.sh <your_hf_token>"
  echo "Or:    export HF_TOKEN=hf_xxx && ./run_transcription.sh"
  echo ""
  echo "Accept license: https://huggingface.co/nyrahealth/faster_CrisperWhisper"
  exit 1
fi

export HF_TOKEN="${1:-$HF_TOKEN}"

echo "============================================================"
echo " CrisperWhisper Batch Transcription (faster-whisper/int8)"
echo "============================================================"
echo ""

conda run -n crisperWhisper python "$SCRIPT_DIR/transcribe_batch.py"
