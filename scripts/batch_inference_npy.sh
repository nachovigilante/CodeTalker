#!/usr/bin/env bash
#
# Batch npy-only inference for CodeTalker.
# Loads model + wav2vec2 + processor ONCE, then iterates over a directory of wavs.
#
set -euo pipefail

AUDIO_DIR=""
CONFIG="config/vocaset/demo.yaml"
CONDITION=""
SUBJECT=""
OUTPUT_DIR=""

usage() {
    cat <<EOF
Usage: $(basename "$0") -d <audio_dir> [options]

Required:
  -d, --audio-dir DIR     Directory containing .wav files

Options:
  -o, --output-dir DIR    Output directory (default: from config demo_npy_save_folder)
  -c, --config PATH       Config YAML (default: $CONFIG)
  --condition NAME        Speaker condition override (default: from config)
  --subject NAME          Subject template override (default: from config)
  --skip-existing         Accepted for backward compatibility — already-done files
                          are detected automatically and skipped.
  -h, --help              Show this help

Output layout:
  <output_dir>/<wav_stem>/condition_<condition>_subject_<subject>.npy
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--audio-dir)  AUDIO_DIR="$2"; shift 2 ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -c|--config)     CONFIG="$2"; shift 2 ;;
        --condition)     CONDITION="$2"; shift 2 ;;
        --subject)       SUBJECT="$2"; shift 2 ;;
        --skip-existing) shift ;;
        -h|--help)       usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$AUDIO_DIR" ]]; then
    echo "Error: --audio-dir (-d) is required."
    usage
fi
if [[ ! -d "$AUDIO_DIR" ]]; then
    echo "Error: Audio directory '$AUDIO_DIR' does not exist."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH=./

CMD=(.venv/bin/python main/batch_inference.py --config "$CONFIG" demo_wav_dir_path "$AUDIO_DIR")
[[ -n "$OUTPUT_DIR" ]] && CMD+=(demo_npy_save_folder "$OUTPUT_DIR")
[[ -n "$CONDITION"  ]] && CMD+=(condition "$CONDITION")
[[ -n "$SUBJECT"    ]] && CMD+=(subject "$SUBJECT")

START=$(date +%s)
"${CMD[@]}"
ELAPSED=$(($(date +%s) - START))
printf "\nWall-clock: %02d:%02d:%02d\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
