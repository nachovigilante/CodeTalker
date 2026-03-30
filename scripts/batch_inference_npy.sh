#!/usr/bin/env bash
#
# Batch npy-only inference script for CodeTalker.
# Runs demo.py on every audio file in a given directory,
# producing only .npy vertex files (no video rendering).
#
set -euo pipefail

# ──────────────────────────────────────────────
# Defaults (override via flags)
# ──────────────────────────────────────────────
AUDIO_DIR=""
CONFIG="config/vocaset/demo.yaml"
CONDITION=""
SUBJECT=""
SKIP_EXISTING=0
OUTPUT_DIR=""
AUDIO_EXTENSIONS="wav,flac,mp3,ogg,m4a,aac"

# ──────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") -d <audio_dir> [options]

Generates .npy vertex files only (no video rendering).

Required:
  -d, --audio-dir DIR       Directory containing audio files

Options:
  -o, --output-dir DIR      Output directory for .npy files (default: from config demo_npy_save_folder)
  -c, --config PATH         Config YAML file          (default: $CONFIG)
  --condition NAME          Speaker condition override (default: from config)
  --subject NAME            Subject template override  (default: from config)
  --skip-existing           Skip audios whose output .npy files already exist
  --extensions EXT          Comma-separated audio extensions (default: $AUDIO_EXTENSIONS)
  -h, --help                Show this help message
EOF
    exit 0
}

# ──────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--audio-dir)      AUDIO_DIR="$2"; shift 2 ;;
        -o|--output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
        -c|--config)         CONFIG="$2"; shift 2 ;;
        --condition)         CONDITION="$2"; shift 2 ;;
        --subject)           SUBJECT="$2"; shift 2 ;;
        --skip-existing)     SKIP_EXISTING=1; shift ;;
        --extensions)        AUDIO_EXTENSIONS="$2"; shift 2 ;;
        -h|--help)           usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$AUDIO_DIR" ]]; then
    echo "Error: --audio-dir (-d) is required."
    echo
    usage
fi

if [[ ! -d "$AUDIO_DIR" ]]; then
    echo "Error: Audio directory '$AUDIO_DIR' does not exist."
    exit 1
fi

# ──────────────────────────────────────────────
# Resolve project root (one level up from scripts/)
# ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH=./

# ──────────────────────────────────────────────
# Read config defaults for condition/subject/output
# ──────────────────────────────────────────────
_read_cfg() {
    .venv/bin/python -c "
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
# Flatten nested sections
flat = {}
for section in cfg.values():
    if isinstance(section, dict):
        flat.update(section)
print(flat.get(sys.argv[2], ''))
" "$CONFIG" "$1"
}

if [[ -z "$CONDITION" ]]; then
    CONDITION="$(_read_cfg condition)"
fi
if [[ -z "$SUBJECT" ]]; then
    SUBJECT="$(_read_cfg subject)"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$(_read_cfg demo_npy_save_folder)"
    if [[ -z "$OUTPUT_DIR" ]]; then
        OUTPUT_DIR="demo/npy"
    fi
fi

echo "Config:    $CONFIG"
echo "Condition: $CONDITION"
echo "Subject:   $SUBJECT"
echo "Output:    $OUTPUT_DIR"
echo ""

# ──────────────────────────────────────────────
# Collect audio files
# ──────────────────────────────────────────────
IFS=',' read -ra EXTS <<< "$AUDIO_EXTENSIONS"
AUDIO_FILES=()
for ext in "${EXTS[@]}"; do
    while IFS= read -r -d '' f; do
        AUDIO_FILES+=("$f")
    done < <(find "$AUDIO_DIR" -maxdepth 1 -type f -iname "*.${ext}" -print0 2>/dev/null)
done

# Sort for deterministic order
IFS=$'\n' AUDIO_FILES=($(sort <<< "${AUDIO_FILES[*]}")); unset IFS

if [[ ${#AUDIO_FILES[@]} -eq 0 ]]; then
    echo "No audio files found in '$AUDIO_DIR' with extensions: $AUDIO_EXTENSIONS"
    exit 1
fi

echo "Found ${#AUDIO_FILES[@]} audio file(s) in '$AUDIO_DIR'"

# ──────────────────────────────────────────────
# Reorder files for prefix diversity
# ──────────────────────────────────────────────
# Files sharing a prefix (name without trailing _DIGITS) are part of the
# same sequence. We interleave prefixes so early results are representative,
# prioritising prefixes with fewer existing outputs.
PRE_EXISTING=0
REORDERED=()
_first_line=1
while IFS= read -r line; do
    if [[ $_first_line -eq 1 ]]; then
        PRE_EXISTING="$line"
        _first_line=0
        continue
    fi
    [[ -n "$line" ]] && REORDERED+=("$line")
done < <(printf '%s\n' "${AUDIO_FILES[@]}" | .venv/bin/python -c "
import os, sys, re
from collections import defaultdict

files = [l.strip() for l in sys.stdin if l.strip()]
out_dir, condition, subject = sys.argv[1], sys.argv[2], sys.argv[3]

def get_prefix(name):
    # TalkingHead-1KH: {id}_{seg}_S{start}_E{end}_L..._T..._R..._B...
    m = re.search(r'_\d+_S\d+_E\d+', name)
    if m:
        return name[:m.start()]
    # Fallback: strip trailing _DIGITS (e.g. speaker1_001 -> speaker1)
    return re.sub(r'_\d+$', '', name)

def get_audio_name(fp):
    return os.path.splitext(os.path.basename(fp))[0]

def output_exists(audio_name):
    return os.path.isfile(os.path.join(out_dir, audio_name, f'condition_{condition}_subject_{subject}.npy'))

existing_count = defaultdict(int)
for f in files:
    name = get_audio_name(f)
    if output_exists(name):
        existing_count[get_prefix(name)] += 1

groups = defaultdict(list)
order = []
for f in files:
    p = get_prefix(get_audio_name(f))
    if p not in groups:
        order.append(p)
    groups[p].append(f)

print(f'Prefix groups ({len(order)} unique):', file=sys.stderr)
for p in sorted(order):
    e = existing_count.get(p, 0)
    print(f'  {p}: {len(groups[p])} input, {e} existing output(s)', file=sys.stderr)

scheduled = dict(existing_count)
result = []
while any(groups[p] for p in order):
    best = min((p for p in order if groups[p]), key=lambda p: scheduled.get(p, 0))
    result.append(groups[best].pop(0))
    scheduled[best] = scheduled.get(best, 0) + 1

# First line: total pre-existing output count
print(sum(existing_count.values()))
for f in result:
    print(f)
" "$OUTPUT_DIR" "$CONDITION" "$SUBJECT")

if [[ ${#REORDERED[@]} -gt 0 ]]; then
    AUDIO_FILES=("${REORDERED[@]}")
fi

# ──────────────────────────────────────────────
# Process each audio file
# ──────────────────────────────────────────────
TOTAL=${#AUDIO_FILES[@]}
SKIPPED=0
PROCESSED=0
FAILED=0
BATCH_START=$(date +%s)
TOTAL_PROCESSING_TIME=0

fmt_duration() {
    local secs=$1
    printf "%02d:%02d:%02d" $((secs/3600)) $(($((secs%3600))/60)) $((secs%60))
}

for idx in "${!AUDIO_FILES[@]}"; do
    audio_path="${AUDIO_FILES[$idx]}"
    audio_name="$(basename "${audio_path%.*}")"
    # Output npy naming: {output_dir}/{audio_name}/condition_{condition}_subject_{subject}.npy
    out_npy="${OUTPUT_DIR}/${audio_name}/condition_${CONDITION}_subject_${SUBJECT}.npy"

    # Check if output already exists
    if [[ "$SKIP_EXISTING" -eq 1 ]]; then
        if [[ -f "$out_npy" ]]; then
            echo "[$((PRE_EXISTING + PROCESSED + SKIPPED + 1))/$TOTAL] SKIP  (exists): $audio_name"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi

    # ETA estimate
    GLOBAL_DONE=$((PRE_EXISTING + PROCESSED))
    remaining=$((TOTAL - GLOBAL_DONE - SKIPPED - FAILED))
    if [[ $PROCESSED -gt 0 ]]; then
        avg=$((TOTAL_PROCESSING_TIME / PROCESSED))
        eta=$((avg * remaining))
        echo "[$((GLOBAL_DONE + 1))/$TOTAL] Processing: $audio_name  (avg $(fmt_duration $avg)/file, ETA $(fmt_duration $eta))"
    else
        echo "[$((GLOBAL_DONE + 1))/$TOTAL] Processing: $audio_name"
    fi

    FILE_START=$(date +%s)

    # Build the command — override wav path, condition, subject, npy output via config opts
    CMD=(
        .venv/bin/python main/demo_npy.py
        --config "$CONFIG"
        demo_wav_path "$audio_path"
    )

    if [[ -n "$CONDITION" ]]; then
        CMD+=(condition "$CONDITION")
    fi
    if [[ -n "$SUBJECT" ]]; then
        CMD+=(subject "$SUBJECT")
    fi
    if [[ -n "$OUTPUT_DIR" ]]; then
        CMD+=(demo_npy_save_folder "$OUTPUT_DIR")
    fi

    if "${CMD[@]}" 2>&1; then
        FILE_END=$(date +%s)
        FILE_ELAPSED=$((FILE_END - FILE_START))
        TOTAL_PROCESSING_TIME=$((TOTAL_PROCESSING_TIME + FILE_ELAPSED))
        PROCESSED=$((PROCESSED + 1))
        echo "  -> Done: $out_npy ($(fmt_duration $FILE_ELAPSED))"
    else
        FILE_END=$(date +%s)
        FILE_ELAPSED=$((FILE_END - FILE_START))
        TOTAL_PROCESSING_TIME=$((TOTAL_PROCESSING_TIME + FILE_ELAPSED))
        FAILED=$((FAILED + 1))
        echo "  -> FAILED: $audio_name ($(fmt_duration $FILE_ELAPSED))"
    fi
done

# ──────────────────────────────────────────────
BATCH_END=$(date +%s)
BATCH_ELAPSED=$((BATCH_END - BATCH_START))
echo ""
echo "===== Batch complete ====="
echo "  Total:        $TOTAL"
echo "  Pre-existing: $PRE_EXISTING"
echo "  Processed:    $PROCESSED"
echo "  Skipped:      $SKIPPED"
echo "  Failed:       $FAILED"
echo "  Elapsed:   $(fmt_duration $BATCH_ELAPSED)"
if [[ $PROCESSED -gt 0 ]]; then
    echo "  Avg/file:  $(fmt_duration $((TOTAL_PROCESSING_TIME / PROCESSED)))"
fi
