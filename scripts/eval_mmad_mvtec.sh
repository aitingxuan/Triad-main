#!/usr/bin/env bash
# MMAD MVTec 0-shot evaluation with CoTM / Triad checkpoint.
#
# Prerequisites (after download):
#   data/MMAD/mmad.json
#   data/MMAD/DS-MVTec/   (recommended; paths match mmad.json)
#     OR
#   data/MMAD/MVTec-AD/   (official MVTec; auto-maps image/ -> test/)
#
# Usage:
#   bash scripts/eval_mmad_mvtec.sh /path/to/cotm_checkpoint [EXP_NAME] [GPU]
#
# Environment:
#   MMAD_ROOT   - default: $TRIAD_ROOT/data/MMAD
#   IMAGE_ROOT  - auto | DS-MVTec | MVTec-AD  (default: auto)
#   MAX_IMAGES  - debug: pass to convert script (optional)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIAD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TRIAD_ROOT}"

MODEL_PATH="${1:?Usage: $0 MODEL_PATH [EXP_NAME] [GPU]}"
EXP_NAME="${2:-$(basename "${MODEL_PATH%/}")}"
GPU="${3:-0}"
MMAD_ROOT="${MMAD_ROOT:-${TRIAD_ROOT}/data/MMAD}"
IMAGE_ROOT="${IMAGE_ROOT:-auto}"

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "=== MMAD MVTec 0-shot eval ==="
echo "Model:     ${MODEL_PATH}"
echo "Exp name:  ${EXP_NAME}"
echo "MMAD root: ${MMAD_ROOT}"
echo "Image root preference: ${IMAGE_ROOT}"
echo "GPU:       ${CUDA_VISIBLE_DEVICES}"

if [[ ! -f "${MMAD_ROOT}/mmad.json" ]]; then
  echo "ERROR: ${MMAD_ROOT}/mmad.json not found."
  echo "Download MMAD from https://huggingface.co/datasets/jiang-cc/MMAD"
  echo "Expected layout (either is OK):"
  echo "  ${MMAD_ROOT}/mmad.json"
  echo "  ${MMAD_ROOT}/DS-MVTec/bottle/image/ ...   # matches mmad.json (recommended)"
  echo "  ${MMAD_ROOT}/MVTec-AD/bottle/test/ ...    # official MVTec-AD"
  exit 1
fi

CONVERT_ARGS=(--mmad_root "${MMAD_ROOT}" --image_root "${IMAGE_ROOT}" --verify_images)
if [[ -n "${MAX_IMAGES:-}" ]]; then
  CONVERT_ARGS+=(--max_images "${MAX_IMAGES}")
fi

python scripts/convert_mmad_mvtec.py "${CONVERT_ARGS[@]}"

ANS_REL="results/${EXP_NAME}/answer_mvtec_0shot.json"
mkdir -p "${MMAD_ROOT}/results/${EXP_NAME}"

python llava_eval/evaluate.py \
  --model-path "${MODEL_PATH}" \
  --model-type triad \
  --ds mmad_mvtec \
  --data_path "${MMAD_ROOT}" \
  --text_src dataset \
  --from question_mvtec_0shot.json \
  --to "${ANS_REL}" \
  --image_num 1 \
  --qa_mode single_qa \
  --max_new_tokens 128 \

python llava_eval/collect_mmad_acc.py "${MMAD_ROOT}/${ANS_REL}"

echo "Done. Answers: ${MMAD_ROOT}/${ANS_REL}"
