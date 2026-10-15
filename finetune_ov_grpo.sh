#!/usr/bin/env bash
# Stage-2 GRPO training for Triad IAD (reward = answer accuracy only).
# Light-memory defaults: fewer rollouts, shorter answers, frozen vision tower, chunked logprob.

set -euo pipefail

export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4000
# Avoid 30min NCCL watchdog when ranks finish GRPO rollouts at different speeds.
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=86400
export NCCL_ASYNC_ERROR_HANDLING=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6}"
NPROC="${NPROC:-5}"

# Base OneVision weights (same as stage-1); stage-1 LoRA ckpt overrides via STAGE1_CKPT.
LLM_VERSION_OV="${LLM_VERSION_OV:-./llava-siglip-so400m-onevision-qwen2-7b_9ROI}"
VISION_MODEL_VERSION="${VISION_MODEL_VERSION:-./google}"
PROMPT_VERSION="${PROMPT_VERSION:-qwen_1_5}"
IMAGE_FOLDER="${IMAGE_FOLDER:-./movable}"
DATA_PATH="${DATA_PATH:-{./movable/CoTM/CoTM_v3.json,./movable/InstructIAD/InstructIAD.json}}"

STAGE1_CKPT="${STAGE1_CKPT:-/home/yz/code/Triad-main/checkpoints/llava-siglip-so400m-onevision-qwen2-7b--Triad_llava_ov_lora_1epoch_4x8x4_1e-5_9ROI_CAprobe_cotmv3_zhenduan}"

exp_name=Triad_llava_ov_grpo
model_path="./checkpoints/llava-siglip-so400m-onevision-qwen2-7b--${exp_name}"
mkdir -p tmp_log

echo "Stage-1 checkpoint: ${STAGE1_CKPT}"
echo "GRPO output: ${model_path}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} (nproc=${NPROC})"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NPROC}" --nnodes="1" --master_addr="localhost" --master_port="29601" \
    LLaVA-NeXT/llava/train/train_grpo_mem.py \
    --deepspeed LLaVA-NeXT/scripts/zero2.json \
    --model_name_or_path "${LLM_VERSION_OV}" \
    --lora_weight_path "${STAGE1_CKPT}" \
    --version "${PROMPT_VERSION}" \
    --data_path="${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_bias none \
    --mm_tunable_parts "mm_mlp_adapter" \
    --vision_tower "${VISION_MODEL_VERSION}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio "anyres_max_9_randomroi" \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type "spatial_avgpool_auto_unpad_randomroi_newl" \
    --multi_image_pad_threshold 2 \
    --max_box_num 2 \
    --bf16 True \
    --run_name "${exp_name}" \
    --output_dir "${model_path}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --attn_implementation sdpa \
    --dataloader_drop_last True \
    --grpo_beta 0.04 \
    --num_generations 2 \
    --grpo_max_new_tokens 16 \
    --grpo_temperature 1.0 \
    --grpo_top_p 0.9 \
    --grpo_logprob_prompt_tail 384 \
    --grpo_logprob_chunk_size 2 \
    --grpo_image_aspect_ratio "anyres_max_4_randomroi" \
    --ddp_find_unused_parameters True \
    --report_to tensorboard 2>&1 | tee "tmp_log/grpo_train_$(date "+%Y%m%d_%H%M%S").log"
