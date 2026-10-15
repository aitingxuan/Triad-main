export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4000

export CUDA_VISIBLE_DEVICES=3,4


LLM_VERSION_OV="/home/yz/code/Triad-main/llava-siglip-so400m-onevision-qwen2-7b_9ROI"
# LLM_VERSION_SI="/hdd/ysh/pretrained_models/models--lmms-lab--llava-onevision-qwen2-7b-si/"
VISION_MODEL_VERSION="/home/yz/code/Triad-main/google"
PROMPT_VERSION="qwen_1_5"

get_model_path(){
    echo ./checkpoints/llava-siglip-so400m-onevision-qwen2-7b--${1}_1epoch_4x8x4_1e-5_9ROI_cotmv6_ecprloss_vislorav2
}

# CoTM_v6 (AB + checklist + process-align on real juice NG; no paste) + InstructIAD; no TNR upsample
exp_name=Triad_llava_ov_lora_cotmv6

# ECPR Week0/1 slots + Week2 aux (docs/EG_CONDITIONAL_PART_ROI.md §6).
#   ECPR_ENABLE=1          → juice sidecar boxes (EG ∪ triggered parts)
#   ECPR_AUX=1 (default 1 when ECPR_ENABLE=1) → localize/attr/agg CE + local-agg BCE
ECPR_ENABLE=${ECPR_ENABLE:-0}
ECPR_AUX=${ECPR_AUX:-${ECPR_ENABLE}}
ECPR_SIDECAR=/home/yz/code/Triad-main/movable/juice_ecpr_sidecar.jsonl
ECPR_ARGS=()
if [ "${ECPR_ENABLE}" = "1" ]; then
  if [ ! -f "${ECPR_SIDECAR}" ]; then
    python3 scripts/build_ecpr_sidecar.py
  fi
  ECPR_ARGS=(--ecpr_enable True --ecpr_sidecar "${ECPR_SIDECAR}")
  if [ "${ECPR_AUX}" = "1" ]; then
    exp_name=Triad_llava_ov_lora_cotmv6_ecpr_aux
    ECPR_ARGS+=(
      --ecpr_aux_loss True
      --ecpr_lambda_localize 0.5
      --ecpr_lambda_attr 0.5
      --ecpr_lambda_agg 0.5
      --ecpr_lambda_local 0.2
      --ecpr_attr_gate_scale 0.25
      --ecpr_aux_log_every 50
      --ecpr_max_slots 4
    )
    echo "ECPR Week2 aux enabled (sidecar=${ECPR_SIDECAR})"
  else
    exp_name=Triad_llava_ov_lora_cotmv6_ecpr
    echo "ECPR slots only (no aux loss): ${ECPR_SIDECAR}"
  fi
fi

model_path=`get_model_path ${exp_name}`
log_name=llava_train_$(date "+%Y%m%d_%H%M%S").log
echo Will save to $model_path

mkdir -p tmp_log

# Build CoTM_v6 once if missing (rebuild: python3 scripts/build_cotm_v6.py --force)
if [ ! -f /home/yz/code/Triad-main/movable/CoTM/CoTM_v6.json ]; then
  python3 scripts/build_cotm_v6.py --force
fi

# zero3.json has stage3_prefetch_bucket_size=0 / overlap_comm=false to avoid
# DeepSpeed "Invalidate trace cache" + "still have inflight params" on Triad's
# variable anyres/ROI graphs. If it still crashes, fall back to zero2.json.
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="2" --nnodes="1" --master_addr="localhost" --master_port="29600" \
    LLaVA-NeXT/llava/train/train_mem.py \
    --deepspeed LLaVA-NeXT/scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION_OV} \
    --version ${PROMPT_VERSION} \
    --data_path=/home/yz/code/Triad-main/movable/train_cotmv6.yaml \
    --image_folder /home/yz/code/Triad-main/movable \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_bias none \
    --mm_tunable_parts "mm_mlp_adapter" \
    --lora_vision_enable True \
    --lora_vision_r 16 \
    --lora_vision_alpha 32 \
    --lora_vision_dropout 0.05 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio "anyres_max_9_randomroi" \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type "spatial_avgpool_auto_unpad_randomroi_newl" \
    --multi_image_pad_threshold 2 \
    --bf16 True \
    --run_name $exp_name \
    --output_dir $model_path \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile False \
    --torch_compile_backend "inductor" \
    --attn_implementation sdpa \
    --decision_token_loss True \
    --decision_token_no_weight 1.0 \
    --decision_token_yes_weight 1.0 \
    --decision_token_log_every 50 \
    --dataloader_drop_last True \
    "${ECPR_ARGS[@]}" \
    --report_to tensorboard 2>&1 | tee tmp_log/llava_train_$(date "+%Y%m%d_%H%M%S").log
