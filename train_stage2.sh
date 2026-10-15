# Stage2: CoTM-simple A/B 决策对齐（Plan A merge 后的 base 上继续训）
# 用法: bash finetune_ov_2stage.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4000

export CUDA_VISIBLE_DEVICES=1,2,3,4,5

LLM_VERSION_OV="/home/yz/code/Triad-main/checkpoints/merged_llava_qwen_triad_9ROI_CAprobe_cotmv3_vis_noMM_zhenduan"
VISION_MODEL_VERSION="/home/yz/code/Triad-main/google"
PROMPT_VERSION="qwen_1_5"

get_stage2_path(){
    echo /home/yz/code/Triad-main/checkpoints/llava_qwen_triad_9ROI_CAprobe_cotmv3_vis_noMM_zhenduan_stage2
}

exp_name=Triad_llava_ov_lora
stage2_path=`get_stage2_path ${exp_name}`
echo Will save to $stage2_path

mkdir -p tmp_log

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="5" --nnodes="1" --master_addr="localhost" --master_port="29600" \
    LLaVA-NeXT/llava/train/train_mem.py \
    --deepspeed LLaVA-NeXT/scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION_OV} \
    --version ${PROMPT_VERSION} \
    --data_path="{/home/yz/code/Triad-main/movable/CoTM/CoTM_simple.json}" \
    --image_folder /home/yz/code/Triad-main/movable \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_bias none \
    --mm_tunable_parts "" \
    --mm_vision_tower_lr=2e-6 \
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
    --run_name ${exp_name}_stage2 \
    --output_dir $stage2_path \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --attn_implementation sdpa \
    --dataloader_drop_last True \
    --report_to tensorboard 2>&1 | tee tmp_log/stage2_$(date "+%Y%m%d_%H%M%S").log
