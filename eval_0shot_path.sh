model_path=$1
# optional: GPU id (default 2)
gpu=${2:-2}

get_exp_name(){
    fname=`basename $1`
    prefix="llava-siglip-so400m-onevision-qwen2-7b--"
    suffix="_1epoch_4x8x4_1e-5"
    echo $fname | sed -n 's/'"$prefix"'//gp' | sed -n 's/'"$suffix"'//gp'
}
exp_name=`get_exp_name $model_path`

if [[ $exp_name == "" ]]
then
    exp_name=`basename $model_path`
fi

echo $exp_name

# Default: musc_v0 + scheme_requery hyp (narrow-band H_A/H_B template re-ask), temp=0.
# NEVER musc_scores as label. Only valid for v0. Does NOT apply to cotm_* (long text).
#
# Inner args: model exp v gpu dataset expert prod [temperature] [mode]
# mode: scheme_requery | scheme_requery_v2 | scheme_requery:lo:hi:tau_ev:delta | ab_tau=0.7 | (empty=free generate)

bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh \
  "$model_path" "$exp_name" v0 "$gpu" mvtec _musc all 0 scheme_requery

# --- other presets ---
# free generate (no gate / no requery):
# bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh \
#   "$model_path" "$exp_name" v0 "$gpu" mvtec _musc all 0

# Scheme requery with explicit band / tau_ev / delta:
# bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh \
#   "$model_path" "$exp_name" v0 "$gpu" mvtec _musc all 0 scheme_requery:0.4:0.7:0.7:1

# A2 soft aggregate (requeryHypv2): soft votes + pair antisymmetry + p_f mix
# bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh \
#   "$model_path" "$exp_name" v0 "$gpu" mvtec _musc all 0 scheme_requery_v2
# bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh \
#   "$model_path" "$exp_name" v0 "$gpu" mvtec _musc all 0 scheme_requery_v2:0.4:0.7:0.7:0.5

# Defv3 (requeryHypDefv3): quantile hard-example gate + contrastive default_v3 checklist
# bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh \
#   "$model_path" "$exp_name" v0 "$gpu" mvtec _musc all 0 scheme_requery_default_v3
# bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh \
#   "$model_path" "$exp_name" v0 "$gpu" mvtec _musc all 0 scheme_requery_default_v3:0.1:0.7:1

# Scheme A (pure P(A) threshold) — v0 only:
# bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh \
#   "$model_path" "$exp_name" v0 "$gpu" mvtec _musc all 0 ab_tau=0.7

# CoTM long text — never pass scheme_requery / ab_tau:
# bash scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh \
#   "$model_path" "$exp_name" cotm_v3 "$gpu" mvtec _musc all 0
