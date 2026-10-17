#! /bin/bash
model_path=$1
model_name=$2
v=$3
gpu=$4
dataset=$5
expert=$6
prod=$7
# Decode / decision kwargs:
#   $8  temperature           (default 0)
#   $9  mode:
#         empty
#         ab_tau=<x>
#         scheme_requery | scheme_requery:<lo>:<hi>:<tau_ev>:<delta>
#         scheme_requery_default | scheme_requery_default:<lo>:<hi>:<tau_ev>:<delta>
#         scheme_requery_default_v2 | scheme_requery_default_v2:<lo>:<hi>:<tau_ev>:<delta>
#         scheme_requery_default_v3 | scheme_requery_default_v3:<top_frac>:<tau_ev>:<delta>
#         scheme_requery_default_v4 | scheme_requery_default_v4:<top_frac>
#         scheme_requery_v2 | scheme_requery_v2:<lo>:<hi>:<tau_ev>:<delta>
#         scheme_requery_v2_default | scheme_requery_v2_default:<lo>:<hi>:<tau_ev>:<delta>
#         scheme_requery_selfq | scheme_requery_selfq:<top_frac>:<max_turns>
#       examples:  scheme_requery
#                  scheme_requery:0.4:0.7:0.7:1
#                  scheme_requery_default   # DEFAULT checklist, max_subq=2 (tag requeryHypDef)
#                  scheme_requery_default_v2  # DEFAULT + geometry Qs, max_subq=3 (tag requeryHypDefv2)
#                  scheme_requery_default_v3  # Defv3: quantile gate + contrastive checklist (tag requeryHypDefv3)
#                  scheme_requery_default_v3:0.1:0.7:1
#                  scheme_requery_default_v4  # Default_v4: attention crop ROI + default_v2 checklist (tag requeryAttnCrop)
#                  scheme_requery_default_v4:0.1
#                  scheme_requery_v2        # A2 soft aggregate (tag requeryHypv2)
#                  scheme_requery_selfq     # T2 self-questioning + Bayesian fusion (tag requerySelfq)
#                  scheme_requery_selfq:0.1:2
#                  ab_tau=0.7
# NOTE: scheme_b (musc_scores) and ROI crop requery have been removed.
temperature=${8:-0}
mode=${9:-}

export CUDA_VISIBLE_DEVICES=$gpu
if [[ $CUDA_VISIBLE_DEVICES == '' ]]
then
    export CUDA_VISIBLE_DEVICES=0
fi

d_path=./evaluation/$dataset

decode_tag="_t${temperature}"
extra_args=(--temperature "$temperature")

_parse_scheme_requery() {
    # $1 = mode string; $2 = decode tag suffix; $3 = checklist; $4 = hyp|hyp_v2
    local m="$1"
    local tag_suffix="$2"
    local checklist_arg="$3"
    local hyp_mode="${4:-hyp}"
    if [[ "$m" == *:* ]]; then
        # scheme_requery[_v2][_default]:lo:hi:tau_ev:delta
        IFS=':' read -r _ lo hi tau_ev delta <<< "$m"
        decode_tag="${decode_tag}_${tag_suffix}_b${lo}-${hi}"
        extra_args+=(--scheme_requery "$hyp_mode" --band_lo "$lo" --band_hi "$hi")
        if [[ -n "${tau_ev:-}" ]]; then
            decode_tag="${decode_tag}_tev${tau_ev}"
            extra_args+=(--requery_tau_ev "$tau_ev")
        fi
        if [[ -n "${delta:-}" ]]; then
            decode_tag="${decode_tag}_d${delta}"
            extra_args+=(--requery_delta "$delta")
        fi
    else
        decode_tag="${decode_tag}_${tag_suffix}"
        extra_args+=(--scheme_requery "$hyp_mode")
    fi
    if [[ -n "$checklist_arg" ]]; then
        extra_args+=(--requery_checklist "$checklist_arg")
    fi
}

if [[ -z "$mode" ]]; then
    :
elif [[ "$mode" == scheme_requery_selfq || "$mode" == scheme_requery_selfq:* ]]; then
    # Scheme 4 (T2): model self-WRITES a yes/no check, answered by forced two-choice
    # readout; evidence is entropy-gated (sharper than main) and Bayesian-fused.
    # Recommended hard-example gate = quantile top ~10%. NO tau_ev / band / delta.
    # Optional: scheme_requery_selfq:<top_frac>:<max_turns>
    decode_tag="${decode_tag}_requerySelfq"
    extra_args+=(
        --scheme_requery selfq
        --requery_gate quantile
    )
    if [[ "$mode" == *:* ]]; then
        IFS=':' read -r _ top_frac max_turns <<< "$mode"
        if [[ -n "${top_frac:-}" ]]; then
            decode_tag="${decode_tag}_tf${top_frac}"
            extra_args+=(--requery_top_frac "$top_frac")
        fi
        if [[ -n "${max_turns:-}" ]]; then
            decode_tag="${decode_tag}_nt${max_turns}"
            extra_args+=(--requery_max_turns "$max_turns")
        fi
    fi
elif [[ "$mode" == scheme_requery_default_v4 || "$mode" == scheme_requery_default_v4:* ]]; then
    # Default_v4 (requeryAttnCrop): attention-guided suspect box -> crop high-fidelity ROI
    # from the ORIGINAL image (EG-RoI view layout) -> default_v2 checklist asked on the crop.
    # Attention only PROPOSES the box; the VLM A/B sub-answers still decide (never MuSc).
    # Hard-example gate = quantile top ~10%. Optional: scheme_requery_default_v4:<top_frac>
    decode_tag="${decode_tag}_requeryAttnCrop"
    extra_args+=(
        --scheme_requery attncrop
        --requery_checklist default
        --requery_gate quantile
        --requery_max_subq 3
    )
    overlay_dir="results/${model_name}/attncrop_overlays_${dataset}_${prod}_${v}"
    extra_args+=(--requery_attncrop_overlay_dir "$overlay_dir")
    if [[ "$mode" == *:* ]]; then
        IFS=':' read -r _ top_frac <<< "$mode"
        if [[ -n "${top_frac:-}" ]]; then
            decode_tag="${decode_tag}_tf${top_frac}"
            extra_args+=(--requery_top_frac "$top_frac")
        fi
    fi
elif [[ "$mode" == scheme_requery_v2_default || "$mode" == scheme_requery_v2_default:* ]]; then
    _parse_scheme_requery "$mode" "requeryHypv2Def" "default" "hyp_v2"
elif [[ "$mode" == scheme_requery_v2 || "$mode" == scheme_requery_v2:* ]]; then
    _parse_scheme_requery "$mode" "requeryHypv2" "per_class" "hyp_v2"
elif [[ "$mode" == scheme_requery_default_v3 || "$mode" == scheme_requery_default_v3:* ]]; then
    # Defv3: quantile hard-example gate (no absolute p_A band) + contrastive default_v3 checklist.
    # Optional: scheme_requery_default_v3:<top_frac>:<tau_ev>:<delta>
    decode_tag="${decode_tag}_requeryHypDefv3"
    extra_args+=(
        --scheme_requery hyp
        --requery_checklist default_v3
        --requery_gate quantile
        --requery_max_subq 3
    )
    if [[ "$mode" == *:* ]]; then
        IFS=':' read -r _ top_frac tau_ev delta <<< "$mode"
        if [[ -n "${top_frac:-}" ]]; then
            decode_tag="${decode_tag}_tf${top_frac}"
            extra_args+=(--requery_top_frac "$top_frac")
        fi
        if [[ -n "${tau_ev:-}" ]]; then
            decode_tag="${decode_tag}_tev${tau_ev}"
            extra_args+=(--requery_tau_ev "$tau_ev")
        fi
        if [[ -n "${delta:-}" ]]; then
            decode_tag="${decode_tag}_d${delta}"
            extra_args+=(--requery_delta "$delta")
        fi
    fi
elif [[ "$mode" == scheme_requery_default_v2 || "$mode" == scheme_requery_default_v2:* ]]; then
    # Expanded DEFAULT checklist (surface + contamination + geometry); ask all 3 pairs.
    _parse_scheme_requery "$mode" "requeryHypDefv2" "default" "hyp"
    extra_args+=(--requery_max_subq 3)
elif [[ "$mode" == scheme_requery_default || "$mode" == scheme_requery_default:* ]]; then
    _parse_scheme_requery "$mode" "requeryHypDef" "default" "hyp"
elif [[ "$mode" == scheme_requery || "$mode" == scheme_requery:* ]]; then
    _parse_scheme_requery "$mode" "requeryHyp" "per_class" "hyp"
elif [[ "$mode" == ab_tau=* ]]; then
    tau="${mode#ab_tau=}"
    decode_tag="${decode_tag}_tau${tau}"
    extra_args+=(--ab_tau "$tau")
elif [[ "$mode" == scheme_b* ]]; then
    echo "ERROR: scheme_b (musc_scores hard gate) has been removed. Use scheme_requery."
    exit 1
else
    echo "Unknown mode='$mode'. Use empty | scheme_requery[_v2][_default|_default_v2|_default_v3|_default_v4] | scheme_requery_selfq | … | ab_tau=<x>"
    exit 1
fi

ans_path=results/${model_name}/ans_${dataset}_${prod}_0shot_randomroi${expert}_${v}${decode_tag}.json
csv_path=$model_path/result_${dataset}_${expert}_${v}_${prod}${decode_tag}.csv

echo "eval $model_name dataset=$dataset v=$v GPU=$gpu temp=$temperature mode=${mode:-free}"
echo "  -> ans: $d_path/$ans_path"
echo "  -> csv: $csv_path"

python llava_eval/evaluate.py \
  --model-path "$model_path" \
  --model-type triad \
  --ds "$dataset" \
  --data_path "$d_path" \
  --text_src "config:$v" \
  --subset "$prod" \
  --from "question${expert}.jsonl" \
  --to "$ans_path" \
  --image_num 1 \
  --qa_mode single_qa \
  "${extra_args[@]}"

python llava_eval/collect_acc.py "$d_path/$ans_path" > "$csv_path"
echo "wrote $csv_path"
