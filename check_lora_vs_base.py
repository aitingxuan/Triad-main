import json
import math
import os
from typing import Dict, Iterable, List, Set, Tuple

import torch
from safetensors.torch import load_file as safe_load_file


def _iter_keys(
    base_sd: Dict[str, torch.Tensor],
    merged_sd: Dict[str, torch.Tensor],
    include_keywords: List[str],
) -> Iterable[str]:
    shared = set(base_sd.keys()) & set(merged_sd.keys())
    for key in sorted(shared):
        if include_keywords and not any(k in key for k in include_keywords):
            continue
        if not (torch.is_floating_point(base_sd[key]) and torch.is_floating_point(merged_sd[key])):
            continue
        yield key


def _tensor_stats(diff: torch.Tensor) -> Tuple[float, float]:
    abs_diff = diff.abs()
    return abs_diff.mean().item(), abs_diff.max().item()


def _load_lora_targets(lora_path: str) -> List[str]:
    cfg_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"adapter_config.json not found in LoRA path: {lora_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    targets = cfg.get("target_modules", [])
    if isinstance(targets, str):
        targets = [targets]
    if not isinstance(targets, list):
        raise ValueError("Invalid target_modules in adapter_config.json")
    return [str(t) for t in targets]


def _is_target_key(key: str, target_modules: List[str]) -> bool:
    # LoRA usually applies to weight matrices of modules listed in target_modules.
    # Match by module token + common weight suffix.
    for t in target_modules:
        token = f".{t}."
        if token in key and key.endswith(".weight"):
            return True
        if key.endswith(f".{t}.weight"):
            return True
    return False


def _load_state_dict_shards(model_dir: str) -> Dict[str, torch.Tensor]:
    """
    Offline checkpoint loader:
    - supports sharded/non-sharded safetensors
    - supports sharded/non-sharded pytorch bin
    - never initializes model classes, so no HF network access is triggered
    """
    st_index = os.path.join(model_dir, "model.safetensors.index.json")
    bin_index = os.path.join(model_dir, "pytorch_model.bin.index.json")
    st_single = os.path.join(model_dir, "model.safetensors")
    bin_single = os.path.join(model_dir, "pytorch_model.bin")

    shard_files: List[str] = []
    shard_type = None

    if os.path.isfile(st_index):
        with open(st_index, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        shard_files = sorted(set(index_data["weight_map"].values()))
        shard_type = "safetensors"
    elif os.path.isfile(bin_index):
        with open(bin_index, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        shard_files = sorted(set(index_data["weight_map"].values()))
        shard_type = "bin"
    elif os.path.isfile(st_single):
        shard_files = [os.path.basename(st_single)]
        shard_type = "safetensors"
    elif os.path.isfile(bin_single):
        shard_files = [os.path.basename(bin_single)]
        shard_type = "bin"
    else:
        raise FileNotFoundError(
            f"No recognized checkpoint files found in: {model_dir}. "
            "Expected model.safetensors(.index.json) or pytorch_model.bin(.index.json)."
        )

    full_state_dict: Dict[str, torch.Tensor] = {}
    for shard in shard_files:
        shard_path = os.path.join(model_dir, shard)
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"Missing shard file: {shard_path}")
        if shard_type == "safetensors":
            part = safe_load_file(shard_path, device="cpu")
        else:
            part = torch.load(shard_path, map_location="cpu")
        full_state_dict.update(part)

    return full_state_dict


def main():
    # ================= 请在这里设置路径和参数 =================
    base_path = "/home/yz/code/Triad-main/checkpoints/merged_llava_qwen_triad_9ROI_CAprobe_struct_all_vis_zhenduan"        # 基础模型路径
    merged_path = "/home/yz/code/Triad-main/checkpoints/merged_llava_qwen_triad_9ROI_CAprobe_struct_all_vis_grpo_zhenduan"    # 合并后的模型路径
    include_keywords_str = ""                # 逗号分隔的关键词，如 "vision_model,lm_head"，留空表示全部
    topk = 20                                # 显示变动最大的参数数量
    lora_path = ""                           # LoRA adapter 路径（可选），留空则跳过LoRA分析
    eps = 0.0                                # 判断参数是否发生变化的阈值
    # ========================================================

    include_keywords = [x.strip() for x in include_keywords_str.split(",") if x.strip()]

    print(f"[Load] base   : {base_path}")
    print(f"[Load] merged : {merged_path}")
    print("[Load] mode=offline state_dict compare (no model init, no network)")
    if include_keywords:
        print(f"[Filter] include keywords: {include_keywords}")

    base_sd = _load_state_dict_shards(base_path)
    merged_sd = _load_state_dict_shards(merged_path)

    changed: List[Tuple[str, float, float]] = []
    changed_keys: Set[str] = set()
    mean_sum = 0.0
    count = 0

    for key in _iter_keys(base_sd, merged_sd, include_keywords):
        # Compute in float32 for stable statistics.
        diff = merged_sd[key].float() - base_sd[key].float()
        mean_abs, max_abs = _tensor_stats(diff)
        if not (math.isfinite(mean_abs) and math.isfinite(max_abs)):
            continue
        changed.append((key, mean_abs, max_abs))
        if mean_abs > eps:
            changed_keys.add(key)
        mean_sum += mean_abs
        count += 1

    if count == 0:
        print("[Result] No comparable floating-point parameters found with current filter.")
        return

    changed.sort(key=lambda x: x[1], reverse=True)
    avg_mean_abs = mean_sum / count
    nonzero = sum(1 for _, m, _ in changed if m > eps)

    print(f"[Result] Compared floating keys: {count}")
    print(f"[Result] Keys with mean_abs_diff > eps({eps}): {nonzero}")
    print(f"[Result] Global avg(mean_abs_diff): {avg_mean_abs:.8e}")
    print("[Result] Top changed keys:")
    for key, mean_abs, max_abs in changed[:topk]:
        print(f"  - {key}: mean_abs={mean_abs:.8e}, max_abs={max_abs:.8e}")

    if lora_path:
        target_modules = _load_lora_targets(lora_path)
        expected_keys = [
            k for k in _iter_keys(base_sd, merged_sd, include_keywords=[]) if _is_target_key(k, target_modules)
        ]
        expected_set = set(expected_keys)
        changed_expected = sorted(expected_set & changed_keys)
        missed_expected = sorted(expected_set - changed_keys)
        hit_rate = (len(changed_expected) / len(expected_set) * 100.0) if expected_set else 0.0

        print(f"[LoRA] target_modules from adapter_config: {target_modules}")
        print(f"[LoRA] expected target keys: {len(expected_set)}")
        print(f"[LoRA] changed target keys (>eps): {len(changed_expected)}")
        print(f"[LoRA] target hit rate: {hit_rate:.2f}%")
        if missed_expected:
            print(f"[LoRA] suspicious untouched target keys (top {topk}):")
            for key in missed_expected[:topk]:
                print(f"  - {key}")

    # Quick verdict for convenience.
    if nonzero == 0:
        print("[Verdict] merged appears numerically identical to base for compared keys.")
    else:
        print("[Verdict] merged differs from base; LoRA merge likely took effect.")


if __name__ == "__main__":
    main()