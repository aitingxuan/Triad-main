import sys
import os
import json
import torch
import shutil
from transformers import AutoTokenizer
from peft import PeftModel

# 添加 LLaVA 路径以便导入自定义模型类
sys.path.append("./LLaVA-NeXT/")
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM


def _collect_meta_param_names(model):
    return [name for name, p in model.named_parameters() if getattr(p, "is_meta", False)]


def _collect_lora_param_names(model):
    return [name for name, _ in model.named_parameters() if "lora_" in name.lower()]


def _load_non_lora_trainables(model, lora_path):
    """Load vision tower / mm_projector weights saved beside the LoRA adapter."""
    non_lora_path = os.path.join(lora_path, "non_lora_trainables.bin")
    if not os.path.exists(non_lora_path):
        print(f"No non_lora_trainables.bin under {lora_path}; skip non-LoRA load.")
        return

    non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
    non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
    if any(k.startswith("model.model.") for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
    # Drop removed-module keys (e.g. legacy mm_spam_*) from older checkpoints.
    non_lora_trainables = {k: v for k, v in non_lora_trainables.items() if "mm_spam" not in k}

    incompatible = model.load_state_dict(non_lora_trainables, strict=False, assign=True)
    print(
        f"Loaded non_lora_trainables.bin: keys={len(non_lora_trainables)}, "
        f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
    )


def _is_valid_vision_tower_id(path):
    """Mirror the recognition rules in llava.model.multimodal_encoder.builder."""
    if not path or not isinstance(path, str):
        return False
    if path.startswith(("openai", "laion", "hf:", "open_clip_hub")):
        return True
    if "siglip" in path.lower() or "ShareGPT4V" in path or path in ["imagebind_huge"]:
        return True
    return False


def _normalize_mm_vision_tower(path, fallback=None):
    """
    Repair truncated vision-tower paths written during training save.

    Some checkpoints store HF id ``google/siglip-...`` as absolute
    ``.../Triad-main/google`` (first path segment only). Builder then fails with
    ``Unknown vision tower`` because the string no longer contains ``siglip``.
    """
    if _is_valid_vision_tower_id(path):
        return path

    if isinstance(path, str) and os.path.isdir(path):
        children = sorted(
            d for d in os.listdir(path)
            if "siglip" in d.lower() and os.path.isdir(os.path.join(path, d))
        )
        if children:
            preferred = [c for c in children if "so400m-patch14-384" in c]
            fixed = os.path.join(path, (preferred or children)[0])
            print(f"Repaired truncated mm_vision_tower: {path!r} -> {fixed!r}")
            return fixed

    if fallback and _is_valid_vision_tower_id(fallback):
        print(f"Invalid mm_vision_tower {path!r}; falling back to {fallback!r}")
        return fallback

    return path


def _apply_lora_multimodal_config(merged_model, lora_path, base_vision_tower=None):
    """Keep training multimodal config instead of overwriting with base."""
    lora_cfg_path = os.path.join(lora_path, "config.json")
    if not os.path.exists(lora_cfg_path):
        return
    with open(lora_cfg_path, "r") as f:
        lora_cfg = json.load(f)

    # Prefer base vision tower as fallback: LoRA configs often corrupt this field.
    if base_vision_tower is None:
        base_vision_tower = getattr(merged_model.config, "mm_vision_tower", None)

    keep_prefixes = ("mm_",)
    keep_keys = {
        "image_aspect_ratio",
        "image_grid_pinpoints",
        "multi_image_pad_threshold",
        "vision_tower",
        "mm_vision_tower",
        "mm_tunable_parts",
    }
    skip_prefixes = ("mm_spam", "mm_enable_spam")
    for k, v in lora_cfg.items():
        if any(k.startswith(p) or k == p for p in skip_prefixes):
            continue
        if k in keep_keys or k.startswith(keep_prefixes):
            setattr(merged_model.config, k, v)

    for key in ("mm_vision_tower", "vision_tower"):
        if not hasattr(merged_model.config, key):
            continue
        raw = getattr(merged_model.config, key)
        if raw is None:
            continue
        setattr(
            merged_model.config,
            key,
            _normalize_mm_vision_tower(raw, fallback=base_vision_tower),
        )


def main():
    base_path = "/home/yz/code/Triad-main/llava-siglip-so400m-onevision-qwen2-7b_9ROI"
    lora_path = "/home/yz/code/Triad-main/checkpoints/llava-siglip-so400m-onevision-qwen2-7b--Triad_llava_ov_lora_cotmv6_ecpr_aux_1epoch_4x8x4_1e-5_9ROI_cotmv6_ecprloss_vislorav2"
    output_path = "/home/yz/code/Triad-main/checkpoints/merged_llava_qwen_triad_9ROI_cotmv6_ecprloss_vislorav2"

    os.makedirs(output_path, exist_ok=True)

    print("Loading base model using LlavaQwenForCausalLM...")
    base_model = LlavaQwenForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map=None,            # Merge 阶段避免分片/惰性加载导致 meta 参数
        low_cpu_mem_usage=False,    # 强制完整加载，降低 no-op 拷贝风险
    )
    base_vision_tower = getattr(base_model.config, "mm_vision_tower", None)
    if not _is_valid_vision_tower_id(base_vision_tower):
        with open(os.path.join(base_path, "config.json")) as f:
            base_vision_tower = json.load(f).get("mm_vision_tower")

    _load_non_lora_trainables(base_model, lora_path)

    base_meta_params = _collect_meta_param_names(base_model)
    if base_meta_params:
        preview = base_meta_params[:20]
        raise RuntimeError(
            "Base model still contains meta parameters before loading LoRA. "
            f"count={len(base_meta_params)}, examples={preview}"
        )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    # Vision-tower LoRA (adapter_name="vision") saved alongside default adapter.
    import os as _os
    for _cand in (["vision", "default_vision"] if True else []):
        _vp = _os.path.join(lora_path, _cand)
        _vc = _os.path.join(_vp, "adapter_config.json")
        if _os.path.isdir(_vp) and _os.path.exists(_vc):
            try:
                model.load_adapter(_vp, adapter_name=_cand)
                print(f"Loaded extra vision adapter: {_cand}")
            except Exception as _e:
                print(f"WARNING: failed to load extra adapter {_cand}: {_e}")
    try:
        _adapters = list(getattr(model, "peft_config", {}).keys())
        if len(_adapters) > 1:
            try:
                model.set_adapter(_adapters)
            except Exception:
                pass
    except Exception:
        pass

    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()

    merged_meta_params = _collect_meta_param_names(merged_model)
    if merged_meta_params:
        preview = merged_meta_params[:20]
        raise RuntimeError(
            "Merged model contains meta parameters. Merge result is unsafe. "
            f"count={len(merged_meta_params)}, examples={preview}"
        )

    left_lora_params = _collect_lora_param_names(merged_model)
    if left_lora_params:
        preview = left_lora_params[:20]
        raise RuntimeError(
            "LoRA parameters are still present after merge_and_unload. "
            f"count={len(left_lora_params)}, examples={preview}"
        )

    _apply_lora_multimodal_config(
        merged_model, lora_path, base_vision_tower=base_vision_tower
    )
    print(f"Final mm_vision_tower: {getattr(merged_model.config, 'mm_vision_tower', None)!r}")

    print("Saving merged model...")
    merged_model.save_pretrained(output_path, safe_serialization=True)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    # 复制 mm_projector.bin（如果存在）
    mm_proj = os.path.join(lora_path, "mm_projector.bin")
    if os.path.exists(mm_proj):
        shutil.copy(mm_proj, output_path)
        print("Copied mm_projector.bin")

    # Do NOT overwrite with base config.json — that drops anyres_randomroi settings.
    print(f"Merge completed. Merged model saved to: {output_path}")

if __name__ == "__main__":
    main()