#!/usr/bin/env python3
"""P0 attention-map probe for Triad (anyres + ROI).

FlashAttention generate stays on the normal eval path. Attention maps come from
a read-only Q/K side score (decision token or last prompt token → keys), then
projected onto BASE / ANYRES / every ROI view. No pruning, no eager attn patching.

Default query is ``decision`` (first A/B/Yes/No gen token) for explaining
correct/incorrect answers. Each layer writes large per-view PNGs plus a panel.

Also reports per-view spatial concentration (entropy / border / ROI overlap) and
optional FP/TN/TP casebook sampling for v0 false-positive studies.

Example (stratified):
  python llava_eval/p0_attention_probe.py \\
    --model-path checkpoints/YOUR_CKPT \\
    --data-path evaluation/mvtec \\
    --from question_musc.jsonl \\
    --text_src config:v0 \\
    --num-samples 24 \\
    --force-triad-roi \\
    --query decision \\
    --output-dir LLM_judge_output/p0_attn

Example (FP casebook + mid layers):
  python llava_eval/build_p0_fp_casebook.py \\
    --ans evaluation/mvtec/results/.../ans_mvtec_all_0shot_randomroi_musc_v0.json \\
    --out LLM_judge_output/casebooks/cotmv6_v0_fp_tn_tp_n48.jsonl
  python llava_eval/p0_attention_probe.py \\
    --model-path checkpoints/merged_llava_qwen_triad_9ROI_cotmv6 \\
    --data-path evaluation/mvtec \\
    --from question_musc.jsonl \\
    --text_src config:v0 \\
    --casebook LLM_judge_output/casebooks/cotmv6_v0_fp_tn_tp_n48.jsonl \\
    --layers mid \\
    --force-triad-roi \\
    --query decision \\
    --output-dir LLM_judge_output/p0_attn_cotmv6_fp_mid
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append("./LLaVA-NeXT/")
sys.path.append(os.path.dirname(__file__))

from attn_map_utils import (  # noqa: E402
    build_views,
    llava_anyres_spatial_hw,
    safe_stem,
    save_layer_maps,
    summarize_attention_mass,
    summarize_spatial_views,
)
from evaluation_config import SUPPORTED_QSET  # noqa: E402
from evaluate import data_read, read_mask  # noqa: E402
from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def parse_answer_yesno(text: str) -> Optional[int]:
    m = re.search(r"<answer>\s*(yes|no)\s*</answer>", text, re.I)
    if m:
        return 1 if m.group(1).lower() == "yes" else 0
    t = text.strip().lower()
    if t.startswith("a.") or t == "a" or "answer: a" in t:
        return 1
    if t.startswith("b.") or t == "b" or "answer: b" in t:
        return 0
    if re.search(r"\byes\b", t) and not re.search(r"\bno\b", t):
        return 1
    if re.search(r"\bno\b", t) and not re.search(r"\byes\b", t):
        return 0
    return None


def build_prompt(question_text: str, template_name: str = "qwen_1_5") -> str:
    conv = conv_templates[template_name].copy()
    if "<image>" not in question_text:
        question_text = "<image>\n" + question_text
    conv.append_message(conv.roles[0], question_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def resolve_question(qs_tmp, subset: str, fallback: str) -> str:
    if qs_tmp is None:
        return fallback
    if isinstance(qs_tmp, dict):
        if subset in qs_tmp:
            return qs_tmp[subset]
        if "default" in qs_tmp:
            return qs_tmp["default"]
        raise KeyError(f"subset {subset} not in prompt config")
    return qs_tmp


def stratified_sample(datas: List[dict], num_samples: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for d in datas:
        buckets[(str(d.get("subset", "?")), int(d.get("gt", 0)))].append(d)
    keys = sorted(buckets.keys())
    for k in keys:
        rng.shuffle(buckets[k])
    picked: List[dict] = []
    while len(picked) < num_samples and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(picked) < num_samples:
                picked.append(buckets[k].pop())
    rng.shuffle(picked)
    return picked


def load_casebook(path: str) -> List[dict]:
    """Load FP/TN/TP casebook (jsonl or json list). Each row needs at least ``id``."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    if path.endswith(".jsonl") or (not raw.startswith("[") and not raw.startswith("{")):
        rows = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    data = json.loads(raw)
    if isinstance(data, dict) and "samples" in data:
        return list(data["samples"])
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported casebook format: {path}")


def select_samples_from_casebook(all_datas: List[dict], casebook: List[dict]) -> List[dict]:
    """Match question-file rows by id; attach casebook fields (group, pred_ref, ...)."""
    by_id = {str(d.get("id")): d for d in all_datas}
    out: List[dict] = []
    missing: List[str] = []
    for row in casebook:
        sid = str(row.get("id"))
        if sid not in by_id:
            missing.append(sid)
            continue
        sample = dict(by_id[sid])
        for k, v in row.items():
            if k == "id":
                continue
            sample[k] = v
        out.append(sample)
    if missing:
        raise KeyError(f"casebook ids not found in question file ({len(missing)}): {missing[:8]}")
    return out


def parse_layer_spec(spec: str, n_layers: int) -> List[int]:
    spec = (spec or "default").strip().lower()
    if spec in ("default", "auto"):
        # Decision-explanation default: mid + last 4 layers.
        mid = [n_layers // 2] if n_layers >= 8 else []
        late = [n_layers - 1 - i for i in range(4) if n_layers - 1 - i >= 0]
        return sorted(set(mid + late))
    if spec in ("mid", "middle", "fp_mid"):
        # FP spatial study: early anchor + mid band + last decision layer.
        wanted = [3, 10, 14, 18, n_layers - 1]
        return sorted({i for i in wanted if 0 <= i < n_layers})
    if spec in ("mid_full", "middle_full"):
        wanted = [3, 8, 10, 12, 14, 16, 18, 20, n_layers - 1]
        return sorted({i for i in wanted if 0 <= i < n_layers})
    if spec in ("early", "fastv"):
        return [i for i in (0, 1, 2, 3) if i < n_layers]
    if spec == "last":
        return [n_layers - 1]
    if spec.startswith("last"):
        m = re.match(r"last(\d+)", spec)
        n = int(m.group(1)) if m else 3
        return list(range(max(0, n_layers - n), n_layers))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx = n_layers + idx
        if not (0 <= idx < n_layers):
            raise ValueError(f"layer {part} out of range for n_layers={n_layers}")
        out.append(idx)
    return sorted(set(out))


def _relpath_under_cwd(path: str) -> str:
    abs_p = os.path.abspath(path)
    try:
        return os.path.relpath(abs_p, os.getcwd())
    except Exception:
        return path


def build_summary(per_sample: List[dict], errors: List[dict]) -> dict:
    """Accuracy + confusion + group-wise spatial means (anyres / key layers)."""
    n_ok = sum(1 for r in per_sample if r.get("correct") == 1)
    n_labeled = sum(1 for r in per_sample if r.get("correct") is not None)
    cm = defaultdict(int)
    for r in per_sample:
        cm[(r.get("gt"), r.get("pred"))] += 1
    n_tp = cm[(1, 1)]
    n_tn = cm[(0, 0)]
    n_fp = cm[(0, 1)]
    n_fn = cm[(1, 0)]
    n_pos = n_tp + n_fn
    n_neg = n_tn + n_fp

    def _safe_div(a, b):
        return (a / b) if b else None

    by_group_counts = defaultdict(int)
    for r in per_sample:
        by_group_counts[str(r.get("group") or "none")] += 1

    # Mean anyres spatial metrics by group × layer
    focus_keys = (
        "entropy_norm", "top5pct", "gini", "centroid_offset",
        "border_mass", "corner_mass", "center_mass", "roi_overlap", "mask_overlap",
        "raw_sum",
    )
    spatial_means: Dict[str, dict] = {}
    groups = sorted({str(r.get("group") or "none") for r in per_sample})
    layers = []
    for r in per_sample:
        for lk in (r.get("spatial_focus") or {}):
            if lk not in layers:
                layers.append(lk)
    layers = sorted(layers, key=lambda x: int(x[1:]) if x.startswith("L") and x[1:].isdigit() else x)

    for g in groups:
        rows = [r for r in per_sample if str(r.get("group") or "none") == g]
        spatial_means[g] = {}
        for lk in layers:
            acc = defaultdict(list)
            mass_acc = defaultdict(list)
            for r in rows:
                sp = (r.get("spatial_focus") or {}).get(lk) or {}
                anyres = sp.get("anyres") or {}
                for k in focus_keys:
                    if k in anyres and isinstance(anyres[k], (int, float)):
                        acc[k].append(float(anyres[k]))
                # de-sink visual mass if present
                mass = (r.get("attn_mass") or {}).get(lk) or {}
                bds = mass.get("buckets_desink") or {}
                braw = mass.get("buckets") or {}
                for name, src in (
                    ("visual_all_desink", bds.get("visual_all")),
                    ("visual_all", braw.get("visual_all")),
                    ("token0_mass", mass.get("token0_mass")),
                    ("text_before", braw.get("text_before_visual")),
                    ("query_self", braw.get("query_self")),
                ):
                    if isinstance(src, (int, float)):
                        mass_acc[name].append(float(src))
                # sum all roi_* view raw spatial focus
                roi_entropy = []
                roi_border = []
                for vn, vv in sp.items():
                    if str(vn).startswith("roi_") and isinstance(vv, dict):
                        if "entropy_norm" in vv:
                            roi_entropy.append(float(vv["entropy_norm"]))
                        if "border_mass" in vv:
                            roi_border.append(float(vv["border_mass"]))
                if roi_entropy:
                    acc["roi_entropy_norm"].append(float(np.mean(roi_entropy)))
                if roi_border:
                    acc["roi_border_mass"].append(float(np.mean(roi_border)))
            entry = {
                k: round(float(np.mean(vs)), 6) for k, vs in acc.items() if vs
            }
            entry.update({k: round(float(np.mean(vs)), 6) for k, vs in mass_acc.items() if vs})
            entry["n"] = len(rows)
            spatial_means[g][lk] = entry

    by_subset = {}
    for r in per_sample:
        sub = str(r.get("subset") or "?")
        slot = by_subset.setdefault(sub, {"n": 0, "ok": 0, "fp": 0, "tn": 0, "tp": 0, "fn": 0})
        slot["n"] += 1
        if r.get("correct") == 1:
            slot["ok"] += 1
        gt, pred = r.get("gt"), r.get("pred")
        if gt == 0 and pred == 1:
            slot["fp"] += 1
        elif gt == 0 and pred == 0:
            slot["tn"] += 1
        elif gt == 1 and pred == 1:
            slot["tp"] += 1
        elif gt == 1 and pred == 0:
            slot["fn"] += 1

    return {
        "n_run": len(per_sample),
        "n_errors": len(errors),
        "accuracy_labeled": _safe_div(n_ok, n_labeled),
        "confusion": {"tp": n_tp, "tn": n_tn, "fp": n_fp, "fn": n_fn},
        "tpr_recall": _safe_div(n_tp, n_pos),
        "tnr": _safe_div(n_tn, n_neg),
        "fpr": _safe_div(n_fp, n_neg),
        "by_group_counts": dict(by_group_counts),
        "by_subset": by_subset,
        "spatial_means_anyres_by_group": spatial_means,
        "note": (
            "spatial_means_anyres_by_group: view-normalized anyres concentration "
            "(entropy_norm↑=more diffuse; border/corner_mass=edge attraction; "
            "roi_overlap=mass inside MuSC boxes). buckets_desink zeros token-0 sink."
        ),
    }


# ---------------------------------------------------------------------------
# Segment recording (BASE / ANYRES / ROI lengths from Triad merge)
# ---------------------------------------------------------------------------

class SegmentRecorder:
    """Wrap ``_concat_global_roi`` to record token lengths / ROI shapes / boxes."""

    def __init__(self):
        self.last = None
        self._orig = None
        self._model = None

    def install(self, model):
        if not hasattr(model, "_concat_global_roi"):
            raise RuntimeError("model has no _concat_global_roi")
        self._model = model
        self._orig = model._concat_global_roi
        recorder = self

        def wrapped(
            image_feature,
            extra_feature=None,
            num_base_tokens=None,
            roi_spatial_shape=None,
            bboxes=None,
            image_size=None,
        ):
            n_global = int(image_feature.shape[0]) if image_feature is not None else 0
            if num_base_tokens is None or int(num_base_tokens) <= 0:
                side = int(math.sqrt(n_global))
                n_base = side * side if side * side == n_global else n_global
            else:
                n_base = min(int(num_base_tokens), n_global)
            n_roi = int(extra_feature.shape[0]) if extra_feature is not None and extra_feature.numel() > 0 else 0
            roi_lens: List[int] = []
            if roi_spatial_shape is not None and len(roi_spatial_shape) == 3:
                n_roi_g, rh, rw = [int(x) for x in roi_spatial_shape]
                per = rh * rw
                if n_roi_g > 0 and per > 0 and n_roi_g * per == n_roi:
                    roi_lens = [per] * n_roi_g
            if not roi_lens and n_roi > 0:
                roi_lens = [n_roi]
            bbox_list = []
            if bboxes is not None:
                for b in bboxes:
                    if b is None:
                        continue
                    if isinstance(b, (list, tuple)) and len(b) >= 4:
                        bbox_list.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
            recorder.last = {
                "n_base": n_base,
                "n_anyres": max(n_global - n_base, 0),
                "n_global": n_global,
                "n_roi": n_roi,
                "roi_lens": roi_lens,
                "roi_spatial_shape": list(roi_spatial_shape) if roi_spatial_shape is not None else None,
                "bboxes": bbox_list,
                "image_size": list(image_size) if image_size is not None else None,
            }
            return recorder._orig(
                image_feature,
                extra_feature=extra_feature,
                num_base_tokens=num_base_tokens,
                roi_spatial_shape=roi_spatial_shape,
                bboxes=bboxes,
                image_size=image_size,
            )

        model._concat_global_roi = wrapped
        return self

    def uninstall(self):
        if self._model is not None and self._orig is not None:
            self._model._concat_global_roi = self._orig
        self._model = None
        self._orig = None


def find_visual_span(input_ids: torch.Tensor, n_visual: int) -> Tuple[int, int]:
    ids = input_ids[0].detach().cpu().tolist()
    try:
        start = ids.index(IMAGE_TOKEN_INDEX)
    except ValueError:
        start = 0
    return start, start + n_visual


def find_decision_query(prompt_len: int, gen_token_ids: Sequence[int], tokenizer) -> Optional[int]:
    """Index of the first A/B or Yes/No answer token in the full sequence."""
    if not gen_token_ids:
        return None
    pieces = [tokenizer.decode([tid], skip_special_tokens=False) for tid in gen_token_ids]
    for i, piece in enumerate(pieces):
        p = piece.strip().lower()
        if p in {"a", "b", "a.", "b.", "yes", "no"} or p.startswith("a.") or p.startswith("b."):
            return prompt_len + i
    # Fallback: first non-space generated token.
    for i, piece in enumerate(pieces):
        if piece.strip():
            return prompt_len + i
    return prompt_len


# ---------------------------------------------------------------------------
# Independent Q/K attention (read-only; does not change decoder state)
# ---------------------------------------------------------------------------

def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


@torch.no_grad()
def independent_qk_attention(
    decoder_layer,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask: Optional[torch.Tensor] = None,
    query_index: Optional[int] = None,
) -> torch.Tensor:
    """Return [seq] head-averaged attention from one query to all keys (FP32 softmax).

    Mirrors xxy FastV's read-only Q/K path; does not touch KV cache / hidden states.
    """
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("independent scoring expects batch size 1 [1, L, H]")
    batch, length, _ = hidden_states.shape
    device = hidden_states.device
    positions = torch.arange(length, device=device).expand(batch, -1)
    valid = torch.ones((batch, length), dtype=torch.bool, device=device)
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape != (batch, length):
            raise ValueError("scoring requires a 2D padding mask [B, L]")
        valid = attention_mask.to(device=device, dtype=torch.bool)
    rows = torch.arange(batch, device=device)
    if query_index is None:
        last = positions.masked_fill(~valid, -1).max(dim=-1).values
    else:
        last = torch.full((batch,), int(query_index), device=device, dtype=torch.long)
    if (last < 0).any() or (last >= length).any() or (~valid[rows, last]).any():
        raise ValueError(f"invalid query_index={query_index} for length={length}")

    attention = decoder_layer.self_attn
    normalized = decoder_layer.input_layernorm(hidden_states)
    query = attention.q_proj(normalized[rows, last].unsqueeze(1))
    key = attention.k_proj(normalized)
    query = query.view(batch, 1, attention.num_heads, attention.head_dim).transpose(1, 2)
    key = key.view(batch, length, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)

    cos, sin = (item.to(device=device).expand(batch, -1, -1) for item in position_embeddings)
    if cos.shape[-1] * 2 == query.shape[-1]:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
    query_cos = cos[rows, last][:, None, None, :]
    query_sin = sin[rows, last][:, None, None, :]
    query = query * query_cos + _rotate_half(query) * query_sin
    key = key * cos[:, None] + _rotate_half(key) * sin[:, None]
    groups = getattr(attention, "num_key_value_groups", None)
    if groups is None:
        groups = attention.num_heads // max(attention.num_key_value_heads, 1)
    key = key.repeat_interleave(groups, dim=1)

    logits = torch.matmul(query.float(), key.float().transpose(-2, -1))
    logits = logits.squeeze(2) / math.sqrt(attention.head_dim)
    visible = valid & (positions <= last[:, None])
    finite = logits.masked_select(visible[:, None])
    if finite.numel() and not torch.isfinite(finite).all():
        raise FloatingPointError("Non-finite independent Q/K scores")
    logits = logits.masked_fill(~visible[:, None], float("-inf"))
    return torch.softmax(logits, dim=-1, dtype=torch.float32).mean(dim=1)[0]


def _position_embeddings(model, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    core = model.model
    if hasattr(core, "rotary_emb"):
        return core.rotary_emb(hidden_states, position_ids)
    raise RuntimeError("model.model.rotary_emb not found; cannot score independently")


@torch.no_grad()
def score_layers(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    layer_indices: Sequence[int],
    query_index: Optional[int],
) -> Dict[int, np.ndarray]:
    """One prefill with hidden states, then read-only Q/K per requested layer."""
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states  # [embed, after_L0, ...]
    pos_emb = _position_embeddings(model, hidden_states[0], position_ids)
    out: Dict[int, np.ndarray] = {}
    for li in layer_indices:
        # Attention at layer li uses the layer input = hidden_states[li]
        scores = independent_qk_attention(
            model.model.layers[li],
            hidden_states[li],
            pos_emb,
            attention_mask=attention_mask,
            query_index=query_index,
        )
        out[int(li)] = scores.detach().float().cpu().numpy()
    del outputs
    return out


# ---------------------------------------------------------------------------
# Per-sample run
# ---------------------------------------------------------------------------

def run_one_sample(
    model,
    tokenizer,
    image_processor,
    sample: dict,
    data_dir: str,
    question_text: str,
    *,
    max_new_tokens: int,
    layer_indices: Sequence[int],
    query_mode: str,
    map_dir: str,
    sample_index: int = 0,
    temperature: float = 0.0,
    display_size: int = 768,
    norm: str = "per_view",
) -> dict:
    recorder = SegmentRecorder().install(model)
    try:
        img_rel = sample.get("image") or (sample.get("src") or [None])[0]
        if isinstance(img_rel, list):
            img_rel = img_rel[0]
        image = Image.open(os.path.join(data_dir, img_rel)).convert("RGB")
        image_sizes = [image.size]

        masks = None
        if sample.get("mask"):
            mask_rel = sample["mask"]
            if isinstance(mask_rel, list):
                mask_rel = mask_rel[0]
            masks = [read_mask(os.path.join(data_dir, mask_rel))]
        boxes_list = sample.get("bbox", None)

        images, roi_bboxes = process_images(
            [image],
            image_processor,
            model.config,
            masks=masks,
            boxes_list=boxes_list,
            return_bboxes=True,
        )
        if isinstance(images, list):
            images = [im.to(model.device, dtype=model.dtype) for im in images]
        else:
            images = images.to(model.device, dtype=model.dtype)

        prompt = build_prompt(question_text)
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        (_, position_ids, attention_mask, _, prompt_embeds, _) = model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None,
            images, modalities=["image"], image_sizes=image_sizes, roi_bboxes=roi_bboxes,
        )
        seg = recorder.last
        if seg is None:
            raise RuntimeError(
                "SegmentRecorder got no callback; need image_aspect_ratio=anyres_max_9_randomroi "
                "(pass --force-triad-roi)"
            )

        n_visual_packed = seg["n_global"] + seg["n_roi"]
        prompt_len = int(prompt_embeds.shape[1])
        n_text_tokens = int(input_ids.shape[1] - 1)
        n_visual_actual = prompt_len - n_text_tokens
        if n_visual_actual < n_visual_packed:
            n_visual_actual = n_visual_packed
        trailing = n_visual_actual - n_visual_packed

        vis_start, _ = find_visual_span(input_ids, n_visual_actual)
        vis_end = vis_start + n_visual_actual

        # --- generate on the normal Triad path (FlashAttention OK) ---
        gen_kwargs = {
            "do_sample": temperature > 1e-5,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.7

        out_ids = model.generate(
            inputs=input_ids,
            images=images,
            image_sizes=image_sizes,
            roi_bboxes=roi_bboxes,
            **gen_kwargs,
        )
        # Prefer only newly generated tokens. HF sometimes returns prompt+new;
        # strip prompt when the prefix matches input_ids exactly.
        in_len = int(input_ids.shape[1])
        if (
            out_ids.shape[1] >= in_len
            and torch.equal(out_ids[0, :in_len].detach().cpu(), input_ids[0].detach().cpu())
        ):
            gen_ids = out_ids[:, in_len:]
        else:
            gen_ids = out_ids
        gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        gen_token_list = gen_ids[0].tolist()

        # --- build scoring sequence ---
        if query_mode == "last_prompt":
            score_embeds = prompt_embeds
            score_mask = attention_mask
            if score_mask is None:
                score_mask = torch.ones(score_embeds.shape[:2], dtype=torch.long, device=model.device)
            elif score_mask.ndim != 2:
                score_mask = torch.ones(score_embeds.shape[:2], dtype=torch.long, device=model.device)
            score_pos = position_ids
            if score_pos is None:
                score_pos = torch.arange(prompt_len, device=model.device).unsqueeze(0)
            query_index = None  # last valid prompt token
            query_meta = {"mode": "last_prompt", "index": prompt_len - 1}
        elif query_mode == "decision":
            gen_embeds = model.get_model().embed_tokens(gen_ids.to(model.device))
            score_embeds = torch.cat([prompt_embeds, gen_embeds.to(prompt_embeds.dtype)], dim=1)
            score_mask = torch.ones(score_embeds.shape[:2], dtype=torch.long, device=model.device)
            if position_ids is not None:
                last = position_ids[0, -1]
                gen_pos = last + torch.arange(1, gen_embeds.shape[1] + 1, device=model.device)
                score_pos = torch.cat([position_ids, gen_pos.unsqueeze(0)], dim=1)
            else:
                score_pos = torch.arange(score_embeds.shape[1], device=model.device).unsqueeze(0)
            q = find_decision_query(prompt_len, gen_token_list, tokenizer)
            if q is None:
                q = prompt_len - 1
            query_index = q
            query_meta = {"mode": "decision", "index": int(q)}
        else:
            raise ValueError(f"unknown query_mode {query_mode}")

        layer_scores = score_layers(
            model, score_embeds, score_mask, score_pos, layer_indices, query_index
        )

        # --- geometry / views ---
        vt = model.get_vision_tower()
        patch_size = int(getattr(vt, "patch_size", 14) or 14)
        vis_size = int(getattr(vt, "image_size", 384) or 384)
        pps = int(getattr(vt, "num_patches_per_side", 27) or 27)
        pinpoints = getattr(model.config, "image_grid_pinpoints", None)
        max_patches = 9
        iar = str(getattr(model.config, "image_aspect_ratio", "") or "")
        m = re.search(r"anyres_max_(\d+)", iar)
        if m:
            max_patches = int(m.group(1))
        any_hw = None
        any_hw_method = "none"
        if seg["n_anyres"] > 0:
            computed = llava_anyres_spatial_hw(
                image.size, pinpoints, vis_size, pps,
                max_num_patches=max_patches, n_anyres=seg["n_anyres"],
            )
            if computed is not None:
                any_hw = (computed[0], computed[1])
                any_hw_method = computed[2]

        bboxes = seg.get("bboxes") or []
        if not bboxes and roi_bboxes and roi_bboxes[0]:
            bboxes = roi_bboxes[0]

        views = build_views(
            image_size_wh=image.size,
            n_base=int(seg["n_base"]),
            n_anyres=int(seg["n_anyres"]),
            roi_lens=seg.get("roi_lens") or [],
            roi_spatial_shape=seg.get("roi_spatial_shape"),
            bboxes=bboxes,
            anyres_hw=any_hw,
            patch_size=patch_size,
            vis_start=vis_start,
        )

        stem = safe_stem(sample, sample_index)
        sample_dir = os.path.join(map_dir, f"sample_{stem}")
        os.makedirs(sample_dir, exist_ok=True)

        pred = parse_answer_yesno(gen_text)
        gt = sample.get("gt", None)
        correct = None if gt is None or pred is None else int(pred == int(gt))
        verdict = "unk" if correct is None else ("ok" if correct else "err")
        q_idx = int(query_meta["index"])

        # Defect mask for spatial mask_overlap (TP/FN); good images usually empty.
        defect_mask_arr = None
        if masks is not None and len(masks) > 0 and masks[0] is not None:
            defect_mask_arr = np.asarray(masks[0])
            if defect_mask_arr.ndim == 3:
                defect_mask_arr = defect_mask_arr.max(axis=-1)
            if defect_mask_arr.size and float(np.max(defect_mask_arr)) <= 0:
                defect_mask_arr = None

        # ROI coverage vs full image (for casebook analysis).
        img_w, img_h = image.size
        img_area = max(img_w * img_h, 1)
        roi_fracs = []
        for b in bboxes:
            if b is None or len(b) < 4:
                continue
            x0, y0, x1, y1 = [int(v) for v in b[:4]]
            # inclusive→half-open approx already applied in views; use raw here
            roi_fracs.append(max(0, (x1 - x0) * (y1 - y0)) / img_area)

        saved_maps: Dict[str, dict] = {}
        mass_by_layer: Dict[str, dict] = {}
        spatial_by_layer: Dict[str, dict] = {}
        for li, scores in layer_scores.items():
            visual = scores[vis_start:vis_end]
            if visual.size < n_visual_packed:
                raise ValueError(
                    f"layer {li}: visual slice length {visual.size} < packed {n_visual_packed}"
                )
            vis_for_map = visual[: n_visual_packed + trailing]
            mass = summarize_attention_mass(
                scores,
                views,
                vis_start=vis_start,
                n_visual=n_visual_actual,
                query_index=q_idx,
            )
            spatial = summarize_spatial_views(
                views,
                vis_for_map,
                roi_boxes_xyxy=bboxes,
                defect_mask=defect_mask_arr,
            )
            key = f"L{li}"
            mass_by_layer[key] = mass
            spatial_by_layer[key] = spatial
            tag = f"L{li:02d}_{query_mode}"
            group_tag = sample.get("group")
            title = (
                f"L{li} query={query_mode}@{q_idx} {verdict} "
                f"gt={gt} pred={pred}"
                + (f" group={group_tag}" if group_tag else "")
                + f" ans={gen_text[:40]!r}"
            )
            paths = save_layer_maps(
                sample_dir,
                tag,
                image,
                views,
                vis_for_map,
                title=title,
                display_size=display_size,
                norm=norm,
                mass=mass,
            )
            saved_maps[key] = {k: _relpath_under_cwd(v) for k, v in paths.items()}

        # Infer analysis group if casebook did not set one.
        group = sample.get("group")
        if group is None and gt is not None and pred is not None:
            if int(gt) == 0 and int(pred) == 1:
                group = "FP"
            elif int(gt) == 0 and int(pred) == 0:
                group = "TN"
            elif int(gt) == 1 and int(pred) == 1:
                group = "TP"
            elif int(gt) == 1 and int(pred) == 0:
                group = "FN"

        return {
            "id": sample.get("id"),
            "subset": sample.get("subset"),
            "group": group,
            "gt": gt,
            "pred": pred,
            "pred_ref": sample.get("pred_ref"),
            "correct": correct,
            "answer": gen_text,
            "origin_path": sample.get("origin_path"),
            "roi_area_fracs": [round(float(x), 6) for x in roi_fracs],
            "roi_area_frac_max": round(float(max(roi_fracs)), 6) if roi_fracs else 0.0,
            "query": query_meta,
            "segments": {
                "vis_span": [vis_start, vis_end],
                "n_base": seg["n_base"],
                "n_anyres": seg["n_anyres"],
                "n_roi": seg["n_roi"],
                "roi_lens": seg["roi_lens"],
                "roi_spatial_shape": seg["roi_spatial_shape"],
                "bboxes": bboxes,
                "trailing_newline_tokens": trailing,
                "anyres_hw": list(any_hw) if any_hw else None,
                "anyres_hw_method": any_hw_method,
                "n_roi_views": sum(1 for v in views if str(v["name"]).startswith("roi")),
            },
            "views": views,
            "layers": list(layer_scores.keys()),
            "maps": saved_maps,
            "attn_mass": mass_by_layer,
            "spatial_focus": spatial_by_layer,
            "viz": {"display_size": display_size, "norm": norm},
            "attn_source": "independent_qk",
            "note": (
                "Maps use read-only Q/K at the listed layers; FlashAttention generate "
                "is unchanged. spatial_focus is view-normalized concentration "
                "(entropy/border/roi_overlap). attn_mass.buckets_desink zeros token-0."
            ),
        }
    finally:
        recorder.uninstall()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="P0 Triad attention-map probe")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--ds", type=str, default="mvtec")
    parser.add_argument("--from", dest="question_file", type=str, default="question_musc.jsonl")
    parser.add_argument("--text_src", type=str, default="config:v0")
    parser.add_argument("--subset", type=str, default="all")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--casebook", type=str, default=None,
        help="JSONL/JSON of sample ids (optional group/pred_ref). Overrides --num-samples stratified draw.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force-triad-roi", action="store_true")
    parser.add_argument(
        "--layers", type=str, default="default",
        help="default(=mid+last4)|mid(=3,10,14,18,last)|mid_full|early|fastv|last|lastN|comma list",
    )
    parser.add_argument(
        "--query", type=str, default="decision", choices=["last_prompt", "decision"],
        help="decision (default): first A/B/Yes/No gen token for right/wrong explanation. "
             "last_prompt: final prompt token (FastV-style).",
    )
    parser.add_argument(
        "--display-size", type=int, default=768,
        help="Per-view PNG side length in pixels (source|overlay each).",
    )
    parser.add_argument(
        "--norm", type=str, default="per_view", choices=["per_view", "global"],
        help="Heatmap normalization: per_view (default) or global max across views.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="JSON report path (optional)")
    args = parser.parse_args()

    model_path = args.model_path.rstrip("/")
    # Keep maps next to the report when only --output is given.
    if args.output_dir:
        out_dir = args.output_dir
    elif args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    else:
        out_dir = os.path.join(model_path, "p0_attn_maps")
    map_dir = os.path.join(out_dir, "maps")
    output_path = args.output or os.path.join(out_dir, "p0_attn_report.json")
    os.makedirs(map_dir, exist_ok=True)

    data_dir = args.data_path
    if data_dir.endswith("/") and not data_dir.rstrip("/").endswith(args.ds):
        data_dir = os.path.join(data_dir, args.ds)
    question_path = os.path.join(data_dir, args.question_file)

    qs_tmp = None
    if args.text_src.startswith("config:"):
        ver = args.text_src.split(":", 1)[1]
        qs_tmp = SUPPORTED_QSET[args.ds][ver]
    elif args.text_src.startswith("input:"):
        qs_tmp = args.text_src.split(":", 1)[1]
    elif args.text_src == "dataset":
        qs_tmp = None
    else:
        raise ValueError(f"unknown text_src {args.text_src}")

    all_datas = data_read(question_path)
    if args.subset != "all":
        keep = set(args.subset.split(","))
        all_datas = [d for d in all_datas if d.get("subset") in keep]

    casebook_meta = None
    if args.casebook:
        casebook = load_casebook(args.casebook)
        samples = select_samples_from_casebook(all_datas, casebook)
        casebook_meta = {
            "path": args.casebook,
            "n": len(samples),
            "groups": sorted({str(s.get("group") or "none") for s in samples}),
        }
        print(f"[P0] casebook {args.casebook}: {len(samples)} samples groups={casebook_meta['groups']}")
    else:
        samples = stratified_sample(all_datas, args.num_samples, args.seed)
        print(f"[P0] sampled {len(samples)} / {len(all_datas)} from {question_path}")

    if args.model_name is None:
        args.model_name = os.path.basename(model_path)

    overwrite_config = {}
    if args.force_triad_roi:
        overwrite_config.update({
            "image_aspect_ratio": "anyres_max_9_randomroi",
            "mm_patch_merge_type": "spatial_avgpool_auto_unpad_randomroi_newl",
        })

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        args.model_name,
        device_map=None,
        device=args.device,
        attn_implementation="flash_attention_2",
        overwrite_config=overwrite_config or None,
    )
    model = model.to(args.device)
    model.eval()

    n_layers = len(model.model.layers)
    layer_indices = parse_layer_spec(args.layers, n_layers)
    print(
        f"[P0] layers={layer_indices} query={args.query} "
        f"attn={getattr(model.config, '_attn_implementation', None)} "
        f"iar={getattr(model.config, 'image_aspect_ratio', None)}"
    )

    per_sample = []
    errors = []
    for si, sample in enumerate(tqdm(samples, desc="P0 attn maps")):
        subset = sample.get("subset", "default")
        fallback = sample.get("conversations", [{}])[0].get("value", sample.get("text", ""))
        question = resolve_question(qs_tmp, subset, fallback)
        try:
            per_sample.append(
                run_one_sample(
                    model, tokenizer, image_processor, sample, data_dir, question,
                    max_new_tokens=args.max_new_tokens,
                    layer_indices=layer_indices,
                    query_mode=args.query,
                    map_dir=map_dir,
                    sample_index=si,
                    temperature=args.temperature,
                    display_size=args.display_size,
                    norm=args.norm,
                )
            )
        except Exception as e:
            errors.append({"id": sample.get("id"), "error": repr(e)})
            print(f"[P0][ERR] {sample.get('id')}: {e}")

    summary = build_summary(per_sample, errors)
    report = {
        "meta": {
            "model_path": model_path,
            "data_path": data_dir,
            "question_file": args.question_file,
            "text_src": args.text_src,
            "num_samples": len(samples),
            "num_run": len(per_sample),
            "num_errors": len(errors),
            "seed": args.seed,
            "casebook": casebook_meta,
            "layers": layer_indices,
            "layers_spec": args.layers,
            "query": args.query,
            "display_size": args.display_size,
            "norm": args.norm,
            "attn_implementation": "flash_attention_2",
            "attn_source": "independent_qk",
            "force_triad_roi": bool(args.force_triad_roi),
            "image_aspect_ratio": getattr(model.config, "image_aspect_ratio", None),
            "mm_patch_merge_type": getattr(model.config, "mm_patch_merge_type", None),
            "map_dir": _relpath_under_cwd(map_dir),
            "note": (
                "Decision-token attention overlays for FP/TN/TP analysis (v0). "
                "Generation uses FlashAttention2; maps use read-only Q/K (FP32). "
                "spatial_focus: per-view concentration; attn_mass.buckets_desink zeros token-0."
            ),
        },
        "summary": summary,
        "errors": errors,
        "per_sample": per_sample,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[P0] maps -> {map_dir}")
    print(f"[P0] report -> {output_path}")
    if summary.get("accuracy_labeled") is not None:
        print(
            f"[P0] labeled accuracy {summary['confusion']['tp'] + summary['confusion']['tn']}"
            f"/{summary['n_run']} = {summary['accuracy_labeled']:.3f} "
            f"TNR={summary.get('tnr')} FPR={summary.get('fpr')} "
            f"groups={summary.get('by_group_counts')}"
        )


if __name__ == "__main__":
    main()
