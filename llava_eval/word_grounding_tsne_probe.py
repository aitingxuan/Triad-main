#!/usr/bin/env python3
"""Word-level reasoning grounding + t-SNE diagnostic probe (single file, read-only).

What it does
------------
For each generated long CoT (<think>...</think><location>...</location><type>...</type>),
this probe treats every *content word* of the reasoning as a separate attention
query, reads that word's image attention map (read-only Q/K, one forward for all
words via ``attn_crop.score_layers_multi``), and scores each map against the
MVTec ground-truth defect mask (per pixel, no MuSc):

  * spearman  : rank correlation between the word map and the GT mask
  * dice@k    : overlap of the top-k cells of map vs GT
  * point     : is argmax(map) inside the GT region
  * mass_gt   : fraction of the word's attention mass inside the GT region
  * *_null    : identical metrics against the GT mask cyclically shifted by
                (G/2, G/2) -> a same-shape "wrong location" control.

All words across all samples are then pooled and embedded (PCA -> t-SNE/UMAP)
and reported per role (type / region / object / process / filler / other),
including downstream classifiers (role separability, sample-outcome AUC) so the
visualisation becomes evidence rather than a pretty plot.

IMPORTANT design choices
------------------------
* GT is used for SCORING ONLY. Images are fed with ``masks=None`` (no GT/MuSc
  ROI) so the attention is never biased toward the answer key.
* By default the CoT is re-generated under the same (GT-free) input so that the
  scored text and the scored context are consistent. Use ``--use-stored`` to
  score the CoT text stored in the ans JSON instead (faster, but that text was
  produced under whatever ROI the eval used).
* Attention maps are averaged over the requested layer band (default ``midband``)
  before scoring; pass ``--layers last4`` etc. to change.

No existing repo file is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

# ---------------------------------------------------------------------------
# Import path setup (mirror p0_attention_probe / causal_probe_common)
# ---------------------------------------------------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
for _p in (_THIS, _REPO, os.path.join(_REPO, "LLaVA-NeXT")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kw):
        return x

from attn_crop import (  # noqa: E402
    score_layers_multi,
    resolve_attn_layers,
    SegmentRecorder,
    find_visual_span,
)
from attn_map_utils import (  # noqa: E402
    build_views,
    view_raw_spatial_grid,
    llava_anyres_spatial_hw,
)
from tracemad.parse import parse_gpt_output  # noqa: E402
from tracemad.taxonomy import canonicalize_type_tokens, extract_region_tags  # noqa: E402

from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402


# ---------------------------------------------------------------------------
# Lexicons for word roles
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "is", "are", "was", "were",
    "be", "been", "being", "am", "do", "does", "did", "has", "have", "had", "having",
    "i", "we", "you", "it", "its", "they", "them", "their", "he", "she", "his", "her",
    "and", "or", "but", "if", "then", "so", "as", "of", "to", "in", "on", "at", "by",
    "for", "with", "from", "into", "onto", "over", "under", "about", "above", "below",
    "between", "through", "during", "before", "after", "up", "down", "out", "off",
    "here", "there", "where", "when", "which", "who", "what", "why", "how", "also",
    "not", "no", "yes", "can", "could", "may", "might", "will", "would", "shall",
    "should", "must", "appear", "appears", "appeared", "seem", "seems", "look",
    "looks", "looking", "seen", "see", "visible", "visibly", "very", "more", "most",
    "less", "least", "some", "any", "all", "each", "every", "other", "such", "than",
    "too", "only", "just", "however", "therefore", "thus", "overall", "image",
    "picture", "photo", "shown", "show", "shows", "object", "product", "item",
}

TAG_WORDS = {"think", "location", "type", "answer", "yes", "no"}

PROCESS_WORDS = {
    "process", "processing", "stage", "step", "steps", "filling", "fill", "filled",
    "pasteurization", "pasteurize", "pasteurized", "assembly", "assemble", "assembled",
    "production", "produce", "produced", "manufacture", "manufacturing", "manufactured",
    "molding", "mold", "molded", "annealing", "anneal", "annealed", "juicing", "juice",
    "crush", "crushed", "press", "pressed", "extract", "extraction", "filter", "filtered",
    "filtering", "seal", "sealed", "sealing", "labeling", "labelling", "print", "printed",
    "printing", "design", "apply", "applied", "applying", "wash", "washing", "peel",
    "peeled", "sterilize", "sterilized", "cooling", "cool", "cooled", "blend", "blending",
    "blended", "selection", "select", "selected", "inspect", "inspection", "pack",
    "packing", "packaging", "transport", "machining", "machined", "cutting", "cut",
}

OBJECT_WORDS = {
    "bottle", "cap", "lid", "label", "liquid", "juice", "glass", "screw", "thread",
    "nut", "pill", "capsule", "carpet", "grid", "tile", "wood", "zipper", "cable",
    "toothbrush", "transistor", "hazelnut", "leather", "banana", "cherry", "orange",
    "brush", "bristle", "connector", "wire", "metal", "surface", "head", "neck",
    "tip", "body", "shaft", "handle", "rim", "edge", "corner", "side", "band",
    "foil", "paper", "sleeve", "tube", "ring", "base", "top", "bottom",
}

ROLE_ORDER = ["type", "region", "object", "process", "filler", "other"]
ROLE_COLORS = {
    "type": "#d62728",
    "region": "#ff7f0e",
    "object": "#2ca02c",
    "process": "#1f77b4",
    "filler": "#7f7f7f",
    "other": "#9467bd",
}


# ---------------------------------------------------------------------------
# Prompt / image / GT helpers (no imports from evaluate to keep this standalone)
# ---------------------------------------------------------------------------

def build_prompt(question_text: str, template_name: str = "qwen_1_5") -> str:
    conv = conv_templates[template_name].copy()
    if "<image>" not in question_text:
        question_text = "<image>\n" + question_text
    conv.append_message(conv.roles[0], question_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def resolve_gt_mask_path(origin_path: str, mvtec_root: str) -> Optional[str]:
    """Map 'screw/test/scratch_neck/006.png' -> GT mask file (good/ok -> None)."""
    if not origin_path or not mvtec_root:
        return None
    parts = origin_path.replace("\\", "/").split("/")
    if len(parts) < 4:
        return None
    cls, _split, defect, fn = parts[0], parts[1], parts[2], parts[3]
    if defect in ("good", "ok", "normal"):
        return None
    base, _ = os.path.splitext(fn)
    cands = [
        os.path.join(mvtec_root, cls, "ground_truth", defect, f"{base}_mask.png"),
        os.path.join(mvtec_root, cls, "ground_truth", defect, fn),
        os.path.join(mvtec_root, cls, "ground_truth", defect, f"{base}.png"),
        os.path.join(mvtec_root, cls, "ground_truth", defect, f"{base}_mask.jpg"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return None


def load_binary_mask(path: str, size_wh: Tuple[int, int]) -> np.ndarray:
    w, h = int(size_wh[0]), int(size_wh[1])
    arr = np.array(Image.open(path).convert("L"))
    if arr.shape[0] != h or arr.shape[1] != w:
        arr = np.array(Image.fromarray(arr).resize((w, h), Image.Resampling.NEAREST))
    return (arr > 0).astype(np.float32)


def load_image(sample: dict, data_path: str, mvtec_root: str) -> Image.Image:
    cands: List[str] = []
    if sample.get("image"):
        cands.append(os.path.join(data_path, sample["image"]))
    op = sample.get("origin_path")
    if op:
        cands.append(os.path.join(mvtec_root, op))
        cands.append(os.path.join(data_path, op))
    for p in cands:
        if p and os.path.isfile(p):
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(
        "image not found; tried: " + " | ".join(c for c in cands if c)
    )


# ---------------------------------------------------------------------------
# Word -> token mapping + role classification
# ---------------------------------------------------------------------------

def _subset_vocab(subset: Optional[str]) -> set:
    out = set(OBJECT_WORDS)
    if subset:
        for part in re.split(r"[^a-z0-9]+", str(subset).lower()):
            if part:
                out.add(part)
    return out


def classify_role(word: str, subset: Optional[str]) -> Optional[str]:
    w = word.lower()
    if w in TAG_WORDS:
        return None
    if w in STOPWORDS:
        return "filler"
    if extract_region_tags(w):
        return "region"
    if canonicalize_type_tokens(w):
        return "type"
    if w in _subset_vocab(subset):
        return "object"
    if w in PROCESS_WORDS:
        return "process"
    return "other"


def decode_with_token_spans(ids: Sequence[int], tokenizer) -> Tuple[str, List[Tuple[int, int]]]:
    """Decode token ids to text while recording each token's char span."""
    pieces = [tokenizer.decode([int(t)], skip_special_tokens=False) for t in ids]
    spans: List[Tuple[int, int]] = []
    c = 0
    for p in pieces:
        spans.append((c, c + len(p)))
        c += len(p)
    return "".join(pieces), spans


def extract_words(text: str, token_spans: Sequence[Tuple[int, int]], subset: Optional[str],
                  max_words: int = 0) -> List[dict]:
    """Content words with their role and the token indices (relative to CoT) they occupy."""
    words: List[dict] = []
    for m in re.finditer(r"[A-Za-z][A-Za-z\-]{1,}", text):
        w = m.group(0)
        c0, c1 = m.span()
        role = classify_role(w, subset)
        if role is None:
            continue
        tok_idxs = [
            i for i, (s, e) in enumerate(token_spans)
            if e > s and s < c1 and e > c0
        ]
        if not tok_idxs:
            continue
        words.append({"word": w, "role": role, "tok_idxs": tok_idxs})
    if max_words and 0 < max_words < len(words):
        step = len(words) / float(max_words)
        words = [words[int(i * step)] for i in range(max_words)]
    return words


# ---------------------------------------------------------------------------
# Grid metrics
# ---------------------------------------------------------------------------

def _resize_grid(g: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(g, dtype=np.float32)
    return np.asarray(
        Image.fromarray(arr, mode="F").resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def _rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    order = x.argsort(kind="mergesort")
    xs = x[order]
    ranks = np.empty(x.size, dtype=np.float64)
    i, n = 0, x.size
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64) - float(np.mean(a))
    b = np.asarray(b, np.float64) - float(np.mean(b))
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_rankdata(a), _rankdata(b))


def _topk_mask(g: np.ndarray, k: int) -> np.ndarray:
    flat = np.asarray(g, np.float64).ravel()
    k = max(1, min(int(k), flat.size))
    idx = np.argpartition(flat, -k)[-k:]
    m = np.zeros(flat.size, dtype=bool)
    m[idx] = True
    return m.reshape(np.asarray(g).shape)


def compute_metrics(H: np.ndarray, M: np.ndarray, Mbin: np.ndarray,
                    k_frac: float) -> Dict[str, float]:
    """Grounding metrics of one word map H (sums to ~1) vs GT map M/Mbin at same GxG."""
    N = H.size
    k = max(1, int(math.ceil(float(k_frac) * N)))
    sp = _spearman(H, M)
    tH = _topk_mask(H, k)
    tM = _topk_mask(M, k)
    dice = float((tH & tM).sum()) / float(k)
    argmax = int(np.argmax(H))
    point = 1.0 if bool(Mbin.ravel()[argmax]) else 0.0
    mass_gt = float(H[Mbin].sum()) if bool(Mbin.any()) else 0.0
    return {"spearman": sp, "dice": dice, "point": point, "mass_gt": mass_gt}


def _shift_mask(M: np.ndarray, di: int, dj: int) -> np.ndarray:
    return np.roll(np.roll(M, int(di), axis=0), int(dj), axis=1)


# ---------------------------------------------------------------------------
# Per-sample attention extraction
# ---------------------------------------------------------------------------

def _select_main_view(views: Sequence[dict]) -> Optional[dict]:
    for name in ("anyres", "base"):
        for v in views:
            if str(v.get("name")) == name:
                return v
    return views[0] if views else None


def _generate_ids(model, tokenizer, input_ids, images, image_sizes, roi_bboxes,
                  max_new_tokens: int) -> torch.Tensor:
    out = model.generate(
        inputs=input_ids,
        images=images,
        image_sizes=image_sizes,
        roi_bboxes=roi_bboxes,
        do_sample=False,
        max_new_tokens=int(max_new_tokens),
        use_cache=True,
    )
    in_len = int(input_ids.shape[1])
    if (out.shape[1] >= in_len
            and torch.equal(out[0, :in_len].detach().cpu(), input_ids[0].detach().cpu())):
        return out[:, in_len:]
    return out


@torch.no_grad()
def run_one_sample(model, tokenizer, image_processor, sample: dict, args,
                   layer_indices: Sequence[int], gt_mask_path: Optional[str]) -> Optional[dict]:
    recorder = SegmentRecorder().install(model)
    try:
        subset = sample.get("subset")
        gt = sample.get("gt")
        image = load_image(sample, args.data_path, args.mvtec_root)
        image_sizes = [image.size]

        # GT is for SCORING ONLY: feed images with masks=None (no ROI leakage).
        images, roi_bboxes = process_images(
            [image], image_processor, model.config,
            masks=None, boxes_list=None, return_bboxes=True,
        )
        if isinstance(images, list):
            images = [im.to(model.device, dtype=model.dtype) for im in images]
        else:
            images = images.to(model.device, dtype=model.dtype)

        question = ""
        convs = sample.get("conversations") or []
        if convs:
            question = convs[0].get("value", "") or ""
        prompt = build_prompt(question)
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        (_, position_ids, attention_mask, _, prompt_embeds, _) = \
            model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, images,
                modalities=["image"], image_sizes=image_sizes, roi_bboxes=roi_bboxes,
            )
        seg = recorder.last
        if seg is None:
            raise RuntimeError(
                "SegmentRecorder got no callback; load a Triad anyres ckpt and pass --force-triad-roi"
            )

        n_packed = int(seg["n_global"]) + int(seg["n_roi"])
        prompt_len = int(prompt_embeds.shape[1])
        n_text = int(input_ids.shape[1]) - 1
        n_visual = prompt_len - n_text
        if n_visual < n_packed:
            n_visual = n_packed
        trailing = n_visual - n_packed
        vis_start, _ = find_visual_span(input_ids, n_visual)
        vis_end = vis_start + n_visual

        # --- CoT text (stored in ans JSON, or freshly generated) ---
        if args.use_stored:
            cot_text_clean = (convs[1].get("value", "") if len(convs) > 1 else "") or ""
        else:
            max_ctx = int(getattr(model.config, "max_position_embeddings", 32768) or 32768)
            room = max_ctx - prompt_len - 8
            max_new = min(int(args.max_new_tokens), max(0, room))
            if max_new <= 0:
                return None
            gen_ids = _generate_ids(
                model, tokenizer, input_ids, images, image_sizes, roi_bboxes, max_new,
            )
            cot_text_clean = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        if not cot_text_clean.strip():
            return None
        # Re-tokenize with char offsets so word char-spans map exactly onto the
        # tokens we embed (self-consistent, independent of generation quirks).
        enc = tokenizer(cot_text_clean, add_special_tokens=False, return_offsets_mapping=True)
        cot_ids = list(enc["input_ids"])
        token_spans = list(enc["offset_mapping"])
        cot_text_raw = cot_text_clean

        parsed = parse_gpt_output(cot_text_clean)

        # --- words -> token positions ---
        words = extract_words(cot_text_raw, token_spans, subset, max_words=args.max_words)
        if not words:
            return None

        cot_ids_t = torch.tensor([cot_ids], device=model.device, dtype=torch.long)
        gen_emb = model.get_model().embed_tokens(cot_ids_t).to(prompt_embeds.dtype)
        score_embeds = torch.cat([prompt_embeds, gen_emb], dim=1)
        score_mask = torch.ones(score_embeds.shape[:2], dtype=torch.long, device=model.device)
        if position_ids is not None:
            last = position_ids[0, -1]
            gen_pos = last + torch.arange(1, gen_emb.shape[1] + 1, device=model.device)
            score_pos = torch.cat([position_ids, gen_pos.unsqueeze(0)], dim=1)
        else:
            score_pos = torch.arange(score_embeds.shape[1], device=model.device).unsqueeze(0)

        query_indices: List[int] = []
        for w in words:
            w["row_slots"] = []
            for ti in w["tok_idxs"]:
                pos = prompt_len + int(ti)
                if 0 <= pos < score_embeds.shape[1]:
                    w["row_slots"].append(len(query_indices))
                    query_indices.append(pos)
        words = [w for w in words if w["row_slots"]]
        if not words or not query_indices:
            return None

        rows_by_layer = score_layers_multi(
            model, score_embeds, score_mask, score_pos,
            list(layer_indices), query_indices,
        )
        if not rows_by_layer:
            return None
        # average attention rows across the requested layer band
        R = np.mean([rows_by_layer[k] for k in sorted(rows_by_layer)], axis=0)  # [n_q, L]

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
            anyres_hw=_anyres_hw(model, image, seg),
            patch_size=_patch_meta(model)[0],
            vis_start=vis_start,
        )
        main_view = _select_main_view(views)
        if main_view is None:
            return None
        visual = R[:, vis_start:vis_end]
        if visual.shape[1] < n_packed:
            return None
        vis_for_map = visual[:, : n_packed + trailing]

        G = int(args.grid)
        word_records: List[dict] = []
        Hs: List[np.ndarray] = []
        for w in words:
            r = vis_for_map[w["row_slots"]].mean(axis=0)
            grid = view_raw_spatial_grid(main_view, r)
            H = _resize_grid(grid, G)
            H = np.maximum(H, 0.0)
            s = float(H.sum())
            if s <= 0:
                H = np.full((G, G), 1.0 / (G * G), dtype=np.float32)
            else:
                H = (H / s).astype(np.float32)
            rec = {
                "word": w["word"],
                "role": w["role"],
                "n_tokens": len(w["tok_idxs"]),
            }
            if gt_mask_path:
                w["H"] = H
            word_records.append(rec)
            Hs.append(H)

        metrics_ok = False
        if gt_mask_path:
            m_full = load_binary_mask(gt_mask_path, image.size)          # HxW {0,1}
            M = _resize_grid(m_full, G)
            M = np.clip(M / max(float(M.max()), 1e-6), 0.0, 1.0).astype(np.float32)
            Mbin = (M >= 0.5)
            if bool(Mbin.any()):
                metrics_ok = True
                Msh = _shift_mask(M, G // 2, G // 2)
                Mbin_sh = (Msh >= 0.5)
                for rec, H in zip(word_records, Hs):
                    m = compute_metrics(H, M, Mbin, args.topk_frac)
                    rec.update(m)
                    mn = compute_metrics(H, Msh, Mbin_sh, args.topk_frac)
                    rec["null_spearman"] = mn["spearman"]
                    rec["null_dice"] = mn["dice"]
                    rec["null_point"] = mn["point"]
                    rec["null_mass_gt"] = mn["mass_gt"]

        # sample group / outcome
        group = None
        if gt is not None:
            if int(gt) == 0:
                group = "normal"
            else:
                group = "defect"

        return {
            "id": sample.get("id"),
            "subset": subset,
            "gt": gt,
            "group": group,
            "origin_path": sample.get("origin_path"),
            "answer": parsed.get("answer"),
            "think_words": len(word_records),
            "vis_span": [int(vis_start), int(vis_end)],
            "n_visual": int(n_visual),
            "n_packed": int(n_packed),
            "layers": list(map(int, sorted(rows_by_layer))),
            "has_gt": bool(metrics_ok),
            "words": word_records,
            "_Hs": Hs,
        }
    finally:
        recorder.uninstall()


def _patch_meta(model) -> Tuple[int, int, int]:
    vt = model.get_vision_tower()
    patch_size = int(getattr(vt, "patch_size", 14) or 14)
    vis_size = int(getattr(vt, "image_size", 384) or 384)
    pps = int(getattr(vt, "num_patches_per_side", 27) or 27)
    return patch_size, vis_size, pps


def _anyres_hw(model, image: Image.Image, seg: dict) -> Optional[Tuple[int, int]]:
    if int(seg.get("n_anyres", 0)) <= 0:
        return None
    patch_size, vis_size, pps = _patch_meta(model)
    pinpoints = getattr(model.config, "image_grid_pinpoints", None)
    max_patches = 9
    m = re.search(r"anyres_max_(\d+)", str(getattr(model.config, "image_aspect_ratio", "") or ""))
    if m:
        max_patches = int(m.group(1))
    computed = llava_anyres_spatial_hw(
        image.size, pinpoints, vis_size, pps,
        max_num_patches=max_patches, n_anyres=int(seg["n_anyres"]),
    )
    return (computed[0], computed[1]) if computed is not None else None


# ---------------------------------------------------------------------------
# Aggregation / normalisation of g
# ---------------------------------------------------------------------------

def finalize_g(rows: List[dict]) -> None:
    """Attach global robust z-based g / g_null / g_gain to every word row (gt only)."""
    keys = ["spearman", "dice", "point", "mass_gt"]
    null_keys = ["null_spearman", "null_dice", "null_point", "null_mass_gt"]
    vals = [[r[k] for r in rows if r.get(k) is not None] for k in keys]
    stats = []
    for vs in vals:
        if vs:
            arr = np.asarray(vs, dtype=np.float64)
            stats.append((float(arr.mean()), float(arr.std()) or 1.0))
        else:
            stats.append((0.0, 1.0))

    def z_of(v, i):
        mu, sd = stats[i]
        return (float(v) - mu) / sd

    for r in rows:
        if all(r.get(k) is not None for k in keys):
            g = float(np.mean([z_of(r[k], i) for i, k in enumerate(keys)]))
            g_null = float(np.mean([
                (float(r[nk]) - stats[i][0]) / stats[i][1]
                for i, nk in enumerate(null_keys)
            ]))
            r["g"] = g
            r["g_null"] = g_null
            r["g_gain"] = g - g_null


# ---------------------------------------------------------------------------
# t-SNE + downstream classifiers (best-effort; skipped if deps missing)
# ---------------------------------------------------------------------------

def run_tsne_and_stats(all_rows: List[dict], X: np.ndarray, out_dir: str,
                       args) -> dict:
    """t-SNE/UMAP embedding + role separability + sample-outcome AUC."""
    stats: dict = {"enabled": False}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except Exception as e:  # pragma: no cover
        stats["enabled"] = False
        stats["error"] = repr(e)
        print(f"[tsne] sklearn/matplotlib unavailable, skipping plot: {e}")
        return stats

    n = X.shape[0]
    if n < 8:
        stats["enabled"] = False
        stats["error"] = "too few words"
        return stats

    roles = np.array([r.get("role", "other") for r in all_rows], dtype=object)
    gvals = np.array([
        (r.get("g") if r.get("g") is not None else np.nan) for r in all_rows
    ], dtype=np.float64)

    # PCA reduce then t-SNE
    k = int(min(args.pca_dim, X.shape[1], n - 1))
    Xp = StandardScaler().fit_transform(X)
    Xp = PCA(n_components=max(2, k), random_state=0).fit_transform(Xp)

    idx = np.arange(n)
    if args.tsne_max and n > args.tsne_max:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n, size=args.tsne_max, replace=False)
    perplexity = float(min(args.perplexity, max(5, len(idx) / 4.0)))
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca",
               learning_rate="auto", random_state=args.seed).fit_transform(Xp[idx])

    # ---- plot: role (left) + g (right) ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sub_roles = roles[idx]
    ax = axes[0]
    for role in ROLE_ORDER:
        m = sub_roles == role
        if not m.any():
            continue
        ax.scatter(emb[m, 0], emb[m, 1], s=14, alpha=0.7,
                   c=ROLE_COLORS.get(role, "#333333"), label=f"{role} ({int(m.sum())})")
    ax.set_title("Word attention maps (color = role)")
    ax.legend(markerscale=2, fontsize=9, loc="best")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    sub_g = gvals[idx]
    finite = np.isfinite(sub_g)
    sc = ax.scatter(emb[finite, 0], emb[finite, 1], s=14, alpha=0.8,
                    c=sub_g[finite], cmap="viridis")
    ax.scatter(emb[~finite, 0], emb[~finite, 1], s=10, alpha=0.3, c="lightgray")
    if bool(finite.any()):
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="grounding g (gt only)")
    ax.set_title("Word attention maps (color = grounding g)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    tsne_path = os.path.join(out_dir, "tsne.png")
    fig.savefig(tsne_path, dpi=160)
    plt.close(fig)

    # ---- role boxplot of g ----
    try:
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        data, labels = [], []
        for role in ROLE_ORDER:
            m = (roles == role) & np.isfinite(gvals)
            if m.any():
                data.append(gvals[m]); labels.append(role)
        if data:
            try:
                ax2.boxplot(data, tick_labels=labels, showfliers=False)
            except TypeError:  # older matplotlib
                ax2.boxplot(data, labels=labels, showfliers=False)
            ax2.axhline(0.0, color="gray", lw=0.8, ls="--")
            ax2.set_ylabel("grounding g")
            ax2.set_title("Grounding by word role")
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "role_grounding.png"), dpi=160)
        plt.close(fig2)
    except Exception as e:
        stats["role_plot_error"] = repr(e)

    # ---- role separability (macro-F1 vs chance) ----
    stats["enabled"] = True
    stats["tsne_path"] = tsne_path
    stats["n_words"] = int(n)
    stats["perplexity"] = perplexity
    role_counts = {r: int((roles == r).sum()) for r in ROLE_ORDER if (roles == r).any()}
    stats["role_counts"] = role_counts
    # keep roles with >= 2 samples and >= 2 classes
    keep = np.array([role_counts.get(r, 0) >= 5 for r in roles])
    try:
        if keep.sum() >= 10 and len(set(roles[keep])) >= 2:
            clf = LogisticRegression(max_iter=1000)
            cv = StratifiedKFold(n_splits=min(5, int(min(np.bincount(
                np.unique(roles[keep], return_inverse=True)[1]).min(), 5))), shuffle=True,
                random_state=args.seed)
            f1 = cross_val_score(clf, Xp[keep], roles[keep], cv=cv, scoring="f1_macro")
            stats["role_macro_f1"] = float(np.mean(f1))
            stats["role_macro_f1_std"] = float(np.std(f1))
    except Exception as e:
        stats["role_clf_error"] = repr(e)

    # ---- sample-outcome AUC (aggregate word embeddings per sample) ----
    try:
        by_sample: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(all_rows):
            by_sample[str(r.get("id"))].append(i)
        Xs, ys = [], []
        for sid, ii in by_sample.items():
            lab = all_rows[ii[0]].get("_sample_correct")
            if lab is None:
                continue
            Xs.append(Xp[ii].mean(axis=0))
            ys.append(int(lab))
        if len(set(ys)) == 2 and len(ys) >= 8:
            Xs = np.asarray(Xs); ys = np.asarray(ys)
            cv = StratifiedKFold(n_splits=min(5, int(min(np.bincount(ys).min(), 5))),
                                 shuffle=True, random_state=args.seed)
            auc = cross_val_score(LogisticRegression(max_iter=1000),
                                  Xs, ys, cv=cv, scoring="roc_auc")
            stats["sample_outcome_auc"] = float(np.mean(auc))
            stats["sample_outcome_auc_std"] = float(np.std(auc))
            stats["n_samples_clf"] = int(len(ys))
    except Exception as e:
        stats["outcome_clf_error"] = repr(e)

    return stats


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _perm_pvalue(a: List[float], b: List[float], n_perm: int = 5000, seed: int = 0) -> Optional[float]:
    a = [x for x in a if np.isfinite(x)]
    b = [x for x in b if np.isfinite(x)]
    if len(a) < 3 or len(b) < 3:
        return None
    a = np.asarray(a); b = np.asarray(b)
    obs = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    na = a.size
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = float(pooled[:na].mean() - pooled[na:].mean())
        if diff >= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def build_summary(all_rows: List[dict], per_sample: List[dict], errors: List[dict]) -> dict:
    by_role: Dict[str, List[float]] = defaultdict(list)
    gain_by_role: Dict[str, List[float]] = defaultdict(list)
    gs_all: List[float] = []
    gnull_all: List[float] = []
    gain_all: List[float] = []
    for r in all_rows:
        if r.get("g") is not None and np.isfinite(r["g"]):
            by_role[r["role"]].append(float(r["g"]))
            gs_all.append(float(r["g"]))
        if r.get("g_gain") is not None and np.isfinite(r["g_gain"]):
            gain_by_role[r["role"]].append(float(r["g_gain"]))
            gain_all.append(float(r["g_gain"]))
        if r.get("g_null") is not None and np.isfinite(r["g_null"]):
            gnull_all.append(float(r["g_null"]))
    role_mean = {k: float(np.mean(v)) for k, v in by_role.items() if v}
    role_n = {k: len(v) for k, v in by_role.items()}
    role_gain = {k: float(np.mean(v)) for k, v in gain_by_role.items() if v}
    evidence = [g for role in ("type", "region", "object") for g in by_role.get(role, [])]
    control = [g for role in ("filler", "process") for g in by_role.get(role, [])]
    p = _perm_pvalue(evidence, control, seed=0)
    return {
        "n_samples_run": len(per_sample),
        "n_samples_with_gt": sum(1 for s in per_sample if s.get("has_gt")),
        "n_words_total": len(all_rows),
        "n_words_with_gt": len(gs_all),
        "n_errors": len(errors),
        "role_counts": role_n,
        "role_mean_g": {k: round(v, 4) for k, v in role_mean.items()},
        "role_mean_g_gain": {k: round(v, 4) for k, v in role_gain.items()},
        "mean_g": round(float(np.mean(gs_all)), 4) if gs_all else None,
        "mean_g_null": round(float(np.mean(gnull_all)), 4) if gnull_all else None,
        "mean_g_gain": round(float(np.mean(gain_all)), 4) if gain_all else None,
        "frac_words_above_null": (
            round(float(np.mean([x > 0 for x in gain_all])), 4) if gain_all else None
        ),
        "evidence_mean_g": round(float(np.mean(evidence)), 4) if evidence else None,
        "control_mean_g": round(float(np.mean(control)), 4) if control else None,
        "evidence_minus_control": (
            round(float(np.mean(evidence)) - float(np.mean(control)), 4)
            if evidence and control else None
        ),
        "perm_p_evidence_gt_control": p,
        "note": (
            "g = mean global z of (spearman, dice@k, point, mass_gt); g_gain = g - g_null "
            "(null = GT cyclically shifted by G/2). evidence=type+region+object, "
            "control=filler+process. GT used for scoring only; images fed with no mask/ROI."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Word-level reasoning grounding + t-SNE probe")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--data-path", default="evaluation/mvtec",
                    help="dir joined with sample['image'] (usually evaluation/mvtec)")
    ap.add_argument("--mvtec-root", required=True,
                    help="MVTec AD root containing class/ground_truth/... (for GT scoring)")
    ap.add_argument("--ans", required=True, help="ans JSON (list) with conversations + gt")
    ap.add_argument("--subset", default="all", help="comma list or 'all'")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--defect-only", action="store_true",
                    help="only score gt=1 samples that have a GT mask")
    ap.add_argument("--use-stored", action="store_true",
                    help="score the CoT text stored in ans JSON instead of re-generating")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-implementation", default="flash_attention_2",
                    choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--force-triad-roi", action="store_true")
    ap.add_argument("--layers", default="midband",
                    help="midband|mid|last|last4|comma list (see attn_crop.resolve_attn_layers)")
    ap.add_argument("--grid", type=int, default=32, help="square grid for map vs GT comparison")
    ap.add_argument("--topk-frac", type=float, default=0.10)
    ap.add_argument("--max-words", type=int, default=0, help="cap words per sample (0=all)")
    ap.add_argument("--center-maps", action="store_true",
                    help="for t-SNE only: subtract per-sample mean map (removes object-location "
                         "common component, reduces class confound; metrics still use raw maps)")
    ap.add_argument("--pca-dim", type=int, default=16)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--tsne-max", type=int, default=4000, help="subsample words for the t-SNE plot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="LLM_judge_output/word_grounding")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- load data ----
    with open(args.ans, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("--ans must be a JSON list")
    if args.subset != "all":
        keep = set(args.subset.split(","))
        data = [d for d in data if d.get("subset") in keep]
    data = data[args.start:]
    if args.max_samples:
        data = data[: args.max_samples]
    print(f"[word-ground] {len(data)} samples from {args.ans}")

    # ---- model ----
    if args.model_name is None:
        args.model_name = os.path.basename(args.model_path.rstrip("/"))
    overwrite = None
    if args.force_triad_roi:
        overwrite = {
            "image_aspect_ratio": "anyres_max_9_randomroi",
            "mm_patch_merge_type": "spatial_avgpool_auto_unpad_randomroi_newl",
        }
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, args.model_name,
        device_map=None, device=args.device,
        attn_implementation=args.attn_implementation,
        overwrite_config=overwrite,
    )
    model = model.to(args.device)
    model.eval()
    n_layers = len(model.model.layers)
    layer_indices = resolve_attn_layers(args.layers, n_layers)
    print(f"[word-ground] layers={layer_indices} attn={args.attn_implementation} "
          f"iar={getattr(model.config, 'image_aspect_ratio', None)}")

    # ---- run ----
    per_sample: List[dict] = []
    errors: List[dict] = []
    all_rows: List[dict] = []
    tsne_rows: List[dict] = []   # same dict objects as all_rows, but only with a map (H)
    X_list: List[np.ndarray] = []

    for si, sample in enumerate(tqdm(data, desc="word-ground")):
        gt = sample.get("gt")
        origin = sample.get("origin_path") or ""
        if args.defect_only and int(gt or 0) != 1:
            continue
        mask_path = resolve_gt_mask_path(origin, args.mvtec_root) if int(gt or 0) == 1 else None
        try:
            rec = run_one_sample(model, tokenizer, image_processor, sample, args,
                                 layer_indices, mask_path)
        except Exception as e:
            errors.append({"id": sample.get("id"), "error": repr(e)})
            print(f"[word-ground][ERR] {sample.get('id')}: {e}")
            continue
        if rec is None:
            continue

        # attach sample-level outcome and flatten rows for stats/tsne
        pred = rec.get("answer")
        correct = None
        if pred is not None and gt is not None:
            correct = int(int(pred) == int(gt))
        rec["correct"] = correct
        Hs = rec.pop("_Hs", None)
        has_map = Hs is not None and len(Hs) == len(rec["words"])
        base_map = None
        if has_map and args.center_maps and len(Hs) > 1:
            base_map = np.mean(Hs, axis=0)
        for j, wr in enumerate(rec["words"]):
            row = {
                "id": rec["id"], "subset": rec["subset"], "gt": rec["gt"],
                "role": wr["role"], "word": wr["word"], "n_tokens": wr["n_tokens"],
                "_sample_group": ("correct" if correct == 1 else "wrong") if correct is not None else "unknown",
                "_sample_correct": correct,
            }
            for k in ("spearman", "dice", "point", "mass_gt",
                      "null_spearman", "null_dice", "null_point", "null_mass_gt"):
                row[k] = wr.get(k)
            all_rows.append(row)
            if has_map:
                tsne_rows.append(row)            # same object -> shares g after finalize
                vec = (Hs[j] - base_map) if base_map is not None else Hs[j]
                X_list.append(vec.reshape(-1))
        per_sample.append(rec)
        if (si + 1) % 10 == 0:
            print(f"[word-ground] {si + 1}/{len(data)} done, words so far={len(all_rows)}")

    # ---- normalise g ----
    finalize_g(all_rows)
    # propagate g back onto per-sample word dicts (all_rows was built in the
    # exact same per_sample/word order, so a sequential cursor is safe)
    cursor = 0
    for rec in per_sample:
        gs_all: List[float] = []
        gs_ev: List[float] = []
        for wr in rec["words"]:
            if cursor < len(all_rows):
                src = all_rows[cursor]
                wr["g"] = src.get("g")
                wr["g_null"] = src.get("g_null")
                wr["g_gain"] = src.get("g_gain")
            gval = wr.get("g")
            if gval is not None and np.isfinite(gval):
                gs_all.append(float(gval))
                if wr["role"] in ("type", "region", "object"):
                    gs_ev.append(float(gval))
            cursor += 1
        rec["best_word_g"] = max(gs_all) if gs_all else None
        rec["mean_evidence_g"] = float(np.mean(gs_ev)) if gs_ev else None

    # ---- t-SNE ----
    tsne_stats = {"enabled": False}
    if X_list and tsne_rows and args.tsne_max != 0:
        X = np.stack(X_list, axis=0).astype(np.float32)
        if X.shape[0] != len(tsne_rows):
            raise RuntimeError(f"t-SNE row misalignment: X={X.shape[0]} rows={len(tsne_rows)}")
        tsne_stats = run_tsne_and_stats(tsne_rows, X, args.out, args)

    # ---- write outputs ----
    summary = build_summary(all_rows, per_sample, errors)
    summary["tsne"] = tsne_stats
    report = {
        "meta": {
            "model_path": args.model_path,
            "ans": args.ans,
            "data_path": args.data_path,
            "mvtec_root": args.mvtec_root,
            "subset": args.subset,
            "use_stored": bool(args.use_stored),
            "layers": layer_indices,
            "grid": args.grid,
            "topk_frac": args.topk_frac,
            "force_triad_roi": bool(args.force_triad_roi),
            "image_aspect_ratio": getattr(model.config, "image_aspect_ratio", None),
            "note": (
                "GT used for scoring only; images fed with masks=None. Word attention = "
                "read-only Q/K averaged over midband, projected to anyres grid -> GxG."
            ),
        },
        "summary": summary,
        "errors": errors,
        "per_sample": per_sample,
    }
    json_path = os.path.join(args.out, "word_grounding.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # flat CSV
    csv_path = os.path.join(args.out, "word_grounding.csv")
    cols = ["id", "subset", "gt", "role", "word", "n_tokens",
            "spearman", "dice", "point", "mass_gt",
            "null_spearman", "null_dice", "null_point", "null_mass_gt",
            "g", "g_null", "g_gain"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in all_rows:
            wr.writerow({k: r.get(k) for k in cols})

    # npz for re-plotting
    if X_list and tsne_rows:
        X = np.stack(X_list, axis=0).astype(np.float32)
        np.savez_compressed(
            os.path.join(args.out, "word_maps.npz"),
            X=X,
            role=np.array([str(r.get("role", "")) for r in tsne_rows], dtype="<U16"),
            g=np.array([
                (r.get("g") if r.get("g") is not None else np.nan) for r in tsne_rows
            ], dtype=np.float32),
        )

    print(f"[word-ground] report -> {json_path}")
    print(f"[word-ground] table  -> {csv_path}")
    if summary.get("evidence_mean_g") is not None:
        print(
            f"[word-ground] evidence_g={summary['evidence_mean_g']} "
            f"control_g={summary['control_mean_g']} "
            f"diff={summary['evidence_minus_control']} p={summary['perm_p_evidence_gt_control']}"
        )
    if tsne_stats.get("enabled"):
        print(f"[word-ground] tsne -> {tsne_stats.get('tsne_path')} "
              f"role_f1={tsne_stats.get('role_macro_f1')} "
              f"outcome_auc={tsne_stats.get('sample_outcome_auc')}")


if __name__ == "__main__":
    main()
