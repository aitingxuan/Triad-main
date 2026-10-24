"""Attention-guided suspect-box extraction for scheme_requery ``attncrop`` (default_v4).

Reuses the Triad EG-RoI view layout: we only produce an ``xyxy`` box from the
decision-token attention map over the ``anyres`` view, then hand it back so the
existing ROI plumbing (``process_images(boxes_list=...)``) crops a high-fidelity
sub-image from the ORIGINAL image. No MuSc, no external detector.

Design (all relative, no absolute thresholds):
  * read-only Q/K attention of the decision token at a mid-layer band
    (FlashAttention generate stays untouched);
  * average over layers, de-sink, normalize the ``anyres`` spatial grid;
  * spatial-concentration gate: if the top-k mass is too small the map is
    diffuse -> no box (fallback to the existing full-image path);
  * largest-attention-mass connected component -> square box, expanded and
    area-clipped to ``[side_lo, side_hi]`` of the short image side.

This module is self-contained (does not import ``evaluate``/``simple_model_worker``)
to avoid circular imports.
"""

from __future__ import annotations

import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from attn_map_utils import (  # noqa: E402
    build_views,
    llava_anyres_spatial_hw,
    view_raw_spatial_grid,
    _jet_rgb,
)
from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402


# ---------------------------------------------------------------------------
# Layer spec + query helpers (mirror p0_attention_probe, kept local)
# ---------------------------------------------------------------------------

def resolve_attn_layers(spec: str, n_layers: int) -> List[int]:
    spec = (spec or "midband").strip().lower()
    if spec in ("midband", "mid_band", "default"):
        wanted = [int(n_layers * f) for f in (0.50, 0.57, 0.64, 0.71)]
        return sorted({i for i in wanted if 0 <= i < n_layers})
    if spec in ("mid", "middle"):
        wanted = [3, 10, 14, 18, n_layers - 1]
        return sorted({i for i in wanted if 0 <= i < n_layers})
    if spec in ("last",):
        return [n_layers - 1]
    if spec == "last4":
        return list(range(max(0, n_layers - 4), n_layers))
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
    if not out:
        raise ValueError(f"empty layer spec {spec!r}")
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Read-only Q/K attention (identical math to p0_attention_probe)
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
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    pos_emb = _position_embeddings(model, hidden_states[0], position_ids)
    out: Dict[int, np.ndarray] = {}
    for li in layer_indices:
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
# Multi-query read-only Q/K attention (ReAL: score MANY reasoning tokens at once)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _independent_qk_attention_multi(
    decoder_layer,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask: Optional[torch.Tensor],
    query_indices: Sequence[int],
) -> torch.Tensor:
    """Full attention rows for a batch of query positions, one layer.

    Returns ``[n_query, L]`` softmax rows (head-averaged). Keys are projected
    ONCE and shared across all queries (cheap: one forward, N query rows).
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
    q_idx = torch.tensor([int(q) for q in query_indices], device=device, dtype=torch.long)
    if (q_idx < 0).any() or (q_idx >= length).any() or (~valid[0, q_idx]).any():
        raise ValueError(f"invalid query_indices for length={length}")

    attention = decoder_layer.self_attn
    normalized = decoder_layer.input_layernorm(hidden_states)
    query = attention.q_proj(normalized[0, q_idx]).unsqueeze(0)  # [1, Nq, H]
    key = attention.k_proj(normalized)                           # [1, L, H]
    query = query.view(1, q_idx.numel(), attention.num_heads, attention.head_dim).transpose(1, 2)
    key = key.view(1, length, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)

    cos, sin = (item.to(device=device).expand(batch, -1, -1) for item in position_embeddings)
    if cos.shape[-1] * 2 == query.shape[-1]:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
    query_cos = cos[0, q_idx][None, None, :, :]   # [1,1,Nq,D]
    query_sin = sin[0, q_idx][None, None, :, :]
    query = query * query_cos + _rotate_half(query) * query_sin
    key = key * cos[:, None] + _rotate_half(key) * sin[:, None]
    groups = getattr(attention, "num_key_value_groups", None)
    if groups is None:
        groups = attention.num_heads // max(attention.num_key_value_heads, 1)
    key = key.repeat_interleave(groups, dim=1)

    logits = torch.matmul(query.float(), key.float().transpose(-2, -1))  # [1, H, Nq, L]
    logits = logits.squeeze(0) / math.sqrt(attention.head_dim)           # [H, Nq, L]
    # causal: query at position q_idx[i] can only see keys <= q_idx[i]
    visible = positions[0][None, :] <= q_idx[:, None]                     # [Nq, L]
    visible = visible[None, :, :] & valid[0][None, None, :]              # [1(H), Nq, L]
    finite = logits.masked_select(visible)
    if finite.numel() and not torch.isfinite(finite).all():
        raise FloatingPointError("Non-finite independent Q/K scores")
    logits = logits.masked_fill(~visible, float("-inf"))
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32).mean(dim=0)  # [Nq, L]
    return probs


@torch.no_grad()
def score_layers_multi(
    model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    layer_indices: Sequence[int],
    query_indices: Sequence[int],
) -> Dict[int, np.ndarray]:
    """One forward, then per layer the full attention rows of many queries.

    Returns ``{layer: ndarray[n_query, L]}``.
    """
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    pos_emb = _position_embeddings(model, hidden_states[0], position_ids)
    out: Dict[int, np.ndarray] = {}
    for li in layer_indices:
        rows = _independent_qk_attention_multi(
            model.model.layers[li],
            hidden_states[li],
            pos_emb,
            attention_mask=attention_mask,
            query_indices=query_indices,
        )
        out[int(li)] = rows.detach().float().cpu().numpy()
    del outputs
    return out


# ---------------------------------------------------------------------------
# Segment recorder (BASE / ANYRES / ROI lengths from Triad merge)
# ---------------------------------------------------------------------------

class SegmentRecorder:
    """Wrap ``_concat_global_roi`` to record token lengths / ROI shapes / boxes."""

    def __init__(self):
        self.last = None
        self._orig = None
        self._model = None

    def install(self, model):
        if not hasattr(model, "_concat_global_roi"):
            raise RuntimeError("model has no _concat_global_roi (need anyres_max_*_randomroi)")
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


# ---------------------------------------------------------------------------
# Grid -> box
# ---------------------------------------------------------------------------

def _label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """8-neighbor connected components (small grids; plain flood fill)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 0
    for i in range(h):
        for j in range(w):
            if mask[i, j] and labels[i, j] == 0:
                cur += 1
                stack = [(i, j)]
                labels[i, j] = cur
                while stack:
                    a, b = stack.pop()
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            na, nb = a + da, b + db
                            if 0 <= na < h and 0 <= nb < w and mask[na, nb] and labels[na, nb] == 0:
                                labels[na, nb] = cur
                                stack.append((na, nb))
    return labels, cur


def _topk_mass(grid: np.ndarray, frac: float) -> float:
    flat = np.asarray(grid, dtype=np.float64).reshape(-1)
    total = float(flat.sum())
    if total <= 0:
        return 0.0
    p = flat / total
    k = max(1, int(math.ceil(float(frac) * p.size)))
    return float(np.sort(p)[-k:].sum())


def grid_to_box(
    grid: np.ndarray,
    view_box_xyxy: Sequence[float],
    image_wh: Tuple[int, int],
    *,
    thr_q: float = 0.90,
    expand: float = 1.3,
    side_lo: float = 0.06,
    side_hi: float = 0.45,
    conc_floor: float = 0.30,
) -> Dict[str, object]:
    """Largest-attention-mass component -> square box in image coordinates."""
    g = np.maximum(np.asarray(grid, dtype=np.float32), 0.0)
    h, w = g.shape
    out: Dict[str, object] = {"kept": False, "box": None, "reason": None}
    if h <= 1 or w <= 1 or float(g.max()) <= 0:
        out["reason"] = "empty"
        return out
    top5 = _topk_mass(g, 0.05)
    out["top5_mass"] = round(float(top5), 6)
    if top5 < float(conc_floor):
        out["reason"] = "diffuse"
        return out

    norm = g / float(g.max())
    blob = norm >= float(thr_q)
    if not blob.any():
        out["reason"] = "no_blob"
        return out
    labels, n = _label_components(blob)
    if n <= 0:
        out["reason"] = "no_blob"
        return out
    # pick component with the largest summed attention mass
    best_c, best_mass = -1, -1.0
    for c in range(1, n + 1):
        m = float(g[labels == c].sum())
        if m > best_mass:
            best_mass, best_c = m, c
    rs, cs = np.where(labels == best_c)
    if rs.size == 0:
        out["reason"] = "no_blob"
        return out

    img_w, img_h = float(image_wh[0]), float(image_wh[1])
    vx0, vy0, vx1, vy1 = [float(v) for v in view_box_xyxy[:4]]
    vw = max(vx1 - vx0, 1e-6)
    vh = max(vy1 - vy0, 1e-6)

    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())
    cx = vx0 + ((c0 + c1) * 0.5 + 0.5) / w * vw
    cy = vy0 + ((r0 + r1) * 0.5 + 0.5) / h * vh
    span_x = (c1 - c0 + 1) / w * vw
    span_y = (r1 - r0 + 1) / h * vh
    side = max(span_x, span_y) * float(expand)

    short = min(img_w, img_h)
    side = max(float(side_lo) * short, min(float(side_hi) * short, side))
    half = side * 0.5
    x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
    # keep inside the frame
    if x0 < 0:
        x1 -= x0
        x0 = 0.0
    if y0 < 0:
        y1 -= y0
        y0 = 0.0
    if x1 > img_w:
        x0 -= (x1 - img_w)
        x1 = img_w
    if y1 > img_h:
        y0 -= (y1 - img_h)
        y1 = img_h
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(img_w, x1), min(img_h, y1)
    area_frac = ((x1 - x0) * (y1 - y0)) / max(img_w * img_h, 1.0)
    out.update({
        "kept": True,
        "box": [x0, y0, x1, y1],
        "area_frac": round(float(area_frac), 6),
        "grid_wh": [h, w],
    })
    return out


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def extract_attn_box_prepared(
    model,
    prepared_images,
    image: Image.Image,
    image_sizes: Sequence[Tuple[int, int]],
    roi_bboxes,
    input_ids: torch.Tensor,
    gen_token_id: int,
    *,
    layers: str = "midband",
    query: str = "decision",
    thr_q: float = 0.90,
    expand: float = 1.3,
    side_lo: float = 0.06,
    side_hi: float = 0.45,
    conc_floor: float = 0.30,
    overlay_path: Optional[str] = None,
    overlay_title: Optional[str] = None,
    display_size: int = 640,
) -> Dict[str, object]:
    """Compute the suspect box from the decision-token attention.

    ``prepared_images`` are the tensors actually fed to the main forward
    (``image_args["images"]``); ``image`` is the ORIGINAL PIL image.
    """
    device = model.device
    n_layers = len(model.model.layers)
    layer_indices = resolve_attn_layers(layers, n_layers)

    recorder = SegmentRecorder().install(model)
    try:
        prepared = prepared_images
        if isinstance(prepared, list):
            prepared = [t.to(device, dtype=model.dtype) for t in prepared]
        else:
            prepared = prepared.to(device, dtype=model.dtype)

        with torch.no_grad():
            (_, position_ids, attention_mask, _, prompt_embeds, _) = \
                model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, None, None, None,
                    prepared, modalities=["image"],
                    image_sizes=list(image_sizes), roi_bboxes=roi_bboxes,
                )
        seg = recorder.last
        if seg is None:
            return {"kept": False, "box": None, "reason": "no_seg"}

        n_packed = int(seg["n_global"]) + int(seg["n_roi"])
        prompt_len = int(prompt_embeds.shape[1])
        n_text = int(input_ids.shape[1]) - 1
        n_visual = prompt_len - n_text
        if n_visual < n_packed:
            n_visual = n_packed
        trailing = n_visual - n_packed
        vis_start, _ = find_visual_span(input_ids, n_visual)
        vis_end = vis_start + n_visual

        # score sequence: prompt + the single decision token
        gen_emb = model.get_model().embed_tokens(
            torch.tensor([[int(gen_token_id)]], device=device, dtype=torch.long)
        ).to(prompt_embeds.dtype)
        score_embeds = torch.cat([prompt_embeds, gen_emb], dim=1)
        score_mask = torch.ones(score_embeds.shape[:2], dtype=torch.long, device=device)
        if position_ids is not None:
            last = position_ids[0, -1]
            gen_pos = (last + torch.arange(1, 2, device=device)).unsqueeze(0)
            score_pos = torch.cat([position_ids, gen_pos], dim=1)
        else:
            score_pos = torch.arange(score_embeds.shape[1], device=device).unsqueeze(0)
        query_index = prompt_len

        layer_scores = score_layers(
            model, score_embeds, score_mask, score_pos, layer_indices, query_index
        )

        # geometry / views
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
        if seg["n_anyres"] > 0:
            computed = llava_anyres_spatial_hw(
                image.size, pinpoints, vis_size, pps,
                max_num_patches=max_patches, n_anyres=seg["n_anyres"],
            )
            if computed is not None:
                any_hw = (computed[0], computed[1])

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

        # average visual scores across layers, keep a slice covering packed views
        vis_sum = None
        n_used = 0
        for _li, scores in layer_scores.items():
            visual = scores[vis_start:vis_end]
            if visual.size < n_packed:
                continue
            sl = visual[: n_packed + trailing].astype(np.float32)
            vis_sum = sl if vis_sum is None else vis_sum + sl
            n_used += 1
        if vis_sum is None or n_used == 0:
            return {"kept": False, "box": None, "reason": "no_visual"}
        vis_mean = vis_sum / float(n_used)

        anyres_view = next((v for v in views if v["name"] == "anyres"), None)
        if anyres_view is None:
            return {"kept": False, "box": None, "reason": "no_anyres"}
        grid = view_raw_spatial_grid(anyres_view, vis_mean)
        # de-sink: drop the single strongest cell if it dominates (attention sink)
        gm = grid.astype(np.float64)
        if gm.size and float(gm.max()) > 0:
            gm = gm.copy()
            gm[np.unravel_index(int(np.argmax(gm)), gm.shape)] *= 0.25
        grid = gm.astype(np.float32)

        res = grid_to_box(
            grid,
            anyres_view["source_box_xyxy"],
            image.size,
            thr_q=thr_q, expand=expand, side_lo=side_lo, side_hi=side_hi, conc_floor=conc_floor,
        )
        res["layers"] = layer_indices
        res["config"] = {
            "thr_q": thr_q, "expand": expand, "side_lo": side_lo,
            "side_hi": side_hi, "conc_floor": conc_floor,
        }
        if isinstance(res.get("box"), (list, tuple)):
            res["box"] = [round(float(v), 2) for v in res["box"]]

        if overlay_path:
            try:
                save_attn_box_overlay(
                    overlay_path, image, anyres_view, vis_mean,
                    res.get("box"), title=overlay_title, display_size=display_size,
                )
                res["overlay"] = overlay_path
            except Exception as e:  # visualization must never break eval
                res["overlay_error"] = repr(e)
        return res
    finally:
        recorder.uninstall()


# ---------------------------------------------------------------------------
# ReAL proposer: harvest reasoning-token attention, filter by S_T + S_I
# (Reasoning-Driven Anomaly Detection, CVPR'26) -> aggregate RDAM -> crop box
# ---------------------------------------------------------------------------

def _spatial_entropy(grid: np.ndarray) -> float:
    """Normalized spatial entropy in [0,1]; low = concentrated."""
    g = np.maximum(np.asarray(grid, dtype=np.float64).reshape(-1), 0.0)
    total = float(g.sum())
    if g.size <= 1 or total <= 0:
        return 1.0
    p = g / total
    ent = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    return float(ent / math.log(g.size))


def _concentration_score(grid: np.ndarray, topk_frac: float = 0.10) -> float:
    """ReAL ``S_I``: spatial concentration in [0,1] (1 - entropy + top-k mass blend)."""
    g = np.maximum(np.asarray(grid, dtype=np.float64), 0.0)
    flat = g.reshape(-1)
    total = float(flat.sum())
    if flat.size <= 1 or total <= 0:
        return 0.0
    ent = _spatial_entropy(g)
    k = max(1, int(math.ceil(topk_frac * flat.size)))
    topk = float(np.sort(flat / total)[-k:].sum())
    # both terms in [0,1]; combine so a single hot blob scores high
    return float(0.5 * (1.0 - ent) + 0.5 * topk)


def _semantic_relevance(row: np.ndarray, anchor_idx: Sequence[int]) -> float:
    """ReAL ``S_T``: attention mass a query puts on anomaly-anchor text tokens."""
    if not anchor_idx:
        return 0.0
    idx = [int(i) for i in anchor_idx if 0 <= int(i) < row.shape[-1]]
    if not idx:
        return 0.0
    return float(np.asarray(row)[idx].sum())


def _find_anchor_positions(tokenizer, input_ids_list: Sequence[int], anchor_words: Sequence[str],
                           image_token_index: int = -200) -> List[int]:
    """Best-effort locate anomaly-anchor tokens within the packed ``input_ids``.

    We map each of the original text tokens (id != IMAGE_TOKEN) to a string and
    keep the positions whose decoded piece contains one of ``anchor_words``
    (case-insensitive). Returns positions in ``input_ids`` coordinates; the
    caller must shift by the number of inserted visual tokens to get the
    prompt-embedding coordinates.
    """
    anchors = [w.lower() for w in anchor_words if w]
    hits: List[int] = []
    if not anchors:
        return hits
    for pos, tid in enumerate(input_ids_list):
        tid = int(tid)
        if tid == int(image_token_index):
            continue
        try:
            piece = tokenizer.decode([tid]).lower()
        except Exception:
            continue
        if any(w in piece for w in anchors):
            hits.append(pos)
    return hits


def _shift_positions_to_prompt(anchor_input_positions: Sequence[int],
                               input_ids_list: Sequence[int],
                               n_visual: int,
                               image_token_index: int = -200) -> List[int]:
    """Convert input_ids positions -> prompt-embedding positions (visual tokens inserted)."""
    # Visual tokens replace the single IMAGE token; net insertion = n_visual - 1.
    ids = [int(t) for t in input_ids_list]
    try:
        img_pos = ids.index(int(image_token_index))
    except ValueError:
        return list(anchor_input_positions)
    shift = n_visual - 1
    out = []
    for p in anchor_input_positions:
        if p <= img_pos:
            out.append(p)
        else:
            out.append(p + shift)
    return out


def propose_box_real(
    model,
    tokenizer,
    prepared_images,
    image: Image.Image,
    image_sizes: Sequence[Tuple[int, int]],
    roi_bboxes,
    input_ids: torch.Tensor,
    reasoning_token_ids: Sequence[int],
    *,
    layers: str = "midband",
    anchor_words: Optional[Sequence[str]] = None,
    tau_t: float = 0.0,
    tau_i: float = 0.0,
    keep_frac: float = 0.5,
    alpha: float = 0.6,
    beta: float = 0.4,
    view_name: str = "anyres",
    thr_q: float = 0.70,
    expand: float = 1.3,
    side_lo: float = 0.06,
    side_hi: float = 0.45,
    conc_floor: float = 0.20,
    overlay_path: Optional[str] = None,
    overlay_title: Optional[str] = None,
    display_size: int = 640,
) -> Dict[str, object]:
    """ReAL-style suspect box from harvested reasoning-token attention.

    Steps (mirrors the CVPR'26 ReAL module, adapted to frozen Triad):
      * take the model-generated reasoning token ids as the query pool;
      * for every reasoning token compute
          - ``S_T`` = attention mass on anomaly-anchor text tokens (semantic);
          - ``S_I`` = spatial concentration of its visual attention (1-entropy);
      * keep the few tokens passing ``S_T``/``S_I`` relative thresholds,
        weighted by ``w = alpha*S_T_hat + beta*S_I_hat``;
      * weighted-aggregate their visual attention maps -> RDAM spatial grid ->
        largest-mass blob -> square crop box (same geometry as ``attn`` proposer).
    """
    device = model.device
    if not reasoning_token_ids:
        return {"kept": False, "box": None, "reason": "no_reasoning_tokens", "proposer": "real"}
    n_layers = len(model.model.layers)
    layer_indices = resolve_attn_layers(layers, n_layers)
    anchor_words = list(anchor_words) if anchor_words else [
        "defect", "fault", "anomal", "abnormal", "peculiar", "damage", "flaw",
    ]

    recorder = SegmentRecorder().install(model)
    try:
        prepared = prepared_images
        if isinstance(prepared, list):
            prepared = [t.to(device, dtype=model.dtype) for t in prepared]
        else:
            prepared = prepared.to(device, dtype=model.dtype)

        with torch.no_grad():
            (_, position_ids, attention_mask, _, prompt_embeds, _) = \
                model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, None, None, None,
                    prepared, modalities=["image"],
                    image_sizes=list(image_sizes), roi_bboxes=roi_bboxes,
                )
        seg = recorder.last
        if seg is None:
            return {"kept": False, "box": None, "reason": "no_seg", "proposer": "real"}

        n_packed = int(seg["n_global"]) + int(seg["n_roi"])
        prompt_len = int(prompt_embeds.shape[1])
        n_text = int(input_ids.shape[1]) - 1
        n_visual = prompt_len - n_text
        if n_visual < n_packed:
            n_visual = n_packed
        trailing = n_visual - n_packed
        vis_start, _ = find_visual_span(input_ids, n_visual)
        vis_end = vis_start + n_visual

        # scoring sequence = prompt + reasoning tokens
        r_ids = [int(t) for t in reasoning_token_ids]
        gen_emb = model.get_model().embed_tokens(
            torch.tensor([r_ids], device=device, dtype=torch.long)
        ).to(prompt_embeds.dtype)
        score_embeds = torch.cat([prompt_embeds, gen_emb], dim=1)
        score_mask = torch.ones(score_embeds.shape[:2], dtype=torch.long, device=device)
        if position_ids is not None:
            last = position_ids[0, -1]
            gen_pos = (last + torch.arange(1, len(r_ids) + 1, device=device)).unsqueeze(0)
            score_pos = torch.cat([position_ids, gen_pos], dim=1)
        else:
            score_pos = torch.arange(score_embeds.shape[1], device=device).unsqueeze(0)

        query_indices = list(range(prompt_len, prompt_len + len(r_ids)))
        layer_rows = score_layers_multi(
            model, score_embeds, score_mask, score_pos, layer_indices, query_indices
        )

        # geometry / views
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
        if seg["n_anyres"] > 0:
            computed = llava_anyres_spatial_hw(
                image.size, pinpoints, vis_size, pps,
                max_num_patches=max_patches, n_anyres=seg["n_anyres"],
            )
            if computed is not None:
                any_hw = (computed[0], computed[1])
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
        target_view = next((v for v in views if v["name"] == view_name), None)
        if target_view is None:
            target_view = next((v for v in views if v["name"] == "base"), None)
        if target_view is None:
            return {"kept": False, "box": None, "reason": "no_view", "proposer": "real"}

        # anchor positions in prompt-embedding coordinates
        ids_list = input_ids[0].detach().cpu().tolist()
        anchor_in = _find_anchor_positions(tokenizer, ids_list, anchor_words, IMAGE_TOKEN_INDEX)
        anchor_pos = _shift_positions_to_prompt(anchor_in, ids_list, n_visual, IMAGE_TOKEN_INDEX)

        # per reasoning-token S_T / S_I (averaged over the layer band)
        n_q = len(r_ids)
        S_T = np.zeros(n_q, dtype=np.float64)
        S_I = np.zeros(n_q, dtype=np.float64)
        n_used = 0
        for _li, rows in layer_rows.items():
            if rows.ndim != 2 or rows.shape[0] != n_q:
                continue
            for qi in range(n_q):
                row = rows[qi]
                if row.shape[-1] < vis_end:
                    continue
                visual = np.maximum(row[vis_start:vis_end], 0.0)[: n_packed + trailing].astype(np.float32)
                S_T[qi] += _semantic_relevance(row, anchor_pos)
                g = view_raw_spatial_grid(target_view, visual).astype(np.float32)
                S_I[qi] += _concentration_score(g)
            n_used += 1
        if n_used == 0:
            return {"kept": False, "box": None, "reason": "no_visual", "proposer": "real"}
        S_T /= float(n_used)
        S_I /= float(n_used)

        # relative normalization + dual-threshold selection (no absolute cut)
        def _norm01(v):
            v = np.asarray(v, dtype=np.float64)
            lo, hi = float(v.min()), float(v.max())
            if hi - lo < 1e-9:
                return np.zeros_like(v)
            return (v - lo) / (hi - lo)

        S_T_hat = _norm01(S_T)
        S_I_hat = _norm01(S_I)
        keep = np.ones(n_q, dtype=bool)
        if keep_frac is not None and 0.0 < keep_frac < 1.0:
            thresh_i = float(np.quantile(S_I_hat, 1.0 - keep_frac))
            keep &= (S_I_hat >= thresh_i)
        if tau_t > 0:
            keep &= (S_T_hat >= tau_t)
        if tau_i > 0:
            keep &= (S_I_hat >= tau_i)
        if not keep.any():
            # fall back to the single most-concentrated token
            keep[int(np.argmax(S_I_hat))] = True
        w = alpha * S_T_hat + beta * S_I_hat
        w = np.where(keep, np.maximum(w, 1e-6), 0.0)
        if float(w.sum()) <= 0:
            w = keep.astype(np.float64)

        # aggregate RDAM: weighted sum (across selected tokens) of their visual maps,
        # each computed by re-scoring the layer band once more (cheap: reuse rows).
        rdam = None
        for _li, rows in layer_rows.items():
            if rows.ndim != 2 or rows.shape[0] != n_q:
                continue
            vis_sum = None
            for qi in range(n_q):
                if not keep[qi]:
                    continue
                row = rows[qi]
                if row.shape[-1] < vis_end:
                    continue
                visual = np.maximum(row[vis_start:vis_end], 0.0)[: n_packed + trailing].astype(np.float32)
                weighted = visual * float(w[qi])
                vis_sum = weighted if vis_sum is None else vis_sum + weighted
            if vis_sum is None:
                continue
            g = view_raw_spatial_grid(target_view, vis_sum / float(n_used)).astype(np.float32)
            rdam = g if rdam is None else rdam + g
        if rdam is None:
            return {"kept": False, "box": None, "reason": "no_rdam", "proposer": "real"}

        # de-sink the single hottest cell (attention sink on token id 0 etc.)
        gm = rdam.astype(np.float64).copy()
        if gm.size and float(gm.max()) > 0:
            gm[np.unravel_index(int(np.argmax(gm)), gm.shape)] *= 0.25
        rdam = gm.astype(np.float32)

        res = grid_to_box(
            rdam,
            target_view["source_box_xyxy"],
            image.size,
            thr_q=thr_q, expand=expand, side_lo=side_lo, side_hi=side_hi, conc_floor=conc_floor,
        )
        res["proposer"] = "real"
        res["layers"] = layer_indices
        res["real"] = {
            "view": target_view["name"],
            "n_reasoning_tokens": int(n_q),
            "n_kept": int(keep.sum()),
            "n_anchor_tokens": len(anchor_pos),
            "anchor_words": anchor_words,
            "S_T_mean": round(float(S_T.mean()), 6) if n_q else 0.0,
            "S_I_mean": round(float(S_I.mean()), 6) if n_q else 0.0,
            "S_I_max": round(float(S_I.max()), 6) if n_q else 0.0,
            "keep_frac": keep_frac,
            "alpha": alpha, "beta": beta,
            "top5_mass": res.get("top5_mass"),
        }
        res["config"] = {
            "proposer": "real", "thr_q": thr_q, "expand": expand,
            "side_lo": side_lo, "side_hi": side_hi, "conc_floor": conc_floor,
        }
        if isinstance(res.get("box"), (list, tuple)):
            res["box"] = [round(float(v), 2) for v in res["box"]]

        if overlay_path:
            try:
                save_attn_box_overlay(
                    overlay_path, image, target_view, rdam,
                    res.get("box"), title=overlay_title, display_size=display_size,
                )
                res["overlay"] = overlay_path
            except Exception as e:  # visualization must never break eval
                res["overlay_error"] = repr(e)
        return res
    finally:
        recorder.uninstall()


# ---------------------------------------------------------------------------
# Visualization: attention heat overlay + crop box on the original image
# ---------------------------------------------------------------------------

def _normalize01(grid: np.ndarray) -> np.ndarray:
    g = np.maximum(np.asarray(grid, dtype=np.float32), 0.0)
    mx = float(g.max()) if g.size else 0.0
    if mx <= 0:
        return np.zeros_like(g, dtype=np.float32)
    return g / mx


def save_attn_box_overlay(
    path: str,
    image: Image.Image,
    view: dict,
    visual_scores: np.ndarray,
    box: Optional[Sequence[float]],
    *,
    title: Optional[str] = None,
    display_size: int = 640,
) -> str:
    """3-panel figure: original(+box) | attention overlay(+box) | crop.

    The attention grid of ``view`` (usually ``anyres``) is upsampled to the full
    image and blended with the original; the crop box is drawn in red.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    grid = _normalize01(view_raw_spatial_grid(view, visual_scores))  # (H, W)
    img = image.convert("RGB")
    w, h = img.size
    heat = np.asarray(
        Image.fromarray(grid.astype(np.float32), mode="F").resize((w, h), Image.Resampling.BILINEAR)
    )
    heat = np.clip(heat, 0.0, 1.0)
    overlay = np.clip(0.55 * np.asarray(img) + 0.45 * _jet_rgb(heat), 0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)

    crop_img = None
    if box is not None and len(box) >= 4:
        x0, y0, x1, y1 = [int(round(v)) for v in box[:4]]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, max(x0 + 1, x1)), min(h, max(y0 + 1, y1))
        crop_img = img.crop((x0, y0, x1, y1))

    def _with_box(im: Image.Image) -> Image.Image:
        im = im.copy()
        if box is not None and len(box) >= 4:
            d = ImageDraw.Draw(im)
            d.rectangle([float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        outline=(255, 0, 0), width=max(2, int(min(w, h) * 0.006)))
        return im

    orig_panel = _with_box(img)
    ov_panel = _with_box(overlay_img)
    panels = [("original", orig_panel), ("attn", ov_panel)]
    if crop_img is not None:
        panels.append(("crop", crop_img))

    def _fit(im: Image.Image) -> Image.Image:
        iw, ih = im.size
        s = display_size / max(iw, ih, 1)
        nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
        r = im.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (display_size, display_size), "white")
        canvas.paste(r, ((display_size - nw) // 2, (display_size - nh) // 2))
        return canvas

    margin, label_h, title_h = 10, 26, (28 if title else 0)
    canvas = Image.new("RGB", (len(panels) * display_size + (len(panels) - 1) * margin,
                               title_h + label_h + display_size), "white")
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((6, 6), str(title)[:160], fill="black")
    for i, (name, im) in enumerate(panels):
        x = i * (display_size + margin)
        draw.text((x + 4, title_h + 4), name, fill="black")
        canvas.paste(_fit(im), (x, title_h + label_h))
    canvas.save(path)
    return path
