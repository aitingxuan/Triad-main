"""ECPR Week-2 auxiliary losses on the LM decision / causal-chain tokens.

Implements the *trainable* part of ``docs/EG_CONDITIONAL_PART_ROI.md`` §6 that
can be attached without a separate ROI scoring tower:

* ``L_localize`` — upweight ``<location>`` / ``<type>`` (and GT interest aliases)
* ``L_attr`` — upweight Step / step_name tokens; heavier only when location
  text in the label is compatible with GT (localization-correct gate)
* ``L_agg`` — upweight the final Yes/No decision token on ECPR-covered samples
* ``L_local`` — BCE between a tiny learnable slot aggregator and Yes/No, using
  offline ``slot_positive`` soft targets as ``p_r`` (evidence scores). This
  trains the soft-agg formula on GT local overlap; gradients do **not** flow
  into the VLM (ZeRO-3 safe). True vision ``p_r`` heads remain future work.

Total used in the trainer:

``L = L_CE_chain + λ_local * L_local_agg``

where ``L_CE_chain`` is token-reweighted CE (localize / attr / agg / base).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from llava.constants import IGNORE_INDEX


def _id_mask(flat_labels, ids):
    m = flat_labels.new_zeros(flat_labels.shape, dtype=torch.bool)
    for _id in ids:
        m |= flat_labels == int(_id)
    return m


def build_decision_token_id_sets(tokenizer):
    no_ids, yes_ids = set(), set()
    for txt in ["Yes", " yes", "YES", "No", " no", "NO"]:
        try:
            tids = tokenizer.encode(txt, add_special_tokens=False)
        except Exception:
            continue
        for t in tids:
            if "yes" in txt.lower():
                yes_ids.add(int(t))
            else:
                no_ids.add(int(t))
    return no_ids, yes_ids


def _last_decision_mask(shift_labels, decision_ids):
    B, T = shift_labels.shape
    mask = torch.zeros((B, T), dtype=torch.bool, device=shift_labels.device)
    d = set(int(x) for x in decision_ids)
    for b in range(B):
        for t in range(T - 1, -1, -1):
            v = int(shift_labels[b, t].item())
            if v == IGNORE_INDEX:
                continue
            if v in d:
                mask[b, t] = True
            break
    return mask


_TAG_LOC = re.compile(r"<location>\s*(.*?)\s*</location>", re.IGNORECASE | re.DOTALL)
_TAG_TYPE = re.compile(r"<type>\s*(.*?)\s*</type>", re.IGNORECASE | re.DOTALL)
_TAG_ANS = re.compile(r"<answer>\s*(Yes|No)\s*</answer>", re.IGNORECASE)


def _as_meta_list(metas: Optional[Sequence[Any]], batch: int) -> List[Optional[Dict[str, Any]]]:
    if metas is None:
        return [None] * batch
    out: List[Optional[Dict[str, Any]]] = []
    for i in range(batch):
        m = metas[i] if i < len(metas) else None
        out.append(m if isinstance(m, dict) else None)
    return out


def _decode_label_row(labels_1d: torch.Tensor, tokenizer) -> str:
    ids = [int(t) for t in labels_1d.tolist() if int(t) != IGNORE_INDEX and int(t) >= 0]
    if not ids:
        return ""
    try:
        return tokenizer.decode(ids, skip_special_tokens=True)
    except Exception:
        return ""


def _encode_needle_id_seqs(tokenizer, text: str) -> List[List[int]]:
    """Token-id sequences for a needle (a few spacing/case variants)."""
    out: List[List[int]] = []
    seen = set()
    raw = str(text)
    if not raw:
        return out
    variants = (raw, f" {raw}", raw.lower(), f" {raw.lower()}", raw.replace("_", " "), f" {raw.replace('_', ' ')}")
    for variant in variants:
        try:
            ids = tokenizer.encode(variant, add_special_tokens=False)
        except Exception:
            continue
        if not ids:
            continue
        key = tuple(int(x) for x in ids)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(key))
    return out


def _token_positions_matching_substrings(
    shift_labels_row: torch.Tensor,
    tokenizer,
    needles: Sequence[str],
) -> torch.Tensor:
    """Boolean mask over positions covered by any needle as a contiguous token span.

    IMPORTANT: do **not** call ``tokenizer.decode`` per token — that is O(T) Python
    decode calls and can stall the first training step for tens of minutes on long
    multimodal sequences. Match on token-id subsequences instead.
    """
    T = shift_labels_row.numel()
    mask = torch.zeros(T, dtype=torch.bool, device=shift_labels_row.device)
    if not needles:
        return mask
    ids = [int(x) for x in shift_labels_row.tolist()]
    for needle in needles:
        if not needle:
            continue
        for nid_seq in _encode_needle_id_seqs(tokenizer, str(needle)):
            L = len(nid_seq)
            if L == 0 or L > T:
                continue
            # Skip extremely broad single-token needles (e.g. bare digit "6") unless
            # they appear as part of a multi-token needle already covered above.
            if L == 1 and len(str(needle).strip()) <= 1:
                continue
            for t in range(T - L + 1):
                if ids[t] == IGNORE_INDEX:
                    continue
                if ids[t : t + L] == nid_seq:
                    for k in range(L):
                        if ids[t + k] != IGNORE_INDEX:
                            mask[t + k] = True
    return mask


def _localization_correct(decoded: str, meta: Dict[str, Any]) -> bool:
    """Gate for attr loss: label location/type compatible with GT interest."""
    loc = ""
    m = _TAG_LOC.search(decoded)
    if m:
        loc = m.group(1).strip().lower()
    interests = [str(x).lower() for x in (meta.get("gt_interests") or [])]
    loc_tag = str(meta.get("gt_location_tag") or "").lower()
    aliases = set(interests)
    if loc_tag:
        aliases.add(loc_tag)
    # CoTM location tags are often top/center/bottom — map from interest.
    for name in interests:
        if "top" in name:
            aliases.update(["top", "top_label", "top label"])
        if "bottom" in name:
            aliases.update(["bottom", "bottom_label", "bottom label"])
        if name == "juice":
            aliases.update(["juice", "center", "liquid"])
    if not aliases:
        return True  # no GT → do not block attr
    if loc and any(a in loc or loc in a for a in aliases if a):
        return True
    # fallback: type text present
    type_txt = str(meta.get("gt_type_text") or "").lower()
    if type_txt and type_txt in decoded.lower():
        return True
    # process_align without tags still may mention interest
    if any(a and a in decoded.lower() for a in interests):
        return True
    # if sample has no location supervision tags at all, allow attr at base weight
    if "<location>" not in decoded.lower():
        return True
    return False


def _attr_needles(meta: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    step = meta.get("gt_step")
    if step is not None:
        # Prefer multi-token / distinctive forms; avoid bare digit alone (too broad).
        out.extend([f"Step {step}", f"step {step}", f"Step{step}"])
    name = meta.get("gt_step_name")
    if name:
        out.append(str(name))
        for w in str(name).replace("-", " ").split():
            if len(w) >= 4:
                out.append(w)
    return out


def _localize_needles(meta: Dict[str, Any]) -> List[str]:
    out: List[str] = ["<location>", "</location>", "<type>", "</type>"]
    for name in meta.get("gt_interests") or []:
        out.append(str(name))
        out.append(str(name).replace("_", " "))
    loc = meta.get("gt_location_tag")
    if loc:
        out.append(str(loc))
    t = meta.get("gt_type_text")
    if t:
        out.append(str(t))
    return out


class EcprLocalAggHead(nn.Module):
    """Soft aggregation over activated-slot ``p_r`` targets → Yes logit.

    ``p`` is detached offline overlap in ``[0,1]`` (slot_positive). Confidence
    ``c`` defaults to 1 for part slots / usable masks and lower for EG-only.
    """

    def __init__(self, max_slots: int = 4):
        super().__init__()
        self.max_slots = int(max_slots)
        self.w0 = nn.Parameter(torch.zeros(()))
        self.w = nn.Parameter(torch.zeros(self.max_slots))
        self.source_gain = nn.Parameter(torch.tensor([1.0, 1.25]))  # eg, part

    def forward(
        self,
        p: torch.Tensor,
        c: torch.Tensor,
        source_idx: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        p, c, source_idx, slot_mask: [B, S]
        returns logits [B]
        """
        S = p.size(1)
        w = self.w[:S]
        gain = self.source_gain[source_idx.clamp(0, 1)]
        g = p * c * gain
        scored = (w * g * slot_mask).sum(dim=-1) + self.w0
        return scored


def build_slot_tensors(
    metas: Sequence[Optional[Dict[str, Any]]],
    device: torch.device,
    max_slots: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return p, c, source_idx, slot_mask, yes_target, sample_mask — all [B, ...]"""
    B = len(metas)
    p = torch.zeros(B, max_slots, device=device)
    c = torch.zeros(B, max_slots, device=device)
    src = torch.zeros(B, max_slots, dtype=torch.long, device=device)
    sm = torch.zeros(B, max_slots, device=device)
    y = torch.zeros(B, device=device)
    smpl = torch.zeros(B, device=device)
    for i, m in enumerate(metas):
        if not m or not m.get("ecpr_active"):
            continue
        smpl[i] = 1.0
        y[i] = 1.0 if m.get("is_anomaly") else 0.0
        pos = list(m.get("slot_positive") or [])
        sources = list(m.get("slot_sources") or [])
        n = min(max_slots, max(len(pos), len(sources), len(m.get("boxes") or [])))
        for s in range(n):
            sm[i, s] = 1.0
            p[i, s] = float(pos[s]) if s < len(pos) else 0.0
            is_part = s < len(sources) and sources[s] == "part"
            src[i, s] = 1 if is_part else 0
            # part / usable evidence gets full confidence; empty EG slot lower
            c[i, s] = 1.0 if is_part or p[i, s] > 0 else 0.5
    return p, c, src, sm, y, smpl


def compute_ecpr_chain_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    metas: Optional[Sequence[Any]],
    tokenizer,
    *,
    no_ids: Set[int],
    yes_ids: Set[int],
    lambda_localize: float = 0.5,
    lambda_attr: float = 0.5,
    lambda_agg: float = 0.5,
    base_weight: float = 1.0,
    attr_gate_scale: float = 0.25,
    num_items_in_batch: Optional[Union[int, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Token-reweighted CE implementing localize / attr / agg on ECPR samples."""
    if labels.dim() != 2 or logits.dim() != 3:
        raise ValueError(f"Expected logits [B,T,V] and labels [B,T], got {tuple(logits.shape)} / {tuple(labels.shape)}")
    t_logits, t_labels = logits.size(1), labels.size(1)
    if t_logits != t_labels:
        if abs(t_logits - t_labels) > 1:
            raise ValueError(
                f"logits T={t_logits} vs labels T={t_labels}: pass image-expanded labels "
                "(model dpo_forward), not raw text labels."
            )
        t = min(t_logits, t_labels)
        logits = logits[:, :t]
        labels = labels[:, :t]

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    B, T, vocab = shift_logits.shape
    meta_list = _as_meta_list(metas, B)

    flat_logits = shift_logits.view(-1, vocab)
    flat_labels = shift_labels.reshape(-1).to(device=flat_logits.device, dtype=torch.long)
    per_tok = F.cross_entropy(flat_logits, flat_labels, reduction="none", ignore_index=IGNORE_INDEX)
    weights = torch.full_like(per_tok, float(base_weight))
    valid = flat_labels.ne(IGNORE_INDEX)

    decision_ids = set(no_ids) | set(yes_ids)
    last_mask = _last_decision_mask(shift_labels, decision_ids)
    flat_last = last_mask.reshape(-1)

    n_ecpr = 0.0
    n_loc = 0.0
    n_attr = 0.0
    n_agg = 0.0

    for b, meta in enumerate(meta_list):
        if not meta or not meta.get("ecpr_active"):
            continue
        n_ecpr += 1.0
        row = shift_labels[b]
        decoded = _decode_label_row(labels[b], tokenizer)

        # localize
        if float(lambda_localize) != 0.0:
            loc_m = _token_positions_matching_substrings(row, tokenizer, _localize_needles(meta))
            if bool(loc_m.any()):
                n_loc += 1.0
                boost = 1.0 + float(lambda_localize)
                idx = loc_m.nonzero(as_tuple=False).flatten()
                flat_idx = b * T + idx
                weights[flat_idx] = torch.maximum(weights[flat_idx], weights.new_full((), boost))

        # attr (gated)
        if float(lambda_attr) != 0.0:
            ok = _localization_correct(decoded, meta)
            scale = 1.0 if ok else float(attr_gate_scale)
            attr_m = _token_positions_matching_substrings(row, tokenizer, _attr_needles(meta))
            if bool(attr_m.any()):
                n_attr += 1.0
                boost = 1.0 + float(lambda_attr) * scale
                idx = attr_m.nonzero(as_tuple=False).flatten()
                flat_idx = b * T + idx
                weights[flat_idx] = torch.maximum(weights[flat_idx], weights.new_full((), boost))

        # agg: final Yes/No
        if float(lambda_agg) != 0.0 and bool(last_mask[b].any()):
            n_agg += 1.0
            boost = 1.0 + float(lambda_agg)
            # slightly stronger when parts were injected (decision should use A)
            if int(meta.get("n_part") or 0) > 0:
                boost += 0.25 * float(lambda_agg)
            idx = last_mask[b].nonzero(as_tuple=False).flatten()
            flat_idx = b * T + idx
            weights[flat_idx] = torch.maximum(weights[flat_idx], weights.new_full((), boost))

    weighted = per_tok * weights
    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            denom = num_items_in_batch.to(dtype=weighted.dtype, device=weighted.device).clamp_min(1.0)
        else:
            denom = weighted.new_tensor(float(max(int(num_items_in_batch), 1)))
        loss = weighted.sum() / denom
    else:
        denom = (weights * valid.to(weights.dtype)).sum().clamp_min(1.0)
        loss = weighted.sum() / denom

    with torch.no_grad():
        stats = {
            "loss": float(loss.detach().item()),
            "n_ecpr": n_ecpr,
            "n_loc": n_loc,
            "n_attr": n_attr,
            "n_agg": n_agg,
            "mean_w": float(weights[valid].mean().item()) if bool(valid.any()) else 1.0,
            "n_yes": float((flat_last & _id_mask(flat_labels, yes_ids) & valid).sum().item()) if yes_ids else 0.0,
            "n_no": float((flat_last & _id_mask(flat_labels, no_ids) & valid).sum().item()) if no_ids else 0.0,
        }
    return loss, stats


def compute_ecpr_local_agg_bce(
    head: EcprLocalAggHead,
    metas: Optional[Sequence[Any]],
    device: torch.device,
    *,
    max_slots: int = 4,
    batch_size: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Soft-agg BCE. Always runs ``head`` so ZeRO-3 ranks stay collective-aligned.

    Batches with no ECPR samples still execute the head forward and return a
    numerically-zero loss that is connected to head parameters (``* 0``).
    Skipping the head on some ranks caused NCCL ALLGATHER timeouts under ZeRO-3.
    """
    if batch_size is None:
        batch_size = len(metas) if metas is not None else 1
    batch_size = max(int(batch_size), 1)
    meta_list = _as_meta_list(metas, batch_size)
    p, c, src, sm, y, smpl = build_slot_tensors(meta_list, device=device, max_slots=max_slots)
    # MUST run on every rank every step (ZeRO-3 allgather of head weights).
    logits = head(p, c, src, sm)
    per = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    active = smpl.sum().clamp_min(0.0)
    if float(active.item()) <= 0:
        # Zero contribution but keeps autograd / param access identical across ranks.
        loss = (logits.float().sum() + per.float().sum()) * 0.0
        stats = {"loss": 0.0, "n": 0.0, "acc": 0.0, "mean_p": 0.0}
        return loss, stats
    loss = (per * smpl).sum() / active.clamp_min(1.0)
    with torch.no_grad():
        pred = (logits.sigmoid() >= 0.5).float()
        acc = ((pred == y).float() * smpl).sum() / active.clamp_min(1.0)
        stats = {
            "loss": float(loss.detach().item()),
            "n": float(active.item()),
            "acc": float(acc.item()),
            "mean_p": float((p * sm).sum().item() / sm.sum().clamp_min(1.0).item()) if float(sm.sum()) > 0 else 0.0,
        }
    return loss, stats
