"""Reweight CE on the final Yes/No decision token to reduce Yes bias / raise TNR."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F

from llava.constants import IGNORE_INDEX


def _single_token_ids(tokenizer, texts: Iterable[str]) -> Set[int]:
    ids: Set[int] = set()
    for text in texts:
        for variant in (text, f" {text}", f"\n{text}"):
            tok = tokenizer.encode(variant, add_special_tokens=False)
            if len(tok) == 1:
                ids.add(int(tok[0]))
    return ids


def build_decision_token_id_sets(tokenizer) -> Tuple[Set[int], Set[int]]:
    """Return (no_ids, yes_ids) for single-token encodings."""
    no_ids = _single_token_ids(tokenizer, ["No", "no", "NO"])
    yes_ids = _single_token_ids(tokenizer, ["Yes", "yes", "YES"])
    return no_ids, yes_ids


def _last_decision_mask(
    shift_labels: torch.Tensor,
    decision_ids: Set[int],
) -> torch.Tensor:
    """Mark only the last supervised Yes/No token in each row. Labels: [B, T]."""
    if not decision_ids:
        return torch.zeros_like(shift_labels, dtype=torch.bool)

    is_dec = torch.zeros_like(shift_labels, dtype=torch.bool)
    for tid in decision_ids:
        is_dec |= shift_labels.eq(tid)
    is_dec &= shift_labels.ne(IGNORE_INDEX)

    B, T = shift_labels.shape
    idx = torch.arange(T, device=shift_labels.device).unsqueeze(0).expand(B, T)
    masked_idx = torch.where(is_dec, idx, idx.new_full((), -1))
    last = masked_idx.max(dim=1).values
    out = torch.zeros_like(is_dec)
    has = last >= 0
    if has.any():
        rows = torch.arange(B, device=shift_labels.device)
        out[rows[has], last[has]] = True
    return out


def _id_mask(labels_flat: torch.Tensor, id_set: Set[int]) -> torch.Tensor:
    mask = torch.zeros_like(labels_flat, dtype=torch.bool)
    for tid in id_set:
        mask |= labels_flat.eq(tid)
    return mask


def compute_decision_weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    no_ids: Set[int],
    yes_ids: Set[int],
    *,
    no_weight: float = 3.0,
    yes_weight: float = 1.0,
    num_items_in_batch: Optional[Union[int, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Token-weighted CE on shifted LM labels.

    ``labels`` must already be multimodal-expanded (same seq length as ``logits``),
    e.g. from LlavaQwen ``dpo_forward=True``. Raw dataloader labels will not match.

    Only the **last** Yes/No token in each sequence is reweighted (CoTM
    ``<answer>Yes/No</answer>``). Earlier Yes/No words inside ``<think>`` stay at 1.0.
    """
    if labels.dim() != 2 or logits.dim() != 3:
        raise ValueError(f"Expected logits [B,T,V] and labels [B,T], got {tuple(logits.shape)} / {tuple(labels.shape)}")
    if logits.size(0) != labels.size(0):
        raise ValueError(f"Batch mismatch: logits {tuple(logits.shape)} vs labels {tuple(labels.shape)}")
    # Truncate to common length if pad/crop differs by a token; hard-fail on large gaps.
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
    vocab = shift_logits.size(-1)

    flat_logits = shift_logits.view(-1, vocab)
    flat_labels = shift_labels.reshape(-1).to(device=flat_logits.device, dtype=torch.long)
    per_tok = F.cross_entropy(flat_logits, flat_labels, reduction="none", ignore_index=IGNORE_INDEX)

    last_mask = _last_decision_mask(shift_labels, set(no_ids) | set(yes_ids))
    flat_last = last_mask.reshape(-1)
    valid = flat_labels.ne(IGNORE_INDEX)
    weights = torch.ones_like(per_tok, dtype=per_tok.dtype)

    if no_ids and float(no_weight) != 1.0:
        weights = torch.where(
            flat_last & _id_mask(flat_labels, no_ids) & valid,
            weights.new_full((), float(no_weight)),
            weights,
        )

    if yes_ids and float(yes_weight) != 1.0:
        weights = torch.where(
            flat_last & _id_mask(flat_labels, yes_ids) & valid,
            weights.new_full((), float(yes_weight)),
            weights,
        )

    weighted = per_tok * weights
    if num_items_in_batch is not None:
        # HF Trainer may pass an int (or a 0-dim tensor).
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
            "n_valid": float(valid.sum().item()),
            "n_no": float((flat_last & _id_mask(flat_labels, no_ids) & valid).sum().item()) if no_ids else 0.0,
            "n_yes": float((flat_last & _id_mask(flat_labels, yes_ids) & valid).sum().item()) if yes_ids else 0.0,
            "mean_w": float(weights[valid].mean().item()) if bool(valid.any()) else 1.0,
            "loss": float(loss.detach().item()),
        }
    return loss, stats
