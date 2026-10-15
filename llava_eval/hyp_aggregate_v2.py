"""
Soft evidence aggregation for scheme_requery hyp_v2 (A2).

Same H_A/H_B sub-questions as hyp; changes only how votes become A/B:
  1) soft vote v = (2*p_yes - 1), gated by tau_ev
  2) pairwise antisymmetric weight (agree boost / conflict shrink)
  3) high-mass low-margin conflict gate -> fallback
  4) mix evidence score with main-question p_f via beta(margin)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def soft_vote(p_yes: float, conf: float, tau_ev: float) -> float:
    """Continuous evidence in [-1, 1]; 0 if below evidence gate."""
    if conf < tau_ev:
        return 0.0
    return float(2.0 * p_yes - 1.0)


def _pair_weight(ra: Dict[str, Any], rb: Dict[str, Any], va: float, vb: float,
                 conflict_lambda: float, agree_lambda: float) -> Tuple[float, str]:
    both_active = (va != 0.0) and (vb != 0.0)
    if both_active:
        if ra["letter"] == rb["letter"]:
            return float(conflict_lambda), "conflict"
        return float(agree_lambda), "agree"
    if va != 0.0 or vb != 0.0:
        return 1.0, "single"
    return 0.0, "inactive"


def aggregate_hyp_v2(
    sub_answers: List[Dict[str, Any]],
    p_f: float,
    llm_letter: str,
    *,
    tau_ev: float = 0.7,
    delta: float = 0.5,
    conflict_lambda: float = 0.3,
    agree_lambda: float = 1.2,
    margin_ref: float = 1.0,
    conflict_ratio: float = 0.35,
    soft_temp: float = 1.0,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Returns (letter, decision_src, diagnostics).

    score > 0 supports defect (A); score < 0 supports normal (B).
    """
    ha = [r for r in sub_answers if r.get("hyp") == "H_A"]
    hb = [r for r in sub_answers if r.get("hyp") == "H_B"]

    score = 0.0
    mass = 0.0
    S_A = 0.0  # defect-supporting soft mass
    S_B = 0.0  # normal-supporting soft mass
    pair_meta: List[Dict[str, Any]] = []
    paired_ha = set()
    paired_hb = set()

    n_pair = min(len(ha), len(hb))
    for i in range(n_pair):
        ra, rb = ha[i], hb[i]
        paired_ha.add(i)
        paired_hb.add(i)
        va = soft_vote(float(ra["p_A"]), float(ra["conf"]), tau_ev)
        vb = soft_vote(float(rb["p_A"]), float(rb["conf"]), tau_ev)
        # H_A Yes -> +defect; H_B Yes -> +normal (== -defect)
        c_a = va
        c_b = -vb
        w, pair_status = _pair_weight(ra, rb, va, vb, conflict_lambda, agree_lambda)

        contrib = w * (c_a + c_b)
        pair_mass = w * (abs(c_a) + abs(c_b))
        score += contrib
        mass += pair_mass
        for c in (w * c_a, w * c_b):
            if c > 0:
                S_A += c
            elif c < 0:
                S_B += -c

        pair_meta.append({
            "i": i,
            "status": pair_status,
            "w": w,
            "v_A": va,
            "v_B": vb,
            "contrib": contrib,
        })

    for i, r in enumerate(ha):
        if i in paired_ha:
            continue
        v = soft_vote(float(r["p_A"]), float(r["conf"]), tau_ev)
        score += v
        mass += abs(v)
        if v > 0:
            S_A += v
        elif v < 0:
            S_B += -v

    for i, r in enumerate(hb):
        if i in paired_hb:
            continue
        v = soft_vote(float(r["p_A"]), float(r["conf"]), tau_ev)
        # H_B Yes (v>0) supports normal
        c = -v
        score += c
        mass += abs(c)
        if c > 0:
            S_A += c
        elif c < 0:
            S_B += -c

    margin = abs(score)
    m_prior = 2.0 * float(p_f) - 1.0
    margin_ref = max(float(margin_ref), 1e-6)
    soft_temp = max(float(soft_temp), 1e-6)

    diag: Dict[str, Any] = {
        "score": score,
        "mass": mass,
        "margin": margin,
        "S_A": S_A,
        "S_B": S_B,
        "pairs": pair_meta,
        "beta": None,
        "e": None,
        "final": None,
        "m_prior": m_prior,
        "gate": None,
    }

    if mass <= 0.0:
        diag["gate"] = "no_evidence"
        return llm_letter, "hyp_v2_fallback", diag

    if (margin / mass) < float(conflict_ratio):
        diag["gate"] = "conflict_ratio"
        return llm_letter, "hyp_v2_fallback_conflict", diag

    if margin < float(delta):
        diag["gate"] = "margin_lt_delta"
        return llm_letter, "hyp_v2_fallback", diag

    beta = min(1.0, margin / margin_ref)
    e = math.tanh(score / soft_temp)
    final = beta * e + (1.0 - beta) * m_prior
    letter = "A" if final >= 0.0 else "B"
    diag.update({"gate": "mix", "beta": beta, "e": e, "final": final})
    return letter, "hyp_v2_test", diag
