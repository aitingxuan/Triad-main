"""
Scheme 4 — self-questioning requery with Bayesian logit fusion (mode ``selfq``).

Idea
----
On the most-uncertain ``top_frac`` of a run (same quantile gate as Defv3), let the
model *generate its own* yes/no verification question (T2: propose -> answer),
instead of a hand-written per-class template. The generated question is answered
by a forced two-choice readout (A=Yes / B=No), so the answer is **always binary
and probabilistic** regardless of how free-form the question is.

Two design choices (no absolute magic thresholds):

1. **Stability by relative sharpness (entropy gate).** A sub-answer is accepted as
   evidence only when its two-way entropy is *lower* than the main question's
   entropy, i.e. the sub-question "sees more clearly" than the thing it is asked
   to disambiguate. This replaces the old ``tau_ev=0.7`` absolute confidence cut.

2. **Bayesian logit fusion.** The final decision is the prior (main) answer updated
   by independent evidence::

       logit_final = logit(p_main_A) + Σ_i  w_i · sub_logit_A(i)
       w_i ∈ {0, 1}   (1 = entropy-gated accept, else dropped)

   No band, no delta, no lambda — the decision boundary is exactly 0.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Saturation guard for a single evidence logit (≈ p=0.9975 already saturated).
# It is a numeric cap, not a tuned decision constant.
DEFAULT_EVIDENCE_CLIP = 6.0

# Support-token normalization: map free-form model words to A / B / neither.
_SUPPORT_A = {"a", "yes", "defective", "anomaly", "abnormal", "anomalous", "1", "true"}
_SUPPORT_B = {"b", "no", "normal", "good", "ok", "nondefective", "non_defective", "0", "false"}
_SUPPORT_NONE = {"neither", "none", "n/a", "na", "both", "-", "uninformative", "unknown", "?"}


SELFQ_INSTRUCTION = (
    "Now act as a careful inspector of the SAME image. Your current answer above may be "
    "wrong, so you must probe it with ONE new yes/no question you write yourself.\n"
    "The question MUST:\n"
    "  - be answerable by looking only at this image (about a concrete visual property),\n"
    "  - have a \"Yes\" and a \"No\" that point to OPPOSITE answers of the original question,\n"
    "  - be discriminative: it should separate the two original options, not restate the question.\n"
    "Output EXACTLY these three lines and nothing else:\n"
    "QUESTION: <your yes/no question>\n"
    "YES_SUPPORTS: <A or B>\n"
    "NO_SUPPORTS: <A or B>\n"
    "where A and B are the two options of the original question."
)


def build_instruction(main_text: Optional[str] = None, prior_qs: Optional[Sequence[str]] = None) -> str:
    """Structured generation instruction, optionally echoing the main question text."""
    parts: List[str] = []
    if main_text:
        parts.append(f'Original question: "{str(main_text).strip()}"')
    if prior_qs:
        done = "; ".join(str(q).strip() for q in prior_qs if str(q).strip())
        if done:
            parts.append(f"Already asked (do not repeat): {done}")
    parts.append(SELFQ_INSTRUCTION)
    return "\n".join(parts)


_LINE_RE = {
    "question": re.compile(r"QUESTION\s*[:：]\s*(.+)", re.IGNORECASE),
    "yes_supports": re.compile(r"YES[_\s]*SUPPORTS\s*[:：]\s*([A-Za-z0-9_/\-?]+)", re.IGNORECASE),
    "no_supports": re.compile(r"NO[_\s]*SUPPORTS\s*[:：]\s*([A-Za-z0-9_/\-?]+)", re.IGNORECASE),
}


def _norm_support(raw: str) -> Optional[str]:
    s = str(raw or "").strip().strip("*` .\n\t").lower()
    if s in _SUPPORT_A:
        return "A"
    if s in _SUPPORT_B:
        return "B"
    if s in _SUPPORT_NONE:
        return "neither"
    # last resort: a bare leading a/b
    if s.startswith("a"):
        return "A"
    if s.startswith("b"):
        return "B"
    return None


def parse_selfq_generation(text: str) -> Optional[Dict[str, str]]:
    """Parse QUESTION / YES_SUPPORTS / NO_SUPPORTS from model text. None if unusable."""
    if not text:
        return None
    body = str(text)
    q_m = _LINE_RE["question"].search(body)
    y_m = _LINE_RE["yes_supports"].search(body)
    n_m = _LINE_RE["no_supports"].search(body)
    if not (q_m and y_m and n_m):
        return None
    question = q_m.group(1).strip().strip("*`").strip()
    # keep only the first sentence-ish; drop trailing fancy
    if not question:
        return None
    question = question.split("\n")[0].strip()
    y = _norm_support(y_m.group(1))
    n = _norm_support(n_m.group(1))
    if y is None or n is None:
        return None
    return {"question": question, "yes_supports": y, "no_supports": n}


def binary_entropy(p_yes: float) -> float:
    """Entropy (nats) of a 2-way distribution {p_yes, 1-p_yes}; in [0, ln 2]."""
    p = min(max(float(p_yes), 1e-9), 1.0 - 1e-9)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def entropy_accept(p_sub_yes: float, p_main_a: float) -> bool:
    """Accept sub-answer as evidence iff it is *sharper* than the main question."""
    return binary_entropy(p_sub_yes) < binary_entropy(p_main_a)


def sub_evidence_logit_for_a(
    p_yes: float,
    yes_supports: str,
    no_supports: str,
    clip: float = DEFAULT_EVIDENCE_CLIP,
) -> Tuple[float, float]:
    """Signed evidence toward option A from one sub-question.

    Returns ``(weight, signed_logit)`` where ``weight`` is 1 only when the question
    is informative (Yes/No point to opposite options) and 0 otherwise.
    """
    y = str(yes_supports).upper()
    n = str(no_supports).upper()
    informative = {y, n} == {"A", "B"}
    if not informative:
        return 0.0, 0.0
    polarity = 1.0 if y == "A" else -1.0
    z = _logit(p_yes)
    z = max(-float(clip), min(float(clip), z))
    return 1.0, polarity * z


def bayes_fuse(
    p_main_a: float,
    evidences: Sequence[Tuple[float, float]],
    clip: float = DEFAULT_EVIDENCE_CLIP,
) -> Dict[str, float]:
    """Update the main A/B belief by independent sub-evidence logits.

    ``evidences`` is a list of ``(weight, signed_logit_toward_A)``. The decision
    boundary is exactly 0 in log-odds space.
    """
    logit_main = _logit(p_main_a)
    acc = 0.0
    n_used = 0
    for w, z in evidences:
        if w <= 0.0:
            continue
        z = max(-float(clip), min(float(clip), float(z)))
        acc += float(w) * z
        n_used += 1
    logit_final = logit_main + acc
    p_final = 1.0 / (1.0 + math.exp(-logit_final))
    return {
        "logit_main": logit_main,
        "evidence_logit": acc,
        "logit_final": logit_final,
        "p_final": p_final,
        "n_evidence": float(n_used),
    }
