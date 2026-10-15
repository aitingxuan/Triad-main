# -*- coding: utf-8 -*-
"""Stage A: deterministic TraceMAD metrics (no LLM)."""
from __future__ import absolute_import, division, print_function, unicode_literals

import re
from collections import Counter, defaultdict

from .parse import extract_defect_folder, parse_gpt_output
from .taxonomy import location_mask_score, type_match_score

# Heuristic cues inside <think>
_YES_CUES = [
    "defect",
    "fault",
    "abnormal",
    "unusual",
    "anomaly",
    "anomal",
    "inconsist",
    "damag",
    "crack",
    "scratch",
    "contaminat",
    "misalign",
    "missing",
    "broken",
    "peculiar",
]
_NO_CUES = [
    "no defect",
    "no visible defect",
    "free from",
    "without any defect",
    "appears to be free",
    "no signs of",
    "no evidence of",
    "meets the standard",
    "no faults",
    "no peculiarities",
    "looks normal",
    "appears normal",
]


def _think_polarity(think):
    """Return 'yes', 'no', 'mixed', or 'unknown' from think text."""
    if not think:
        return "unknown"
    t = think.lower()
    has_no = any(c in t for c in _NO_CUES)
    # yes cues but exclude "no defect" style
    has_yes = False
    for c in _YES_CUES:
        if c in t:
            # skip if part of explicit negation phrases already counted
            has_yes = True
            break
    if has_no and has_yes:
        # if strong no phrases dominate short think claiming free-from
        if has_no and sum(1 for c in _NO_CUES if c in t) >= 1:
            # still mixed if also strong defect claims outside negation
            if re.search(r"\b(defect|fault|abnormal|unusual)\b", t) and not has_no:
                return "mixed"
            # e.g. "no defect" alone -> no
            if has_no and not re.search(
                r"(there (is|are) (a |an )?(defect|fault|abnormal))", t
            ):
                # if only negated mentions
                return "no" if not re.search(
                    r"(defective|faulty|abnormality|anomaly)", t
                ) else "mixed"
        return "mixed"
    if has_no:
        return "no"
    if has_yes:
        return "yes"
    return "unknown"


def _process_anchored(think, human_prompt):
    if not think:
        return 0.0
    t = think.lower()
    score = 0.0
    # generic process language
    if any(
        k in t
        for k in (
            "process",
            "stage",
            "during the",
            "manufactur",
            "assembly",
            "production",
        )
    ):
        score += 0.5
    # extract numbered step titles from prompt if present
    steps = re.findall(
        r"\d+\.\s*([A-Za-z][A-Za-z /-]{2,40})", human_prompt or ""
    )
    hit = 0
    for s in steps:
        key = s.strip().lower().split(":")[0].strip()
        if len(key) >= 4 and key in t:
            hit += 1
    if steps:
        score += 0.5 * min(1.0, hit / float(min(2, len(steps))))
    return min(1.0, score)


def _field_entailed(think, location, typ):
    """Check location/type tokens appear in think (soft)."""
    if not think:
        return 0.0 if (location or typ) else 1.0
    t = think.lower()
    scores = []
    if location:
        words = [w for w in re.split(r"[^a-z0-9]+", location.lower()) if len(w) > 3]
        if words:
            scores.append(sum(1 for w in words if w in t) / float(len(words)))
        else:
            scores.append(1.0 if location.lower() in t else 0.0)
    if typ:
        words = [w for w in re.split(r"[^a-z0-9]+", typ.lower()) if len(w) > 3]
        if words:
            scores.append(sum(1 for w in words if w in t) / float(len(words)))
        else:
            scores.append(1.0 if typ.lower() in t else 0.0)
    if not scores:
        return 1.0
    return sum(scores) / float(len(scores))


def schema_score(parsed, gt):
    """Return (score in [0,1], issues list)."""
    issues = []
    # completeness
    if not parsed["has_think"]:
        issues.append("missing_think")
    if not parsed["has_answer"]:
        issues.append("missing_answer")

    pred = parsed["answer"]
    if pred == 1:
        if not parsed["has_location"]:
            issues.append("yes_missing_location")
        if not parsed["has_type"]:
            issues.append("yes_missing_type")
    elif pred == 0:
        if parsed["has_location"] or parsed.get("location"):
            # allow empty tags? has_location already requires content
            if parsed["has_location"]:
                issues.append("no_with_location")
        if parsed["has_type"]:
            issues.append("no_with_type")

    # purity
    if parsed["preamble"]:
        issues.append("preamble_outside_tags")
    if parsed["trailing"]:
        issues.append("trailing_outside_tags")
    if not parsed["starts_with_tag"] and parsed["preamble"]:
        issues.append("does_not_start_with_tag")

    # order
    order = parsed["tag_order"]
    expected_yes = ["think", "location", "type", "answer"]
    expected_no = ["think", "answer"]
    order_ok = True
    if pred == 1 and parsed["has_think"] and parsed["has_answer"]:
        # check relative order of present required tags
        filt = [t for t in order if t in expected_yes]
        # think before answer; location before type if both
        if "think" in filt and "answer" in filt:
            if filt.index("think") > filt.index("answer"):
                order_ok = False
        if "location" in filt and "type" in filt:
            if filt.index("location") > filt.index("type"):
                order_ok = False
        if "type" in filt and "answer" in filt:
            if filt.index("type") > filt.index("answer"):
                order_ok = False
    elif pred == 0:
        filt = [t for t in order if t in expected_no]
        if filt != ["think", "answer"] and set(filt) == {"think", "answer"}:
            if filt.index("think") > filt.index("answer"):
                order_ok = False
        elif parsed["has_think"] and parsed["has_answer"]:
            if order and order[0] != "think":
                order_ok = False
    if not order_ok:
        issues.append("bad_tag_order")

    # component scores
    complete = 1.0
    if "missing_think" in issues or "missing_answer" in issues:
        complete = 0.0
    elif pred == 1 and (
        "yes_missing_location" in issues or "yes_missing_type" in issues
    ):
        complete = 0.4
    elif pred is None:
        complete = 0.0

    purity = 1.0
    if "preamble_outside_tags" in issues:
        purity -= 0.6
    if "trailing_outside_tags" in issues:
        purity -= 0.4
    purity = max(0.0, purity)

    mutex = 1.0
    if "no_with_location" in issues or "no_with_type" in issues:
        mutex = 0.0

    order_s = 0.0 if "bad_tag_order" in issues else 1.0

    score = 0.4 * complete + 0.2 * order_s + 0.2 * purity + 0.2 * mutex
    return score, issues


def logic_proxy_score(parsed, human_prompt):
    """Rule-based proxy for Trace Logic (0-1). Stage B refines with LLM."""
    think = parsed.get("think") or ""
    pred = parsed.get("answer")
    polarity = _think_polarity(think)

    # think-answer consistency
    if pred is None:
        cons = 0.0
    elif polarity == "unknown":
        cons = 0.5
    elif polarity == "mixed":
        cons = 0.4
    elif (pred == 1 and polarity == "yes") or (pred == 0 and polarity == "no"):
        cons = 1.0
    else:
        cons = 0.0

    evidence = 0.0
    if think:
        # length + concrete visual words
        words = think.split()
        evidence += min(0.5, len(words) / 80.0)
        vis = [
            "color",
            "shape",
            "surface",
            "edge",
            "left",
            "right",
            "top",
            "bottom",
            "visible",
            "appear",
            "bristle",
            "scratch",
            "crack",
            "hole",
            "stain",
        ]
        if any(v in think.lower() for v in vis):
            evidence += 0.5
        evidence = min(1.0, evidence)

    process = _process_anchored(think, human_prompt)
    entail = _field_entailed(
        think, parsed.get("location"), parsed.get("type")
    )

    # if answer yes, fields should be entailed; if no, entail defaults high
    logic = 0.35 * cons + 0.25 * evidence + 0.2 * process + 0.2 * entail
    return {
        "logic_proxy": logic,
        "think_polarity": polarity,
        "think_answer_consistent": cons,
        "evidence_explicit": evidence,
        "process_anchored": process,
        "field_entailed": entail,
    }


def completeness_score(parsed, gt):
    """Diagnostic completeness 0-1."""
    pred = parsed.get("answer")
    think = parsed.get("think") or ""
    if gt == 1:
        parts = []
        parts.append(1.0 if pred == 1 else 0.0)
        parts.append(1.0 if parsed.get("has_location") else 0.0)
        parts.append(1.0 if parsed.get("has_type") else 0.0)
        # root-cause hint
        if any(
            k in think.lower()
            for k in ("during", "process", "stage", "caused", "occurred")
        ):
            parts.append(1.0)
        else:
            parts.append(0.0)
        return sum(parts) / float(len(parts))
    else:
        # normal sample
        parts = []
        parts.append(1.0 if pred == 0 else 0.0)
        parts.append(1.0 if len(think.split()) >= 20 else 0.5 if think else 0.0)
        # over-diagnosis penalty already in pred
        return sum(parts) / float(len(parts))


def assign_failure_modes(parsed, gt, schema_issues, logic):
    modes = []
    pred = parsed.get("answer")
    if pred is None:
        modes.append("schema_drift")
    if any(
        x in schema_issues
        for x in (
            "preamble_outside_tags",
            "bad_tag_order",
            "missing_think",
            "missing_answer",
        )
    ):
        modes.append("schema_drift")
    if pred is not None and pred != gt:
        modes.append("fragile_grounding")
    if gt == 1 and (
        not parsed.get("has_location")
        or not parsed.get("has_type")
        or (logic.get("evidence_explicit", 0) < 0.3)
    ):
        modes.append("incomplete_diagnosis")
    if logic.get("think_answer_consistent", 1) < 0.5:
        modes.append("weak_logical_grounding")
    if gt == 1 and pred == 0 and logic.get("think_polarity") == "no":
        modes.append("overconfident_normal")
    if logic.get("process_anchored", 0) < 0.2 and pred == 1:
        modes.append("process_hallucination_risk")
    # unique preserve order
    seen = set()
    out = []
    for m in modes:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def overall_score(acc01, schema, logic_proxy, completeness, grounding):
    """Weighted overall in [0, 5], with hard caps."""
    # map to 1-5 style like RobustMAD, but allow 0 for total failure
    def to5(x):
        return 1.0 + 4.0 * max(0.0, min(1.0, x))

    a = to5(acc01 if acc01 is not None else 0.0)
    s = to5(schema)
    l = to5(logic_proxy)
    c = to5(completeness)
    g = to5(grounding if grounding is not None else 0.5)

    overall = 0.30 * a + 0.20 * s + 0.25 * l + 0.15 * c + 0.10 * g

    if acc01 is None:
        return 0.0
    if schema < 0.5:
        overall = min(overall, 2.0)
    if acc01 == 0.0 and logic_proxy >= 0.7:
        # confident wrong
        overall = min(overall, 1.5)
    return overall


def evaluate_sample_stage_a(sample, mask_arr=None):
    """Evaluate one ans json sample. mask_arr optional numpy array."""
    human = ""
    gpt = ""
    for turn in sample.get("conversations", []):
        if turn.get("from") == "human":
            human = turn.get("value") or ""
        elif turn.get("from") == "gpt":
            gpt = turn.get("value") or ""

    parsed = parse_gpt_output(gpt)
    gt = int(sample.get("gt", 0))
    defect = extract_defect_folder(sample.get("origin_path"))
    subset = sample.get("subset")

    schema, issues = schema_score(parsed, gt)
    logic = logic_proxy_score(parsed, human)
    comp = completeness_score(parsed, gt)

    pred = parsed["answer"]
    acc01 = None if pred is None else (1.0 if pred == gt else 0.0)

    tscore = None
    if gt == 1:
        tscore = type_match_score(parsed.get("type"), defect)
    elif pred == 1:
        tscore = 0.0  # false positive typed somehow
    else:
        tscore = 1.0 if pred == 0 else None

    lscore = None
    if gt == 1 and mask_arr is not None:
        lscore = location_mask_score(parsed.get("location"), mask_arr)
    elif gt == 1 and parsed.get("has_location"):
        lscore = None  # unknown without mask
    elif gt == 0:
        lscore = 1.0 if not parsed.get("has_location") else 0.5

    # grounding combine
    g_parts = []
    if tscore is not None:
        g_parts.append(tscore)
    if lscore is not None:
        g_parts.append(lscore)
    # object name consistency
    if subset and parsed.get("think"):
        g_parts.append(1.0 if subset.replace("_", " ") in parsed["think"].lower()
                        or subset.split("_")[0] in parsed["think"].lower()
                        else 0.0)
    grounding = sum(g_parts) / float(len(g_parts)) if g_parts else None

    modes = assign_failure_modes(parsed, gt, issues, logic)
    overall = overall_score(acc01, schema, logic["logic_proxy"], comp, grounding)

    return {
        "id": sample.get("id"),
        "subset": subset,
        "gt": gt,
        "defect_folder": defect,
        "pred": pred,
        "decision_correct": None if acc01 is None else bool(acc01),
        "schema_score": schema,
        "schema_issues": issues,
        "logic_proxy": logic["logic_proxy"],
        "think_polarity": logic["think_polarity"],
        "think_answer_consistent": logic["think_answer_consistent"],
        "evidence_explicit": logic["evidence_explicit"],
        "process_anchored": logic["process_anchored"],
        "field_entailed": logic["field_entailed"],
        "completeness": comp,
        "type_match": tscore,
        "location_mask_match": lscore,
        "grounding": grounding,
        "overall": overall,
        "failure_modes": modes,
        "think_n_words": len((parsed.get("think") or "").split()),
        "has_preamble": bool(parsed.get("preamble")),
        "location": parsed.get("location"),
        "type": parsed.get("type"),
        "parsed": parsed,
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / float(len(xs))


def aggregate_stage_a(rows):
    """Aggregate per-sample Stage A rows into summary + per-subset."""
    by_sub = defaultdict(list)
    for r in rows:
        by_sub[r.get("subset") or "unknown"].append(r)

    def summarize(group):
        n = len(group)
        decided = [r for r in group if r["pred"] is not None]
        tp = sum(1 for r in decided if r["gt"] == 1 and r["pred"] == 1)
        tn = sum(1 for r in decided if r["gt"] == 0 and r["pred"] == 0)
        fp = sum(1 for r in decided if r["gt"] == 0 and r["pred"] == 1)
        fn = sum(1 for r in decided if r["gt"] == 1 and r["pred"] == 0)
        und = sum(1 for r in group if r["pred"] is None)
        eps = 1e-12
        mode_c = Counter()
        for r in group:
            for m in r.get("failure_modes") or []:
                mode_c[m] += 1
        return {
            "n": n,
            "acc": (tp + tn) / float(tp + tn + fp + fn + eps) if decided else None,
            "precision": tp / float(tp + fp + eps),
            "recall": tp / float(tp + fn + eps),
            "tnr": tn / float(tn + fp + eps),
            "fnr": fn / float(tp + fn + eps),
            "undecided": und,
            "schema_mean": _mean([r["schema_score"] for r in group]),
            "logic_proxy_mean": _mean([r["logic_proxy"] for r in group]),
            "completeness_mean": _mean([r["completeness"] for r in group]),
            "grounding_mean": _mean([r["grounding"] for r in group]),
            "type_match_mean": _mean([r["type_match"] for r in group]),
            "overall_mean": _mean([r["overall"] for r in group]),
            "preamble_rate": _mean([1.0 if r["has_preamble"] else 0.0 for r in group]),
            "failure_mode_counts": dict(mode_c),
        }

    per_subset = {k: summarize(v) for k, v in sorted(by_sub.items())}
    overall = summarize(rows)
    return {"overall": overall, "per_subset": per_subset}
