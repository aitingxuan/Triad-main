#!/usr/bin/env python3
"""Build a v0 FP/TN/TP casebook for P0 attention probing from a full ans JSON.

Default quota (48): FP=24, TN=12, TP=12, weighted toward low-TNR classes.

Example:
  python llava_eval/build_p0_fp_casebook.py \\
    --ans evaluation/mvtec/results/merged_llava_qwen_triad_9ROI_cotmv6/ans_mvtec_all_0shot_randomroi_musc_v0.json \\
    --out LLM_judge_output/casebooks/cotmv6_v0_fp_tn_tp_n48.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def parse_pred(ans_text: str) -> Optional[int]:
    u = (ans_text or "").strip().upper()
    if not u:
        return None
    if u.startswith("A") or u == "YES" or "<ANSWER>YES" in u.replace(" ", ""):
        return 1
    if u.startswith("B") or u == "NO" or "<ANSWER>NO" in u.replace(" ", ""):
        return 0
    return None


def load_ans(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("ans file must be a JSON list")
    rows = []
    for a in data:
        pred = parse_pred(a.get("conversations", [{}, {}])[1].get("value", ""))
        gt = a.get("gt")
        if pred is None or gt is None:
            continue
        if int(gt) == 0 and pred == 1:
            group = "FP"
        elif int(gt) == 0 and pred == 0:
            group = "TN"
        elif int(gt) == 1 and pred == 1:
            group = "TP"
        elif int(gt) == 1 and pred == 0:
            group = "FN"
        else:
            continue
        rows.append({
            "id": str(a.get("id")),
            "subset": a.get("subset"),
            "gt": int(gt),
            "pred_ref": int(pred),
            "group": group,
            "origin_path": a.get("origin_path"),
            "image": a.get("image"),
            "answer_ref": a.get("conversations", [{}, {}])[1].get("value"),
        })
    return rows


# Default FP quota by subset (sum=24). Heavy on bottle/pill/transistor.
DEFAULT_FP_QUOTA = {
    "bottle": 6,
    "pill": 6,
    "transistor": 6,
    "cable": 3,
    "screw": 3,
}


def _take(pool: List[dict], n: int, rng: random.Random) -> List[dict]:
    if n <= 0 or not pool:
        return []
    xs = list(pool)
    rng.shuffle(xs)
    return xs[: min(n, len(xs))]


def build_casebook(
    rows: List[dict],
    *,
    seed: int = 0,
    n_fp: int = 24,
    n_tn: int = 12,
    n_tp: int = 12,
    fp_quota: Optional[Dict[str, int]] = None,
) -> List[dict]:
    rng = random.Random(seed)
    by_group_sub: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        by_group_sub[(r["group"], str(r["subset"]))].append(r)

    fp_quota = dict(fp_quota or DEFAULT_FP_QUOTA)
    # Scale / trim quota to n_fp
    qsum = sum(fp_quota.values())
    if qsum != n_fp and qsum > 0:
        # keep proportions, then fix remainder
        scaled = {k: max(0, int(round(v * n_fp / qsum))) for k, v in fp_quota.items()}
        while sum(scaled.values()) > n_fp:
            k = max(scaled, key=scaled.get)
            scaled[k] -= 1
        while sum(scaled.values()) < n_fp:
            # add to subsets that still have unused FPs
            candidates = [
                k for k, v in scaled.items()
                if v < len(by_group_sub.get(("FP", k), []))
            ]
            if not candidates:
                break
            k = max(candidates, key=lambda x: len(by_group_sub.get(("FP", x), [])) - scaled[x])
            scaled[k] += 1
        fp_quota = scaled

    picked: List[dict] = []
    used_ids = set()

    # FP by quota
    for sub, need in sorted(fp_quota.items(), key=lambda kv: -kv[1]):
        pool = [r for r in by_group_sub.get(("FP", sub), []) if r["id"] not in used_ids]
        chunk = _take(pool, need, rng)
        for r in chunk:
            used_ids.add(r["id"])
        picked.extend(chunk)

    # Fill remaining FP from leftover pool (any subset)
    if len([r for r in picked if r["group"] == "FP"]) < n_fp:
        need = n_fp - len([r for r in picked if r["group"] == "FP"])
        pool = [r for r in rows if r["group"] == "FP" and r["id"] not in used_ids]
        chunk = _take(pool, need, rng)
        for r in chunk:
            used_ids.add(r["id"])
        picked.extend(chunk)

    fp_subs = sorted({r["subset"] for r in picked if r["group"] == "FP"})

    def _paired(group: str, n: int) -> List[dict]:
        """Prefer same subsets as FP; round-robin then fill."""
        out: List[dict] = []
        if not fp_subs or n <= 0:
            return out
        per = max(1, n // max(len(fp_subs), 1))
        for sub in fp_subs:
            pool = [r for r in by_group_sub.get((group, sub), []) if r["id"] not in used_ids]
            chunk = _take(pool, per, rng)
            for r in chunk:
                used_ids.add(r["id"])
            out.extend(chunk)
            if len(out) >= n:
                return out[:n]
        # fill
        need = n - len(out)
        # first remaining in fp_subs, then any
        pool = []
        for sub in fp_subs:
            pool.extend([r for r in by_group_sub.get((group, sub), []) if r["id"] not in used_ids])
        pool.extend([r for r in rows if r["group"] == group and r["id"] not in used_ids])
        # dedupe preserve order
        seen = set()
        uniq = []
        for r in pool:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            uniq.append(r)
        chunk = _take(uniq, need, rng)
        for r in chunk:
            used_ids.add(r["id"])
        out.extend(chunk)
        return out[:n]

    picked.extend(_paired("TN", n_tn))
    picked.extend(_paired("TP", n_tp))

    rng.shuffle(picked)
    return picked


def main():
    parser = argparse.ArgumentParser(description="Build P0 FP/TN/TP casebook from v0 ans JSON")
    parser.add_argument("--ans", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-fp", type=int, default=24)
    parser.add_argument("--n-tn", type=int, default=12)
    parser.add_argument("--n-tp", type=int, default=12)
    args = parser.parse_args()

    rows = load_ans(args.ans)
    counts = defaultdict(int)
    for r in rows:
        counts[r["group"]] += 1
    print(f"[casebook] loaded {len(rows)} labeled | {dict(counts)}")

    picked = build_casebook(
        rows, seed=args.seed, n_fp=args.n_fp, n_tn=args.n_tn, n_tp=args.n_tp,
    )
    gcounts = defaultdict(int)
    by_sub = defaultdict(lambda: defaultdict(int))
    for r in picked:
        gcounts[r["group"]] += 1
        by_sub[r["subset"]][r["group"]] += 1
    print(f"[casebook] picked {len(picked)} | {dict(gcounts)}")
    for sub in sorted(by_sub):
        print(f"  {sub}: {dict(by_sub[sub])}")

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[casebook] wrote {args.out}")


if __name__ == "__main__":
    main()
