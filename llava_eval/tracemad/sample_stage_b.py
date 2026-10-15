# -*- coding: utf-8 -*-
"""Select a stratified subset of samples for Stage-B LLM judging."""
from __future__ import absolute_import, division, print_function, unicode_literals

import random
from collections import defaultdict


def select_stage_b_ids(stage_a_rows, budget=200, seed=0, correct_frac=0.2):
    """Pick sample ids for LLM judge.

    Priority:
      1) all undecided / schema-broken / incorrect (up to budget)
      2) fill remaining with correct samples (correct_frac of budget as soft target,
         but never exceed remaining slots)
    """
    rng = random.Random(seed)
    bad = []
    good = []
    for r in stage_a_rows:
        is_bad = (
            r.get("pred") is None
            or r.get("decision_correct") is False
            or (r.get("schema_score") is not None and r["schema_score"] < 0.7)
            or (r.get("failure_modes") and len(r["failure_modes"]) > 0 and r.get("decision_correct") is False)
        )
        # always include schema preamble / missing answer
        if r.get("has_preamble") or r.get("pred") is None or r.get("decision_correct") is False:
            is_bad = True
        if is_bad:
            bad.append(r)
        else:
            good.append(r)

    # stratify bad by subset
    def stratified_take(rows, k):
        if k <= 0 or not rows:
            return []
        by = defaultdict(list)
        for r in rows:
            by[r.get("subset") or "unk"].append(r)
        for v in by.values():
            rng.shuffle(v)
        keys = list(by.keys())
        rng.shuffle(keys)
        picked = []
        while len(picked) < k and keys:
            progress = False
            for key in list(keys):
                if by[key]:
                    picked.append(by[key].pop())
                    progress = True
                    if len(picked) >= k:
                        break
                else:
                    keys.remove(key)
            if not progress:
                break
        return picked

    n_bad = min(len(bad), budget)
    # reserve some slots for correct if budget allows
    n_good_target = int(budget * correct_frac)
    if n_bad + n_good_target > budget:
        n_good_target = max(0, budget - n_bad)
    # if bad is small, give leftover to good
    remaining = budget - n_bad
    n_good = min(len(good), max(n_good_target, remaining))

    picked_bad = stratified_take(bad, n_bad)
    # recompute leftover
    leftover = budget - len(picked_bad)
    picked_good = stratified_take(good, min(n_good, leftover))

    selected = picked_bad + picked_good
    rng.shuffle(selected)
    return [r["id"] for r in selected]
