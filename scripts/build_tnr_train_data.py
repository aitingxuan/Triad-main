#!/usr/bin/env python3
"""Build CoTM No / InstructIAD normal subsets for TNR-oriented finetune.

Supports CoTM_v6 (and older CoTM / CoTM_v3) answer formats:
  - <answer>Yes/No</answer>  (process_align / v3)
  - Overall, the answer is: Yes/No  (checklist)
  - A/B letter mapped through the question options (order can flip)

Those stay in the full CoTM set but are not repeated as normal samples when
decision_label returns None (e.g. checklist-writing prompts with no image decision).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_OPT_A = re.compile(r"A\.\s*(Yes|No)\b", re.I)
_OPT_B = re.compile(r"B\.\s*(Yes|No)\b", re.I)
_OVERALL = re.compile(r"Overall, the answer is:\s*(Yes|No)\b", re.I)
_ANSWER_TAG = re.compile(r"<answer>\s*(Yes|No)\s*</answer>", re.I)
_ANSWER_LETTER = re.compile(r"(?:answer is|Answer)\s*[:：]?\s*([ABab])\b")


def _gpt_text(ex) -> str:
    for turn in ex.get("conversations", []):
        if turn.get("from") == "gpt":
            return turn.get("value", "")
    return ""


def _human_text(ex) -> str:
    for turn in ex.get("conversations", []):
        if turn.get("from") == "human":
            return turn.get("value", "")
    return ""


def _chosen_letter(answer: str):
    text = answer.strip()
    if re.fullmatch(r"[ABab]", text):
        return text.upper()
    found = list(_ANSWER_LETTER.finditer(text))
    if found:
        return found[-1].group(1).upper()
    leading = re.match(r"^\s*([ABab])\b", text)
    if leading:
        return leading.group(1).upper()
    return None


def decision_label(ex):
    """Return 'yes', 'no', or None when the sample is not a Yes/No decision."""
    human = _human_text(ex)
    answer = _gpt_text(ex)

    tag = _ANSWER_TAG.search(answer)
    if tag:
        return tag.group(1).lower()

    overall = _OVERALL.search(answer)
    if overall:
        return overall.group(1).lower()

    opt_a = _OPT_A.search(human)
    opt_b = _OPT_B.search(human)
    if opt_a and opt_b:
        letter = _chosen_letter(answer)
        if letter is None:
            return None
        return {"A": opt_a.group(1), "B": opt_b.group(1)}[letter].lower()

    return None


def extract_cotm_no(cotm):
    return [ex for ex in cotm if decision_label(ex) == "no"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cotm",
        default="/home/yz/code/Triad-main/movable/CoTM/CoTM_v6.json",
    )
    ap.add_argument(
        "--iad",
        default="/home/yz/code/Triad-main/movable/InstructIAD/InstructIAD.json",
    )
    ap.add_argument(
        "--out-cotm-no",
        default="/home/yz/code/Triad-main/movable/CoTM/CoTM_v6_no.json",
    )
    ap.add_argument(
        "--out-iad-normal",
        default="/home/yz/code/Triad-main/movable/InstructIAD/InstructIAD_normal.json",
    )
    args = ap.parse_args()

    with open(args.cotm) as f:
        cotm = json.load(f)
    with open(args.iad) as f:
        iad = json.load(f)

    cotm_no = extract_cotm_no(cotm)
    iad_normal = [ex for ex in iad if ex.get("is_anomaly") in (0, False)]
    n_yes = sum(1 for ex in cotm if decision_label(ex) == "yes")
    n_skip = sum(1 for ex in cotm if decision_label(ex) is None)

    Path(args.out_cotm_no).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_iad_normal).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_cotm_no, "w") as f:
        json.dump(cotm_no, f, ensure_ascii=False)
    with open(args.out_iad_normal, "w") as f:
        json.dump(iad_normal, f, ensure_ascii=False)

    print(
        f"CoTM: {len(cotm)} (yes={n_yes}, no={len(cotm_no)}, non_decision={n_skip}) "
        f"-> {args.out_cotm_no}"
    )
    print(f"InstructIAD_normal: {len(iad_normal)} -> {args.out_iad_normal}")


if __name__ == "__main__":
    main()
