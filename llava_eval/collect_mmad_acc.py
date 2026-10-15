#!/usr/bin/env python3
"""Collect MMAD multiple-choice accuracy from Triad evaluate.py answer JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from mmad_utils import normalize_question_type, parse_mmad_answer  # noqa: E402


def load_answers(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_mmad_acc(datas: list) -> dict:
    stats = {
        "overall": {"total": 0, "correct": 0, "unparseable": 0},
        "by_question_type": {},
        "by_subset": {},
    }
    details = []

    for data in datas:
        conv = data.get("conversations", [])
        if not conv:
            continue
        raw = conv[-1].get("value", "")
        gt = data.get("gt")
        options = data.get("options", {})
        qtype = normalize_question_type(data.get("question_type", "unknown"))
        subset = data.get("subset", "unknown")
        image = data.get("image") or data.get("origin_path", "")

        pred = parse_mmad_answer(raw, options)
        correct = pred is not None and pred == gt

        stats["overall"]["total"] += 1
        if pred is None:
            stats["overall"]["unparseable"] += 1
        elif correct:
            stats["overall"]["correct"] += 1

        for bucket, key in (("by_question_type", qtype), ("by_subset", subset)):
            if key not in stats[bucket]:
                stats[bucket][key] = {"total": 0, "correct": 0, "unparseable": 0}
            stats[bucket][key]["total"] += 1
            if pred is None:
                stats[bucket][key]["unparseable"] += 1
            elif correct:
                stats[bucket][key]["correct"] += 1

        details.append(
            {
                "id": data.get("id"),
                "image": image,
                "subset": subset,
                "question_type": qtype,
                "gt": gt,
                "pred": pred,
                "correct": correct,
                "raw_output": raw[:500],
            }
        )

    def _acc(entry):
        t = entry["total"]
        if t == 0:
            return 0.0
        return entry["correct"] / t

    summary = {
        "overall_acc": _acc(stats["overall"]),
        "overall_total": stats["overall"]["total"],
        "overall_correct": stats["overall"]["correct"],
        "unparseable": stats["overall"]["unparseable"],
        "unparseable_rate": (
            stats["overall"]["unparseable"] / stats["overall"]["total"]
            if stats["overall"]["total"]
            else 0.0
        ),
        "by_question_type": {
            k: {
                "acc": _acc(v),
                "total": v["total"],
                "correct": v["correct"],
                "unparseable": v["unparseable"],
            }
            for k, v in sorted(stats["by_question_type"].items())
        },
        "by_subset": {
            k: {
                "acc": _acc(v),
                "total": v["total"],
                "correct": v["correct"],
                "unparseable": v["unparseable"],
            }
            for k, v in sorted(stats["by_subset"].items())
        },
        "macro_subset_acc": 0.0,
    }

    subset_accs = [v["acc"] for v in summary["by_subset"].values() if v["total"] > 0]
    if subset_accs:
        summary["macro_subset_acc"] = sum(subset_accs) / len(subset_accs)

    return summary, details


def print_summary(summary: dict) -> None:
    print(f"Overall Acc: {summary['overall_acc']:.4f} "
          f"({summary['overall_correct']}/{summary['overall_total']})")
    print(f"Unparseable: {summary['unparseable']} ({summary['unparseable_rate']:.2%})")
    print(f"Macro subset Acc: {summary['macro_subset_acc']:.4f}")
    print("\n=== By question type ===")
    for k, v in summary["by_question_type"].items():
        print(f"  {k}: {v['acc']:.4f} ({v['correct']}/{v['total']}, unparseable={v['unparseable']})")
    print("\n=== By MVTec subset ===")
    for k, v in summary["by_subset"].items():
        print(f"  {k}: {v['acc']:.4f} ({v['correct']}/{v['total']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="MMAD MVTec-AD multiple-choice accuracy collector.")
    parser.add_argument("answer_json", type=str, help="Path to answer JSON from evaluate.py")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write summary JSON (default: <answer_json>.summary.json)",
    )
    parser.add_argument(
        "--details",
        type=str,
        default=None,
        help="Write per-sample details JSONL (default: <answer_json>.details.json)",
    )
    args = parser.parse_args()

    answer_path = os.path.abspath(args.answer_json)
    datas = load_answers(answer_path)
    summary, details = collect_mmad_acc(datas)

    out_path = args.out or (answer_path + ".summary.json")
    details_path = args.details or (answer_path.replace(".json", "") + ".details.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\nWrote summary -> {out_path}")
    print(f"Wrote details -> {details_path}")


if __name__ == "__main__":
    main()
