# -*- coding: utf-8 -*-
"""CLI for TraceMAD evaluation (does not modify existing eval scripts).

Stage A (default, offline rules):
  python -m llava_eval.tracemad.run_tracemad \\
    --ans evaluation/mvtec/results/.../ans_..._cotm_v3.json \\
    --out_dir evaluation/mvtec/results/.../tracemad_out

Stage B (optional LLM judge, OpenAI-compatible HTTP API):
  export OPENAI_API_KEY=sk-...
  # optional: export OPENAI_BASE_URL=https://api.openai.com/v1
  # optional: export TRACEMAD_JUDGE_MODEL=gpt-4o-mini
  python -m llava_eval.tracemad.run_tracemad ... --stage_b --stage_b_budget 100

Mask / image roots are optional. If --mask_root points to a folder containing
relative paths from each sample's \"mask\" field, Stage A adds location-mask scores.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import csv
import json
import os
import sys

# Allow `python llava_eval/tracemad/run_tracemad.py` from repo root
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from llava_eval.tracemad.metrics_stage_a import (  # noqa: E402
    aggregate_stage_a,
    evaluate_sample_stage_a,
)
from llava_eval.tracemad.sample_stage_b import select_stage_b_ids  # noqa: E402


def _load_ans(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("expected a JSON list of samples")
    return data


def _try_load_mask(mask_root, mask_rel):
    if not mask_root or not mask_rel:
        return None
    path = mask_rel
    if not os.path.isabs(path):
        path = os.path.join(mask_root, mask_rel)
    if not os.path.isfile(path):
        # also try basename under mask_root
        alt = os.path.join(mask_root, os.path.basename(mask_rel))
        if os.path.isfile(alt):
            path = alt
        else:
            return None
    try:
        from PIL import Image
        import numpy as np

        arr = np.array(Image.open(path))
        return arr
    except Exception:
        return None


def _write_csv(path, rows):
    if not rows:
        return
    # flatten without nested parsed
    keys = [
        "id",
        "subset",
        "gt",
        "defect_folder",
        "pred",
        "decision_correct",
        "schema_score",
        "logic_proxy",
        "think_polarity",
        "think_answer_consistent",
        "evidence_explicit",
        "process_anchored",
        "field_entailed",
        "completeness",
        "type_match",
        "location_mask_match",
        "grounding",
        "overall",
        "think_n_words",
        "has_preamble",
        "location",
        "type",
        "schema_issues",
        "failure_modes",
    ]
    with open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = dict(r)
            row["schema_issues"] = "|".join(r.get("schema_issues") or [])
            row["failure_modes"] = "|".join(r.get("failure_modes") or [])
            w.writerow({k: row.get(k) for k in keys})


def _print_summary(summary):
    o = summary["overall"]
    print("========== TraceMAD Stage A ==========")
    print(
        "n=%s acc=%.4f prec=%.4f rec=%.4f tnr=%.4f fnr=%.4f undecided=%s"
        % (
            o["n"],
            o["acc"] or -1,
            o["precision"],
            o["recall"],
            o["tnr"],
            o["fnr"],
            o["undecided"],
        )
    )
    print(
        "schema=%.4f logic_proxy=%.4f completeness=%.4f grounding=%s overall=%.4f preamble_rate=%.4f"
        % (
            o["schema_mean"],
            o["logic_proxy_mean"],
            o["completeness_mean"],
            ("%.4f" % o["grounding_mean"]) if o["grounding_mean"] is not None else "NA",
            o["overall_mean"],
            o["preamble_rate"],
        )
    )
    print("failure_modes:", o["failure_mode_counts"])
    print("---------- per subset (acc / overall / schema) ----------")
    for sub, s in summary["per_subset"].items():
        print(
            "  %-14s n=%4d acc=%s overall=%.3f schema=%.3f"
            % (
                sub,
                s["n"],
                ("%.3f" % s["acc"]) if s["acc"] is not None else "NA",
                s["overall_mean"],
                s["schema_mean"],
            )
        )


def main(argv=None):
    p = argparse.ArgumentParser(description="TraceMAD long-form CoT evaluation")
    p.add_argument("--ans", required=True, help="path to ans_*.json (list)")
    p.add_argument(
        "--out_dir",
        default=None,
        help="output directory (default: <ans_dir>/tracemad_out)",
    )
    p.add_argument(
        "--mask_root",
        default=None,
        help="directory containing sample['mask'] relative paths (optional)",
    )
    p.add_argument(
        "--image_root",
        default=None,
        help="optional image root (reserved for Stage B multimodal; unused in text judge)",
    )
    p.add_argument("--stage_b", action="store_true", help="enable LLM judge stage")
    p.add_argument("--stage_b_budget", type=int, default=100)
    p.add_argument("--stage_b_seed", type=int, default=0)
    p.add_argument("--stage_b_correct_frac", type=float, default=0.2)
    p.add_argument("--api_key", default=None)
    p.add_argument("--base_url", default=None)
    p.add_argument("--judge_model", default=None)
    p.add_argument("--sleep", type=float, default=0.0, help="sleep between Stage B calls")
    p.add_argument(
        "--export_stage_b_prompts",
        action="store_true",
        help="only export Stage-B prompt JSONL (no API calls)",
    )
    args = p.parse_args(argv)

    ans_path = os.path.abspath(args.ans)
    samples = _load_ans(ans_path)
    out_dir = args.out_dir or os.path.join(os.path.dirname(ans_path), "tracemad_out")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # resolve default mask_root next to ans if musc/ exists
    mask_root = args.mask_root
    if mask_root is None:
        cand = os.path.join(os.path.dirname(ans_path), "musc")
        if os.path.isdir(cand):
            mask_root = os.path.dirname(ans_path)

    rows = []
    for i, sample in enumerate(samples):
        mask_arr = _try_load_mask(mask_root, sample.get("mask"))
        row = evaluate_sample_stage_a(sample, mask_arr=mask_arr)
        # drop bulky parsed from saved row copy later
        rows.append(row)
        if (i + 1) % 500 == 0:
            print("[TraceMAD StageA] %d/%d" % (i + 1, len(samples)))

    summary = aggregate_stage_a(rows)
    _print_summary(summary)

    # persist
    slim_rows = []
    for r in rows:
        rr = dict(r)
        rr.pop("parsed", None)
        slim_rows.append(rr)

    with open(os.path.join(out_dir, "stage_a_per_sample.json"), "w") as f:
        json.dump(slim_rows, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "stage_a_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _write_csv(os.path.join(out_dir, "stage_a_per_sample.csv"), slim_rows)

    # Stage B selection always exported for convenience
    ids = select_stage_b_ids(
        rows,
        budget=args.stage_b_budget,
        seed=args.stage_b_seed,
        correct_frac=args.stage_b_correct_frac,
    )
    with open(os.path.join(out_dir, "stage_b_selected_ids.json"), "w") as f:
        json.dump(ids, f, indent=2)

    samples_by_id = {s.get("id"): s for s in samples}
    row_by_id = {r["id"]: r for r in rows}

    if args.export_stage_b_prompts or args.stage_b:
        from llava_eval.tracemad.judge_prompts import build_judge_messages

        prompt_path = os.path.join(out_dir, "stage_b_prompts.jsonl")
        with open(prompt_path, "w") as f:
            for sid in ids:
                sample = samples_by_id[sid]
                row = row_by_id[sid]
                msgs = build_judge_messages(sample, row)
                f.write(
                    json.dumps({"id": sid, "messages": msgs}, ensure_ascii=False) + "\n"
                )
        print("[TraceMAD] wrote Stage-B prompts:", prompt_path)

    if args.stage_b:
        from llava_eval.tracemad.stage_b_openai import (
            OpenAICompatibleJudge,
            aggregate_stage_b,
            run_stage_b,
        )

        judge = OpenAICompatibleJudge(
            api_key=args.api_key,
            base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
            model=args.judge_model or os.environ.get("TRACEMAD_JUDGE_MODEL"),
        )
        b_rows = run_stage_b(
            samples_by_id, rows, ids, judge, sleep_s=args.sleep
        )
        b_sum = aggregate_stage_b(b_rows)
        with open(os.path.join(out_dir, "stage_b_per_sample.json"), "w") as f:
            json.dump(b_rows, f, indent=2, ensure_ascii=False)
        with open(os.path.join(out_dir, "stage_b_summary.json"), "w") as f:
            json.dump(b_sum, f, indent=2, ensure_ascii=False)
        print("========== TraceMAD Stage B ==========")
        print(json.dumps(b_sum, indent=2))
    else:
        print(
            "[TraceMAD] Stage B skipped (no LLM call). "
            "Re-run with --stage_b after setting OPENAI_API_KEY, "
            "or --export_stage_b_prompts to dump prompts only. "
            "Selected %d ids -> %s"
            % (len(ids), os.path.join(out_dir, "stage_b_selected_ids.json"))
        )

    print("[TraceMAD] outputs in", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
