#!/usr/bin/env python3
"""Decision-probe + confidence-distribution diagnostic (single self-contained file).

Two analyses in ONE run, to settle the "how to mine hard examples?" question with data:

  (1) DISTRIBUTION / CAPTURE  (needs only P(Yes)=p_A per sample)
      - For normals: distribution of p_A for FP (pred=Yes) vs TN (pred=No).
        Tests the "Yes-bias makes FPs *confidently* wrong" hypothesis.
      - Capture table: for the TNR-relevant errors (FPs), compare two mining schemes
        at matched budget k:
          S1 "bottom-confidence"  (current approach): lowest max(p_A,p_B) over ALL samples
          S2 "error-stream Top-K": highest p_A among normals only
        Report, for each budget fraction, FP-recall / FP-precision / error-recall.
        If S2 >> S1 on FP, the confidence sieve is (empirically) missing your FPs.

  (2) FROZEN-FEATURE PROBE  (needs hidden states; one forward per sample)
      - Linear probe (standardized LogisticRegression, cross-validated) on the
        decision-token last-layer hidden state (and optionally mean-pooled visual
        tokens), per class, predicting normal/defect.
      - Compare probe AUC/acc vs the LM's own p_A AUC/acc on the SAME samples.
          probe >> LM : the info is in the representation but the LM decision/prior
                        squashes it  -> a decision-head / calibration problem.
          probe ~= LM (both weak) : the representation itself lacks the info
                        -> you need new signal (AD score, etc.).

Nothing in the existing repo is modified. Reuses helpers from evaluate.py /
causal_probe_common.py / v0_softmax_confidence_probe.py.

Examples
--------
Smoke (16 samples, both analyses):
  conda activate triad
  cd /home/yz/code/Triad-main
  CUDA_VISIBLE_DEVICES=0 python llava_eval/decision_probe.py \
    --model-path checkpoints/merged_llava_qwen_triad_9ROI_cotmv6_ecprloss \
    --num-samples 16 --text_src config:v0 \
    --out LLM_judge_output/decision_probe_smoke.json

Full MVTec (all samples, both analyses):
  CUDA_VISIBLE_DEVICES=0 python llava_eval/decision_probe.py \
    --model-path checkpoints/merged_llava_qwen_triad_9ROI_cotmv6_ecprloss \
    --text_src config:v0 \
    --out LLM_judge_output/decision_probe_full.json

Distribution only, reuse an existing v0_softmax JSON (no GPU needed):
  python llava_eval/decision_probe.py \
    --ans LLM_judge_output/v0_softmax_conf_ecprloss.json \
    --out LLM_judge_output/decision_probe_from_ans.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Import path setup (mirror the other probes)
# ---------------------------------------------------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, ".."))
for _p in (_THIS, _REPO, os.path.join(_REPO, "LLaVA-NeXT")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llava.constants import (  # noqa: E402
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402

from causal_probe_common import (  # noqa: E402
    infer_subset,
    load_prompt_config,
    load_worker,
    resolve_prompt,
    stratified_pick,
)
from evaluate import data_read  # noqa: E402
from v0_softmax_confidence_probe import (  # noqa: E402
    build_chat_prompt,
    confusion_cell,
    load_sample_image,
    load_sample_mask,
    resolve_ab_token_ids,
    score_ab_first_token,
    two_way_softmax,
)

CELL_ORDER = ("TP", "FP", "TN", "FN", "UN")


# ---------------------------------------------------------------------------
# Small stats helpers
# ---------------------------------------------------------------------------
def _qstats(xs: Sequence[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=0)),
        "q10": float(np.quantile(a, 0.10)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
        "q90": float(np.quantile(a, 0.90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _metrics(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, tn, fn, un = (counts["TP"], counts["FP"], counts["TN"], counts["FN"], counts["UN"])
    eps = 1e-15
    decided = tp + fp + tn + fn
    return {
        "acc": (tp + tn) / (decided + un + eps),
        "pre": tp / (tp + fp + eps),
        "rec": tp / (tp + fn + eps),
        "tnr": tn / (tn + fp + eps),
        "yes_rate": (tp + fp) / (decided + eps),
    }


# ---------------------------------------------------------------------------
# Model: one forward -> P(Yes) + hidden features (decision token / visual pool)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def score_with_hidden(
    worker,
    prompt: str,
    images: List[Image.Image],
    masks: Optional[List[Any]],
    boxes_list: Optional[List[Any]],
    ab_ids: Dict[str, List[int]],
    feature_mode: str = "dec",
) -> Dict[str, Any]:
    """Single forward with output_hidden_states: returns p_A plus feature vectors.

    Falls back to the proven generate-based scorer (p_A only) if the manual
    forward raises for any reason.
    """
    sm = worker.model
    tokenizer, model = sm.tokenizer, sm.model
    try:
        image_sizes = [im.size for im in images]
        images_t, roi_bboxes = process_images(
            images, sm.image_processor, model.config,
            masks=masks, boxes_list=boxes_list, return_bboxes=True,
        )
        if isinstance(images_t, list):
            images_t = [im.to(model.device, dtype=model.dtype) for im in images_t if not isinstance(im, dict)]
        else:
            images_t = images_t.to(model.device, dtype=model.dtype)

        replace_token = DEFAULT_IMAGE_TOKEN
        if getattr(model.config, "mm_use_im_start_end", False):
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        prompt_use = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        input_ids = tokenizer_image_token(
            prompt_use, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(sm.device)

        (_, position_ids, attention_mask, _, inputs_embeds, _) = \
            model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None, images_t,
                modalities=["image"], image_sizes=image_sizes, roi_bboxes=roi_bboxes,
            )
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits[0, -1].float().cpu()
        z_a = float(max(logits[i].item() for i in ab_ids["A"]))
        z_b = float(max(logits[i].item() for i in ab_ids["B"]))
        p_a, p_b = two_way_softmax(z_a, z_b)
        pred = 1 if p_a >= p_b else 0

        feats: Dict[str, np.ndarray] = {}
        if feature_mode in ("dec", "vis", "dec+vis", "both"):
            hs = out.hidden_states[-1]  # [1, L, H]
            if feature_mode in ("dec", "dec+vis", "both"):
                feats["dec"] = hs[0, -1].float().cpu().numpy().astype(np.float32)
            if feature_mode in ("vis", "dec+vis", "both"):
                n_text = int(input_ids.shape[1]) - 1
                n_visual = int(inputs_embeds.shape[1]) - n_text
                try:
                    vis_start = input_ids[0].detach().cpu().tolist().index(IMAGE_TOKEN_INDEX)
                except ValueError:
                    vis_start = 0
                vis_end = max(vis_start, min(vis_start + n_visual, hs.shape[1]))
                if vis_end > vis_start:
                    feats["vis"] = hs[0, vis_start:vis_end].mean(0).float().cpu().numpy().astype(np.float32)
                else:
                    feats["vis"] = hs[0, -1].float().cpu().numpy().astype(np.float32)
        del out
        return {
            "pred": pred,
            "p_A": p_a,
            "p_B": p_b,
            "z_A": z_a,
            "z_B": z_b,
            "margin": z_a - z_b,
            "feats": feats,
        }
    except Exception as e:
        print(f"[decision-probe] hidden-forward failed ({repr(e)}); falling back to generate-only p_A")
        scored = score_ab_first_token(
            worker, prompt, images, masks=masks, boxes_list=boxes_list, ab_ids=ab_ids,
        )
        scored["feats"] = {}
        return scored


# ---------------------------------------------------------------------------
# (1) Distribution + capture analysis
# ---------------------------------------------------------------------------
def distribution_analysis(rows: List[dict]) -> Dict[str, Any]:
    def pack(rs: List[dict]) -> Dict[str, Any]:
        cells: Dict[str, List[dict]] = {c: [] for c in CELL_ORDER}
        for r in rs:
            cells[r["cell"]].append(r)
        counts = {c: len(cells[c]) for c in CELL_ORDER}
        return {
            "n": len(rs),
            "counts": counts,
            "metrics": _metrics(counts),
            "pA_by_cell": {
                c: _qstats([r["p_A"] for r in cells[c] if r.get("p_A") is not None])
                for c in CELL_ORDER
            },
            "conf_by_cell": {
                c: _qstats([r["confidence"] for r in cells[c] if r.get("confidence") is not None])
                for c in CELL_ORDER
            },
        }

    by_sub: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_sub[r["subset"]].append(r)
    return {
        "all": pack(rows),
        "per_subset": {s: pack(v) for s, v in sorted(by_sub.items())},
    }


def capture_analysis(rows: List[dict], fracs: Sequence[float]) -> Dict[str, Any]:
    """Compare bottom-confidence (S1) vs error-stream Top-K-on-normals (S2) for FP capture."""

    def one(rs: List[dict]) -> Dict[str, Any]:
        n = len(rs)
        FPs = [r for r in rs if r["gt"] == 0 and r["pred"] == 1]
        FNs = [r for r in rs if r["gt"] == 1 and r["pred"] == 0]
        errors = [r for r in rs if r["pred"] != r["gt"]]
        normals = [r for r in rs if r["gt"] == 0]
        nFP, nerr, nN = len(FPs), len(errors), len(normals)

        by_frac: Dict[str, Any] = {}
        for f in fracs:
            k = max(1, min(int(math.ceil(f * n)), n)) if n else 1

            # S1: lowest confidence over ALL samples
            s1 = sorted(rs, key=lambda r: (r["confidence"] if r.get("confidence") is not None else 1.0))[:k]
            fp1 = sum(1 for r in s1 if r["gt"] == 0 and r["pred"] == 1)
            err1 = sum(1 for r in s1 if r["pred"] != r["gt"])
            comp = {c: 0 for c in CELL_ORDER}
            for r in s1:
                comp[r["cell"]] += 1

            # S2: highest p_A among normals only, same budget k (capped to #normals)
            k2 = min(k, nN)
            s2 = sorted(normals, key=lambda r: -(r["p_A"] if r.get("p_A") is not None else -1.0))[:k2]
            fp2 = sum(1 for r in s2 if r["pred"] == 1)

            by_frac[f"{f:.2f}"] = {
                "budget_frac": f,
                "S1_bottom_conf": {
                    "k": k,
                    "fp_caught": fp1,
                    "fp_recall": (fp1 / nFP) if nFP else None,
                    "fp_precision": (fp1 / k) if k else None,
                    "error_recall": (err1 / nerr) if nerr else None,
                    "bucket_cell_comp": comp,
                },
                "S2_normal_top_pA": {
                    "k": k2,
                    "fp_caught": fp2,
                    "fp_recall": (fp2 / nFP) if nFP else None,
                    "fp_precision": (fp2 / k2) if k2 else None,
                },
            }
        return {
            "n": n,
            "n_FP": nFP,
            "n_FN": len(FNs),
            "n_errors": nerr,
            "n_normals": nN,
            "by_frac": by_frac,
        }

    by_sub: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_sub[r["subset"]].append(r)
    return {
        "all": one(rows),
        "per_subset": {s: one(v) for s, v in sorted(by_sub.items())},
    }


# ---------------------------------------------------------------------------
# (2) Frozen-feature linear probe vs LM
# ---------------------------------------------------------------------------
def probe_analysis(
    feat_rows: List[dict],
    feature_name: str,
    C: float = 0.1,
    cv: int = 5,
) -> Dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:  # pragma: no cover
        return {"feature": feature_name, "enabled": False, "error": repr(e)}

    def one(rs: List[dict]) -> Dict[str, Any]:
        y = np.array([int(r["gt"]) for r in rs], dtype=int)
        pa = np.array([float(r["p_A"]) for r in rs], dtype=np.float64)
        n = len(y)
        if n < 8 or len(set(y.tolist())) < 2:
            return {"skipped": "too_few_or_single_class", "n": n}
        nmin = int(min(np.bincount(y)))
        k = int(min(cv, nmin))
        if k < 2:
            return {"skipped": "too_few_per_class", "n": n}

        X = np.stack([r["X"] for r in rs], axis=0)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        try:
            proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
        except Exception as e:  # pragma: no cover
            return {"skipped": f"probe_failed:{e}", "n": n}

        probe_pred = (proba >= 0.5).astype(int)
        try:
            probe_auc = float(roc_auc_score(y, proba))
        except Exception:
            probe_auc = None
        probe_acc = float(accuracy_score(y, probe_pred))

        lm_pred = (pa >= 0.5).astype(int)
        try:
            lm_auc = float(roc_auc_score(y, pa))
        except Exception:
            lm_auc = None
        lm_acc = float(accuracy_score(y, lm_pred))

        gap = (probe_auc - lm_auc) if (probe_auc is not None and lm_auc is not None) else None
        return {
            "n": int(n),
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
            "probe_auc": probe_auc,
            "probe_acc": probe_acc,
            "lm_auc": lm_auc,
            "lm_acc": lm_acc,
            "auc_gap_probe_minus_lm": gap,
            "cv_folds": int(k),
            "C": float(C),
        }

    by_sub: Dict[str, List[dict]] = defaultdict(list)
    for r in feat_rows:
        by_sub[r["subset"]].append(r)
    return {
        "feature": feature_name,
        "enabled": True,
        "per_subset": {s: one(v) for s, v in sorted(by_sub.items())},
        "all": one(feat_rows),
    }


# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------
def plot_distribution(rows: List[dict], out_png: str) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[decision-probe] matplotlib unavailable, skip plot: {e}")
        return None
    groups = {"TP": "#2ca02c", "FP": "#d62728", "TN": "#1f77b4", "FN": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(9, 5))
    any_data = False
    for c, color in groups.items():
        vals = [r["p_A"] for r in rows if r["cell"] == c and r.get("p_A") is not None]
        if not vals:
            continue
        any_data = True
        ax.hist(vals, bins=20, range=(0, 1), alpha=0.55, label=f"{c} (n={len(vals)})", color=color)
    if not any_data:
        plt.close(fig)
        return None
    ax.axvline(0.5, color="black", lw=1, ls="--")
    ax.set_xlabel("P(Yes) = p_A  (decision threshold at 0.5)")
    ax.set_ylabel("count")
    ax.set_title("P(Yes) distribution by confusion cell (all samples)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Decision probe + confidence-distribution diagnostic")
    ap.add_argument("--model-path", type=str, default=None,
                    help="required unless --ans is given")
    ap.add_argument("--model-base", type=str, default=None)
    ap.add_argument("--use-flash-attn", action="store_true")
    ap.add_argument("--ds", type=str, default="mvtec")
    ap.add_argument("--data_path", type=str, default="./evaluation/mvtec")
    ap.add_argument("--from", dest="question_file", type=str, default="question_musc.jsonl")
    ap.add_argument("--text_src", type=str, default="config:v0")
    ap.add_argument("--subset", type=str, default="all", help="all or comma-separated class names")
    ap.add_argument("--num-samples", type=int, default=-1, help="-1 = all; else stratified subset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature", type=str, default="dec",
                    choices=["dec", "vis", "dec+vis", "none"],
                    help="frozen feature to probe: decision-token hidden (dec), "
                         "mean-pooled visual tokens (vis), both, or none (dist only)")
    ap.add_argument("--probe-C", type=float, default=0.1, help="LogisticRegression C (inverse reg)")
    ap.add_argument("--probe-cv", type=int, default=5, help="max CV folds")
    ap.add_argument("--bottom-fracs", type=str, default="0.05,0.10,0.20,0.30")
    ap.add_argument("--ans", type=str, default=None,
                    help="optional: reuse a v0_softmax JSON (rows+summary) to skip inference "
                         "(distribution/capture only; probe needs --model-path)")
    ap.add_argument("--out", type=str, required=True, help="JSON output path")
    ap.add_argument("--csv-prefix", type=str, default=None,
                    help="CSV prefix (default: out path without extension)")
    ap.add_argument("--no-rows", action="store_true", help="omit per-sample rows from JSON")
    ap.add_argument("--save-feats", action="store_true", help="save features to <out>.npz")
    args = ap.parse_args()

    fracs = [float(x) for x in args.bottom_fracs.split(",") if x.strip()]

    # ---- Build rows: either from an existing ans JSON, or by running the model ----
    rows: List[dict] = []
    feat_store: Dict[str, List[dict]] = defaultdict(list)
    meta: Dict[str, Any] = {
        "ds": args.ds,
        "text_src": args.text_src,
        "subset": args.subset,
        "feature": args.feature,
        "bottom_fracs": fracs,
        "source": "ans" if args.ans else "model",
    }

    if args.ans:
        if args.feature not in ("none",):
            print("[decision-probe] WARNING: --ans given; probe needs model forward, "
                  "so only distribution/capture will be computed (feature ignored).")
        with open(args.ans, "r", encoding="utf-8") as f:
            payload = json.load(f)
        ans_rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(ans_rows, list):
            raise ValueError("--ans JSON has no 'rows' list")
        for r in ans_rows:
            p_a = r.get("p_A")
            p_b = r.get("p_B")
            pred = r.get("pred")
            if pred is None and p_a is not None:
                pred = 1 if float(p_a) >= 0.5 else 0
            gt = int(r.get("gt", 0))
            subset = str(r.get("subset") or infer_subset(r))
            cell = r.get("cell") or confusion_cell(gt, pred)
            conf = max(float(p_a), float(p_b)) if (p_a is not None and p_b is not None) else (
                max(float(p_a), 1.0 - float(p_a)) if p_a is not None else None)
            rows.append({
                "id": r.get("id"), "subset": subset, "gt": gt, "pred": pred,
                "cell": cell, "p_A": (float(p_a) if p_a is not None else None),
                "p_B": (float(p_b) if p_b is not None else None),
                "margin": r.get("margin"), "confidence": conf,
            })
        meta["ans"] = args.ans
        # keep only decided rows for analysis
        rows = [r for r in rows if r["pred"] is not None and r["p_A"] is not None]
    else:
        if not args.model_path:
            raise SystemExit("--model-path is required unless --ans is provided")

        data_dir = args.data_path
        if data_dir.endswith("/") and not data_dir.rstrip("/").endswith(args.ds):
            data_dir = os.path.join(data_dir, args.ds)
        question_path = os.path.join(data_dir, args.question_file)

        qs_tmp, prompt_ver = load_prompt_config(args.ds, args.text_src)
        all_datas = data_read(question_path)
        if args.subset != "all":
            want = set(args.subset.split(","))
            all_datas = [d for d in all_datas if infer_subset(d) in want]
        if args.num_samples and args.num_samples > 0:
            all_datas = stratified_pick(all_datas, args.num_samples, seed=args.seed, defect_only=False)

        meta.update({"model_path": args.model_path, "prompt_ver": prompt_ver,
                     "question_file": question_path, "n_requested": len(all_datas)})
        print(f"samples={len(all_datas)}  prompt={prompt_ver}  questions={question_path}")

        worker = load_worker(args.model_path, args.model_base, args.use_flash_attn)
        ab_ids = resolve_ab_token_ids(worker.model.tokenizer)
        print(f"A/B token ids: {ab_ids}")
        meta["ab_token_ids"] = ab_ids

        want_feats = args.feature not in ("none",)

        try:
            from tqdm import tqdm
        except Exception:  # pragma: no cover
            def tqdm(x, **kw):
                return x

        for sample in tqdm(all_datas, desc="decision-probe"):
            subset = infer_subset(sample)
            gt = int(sample.get("gt", 0))
            prompt = resolve_prompt(qs_tmp, subset)
            img = load_sample_image(data_dir, sample)
            musc = load_sample_mask(data_dir, sample)
            masks = [musc] if musc is not None else None
            chat_prompt = build_chat_prompt(worker, prompt, image_num=1)

            if want_feats:
                scored = score_with_hidden(worker, chat_prompt, [img], masks, None, ab_ids, args.feature)
            else:
                scored = score_ab_first_token(worker, chat_prompt, [img], masks=masks,
                                              boxes_list=None, ab_ids=ab_ids)
                scored["feats"] = {}

            pred = scored.get("pred")
            p_a = scored.get("p_A")
            p_b = scored.get("p_B")
            cell = confusion_cell(gt, pred)
            conf = max(p_a, p_b) if (p_a is not None and p_b is not None) else None
            row = {
                "id": sample.get("id"), "subset": subset,
                "origin_path": sample.get("origin_path"), "gt": gt,
                "pred": pred, "cell": cell, "p_A": p_a, "p_B": p_b,
                "margin": scored.get("margin"), "confidence": conf,
                "correct": (pred == gt) if pred is not None else None,
            }
            rows.append(row)
            for k, v in scored.get("feats", {}).items():
                feat_store[k].append({
                    "id": row["id"], "subset": subset, "gt": gt,
                    "p_A": p_a, "cell": cell, "X": v,
                })

    # keep decided rows only (consistent with analysis)
    rows = [r for r in rows if r["pred"] is not None and r["p_A"] is not None]

    # ---- Analyses ----
    distribution = distribution_analysis(rows)
    capture = capture_analysis(rows, fracs)

    probes: Dict[str, Any] = {}
    for name, frs in feat_store.items():
        if frs:
            probes[name] = probe_analysis(frs, name, C=args.probe_C, cv=args.probe_cv)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    png_path = os.path.splitext(out_path)[0] + "_pA_by_cell.png"
    plot_path = plot_distribution(rows, png_path)

    payload = {
        "meta": {**meta, "n_analyzed": len(rows), "plot": plot_path,
                 "definition": {
                     "p_A": "P(Yes/defect) from A/B two-way softmax at the decision token",
                     "confidence": "max(p_A,p_B) = probability of the chosen letter",
                     "S1_bottom_conf": "lowest-confidence budget over ALL samples (current mining)",
                     "S2_normal_top_pA": "highest p_A among normals only, same budget (error-stream)",
                     "probe": "StandardScaler+LogisticRegression on frozen feature, "
                              "cross-validated; compare AUC/acc to LM p_A",
                 }},
        "distribution": distribution,
        "capture": capture,
        "probe": probes,
    }
    if not args.no_rows:
        payload["rows"] = rows
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}")

    # ---- CSV: distribution ----
    csv_prefix = args.csv_prefix or os.path.splitext(out_path)[0]
    dist_csv = csv_prefix + "_distribution.csv"
    with open(dist_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subset", "n", "acc", "pre", "rec", "tnr", "yes_rate",
                    "TP", "FP", "TN", "FN",
                    "mean_pA_TP", "mean_pA_FP", "mean_pA_TN", "mean_pA_FN",
                    "med_pA_FP", "med_pA_TN", "med_conf_FP", "med_conf_TN"])
        blocks = [("all", distribution["all"])] + list(distribution["per_subset"].items())
        for name, b in blocks:
            c = b["counts"]; m = b["metrics"]
            pac = b["pA_by_cell"]; cfc = b["conf_by_cell"]

            def mp(d, key, field):
                return d[key].get(field) if d[key].get("n", 0) else None

            w.writerow([
                name, b["n"], m["acc"], m["pre"], m["rec"], m["tnr"], m["yes_rate"],
                c["TP"], c["FP"], c["TN"], c["FN"],
                mp(pac, "TP", "mean"), mp(pac, "FP", "mean"),
                mp(pac, "TN", "mean"), mp(pac, "FN", "mean"),
                mp(pac, "FP", "median"), mp(pac, "TN", "median"),
                mp(cfc, "FP", "median"), mp(cfc, "TN", "median"),
            ])
    print(f"wrote {dist_csv}")

    # ---- CSV: capture ----
    cap_csv = csv_prefix + "_capture.csv"
    with open(cap_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subset", "n", "n_FP", "n_FN", "budget_frac", "k",
                    "S1_fp_recall", "S1_fp_precision", "S1_error_recall",
                    "S2_k", "S2_fp_recall", "S2_fp_precision"])
        blocks = [("all", capture["all"])] + list(capture["per_subset"].items())
        for name, b in blocks:
            for fk, fb in b["by_frac"].items():
                s1 = fb["S1_bottom_conf"]; s2 = fb["S2_normal_top_pA"]
                w.writerow([
                    name, b["n"], b["n_FP"], b["n_FN"], fb["budget_frac"], s1["k"],
                    s1["fp_recall"], s1["fp_precision"], s1["error_recall"],
                    s2["k"], s2["fp_recall"], s2["fp_precision"],
                ])
    print(f"wrote {cap_csv}")

    # ---- CSV: probe ----
    probe_csv = None
    if probes:
        probe_csv = csv_prefix + "_probe.csv"
        with open(probe_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["feature", "subset", "n", "n_pos", "n_neg",
                        "probe_auc", "probe_acc", "lm_auc", "lm_acc",
                        "auc_gap_probe_minus_lm", "note"])
            for feat, pb in probes.items():
                if not pb.get("enabled"):
                    w.writerow([feat, "ALL", "", "", "", "", "", "", "", "", pb.get("error")])
                    continue
                blocks = [("all", pb["all"])] + list(pb["per_subset"].items())
                for name, b in blocks:
                    w.writerow([
                        feat, name, b.get("n"), b.get("n_pos"), b.get("n_neg"),
                        b.get("probe_auc"), b.get("probe_acc"),
                        b.get("lm_auc"), b.get("lm_acc"),
                        b.get("auc_gap_probe_minus_lm"), b.get("skipped", ""),
                    ])
        print(f"wrote {probe_csv}")

    # ---- optional features dump ----
    if args.save_feats and feat_store:
        npz_path = os.path.splitext(out_path)[0] + "_features.npz"
        dump = {}
        for name, frs in feat_store.items():
            dump[f"X_{name}"] = np.stack([r["X"] for r in frs], axis=0)
            dump[f"gt_{name}"] = np.array([int(r["gt"]) for r in frs])
            dump[f"subset_{name}"] = np.array([str(r["subset"]) for r in frs])
        np.savez_compressed(npz_path, **dump)
        print(f"wrote {npz_path}")

    # ---- console verdict ----
    _print_verdict(distribution, capture, probes)


def _print_verdict(distribution, capture, probes):
    print("\n================ VERDICT ================")
    a = distribution["all"]
    m = a["metrics"]; c = a["counts"]
    print(f"overall: acc={m['acc']:.4f} rec={m['rec']:.4f} tnr={m['tnr']:.4f} "
          f"yes_rate={m['yes_rate']:.4f}  counts={c}")
    pac = a["pA_by_cell"]

    def g(cell, field):
        d = pac[cell]
        return d.get(field) if d.get("n", 0) else None

    print(f"median P(Yes): FP={g('FP','median')}  TN={g('TN','median')}  "
          f"TP={g('TP','median')}  FN={g('FN','median')}")
    fp_med = g("FP", "median"); tn_med = g("TN", "median")
    if fp_med is not None and tn_med is not None:
        if fp_med >= 0.7 and fp_med > tn_med + 0.15:
            print("=> FPs are CONFIDENTLY wrong (high P(Yes)); a low-confidence sieve will MISS them.")
        else:
            print("=> FPs sit closer to the boundary; confidence sieve may still catch them.")

    print("\nFP capture, budget=10% of samples (overall):")
    fb = capture["all"]["by_frac"].get("0.10") or next(iter(capture["all"]["by_frac"].values()))
    s1 = fb["S1_bottom_conf"]; s2 = fb["S2_normal_top_pA"]
    print(f"  S1 bottom-confidence: FP_recall={_fmt(s1['fp_recall'])} FP_precision={_fmt(s1['fp_precision'])} "
          f"err_recall={_fmt(s1['error_recall'])}  bucket_cells={s1['bucket_cell_comp']}")
    print(f"  S2 normal-Top-p_A   : FP_recall={_fmt(s2['fp_recall'])} FP_precision={_fmt(s2['fp_precision'])}")
    if s1["fp_recall"] is not None and s2["fp_recall"] is not None:
        if s2["fp_recall"] > s1["fp_recall"] + 0.1:
            print("  => error-stream Top-K catches more FPs than bottom-confidence at equal budget.")
        else:
            print("  => comparable; bottom-confidence is not obviously worse for FP capture.")

    for feat, pb in probes.items():
        if not pb.get("enabled"):
            print(f"\nprobe[{feat}]: disabled ({pb.get('error')})")
            continue
        b = pb["all"]
        print(f"\nprobe[{feat}] overall: n={b.get('n')} probe_auc={_fmt(b.get('probe_auc'))} "
              f"probe_acc={_fmt(b.get('probe_acc'))} | lm_auc={_fmt(b.get('lm_auc'))} "
              f"lm_acc={_fmt(b.get('lm_acc'))} | gap={_fmt(b.get('auc_gap_probe_minus_lm'))}")
        pa_, la_ = b.get("probe_auc"), b.get("lm_auc")
        if pa_ is not None and la_ is not None:
            if pa_ - la_ >= 0.05:
                print("  => probe >> LM: info is in the representation; decision/prior squashes it "
                      "(decision-head / calibration problem).")
            elif pa_ < 0.70 and la_ < 0.70:
                print("  => both weak: representation itself lacks separability "
                      "(need new signal: AD score etc.).")
            else:
                print("  => probe ~ LM: the LM decision is roughly as good as a linear readout of features.")
    print("========================================\n")


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)


if __name__ == "__main__":
    main()
