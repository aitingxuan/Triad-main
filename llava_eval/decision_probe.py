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
# Representation-level DIRECTION analysis (evidence vs verbalizer vs bias)
# ---------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (handles ties), scipy-free."""
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort(kind="mergesort")
    sa = a[order]
    ranks = np.empty(a.size, dtype=np.float64)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return None
    r = _rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _tnr_at_recall(y: np.ndarray, s: np.ndarray, rec_target: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    """Single global Neyman-Pearson operating point: threshold so recall >= rec_target, return TNR."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    pos = s[y == 1]
    if pos.size == 0:
        return None, None
    thr = float(np.quantile(pos, 1.0 - rec_target))
    pred = (s >= thr).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tnr = (tn / (tn + fp)) if (tn + fp) > 0 else None
    return tnr, thr


def _randomized_range(R: np.ndarray, k: int, seed: int = 0, n_iter: int = 1) -> np.ndarray:
    """Orthonormal basis (N x k) approximating the range of R (N x H), for H >> k."""
    N, H = R.shape
    k = int(min(k, max(N - 1, 1), H))
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((H, k))
    Y = R @ omega
    for _ in range(max(0, n_iter)):
        Y = R @ (R.T @ Y)
    Q, _ = np.linalg.qr(Y)
    return Q[:, :k]


def _metrics_from_pred(pred: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    pred = np.asarray(pred).astype(int)
    y = np.asarray(y).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    eps = 1e-9
    return {"acc": (tp + tn) / (y.size + eps), "rec": tp / (tp + fn + eps),
            "tnr": tn / (tn + fp + eps), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _per_class_intercept(s: np.ndarray, sub: np.ndarray, y: np.ndarray, classes) -> Dict[str, float]:
    """Per-class threshold = midpoint of the class's normal/defect medians of the score."""
    b: Dict[str, float] = {}
    for c in classes:
        m = sub == c
        sn = s[m & (y == 0)]
        sd = s[m & (y == 1)]
        if sn.size and sd.size:
            b[c] = float(0.5 * (np.median(sn) + np.median(sd)))
        elif sn.size:
            b[c] = float(np.median(sn))
        elif sd.size:
            b[c] = float(np.median(sd))
    return b


def _predict_with_b(s: np.ndarray, sub: np.ndarray, b: Dict[str, float]) -> np.ndarray:
    barr = np.array([b.get(str(sc), np.inf) for sc in sub], dtype=np.float64)
    return (s > barr).astype(int)


def _cv_intercept_oof(s: np.ndarray, sub: np.ndarray, y: np.ndarray, classes, K: int, seed: int) -> np.ndarray:
    """Out-of-fold per-class-intercept predictions (K-fold within each class)."""
    rng = np.random.default_rng(seed)
    pred = np.full(s.size, -1, dtype=int)
    K = max(int(K), 2)
    for c in classes:
        idx = np.where(sub == c)[0]
        if idx.size < 2:
            continue
        fold = np.zeros(idx.size, dtype=int)
        for g in (0, 1):
            pos = np.where(y[idx] == g)[0]
            if pos.size:
                rng.shuffle(pos)
                fold[pos] = np.arange(pos.size) % K
        for f in range(K):
            te = idx[fold == f]
            tr = idx[fold != f]
            if te.size == 0 or tr.size == 0:
                continue
            sn = s[tr][y[tr] == 0]
            sd = s[tr][y[tr] == 1]
            if sn.size and sd.size:
                bt = 0.5 * (np.median(sn) + np.median(sd))
            elif sn.size:
                bt = float(np.median(sn))
            elif sd.size:
                bt = float(np.median(sd))
            else:
                continue
            pred[te] = (s[te] > bt).astype(int)
    if (pred < 0).any():
        b_all = _per_class_intercept(s, sub, y, classes)
        miss = pred < 0
        barr = np.array([b_all.get(str(sc), np.inf) for sc in sub[miss]], dtype=np.float64)
        pred[miss] = (s[miss] > barr).astype(int)
    return pred


def shared_model_analysis(feat_rows: List[dict], ridge: float = 0.05, cv: int = 5,
                          seed: int = 0, k_white: int = 128) -> Dict[str, Any]:
    """Class-invariant anomaly direction with whitening + shared-w / per-class-intercept model.

    Directions (all class-invariant, estimated from normal/defect centroids):
      ev   : mean_c(mu_def_c - mu_norm_c)                       (raw mean-diff)
      diag : ev divided by per-dim within-cell variance          (diagonal Mahalanobis)
      lda  : (Sigma_w + ridge*I)^-1 ev, via low-rank randomized whitening (Mahalanobis)
    For each direction we report in-sample AUC and leave-one-class-out zero-shot AUC.
    Then a "shared w + per-class intercept b_c" model (score = <h,w> > b_c) is evaluated
    in-sample (ceiling) and with within-class K-fold CV on b_c (honest-ish).
    """
    out: Dict[str, Any] = {"enabled": False}
    if not feat_rows:
        out["reason"] = "no dec features"
        return out
    X = np.stack([np.asarray(r["X"], dtype=np.float64) for r in feat_rows], 0)
    y = np.array([int(r["gt"]) for r in feat_rows])
    sub = np.array([str(r["subset"]) for r in feat_rows])
    classes = sorted(set(sub.tolist()))
    N, H = X.shape

    gm: Dict[Any, np.ndarray] = {}
    for c in classes:
        for g in (0, 1):
            m = (sub == c) & (y == g)
            if m.sum():
                gm[(c, g)] = X[m].mean(0)
    d_by_class = {c: gm[(c, 1)] - gm[(c, 0)] for c in classes if (c, 1) in gm and (c, 0) in gm}
    if not d_by_class:
        out["reason"] = "need both normal & defect per class"
        return out
    d_global = np.mean(list(d_by_class.values()), 0)

    gmean = np.empty_like(X)
    for i in range(N):
        gmean[i] = gm[(sub[i], int(y[i]))]
    R = X - gmean
    ss_dim = (R * R).sum(0)                     # per-dim sum of squared residuals
    denom = max(N - 1, 1)
    var_w = ss_dim / denom
    trace_C = float(ss_dim.sum() / denom)
    lam = max(ridge * trace_C / H, 1e-12)
    lam_d = max(ridge * float(var_w.mean()), 1e-12)
    ss_by_class = {c: (R[sub == c] ** 2).sum(0) for c in classes}
    n_class = {c: int((sub == c).sum()) for c in classes}

    w_ev = _unit(d_global)
    w_diag = _unit(d_global / (var_w + lam_d))

    lda_ok = True
    try:
        Q = _randomized_range(R, k_white, seed=seed)
        Bc = (Q.T @ R) / np.sqrt(denom)         # k x H;  Bc^T Bc ~ R^T R/(N-1)
        M = Bc @ Bc.T + lam * np.eye(Bc.shape[0])

        def _whiten(d):
            Bd = Bc @ d
            yv = np.linalg.solve(M, Bd)
            return (d - Bc.T @ yv) / lam
        w_lda = _unit(_whiten(d_global))
    except Exception as e:  # noqa: BLE001
        lda_ok = False
        out["lda_error"] = repr(e)
        _whiten = None
        w_lda = _unit(d_global)

    def _auc_of(w):
        return _auc(y, X @ w)

    def _zs(direction_fn):
        vals = []
        for c in classes:
            others = [cc for cc in classes if cc != c and cc in d_by_class]
            m = sub == c
            Xc = X[m]
            yc = y[m]
            if not others or int(yc.sum()) == 0 or int((1 - yc).sum()) == 0:
                continue
            d_o = np.mean([d_by_class[cc] for cc in others], 0)
            try:
                w = _unit(direction_fn(d_o, c))
            except Exception:  # noqa: BLE001
                continue
            vals.append(_auc(yc, (Xc - Xc.mean(0)) @ w))
        return float(np.mean(vals)) if vals else None

    def _dir_diag_loo(d, c):
        den = max((N - n_class[c]) - 1, 1)
        var_o = (ss_dim - ss_by_class[c]) / den
        return d / (var_o + lam_d)

    zs_ev = _zs(lambda d, c: d)
    zs_diag = _zs(_dir_diag_loo)
    zs_lda = _zs(lambda d, c: _whiten(d)) if lda_ok else None

    def _intercept_model(w):
        s = X @ w
        b = _per_class_intercept(s, sub, y, classes)
        pred_in = _predict_with_b(s, sub, b)
        pred_cv = _cv_intercept_oof(s, sub, y, classes, cv, seed)
        per = {}
        for c in classes:
            m = sub == c
            per[c] = {"b": b.get(c),
                      "in": _metrics_from_pred(pred_in[m], y[m]),
                      "cv": _metrics_from_pred(pred_cv[m], y[m])}
        return {"in_sample": _metrics_from_pred(pred_in, y),
                "cv": _metrics_from_pred(pred_cv, y),
                "per_class": per}

    intercept = {"ev": _intercept_model(w_ev), "lda": _intercept_model(w_lda)}

    out.update({
        "enabled": True, "n": int(N), "hidden_dim": int(H),
        "ridge": float(ridge), "cv": int(cv), "k_white": int(min(k_white, max(N - 1, 1), H)),
        "lda_ok": bool(lda_ok),
        "directions": {
            "ev": {"in_sample_auc": _auc_of(w_ev), "zeroshot_auc": zs_ev},
            "diag": {"in_sample_auc": _auc_of(w_diag), "zeroshot_auc": zs_diag},
            "lda": {"in_sample_auc": _auc_of(w_lda), "zeroshot_auc": zs_lda},
        },
        "intercept": intercept,
    })
    return out


def _cv_proba(X: np.ndarray, y: np.ndarray, C: float = 0.1, cv: int = 5, seed: int = 0):
    """Out-of-fold P(defect) from a standardized logistic probe; None if sklearn unavailable."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:  # noqa: BLE001
        return None
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return None
    k = int(min(cv, n1, n0))
    try:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C))
        if k < 2:
            clf.fit(X, y)
            return clf.predict_proba(X)[:, 1]
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    except Exception:  # noqa: BLE001
        return None


def axis_residual_diagnostics(feat_rows: List[dict], C: float = 0.1, cv: int = 5,
                              seed: int = 0, n_bins: int = 10) -> Dict[str, Any]:
    """D2: is there normal/defect signal OFF the shared evidence axis? (residual probe + binned AUC)

    Projects out the shared mean-diff axis w_ev, then:
      - runs a linear probe on the OFF-AXIS residual (does separation survive off-axis?),
      - bins samples by axis score t=<h,w_ev> and reports within-bin residual AUC
        (can same-axis samples still be separated by their off-axis residual?).
    """
    out: Dict[str, Any] = {"enabled": False}
    if not feat_rows:
        out["reason"] = "no dec features"
        return out
    X = np.stack([np.asarray(r["X"], dtype=np.float64) for r in feat_rows], 0)
    y = np.array([int(r["gt"]) for r in feat_rows])
    sub = np.array([str(r["subset"]) for r in feat_rows])
    classes = sorted(set(sub.tolist()))
    N, H = X.shape

    mu_norm: Dict[str, np.ndarray] = {}
    mu_def: Dict[str, np.ndarray] = {}
    for c in classes:
        m = sub == c
        xn = X[m & (y == 0)]
        xd = X[m & (y == 1)]
        if xn.shape[0]:
            mu_norm[c] = xn.mean(0)
        if xd.shape[0]:
            mu_def[c] = xd.mean(0)
    d_by_class = {c: mu_def[c] - mu_norm[c] for c in classes if c in mu_def and c in mu_norm}
    if not d_by_class:
        out["reason"] = "need both labels per class"
        return out
    w = _unit(np.mean(list(d_by_class.values()), 0))
    t = X @ w
    resid = X - np.outer(t, w)

    full_p = _cv_proba(X, y, C, cv, seed)
    res_p = _cv_proba(resid, y, C, cv, seed)
    out.update({
        "enabled": True, "n": int(N), "hidden_dim": int(H),
        "full_probe_auc": _auc(y, full_p) if full_p is not None else None,
        "residual_probe_auc": _auc(y, res_p) if res_p is not None else None,
    })

    per = []
    for c in classes:
        m = sub == c
        yc = y[m]
        if int(yc.sum()) and int((1 - yc).sum()):
            p = _cv_proba(resid[m], yc, C, cv, seed)
            per.append({"subset": c, "n": int(m.sum()),
                        "residual_probe_auc": _auc(yc, p) if p is not None else None})
    out["per_class_residual_auc"] = per

    cond = []
    if res_p is not None:
        qs = np.quantile(t, np.linspace(0, 1, n_bins + 1))
        for b in range(n_bins):
            lo, hi = float(qs[b]), float(qs[b + 1])
            m = (t >= lo) & (t <= hi) if b == n_bins - 1 else (t >= lo) & (t < hi)
            yb = y[m]
            cond.append({
                "bin": b, "t_lo": lo, "t_hi": hi, "n": int(m.sum()),
                "auc_resid_within_bin": (_auc(yb, res_p[m])
                                         if (int(yb.sum()) and int((1 - yb).sum())) else None),
            })
    out["conditional_auc_by_axis_bin"] = cond
    return out


def lowrank_gaussian_diagnostics(feat_rows: List[dict], r_list=(1, 2, 4, 8, 16, 32),
                                 ridge_cov: float = 1e-2, rho: float = 10.0,
                                 cv: int = 5, seed: int = 0) -> Dict[str, Any]:
    """D1+D3: shared-subspace rank spectrum + class-conditional low-rank Gaussian LLR.

    Build shared anomaly basis U_r from SVD of the per-class mean-diff matrix D (H x C).
    In z=U_r^T h space, fit a class-conditional Gaussian (shrinkage covariance toward the
    pooled within-cell covariance, estimated from NORMALS-heavy counts) and score with a
    Gaussian log-likelihood-ratio. Evaluated with within-class K-fold CV (honest-ish).
    A rank-r 2nd-order readout that beats rank-1 / the 1-D axis indicates the structural win.
    """
    out: Dict[str, Any] = {"enabled": False}
    if not feat_rows:
        out["reason"] = "no dec features"
        return out
    X = np.stack([np.asarray(r["X"], dtype=np.float64) for r in feat_rows], 0)
    y = np.array([int(r["gt"]) for r in feat_rows])
    sub = np.array([str(r["subset"]) for r in feat_rows])
    classes = sorted(set(sub.tolist()))
    N, H = X.shape

    gm: Dict[Any, np.ndarray] = {}
    for c in classes:
        for g in (0, 1):
            m = (sub == c) & (y == g)
            if m.sum():
                gm[(c, g)] = X[m].mean(0)
    d_by_class = {c: gm[(c, 1)] - gm[(c, 0)] for c in classes if (c, 1) in gm and (c, 0) in gm}
    if not d_by_class:
        out["reason"] = "need both labels per class"
        return out
    D = np.stack(list(d_by_class.values()), axis=1)  # H x C
    U, S, _ = np.linalg.svd(D, full_matrices=False)
    s2 = S ** 2
    energy = [float(e) for e in (s2 / max(float(s2.sum()), 1e-12))[:min(len(s2), 16)]]

    by_rank = []
    best = None
    for r in sorted(set(int(x) for x in r_list)):
        r = int(min(r, U.shape[1]))
        if r < 1:
            continue
        Ur = U[:, :r]
        z = X @ Ur

        zc = np.empty_like(z)
        for i in range(N):
            zc[i] = gm[(sub[i], int(y[i]))] @ Ur
        Rz = z - zc
        Spool = (Rz.T @ Rz) / max(N - len(gm), 1) + ridge_cov * np.eye(r)

        rng = np.random.default_rng(seed)
        oof = np.full(N, np.nan)
        for c in classes:
            idx = np.where(sub == c)[0]
            if idx.size < 4:
                continue
            fold = np.zeros(idx.size, dtype=int)
            for g in (0, 1):
                pos = np.where(y[idx] == g)[0]
                if pos.size:
                    rng.shuffle(pos)
                    fold[pos] = np.arange(pos.size) % max(int(cv), 2)
            for f in range(max(int(cv), 2)):
                te = idx[fold == f]
                tr = idx[fold != f]
                if te.size == 0:
                    continue
                mn = z[tr][y[tr] == 0]
                md = z[tr][y[tr] == 1]
                if mn.shape[0] < 2 or md.shape[0] < 2:
                    continue
                mun = mn.mean(0)
                mud = md.mean(0)
                Rn = mn - mun
                Rd = md - mud
                Sp = (Rn.T @ Rn + Rd.T @ Rd) / max((mn.shape[0] + md.shape[0] - 2), 1)
                alpha = rho / (rho + max(mn.shape[0], 1))
                Sc = (1.0 - alpha) * Sp + alpha * Spool
                inv = np.linalg.pinv(Sc)

                def _maha(a, mu, inv=inv):
                    d = a - mu
                    return np.einsum("ij,jk,ik->i", d, inv, d)

                oof[te] = 0.5 * (_maha(z[te], mun) - _maha(z[te], mud))

        mask = np.isfinite(oof)
        auc = _auc(y[mask], oof[mask]) if int(mask.sum()) else None
        tnr, _ = _tnr_at_recall(y[mask], oof[mask], 0.95) if int(mask.sum()) else (None, None)
        by_rank.append({"r": r, "oof_auc": auc, "tnr_at_recall95": tnr})
        if auc is not None and (best is None or auc > best["auc"]):
            best = {"r": r, "auc": auc, "oof": oof.copy()}

    per = []
    if best is not None:
        oof = best["oof"]
        mask = np.isfinite(oof)
        for c in classes:
            m = (sub == c) & mask
            yc = y[m]
            if int(yc.sum()) and int((1 - yc).sum()):
                per.append({"subset": c, "n": int(m.sum()), "auc": _auc(yc, oof[m])})

    out.update({
        "enabled": True, "n": int(N), "hidden_dim": int(H), "n_classes": len(d_by_class),
        "singular_energy_top": energy,
        "by_rank": by_rank,
        "best_r": (best["r"] if best else None),
        "per_class_auc_best_r": per,
    })
    return out


def direction_analysis(feat_rows: List[dict], d_verb: np.ndarray, rec_target: float = 0.95) -> Dict[str, Any]:
    """Decompose the Yes/No decision in representation space.

    Directions (all in the decision-token hidden space that ``lm_head`` reads):
      d_verb : lm_head[A] - lm_head[B]  (the model's verbalizer readout direction)
      w_ev   : mean_c( mean_defect_c - mean_normal_c )  (evidence direction, class-centered)
      w_bias : mean_c( mean_normal&FP_c - mean_normal&TN_c )  (what pushes normals to Yes)
    Reports cosines (global + per-class), variance decomposition of the verbalizer score
    into evidence vs off-evidence parts, AUC/TNR of four readouts, and a leave-one-class-out
    zero-shot transfer AUC of the shared evidence direction.
    """
    out: Dict[str, Any] = {"enabled": False}
    if not feat_rows:
        out["reason"] = "no dec features"
        return out
    d_verb = np.asarray(d_verb, dtype=np.float64)
    X = np.stack([np.asarray(r["X"], dtype=np.float64) for r in feat_rows], 0)
    if X.shape[1] != d_verb.shape[0]:
        out["reason"] = f"feature dim {X.shape[1]} != d_verb dim {d_verb.shape[0]}"
        return out
    y = np.array([int(r["gt"]) for r in feat_rows])
    sub = np.array([str(r["subset"]) for r in feat_rows])
    cell = np.array([str(r.get("cell", "")) for r in feat_rows])
    margin = np.array(
        [r.get("margin") if r.get("margin") is not None else np.nan for r in feat_rows],
        dtype=np.float64,
    )
    classes = sorted(set(sub.tolist()))

    mu_norm: Dict[str, np.ndarray] = {}
    mu_def: Dict[str, np.ndarray] = {}
    for c in classes:
        m = sub == c
        xn = X[m & (y == 0)]
        xd = X[m & (y == 1)]
        if xn.shape[0]:
            mu_norm[c] = xn.mean(0)
        if xd.shape[0]:
            mu_def[c] = xd.mean(0)

    diffs = [mu_def[c] - mu_norm[c] for c in classes if c in mu_def and c in mu_norm]
    w_ev = np.mean(diffs, 0) if diffs else np.zeros(X.shape[1])
    w_ev_hat = _unit(w_ev)

    is_fp = cell == "FP"
    is_tn = cell == "TN"
    bias_diffs = []
    for c in classes:
        m = sub == c
        xf = X[m & is_fp]
        xt = X[m & is_tn]
        if xf.shape[0] and xt.shape[0]:
            bias_diffs.append(xf.mean(0) - xt.mean(0))
    w_bias = np.mean(bias_diffs, 0) if bias_diffs else np.zeros(X.shape[1])
    w_bias_hat = _unit(w_bias)

    dvh = _unit(d_verb)
    cos = {
        "dverb_ev": float(dvh @ w_ev_hat) if np.any(w_ev) else None,
        "dverb_bias": float(dvh @ w_bias_hat) if np.any(w_bias) else None,
        "ev_bias": float(w_ev_hat @ w_bias_hat) if np.any(w_ev) and np.any(w_bias) else None,
    }

    # Sanity: does <h, d_verb> reproduce the model's stored logit margin?
    corr = None
    if int(np.isfinite(margin).sum()) > 3:
        mask = np.isfinite(margin)
        a = X[mask] @ d_verb
        if a.std() > 0 and margin[mask].std() > 0:
            corr = float(np.corrcoef(a, margin[mask])[0, 1])

    if mu_norm:
        gnorm = np.mean(list(mu_norm.values()), 0)
    else:
        gnorm = np.zeros(X.shape[1])
    per_class = []
    for c in classes:
        row: Dict[str, Any] = {"subset": c}
        if c in mu_def and c in mu_norm:
            wev_c = _unit(mu_def[c] - mu_norm[c])
            row["cos_dverb_ev_c"] = float(dvh @ wev_c)
        if c in mu_norm:
            v = mu_norm[c] - gnorm
            row["norm_centroid_on_ev"] = float(v @ w_ev_hat) if np.any(w_ev) else None
            row["norm_centroid_on_bias"] = float(v @ w_bias_hat) if np.any(w_bias) else None
        per_class.append(row)

    # Variance decomposition of the verbalizer score into evidence vs off-evidence parts.
    a_ev = float(dvh @ w_ev_hat) if np.any(w_ev) else 0.0
    d_along = a_ev * w_ev_hat
    d_perp = dvh - d_along
    s_verb = X @ d_verb
    s_along = X @ d_along
    s_perp = X @ d_perp

    def _var(x):
        return float(np.var(x)) if x.size else None

    def _mean_where(arr, sel):
        return float(arr[sel].mean()) if int(sel.sum()) else None

    norm_mask = y == 0
    var_decomp = {
        "var_s_verb": _var(s_verb),
        "var_evidence_part": _var(s_along),
        "var_offev_part": _var(s_perp),
        "frac_var_from_evidence": (
            float(np.var(s_along) / (np.var(s_verb) + 1e-12)) if np.var(s_verb) > 0 else None
        ),
        "normals_mean_evidence_part": _mean_where(s_along, norm_mask),
        "normals_mean_offev_part": _mean_where(s_perp, norm_mask),
        "FP_mean_evidence_part": _mean_where(s_along, is_fp),
        "TN_mean_evidence_part": _mean_where(s_along, is_tn),
        "FP_mean_offev_part": _mean_where(s_perp, is_fp),
        "TN_mean_offev_part": _mean_where(s_perp, is_tn),
    }

    d_debias = dvh - (float(dvh @ w_bias_hat) * w_bias_hat if np.any(w_bias) else 0.0)
    w_ev_debias = _unit(
        w_ev_hat - (float(w_ev_hat @ w_bias_hat) * w_bias_hat if np.any(w_bias) else 0.0)
    )
    readout_scores = {
        "verbalizer": s_verb,
        "evidence_supervised": X @ w_ev_hat,
        "verbalizer_minus_bias": X @ d_debias,
        "evidence_minus_bias": X @ w_ev_debias,
    }
    readouts: Dict[str, Any] = {}
    for name, s in readout_scores.items():
        tnr, thr = _tnr_at_recall(y, s, rec_target)
        readouts[name] = {"auc": _auc(y, s), "tnr_at_recall": tnr, "thr": thr}

    # Cross-class zero-shot: leave-one-class-out shared evidence direction.
    loo = []
    for c in classes:
        others = [cc for cc in classes if cc != c and cc in mu_def and cc in mu_norm]
        m = sub == c
        Xc = X[m]
        yc = y[m]
        if not others or int(yc.sum()) == 0 or int((1 - yc).sum()) == 0:
            loo.append({"subset": c, "skipped": "single_class_or_no_others"})
            continue
        w_loo = _unit(np.mean([mu_def[cc] - mu_norm[cc] for cc in others], 0))
        Xc_c = Xc - Xc.mean(0)  # label-free centering by the held-out class mean
        loo.append({"subset": c, "n": int(m.sum()), "auc_zeroshot": _auc(yc, Xc_c @ w_loo)})
    loo_aucs = [r["auc_zeroshot"] for r in loo if r.get("auc_zeroshot") is not None]

    out.update({
        "enabled": True,
        "n": int(X.shape[0]),
        "hidden_dim": int(X.shape[1]),
        "cosines": cos,
        "decision_sanity_corr_score_vs_margin": corr,
        "per_class": per_class,
        "variance_decomposition": var_decomp,
        "readouts": readouts,
        "rec_target": float(rec_target),
        "cross_class_zeroshot": {
            "mean_auc": (float(np.mean(loo_aucs)) if loo_aucs else None),
            "per_class": loo,
        },
    })
    return out


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
    ap.add_argument("--features-npz", type=str, default=None,
                    help="offline: load a *_features.npz and run ONLY feature-space diagnostics "
                         "(direction / shared / axis-residual / low-rank Gaussian); CPU only, no model.")
    args = ap.parse_args()

    if args.features_npz:
        _run_offline_features(args)
        return

    fracs = [float(x) for x in args.bottom_fracs.split(",") if x.strip()]

    # ---- Build rows: either from an existing ans JSON, or by running the model ----
    rows: List[dict] = []
    feat_store: Dict[str, List[dict]] = defaultdict(list)
    d_verb: Optional[np.ndarray] = None
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

        # Verbalizer direction in hidden space: lm_head[A] - lm_head[B] (the readout axis).
        try:
            lm_head = worker.model.model.lm_head
            W = lm_head.weight.detach().float().cpu().numpy()
            d_verb = W[ab_ids["A"]].mean(0) - W[ab_ids["B"]].mean(0)
            print(f"verbalizer d_verb: shape={tuple(d_verb.shape)} "
                  f"norm={float(np.linalg.norm(d_verb)):.4f}")
        except Exception as e:
            print(f"[decision-probe] could not build d_verb from lm_head: {repr(e)}")
            d_verb = None

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
                    "p_A": p_a, "cell": cell, "margin": scored.get("margin"), "X": v,
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

    # ---- Representation-level direction / mechanism analysis (needs dec features + d_verb) ----
    direction: Dict[str, Any]
    if feat_store and feat_store.get("dec") and d_verb is not None:
        direction = direction_analysis(feat_store["dec"], d_verb)
    elif feat_store:
        direction = {"enabled": False,
                     "reason": "dec feature / lm_head direction unavailable; "
                               "run with --feature dec (or dec+vis) and --model-path"}
    else:
        direction = {"enabled": False, "reason": "no features (e.g. --ans / --feature none)"}

    # ---- Shared whitened direction + per-class-intercept model (needs dec features) ----
    shared: Dict[str, Any]
    if feat_store and feat_store.get("dec"):
        shared = shared_model_analysis(feat_store["dec"], ridge=0.05, cv=5, seed=args.seed)
    else:
        shared = {"enabled": False, "reason": "no dec features (need --feature dec and model mode)"}

    # ---- D2: off-axis residual signal ; D1/D3: shared-subspace rank + low-rank Gaussian ----
    axis: Dict[str, Any]
    lowrank: Dict[str, Any]
    if feat_store and feat_store.get("dec"):
        axis = axis_residual_diagnostics(feat_store["dec"], C=args.probe_C, cv=args.probe_cv, seed=args.seed)
        lowrank = lowrank_gaussian_diagnostics(feat_store["dec"], seed=args.seed, cv=args.probe_cv)
    else:
        axis = {"enabled": False, "reason": "no dec features"}
        lowrank = {"enabled": False, "reason": "no dec features"}

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
        "direction": direction,
        "shared_model": shared,
        "axis_residual": axis,
        "lowrank_gaussian": lowrank,
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

    # ---- CSV: direction / mechanism ----
    if direction and direction.get("enabled"):
        dir_csv = csv_prefix + "_direction_readouts.csv"
        with open(dir_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            c = direction["cosines"]
            w.writerow(["cos_dverb_ev", "cos_dverb_bias", "cos_ev_bias",
                        "sanity_corr_score_vs_margin", "frac_var_from_evidence",
                        "cross_class_zeroshot_mean_auc"])
            w.writerow([c.get("dverb_ev"), c.get("dverb_bias"), c.get("ev_bias"),
                        direction.get("decision_sanity_corr_score_vs_margin"),
                        direction["variance_decomposition"].get("frac_var_from_evidence"),
                        direction["cross_class_zeroshot"].get("mean_auc")])
            w.writerow([])
            w.writerow(["readout", "auc", "tnr_at_recall", "threshold"])
            for name in ("verbalizer", "evidence_supervised", "verbalizer_minus_bias", "evidence_minus_bias"):
                r = direction["readouts"].get(name, {})
                w.writerow([name, r.get("auc"), r.get("tnr_at_recall"), r.get("thr")])
        print(f"wrote {dir_csv}")

        pc_csv = csv_prefix + "_direction_perclass.csv"
        zs = {r["subset"]: r for r in direction["cross_class_zeroshot"]["per_class"]}
        with open(pc_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["subset", "cos_dverb_ev_c", "norm_centroid_on_ev",
                        "norm_centroid_on_bias", "zeroshot_loo_auc"])
            for row in direction["per_class"]:
                z = zs.get(row["subset"], {})
                w.writerow([row.get("subset"), row.get("cos_dverb_ev_c"),
                            row.get("norm_centroid_on_ev"), row.get("norm_centroid_on_bias"),
                            z.get("auc_zeroshot")])
        print(f"wrote {pc_csv}")

    # ---- CSV: shared whitened direction + per-class-intercept model ----
    if shared and shared.get("enabled"):
        sh_csv = csv_prefix + "_shared_model.csv"
        with open(sh_csv, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["kind", "name", "in_sample_auc", "zeroshot_auc",
                           "is_acc", "is_rec", "is_tnr", "cv_acc", "cv_rec", "cv_tnr"])
            for name in ("ev", "diag", "lda"):
                r = shared["directions"].get(name, {})
                wcsv.writerow(["direction", name, r.get("in_sample_auc"),
                               r.get("zeroshot_auc"), None, None, None, None, None, None])
            for key in ("lda", "ev"):
                im = shared["intercept"][key]
                ins = im["in_sample"]; cvs = im["cv"]
                wcsv.writerow(["intercept", key, None, None,
                               ins["acc"], ins["rec"], ins["tnr"],
                               cvs["acc"], cvs["rec"], cvs["tnr"]])
        print(f"wrote {sh_csv}")

        spc_csv = csv_prefix + "_shared_perclass.csv"
        with open(spc_csv, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["subset", "b_lda", "in_rec_lda", "in_tnr_lda",
                           "cv_rec_lda", "cv_tnr_lda"])
            per = shared["intercept"]["lda"]["per_class"]
            for c in sorted(per.keys()):
                p = per[c]
                wcsv.writerow([c, p.get("b"), p["in"]["rec"], p["in"]["tnr"],
                               p["cv"]["rec"], p["cv"]["tnr"]])
        print(f"wrote {spc_csv}")

    # ---- optional features dump ----
    if args.save_feats and feat_store:
        npz_path = os.path.splitext(out_path)[0] + "_features.npz"
        dump = {}
        for name, frs in feat_store.items():
            dump[f"X_{name}"] = np.stack([r["X"] for r in frs], axis=0)
            dump[f"gt_{name}"] = np.array([int(r["gt"]) for r in frs])
            dump[f"subset_{name}"] = np.array([str(r["subset"]) for r in frs])
            dump[f"cell_{name}"] = np.array([str(r.get("cell", "")) for r in frs])
        if d_verb is not None:
            dump["d_verb"] = np.asarray(d_verb, dtype=np.float32)
        np.savez_compressed(npz_path, **dump)
        print(f"wrote {npz_path}")

    # ---- console verdict ----
    _print_verdict(distribution, capture, probes, direction, shared, axis, lowrank)


def _print_direction(direction: Dict[str, Any]) -> None:
    if not direction or not direction.get("enabled"):
        reason = direction.get("reason") if direction else "not run"
        print(f"\n[DIRECTION] disabled ({reason})")
        return
    c = direction["cosines"]
    print("\n---------------- DIRECTION / MECHANISM ----------------")
    print(f"n={direction['n']} hidden_dim={direction['hidden_dim']}  "
          f"cos(d_verb,w_ev)={_fmt(c.get('dverb_ev'))}  "
          f"cos(d_verb,w_bias)={_fmt(c.get('dverb_bias'))}  "
          f"cos(w_ev,w_bias)={_fmt(c.get('ev_bias'))}")
    print(f"decision sanity corr(<h,d_verb>, margin)="
          f"{_fmt(direction.get('decision_sanity_corr_score_vs_margin'))}")
    ro = direction["readouts"]
    rt = direction["rec_target"]
    for name in ("verbalizer", "evidence_supervised", "verbalizer_minus_bias", "evidence_minus_bias"):
        r = ro.get(name, {})
        print(f"  readout {name:22s}: AUC={_fmt(r.get('auc'))}  "
              f"TNR@recall{rt:.2f}={_fmt(r.get('tnr_at_recall'))}")
    vd = direction["variance_decomposition"]
    print(f"  normals: evidence-part={_fmt(vd.get('normals_mean_evidence_part'))}  "
          f"off-evidence-part={_fmt(vd.get('normals_mean_offev_part'))}  "
          f"(frac var from evidence={_fmt(vd.get('frac_var_from_evidence'))})")
    print(f"  FP off-ev={_fmt(vd.get('FP_mean_offev_part'))}  TN off-ev={_fmt(vd.get('TN_mean_offev_part'))}")
    cc = direction["cross_class_zeroshot"]
    print(f"  cross-class zero-shot LOO mean AUC={_fmt(cc.get('mean_auc'))}")
    dvh_ev = c.get("dverb_ev")
    dvh_bias = c.get("dverb_bias")
    if dvh_ev is not None and dvh_bias is not None:
        if dvh_bias > dvh_ev:
            print("  => verbalizer aligns MORE with the bias direction than evidence: "
                  "the Yes/No readout is prior-driven.")
        else:
            print("  => verbalizer aligns more with evidence than bias (readout mostly evidence-driven).")
    if cc.get("mean_auc") is not None:
        if cc["mean_auc"] >= 0.90:
            print("  => a SHARED evidence direction transfers to held-out classes: "
                  "a class-invariant anomaly direction exists.")
        else:
            print("  => shared-direction transfer is weak; the anomaly representation is partly class-specific.")
    print("-------------------------------------------------------\n")


def _print_shared(shared: Dict[str, Any]) -> None:
    if not shared or not shared.get("enabled"):
        reason = shared.get("reason") if shared else "not run"
        print(f"\n[SHARED-MODEL] disabled ({reason})")
        return
    d = shared["directions"]
    print("\n----------- SHARED DIRECTION / WHITENED MODEL -----------")
    for name in ("ev", "diag", "lda"):
        r = d.get(name, {})
        print(f"  dir {name:5s}: in-sample AUC={_fmt(r.get('in_sample_auc'))}  "
              f"zero-shot LOO AUC={_fmt(r.get('zeroshot_auc'))}")
    if not shared.get("lda_ok"):
        print(f"  (LDA whitening failed -> fell back to mean-diff: {shared.get('lda_error')})")
    print(f"  whitening: ridge={shared['ridge']}  k={shared['k_white']}  "
          f"hidden_dim={shared['hidden_dim']}")
    for key, label in (("lda", "w_lda + per-class intercept"),
                       ("ev", "w_ev  + per-class intercept")):
        im = shared["intercept"][key]
        ins = im["in_sample"]; cvs = im["cv"]
        print(f"  [{label}]")
        print(f"     in-sample: acc={_fmt(ins['acc'])} rec={_fmt(ins['rec'])} tnr={_fmt(ins['tnr'])} "
              f"(tp={ins['tp']} fp={ins['fp']} tn={ins['tn']} fn={ins['fn']})")
        print(f"     CV({shared.get('cv', '?')}):     acc={_fmt(cvs['acc'])} rec={_fmt(cvs['rec'])} "
              f"tnr={_fmt(cvs['tnr'])}")
    print("  (reference: deployed 0.5 threshold gave acc~0.916 rec~0.973 tnr~0.760)")
    print("---------------------------------------------------------\n")


def _print_axis(axis: Dict[str, Any]) -> None:
    if not axis or not axis.get("enabled"):
        r = axis.get("reason") if axis else "not run"
        print(f"\n[AXIS-RESIDUAL] disabled ({r})")
        return
    print("\n--------------- AXIS-OFF RESIDUAL (D2) ---------------")
    print(f"  full-state probe AUC        = {_fmt(axis.get('full_probe_auc'))}")
    print(f"  OFF-AXIS residual probe AUC = {_fmt(axis.get('residual_probe_auc'))}")
    per = [p for p in axis.get("per_class_residual_auc", []) if p.get("residual_probe_auc") is not None]
    if per:
        per = sorted(per, key=lambda x: x["residual_probe_auc"])
        print("  per-class residual AUC (worst 6): " +
              ", ".join(f"{p['subset']}={p['residual_probe_auc']:.3f}" for p in per[:6]))
    cond = [c for c in axis.get("conditional_auc_by_axis_bin", [])
            if c.get("auc_resid_within_bin") is not None]
    if cond:
        print("  within-axis-bin residual AUC (low->high t deciles): " +
              ", ".join(f"{c['auc_resid_within_bin']:.2f}" for c in cond))
    print("  (residual AUC stays high, esp. hard classes & high-t bins => off-axis info exists)")
    print("------------------------------------------------------\n")


def _print_lowrank(lr: Dict[str, Any]) -> None:
    if not lr or not lr.get("enabled"):
        r = lr.get("reason") if lr else "not run"
        print(f"\n[LOWRANK-GAUSS] disabled ({r})")
        return
    print("\n------ SHARED-SUBSPACE RANK + 2nd-ORDER GAUSSIAN (D1/D3) ------")
    spec = lr.get("singular_energy_top", [])
    if spec:
        print("  singular energy (top): " + ", ".join(f"{e:.2f}" for e in spec[:8]))
    for blk in lr.get("by_rank", []):
        print(f"  r={blk['r']:>2}: OOF AUC={_fmt(blk.get('oof_auc'))}  "
              f"TNR@recall0.95={_fmt(blk.get('tnr_at_recall95'))}")
    per = [p for p in lr.get("per_class_auc_best_r", []) if p.get("auc") is not None]
    if per:
        per = sorted(per, key=lambda x: x["auc"])
        print(f"  best_r={lr.get('best_r')} worst-6 per-class AUC: " +
              ", ".join(f"{p['subset']}={p['auc']:.3f}" for p in per[:6]))
    print("  (some r>1 beats r=1 and TNR@recall0.95 rises => low-rank 2nd-order readout helps)")
    print("----------------------------------------------------------------\n")


def _run_offline_features(args) -> None:
    npz = np.load(args.features_npz, allow_pickle=True)
    key = None
    for cand in ("dec", "vis"):
        if f"X_{cand}" in npz:
            key = cand
            break
    if key is None:
        raise SystemExit("features npz has no X_dec / X_vis array")
    X = np.asarray(npz[f"X_{key}"])
    gt = np.asarray(npz[f"gt_{key}"])
    sub = np.asarray(npz[f"subset_{key}"])
    cell = np.asarray(npz[f"cell_{key}"]) if f"cell_{key}" in npz else np.array([""] * len(gt))
    rows = [{"id": int(i), "subset": str(sub[i]), "gt": int(gt[i]), "p_A": None,
             "cell": str(cell[i]), "margin": None, "X": X[i]} for i in range(len(gt))]
    d_verb = np.asarray(npz["d_verb"]) if "d_verb" in npz else None

    print(f"[offline] loaded {args.features_npz}  feature={key}  n={len(gt)}  dim={X.shape[1]}")
    direction = (direction_analysis(rows, d_verb) if d_verb is not None
                 else {"enabled": False, "reason": "no d_verb in npz"})
    shared = shared_model_analysis(rows, seed=args.seed)
    axis = axis_residual_diagnostics(rows, C=args.probe_C, cv=args.probe_cv, seed=args.seed)
    lowrank = lowrank_gaussian_diagnostics(rows, seed=args.seed, cv=args.probe_cv)

    payload = {
        "meta": {"mode": "offline_features", "features_npz": args.features_npz,
                 "feature": key, "n": int(len(gt)), "seed": args.seed},
        "direction": direction, "shared_model": shared,
        "axis_residual": axis, "lowrank_gaussian": lowrank,
    }
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}")

    csv_prefix = args.csv_prefix or os.path.splitext(out_path)[0]

    ar_csv = csv_prefix + "_axis_residual.csv"
    with open(ar_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["full_probe_auc", axis.get("full_probe_auc")])
        w.writerow(["residual_probe_auc", axis.get("residual_probe_auc")])
        w.writerow([])
        w.writerow(["subset", "n", "residual_probe_auc"])
        for p in axis.get("per_class_residual_auc", []):
            w.writerow([p.get("subset"), p.get("n"), p.get("residual_probe_auc")])
        w.writerow([])
        w.writerow(["bin", "t_lo", "t_hi", "n", "auc_resid_within_bin"])
        for c in axis.get("conditional_auc_by_axis_bin", []):
            w.writerow([c.get("bin"), c.get("t_lo"), c.get("t_hi"), c.get("n"),
                        c.get("auc_resid_within_bin")])
    print(f"wrote {ar_csv}")

    lr_csv = csv_prefix + "_lowrank_gaussian.csv"
    with open(lr_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["r", "oof_auc", "tnr_at_recall95"])
        for b in lowrank.get("by_rank", []):
            w.writerow([b.get("r"), b.get("oof_auc"), b.get("tnr_at_recall95")])
        w.writerow([])
        w.writerow(["subset", "n", "auc_best_r"])
        for p in lowrank.get("per_class_auc_best_r", []):
            w.writerow([p.get("subset"), p.get("n"), p.get("auc")])
    print(f"wrote {lr_csv}")

    _print_direction(direction)
    _print_shared(shared)
    _print_axis(axis)
    _print_lowrank(lowrank)


def _print_verdict(distribution, capture, probes, direction, shared, axis=None, lowrank=None):
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

    _print_direction(direction)
    _print_shared(shared)
    _print_axis(axis) if axis is not None else None
    _print_lowrank(lowrank) if lowrank is not None else None
    print("========================================\n")


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)


if __name__ == "__main__":
    main()
