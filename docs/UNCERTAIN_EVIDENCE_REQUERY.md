# 不确定样本：图像证据 / 子问题追问方案（替代 Scheme B）

> 动机：Scheme B（窄带内回看 `musc_scores`）是测试侧分数拼接，不是「决策吃视觉证据」的机制解。  
> 本文档保存替代方案；**当前落地为方案 3（固定模板双向追问）**。方案 2（ROI crop 再问）已移除。  
> 相关：`docs/V0_SOFTMAX_CONFIDENCE_PROBE.md`、`docs/DECISION_DIAG_HANDOFF.md`。  
> 参考论文：Srinivasan et al., *Selective “Selective Prediction”: Reducing Unnecessary Abstention in Vision-Language Reasoning* (ACL Findings 2024) — ReCoVERR。

**日期：** 2026-10-08（方案 3 hyp requery 落地；方案 2 ROI 已删除）  
**仓库：** `/home/yz/code/Triad-main`  
**触发条件：** \(P(A)\in[\mathrm{band\_lo},\mathrm{band\_hi}]\)（默认 `[0.4, 0.7]`）。  
**带外样本：** LLM argmax，不进入追问。  
**硬约束：** 决策路径**禁止**回看 `musc_scores` / `musc_tau` 硬切。

---

## 0. 为何要换掉「窄带回看 MuSc」

| 观察（ecprloss + schemeB，MVTec musc_v0 全量 1725） | 含义 |
|---|---|
| 不确定带内 175 样本；用 MuSc 硬判 Acc≈80.6%，AUROC(musc)≈0.89 | MuSc 分数有排序力，但 `musc_tau` 硬切会损 Acc |
| 同批若用 LLM argmax：Acc≈46.9%，AUROC(\(P(A)\))≈0.55 | 窄带内 LLM 几乎无区分力，需要**额外视觉证据** |
| 带外 1550：Acc≈96.6%，AUROC(\(P(A)\))≈0.99 | 主路径可保留；改造应聚焦不确定子集 |
| `metal_nut` 等：MuSc AUROC 高但硬阈值 Acc 低 | 典型「排序对、阈值错」；不宜把专家标量当主裁判 |

**原则：** 最终 **是什么** 由 VLM 在全图上经子问题证据判定。无法决策时**回退全图 LLM argmax**，绝不回退到 `musc_scores`。

---

## 方案 1：ReCoVERR 式「子问题证据确认」（未实现）

### 1.1 思路

窄带不确定时不弃权、也不直接交 MuSc：用 LLM 提出与当前假设相关的**可视觉核验**子问题，向同一 VLM 追问，只收集**高置信**证据；证据充分则作答，否则弃权或兜底。

### 1.2 与现状关系

开放子问题生成噪声大；当前用**方案 3 固定模板**共享同一子问题执行器，完整开放出题留作后续消融（P2）。

---

## 方案 2：ROI / mask 引导的「看图证据」再问（**已删除**）

曾实现过 top-1 MuSc-mask ROI crop + \(\lambda\) 融合；效果不理想且与主问题同属泛化 A/B 缺陷问法。  
相关入口（`--scheme_requery roi`、`--requery_k`、`--requery_lambda`、`--requery_roi_delta`、ROI crop helpers）已从代码中移除。

---

## 方案 3：假设检验式双向追问（**已实现**）

### 3.1 思路

对 \(H_A\)（有缺陷）与 \(H_B\)（正常）**同时**收集高置信视觉证据，比较后再决，减轻 Yes-bias / 确认偏误。

### 3.2 已落地决策表（`eval_0shot_path.sh` 默认）

记全图首 token \(p_f=P_{\mathrm{full}}(A)\)，默认 \(\tau_{\mathrm{ev}}=0.7\)，\(\delta=1\)，每侧最多 `max_subq=2` 条子问题。

```text
全图 anyres+randomroi 得 p_f、llm_letter = argmax(p_f)

if p_f not in [band_lo, band_hi]:          # 默认 [0.4, 0.7]
    → letter = llm_letter
      decision_src = llm_argmax_out_of_band

else:
    按 subset 取固定模板 checklist（H_A / H_B 各 ≤ max_subq 条）
    对每条子问题（同一张已处理全图）再跑 VLM 单 token A/B:
      kept = max(p_A, p_B) >= tau_ev
      H_A 答 Yes → S_A += 1；答 No → S_B += 1   （仅 kept）
      H_B 答 Yes → S_B += 1；答 No → S_A += 1   （仅 kept）

    if S_A - S_B >= δ:   → letter = A, decision_src = hyp_test
    elif S_B - S_A >= δ: → letter = B, decision_src = hyp_test
    else:                → letter = llm_letter, decision_src = hyp_fallback
```

**禁止：** 任何分支使用 `musc_scores ≥ τ` / Scheme B 式专家标量裁判。

### 3.3 实现要点

- 清单：`llava_eval/hyp_checklists.py`（MVTec 15 类 + default fallback）
- 子问题须**可在图上核验**；执行时复用主问题已处理的 `image_args`（与主 forward 同视野）
- Sidecar：`S_A`、`S_B`、`sub_answers`、`decision_src`、`in_band`、`subset`

### 3.4 代码入口

| 位置 | 作用 |
|---|---|
| `eval_0shot_path.sh` | 默认 `scheme_requery`（hyp） |
| `scripts/eval_triad_0shot_v_gpu_dataset_expert_prod.sh` | mode `scheme_requery[_v2][_default]` / `…:lo:hi:tau_ev:delta` |
| `llava_eval/evaluate.py` | `--scheme_requery hyp|hyp_v2`、v2 聚合超参；与 `--ab_tau` 互斥 |
| `llava_eval/simple_model_worker.py` | 窄带 H_A/H_B 追问 + hyp/hyp_v2 聚合；sidecar `ab_decision` |
| `llava_eval/hyp_checklists.py` | 按 subset 的固定模板 |
| `llava_eval/hyp_aggregate_v2.py` | A2 软聚合（hyp_v2） |

Sidecar 字段：`p_A`（=p_f）、`S_A`、`S_B`、`sub_answers`、`in_band`、`decision_src`、`requery`、`subset`；hyp_v2 另有 `hyp_v2`。  
**已删除：** `--scheme_b` / ROI requery 相关参数与 `decision_src=roi_*`。

### 3.5 消融建议

- \(\tau_{\mathrm{ev}}\)、\(\delta\)、`max_subq`；  
- 仅 \(H_A\) vs 双向；  
- 在 FP 多的类别上重点看 TNR；  
- 后续可叠 ROI crop 输入或开放 LLM 出题（方案 1）。

---

## 方案 3b：A2 软证据聚合（`hyp_v2` / `requeryHypv2`，已实现）

与方案 3 **共用同一套 checklist 与子问题 forward**，只改聚合：

1. **软票** \(v=2p_{\mathrm{Yes}}-1\)（`conf<τ_ev` 则 0）  
2. **成对反对称**：同 index 的 H_A/H_B 同答 Yes/No → 权重 `conflict_lambda`（默认 0.3）；反向一致 → `agree_lambda`（默认 1.2）  
3. **冲突门控**：`margin/mass < conflict_ratio`（默认 0.35）→ fallback  
4. **与 \(p_f\) 混合**：`β=min(1, margin/margin_ref)`，`final = β·tanh(score/T) + (1-β)·(2p_f-1)`

```text
--scheme_requery hyp_v2
# shell: scheme_requery_v2  → decode tag requeryHypv2
#        scheme_requery_v2:0.4:0.7:0.7:0.5
```

可选超参：`--requery_v2_conflict` / `_agree` / `_margin_ref` / `_conflict_ratio` / `_temp`。  
Sidecar 额外字段：`hyp_v2`（score/mass/margin/beta/final/pairs/gate）。  
`decision_src`：`hyp_v2_test` | `hyp_v2_fallback` | `hyp_v2_fallback_conflict`。

---

## 方案 3c：Defv3（`requeryHypDefv3`，已实现）

在 HypDefv2 上叠加：

1. **v3-a 困难例选取（quantile gate）**：两遍评测；pass-1 只算 \(u=1-|2p_A-1|\)；按本 run 的 `top_frac`（默认 0.1）取最不确定子集再追问。**无绝对 \(p_A\) 带宽超参**。  
2. **v3-b 对比式 default_v3 checklist**：相对连续性 / 局部一致性 / 几何对齐问句，避免 scratch/crack/stain 等缺陷 taxonomy 词与类名。

```text
--scheme_requery hyp --requery_checklist default_v3 \
  --requery_gate quantile --requery_top_frac 0.1 \
  --requery_tau_ev 0.7 --requery_delta 1.0 --requery_max_subq 3

# shell: scheme_requery_default_v3
#        scheme_requery_default_v3:0.1:0.7:1   # top_frac:tau_ev:delta
```

Sidecar 额外：`u`、`u_thr`、`gate=quantile`。

---

## 组合与落地优先级

| 优先级 | 组合 | 状态 |
|---|---|---|
| ~~P0~~ | 方案 2 ROI | **已删除** |
| **P1** | **方案 3 固定模板双向** | **已落地** |
| **P1b** | **方案 3b 软聚合 hyp_v2** | **已落地** |
| **P1c** | **方案 3c Defv3（quantile + contrastive）** | **已落地** |
| P2 | 方案 1 完整 ReCoVERR（开放出题） | 未做 |
| 兜底 | 证据仍不足 → **回退全图 llm_letter** | **禁止** `musc_scores` |

**建议指标（除 Acc/Pre/Rec/TNR）：**

- 带内 / 带外分层 Acc；  
- `decision_src` 分布（out_of_band / hyp_test / hyp_fallback）；  
- 额外 forward 次数（窄带约 +4×，每侧 2 题）。

---

## 正式评测接口（方案 3 已实现）

```text
# 推荐（eval_0shot_path.sh 默认 = hyp 硬票）
--scheme_requery hyp --band_lo 0.4 --band_hi 0.7 \
  --requery_tau_ev 0.7 --requery_delta 1.0 --requery_max_subq 2

# A2 软聚合（requeryHypv2）
--scheme_requery hyp_v2 --band_lo 0.4 --band_hi 0.7 \
  --requery_tau_ev 0.7 --requery_delta 0.5 --requery_max_subq 2

# Shell mode 简写
scheme_requery                         # hyp，tag=requeryHyp
scheme_requery:0.4:0.7:0.7:1           # lo:hi:tau_ev:delta
scheme_requery_v2                      # hyp_v2，tag=requeryHypv2
scheme_requery_v2:0.4:0.7:0.7:0.5
```

与 `--ab_tau` 互斥；仅 `config:v0`。  
旧 `--scheme_b` 与 `--scheme_requery roi` **已移除**。

---

## 参考

1. Tejas Srinivasan et al. Selective “Selective Prediction”: Reducing Unnecessary Abstention in Vision-Language Reasoning. ACL Findings 2024. https://aclanthology.org/2024.findings-acl.767/  
2. 代码参考：https://github.com/tejas1995/ReCoVERR  
3. 本仓库 Scheme A 与置信度探针：`docs/V0_SOFTMAX_CONFIDENCE_PROBE.md`
