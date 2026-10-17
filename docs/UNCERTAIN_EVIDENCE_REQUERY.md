# 不确定样本：图像证据 / 子问题追问方案（替代 Scheme B）

> 动机：Scheme B（窄带内回看 `musc_scores`）是测试侧分数拼接，不是「决策吃视觉证据」的机制解。  
> 本文档保存替代方案；**当前落地为方案 3（固定模板双向追问）与方案 4（`selfq`：T2 自生成问题 + 贝叶斯融合）**。方案 2（ROI crop 再问）已移除。  
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

## 方案 4：自生成问题 + 贝叶斯证据融合（`selfq`，T2，已实现）

> 恢复初衷：让模型**自己写核验问题**（T2：propose→answer），而不是查手写模板；并用**无绝对阈值**的方式决定是否采信。

### 4.1 动机（修正方案 3 的两个痛点）

- 痛点 A：困难例选取依赖**窄带** `[lo,hi]`，或即使 quantile 也要 `tau_ev=0.7` 这类绝对阈值——脆弱、难迁移。
- 痛点 B：子问题来自 `hyp_checklists.py` 的**固定模板**，与图像无关、且按类别泄露缺陷 taxonomy，不通用。

### 4.2 设计（零 band / 零 tau_ev / 零 delta）

1. **选取**：沿用 quantile gate（两遍评测），默认 `top_frac=0.1`（最不确定的 10%）。**去掉 band**。
2. **出题（T2，自由文本）**：模型针对同一张图**自己生成**一条 yes/no 核验问题，并声明映射：
   ```text
   QUESTION: <一句可在图上核验的 yes/no 问题>
   YES_SUPPORTS: <A or B>
   NO_SUPPORTS:  <A or B>
   ```
3. **作答（强制二分）**：把 `QUESTION` 用 `A. Yes / B. No` 拼进去跑单 token 打分 → `p_Yes`。
   **无论问题多开放，答案永远是二分类且带概率**（关键：生成自由、作答闭合）。
4. **稳定判据=相对锐度（替代 `tau_ev`）**：仅当 `H(sub) < H(main)`（子问题二分类熵 < 主问题熵）才采信。
5. **融合同样无阈值**：`logit_final = logit(p_main_A) + Σ w_i·sub_logit_A`，`w_i∈{0,1}`，以 **0** 决断；
   单条证据 logit 仅做 `±6` 饱和保护（数值守卫，非决策常数，≈ p=0.9975）。
6. **永不** `musc_scores`；解析失败/无信息（Yes/No 指向同侧或 `neither`）→ 丢弃该条；无任何采信证据 → 回退主答案。

### 4.3 代码入口

| 位置 | 作用 |
|---|---|
| `eval_0shot_path.sh` / `scripts/...prod.sh` | mode `scheme_requery_selfq[:top_frac[:max_turns]]`（tag `requerySelfq`） |
| `llava_eval/evaluate.py` | `--scheme_requery selfq`（`choices` 已含）；`--requery_max_turns`、`--requery_selfq_clip`、`--requery_selfq_gen_max_new_tokens` |
| `llava_eval/simple_model_worker.py` | `mode="selfq"` 分支：生成→强制二分→熵门→`bayes_fuse` |
| `llava_eval/hyp_selfq.py` | 自生成 prompt、结构化解析、`binary_entropy`、`sub_evidence_logit_for_a`、`bayes_fuse` |

Sidecar：`selfq`（`logit_main/evidence_logit/logit_final/p_final/n_evidence`）、`sub_answers[*]`（`gen_text/q/yes_supports/no_supports/p_A/p_B/accept/kept/reason`）。
`decision_src`：`selfq_bayes` | `hyp_fallback`（另有 gate 侧 `llm_argmax_out_of_band` / `llm_argmax_score_only`）。

### 4.4 推荐调用

```text
--scheme_requery selfq --requery_gate quantile --requery_top_frac 0.1 --requery_max_turns 2

# shell:
scheme_requery_selfq            # 默认 top_frac=0.1, max_turns=2
scheme_requery_selfq:0.1:2
```

### 4.5 消融建议

- `top_frac`∈{0.05,0.1,0.2}；`max_turns`∈{1,2,3}；
- 熵门开/关（关=只看是否 informative）；
- 融合 vs 挑战者（默认保留主答案、稳定一致指向反面才翻案）；
- 重点指标：选中集 Acc/TNR、**翻案率**、`decision_src` 分布、额外 forward 数、解析失败率；
- 盯"翻对 vs 误伤"（把原本对的改成错的量）。

---

## 方案 5：注意力引导裁图 + default_v2 追问（`attncrop` / `requeryAttnCrop`，default_v4，已实现）

> 复用 EG-RoI 视图框架：只用 decision-token 注意力**供框**，框交给 `process_images(boxes_list=...)` 从**原图**裁高保真 ROI，再把 `default_v2` 清单在裁剪视图上重问。注意力**不做裁判**，最终仍由 VLM A/B 决定（永不 `musc_scores`）。

### 5.1 数据流

```text
主路 全图 anyres+randomroi → 单token A/B → p_f, llm_letter
  │ 量化门(quantile top_frac，默认0.1) 选中困难例
  ▼
attn_crop：只读Q/K 取 decision-token 在 midband 层的注意力
  → 平均层→anyres网格→去sink→top5%质量门(conc_floor)→最大质量连通域
  → 正方形外扩(expand) + 面积clip[side_lo,side_hi] → suspect box
  ▼
process_images(raw_images, boxes_list=[[box]]) 从原图裁 ROI（EG-RoI 多视图）
  ▼
default_v2 清单(H_A/H_B×3) 在裁剪视图上 forced A/B → 硬计票 S_A/S_B
  ▼
|S_A-S_B|>=delta → 翻案(attncrop_test)；否则回退 llm_letter(attncrop_fallback)
```

### 5.2 关键约束（全部相对，无绝对阈值）

- **层**：`midband`（默认，≈0.50/0.57/0.64/0.71·n）；可 `mid|last|last4|逗号列表`。
- **供框门**：`attncrop_conc_floor`（top5% 质量，默认 0.30）——散的直接不出框，落回全图。
- **框**：`thr_q=0.90` 相对阈值、`expand=1.3`、面积 `[0.06,0.45]·min(W,H)`；从原图裁，偏大宁可含上下文。
- **冻结约束**：不读 `musc_scores`；`slot` 由注意力框提供，判据仍是 VLM。

### 5.3 代码入口

| 位置 | 作用 |
|---|---|
| `eval_0shot_path.sh` / `scripts/...prod.sh` | mode `scheme_requery_default_v4[:top_frac]`（tag `requeryAttnCrop`） |
| `llava_eval/evaluate.py` | `--scheme_requery attncrop`；`--requery_attncrop_{layers,thr_q,expand,side_lo,side_hi,conc_floor,overlay_dir}` |
| `llava_eval/simple_model_worker.py` | `mode="attncrop"` 分支：注意力→框→裁图→default_v2 追问→硬计票 |
| `llava_eval/attn_crop.py` | 只读Q/K `score_layers`、`SegmentRecorder`、`grid_to_box`、`save_attn_box_overlay` |

Sidecar 额外：`attncrop{box,used_crop,attn{kept,reason,top5_mass,area_frac,layers}}`。  
`decision_src`：`attncrop_test` | `attncrop_fallback` | `attncrop_fallback_nobox`。

### 5.4 注意力叠加可视化

`--requery_attncrop_overlay_dir <dir>` 开时，每个被选中样本输出一张三栏 PNG：`original(+红框) | 注意力热力叠加(+红框) | crop`。

### 5.5 推荐调用

```text
# shell：
scheme_requery_default_v4              # 默认 quantile top 0.1
scheme_requery_default_v4:0.1

# evaluate.py 直连：
--scheme_requery attncrop --requery_checklist default --requery_gate quantile \
  --requery_top_frac 0.1 --requery_max_subq 3 \
  --requery_attncrop_layers midband --requery_attncrop_overlay_dir results/attncrop_viz
```

### 5.6 消融建议

- `attncrop_layers`∈{midband, mid, last4}；`conc_floor`∈{0.2,0.3}；
- `expand`∈{1.2,1.3,1.5}；`side_hi`∈{0.35,0.45}；
- 对比 `scheme_requery_default_v2`（不裁图）；重点看选中集 Acc/TNR、**翻对 vs 误伤**、`decision_src` 分布、`used_crop` 比例。

---

## 组合与落地优先级

| 优先级 | 组合 | 状态 |
|---|---|---|
| ~~P0~~ | 方案 2 ROI | **已删除** |
| **P1** | **方案 3 固定模板双向** | **已落地** |
| **P1b** | **方案 3b 软聚合 hyp_v2** | **已落地** |
| **P1c** | **方案 3c Defv3（quantile + contrastive）** | **已落地** |
| **P1d** | **方案 4 selfq（T2 自生成 + 贝叶斯融合，无绝对阈值）** | **已落地** |
| **P1e** | **方案 5 default_v4（注意力裁图 + default_v2 追问，attncrop）** | **已落地** |
| P2 | 方案 1 完整 ReCoVERR（开放出题，多轮反驳） | 未做 |
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
