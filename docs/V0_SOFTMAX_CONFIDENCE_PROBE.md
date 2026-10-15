# v0 二分类 Softmax 置信度探针

> 独立评测脚本 + 两模型对照结论。  
> 正式 `evaluate.py` 测试侧决策（**仅 `config:v0`**）：  
> - 方案 A：`--ab_tau`（纯 \(P(A)\) 门控）  
> - 方案 requery：`--scheme_requery hyp`（窄带内固定模板 \(H_A\)/\(H_B\) 双向追问；**禁止** `musc_scores` 硬切）  
> - 旧方案 B（`--scheme_b`）与 ROI crop requery（`--scheme_requery roi`）**已移除**  
> `cotm_*` 长文本禁止上述门控。详见 `docs/UNCERTAIN_EVIDENCE_REQUERY.md`。  
> 相关：`docs/DECISION_DIAG_HANDOFF.md`、`docs/TNR_TRAIN_HANDOFF.md`。

**日期：** 2026-09-29  
**仓库：** `/home/yz/code/Triad-main`  
**环境：** `conda activate triad`  
**脚本：** `llava_eval/v0_softmax_confidence_probe.py`

---

## 1. 动机

正式 musc_v0 评测只存硬标签 `A`/`B`（A=Yes/缺陷，B=No/正常），CSV 只有 acc/pre/rec/tnr，**没有**「模型说 Yes 有多自信」。

本探针回答：

1. 单样本 \(P(\text{Yes})\equiv P(A)\) 是多少？  
2. 按混淆矩阵四格（TP/FP/TN/FN）各自的平均自信度是多少？  
3. 不同 checkpoint 在「嘴上的自信形态」上差在哪里？

**范围边界（刻意不做）：** 这是 **输出层 / lm_head** 上的二类 softmax（「嘴」），不是内部 hidden 探针或干预稳定性（「心」）。后者可后续单独做。

---

## 2. 探针逻辑

### 2.1 与正式 v0 对齐的部分

| 项 | 做法 |
|----|------|
| 数据 | `evaluation/mvtec/question_musc.jsonl`（默认） |
| Prompt | `--text_src config:v0` → `MVTEC_PROMPT_V0`（A. Yes / B. No） |
| 图像 | 单图 + MuSc mask → `anyres_max_9_randomroi`（与 eval 一致） |
| 加载 | `SimpleModelWrapper` / `causal_probe_common.load_worker` |
| 标签 | `gt=1` 异常 → 正类；`pred=1` 当预测为 A |

**不修改** `evaluate.py`、`simple_model_worker.generate_batch`、`collect_acc.py`；图像预处理在探针内复刻 `generate_batch` 路径。

### 2.2 置信度定义（核心）

对每个样本：

1. 拼好与 `single_qa` 相同的 chat prompt（`qwen_1_5`）。  
2. `model.generate(..., do_sample=False, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)`。  
3. 取**第一个生成步**的 logits \(z\in\mathbb{R}^{V}\)。  
4. 解析 tokenizer 中 A/B 的单 token 表面形式（实测 Qwen：`A∈{32,362}`，`B∈{33,425}`，含前导空格变体）。  
5. \(z_A=\max_{i\in A} z_i\)，\(z_B=\max_{i\in B} z_i\)，再二类 softmax：

\[
P(A)=\frac{e^{z_A}}{e^{z_A}+e^{z_B}},\quad P(B)=1-P(A)
\]

6. **硬预测** \(\mathrm{pred}=\arg\max(P(A),P(B))\)（与置信度同一套分数，greedy，避免与正式 eval 的 `temp=0.2` 采样不一致）。

语义：\(P(A)\) = **「模型说 Yes/缺陷的自信百分比」**（在 A/B 强制二选一意义下）。

辅助量：

- `p_pred`：所选字母的概率（pred=A 时 = \(P(A)\)，否则 = \(P(B)\)）  
- `margin`：\(z_A-z_B\)  
- `p_A_fullvocab`：全词表 softmax 后 A 表面形式质量（诊断用，通常很小）

### 2.3 四格与汇总

| 格子 | 条件 |
|------|------|
| TP | gt=1, pred=A |
| FP | gt=0, pred=A |
| TN | gt=0, pred=B |
| FN | gt=1, pred=B |

每格统计：`n`、`mean/median/std` of \(P(A)\)、`p_pred`、`margin`。  
另报 overall / per-subset 的 acc、pre、rec、tnr（与 `collect_acc` 同口径；因 greedy 可能与正式 `temp=0.2` 结果有微小差别）。

### 2.4 输出物

| 文件 | 内容 |
|------|------|
| `*.json` | `meta`（定义、token ids）、`summary`（all / per_subset / cells）、默认含 `rows` 逐样本 |
| `*_summary.csv` | 每类 + mean + all：硬指标 + `mean_pA_{TP,FP,TN,FN}` + `mean_pPred_*` |

### 2.5 运行命令

```bash
conda activate triad
cd /home/yz/code/Triad-main

# 冒烟
CUDA_VISIBLE_DEVICES=0 python llava_eval/v0_softmax_confidence_probe.py \
  --model-path checkpoints/merged_llava_qwen_triad_9ROI_cotmv6_ecprloss \
  --num-samples 16 --text_src config:v0 \
  --out LLM_judge_output/v0_softmax_conf_smoke.json

# 全量 musc_v0（1725）
CUDA_VISIBLE_DEVICES=0 python llava_eval/v0_softmax_confidence_probe.py \
  --model-path <MERGED_CKPT> \
  --text_src config:v0 \
  --out LLM_judge_output/v0_softmax_conf_<tag>.json
```

常用参数：`--subset bottle,pill`、`--num-samples N`、`--no-rows`（只要 summary）、`--csv path`。

### 2.6 如何读指标（避免误解）

- **格子归属**仍靠硬 A/B；**格子内的 `mean_pA_*`** 才是自信百分比。  
- 正式 eval CSV 与探针硬指标可能略有差别（采样 vs greedy）。比自信形态时以探针 JSON/CSV 为准。  
- `musc_scores` **不是** LLM 置信度，本探针不用。  
- 训练侧 decision/ECPR 盯的是 CoTM **Yes/No** token；v0 决策字母是 **A/B**——机制可类比，token 集合不同。

---

## 3. 对照实验：ecprloss vs cotmv3_tnr

### 3.1 Checkpoint 与产物

| Tag | Merged 路径 | Softmax 全量输出 |
|-----|-------------|------------------|
| ecprloss | `checkpoints/merged_llava_qwen_triad_9ROI_cotmv6_ecprloss` | `LLM_judge_output/v0_softmax_conf_ecprloss.json` (+ `_summary.csv`) |
| cotmv3tnr | `checkpoints/merged_llava_qwen_triad_9ROI_cotmv3_tnr` | `LLM_judge_output/v0_softmax_conf_cotmv3tnr.json` (+ `_summary.csv`) |

设置：MVTec musc_v0，全量 1725，UN=0。

### 3.2 硬指标（探针 all）

| | acc | pre | rec | tnr | FP | FN |
|--|-----|-----|-----|-----|----|----|
| ecprloss | 0.915 | 0.916 | **0.973** | 0.760 | **112** | 34 |
| cotmv3tnr | **0.928** | **0.968** | 0.932 | **0.917** | 39 | **86** |

- tnr 路线：用更多漏检换大幅降虚警。  
- ecprloss：高召回、Yes bias 更明显。

### 3.3 四格平均 \(P(A)\)（说 Yes 的自信）

| 格子 | ecprloss | cotmv3tnr | 解读 |
|------|----------|-----------|------|
| TP | 0.913 | 0.876 | 真阳性两边都较笃定；tnr 略压低 Yes |
| FP | 0.651 | 0.667 | 虚警多为「略偏 Yes」，很少极端 0.99 |
| TN | **0.207** | **0.117** | tnr 说 No 时对 Yes 的残留更低（拒识更干净） |
| FN | 0.359 | 0.237 | tnr 漏检更偏向 No；ecpr 漏检更贴决策边界 |

补充分布：

| 诊断计数 | ecprloss | cotmv3tnr |
|----------|----------|-----------|
| FP 且 \(P(A)\ge 0.9\) | 5 | **0** |
| FP 且 \(P(A)\ge 0.8\) | 14 | 6 |
| FP 且 \(P(A)< 0.6\) | 47 | 18 |
| TN 且 \(P(A)\ge 0.4\)（No 不稳） | 38 | 13 |
| TP 且 \(P(A)< 0.8\)（弱 Yes） | 121 | 227 |

→ tnr 几乎没有「极其自信的虚警」；正确拒识更稳；正确检出时自信略低（与整体压 Yes 一致）。

### 3.4 难类

**ecprloss TNR 极差：** bottle (0.10)、pill (0.27)、toothbrush (0.33)、transistor (0.55)；这些类 FP 的 mean \(P(A)\) 约 0.59–0.70（犹豫着说 Yes）。

**cotmv3tnr：** 上述类 TNR 明显抬升（如 bottle 0.85、toothbrush 0.92）；代价包括 **metal_nut** 召回大跌（FN 约 6→23）。

pill / transistor 在两边仍是相对难点（tnr 上 pill TNR≈0.65，transistor≈0.78）。

### 3.5 一句话结论

两边 **TP 都高、FP 自信都中等**；真正拉开的是：

1. **还敢不敢说 No**（FP：112 → 39）  
2. **说 No 有多干净**（TN 的 \(P(A)\)：0.21 → 0.12）

若目标是压 Yes bias / 抬 TNR，`cotmv3_tnr` 的置信度形态更符合；`ecprloss` 更像高召回、边界虚警多的模型。

---

## 4. 后续可选

按优先级，不必一次做完：

1. **阈值校准（方案 A）：** `--ab_tau`（v0 only，opt-in）。测试侧压偏置，非机制解。  
2. **窄带假设追问（方案 requery hyp，已接入）：** `--scheme_requery hyp`（v0 only）。  
   - 规则：若 \(P(A)\in[\mathrm{band\_lo},\mathrm{band\_hi}]\)，按 subset 固定模板对 \(H_A\)/\(H_B\) 子问题追问；仅保留 \(\max(p_A,p_B)\ge\tau_{\mathrm{ev}}\) 的证据，聚合 \(S_A,S_B\)；\(|S_A-S_B|\ge\delta\) 则判决，否则回退 `llm_letter`。  
   - **不**使用 `musc_scores`。入口：`bash eval_0shot_path.sh <merged_ckpt> [gpu]`（默认 `scheme_requery`）。  
   - 答案 sidecar：`ab_decision`（含 `in_band` / `decision_src` / `S_A` / `S_B` / `sub_answers`）。详见 `docs/UNCERTAIN_EVIDENCE_REQUERY.md`。  
3. **「心」侧：** 干预稳定性与 \(P(A)\) 对齐。  
4. **层间 / ECPR 旁路** 作第三通道。

---

## 5. 关键路径速查

| 用途 | 路径 |
|------|------|
| 探针脚本 | `llava_eval/v0_softmax_confidence_probe.py` |
| 共用加载 | `llava_eval/causal_probe_common.py` |
| v0 prompt | `llava_eval/prompt/mvtec.py` → `MVTEC_PROMPT_V0` |
| 正式硬指标 CSV | `checkpoints/<merged>/result_mvtec__musc_v0_all.csv` |
| 本对照 JSON/CSV | `LLM_judge_output/v0_softmax_conf_{ecprloss,cotmv3tnr}*` |
