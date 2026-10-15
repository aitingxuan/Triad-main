# Triad TNR / 降 Yes bias 训练 — 会话交接文档

> 给下一个 Cursor 会话用。读完本文即可接续，无需翻完整旧对话。  
> 前置结论：`docs/DECISION_DIAG_HANDOFF.md`（attn-aux **已否决**；瓶颈是 Yes bias / TNR）。  
> 相关：`docs/ATTN_AUX_HANDOFF.md`（aux 实现历史，勿再训）。

**日期：** 2026-09-17 → 2026-09-18  
**仓库：** `/home/yz/code/Triad-main`  
**环境：** `conda activate triad`

---

## 0. 用户真实目标（不变）

**提高异常二分类（Yes/No）准确率，尤其改善 TNR / 降 Yes bias（过检）。**  
可以重训；不要再走 attn aux / P0 attention probe。

基线参考（`merged_..._vis_ori`，musc_v0）：

| 指标 | 大约 |
|------|------|
| acc | 0.916 |
| rec | 0.979（已饱和） |
| tnr | 0.745 |
| 最差类 TNR | pill≈0.12, bottle≈0.20, toothbrush≈0.42, transistor≈0.53 |

---

## 1. 本会话结论与选型

### 1.1 路线（已落地）

按 `DECISION_DIAG_HANDOFF` 的 **P0**：训练侧打 Yes bias，**关掉 attn_aux**。

两板斧一起上：

1. **数据**：上采样 CoTM 的 No 答案 + InstructIAD 正常样本（**不用** musc_v0 评测 FP，避免 MVTec test 泄漏）
2. **损失**：对每条样本**最后一个** Yes/No token 加重 CE（默认 No×3）

### 1.2 明确不做

- 再训 / 调 `attn_aux_*`
- 用 musc_v0 FP 当训练集 hard negative（泄漏）
- 用 rec / P0 `W2_vision` 当成功标准

---

## 2. 新增 / 修改文件一览

### 2.1 新增

| 路径 | 作用 |
|------|------|
| `LLaVA-NeXT/llava/train/decision_token_loss.py` | 决策 token 加权 CE：只重加权**最后一次** Yes/No |
| `scripts/build_tnr_train_data.py` | 从 CoTM / InstructIAD 抽出 No / normal 子集 |
| `movable/CoTM/CoTM_v3_no.json` | CoTM 中答案为 No 的样本（约 868） |
| `movable/InstructIAD/InstructIAD_normal.json` | `is_anomaly==0`（约 5428） |
| `movable/train_tnr_cotmv3.yaml` | 训练数据配方（all + repeat） |
| `docs/TNR_TRAIN_HANDOFF.md` | **本文** |

### 2.2 修改

| 路径 | 改动要点 |
|------|----------|
| `finetune_ov_0shot.sh` | 实验名 `Triad_llava_ov_lora_tnr`；`attn_aux_loss False`；`data_path=train_tnr_cotmv3.yaml`；打开 `decision_token_loss` |
| `LLaVA-NeXT/llava/train/llava_trainer.py` | `compute_loss`：若 `decision_token_loss`，走 `dpo_forward=True` 拿 **expanded labels**，再算加权 CE |
| `LLaVA-NeXT/llava/train/train.py` | 新超参 `decision_token_*`；yaml `sampling_strategy: repeat:N` |

---

## 3. 训练配方细节

### 3.1 数据 mix（`movable/train_tnr_cotmv3.yaml`）

```text
CoTM_v3.json              all          1759
CoTM_v3_no.json           repeat:2     868×2 = 1736   → No 样本相对原 CoTM 约 3×
InstructIAD.json          all          9844
InstructIAD_normal.json   repeat:1     5428           → normal 相对原 IAD 约 2×
────────────────────────────────────────────
合计约 18767（原 CoTM+IAD = 11603）
```

重建子集（幂等）：

```bash
python3 scripts/build_tnr_train_data.py
```

### 3.2 关键超参（`finetune_ov_0shot.sh`）

```text
--attn_aux_loss False
--decision_token_loss True
--decision_token_no_weight 3.0
--decision_token_yes_weight 1.0
--decision_token_log_every 50
--data_path .../movable/train_tnr_cotmv3.yaml
exp / output: ..._cotmv3_tnr
```

其余（LoRA r=32、lr=1e-5、1epoch、5 卡 zero3、anyres_max_9_randomroi 等）与原 vis_ori 配方一致。

### 3.3 决策 loss 实现要点（易踩坑）

1. **必须用图像展开后的 labels**  
   dataloader 的 `labels` 是文本长（~500），logits 是展开后（~8192）。  
   正确做法：`model(..., dpo_forward=True) -> (logits, expanded_labels)`，再 CE。

2. **只加权每条序列最后一个 Yes/No**  
   避免 CoTM `<think>` 里 “no visible…” 被误加重。

3. **InstructIAD 是 A/B，不是 Yes/No**  
   当前加权**不碰** A/B（选项顺序还会翻转）。IAD 正常类主要靠数据上采样起作用。

4. **`num_items_in_batch` 可能是 int**  
   已兼容 int / tensor，勿再 `.to()` 直接调。

---

## 4. 训练排错记录（已修）

| 现象 | 原因 | 修复 |
|------|------|------|
| `batch_size (8191) vs (473)` | 用了未展开的 text labels | `dpo_forward=True` 取 expanded labels |
| `'int' object has no attribute 'to'` | `num_items_in_batch` 是 int | `decision_token_loss.py` 分支处理 int/tensor |

---

## 5. 当前训练状态（写文档时）

- **脚本：** `bash finetune_ov_0shot.sh`（第三次启动成功）
- **日志：** `tmp_log/llava_train_20260917_203536.log`
- **输出目录：**  
  `checkpoints/llava-siglip-so400m-onevision-qwen2-7b--Triad_llava_ov_lora_tnr_1epoch_4x8x4_1e-5_9ROI_cotmv3_tnr`
- **进度：** 约 **90% / 117 steps** 仍在跑（`save_strategy=no`，结束前可能只有 `runs/`，需等写完 adapter）
- **观察：** `decision_tok/n_no` / `n_yes` 在 step 50/100 日志里曾为 **0**（该 sync step 的 microbatch 很可能是纯 IAD A/B，没有 Yes/No token）。**不等于 loss 路径挂了**；要以 CoTM batch 上的 `n_no>0` 或训后评测为准。

训完后预期产物：

```text
LoRA:    checkpoints/..._cotmv3_tnr/
Merged:  checkpoints/merged_llava_qwen_triad_9ROI_cotmv3_tnr/   # 需手动 merge
```

Merge 参考既有：

```bash
# 按仓库现有 merge_lora_weights.py 用法，指向 tnr LoRA 目录
python merge_lora_weights.py ...
```

---

## 6. 下一会话建议（按优先级）

### P0 — 等训完 → merge → 评测

```bash
conda activate triad
cd /home/yz/code/Triad-main

# 1) 确认 ckpt 写完（应有 adapter_model / trainer_state 等，不只 runs/）
ls checkpoints/llava-siglip-so400m-onevision-qwen2-7b--Triad_llava_ov_lora_tnr_1epoch_4x8x4_1e-5_9ROI_cotmv3_tnr/

# 2) merge（沿用仓库 merge 脚本）

# 3) 全量 v0
bash eval_0shot_path.sh checkpoints/merged_llava_qwen_triad_9ROI_cotmv3_tnr

# 4) 对照 vis_ori：看 mean TNR、pill/bottle/transistor；rec 允许小幅下降
```

验收 KPI：**TNR ↑**（尤其 pill/bottle）；决策探针 `--include-good` 看 gt=0 的 yes_rate 是否下降。

### P1 — 若 TNR 几乎不动

可能原因与对策：

| 怀疑 | 动作 |
|------|------|
| 加权几乎打不到（多数 step 是 IAD A/B） | 提高 CoTM 占比 / 降 IAD；或给 A/B 做「按 prompt 映射到 normal/anomaly」再加权 |
| No×3 不够 | `decision_token_no_weight` → 4～5 再短训 |
| 仅数据上采样不够 | 难正常 hard mining（**训练域**内挖，仍勿用 musc test FP） |

### P2 — 仍不够再考虑

- 不重训：Yes 概率阈值校准（半天对照）
- `DECISION_DIAG` 的主图 crop 探针（P1）；crop 无效则别上双阶段

---

## 7. 关键路径速查

| 路径 | 用途 |
|------|------|
| `docs/DECISION_DIAG_HANDOFF.md` | 决策探针结论：停 attn-aux，转 TNR |
| `docs/TNR_TRAIN_HANDOFF.md` | **本文：TNR 训练实现与状态** |
| `finetune_ov_0shot.sh` | 当前 TNR 训练入口 |
| `movable/train_tnr_cotmv3.yaml` | 数据配方 |
| `LLaVA-NeXT/llava/train/decision_token_loss.py` | 加权 CE |
| `LLaVA-NeXT/llava/train/llava_trainer.py` | `dpo_forward` + decision loss 挂接 |
| `checkpoints/..._cotmv3_vis_ori/` | 评测对照基线 |
| `checkpoints/..._cotmv3_tnr/` | 本会话训练输出（进行中） |
| `tmp_log/llava_train_20260917_203536.log` | 当前训练日志 |

---

## 8. 一句话现状

**已实现并启动「关 attn-aux + No/normal 上采样 + 末位 Yes/No CE 加权」的 TNR 训练；多模态 labels 对齐与 int denom 两个坑已修。下一会话：等训完 → merge → musc_v0 看 TNR（pill/bottle），不要再碰注意力 aux。**
