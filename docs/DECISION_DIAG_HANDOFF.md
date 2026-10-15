# Triad 决策诊断 / attn-aux 结论 — 会话交接文档

> 给下一个 Cursor 会话用。读完本文即可接续，无需翻完整旧对话。  
> 相关旧文档：`docs/ATTN_AUX_HANDOFF.md`（注意力 aux 实现细节，仍有效但**路线已降级**）。

**日期：** 2026-09-17  
**仓库：** `/home/yz/code/Triad-main`  
**环境：** `conda activate triad`

---

## 0. 用户真实目标（重要）

**最终要的是：提高异常二分类（Yes/No）准确率，尤其改善 TNR / 降 Yes bias。**  
「多看图 / 注意力」只是曾设想的手段，**不是 KPI**。  
用户对 P0 attention probe **无兴趣**；不要把下一会话拉回刷 P0。

---

## 1. 本会话做了什么

### 1.1 重写了训练侧 attn aux（attnauxv2）

相对第一版（全序列 softmax 激活代理，`m_vis≡0` 无效）：

| 改动 | 说明 |
|------|------|
| 挂图 | `lm_head` 输入 hook（与 CE 同图），避免 GC 断梯度 |
| 损失 | 视觉 vs 文本前缀 **两桶 cosine + LME + softplus** |
| 组合 | `CE + α·aux`，去掉 `× stopgrad(CE)` |
| 样本 | answer `< min_answer_tokens(32)` 跳过（少打 InstructIAD） |
| query | 偏尾部，且要求在 `vis_end` 之后 |

**改动文件：**

- `LLaVA-NeXT/llava/train/attn_aux_loss.py`
- `LLaVA-NeXT/llava/train/llava_trainer.py`
- `LLaVA-NeXT/llava/train/train.py`（新超参）
- `finetune_ov_0shot.sh`（默认 α=0.1 等）
- `docs/ATTN_AUX_HANDOFF.md`（实现说明已更新）

**产物 ckpt：**

- LoRA：`checkpoints/llava-siglip-so400m-onevision-qwen2-7b--Triad_llava_ov_lora_attnaux_1epoch_4x8x4_1e-5_9ROI_cotmv3_attnauxv2`
- Merged：`checkpoints/merged_llava_qwen_triad_9ROI_cotmv3_attnauxv2`
- 对照基线 merged：`checkpoints/merged_llava_qwen_triad_9ROI_cotmv3_vis_ori`
- 更早无效 aux：`.../cotmv3_attnaux`（v1，可忽略）

### 1.2 评测与诊断

1. **全量 musc_v0**：attnauxv2 ≈ vis_ori（acc/TNR 无实质提升）  
2. **决策因果对照**（本会话最终定的实验，已跑完）：  
   `llava_eval/causal_a1_a3_probe.py` + `config:v0` + `--include-good` + n=48

结果文件：

```text
LLM_judge_output/decision_a1a3_vis_ori_v0_n48.json
LLM_judge_output/decision_a1a3_vis_attnauxv2_v0_n48.json
```

---

## 2. 关键结论（已坐实，勿重复争论）

### 2.1 attnauxv2 没有改变二分类决策

同 48 题、v0、同 seed：

| 指标 | vis_ori | attnauxv2 |
|------|---------|-----------|
| baseline acc | 0.833 | 0.812 |
| baseline 答案一致率 | — | **47/48 ≈ 97.9%** |
| gt/bg_occlude、roi_* | 几乎相同 | 几乎相同（一致率 98–100%） |

全量 csv 同样：ori acc≈0.916 / tnr≈0.745；attnauxv2 acc≈0.914 / tnr≈0.741。

**结论：停掉「靠 attn aux / 多看图训练」提二分类这条线。不要再为 aux 调 α、重训。**

### 2.2 当前模型 Yes 决策几乎不依赖缺陷像素 / ROI

在 **vis_ori** 的决策探针上（更重要）：

- 缺陷样本 baseline：**yes_rate = 1.0**（24/24）  
- **gt_occlude flip = 0**（涂掉 GT 缺陷，答案不变，仍全 Yes）  
- **causal_gap(gt − bg) ≈ −0.04**（涂缺陷并不比涂背景更能掀答案）  
- **roi_gt / roi_musc / roi_none / roi_random** 几乎不改变答案  

正常样本：gt=0 时 yes_rate≈0.33，acc≈0.67 → **TNR 弱、有 Yes bias**。

**解释：** 瓶颈是「缺陷图 → 倾向说 Yes」的先验 + 正常类误杀，不是「图像 attention mass 不够」。  
ROI token 对 v0 决策是弱因果；再灌注意力损失也不会自动变成「看见缺陷才说 Yes」。

### 2.3 训练侧旁证（次要）

- attnauxv2 的 `trainer_state` **没有** `attn_aux/*` 日志（可能被 `min_answer_tokens` + `log_every=50` 错过），但 mean train loss≈1.67（ori≈0.73），暗示 aux 可能进了 loss。  
- 即便进了 loss，优化的是 **lm_head 前 hidden 两桶 cosine**，与 **v0 Yes/No 决策**、与真 QK attention 都可不一致。  
- 对用户目标而言：**决策探针 + 全量 v0 已足够判死刑**，不必再为「aux 有没有挂上图」纠缠。

---

## 3. 已否定 / 不要再做

- 假 `<evidence_roi>` ID / InstructIAD 假框当 GT  
- 默认恢复 SPAM  
- 继续训 attn aux、扫 α、以 P0 `W2_vision` 当成功标准  
- 用 musc_v0 已饱和的 **rec≈98%** 当「看图是否变好」的指标  

---

## 4. 下一会话建议（按优先级）

### P0 — 打 Yes bias / 抬 TNR（与「看图」解耦）

方向示例（择一先做小实验）：

- 训练：加重难正常样本 / 类权重 / 对 answer 的 Yes 惩罚  
- 或推理：对 Yes 概率做阈值校准（先不重训）  
- 验收：**全量 musc_v0 的 TNR、各类 bottle/pill/transistor**；以及决策探针里 gt=0 的 yes_rate  

### P1 — 验证「主图 crop 喂证据」（不重训）

现有 A3 只改 ROI token，已被证伪为弱因果。  
若仍想走「证据进输入」：加条件 **GT/MusC 外接框 crop 成主图再问 v0**，看 TNR/acc 是否动。

- crop 明显更好 → 上双阶段（MusC→crop→VLM）  
- crop 也不更好 → 继续 P0（先验/数据），别做定位融合  

### P2 — 仅当 P1 证明 crop 有用后

再考虑训练侧「强制依据可见缺陷作答」或 decode-time 干预；验收仍用 **acc/TNR/遮挡 flip**，不用 P0。

---

## 5. 复现命令（决策对照）

```bash
conda activate triad
cd /home/yz/code/Triad-main

CUDA_VISIBLE_DEVICES=0 python llava_eval/causal_a1_a3_probe.py \
  --model-path checkpoints/merged_llava_qwen_triad_9ROI_cotmv3_vis_ori \
  --data_path evaluation/mvtec \
  --from question_musc.jsonl \
  --text_src config:v0 \
  --mvtec-root data/MMAD/MVTec-AD \
  --num-samples 48 \
  --include-good \
  --seed 0 \
  --max_new_tokens 64 \
  --out LLM_judge_output/decision_a1a3_vis_ori_v0_n48.json
```

全量 v0：

```bash
bash eval_0shot_path.sh checkpoints/merged_llava_qwen_triad_9ROI_cotmv3_vis_ori
```

---

## 6. 关键路径速查

| 路径 | 用途 |
|------|------|
| `docs/ATTN_AUX_HANDOFF.md` | aux 实现 / ZeRO-3 坑（历史） |
| `docs/DECISION_DIAG_HANDOFF.md` | **本文：决策结论与下一步** |
| `docs/TNR_TRAIN_HANDOFF.md` | **后续会话：已落地的 TNR/降 Yes bias 训练（关 aux + 数据上采样 + decision-token CE）** |
| `llava_eval/causal_a1_a3_probe.py` | A1 遮挡 + A3 ROI 决策探针 |
| `llava_eval/causal_probe_common.py` | 共用工具 |
| `LLM_judge_output/decision_a1a3_vis_*_v0_n48.json` | 已跑完的对照结果 |
| `checkpoints/merged_llava_qwen_triad_9ROI_cotmv3_vis_ori/` | 当前主基线 |
| `checkpoints/merged_llava_qwen_triad_9ROI_cotmv3_attnauxv2/` | auxv2（决策上≈ori） |

---

## 7. 一句话现状

**attn aux（含 v2）未能改变 v0 二分类；因果探针表明缺陷像素与 ROI 对 Yes 决策几乎无因果。下一会话应转向降 Yes bias / 抬 TNR，或用不重训的主图 crop 验证「喂证据」是否真有用——不要再训注意力。**
