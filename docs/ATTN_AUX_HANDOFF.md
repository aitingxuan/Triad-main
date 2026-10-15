# Triad 注意力辅助损失 / 证据回看 — 会话交接文档

> 给下一个 Cursor 会话用。读完本文即可接续，无需翻完整旧对话。

**日期：** 2026-09-16（aux 重写：两桶对比 + lm_head 挂图）  
**仓库：** `/home/yz/code/Triad-main`  
**环境：** `conda activate triad`

---

## 1. 在做什么

目标：增强 Triad（LLaVA-NeXT + anyres + ROI）在 CoTM 生成时对**图像/视觉 token**的注意力，训练侧养成「看图」习惯。

已否定/弱化：

- 在 CoTM 上编造 `<evidence_roi>` 数字 ID → **不要当主训数据**
- InstructIAD bbox 当缺陷 ROI（分布不像 GT）
- **旧版全序列 softmax 激活代理**：训练日志里 `m_vis≡0`、`m_prefix≡1`，且与 CE 图断开 → 评测与无 aux 几乎无差

更靠谱方向：真 mask→ROI、推理双阶段、**注意力辅助损失（已重写）**、decode-time AttnReal（未做）。

---

## 2. 当前实现：两桶 cosine 对比 aux

### 公式（概念）

在 **lm_head 输入 hidden**（与 CE 同一张量）上，对 answer query 与 key 做 cosine，再对视觉 / 文本前缀两桶做 log-mean-exp：

\[
s_{\mathrm{vis}}=\mathrm{LME}(\hat q\cdot\hat k_{\mathrm{vis}}/T),\quad
s_{\mathrm{pre}}=\mathrm{LME}(\hat q\cdot\hat k_{\mathrm{pre}}/T)
\]

\[
\mathcal{L}_{\mathrm{attn}}=\mathrm{softplus}(s_{\mathrm{pre}}-s_{\mathrm{vis}}+m)
+\mathrm{ReLU}(\tau_v-p_{\mathrm{vis}})^2+\mathrm{ReLU}(p_{\mathrm{pre}}-\tau_p)^2
\]

\[
\mathcal{L}=\mathcal{L}_{\mathrm{CE}}+\alpha\cdot\mathcal{L}_{\mathrm{attn}}
\]

- \(p\) 为 \([s_{\mathrm{vis}},s_{\mathrm{pre}}]\) 的两桶 softmax（监控用；**不是**对 8k token 的 softmax）
- prefix 桶默认丢掉前 `sink_tokens` 个 BOS/system sink
- **不再** `× stopgrad(CE)`（旧写法在 aux 恒定时只相当于把 CE 乘 1.01）
- 短答（`< min_answer_tokens`，多为 InstructIAD）跳过 aux
- query 偏尾部（evidence / `<answer>`）；且必须在 `vis_end` 之后

### 为何挂 lm_head

Gradient checkpointing 下对 layer 的 pre-hook 常吃到 **no_grad** 激活，aux 变成常数。  
`lm_head` 的输入与 CE 共用，**保证 `aux.requires_grad`**。

ZeRO-3 约束不变：不重算 `q_proj/k_proj`、不读分片 weight。

### 默认超参（`finetune_ov_0shot.sh`）

```text
α=0.1  τ_vis=0.45  τ_prefix=0.55  margin=0.1  T=2.0
sink_tokens=4  min_answer_tokens=32  last_n_layers=1  max_queries=8
```

`attn_aux/m_vis`、`m_prefix` 现为**两桶质量**，健康时不应长期卡在 `0.0 / 1.0`。

---

## 3. 改动文件清单

| 路径 | 作用 |
|------|------|
| `LLaVA-NeXT/llava/train/attn_aux_loss.py` | 两桶 cosine LME + softplus |
| `LLaVA-NeXT/llava/train/llava_trainer.py` | lm_head hook；`CE + α·aux` |
| `LLaVA-NeXT/llava/train/train.py` | 新 `attn_aux_*` 参数 |
| `finetune_ov_0shot.sh` | 打开 aux 与新默认 |
| `LLaVA-NeXT/llava/model/llava_arch.py` | SPAM 默认关闭 |

训练数据仍为：`CoTM_v3.json + InstructIAD.json`

---

## 4. 怎么跑

```bash
conda activate triad
cd /home/yz/code/Triad-main
bash finetune_ov_0shot.sh
```

训完 merge LoRA 后：

```text
llava_eval/p0_attention_probe.py   # 主验收：W2_vision↑、text_prefix↓
# musc_v0 看 TNR，不要只看已饱和的 rec/acc
```

---

## 5. 下一会话建议

1. 确认新 run：`attn_aux/m_vis` 离开 0，且 `loss/grad_norm` 正常  
2. merge → P0 对比 `cotmv3_vis_ori`  
3. 若两桶质量升但 P0 真 attention 仍低：再上 decode-time AttnReal 或真 QK（hook 当次 q/k 输出，仍禁重算 Linear）  
4. 不要默认恢复 SPAM / 假 evidence_roi

---

## 6. 一句话现状

**旧 aux 因全序列 softmax 饱和 + checkpoint 断图无效；已改为 lm_head 上的视觉/前缀两桶对比，可重新开训。用 `m_vis` 是否离开 0 和 P0 probe 判断是否生效。**
