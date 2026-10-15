# 工艺对齐的伪异常 — 方案

> **训练优先改用 CoTM_v6（真异常 + 工艺对齐文本，无贴伪）。** 见 `docs/COTM_V6.md`。  
> 本文保留贴伪方案备忘；效果不佳时不要用贴伪图开训。  
> 不要回到 attn-aux，也不要再做「加重 No / 压低 Yes」的 TNR。  
> 相关：`docs/DECISION_DIAG_HANDOFF.md`、`docs/TNR_TRAIN_HANDOFF.md`。

**日期：** 2026-09-22（修订：去掉错部件对照；指向 v6）  
**仓库：** `/home/yz/code/Triad-main`

---

## 0. 目标

让伪异常的**视觉改动**与提示词里的**工艺步骤 / 位置 / 类型**对齐，用来加强「看见的缺陷 ↔ 文本描述」的对应。  
不是靠少说 Yes，也不是靠认出随机贴图痕迹。

题面仍是「产品有没有异常」。因此：

- 原正常图 → No（沿用数据里已有样本，不复制）
- 按闭集表改在对应部件上 → Yes，且 `<think>` / `<location>` / `<type>` 与该行一致

**不做**「同一种改动落到别的部件上标 No」。错部件图肉眼仍是缺陷，标 No 与「有没有异常」题面冲突。

验收：肉眼看生成图，改动落在文本写的部件内，步骤号与类型表一致。再视情况报 musc_v0 的 acc / rec / tnr，只作迁移参考。

---

## 1. 不做的事

| 做法 | 为什么不做 |
|------|------------|
| 决策 token 上 No×3、上采样正常样本（TNR） | 只改判分门槛。全量 TNR 能从 0.745 到约 0.92，但涂掉缺陷几乎不翻答案 |
| 缺陷涂成均值色再标 No（CF） | 涂色块不像正常件。模型把原图和涂后图分开背即可，决策仍不依赖缺陷像素 |
| 全图随机 CutPaste / CutPaste-Scar | 无上下文的伪异常。位置和类型与工艺步骤无关，模型学接缝 |
| 只用 CoTM 的 `bbox` 当粘贴位置 | 正常图和异常图都是若干大小相近的框，不是「上标签 / 果汁」部件 |
| 在 Real-IAD / InstructIAD 上自动对齐 | 工艺是笼统段落，没有步骤号，多数没有可用的部件框 |
| 错部件对照标 No | 题面是有无异常；错部件仍是异常。监督目标改为「视觉 ↔ 步骤/位置/类型对齐」，不靠错部件压接缝 |

已有数字（musc_v0，1725）：

| | acc | rec | tnr |
|---|---:|---:|---:|
| vis_ori | 0.916 | 0.979 | 0.745 |
| cotmv3_tnr | 0.926 | 0.929 | 0.919 |
| cotmv4_cf | 0.927 | 0.926 | 0.929 |

cf 与 tnr 的遮挡探针相同：涂缺陷翻转约 0.042，不高于涂背景。

---

## 2. 可行性边界

CoTM_v3 共 1759 条。带「工艺」字样的不只有三种果汁，但能拿来贴图的仍然只有有图像、有编号步骤、并且缺陷类型能落到部件上的样本。

| 子集 | 条数 | 有图 | 工艺形式 | 能否贴 |
|------|-----:|------|----------|--------|
| LOCO 三种果汁 | 812 | 有 | 编号步骤，但三种果汁步号不一致 | 能。还缺部件框 |
| Real-IAD | 356 | 有 | 一段散文，列出可能缺陷，没有步骤号 | 不能。对不上某一步 |
| `textual_v4` 的 cable / pill / T-nut | 562 | 无 | 编号步骤（电缆 7 步、药片 6 步、T 型螺母 8 步），type1 与 type2 各一份 | 不能贴。没有图像 |
| `PP2Checklist` | 约 20 | 无 | 每类一条编号工艺，含 bottle、screw、pill 等 | 不能贴。是模板，没有图像 |
| `ND2Checklist` | 15 | 无 | 外观描述，不是编号工艺 | 不能 |

三种果汁的步号不能共用一张表：

- banana：6 步。灌装是第 5 步，贴标是第 6 步
- cherry：4 步。贴标是第 3 步，灌装是第 4 步
- orange：4 步。标签是第 3 步，装配是第 4 步

果汁这边已经有的：`defect_type`，回答里的 `<type>`、`<location>`，以及约 400 张正常图。没有的是部件掩码。`movable/images/mvtec_loco_bottle_annotated/` 下只有按类型分的图像。json 里的 `bbox` 仍不能当部件位置。

不要把 `PP2Checklist` / `textual_v4` 的工艺文本配到 MVTec 测试图上，那会泄漏评测。Real-IAD 在没有步骤号和部件框之前不要生成。

---

## 3. 推荐的伪异常做法

不要用随机 CutPaste 当生成器。编辑方式由闭集表决定，一种缺陷类型只对应一种编辑。泊松融合只用于「往正确区域里放一块异物」，用来去掉硬边，不是默认贴法。

闭集表按产品各写一份，不要共用步号。下表是 banana 的步号。cherry 把贴标写成第 3 步、灌装写成第 4 步；orange 把标签写成第 3 步、装配写成第 4 步。表外的类型不生成。

| 缺陷类型 | banana 步骤 | 部件 | 编辑 |
|----------|-------------|------|------|
| `missing_top_label` | 6 贴标 | 上标签 | 用周围瓶身纹理盖住上标签 |
| `missing_bottom_label` | 6 贴标 | 下标签 | 用周围瓶身纹理盖住下标签 |
| `broken_top_label` | 6 贴标 | 上标签 | 只在上标签内部做细长破损，不超出标签 |
| `too_little_juice` | 5 灌装 | 果汁液面 | 下移液面（避开标签框） |
| `too_much_juice` | 5 灌装 | 果汁液面 | 上移液面（避开标签框） |
| `foreign_object_in_juice` | 5 灌装 | 果汁内部 | 异物块只放进果汁区域（避开标签），边缘软融合 |

`broken_*` 可以用 CutPaste-Scar 的细长条，但位置必须限制在该标签框内，不能在整张图上随机撒。其余行不要用 CutPaste。

部件区域单独标，每张正常图三个区域：上标签、下标签、果汁。标不出的图跳过。不要从 CoTM `bbox` 换算。灌装类编辑时，果汁有效区应与标签框互斥（优先用两标签之间的液带，或肩部液面）。

---

## 4. 怎么造一条样本

对每张用上的正常图，选表中的一行，只生成 **一张** 对齐伪异常：

1. 改在表里的部件上。答案 **Yes**。`<think>` 写该步骤号和表里的类型。`<location>` / `<type>` 与这一行一致。
2. **原图：** 保持数据里已有的 No，不复制、不另造对照 No。

提示词沿用该正常样本里的工艺步骤原文，不要改步骤编号。

伪异常只作「工艺文本 ↔ 视觉」对齐的补充监督，不替代 InstructIAD 里的真缺陷。

---

## 5. 规模

- 只从三种果汁的正常图里抽。全部可用大约 400 张；实现时跳过标不出三个部件的图。
- 每张只做 1 行表、1 张 Yes。不要一张图生成多种缺陷。
- 可留出少量正常图不做生成，供肉眼对照（原图 No vs 合成 Yes）。
- 新增大约数百～一千条量级的 **Yes**，加在原 CoTM + InstructIAD 上。不要再 `repeat` 正常样本。
- 合成 Yes 应明显少于 InstructIAD 里已有的真异常（约 4400），避免假缺陷盖过真缺陷。

---

## 6. 训练

配方对齐 vis_ori，而不是 TNR / CF：

- 数据：原来的 CoTM 与 InstructIAD 全部保留，加上第 4 节的对齐 Yes
- `attn_aux_loss False`
- `decision_token_loss` 可留着，但 `decision_token_no_weight=1`、`decision_token_yes_weight=1`。不要 No×3
- 不要为多张对照图各跑一次前向，也不要在损失里比 logit。ZeRO-3 下按样本分支的额外前向会卡死
- 1 epoch、LoRA、学习率与 `finetune_ov_0shot.sh` 里 vis_ori 相同

开训前先肉眼看 20 张：改动是否在文本写的部件内，步骤号是否和类型表一致。有一组对不上就停。

---

## 7. 实现时要新增的东西

| 路径 | 作用 |
|------|------|
| `scripts/build_process_pseudo.py` | 读正常果汁样本，按闭集表生成对齐 Yes 图像和 json |
| `movable/process_pseudo_map.json` | 第 3 节的闭集表 |
| `movable/CoTM/CoTM_process_pseudo.json` | 生成的对齐 Yes |
| `movable/CoTM/CoTM_v5.json` | `CoTM.json`（果汁 + textual 等混合）+ 对齐 Yes |
| `movable/images_process_pseudo/` | 生成图像，路径相对 `movable/` |
| `movable/train_cotmv5.yaml` | CoTM_v5 + InstructIAD，全部 `sampling_strategy: all` |

部件框：手标放在 `movable/juice_parts/`；未手标的图用 banana 中位框按瓶身中心平移。不要写进 CoTM 原 json。

```bash
# 全量贴伪 + 写出 CoTM_v5（默认每图一种缺陷；加 --all-defects 则每图全表）
python3 scripts/build_process_pseudo.py --full --all-defects
# 然后 bash finetune_ov_0shot.sh（已指向 train_cotmv5.yaml，decision Yes/No 权重均为 1）
```

---

## 8. 成功与失败

成功：合成图上的改动与文本步骤 / 部件 / type 一致；原正常样本仍答 No；再视情况看 musc_v0。

失败：

- 20 张里步骤和部件对不上 → 不要开训
- 合成痕迹像随机 CutPaste、压到错误部件 → 先修编辑与部件框，不要靠加重 No 补
- musc_v0 不动 → 果汁上的逻辑编辑没有迁到 MVTec 外观缺陷，不要把随机 CutPaste 加回来补
