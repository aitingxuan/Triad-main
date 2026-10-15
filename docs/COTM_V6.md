# CoTM_v6 — 三分混合（真图，无贴伪）

> 替代「工艺贴伪 v5」作为当前推荐训练语料。  
> 相关：`docs/PROCESS_AWARE_PSEUDO.md`（贴伪方案，本版不采用）、`movable/process_step_map.json`。

**日期：** 2026-09-22  
**仓库：** `/home/yz/code/Triad-main`

---

## 0. 目标

在**不贴伪异常**的前提下，用三种回答风格混合训练：

| 档 | 风格 | 作用 | 来源 |
|----|------|------|------|
| ① | 只答 A/B | 判断 / 决策 | `CoTM.json` 原样保留 |
| ② | 步骤 + 位置 + 类型 | 工艺对齐 | 果汁**真实 NG** 新写一条 |
| ③ | checklist 逐步 Defective | 细推理 | `CoTM.json` 原样保留 |

非果汁（textual / Real-IAD 等）原样保留。不加 `images_process_pseudo`。

---

## 1. 为什么不用贴伪

贴伪图接缝明显，模型容易学痕迹而不是缺陷语义。果汁真实异常已覆盖主要类型；缺的是 **gpt 把缺陷挂到工艺步骤**，不是再造假图。

---

## 2. 缺陷 → 步骤（全类型）

表在 `movable/process_step_map.json`。三种果汁步号不同，**不要共用**：

| 产品 | 灌装相关 | 标签相关 | 汁液外观 |
|------|----------|----------|----------|
| banana（6 步） | Step 5 Filling | Step 6 Label Design and Application | Step 2 Juicing / Step 3 Pasteurization |
| cherry（4 步） | Step 4 Filling the Bottle | Step 3 Label Application | Step 1 Cherry Juice Production |
| orange（4 步） | Step 4 Assembly（灌装） | Step 3 Label Production（印刷类）或 Step 4 Assembly（贴附类） | Step 1 Juice Production |

orange 刻意拆开：缺印 / 错标 / 破损印刷 → Step 3；缺标 / 错位 / 歪斜贴附 → Step 4。

几乎所有 `defect_type`（含 `top_label_skewed_logo`、`missing_two_label`、`invalid_juice_type`、`discolor_juice` 等）都有映射；`location` / `type_text` 优先复用同图 `CoTM_v3` 的 tag，否则用表默认。

---

## 3. 怎么造 v6

对 `CoTM.json` 里每张果汁 NG 图（以其 **AB** 行为种子，共 203 张）：

- human：沿用 AB 的工艺提示 + 提问（不改图）
- gpt：`<think>…Step k…</think><location>…</location><type>…</type><answer>Yes</answer>`
- 元数据：`cotm_style=process_align`

原 AB / checklist 行保留，并标 `cotm_style`。同一张图因此可有：AB（①）+ checklist（③）+ process_align（②）。

```bash
python3 scripts/build_cotm_v6.py --force
```

产物：`movable/CoTM/CoTM_v6.json`（约 1759 + 203）。

训练：`movable/train_cotmv6.yaml` = CoTM_v6 + InstructIAD，`sampling_strategy: all`。  
`finetune_ov_0shot.sh` 指向该 yaml；`decision_token_no/yes_weight=1`，无 TNR 上采样。

---

## 4. 规模（构建后核对）

- ① AB：与 `CoTM.json` 相同（果汁约 406，含 good/NG）
- ③ checklist：同上（果汁约 406）
- ② process_align：203（每张真实 NG 一条 Yes）
- 合成贴伪：0

---

## 5. 成功 / 失败

成功：抽查 process_align 的 step 与产品工艺、defect_type 一致；AB 仍短答；checklist 仍逐步。再报 musc_v0。

失败：

- 步骤号与该产品提示词不一致 → 改 `process_step_map.json` 后重建
- 又加回贴伪图 → 违背本版前提
- 用 No×3 / 正常上采样当主手段 → 回到 TNR，不是本路线
