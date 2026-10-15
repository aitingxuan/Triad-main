# EG-RoI 条件性部件补位（ECPR）

> **视觉：EG-RoI 为主，部件 ROI 为条件性补充。**  
> **文本：interest → step 归因表作工艺反查（非「当前故障步」先验）。**  
> **决策：只在激活集合 \(\mathcal{A}\) 上可学习聚合（非硬 OR、非全槽投票）。**  
>  
> 相对 `docs/PGRB_PROCESS_GROUNDED_REGION_BUS.md`：**不做「每图 SAM 分割并注入全部部件」**；保留 Triad EG-RoI 的稀疏异常导向，仅为逻辑异常 / EG 盲区提供事件驱动补位。  
> 相关：`docs/PGRB_PROCESS_GROUNDED_REGION_BUS.md`、`docs/COTM_V6.md`、`docs/DECISION_DIAG_HANDOFF.md`、`movable/process_interest_graph.json`、`movable/process_step_map.json`。  
> 部件 mask 缓存：`Seg_juice/juice_interest_masks_cotm/`（果汁 CoTM；非每图必挂）。

**日期：** 2026-09-24  
**仓库：** `/home/yz/code/Triad-main`  
**状态：** 推荐主路线（相对全槽 PGRB 更贴现管线与指标）

---

## 0. 一句话

**EG-RoI 回答「哪里像异常」；部件 ROI 只在工艺规则要求且 EG 盖不住时补「验哪个部件」；归因用 interest→step（有细类时用 defect 表）；决策只在激活槽集合上 soft 聚合。**

---

## 1. 与 PGRB / 旧路径的关系

| | 全槽 PGRB | **本方案 ECPR** | 已否定路径 |
|--|-----------|-----------------|------------|
| 主视觉证据 | 每图全部件 SAM crop | **IAD EG-RoI（≤4）** | MusC 分数直接替 Yes/No |
| 部件 ROI | 每图必注 | **规则触发 + 覆盖不足才注** | CoTM 匿名 bbox 当部件 |
| 聚合范围 | 全 interest 的 \(p_r\) | **仅 \(\mathcal{A}=\) EG∪触发部件** | 裸硬 OR / No×3 / TNR 主训 |
| 工艺文本 | 归因字典 | 同左：**反查用，不作当前步先验** | 先选 Step k 再只开该步 ROI |
| 训练语料 | — | 优先 **CoTM_v6（真图）** | 贴伪 v5 作主训 |

保留 PGRB 里仍然正确的部分：

- 因果顺序：**看图发现 → 定位 interest → 反查 Step → Yes/No**（不是先 Step 后看图）
- `interest_to_step` / 细表作归因监督
- soft \(p_r\) + 置信门控 + 可学习聚合（§ 决策）
- 涂 GT 证据槽后 Yes 应显著下降

刻意不做的 PGRB 部分：

- 每图强制扫全套 SAM 部件
- 全槽投票 / 全槽进聚合（易稀释、易 Yes bias）
- 推理时假设已知故障工序

---

## 2. 目标因果链

题面仍是「有没有异常」。工艺全文 = **检查清单 + 归因字典**。

```text
图像 + 全套工艺文本
        │
        ▼
 ① EG 主扫：IAD 专家 → ≤4 异常候选 EG-RoI → 局部 p_eg
        │
        ▼
 ② 规则 / 覆盖判定：
        工艺是否要求校验某部件 r？
        EG 是否已充分覆盖 r？
        │
        ├─ 已覆盖 → 不注入部件 ROI
        └─ 要求校验且覆盖不足 → 注入部件 ROI → p_part
        │
        ▼
 ③ 定位：异常落在哪个 interest（<location> / <type>）
        │
        ▼
 ④ 归因：interest → Step（查表写入 <think>；非先验）
        │
        ▼
 ⑤ 决策：可学习聚合(𝒜) → Yes/No
```

| 错误顺序（禁止） | 正确顺序（本版） |
|------------------|------------------|
| 先选 Step k → 只看该步 ROI | **先 EG 发现 →（按需）补部件 → 定位 → 归因 Step → 决策** |
| 每图挂齐全部件再 OR | **只对激活集合 \(\mathcal{A}\) 打分并聚合** |
| 测试时已知故障步 | 测试时**不知道**哪步坏；靠视觉证据 + 反查 |

已坐实问题（`DECISION_DIAG_HANDOFF.md`）：决策几乎不依赖缺陷像素。本方案要求：**Yes 必须消费 \(\mathcal{A}\) 内局部证据；Step 必须由定位到的 interest 反查得到。**

---

## 3. 视觉端：EG 为主，部件为事件驱动补位

### 3.1 EG-RoI（主证据，每图）

- 来源：Triad 现有路径 — IAD 专家模型（如 MusC）热图 → box，或评测/训练已有的 mask→bbox。
- 配额：与现管线一致，**最多 4 个**异常候选（`max_box_num=4`）。
- 角色：稀疏、异常导向；外观缺陷主路径，尽量不改 vis_ori 行为。
- anyres 全局缩略 / 多尺度：**保留**，作语境，不替代 EG。

### 3.2 部件 ROI（补位，非每图必注）

- 来源：离线闭集 interest mask（果汁：`Seg_juice/juice_interest_masks_cotm/`；`classes`: background / juice / top_label / bottom_label / bottle_body）。
- **不是**「每图 SAM 再跑一遍并全部注入」。
- 激活语义（叙述上）：CoT 推进到「部件状态校验」时；  
  **实现上**：用可复现规则，避免生成式 CoT 漂移导致注槽不稳定（见 §3.3）。

### 3.3 触发条件（训练与推理同一套）

对候选部件 \(r\)，**同时**满足才注入：

1. **规则命中**：当前样本工艺闭集 / checklist 明确要求检查 \(r\)  
   - 果汁：由 `defect_type` 或工艺行映射到 interest（见 §4）  
   - 通用：规则模板中的部件名；无规则则永不触发  
2. **EG 覆盖不足**：对所有 EG 槽 \(i\)，\(\mathrm{IoU}(\mathrm{EG}_i,\,r) < \tau\)（或 EG 中心不在 \(r\) 的 mask 内）  
3. **mask 可用**：缓存中 `usable=true`，质量门通过  

否则 \(r \notin \mathcal{A}\)。

典型该触发的情况（EG 常盲）：

- `missing_two_label` / 双标齐全类逻辑缺陷  
- 标签–汁液颜色一致性等跨槽组合  
- 工艺写明「检查片剂边缘」类、但 EG 只打在纹理噪声上的逻辑/结构错位  

典型不该触发：

- `foreign_object_in_juice` 且 MusC 已盖住汁内异物 → 只用 EG  
- 无工艺、无 interest 缓存的域外样本 → \(\mathcal{A}=\) 仅 EG（自然退化）

### 3.4 槽位配额（钉死）

默认仍在 `max_box_num=4` 内：

1. 先放入 top‑k EG（按专家分数）  
2. 若有触发的部件 ROI：  
   - 有空位 → 追加  
   - 已满 → **替换**与目标部件 IoU 最低（覆盖最差）的一个 EG 槽  
3. 记录元数据：`slot_names` / `slot_sources` ∈ {`eg`, `part`}，供 loss 与探针使用  

不在默认路径把 box 数扩到 8（成本与假阳性双升）。

### 3.5 激活集合

\[
\mathcal{A}
= \{\text{本样本采用的 EG 槽}\}
\cup \{\text{本样本触发的部件槽}\}
\]

局部异常分 **只**对 \(r \in \mathcal{A}\) 计算；未激活部件不进前向、不进聚合。

---

## 4. 文本端：interest → step 反查

### 4.1 原则

- 工艺全文提供步骤语义与 checklist，**不**提供「当前故障步」先验。
- 训练教模型：**先视觉定位异常 interest，再反查 Step**。
- 禁止：先写 Step k 再只开对应区。

### 4.2 表怎么用（果汁分层，避免粗表踩坑）

| 用途 | 表 | 路径 |
|------|-----|------|
| 有 `defect_type` 时的归因 GT | **defect → Step** | `movable/process_step_map.json` |
| 推理默认 / 无细类型 | **interest → Step** | `movable/process_interest_graph.json` → `interest_to_step` |
| 逆索引（可选） | step → interests | 同 graph；**只**用于「该步应查哪些区」的规则提示，不作决策门 |

三种果汁步号**不要共用**一张表（banana 灌装 Step 5 / 贴标 6；cherry、orange 为 4 步体系；orange 印刷 vs 贴附需按 defect 拆开，见 `process_step_map`）。

粗 `interest_to_step`（如一切 `juice`→灌装）**不能**单独当 GT：`invalid_juice_type` / `discolor_juice` 等应对应更早步骤。有细类时以 `process_step_map` 为准。

### 4.3 生成监督模板（process_align，可选收紧）

在 CoTM_v6 的 process_align 风格上体现顺序，不强制推理时真串行跑 K 次：

```text
<think>
EG: ...
Part-check (if triggered): top_label=...
Locate: top_label
Attribute: Step 6 ... (from map)
</think>
<location>...</location>
<type>...</type>
<answer>Yes|No</answer>
```

AB / checklist 档可保持短答或逐步 checklist；sidecar 仍可按规则决定是否注部件槽。

---

## 5. 决策端：可学习聚合（仅 \(\mathcal{A}\)）

沿用 PGRB §6 的容错思想，**缩小求和范围**：

\[
P(\mathrm{Yes}\mid\{p_r\}_{r\in\mathcal{A}})
=\sigma\!\left(
  w_0
  + \sum_{r\in\mathcal{A}} w_r\, g(p_r, c_r)
  + \sum_{k\in\mathrm{pairs}\cap\mathcal{A}} u_k\, g(p_k, c_k)
\right)
\]

- \(p_r \in [0,1]\)：soft 局部异常分；允许 abstain  
- \(c_r\)：置信 / mask 或 EG 质量 / 非 abstain；低置信时 \(g \to 0\)（假阳性买不到票）  
- \(w_r, u_k\)：与 Yes/No **端到端**学  
- **禁止**裸硬 \(\mathrm{Yes}=\bigvee_r \mathrm{anom}(r)\)  
- **禁止**对未激活的 SAM 部件算 \(p_r\)（避免全槽稀释）

跨槽逻辑（双标、tag↔汁液等）：用 **pair 头**，且只在相关槽均属于 \(\mathcal{A}\)（或规则显式要求的 pair）时启用；不要塞进「每槽独立再 OR」。

局部头允许：**本 ROI + 低权重全局缩略**（或邻接已激活槽）；禁止「零上下文极端独立 crop」作为唯一实现。

---

## 6. 训练监督（因果链对齐，非话术背诵）

| 信号 | 含义 | 备注 |
|------|------|------|
| \(\mathcal{L}_{\mathrm{local}}\) | \(p_r\) 与 GT 缺陷区域对齐 | 仅 \(r\in\mathcal{A}\)；GT 落在 EG 或触发部件上 |
| \(\mathcal{L}_{\mathrm{localize}}\) | `<location>` / `<type>` ↔ GT interest | 有部件标注 / defect→interest 映射时 |
| \(\mathcal{L}_{\mathrm{attr}}\) | Step ↔ 归因表 | **仅定位正确时**重惩；细类优先 `process_step_map` |
| \(\mathcal{L}_{\mathrm{agg}}\) | 聚合 ↔ Yes/No | 必须消费局部证据；涂 GT 槽后 Yes 应显著下降 |
| \(\mathcal{L}_{\mathrm{CE}}\) | 原有生成 CE | CoTM_v6 + InstructIAD 等 |

示意：

\[
\mathcal{L}
= \mathcal{L}_{\mathrm{CE}}
+ \lambda_1 \mathcal{L}_{\mathrm{local}}
+ \lambda_2 \mathcal{L}_{\mathrm{localize}}
+ \lambda_3 \mathcal{L}_{\mathrm{attr}}
+ \lambda_4 \mathcal{L}_{\mathrm{agg}}
\]

**不做：** No×3、TNR 上采样主训、贴伪主训、attn-aux 调 α、先 step-gate 再看图、全槽硬 OR。

数据：继续以 `docs/COTM_V6.md` 为主；部件 mask **旁路缓存**，勿污染 CoTM 原 json。无 mask / 无工艺子集：\(\mathcal{A}=\) 仅 EG，辅损自动关掉部件相关项。

---

## 7. 离线产物与建议路径

### 7.1 已有

| 路径 | 作用 |
|------|------|
| `Seg_juice/juice_interest_masks_cotm/` | 果汁 CoTM interest mask + `index.jsonl`（按需查询，非每图注入） |
| `movable/process_interest_graph.json` | interests、`interest_to_step`、pair_rules |
| `movable/process_step_map.json` | defect→Step 细表（归因 GT 优先） |
| Triad `anyres_max_9_randomroi` | EG / mask / bbox → ≤4 ROI crop |

### 7.2 建议新增（不覆盖 PGRB 原文）

| 路径 | 作用 |
|------|------|
| `movable/ecpr_trigger_rules.json` | 规则：哪些工艺行 / defect → 候选 interest；默认 \(\tau\) |
| `scripts/build_ecpr_sidecar.py` | 每样本：EG boxes（若有）+ 是否触发部件 + 最终 slot 列表 + GT interest/step |
| `movable/juice_ecpr_sidecar.jsonl` | 上述旁路索引（勿写入 CoTM 原字段除非必要） |
| `LLaVA-NeXT/llava/train/ecpr_slots.py` | 组 \(\mathcal{A}\)、配额替换、写出 `slot_sources` |
| `llava_eval/ecpr_causal_probe.py` | 涂 EG/部件 GT 槽、扰动工艺文、对比「强制全部件」消融 |

`process_interest_graph.json` 的 `mask_root` 可指向 `Seg_juice/juice_interest_masks_cotm`；增加 `defect_to_interests` 供触发与定位 GT。粗 `interest_to_step` 仅作回退。

---

## 8. 验收（按因果链，不按话术）

| 探针 | 期望 |
|------|------|
| 涂掉 GT 所在证据槽（EG 或触发部件） | \(P(\mathrm{Yes})\) / Yes→No 下降 **明显高于** 涂背景或未激活部件 |
| 仅抬高某个正常激活槽的假阳性 | 聚合后不应轻易全局 Yes |
| 破坏工艺文本、图与 EG 不变 | 定位与 Yes 基本稳定；最多归因话术变差 |
| 错改归因表监督 | Step 错，但定位 interest 仍应对 |
| 强迫跳过本应触发的部件补位（逻辑缺陷子集） | Yes / 定位应变差（证明补位有用） |
| 对外观缺陷子集关闭全部件触发 | 指标应接近现 EG 基线（证明未伤害主路径） |

再报 musc_v0 acc/rec/tnr 作迁移参考。  
**不是**成功标准：think 里单独写出漂亮 Step 句，而决策不依赖任何局部视觉证据。

---

## 9. 分周落地

| 周 | 内容 | 通过门槛 |
|----|------|----------|
| Week 0 | 固化触发规则 + \(\tau\)；sidecar；抽检 overlay（触发 vs 不触发） | 逻辑缺陷该触发、异物已盖住不触发；路径对齐 CoTM |
| Week 1 | train/eval 接入：默认 EG；触发时替换/追加部件槽；记录 `slot_sources` | 前向稳定；`max_box_num=4` |
| Week 2 | 辅损：local + localize + attr（定位对才重惩）+ agg；1 epoch 对齐 vis_ori 配方 | 涂 GT 证据槽降 Yes |
| Week 3 | pair 头（仅相关激活槽）；消融：无补位 / 强制全部件 / 仅 EG | 补位增益集中在逻辑子集；全槽不优于 ECPR |

### 9.1 Week 2 怎么开训（已接线）

实现：

- 旁路监督字段：`scripts/build_ecpr_sidecar.py` → `movable/juice_ecpr_sidecar.jsonl`（`slot_positive` / `gt_interests` / `gt_step*`）
- 组槽：`LLaVA-NeXT/llava/train/ecpr_slots.py`
- 辅损：`LLaVA-NeXT/llava/train/ecpr_aux_loss.py`（token 重加权 CE + soft \(p_r\) 聚合 BCE）
- Trainer：`ecpr_aux_loss=True` 时走 chain CE，并把 `ecpr_local_agg_head` 挂到模型上优化

```bash
# 重建 sidecar（改规则 / step_map 后）
python3 scripts/build_ecpr_sidecar.py

# Week2：slots + localize/attr/agg + local-agg（ECPR_AUX 默认随 ENABLE=1）
ECPR_ENABLE=1 ECPR_AUX=1 bash finetune_ov_0shot.sh

# 仅 slots、不开辅损（对照 Week1）
ECPR_ENABLE=1 ECPR_AUX=0 bash finetune_ov_0shot.sh
```

默认 \(\lambda\)：`localize=attr=agg=0.5`，`local=0.2`，`attr_gate_scale=0.25`。  
日志键：`ecpr/chain_loss`、`ecpr/n_*`、`ecpr/local_loss`、`ecpr/local_acc`。  
**注意：** `L_local` 用 sidecar 离线 `slot_positive` 当 \(p_r\)，梯度不回灌 VLM 视觉塔；真正改 VLM 行为的是 localize/attr/agg 的 token CE。科学验收仍以涂 GT 证据槽降 Yes 为准，不以 musc_v0 单独抬分为准。

---

## 10. 成功 / 失败

**成功：**

- 行为符合「EG 发现 → 按需部件校验 → 定位 → 归因 → 决策」
- 外观缺陷不伤 EG 基线；逻辑 / EG 盲区上补位可测
- 涂证据槽能压 Yes；单槽噪点不轻易全局炸
- Step 随定位走，不随「先验步号」走

**失败：**

- 又变回每图强制全 SAM 部件注入
- 落地成裸硬 OR 或全槽投票
- 先写 Step 再找区，或决策不依赖 \(\mathcal{A}\)
- 用贴伪 / No×3 / 纯话术挂步当主手段
- 无规则域外样本仍强行注果汁部件

---

## 11. 与 PGRB 文档的阅读顺序

1. 本文 **ECPR**：当前推荐实现口径  
2. `docs/PGRB_PROCESS_GROUNDED_REGION_BUS.md`：归因方向、soft 聚合与硬问题（§6）的详尽讨论；其中「每图全部件总线」改为本文的条件激活即可  
3. `docs/COTM_V6.md`：训练语料  
4. `docs/DECISION_DIAG_HANDOFF.md`：为何必须做涂抹因果探针  

---

## 12. 一句话（复述）

**工艺用于归因，EG 用于发现，部件 ROI 用于补 EG 盲区；训练对齐这条链。聚合只在激活集合上做，不做全槽 PGRB 扫描。**
