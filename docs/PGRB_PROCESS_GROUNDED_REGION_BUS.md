# Process-Grounded Region Bus（PGRB）— 看图定位再归因工序

> **诉求（训练与推理同一条因果链）：**  
> **看图 → 发现缺陷 → 定位部件 → 结合工艺文本归因工序 → Yes/No。**  
> 测试时**不知道**哪一步坏了；要**遍历区域**找出异常部件，再映射到步骤。  
>  
> **不是**「已知 Step k 再只看对应区」。  
> **不是**贴伪 / v6 查表改写 / No×3 / attn-aux 调 α。  
> 相关：`docs/COTM_V6.md`、`docs/PROCESS_AWARE_PSEUDO.md`、`docs/DECISION_DIAG_HANDOFF.md`。  
> 论文：LogSAD (CVPR 2025) — [arXiv:2503.18325](https://arxiv.org/abs/2503.18325) | [代码](https://github.com/zhang0jhon/LogSAD)。

**日期：** 2026-09-22（修订：区域优先、步骤归因）  
**仓库：** `/home/yz/code/Triad-main`

---

## 0. 目标因果链

题面仍是「有没有异常」。工艺全文是**检查清单 + 归因字典**，不是推理前给定的「当前故障步骤」。

```text
图像 + 全套工艺文本
        │
        ▼
 ① 看图：在可寻址部件 ROI 上扫描（遍历全部兴趣区）
        │
        ▼
 ② 发现：哪些部件异常 / 正常（局部判定）
        │
        ▼
 ③ 定位：异常落在哪个 interest（location / type）
        │
        ▼
 ④ 归因：interest → 工艺 Step（查工艺图，写入 think）
        │
        ▼
 ⑤ 决策：任一关键部件异常 → Yes；全部正常 → No
```

| 错误顺序（已弃） | 正确顺序（本版） |
|------------------|------------------|
| 先选 Step k → 只看该步 ROI → 再答 | **先扫全部 ROI → 定位缺陷 → 再归因 Step → 再答** |
| 测试时假设已知故障工序 | 测试时**不知道**哪步坏；靠看图找出来 |
| 训练教模型「背 Step 话术」 | 训练教模型 **扫区→定位→归因→决策** |

已坐实问题（`DECISION_DIAG_HANDOFF.md`）：决策几乎不依赖缺陷像素。本方案用机制强迫：**Yes 必须由某个异常 ROI 的局部证据触发；Step 必须由该 ROI 反查工艺得到。**

| 不做 | 原因 |
|------|------|
| 整套搬 LogSAD 替 Yes/No | 丢掉 CoT；主线仍是训 LMM 学会这条链 |
| 只把 SAM mask 写进 json 再 SFT | 预处理；模型仍可跳过扫区 |
| 贴伪 / No×3 / 重训 attn-aux | 已否定或证伪 |
| 先 step-gate 再看图 | 因果倒置；测试不可用 |

---

## 1. LogSAD 借什么、不借什么

| 借 | 不借 |
|----|------|
| Interest：SAM+CLIP 把部件变成可寻址区 | 用分数 max 融合**替代** LMM 决策 |
| 对**多个** interest 做核验再汇总（遍历思想） | 仅离线 MoT、无训练 |
| Composition 作独立核验器 | 没有「工序归因」——这是我们相对 LogSAD 的增量 |

相对 LogSAD 的差异化：**interest → process step 归因表** + **训 LMM 按「扫区→定位→归因→决策」生成**。

---

## 2. 方案总览

**PGRB = 区域总线（扫遍）+ 局部判定 + 工艺反查归因 + 聚合决策**

```text
工艺原文 / process_step_map
        │
        ▼
 Process-MoT ──► interest 闭集
                  interest→step 归因表（路由表反向用）
                  composition rules（可选核验）
        │
        ▼
 每图 SAM+CLIP ──► 全部部件 mask → anyres/ROI token
        │
        ├──► 并行各 <roi:*>：soft \(p_r\) + 跨槽 pair（非裸独立）
        ├──► 定位：异常 interest → <location>/<type>
        ├──► 归因：interest→Step，写入 <think>
        └──► 可学习聚合 → <answer> Yes/No（见 §6）
```

---

## 3. 离线：Process-MoT（归因图，不是「当前步」）

### 3.1 输入

- 果汁正常图若干 + 编号工艺原文（步号按产品，勿混用）
- `movable/process_step_map.json`（闭集优先）

### 3.2 产出

1. **Interests（闭集）**  
   `top_label` / `bottom_label` / `juice` / `bottle_body` / `logo` …

2. **Interest → Step 归因表**（推理主用；与旧「Step→ROI 门控」方向相反）

   | 异常 interest | banana 归因 | cherry 归因 | orange 归因（示例） |
   |---------------|-------------|-------------|---------------------|
   | `juice`（液量/异物/变色等） | Step 5 Filling | Step 4 Filling | Step 4 Assembly（灌装） |
   | `top_label` / `bottom_label`（缺/破/歪） | Step 6 Label | Step 3 Label | Step 3 或 4（印刷 vs 贴附拆开） |
   | `logo` / 印刷类 | Step 6 | Step 3 | Step 3 Label Production |

   训练与评测都以 **先定位 interest、再填 Step** 为准。`step→interest` 仅作表的逆索引（例如知道灌装步应检查哪些区），**不**用作「只开一个槽」的决策门。

3. **Composition rules**（可选）  
   `fruit_tag ↔ liquid_color`、双标齐全、液面相对肩部等——用于局部核验，不替代扫区。

### 3.3 落盘

`movable/process_interest_graph.json`：interests、`interest_to_step`、可选 `step_to_interests`（逆索引）、composition rules。

三种果汁步号不要共用一张表。

---

## 4. 每图：实例化全部 ROI（必须可遍历）

1. 闭集 interest → SAM mask（可加背景类；少量手标校准）  
2. 映射到 Triad anyres/ROI → 每个 interest 一段视觉 token  
3. 可选：每槽 CLIP composition / 局部异常分

```json
{
  "interests": {"juice": 0, "top_label": 1, "bottom_label": 2},
  "interest_to_step": {"juice": 5, "top_label": 6, "bottom_label": 6},
  "local": {"juice": {"anom": 0.91, "type": "too_little_juice"}, "top_label": {"anom": 0.05}}
}
```

推理时模型（或机制）应对 **所有** interest 有局部结论，再聚合；不能只抽一个步的槽。

---

## 5. 在线机制：对齐「看图→定位→归因→决策」

### 5.1 层 1 — 可寻址区域总线（支撑「看图 / 扫区」）

```text
<roi:juice> … </roi:juice>
<roi:top_label> … </roi:top_label>
<roi:bottom_label> … </roi:bottom_label>
```

局部判定时 **主证据在本 ROI**，允许受控上下文 / 相关 pair（见 §6.1–6.2）；不是「全局只开一个工艺步」。  
强迫「发现 / 定位」落在真实部件像素上，而不是工艺散文。

### 5.2 层 2 — 遍历局部判定 → 聚合决策（支撑「发现缺陷 → Yes/No」）

- 对每个 interest \(r\)：产出 **soft** 局部异常分 \(p_r\)（及 type）；允许 abstain  
- 全局：由各槽（及跨槽 pair）**可学习聚合**得 Yes/No——**不要**落地成裸硬 OR（见 §6.3）  
- 语义约束：answer 必须消费局部/组合证据，不能绕开扫区直接 Yes；涂掉 GT 缺陷槽后 Yes 应明显下降  

训练：GT 缺陷在 juice → **juice 槽 \(p_r\) 应升高并主导聚合**；禁止纯文本先验。

### 5.3 层 3 — 定位后归因工序（支撑「结合工艺文本归因」）

- 在已定位的异常 interest 上，用 `interest_to_step` + 工艺原文写 Step  
- `<think>` / 步骤号 **依赖定位结果**，不是先写 Step 再找区  
- 监督：location/type 与 GT 部件一致，且 Step 与 `interest_to_step` 一致  

工艺文本的作用：提供步骤语义与归因依据；**不**提供「当前故障步」先验。

### 5.4 层 4 — Composition verifier（可选）

独立 CLIP 核验与 LMM 局部结论对质；高置信冲突再加一致性损失。不当最终 decider。

---

## 6. 实现硬问题（逻辑对 ≠ 能直接 OR 落地）

因果链方向对了，但若把「每槽硬判 0/1 + 全局硬 OR」当真，会立刻撞上可靠性 / 独立性 / 容错性。下面是必须正视的约束与默认折中——**实现时以本节为准，修正 §5.2 的理想化表述**。

### 6.1 局部判定可靠性

| 难点 | 表现 |
|------|------|
| Mask 不准 | SAM 切坏液面/标签；局部头看错区 |
| 单槽缺上下文 | 歪标、液面高低常要对照瓶肩/双标，纯 crop 不够 |
| LMM 本就不擅局部 AD | 你们涂缺陷探针已说明：直接让槽内「讲故事式」判异常很脆 |
| 外部核验器也噪 | CLIP composition / 开词表分时常飘 |

**折中（默认）：**

- 局部输出用 **soft \(p_r \in [0,1]\) + abstain**，不要早变成硬 0/1  
- 局部头允许 **本 ROI + 低权重全局缩略**（或邻接槽），禁止「只看本 crop、零上下文」极端独立  
- 先用 **弱监督**：图像级 Yes + GT interest（有则）拉高对应 \(p_r\)；不要一上来要求每槽都有可靠 CE  
- Week 0 过不了「涂 GT 区 vs 背景」的槽，**不允许**该槽拥有一票否决权

### 6.2 局部独立性（经常不成立）

| 难点 | 表现 |
|------|------|
| 跨槽逻辑缺陷 | `tag ↔ liquid color`、双标齐全，**必须**联合看两区 |
| Mask 重叠 / 互斥 | 汁液带与标签争像素；切分误差相关 |
| 联合事件 | `missing_two_label` 不是两个独立伯努利简单加 |

**折中（默认）：**

- 缺陷分两类：**单槽结构类**（破损、异物在汁内）vs **跨槽组合类**（标签-汁液一致、双标）  
- 单槽：主用局部 \(p_r\)  
- 跨槽：显式 **pair / composition 头**（只看相关 ROI 集合），**不要**硬塞进「每槽独立再 OR」  
- 「局部门控」改为：**主证据在目标 ROI，允许只读相关槽**；不是数学上的条件独立

### 6.3 聚合容错性

硬 \(\mathrm{Yes}=\bigvee_r \mathrm{anom}(r)\) 的问题：

- 任一槽假阳性 → **系统性 Yes bias**（与现有病同构）  
- 提高阈值又漏检；槽数越多，OR 越炸  

**折中（默认）——可学习聚合，不是手写 OR：**

\[
P(\mathrm{Yes}\mid\{p_r\})
=\sigma\!\left(w_0+\sum_r w_r\,g(p_r,c_r)+\sum_{k\in\mathrm{pairs}}u_k\,g(p_k,c_k)\right)
\]

- \(c_r\)：该槽置信 / mask 质量 / 非 abstain；\(g\) 在低置信时接近 0（**假阳性买不到票**）  
- \(w_r,u_k\) 与最终 Yes/No **端到端**学，不手工 OR  
- 可选 noisy-OR 参数化，但必须带 **每槽 FP 先验** 或温度，禁止裸 OR  
- 训练仍要求：涂掉 GT 缺陷槽后 \(P(\mathrm{Yes})\) 明显下降（容错 ≠ 允许无视证据）

### 6.4 其它硬问题（简表）

| 问题 | 风险 | 折中 |
|------|------|------|
| **归因歧义** | 一症多步、一步多区 | 允许多候选 Step；主 KPI 仍是 Yes/No + 定位；归因作辅损 |
| **级联错误** | 定位错 → 步必错 | 分项验收：定位探针与归因探针分开；定位错时不猛罚 Step |
| **槽监督稀疏** | 多数图只有图像级标签 | 用「GT 缺陷槽 multi-label」+ 其余槽只作 bag 约束；不要假造每槽硬标签 |
| **生成式扫区成本** | 逐槽长 CoT 贵且不稳 | 优先 **并行局部打分 + 一次聚合生成**；checklist 文本作监督模板，不强制推理时真串行跑 K 次 |
| **域外无工艺/无 interest** | musc_v0 等 | 总线可退化：无表时仅全局+通用 ROI；工艺归因只在有图的子集启用 |
| **Mask 失败传播** | 坏 mask → 坏局部 → 坏聚合 | mask 质量门控；失败则回退整图路径并打标，避免静默错判 |

### 6.5 实现原则（压风险）

1. **因果链保留，模块可靠性降级实现**：扫区→定位→归因→决策仍是监督与探针语义，不要求每层都是硬符号电路。  
2. **先并行 soft 局部 + 可学习聚合，再考虑强符号 OR。**  
3. **跨槽逻辑单独建模**，不要装进独立槽假设。  
4. **能投票的槽必须先通过可靠性门槛**（分割+涂抹探针）。  
5. **归因可错，定位与决策不能一起漂**——损失与验收解耦。

---

## 7. 训练在教什么

训练目标 = 学会同一条链，而不是背答案话术；**实现按 §6 降级，不按裸 OR。**

| 阶段 | 模型应学会 | 监督信号 |
|------|------------|----------|
| 看图扫区 | 相关 interest ROI 被真正看过 | 局部/定位时 attn 落在对应 `<roi:*>`（及允许的 pair） |
| 发现缺陷 | 提高对的 \(p_r\)、压低假阳性槽 | soft 局部 + 图像级 Yes；有 GT interest 则对齐 |
| 定位 | 指出异常部件 | `<location>` / `<type>` 与 GT interest 一致 |
| 归因工序 | 由部件反查 Step | Step 与 `interest_to_step` 一致（**仅当定位正确时**重惩） |
| 决策 | 由局部/组合证据聚合 | 可学习聚合 ↔ Yes/No；涂 GT 槽应显著降 Yes |

损失示意：

\[
\mathcal{L}=\mathcal{L}_{\mathrm{CE}}
+\lambda_1\mathcal{L}_{\mathrm{local\text{-}roi}}
+\lambda_2\mathcal{L}_{\mathrm{localize}}
+\lambda_3\mathcal{L}_{\mathrm{attr\text{-}step}}
+\lambda_4\mathcal{L}_{\mathrm{agg\text{-}decision}}
\]

| 项 | 含义 |
|----|------|
| \(\mathcal{L}_{\mathrm{local\text{-}roi}}\) | 各槽 soft 判定；主视野在本 ROI（允许受控上下文） |
| \(\mathcal{L}_{\mathrm{localize}}\) | location/type ↔ GT interest |
| \(\mathcal{L}_{\mathrm{attr\text{-}step}}\) | 归因 Step ↔ interest_to_step（定位正确条件下） |
| \(\mathcal{L}_{\mathrm{agg\text{-}decision}}\) | 可学习聚合 ↔ Yes/No；低置信槽抑制；**禁止**无证据高 Yes |

**不做：** No×3、TNR 上采样、贴伪主训、先 step 后看图的 gate、裸硬 OR 当唯一决策器。

数据：CoTM + InstructIAD；果汁挂 ROI 总线与归因表即可。生成监督可体现：`定位/类型 → 步骤归因 → answer`；推理实现优先并行打分 + 一次汇总生成。

---

## 8. 验收（按因果链，而非话术）

| 探针 | 期望 |
|------|------|
| 涂 GT 缺陷所在 ROI | Yes→No（或 \(P(\mathrm{Yes})\)）下降 **明显高于** 涂背景 / 涂无关部件 |
| 抬高单个正常槽假阳性 | 聚合后不应轻易全局 Yes（容错；§6.3） |
| 仅破坏工艺文本、图不变 | 定位与 Yes 应基本稳定；至多归因话术变差 |
| 错改 `interest_to_step` 监督 | 归因 Step 变错，但定位 ROI 仍应对 |
| 强迫跳过某 GT 槽的局部证据 | 不应还能稳定 Yes |

再报 musc_v0 acc/rec/tnr 作迁移参考。  
**不是**成功标准：think 里是否单独写出漂亮 Step 句，而定位/决策不依赖图。

---

## 9. 分周落地

| 周 | 内容 | 通过门槛 |
|----|------|----------|
| Week 0 | SAM 闭集部件；**每槽**涂 ROI 可靠性探针 | 缺陷区 vs 背景可分；不过关的槽不进投票 |
| Week 1 | `<roi:*>` + soft 局部头（许受控上下文）；分单槽/跨槽 | attn 与 \(p_r\) 落在对的槽/ pair |
| Week 2 | **可学习聚合**（非裸 OR）+ 因果探针 | 涂 GT 槽降 Yes；单槽 FP 不炸全局 |
| Week 3 | interest→step 归因（定位对才重惩）；可选 composition | 定位对时 Step 对；错表只伤归因 |

---

## 10. 建议路径

| 路径 | 作用 |
|------|------|
| `movable/process_interest_graph.json` | interests、`interest_to_step`、pair 规则、逆索引 |
| `scripts/build_process_interest_masks.py` | SAM 出全套部件 mask + 质量分 |
| `movable/juice_interest_masks/` | 每图 mask（勿污染 CoTM 原 json） |
| `LLaVA-NeXT/llava/train/pgrb_region_scan.py` | soft 局部 + pair 头 + 可学习聚合 + 归因损失 |
| `llava_eval/pgrb_causal_probe.py` | 涂缺陷槽 / 涂无关槽 / 单槽 FP / 扰动工艺文本 |

---

## 11. 成功 / 失败

**成功：** 行为符合「扫区→发现→定位→归因→决策」；涂缺陷槽能压 Yes；单槽噪点不轻易全局炸；Step 随定位走。

**失败：**

- 仍先写 Step 再找区，或决策不依赖任何 ROI  
- 落地成裸硬 OR，Yes bias 回潮  
- 只训出漂亮工艺句子，因果探针不动  
- 又退回贴伪 / No×3 / 纯 v6 文本挂步  
- 用 LogSAD 分数直接替 Yes/No  

---

## 12. 一句话

**工艺用于归因，区域用于发现；训练对齐这条链。实现上用 soft 局部 + 跨槽显式建模 + 可学习聚合扛可靠性/独立性/容错，而不是理想化的硬独立 OR。**
