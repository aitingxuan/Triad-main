# 多 IAD 模型「自适应异常图融合」方案（Agent 调度 + 门控）

> 版本：v1（设计稿 / 实验设计）
> 范围：如何用多个 IAD（工业异常检测）模型，经一个"Agent 调度 + 门控"策略，
> **综合输出一张最好的 anomaly map**（而非 VLM 的 Yes/No 判定），再交 MeBin 二值化。
> **基线（baseline）以 `D:\triad\labelingcopilot_my\annotate` 现有管线为准。**
> 本文只做设计与实验规划，不强制改动现有代码。

---

## 0. TL;DR

- 目标不是复刻 Triad（Triad 是"VLM 读局部 crop 判 Yes/No"），而是**map 级融合**：
  多专家各出一张 anomaly map，由一个**调度/门控策略（"Agent"）**决定"用哪些专家、如何在像素/区域上加权"，
  **合成一张比任何单模型都好的单图**，再交给 MeBin。
- 科学问题：**测试时无 GT 的前提下，学习一个 GT-free 的调度 + 门控函数，把异构 IAD 图融合成最优单图**。
- 铁律：判据必须 GT-free（来自"标定集学到的可靠性 / 一致性 / 不确定性 / 先验"）；
  **Agent 只出"融合计划"，像素合成由确定性融合算子完成**（LLM 不直接吐像素）。
- 落地路径：先修基线 bug → 零训练 rank-mean 融合 → 图像级门控 → 像素级门控(MoE) → 级联 + 逻辑专家 + 人在回路(主动学习)。

---

## 1. 背景与动机

### 1.1 为什么"综合成一张图"是独立问题
- `labelingcopilot_my` 当前把 MuSc 与 AnomalyCLIP 的图**逐像素相乘（product）**再 MeBin。
  product 要求"两家都高才算异常"，会**互相否决**：在某专家正确的区域被另一专家压低，
  逻辑/结构缺陷在 cable 上大面积塌陷（见 §2.3）。**这不是"融合"，是"求交集"。**
- Triad（`Triad-main`）的 EG-RoI 路线是 **VLM 分类**：IAD 图只用来定位 crop，最终输出 Yes/No + 缺陷类型，
  **不产出"最优 anomaly map"**。所以 Triad 解决不了本目标，但它的"map 只负责定位、判定消费局部证据"的思路可借鉴。

### 1.2 想让"Agent 做调度决策"有意义，必须回答三个问题
1. **判据从哪来（无 GT）？** → 校准可靠性 / 一致性 / 不确定性 / 规格先验（GT-free）。
2. **谁出像素？** → 确定性融合算子；Agent 只给"权重/策略"。
3. **如何证明真的更好（不靠 GT 选优）？** → 训练/验证/测试三分 + 只比基线/消融 + per-type 报告。

---

## 2. 基线定义（以 `labelingcopilot_my/annotate` 为准）

### 2.1 现有管线（当前代码）
```
一批图 ─► MuSc（批量互评分，/annotate_batch，每请求自成 batch）──┐
        └► AnomalyCLIP（逐图，/annotate）─────────────────────┤
                       ▼
   L0'  batch 级 percentile(2,98) 线性归一化（每模型一个全局 lo/hi）
                       ▼
   L3'  逐像素相乘 product 融合  F = Π_k M̃_k
                       ▼
   L4'  MeBin 自适应二值化（整批一次，sample_rate=4, min_interval=2, erode 6x6）→ mask
```
- 入口：`annotate/orchestrator/app.py`（`/annotate`、`/annotate/batch`、`/annotate/segment`、`/health`）。
- 模型服务：`annotate/models/musc/app.py`（`/annotate`、`/annotate_batch`）、`annotate/models/anomaly_clip/app.py`（`/annotate`）。
- 二值化：`annotate/shared/mebin.py`（`MEBin` 类，批量 `binarize()`）。
- 评测脚本：`annotate/test_mvtec_batch.py`（`/annotate/batch`，默认 `--chunk 32`，输出 `summary.csv` + pixel/image AUROC）。
- 服务编排：`annotate/orchestrator/model_urls.json`（MuSc→:8001、AnomalyCLIP→:8002），`annotate/run_local.ps1`。
- 现有可复用的其它模型服务（目前**未接入 IAD 编排**）：`detic`、`owl_vit`、`grounding_dino`（开放词表检测，`DetectionResponse`）、`seem`、`sam`（分割，`SEEMResponse`/`SAMResponse`）。

> 说明：`annotate/MuSc_AnomalyCLIP_进展记录.md` 早期设计是 percentile(1,99)+平均；**当前代码已演进为
> batch 2–98 归一化 + product 融合**。本文以"当前代码"为 baseline。

### 2.2 基线入口/数据契约（便于后续替换融合层）
- 统一响应：`IADResponse{ anomaly_map_b64(float32), mask_b64(PNG), shape, image_width, image_height, threshold, est_anomaly_num }`（见 `annotate/shared/schemas.py`）。
- 各专家原始图：`AnomalyMapResponse{ anomaly_map_b64, shape, image_width, image_height }`。
- 即：**只要新融合层仍产出 `IADResponse`，就能零侵入替换** `orchestrator._annotate_batch` 的融合段。

### 2.3 基线现状体检（cable，当前代码实测）

| 指标 | 数值 |
|---|---|
| pixel AUROC | **0.7991** |
| image AUROC | **0.7186** |
| 缺陷 IoU 均值 / 中位 | **0.127 / 0.056** |
| IoU≥0.3 / ≥0.1 / =0 的缺陷数 | 12 / 37 / 21（共 92） |

按缺陷类 mean IoU：

| 类型 | meanIoU | 备注 |
|---|---|---|
| bent_wire | 0.290 | 结构型，最好 |
| poke_insulation | 0.225 | 结构型 |
| cut_outer_insulation | 0.195 | 结构型 |
| combined | 0.129 | |
| cut_inner_insulation | 0.107 | |
| missing_wire | 0.057 | 逻辑型 |
| missing_cable | 0.010 | 逻辑型，近乎全灭 |
| cable_swap | 0.005 | 逻辑型，近乎全灭 |

正常图（good）假阳性：58 张里 **20 张 fg>1000px、9 张 fg>5000px、最大 18292px**；
`img_score=max` 有 **13/58 顶到 1.0**（异常图 47/92 也是 1.0）→ image 级大量并列，AUROC 被压低。

### 2.4 基线已知问题（融合方案必须顺带修掉）
1. **batch 级 2–98 全局裁剪**：2% 像素被 clip 到 1.0 → 大量 `max=1.0` 并列；且**batch 组成决定结果**
   （纯 good 的第一批会让正常图整体抬到 ~1.0；实测 good[0:32] 有 9 张 fg>5000，good[32:58] 为 0）。
2. **product 融合互相否决**：逻辑型缺陷被 AnomalyCLIP 拖没（cable_swap / missing_cable ≈0）。
3. **图像级分数用 `fused.max()`**：最易被单像素/裁剪顶穿 → image AUROC 结构性偏低。
4. **MeBin 的 batch = 每个 HTTP chunk(32)**：阈值随分块/顺序漂移，不可复现。
5. **测试脚本输出命名用 basename**：不同缺陷子目录 `000.png…` 互相覆盖 `amap_*/mask_*`（只影响可视化，不影响 `summary.csv` 指标）。

> 注：bottle 旧结果为此前版本代码所跑，**须用当前基线重跑**后再做对比。

---

## 3. 科学问题（形式化）

设 K 个 IAD 专家 \(\{E_k\}_{k=1..K}\)，对图 \(x\) 输出异常图 \(M_k(x)\in\mathbb{R}^{H\times W}\) 与标量分 \(s_k(x)\)。
求一个**调度+门控策略** \(\pi\) 与**融合算子** \(\mathcal{F}\)，在**测试时无 GT** 下：

\[
\tilde{F}(x)=\mathcal{F}\big(\pi(\{M_k(x),s_k(x)\},\,c);\ \{M_k(x)\}\big)
\]

使得 \(\tilde{F}\) 在混合分布上优于任意单专家，并优于 naive 融合（mean / rank-mean / max / **product**）。
其中 \(c\) = 类别/规格等已知上下文；\(\pi\) 的参数**只在标定集上学、推理时冻结**。

---

## 4. 设计原则（硬约束）

1. **GT-free 判据**：测试时的"选谁/怎么混"只能来自
   (a) 标定集学到的可靠性/门控（离线学、冻结）；
   (b) 专家间一致性/分歧；
   (c) 不确定性（TTA 方差、map 熵）；
   (d) 规格/领域先验（结构 vs 逻辑、成本预算）。
2. **Agent 出"计划"，算子出"像素"**：LLM/策略只给"专家子集 + 权重/策略"；像素由确定性/可微融合算子完成。
3. **不互相否决**：用加权/门控/max-with-agreement，而非 product。
4. **可评测**：训练/验证/测试三分；测试 GT 只用于报告；必比基线与消融；按缺陷族报告。
5. **非侵入基线**：新方案须仍输出 `IADResponse`，可替换 `orchestrator` 融合段而不动模型服务。

---

## 5. 总体架构

```
            ┌──────────────────────── 专家库(模型服务, 可增删) ────────────────────────┐
            │ MuSc(结构,batch)  AnomalyCLIP(语义)  LogicExpert(逻辑)  Detector/SEEM(部件) │
            └───────────────┬──────────────────────────────────────────────┬──────────┘
                            │  各出 raw map M_k / score s_k                 │
                            ▼                                               │
  L0 校准       逐图 robust 归一化(rank / percentile) → M̃_k ∈[0,1]（尺度可比，去掉 batch 全局裁剪）
                            ▼
  L1 门控(Agent) π(scorecard, 上下文) → 融合计划 {专家子集, 全局权重 g_k, 逐像素权重 w_k(x)}
                            ▼
  L2 调度        cost-aware 级联：便宜专家先跑→看分歧/不确定性→决定是否升级贵专家/请求复核
                            ▼
  L3 融合算子    F(x) = Σ_k g_k·w_k(x)·M̃_k(x)   （或 rank/logit 版）→ 单张最优图
                            ▼
  L4 决策        MeBin(F) 出 mask；低置信/冲突 → 弃权/复核（GT-free）
```

要点：L0 修掉 §2.4 的根因；L3 用"加权取证"替代 product；MeBin 作用在**单一、尺度合理**的 \(F\) 上才成立。

---

## 6. 关键组件细则

### 6.1 L0 校准（先做，收益最大、风险最低）
- 目标：让不同专家的图**可比**，且**不跨图被全局裁剪**。
- 做法（择一，建议先从 rank 起步）：
  - **rank 变换**：把 \(M_k\) 展平成排名 → \([0,1]\)（对尺度/分布鲁棒，零假设）。
  - **逐图 percentile(1,99)** 截断后归一化（相对稳健，但仍建议逐图而非整批）。
  - **可学习温度**（有标定集时）：\( \tilde M_k = \sigma(a_k\log M_k + b_k)\)，\(a_k,b_k\) 用标定集学到。
- 禁忌：**不要**再用"整批所有像素的 2/98 百分位"做统一线性映射（§2.4-1 的根因）。

### 6.2 L1 门控（= "Agent 的决策能力"）两档

**档 A：零训练（先跑通 / baseline）**
- 权重来自 GT-free 信号：
  - 专家间**一致性**（如逐像素 rank 相关 / IoU@τ）；
  - **不确定性**（TTA 方差、mask 熵）→ 高不确定降权；
  - **分专家可靠性先验**（小标定集按类别/缺陷族统计的 P/R，不是逐图 GT）。
- "Agent"= 规则调度器（结构域↑MuSc、逻辑域↑LogicExpert）。
- 融合：**rank-mean / 加权校准和**。

**档 B：有训练（冲上限）**
- **图像级门控** \(g_k\)：小 MLP/GBM，输入"各专家标量 scorecard（峰值、异常面积、熵、两两一致性、图像特征、类别）"，输出 K 维 softmax。
- **像素级门控** \(w_k(x)\)：MoE 式；输入"K 张图堆叠 + 局部特征"→ 逐像素 softmax 权重。
- 训练目标：直接优化**融合后**的可微指标（soft-Dice / 加权 BCE），而非回归某个单模型。

### 6.3 L2 调度（"Agent"真正擅长、且省钱的位）
- **级联**：先跑便宜专家（MuSc batch）→ 算分歧/不确定性 → 共识高则直接融合出图；灰区才调贵专家/检测器。
- **策略选择**：把"scorecard + 规格"给 LLM/VLM，使其在小策略库中选（"仅MuSc" / "MuSc+逻辑" / "全加权"）并给理由；
  **粗粒度、可审计、可复现**；细粒度像素权限仍在 6.2 的学习门控。
- **禁止**把 LLM 放进逐像素回路。

### 6.4 L3 融合算子（为什么不再用 product）
- 线性加权：\( F(x)=\sum_k g_k\,w_k(x)\,\tilde M_k(x) \)，\(g_k,w_k\ge0\)。
- rank 融合：\( F=\text{mean}_k \text{rank}(\tilde M_k) \)（零训练强基线）。
- 校准 logit 融合：\( F=\sigma(\sum_k w_k\,\text{logit}(\tilde M_k))\)。
- 关键：某专家在某区域强就让它主导，**不强求同高**；这与"求交集"的 product 本质不同。

### 6.5 L4 MeBin + 弃权
- MeBin 作用在单一 \(F\) 上（scale 合理后其"连通域稳定序列"假设才成立）；batch 用**整类**，不用 chunk。
- **弃权**：一致性低 / 熵高 / 门控权重接近均匀 → 输出"待复核"而非硬塞结果。

### 6.6 专家库（可逐步扩充）
| 专家 | 角色 | 现状 | 备注 |
|---|---|---|---|
| MuSc | 结构型主力（零样本互评分, batch） | ✅ 已接入 | 当前主专家 |
| AnomalyCLIP | 语义/全局 | ✅ 已接入 | 在 cable 上对逻辑缺陷弱 |
| LogicExpert（PRN/GCAD 等或"检测+规则图"） | 逻辑型（缺件/错位/计数） | ❌ 待接入 | 可用 `detic`/`SEEM` 生成"约束违反"图 |
| Detector/SEEM 派生的部件 map | 逻辑证据/部件 ROI | 🟡 服务已存在 | `detic`/`owl_vit`/`grounding_dino`/`seem`/`sam` |

---

## 7. 训练 / 标定协议（防自欺）

- **三分数据**：train（学门控 + 校准）、val（选策略 + 调 MeBin 阈值）、test（**只报指标**）。
- **GT 使用边界**：GT 只进 train/val；测试 GT 仅用于报告，**绝不用于选模型/调权/调 prompt**。
- **优化目标**：像素 AUROC + soft-IoU（可微代理，如 soft-Dice）。
- **冻结**：推理时门控/校准全部冻结；线上只做前向。

---

## 8. 评测协议

### 8.1 必比基线
| 名称 | 说明 |
|---|---|
| B0 best-single | 各类上各自最佳单专家（上界参照） |
| B1 mean | 归一化后逐像素均值 |
| B2 rank-mean | rank 后均值（零训练强基线） |
| B3 max | 逐像素取最大 |
| **B4 product** | **当前基线（失败的那个）** |
| P1 image-gate | 6.2 档B 图像级门控 |
| P2 pixel-gate | 6.2 档B 像素级门控(MoE) |

### 8.2 必做消融
1. 固定权重 vs 图像级门控 vs 像素级门控（量化"决策"的增益）；
2. 级联 vs 全跑（成本 vs 掉点）；
3. 去掉某专家（互补性：结构 + 逻辑的 per-type 覆盖是否 > 各自）；
4. L0 用 rank vs percentile vs 学习温度（校准的影响）。

### 8.3 报告维度
- 总体：pixel AUROC、best-F1 / IoU、image AUROC（用**基于 mask 的分数**而非 `max`）。
- **per-type**：分别报 IoU/AUROC（结构型 vs 逻辑型），因为融合价值就在覆盖不同缺陷族。
- **正常图假阳性**：`fg>1000 / >5000 px` 占比、top-1% 误报。
- **互补性/因果**：门控在各区域是否按"结构区→结构专家、逻辑区→逻辑专家"分派。

### 8.4 成功判据（诚实设定）
- 在**混合分布**上优于任意单专家，且优于 mean/rank-mean/product；
- 在专家专长类上**不劣化**；
- 级联能省成本且不掉点；
- image 级分数不再被 `max` 饱和（并列显著减少）。

---

## 9. 迭代路线（里程碑）

- **M0｜修基线 bug（不改模型）**
  - 逐图归一化替代 batch 2–98；图像分数改为 mask 均值/面积×均值；MeBin batch 改为整类；修测试脚本命名冲突。
  - 产出：**修正版基线**在 cable 上的干净数字（重跑 bottle）。
- **M1｜零训练融合**
  - 上 rank-mean + 一致性/TTA 加权（档A 门控）；对比 B0–B4。判定"融合是否值得"。
- **M2｜图像级门控**
  - 用标定集学 \(g_k\)（GBM/小 MLP）；对比 M1。
- **M3｜像素级门控(MoE)**
  - 端到端 soft-Dice 训练 \(w_k(x)\)；对比 M2。
- **M4｜级联 + 逻辑专家 + 人在回路**
  - 接入 LogicExpert（或检测+规则图），加 L2 级联；构建"不确定→请求人工标注→在线更新门控/校准"的主动学习闭环（复用 `labelingcopilot_my/retrieval/active_learning_session.py`、`retrieval/samplers/uncertainty.py` 思路）。

---

## 10. 与 `labelingcopilot_my` 的集成点（示意，不强制改码）

- **替换点**：`annotate/orchestrator/app.py` 的 `_annotate_batch` 中"L0 归一化 + L3 融合"两段；
  输入仍是各专家的 `AnomalyMapResponse`，输出仍是 `IADResponse` → 对上游 `test_mvtec_batch.py`、下游 viewer 零侵入。
- **专家接入**：新专家沿用 `/annotate` 或 `/annotate_batch`（返回 `AnomalyMapResponse`）即可被调度；
  `detic`/`seem` 派生的"部件/规则图"可作为特殊专家（输出 map 形式，或先转成 map 再融合）。
- **配置**：扩展 `orchestrator/model_urls.json`（增加 `LogicExpert` 等）；
  新增"门控/融合配置"（专家集合、归一化方式、默认策略、级联预算）。
- **评测**：在 `test_mvtec_batch.py` 基础上加 `--fusion {mean,rankmean,product,gate}` 与 per-type 报告（或用独立评测脚本，保持基线脚本不变）。

---

## 11. 风险与开放问题

- **"融合 > 最佳单专家"不是必然**：在某专家专长类上很难超过它本身；价值在"混合分布/per-type 覆盖"。需诚实报告。
- **可靠性先验的迁移性**：标定集的 per-category 先验对新类别/新产线未必成立，需要少量在线校准。
- **门控过拟合**：像素级门控容量大，必须用严格 train/val/test + 早停。
- **LLM 调度不可复现**：只允许其在"离散策略库"里选，且规则/权重落成结构化配置再执行。
- **逻辑专家的落地形态**：纯逻辑 IAD（PRN/GCAD，LOCO 系）需域适配；检测+规则图更可控但依赖规格与部件词表。
- **成本**：级联/多专家有延迟；需给预算与回退策略。

---

## 12. 复现基线（供对照）

```powershell
# 1) 启动服务（MuSc:8001, AnomalyCLIP:8002, Orchestrator:8080/8090）
powershell -ExecutionPolicy Bypass -File D:\triad\labelingcopilot_my\annotate\run_local.ps1

# 2) 跑整类（示例：cable）
& "D:\triad\venvs\musc\Scripts\python.exe" D:\triad\labelingcopilot_my\annotate\test_mvtec_batch.py `
    --category cable --all --orch http://127.0.0.1:8090

# 输出：annotate\mvtec_out\cable\summary.csv（pixel/image AUROC 打印在 stdout）
```

### 关键文件速查（baseline）
```
labelingcopilot_my/annotate/orchestrator/app.py      # 融合/编排入口（替换点）
labelingcopilot_my/annotate/orchestrator/model_urls.json
labelingcopilot_my/annotate/shared/mebin.py          # MeBin
labelingcopilot_my/annotate/shared/schemas.py        # IADResponse / AnomalyMapResponse
labelingcopilot_my/annotate/models/musc/{app.py,model.py}
labelingcopilot_my/annotate/models/anomaly_clip/{app.py,model.py}
labelingcopilot_my/annotate/models/{detic,owl_vit,grounding_dino,seem,sam}/  # 可复用为部件/规则证据
labelingcopilot_my/annotate/test_mvtec_batch.py      # 评测脚本
labelingcopilot_my/annotate/mvtec_out/cable/summary.csv
labelingcopilot_my/retrieval/active_learning_session.py, retrieval/samplers/uncertainty.py  # 主动学习闭环
```

---

## 13. 术语

- **IAD**：工业异常检测（输出 anomaly map）。
- **专家/Expert**：一个 IAD 模型（或其派生的证据图）。
- **门控/Gating**：决定各专家在某图/某像素上权重的函数（"Agent 决策"的载体）。
- **调度/Scheduling**：决定调用哪些专家、按什么顺序/预算（级联）。
- **GT-free**：测试时刻不使用真实标注。
- **MoE**：Mixture-of-Experts，按输入自适应组合多专家。
- **EG-RoI**：Triad 的"证据导向局部裁剪"（本项目借鉴其"map 只负责定位"的思想，但目标不同）。
