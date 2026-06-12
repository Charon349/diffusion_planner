# BridgeDrive Alignment: K-Parallel Bridge（对齐论文实际实现）

## 正确的 BridgeDrive 算法理解

BridgeDrive 论文（ICLR 2026）的实际实现（LEAD adaptation 和 DiffusionDrive adaptation 代码均一致）使用的是：

**训练**：K 条并行 bridge + winner-take-all 回归 + 独立 cls 监督  
**推理**：AnchorClassifier 先跑（cls-first）→ K 条并行去噪 → gather 最优 mode

参见论文代码：
- `BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py`
  - `forward_train` 第 261-316 行：K 条并行 bridge 构建与 winner-take-all 回归
  - `forward_test` 第 340-388 行：cls-first → K 条并行去噪 → gather

---

## 实验背景（保留旧数据供参考）

| 配置 | PDMS | 主要症状 |
|---|---|---|
| baseline (const-vel) | 0.6814 | — |
| anc_k1_bd2 | 0.3435 | 大量静止 |
| anc_k15_bd1 | 0.4153 | 大量追尾 |
| anc_k15_bd2 | 0.3659 | 追尾更严重 |

---

## 改动列表

### C1 — `loss.py` K>1 路径：K 条并行 bridge（对齐论文 forward_train）
**论文对应**：LEAD `forward_train` 第 261-275 行；`DDBMLossComputer.forward` 第 181-219 行

**旧行为**（错误对齐）：`argmin` 最近 anchor → 单桥 `x_t_ego [B, T, 4]`，单个 ego slot 进 decoder

**新行为**：
1. 构建 K 条并行 ego bridge（每条对应一个 anchor）：
   - `gt_ego_norm` 作为 x_0，向所有 K 个 anchor 方向同时加噪
   - `xT_all = ego_anchors_dev.unsqueeze(0).expand(B, K, T, 4)` — K 个端点
   - `z_ego = torch.randn(B, K, T, 4, ...)`
   - `x_t_ego = bridge.add_noise(x0=gt_ego_norm[:,None].expand(B,K,T,4), xT=xT_all, noise=z_ego, t=t)` → `[B, K, T, 4]`
2. 邻车**不使用 bridge**，沿用 VP diffusion（`_NB_SDE.marginal_prob`）：`x_t_nb [B, P_nb, T, 4]`，无 bridge 端点
3. 拼接：`x_t_combined = cat([x_t_ego, x_t_nb], dim=1)` → `[B, K+P_nb, T, 4]`
4. 构建 endpoint：K 个 anchor 端点 + 邻车全零占位 → `endpoint_combined [B, K+P_nb, T, 4]`
5. 单次 forward：`score [B, K+P_nb, T, 4]`，`cls_logits [B, K]`（来自 AnchorClassifier）
6. `assigned = argmin dist(GT_pos, anchor_pos)` → cls 监督目标（最近 anchor 的索引）
7. `best_slot = cls_logits.argmax(dim=1)` → gather `score[:, :K]` by `best_slot` → `best_ego [B, T, 4]`
8. 回归 loss：`(best_ego - gt_ego_norm).pow(2).sum(-1)`
9. cls loss：`cross_entropy(cls_logits, assigned)`

**参考代码位置**：LEAD `forward_train` 261-316；`DDBMLossComputer.forward` 181-219

- [x] 已完成

---

### C2 — `decoder.py` `AnchorClassifier`：保持不变 ✅
**论文对应**：LEAD `diff_decoder_cls`（在干净 anchor 特征上运行，独立于去噪路径）

当前实现已与论文对齐，无需修改。

- [x] 已完成

---

### C3 — `decoder.py` 训练分支：改回 K ego slots（对齐论文 forward_train）
**论文对应**：LEAD `forward_train` 第 298-302 行（`diff_decoder` 处理 K 个 ego token）

**旧行为**（错误对齐）：`sampled_trajectories [B, 1+P_nb, T, 4]`，固定 1 个 ego slot

**新行为**：
- `sampled_trajectories [B, K+P_nb, T, 4]`（K 个 ego slot + P_nb 邻车 slot）
- `bridge_endpoint [B, K+P_nb, T, 4]`（K 个 anchor 端点 + 邻车全零占位）
- DiT 处理 K+P_nb tokens → `score [B, K+P_nb, T, 4]`
- AnchorClassifier 仍独立运行：`cls_logits [B, K]`
- gathering 与 loss 计算在 loss.py 完成（decoder 只负责输出 score 和 cls_logits）

**参考代码位置**：LEAD `forward_train` 298-316

- [x] 已完成

---

### C4 — `decoder.py` 推理分支：cls-first → K 条并行去噪 → gather（对齐论文 forward_test）
**论文对应**：LEAD `forward_test` 第 340-388 行

**旧行为**（错误对齐）：`best_anchor_idx` → 单桥（仅 1 个 ego slot 去噪）

**新行为**：
1. `AnchorClassifier` 先跑 → `best_anchor_idx [B]`（cls-first，保持正确）
2. 构建所有 K 个 anchor 作为 ego 端点：`ego_endpoints = _ego_anchors.unsqueeze(0).expand(B, K, T, 4)` → `[B, K, T, 4]`
3. 拼接邻车零端点：`endpoint [B, K+P_nb, T, 4]`
4. `bridge_sampler` 运行 K+P_nb slots 的去噪
5. 采样结束后 gather：`x0[:, :K][range(B), best_anchor_idx]` → `[B, T, 4]`

**参考代码位置**：LEAD `forward_test` 331-388

- [x] 已完成

---

### C5 — `DiT.forward`：支持 K ego slots（对齐论文 diff_decoder 处理 K token）
**论文对应**：LEAD `forward_train`/`forward_test` 中 `diff_decoder` 接受 `traj_feature [B, K, D]`（K ego token 作为序列元素）

**旧行为**：`P_nb = P_total - 1`，agent_embedding 固定 1 ego（type 0）+ P_nb neighbor（type 1），`attn_mask[:, 1:]`

**新行为**（新增 `num_ego_slots` 参数，默认 1 保持向下兼容）：
- `P_nb = P_total - num_ego_slots`
- agent embedding：前 `num_ego_slots` 位用 ego type（0），后 P_nb 位用 neighbor type（1）
- `attn_mask[:, num_ego_slots:]`（K 个 ego slot 永不被 mask）

**参考代码位置**：LEAD `forward_train` 中 `diff_decoder` 处理 K ego token 的序列方式

- [x] 已完成

---

### C6 — `sampling.py` `bridge_sampler`：确认支持 K+P_nb slots
**论文对应**：LEAD `forward_test` 去噪循环第 350-382 行

**旧行为**：调用方传入 1+P_nb slots；bridge_sampler 本身无硬编码假设（待确认）

**新行为**：
- 调用方传入 K+P_nb slots 的 endpoint 和初始 x_t
- bridge_sampler 逻辑对所有 slot 统一执行去噪，不依赖 slot 数量
- 检查是否有 `P_total = 1 + P_nb` 的硬编码，有则修改；无则仅更新调用方传参

**参考代码位置**：LEAD `forward_test` 第 349-382 行

- [x] 已完成

---

### C7 — `DiT` 内 cls_head 已删除：保持不变 ✅
论文对应：分类器与去噪器独立（LEAD `diff_decoder_cls` 独立于 `diff_decoder`），已实施，无需修改。

- [x] 已完成

---

### C8 — `AnchorClassifier.out_proj` 零初始化：保持不变 ✅
训练初期对所有 anchor 无偏好，已实施，无需修改。

- [x] 已完成

---

## 文件变更摘要

| 文件 | 改动 |
|---|---|
| `diffusion_planner/loss.py` | K>1 路径：恢复 K 条并行 bridge；reg loss 改为 gather by cls_argmax |
| `diffusion_planner/model/module/decoder.py` | 训练分支：K+P_nb slots；推理分支：K 条并行去噪 → gather；DiT.forward 支持 K ego slots |
| `diffusion_planner/model/diffusion_utils/sampling.py` | 确认/修复 K+P_nb slots 兼容性 |

---

## 不变的部分

- `sde.py`：VPSDEBridge 数学不变
- `encoder.py`：场景编码不变
- `AnchorClassifier`：架构不变（与论文 `diff_decoder_cls` 对齐）
- `DiTBlock`、`FinalLayer`：架构不变
- `endpoint_proj`：保留（x_T 条件化，对应论文签名 `x_θ(x_t, t, x_T, z)`）
- K=1 路径（loss.py）：原始单模态路径不变
- `compute_ego_anchors.py`：anchor 生成逻辑不变

---

## Q1 — 邻车 bridge endpoint 的合理性（保留旧讨论）

BridgeDrive 论文仅针对 ego 定义 bridge，不涉及邻车联合预测。
K>1 路径中邻车沿用 VP diffusion（`_NB_SDE`），endpoint 为全零占位，无论文依据但与原始 Diffusion-Planner 设计兼容。
K=1 路径中邻车仍使用 const-vel bridge（原始路径不动）。
此不一致已知，暂不修改，等实验验证后再决定是否对齐。

---

## 迁移核查结果（2026-06-11，请逐项确认）

对照参考实现做了一次完整核查。参考实现有两份且**完全一致**：
- LEAD：`BridgeDrive_adaptation_LEAD/lead/tfv6/diffusion_modules/model_diffusion_head_ddbm.py` + `multimodal_loss_ddbm.py`
- DiffusionDrive：`BridgeDrive_adaptation_DiffusionDrive/navsim/agents/diffusiondrive/model_diffusion_head_ddbm_v5.py` + `modules/multimodal_loss.py`

### ✅ 已正确对齐（直接移植，无新造）

| 项 | 实现位置 | 参考位置 | 说明 |
|---|---|---|---|
| DDBM 桥数学（a_t/b_t/c_t、add_noise、sample_step、is_T 首步注噪） | `sde.py:191-241` | LEAD `DDBMScheduler` 44-123 | 公式逐行一致，纯移植 |
| 训练：K 条并行桥（x_0 同时向所有 K 个 anchor 加噪） | `loss.py:120-128` | LEAD `forward_train` 261-275 | 一致 |
| 推理：cls-first → K 条并行去噪 → gather | `decoder.py:262-319` | LEAD `forward_test` 340-388 | 一致 |
| `best_reg` 用 **cls argmax** 来 gather（不是用最近 anchor） | `loss.py:156-162` | LEAD 309-311 / DiffDrive head 244-247 | 一致（两份参考都用 cls argmax） |
| cls 监督目标 = 最近 anchor（argmin dist） | `loss.py:117-118` | LEAD loss 126-129 / DiffDrive loss 140 | 一致 |
| AnchorClassifier 独立于 x_t（干净 anchor 上跑，去噪循环外只跑一次） | `decoder.py:20-84,253-268` | LEAD `diff_decoder_cls` | 一致（§3.4） |

### ✅ D1/D2 已对齐论文（2026-06-11 修改）

| # | 方面 | 论文参考做法（及代码） | 修改后实现 | 状态 |
|---|---|---|---|---|
| **D1** | **分类损失** | **focal loss**（sigmoid-BCE，γ=2，α=0.25，多标签 one-hot），`py_sigmoid_focal_loss` —— LEAD `multimodal_loss_ddbm.py:138-146`；DiffDrive `multimodal_loss.py:154,206` | 已移植 `py_sigmoid_focal_loss`（`loss.py:14-35`），cls loss 改用 focal + one-hot 目标（`loss.py:181-187`），不再用 `F.cross_entropy` | ✅ 已对齐 |
| **D2** | **回归损失** | **`F.l1_loss(best_reg, gt)`（L1）** —— LEAD `multimodal_loss_ddbm.py:149`；DiffDrive `multimodal_loss.py:167,217` | ego reg loss 改为 L1 `.abs().sum(-1)`（`loss.py:166`）；因 ego 训练于速度空间，L1 取在全部 4 维 | ✅ 已对齐（速度空间适配） |

### ⚠️ 其余 host repo 适配（非 BridgeDrive，列出供你判断是否保留）

| # | 方面 | 论文参考做法（及代码） | 当前实现做法（及代码） | 性质 |
|---|---|---|---|---|
| **D3** | **waypoint hybrid 损失** | 无（BridgeDrive 无此项） | 额外加 `hybrid_waypoint_weight * waypoint_loss` —— `loss.py:173-176,192-195` | host repo 附加项，非 BridgeDrive。属保留原 Diffusion-Planner 行为，但不在论文内 |
| **D4** | **邻车联合建模** | 无（BridgeDrive 只对 ego 定义桥） | K>1 时邻车走 VP diffusion（训练 `loss.py:136-139`）/ VP-DDIM（推理 `sampling.py:147-151`） | 已在本文档 §Q1 记录为已知适配，无论文依据 |
| **D5** | **ego 目标空间** | 位置 (x,y)（+可选 speed holder） | 速度 (vx,vy)+cos/sin，推理时 cumsum 积分回位置 —— `decoder.py:329-335` | host repo 设计；为此 anchor 也存成速度空间，最近-anchor 比较需先把 anchor 速度积分回位置 `loss.py:113-118` |
| **D6** | **AnchorClassifier 结构** | cross-BEV attn + cross-ego attn，anchor 用 `gen_sineembed` 位置编码，2 层 —— LEAD `CustomTransformerDecoderLayerCls` | MLP(flatten anchor) + 1 层 cross-attn 到融合 scene enc + FFN —— `decoder.py:37-84` | 适配 host backbone（无 BEV 网格，用融合 token）；符合 §3.4 文字描述但结构简化 |

### 结论

- **核心 BridgeDrive 算法（桥扩散数学 + K 并行训练/推理 + cls-first 选择）已正确迁移**，与两份参考实现结构一致。
- **D1、D2 已于 2026-06-11 改回论文做法**（focal loss + L1），与两份参考实现一致。
- D3–D6 均为 host repo（Diffusion-Planner / nuPlan）适配，各有原因，非"凭空新造"，但同样不在 BridgeDrive 论文范围内，列出供你判断是否保留。

---

## anchor_cls_loss_weight 调参指引（D1 改 focal 后）

**背景**：D1 把分类损失从 softmax cross-entropy 换成 sigmoid focal 后，**量级骤降约 20–30 倍**：
- 旧 CE 初始 ≈ `log(K)`（K=60 → ≈4.1），随训练 → 0。
- 新 focal 初始 ≈ **0.13**（基本与 K 无关），随训练 → ~0.01。

**聚合链路**（`train_epoch.py:127-133`）：
```
raw = neighbor_loss + alpha_planning_loss(=1.0) · mean( velocity_L1 + waypoint_w·waypoint + cls_w·cls_focal )
```
等权（`anchor_cls_loss_weight=1.0`）下，focal 只占总 loss 的零头，分类头可能欠拟合。
而**推理按 cls argmax 取桥** —— 分类器选错 anchor 直接产生坏轨迹（对应历史实验的"追尾/静止"）。

**建议流程**：
1. 先按论文设 `--anchor_cls_loss_weight 1.0`（参考实现 cls 权重=reg 权重=1.0）。
2. 盯 3 个判读信号：
   - 训练进度条 `cls` 数值（focal 应在 0.0x–0.1x 平稳下降，不降=没学）；
   - **anchor 选择准确率**（`diagnose_cls_bias.py` / `diagnose_anchors.py`）—— 决定 clsw 是否要加大的直接依据；
   - 最终 PDMS / 闭环指标。
3. 若分类准确率偏低，把 `anchor_cls_loss_weight` 抬到 **5–10**，补回 focal 弱梯度。这只放大 cls 相对 velocity/waypoint 的占比，**保留 focal 形式（仍对齐论文）**，不是新造方法。

**对比实验**（`sweep.sh` Group 3，固定 K=15 / beta_d=2.0）：
- `clsw=1.0` == `anc_k15_bd2`（默认，已在 Group 1）
- `anc_k15_clsw5`（5.0）、`anc_k15_clsw10`（10.0）

---

## 新开窗口恢复上下文

如果在新窗口中继续本次改动，需要：
1. 阅读本文件确认已完成项（打钩的）和未完成项
2. 阅读以下文件了解当前状态：
   - `diffusion_planner/loss.py` — K>1 路径是否已恢复 K 条并行 bridge
   - `diffusion_planner/model/module/decoder.py` — 训练/推理分支 ego slots 数量；DiT.forward 是否支持 K ego slots
   - `diffusion_planner/model/diffusion_utils/sampling.py` — bridge_sampler 是否兼容 K+P_nb
3. 从第一个未完成项（C1）继续

---

## 理由与确认要求

每项改动均有论文代码位置对应（LEAD adaptation `model_diffusion_head_ddbm.py`）。
不得在未说明论文依据的情况下引入新逻辑。
若某处有多种实现方式，先列出并标明推荐，再执行。

---
---

# v2（2026-06-12）：P1–P3 对齐修复

> 本节为新版本改动记录，不覆盖上文 v1 内容。背景：2026-06-12 对照两份参考实现
> 复核，发现 3 处 v1 文档未记录的实质偏差（P1–P3，已全部修复），另记录 3 处
> 暂不修改的小偏差（P4–P6）。**v1 的全部 K>1 实验结果（anc_k*）受 P1/P3 影响，
> 作废重跑**；K=1 baseline 的训练路径不受任何影响，旧 checkpoint 可复用
> （仅推理时间表默认值变化，见 P2，可对旧 checkpoint 重评估）。

## P1 — K 个 ego mode 在 DiT 中互相可见（架构级偏差，已修复）

**问题**：参考实现中 K 个 mode 之间**没有任何注意力交互**（LEAD
`CustomTransformerDecoderLayer` 只有 per-mode cross-BEV attn 和对 ego query 的
cross-attn，mode 维度从不混合）。宿主 DiTBlock 对全部 K+P_nb token 做完整
self-attention，后果：① winner-take-all 训练下各 ego slot 可互抄答案，易模式
塌缩（所有 slot 收敛到同一轨迹、anchor 条件被忽略）；② 邻车 token attend 到
K 个假设性 ego，对邻车预测是噪声。这是 v1 实验"加 anchor 反而劣于 baseline"
最可能的架构性原因。

**修复（mode 隔离 mask）**：
- ego slot k 只 attend：自己 + 邻车 token（cross-attn 到 encoder context 不受影响）
- 邻车只 attend：邻车 + **cls-argmax 选中的 WTA ego slot**
  （训练与推理使用同一索引来源——AnchorClassifier argmax——保证一致性，
  且与 loss.py 回归 gather、推理输出 gather 同源）
- K=1 路径不构造该 mask，行为与 v1 完全一致。

**实现位置**：
| 文件 | 改动 |
|---|---|
| `model/module/dit.py` | `DiTBlock.forward` 新增 `mode_attn_mask` 参数（`[B*heads, S, S]` bool，True=屏蔽），与原 `key_padding_mask` 同时传入 `nn.MultiheadAttention` |
| `model/module/decoder.py` `DiT` | `__init__` 记录 `_num_heads`；`forward` 新增 `wta_idx [B]` 参数，K>1 时构造 mode mask（ego↔ego 仅对角放行；邻车→ego 仅放行 WTA slot，`wta_idx=None` 时全屏蔽作为保守回退） |
| `model/module/decoder.py` `Decoder` 训练分支 | AnchorClassifier 调用移至 DiT **之前**，`wta_idx = cls_logits.argmax(1)` 传入 DiT |
| `model/module/decoder.py` `Decoder` 推理分支 | `best_anchor_idx` 经 `bridge_sampler` 的 `other_model_params["wta_idx"]` 传入每步去噪 |

**论文依据**：mode 独立性来自参考实现结构（mode 间无 self-attn）；"邻车 attend
WTA ego" 是宿主联合预测架构的适配（参考实现无邻车 token），属 §Q1/D4 同类适配，
取"保留联合预测语义"与"mode 独立"的交集。

- [x] 已完成

## P2 — 推理时间表默认 quadratic → linear（已修复）

两份参考实现均为**均匀线性网格**（LEAD `forward_test:328` / DiffusionDrive
`v5:273`：`arange(0, 1001, 1000//steps)`）。v1 的 "quadratic"（t→0 加密、t→T
步距变大）为宿主自创且未消融，且 t≈T 段恰是 anchor→轨迹结构变换处，大步距
可能放大离散误差。已将默认值改回 `linear`：
- `model/module/decoder.py`：`diffusion_bridge_schedule` 默认 `"linear"`
- `model/diffusion_utils/sampling.py`：`bridge_sampler` 默认 `"linear"`，docstring 更新

quadratic 保留为消融选项（config `diffusion_bridge_schedule="quadratic"`）。
该项**仅影响推理**，对 v1 旧 checkpoint 可直接用新默认重评估。
注意：`diagnose_cls_bias.py` 硬编码 `diffusion_bridge_steps: 10` 偏少
（sampling.py 自家注释即称 "10 is too few"），诊断结论可能被低估，建议改 20。

- [x] 已完成

## P3 — AnchorClassifier out_proj 初始化（已修复，v1 C8 记录不准确）

**v1 做法**（C8）：weight 与 bias 全零 → 所有 logits 严格相等，初始 sigmoid
p=0.5。两个问题：① focal loss 初期被 K−1 个负类主导（每元素 ≈0.13），梯度
先花在无信息的"全体压低"过渡上；② logits 全等使 argmax **早期恒选 slot 0**，
与 P1 的 WTA mask、loss.py 的回归 gather 连锁，开训时全部锁死在 anchor 0。

**v2 做法**（对齐参考 `bias_init_with_prob(0.01)`，LEAD
`model_diffusion_head_ddbm.py:428-429`）：weight 保留 `_basic_init` 的 xavier
初始化（logits 从第 0 步起有区分度），bias = `-log((1-0.01)/0.01) ≈ -4.595`
（初始 p=0.01，RetinaNet focal 先验初始化）。位置：`model/diffusion_planner.py`
`Diffusion_Planner_Decoder.initialize_weights`。

**连带影响**：focal 初始量级从 ≈0.13 降至 ≈0.02（集中在正类上）。v1 文档
"anchor_cls_loss_weight 调参指引"中的量级估算（focal 初始 ≈0.13）按 v2 作废，
clsw 合理区间可能需进一步上移，由 sweep Group 3 实验判定。

- [x] 已完成

## 记录在案、暂不修改的小偏差（P4–P6）

| # | 方面 | 参考做法 | 当前做法 | 评估 |
|---|---|---|---|---|
| P4 | reg loss 量级 | `F.l1_loss`（按元素 mean） | 4 维 `sum(-1)` 再 mean（≈4×），另叠加 waypoint/neighbor 项 | cls:reg 有效比例比参考小 ≈4×+；由 clsw sweep 覆盖，不单独改 |
| P5 | 训练 t 采样 | `randint(1,1001)/1000`，含 t=T 整点（推理首步输入=anchor 本身） | `rand∈[eps,1)`，不含 T | 连续性下影响极小 |
| P6 | 最近 anchor 度量 | 逐时刻 L2 范数取 mean | 平方距离求和（`loss.py`） | argmin 在边界样本可能不同，影响 cls 标签一致性；后续可一行修正 |

## 验证状态与 v2 实验计划

- 本地环境无 torch，`test_decoder_smoke.py` 未能本地运行。**sweep.sh 已加入
  preflight：服务器上先跑 smoke test（覆盖 K=1/K=4 训练+推理分支），失败即停。**
- `sweep.sh` 已更新为 v2：结果目录 `results/sweep_v2`（不覆盖 v1）；
  K=1 baseline 不重训（训练路径未变，复用 v1 checkpoint，可选 linear 重评估）；
  Group 1: K∈{15,60}，Group 2: beta_d∈{1,4}（K=15），Group 3: clsw∈{5,10}（K=15）。
- 判读顺序不变：① 训练 `cls` 曲线 → ② `diagnose_cls_bias.py` 分类准确率
  （建议补充观测 K 个 ego slot 输出的两两距离，验证 P1 后 mode 多样性恢复）→ ③ PDMS。
