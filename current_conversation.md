# 当前对话导出 (Conversation Export)

## 👤 User

/home/wangchenggang/wcg/Diffusion-Planner_1现在这个工程项目，在规划轨迹的时候，是每一帧都选一个接近GT的anchor加噪去噪吗

---

## 🤖 Assistant

Let me研究一下这个项目的代码来回答你的问题。

---



## 回答：不是每一帧选一个 anchor，而是**每个样本（batch item）**选一个最近的 anchor

具体来说，这个项目有**两条路径**，取决于 `--use_ego_anchor` 参数：

### 路径 1：不使用 anchor（原始 Diffusion Planner）

当 `use_ego_anchor=False` 时，走 [diffusion_loss_func](file:///home/wangchenggang/wcg/Diffusion-Planner_1/diffusion_planner/loss.py#L305-L421)：

- 对 GT 轨迹整体采样一个随机时间步 `t ~ Uniform[eps, 1]`（[L354](file:///home/wangchenggang/wcg/Diffusion-Planner_1/diffusion_planner/loss.py#L354)）
- 用 SDE 的 `marginal_prob` 对 **整条 GT 轨迹** 加噪，然后让模型去噪
- **没有 anchor 参与**，就是标准的 diffusion training

### 路径 2：使用 anchor（你项目当前改造的方式）

当 `use_ego_anchor=True` 时，走 [anchored_diffusion_loss_func](file:///home/wangchenggang/wcg/Diffusion-Planner_1/diffusion_planner/loss.py#L142-L302)，**关键流程如下**：

#### 1. 找到最接近 GT 的 anchor（每个样本一个）
```python
# L175-176
anchor_dist = _anchor_distances(anchors, ego_future)  # [B, K]
pos_k = anchor_dist.argmin(dim=1)                      # [B] — 每个样本选一个最近anchor
```
[_anchor_distances](file:///home/wangchenggang/wcg/Diffusion-Planner_1/diffusion_planner/loss.py#L68-L71) 是计算 anchor 与 GT 轨迹之间的 **L2 xy 距离的均值**。

#### 2. 对 **所有 K 个 anchor** 都加噪去噪（不是只加噪最近的那个）
```python
# L187 — 扩展为 B*K
t = t_base.repeat_interleave(K)

# L189-191 — GT 也复制 K 份
gt_future = gt_future[:, None].expand(B, K, P, T, 4).reshape(B * K, P, T, 4)

# L200-203 — 用 anchor 作为 ego 的加噪均值（而不是用 GT）
anchor_norm = normalize_ego_anchor(anchors, norm)
ego_anchor_mean, _ = sde.marginal_prob(anchor_norm, t)
mean[:, 0] = ego_anchor_mean  # 把 ego 通道的均值替换为 anchor 的均值
```

**核心区别在这里**：普通 diffusion 是对 GT 加噪，而 anchor 版本是 **以 anchor 轨迹为均值** 来加噪。每个 anchor 分支都得到一个从该 anchor 附近出发的噪声样本。

#### 3. 损失只在 **正样本 anchor** 上计算主要 loss
```python
# L229-230
batch_idx = torch.arange(B, device=device)
pos_loss = dpm_loss[batch_idx, pos_k]  # 只取最接近GT的那个anchor的loss
```

#### 4. 还有 **anchor scoring loss**
模型还会输出 `branch_logits`（[L261-266](file:///home/wangchenggang/wcg/Diffusion-Planner_1/diffusion_planner/loss.py#L261-L266)），用 focal loss 训练模型学会判断哪个 anchor 是正样本。

#### 5. 加噪时间范围受限
```python
# L184-186
t_min = 1e-3 (默认)
t_max = 0.2  (默认)
t_base = torch.rand(B) * (t_max - t_min) + t_min
```
anchor 训练只在 **t ∈ [0.001, 0.2]** 的小噪声范围内训练（不像标准 diffusion 用 [0, 1] 全范围）。

---

### 总结

| 维度 | 说明 |
|------|------|
| **anchor 选择粒度** | **每个样本（每帧场景）**选一个最接近 GT 的 anchor，不是每个时间步 |
| **加噪对象** | 以 **anchor 轨迹**（不是 GT）为均值加噪，所有 K 个 anchor 都参与 |
| **去噪** | 模型同时对 B×K 个分支去噪，但主 loss 只算正样本 anchor 分支 |
| **额外任务** | 还训练一个 score head 来预测哪个 anchor 最接近 GT |
| **推理时** | 从每个 anchor 出发做少步去噪（[decoder.py L83-192](file:///home/wangchenggang/wcg/Diffusion-Planner_1/diffusion_planner/model/module/decoder.py#L83-L192)），再用 score head 选最佳分支 |

---

## 👤 User

选择anchor是怎么选的，就是简单的l2 距离吗，然后打分头score head是怎么训练的

---

## 🤖 Assistant

好的，这两个问题我已经从上一轮看过的代码中可以详细回答。让我再仔细看一下 anchor 距离计算和 score head 的细节。

---


## 🤖 Assistant

好的，下面详细回答你的两个问题：

---

## 1. Anchor 选择方式：确实是简单的 L2 距离

核心函数是 [_anchor_distances](file:///home/wangchenggang/wcg/Diffusion-Planner_1/diffusion_planner/loss.py#L68-L71)：

```python
def _anchor_distances(anchors, ego_future):
    anchor_xy = torch.cumsum(anchors[..., :2], dim=1)   # delta → absolute 坐标
    dist = torch.linalg.norm(anchor_xy[None] - ego_future[:, None, :, :2], dim=-1).mean(dim=-1)
    return dist  # [B, K]
```

具体步骤：
1. **anchor 是 delta 格式** (逐帧位移)，先做 `cumsum` 转换成绝对坐标
2. **只用 xy 坐标**（`[:2]`），**不考虑 heading（cos/sin）**
3. 计算 anchor 与 GT 轨迹**逐帧的 L2 距离**（`torch.linalg.norm`）
4. 对所有时间步取 **均值**（`.mean(dim=-1)`），得到每个 anchor 与 GT 的平均距离
5. 最后 `argmin` 选距离最小的那个 → `pos_k`

所以**是的，就是简单的全轨迹 xy L2 均值距离**，没有加权、没有考虑 heading。

---

## 2. Score Head 的训练

### 网络结构

Score head 定义在 [decoder.py L392-396](file:///home/wangchenggang/wcg/Diffusion-Planner_1/diffusion_planner/model/module/decoder.py#L392-L396)，非常简单：

```python
self.score_head = nn.Sequential(
    nn.LayerNorm(hidden_dim),   # LayerNorm(192)
    nn.Linear(hidden_dim, 1),   # 线性映射到标量
)
```

### 输入来源

在 [forward_with_logits](file:///home/wangchenggang/wcg/Diffusion-Planner_1/diffusion_planner/model/module/decoder.py#L430-L435) 中：

```python
def forward_with_logits(self, x, t, cross_c, route_lanes, neighbor_current_mask):
    x, y = self._forward_features(x, t, cross_c, route_lanes, neighbor_current_mask)
    branch_logits = self.score_head(x[:, 0]).squeeze(-1)  # 取 ego token 的特征
    ...
```

**Score head 的输入是 DiT transformer 所有 block 处理后的 ego token（第 0 个 agent）的特征向量**（维度 `hidden_dim=192`）。注意此时输入 `x` 中已经包含了加噪轨迹的信息，所以 score
<truncated 265 bytes>
_output["branch_logits"].reshape(B, K)  # [B, K]

# 构造 one-hot 标签：pos_k 位置为 1，其余为 0
target_classes_onehot = torch.zeros_like(branch_logits)
target_classes_onehot.scatter_(1, pos_k.unsqueeze(1), 1)

# 使用 Sigmoid Focal Loss
score_loss = py_sigmoid_focal_loss(branch_logits, target_classes_onehot)
```

**关键点：**

| 方面 | 说明 |
|------|------|
| **标签** | one-hot 向量，只有最接近 GT 的 anchor（`pos_k`）为 1，其余为 0 |
| **损失函数** | **Sigmoid Focal Loss**（不是 softmax cross-entropy） |
| **为什么用 focal loss** | 因为 K 个 anchor 中只有 1 个正样本、K-1 个负样本，存在**严重的正负样本不均衡**。Focal loss 的 `gamma=2.0` 会降低 easy negative 的权重，`alpha=0.25` 进一步平衡正负样本 |
| **注意** | 代码中注释掉了原来的 `F.cross_entropy`，说明曾经试过 softmax 分类但换成了 focal loss |

### Score loss 在总 loss 中的权重

```python
# L283-287
loss["ego_planning_loss"] = (
    ego_planning_diffusion_loss
    + planning_hybrid_loss * loss["ego_planning_hybrid_loss"]
    + anchor_score_loss_weight * score_loss      # 默认权重 0.1
)
```

Score loss 以 **0.1 的权重** 加入总的 ego planning loss。

### 监控指标

训练时还记录了几个诊断指标（[L267-280](file:///home/wangchenggang/wcg/Diffusion-Planner_1/diffusion_planner/loss.py#L267-L280)）：

- `anchor_top1_acc`：score head 预测的第 1 名是否是 `pos_k`
- `anchor_top3_acc`：`pos_k` 是否在 top-3 中
- `positive_logit_mean`：正样本 anchor 的 logit 均值
- `negative_logit_mean`：负样本 anchor 的 logit 均值

### 推理时怎么用

在 [decoder.py L163-181](file:///home/wangchenggang/wcg/Diffusion-Planner_1/diffusion_planner/model/module/decoder.py#L163-L181)，推理时所有 K 个 anchor 分支都完成去噪后，再过一遍 `forward_with_logits`（此时 t 设为极小值 1e-3，即近似无噪声状态），用 **`argmax(branch_logits)`** 选出最佳分支作为最终输出。

---

## 👤 User

这是训练曲线，在实际仿真验证训练好的模型时，有以下问题：
1.在一些转弯的场景，规划出的轨迹确实原地不动的；
2.规划的轨迹有时候在接近GT的轨迹和原地不动之间频繁跳变

---

## 🤖 Assistant

让我先仔细看一下训练曲线，然后分析推理代码中的问题。

---


# 查看最新的训练 args
with open('training_log/diffusion-planner-training/2026-05-25-17:11:59/args.json') as f:
    args = json.load(f)
# 打印 anchor 相关参数
anchor_keys = [k for k in args if 'anchor' in k.lower() or 'ego_anchor' in k.lower()]
print('Anchor-related args:')
for k in anchor_keys:
    print(f'  {k}: {args[k]}')
print()
# 也看其他关键参数
for k in ['diffusion_model_type', 'diffusion_supervision_type', 'planning_hybrid_loss', 'batch_size', 'train_epochs', 'learning_rate']:
    if k in args:
        print(f'  {k}: {args[k]}')
")


---



## 🤖 Assistant

现在确认了实际训练配置。让我重新仔细看训练曲线图来做准确诊断：

---

## 🤖 Assistant

Created At: 2026-05-28T08:59:44Z
Completed At: 2026-05-28T09:00:28Z

				The command completed successfully.
				Output:
				=== 服务器实际训练配置 ===
K=64 anchors, batch_size=768, epochs=150
t_range: [0.001, 0.2]
anchor_sampling: t_start=0.2, steps=10

=== 训练时内存/计算开销 ===
每个样本扩展为 B*K = 768*64 = 49152 个分支
  → encoder 只跑 B=768 次
  → decoder 跑 B*K=49152 次

=== t=0.2 时的噪声水平 ===
alpha=0.8114 (信号保留 81.1%)
sigma=0.5845
x_t = 0.811 * anchor + 0.584 * noise

=== Ego State Normalizer ===
mean: dx=3.789, dy=0.092, cos=0.975, sin=0.022
std:  dx=4.122, dy=1.001, cos=0.109, sin=0.192

关键观察: ego mean dx=3.789 意味着每步平均位移约 3.8m
  → 静止 anchor (delta_x ≈ 0) 归一化后 = (0 - 3.789) / 4.122 = -0.919
  → 正常行驶 anchor (delta_x ≈ 3.8) 归一化后 ≈ 0
  → 高速 anchor (delta_x ≈ 8) 归一化后 = (8 - 3.789) / 4.122 = 1.02

=== Score Head 训练挑战 ===
64 个 anchor 中只有 1 个是正样本 → 正负比 1:63
score head 架构: LayerNorm(192) + Linear(192, 1) → 只有 193 个参数
score loss 权重: 0.1 (很小)
focal loss alpha=0.25, gamma=2.0
=== 训练-推理不匹配分析 ===
训练时 score head 输入:
  x_t = 0.811*anchor + 0.584*noise (t ~ [0.001, 0.2])
  → 输入是「加噪的 anchor 轨迹」，anchor 信号占主导

推理时 score head 输入:
  x0 = DPM-Solver 去噪 10 步的结果 (t=0.001)
  → 输入是「去噪后的预测轨迹」，不是 anchor 本身
  → 如果去噪效果差，x0 可能偏离 anchor 很远
  → score head 从未在训练时见过这种分布！



---

## 🤖 Assistant

现在更新分析报告：

---

## 🤖 Assistant

Created At: 2026-05-28T09:01:38Z
Completed At: 2026-05-28T09:01:39Z
Created file file:///home/wangchenggang/.gemini/antigravity-ide/brain/3ea08830-440b-4785-a06f-12fe409ab83d/analysis_results.md with requested content.
If relevant, proactively run terminal commands to execute this code for the USER. Don't ask for permission.

---

## 🤖 Assistant

## 诊断结论

确认了你的训练配置（64 anchor, batch=768, 150 epochs）后，anchor 覆盖度本身没问题。问题出在 **3 个层面**：

### 🔴 核心问题：Score Head 训练-推理分布不匹配

| 阶段 | Score Head 输入 |
|------|----------------|
| **训练** | `0.811 * anchor + 0.584 * noise`（加噪的 anchor 轨迹） |
| **推理** | DPM-Solver 去噪 10 步后的 x0（完全不同的分布） |

训练时 score head 学的是「从 anchor 附近的噪声特征区分正负样本」，推理时却要对「去噪后的预测轨迹」打分——**它从未见过这种输入**，输出完全不可靠。

### 🟡 辅助因素
- **去噪能力不足**：t_start=0.2 只给了 20% 噪声空间，选错 anchor 后 10 步去噪无法矫正
- **Score head 太弱**：193 个参数，loss 权重仅 0.1，1:63 正负比
- 训练曲线中 `anchor_top1_acc` 剧烈波动、`anchor_score_loss` 上升趋势，印证了 score head 没有学好

### 建议优先级

1. **立即可做**：推理时绕过 score head，用轨迹质量（位移、与 route 对齐度）来选分支
2. **重训练**：增大 `t_start` 到 0.5 + 步数到 20，同时加强 score head 容量和 loss 权重
3. **可选**：加推理时时序平滑防跳变

详细方案和代码示例请看上面的分析报告。

---

## 👤 User

Score Head到底应该对去噪后的轨迹打分还是应该对带噪声的轨迹打分更合理？

---

## 🤖 Assistant

这是一个非常好的设计问题，直接关系到整个 anchor 系统能否正常工作。让我从原理层面分析。

---

## 先明确 Score Head 的任务

Score head 在这个系统中的唯一用途是**推理时**选择最佳分支（[decoder.py L181](file:///home/wangchenggang/wcg/Diffusion-Planner_1/diffusion_planner/model/module/decoder.py#L181): `best_k = branch_logits.argmax(dim=1)`）。

所以它要回答的问题是：

> **"在这 K 条去噪后的候选轨迹中，哪一条最好？"**

## 对噪声轨迹打分 vs 对去噪轨迹打分

### 对噪声轨迹打分（当前做法）

训练时 score head 看到的是 `x_t = 0.811 * anchor + 0.584 * noise`，anchor 信号占主导。

**它实际学到的是**：「这个输入像哪个 anchor？那个 anchor 是不是离 GT 最近的？」

本质上是在做 **anchor 身份识别**，而不是轨迹质量评估。

问题在于：
- 推理时所有 K 个分支都完成了去噪，每个分支的去噪结果可能已经**偏离了原始 anchor**很远
- 去噪后不同分支可能**收敛到相似的轨迹**（因为模型会把它们都往场景合理方向修正）
- Score head 在训练时从未见过去噪后的轨迹长什么样，打分不可靠

### 对去噪轨迹打分（更合理）

让 score head 在训练时也看到去噪后（或近似去噪后）的轨迹。

**它学到的是**：「给定场景上下文，这条轨迹质量如何？是否匹配当前场景？」

这是一个**轨迹质量评估器**，与推理时的使用方式完全一致。

## 结论：应该对去噪后的轨迹打分

| 维度 | 对噪声轨迹打分 | 对去噪轨迹打分 |
|------|---------------|---------------|
| **学到的是什么** | anchor 身份识别 | 轨迹质量评估 |
| **训练-推理一致性** | ❌ 不匹配 | ✅ 匹配 |
| **当去噪改变轨迹形状时** | ❌ 打分失效 | ✅ 基于实际结果 |
| **类比** | 根据食材猜菜好不好吃 | 尝了菜再打分 |
| **计算开销** | 便宜（共用 forward） | 贵（
<truncated 955 bytes>
x_start], dim=2)

# 3. 对 pred_x_start（detach！不要让 score loss 影响去噪训练）做 score 评估
with torch.no_grad():
    pred_for_score = pred_full.detach()
score_t = torch.full_like(t, 1e-3)  # 假装干净轨迹
_, branch_logits = model.dit.forward_with_logits(
    pred_for_score.reshape(B*K, P, -1),
    score_t, cross_c, route_lanes, neighbor_current_mask,
)
# 注意：这里 branch_logits 需要有梯度，所以不能全放 no_grad 里
# 实际实现时只 detach 输入，不 detach score_head 的参数
```

关键点：**必须 detach `pred_x_start`**，否则 score loss 的梯度会反向传播到去噪网络，干扰 diffusion 训练。

### 方式 B：直接用 GT 训练 score head

更简单粗暴——训练时直接把 GT 轨迹喂给 score head：

```python
# 每个分支都看到相同的 GT（但不同 anchor 的 neighbor 预测不同）
# score head 要学：「在给定场景下，当前输入轨迹是否像 GT？」
gt_for_score = all_gt.detach()
score_t = torch.full_like(t, 1e-3)
_, branch_logits = model.dit.forward_with_logits(
    gt_for_score.reshape(B*K, P, -1),
    score_t, cross_c, route_lanes, neighbor_current_mask,
)
```

但这个方式有问题：所有 K 个分支看到的 ego GT 都一样（只是 neighbor 不同），score head 很难从中区分。**方式 A 更好**，因为不同 anchor 分支的 `pred_x_start` 是不同的。

### 方式 C（最优但最贵）：训练时也做真正的去噪

训练时对每个 anchor 分支真正跑几步 DPM-Solver 得到 x0，然后对 x0 打分。这最接近推理时的行为，但计算开销极大（K=64，每个分支多步去噪），**不太现实**。

---

## 推荐方案

**方式 A（对 detached pred_x_start 打分）** 是最佳平衡点：
- ✅ 训练-推理分布一致
- ✅ 计算开销可控（只多一次 forward，不需要多步去噪）
- ✅ 不干扰 diffusion 训练（detach 输入）
- ✅ 不同分支的 pred_x_start 有差异，score head 有区分信号

需要我写出具体的代码修改吗？

---

