# 训练曲线分析：为什么对齐后的 score head 反而更差？

## 三个模型对比

| 模型 | Anchor 数 | Score Head 训练输入 | Score Head 推理输入 |
|---|---|---|---|
| `ego_anchor_64` | 64 | noisy trajectory | 去噪后 x0 |
| `ego_anchor_128` | 128 | noisy trajectory | 去噪后 x0 |
| `ego_anchor_128_score_head` | 128 | 单步 pred_x_start（对齐版） | 去噪后 x0 |

---

## 问题一：为什么 `ego_anchor_128_score_head`（对齐版）表现最差？

### 核心原因：第二次 forward 破坏了去噪主任务的训练效率

> [!CAUTION]
> 对齐 score head 的做法引入了**第二次完整 DiT forward**，这个代价不仅仅是"计算量翻倍"那么简单，它从多个维度损害了去噪主任务的训练。

#### 1. 显存压力导致有效 batch size 下降

在 [decoder.py L246-273](file:///home/wangchenggang/wcg/Diffusion-Planner_1/diffusion_planner/model/module/decoder.py#L246-L273) 中，对齐版本需要：

- 第一次 forward：正常去噪（B\*K 规模的完整 DiT forward + backward）
- 第二次 forward：`forward_with_logits`（又一次 B\*K 规模的 DiT forward，虽然 `no_grad` 但仍占用显存存 activation）

当 K=128 时，实际 batch 是 B\*128。两次 forward 的**峰值显存**远高于原版，这意味着你要么：
- 降低了 B（有效场景多样性下降）
- 开启了 gradient checkpointing 等节省显存的手段（训练变慢）

**无论哪种，去噪主任务的训练质量都会受损。**

#### 2. 单步 pred_x_start 在训练早期极其不准确

```
训练早期:  DiT 还没学好 → pred_x_start ≈ 噪声 → score head 学到的是垃圾信号
训练中期:  DiT 逐渐改善 → pred_x_start 分布快速变化 → score head 追逐移动目标
训练后期:  DiT 较好 → pred_x_start 比较准 → 但 score head 已经被前期错误信号带偏
```

这是一个典型的 **non-stationary target** 问题。score head 的监督信号（pred_x_start 的质量）依赖于 DiT backbone 的训练进度，而 backbone 本身在不断变化。

相比之下，原版（`ego_anchor_128`）的 score head 直接对 noisy anchor 轨迹打分，输入分布是**稳定的**（由 anchor + SDE noise schedule 决定），不依赖 backbone 质量。

#### 3. detach 制造的梯度断裂反而是双刃剑

```python
# decoder.py L257-262
with torch.no_grad():
    pred_x_start = self._sde.transform(...)
    pred_full = torch.cat([...]).reshape(B, P, -1)

# L266-272: forward_with_logits with detach_features=True
```

你的设计意图是"不让 score loss 影响 backbone"，这本身是对的。但 **完全的梯度断裂意味着 score head 永远无法反向告诉 backbone 需要什么样的特征**。

在原版中，虽然 score head 对 noisy 轨迹打分看似"不对齐"，但 backbone 的 feature 和 score head 的 feature 来自**同一次 forward**。即使 score head 的梯度被 detach 了，backbone 也自然会学到对不同 anchor 分支产生有区分性的 feature（因为去噪 loss 本身就要求 backbone 对不同 anchor 区分对待）。

而对齐版的第二次 forward 是完全独立的，feature 质量完全取决于一步 pred_x_start 的质量——这是一个更弱的信号。

#### 4. "对齐"本身可能是个伪需求

> [!IMPORTANT]
> 训练-推理 mismatch 不一定是负面的。原版的"不对齐"可以理解为一种**隐式数据增强**。

原版的 score head 在训练时看到的是带噪声的轨迹，推理时看到的是干净的轨迹。这意味着：
- 如果 score head 能从**噪声轨迹**中学会识别"哪个 anchor 更好"，那它在**干净轨迹**上只会表现得**更好**
- 噪声输入迫使 score head 学到更鲁棒的判别特征，而不是过拟合到精确的轨迹形状

这类似于"在模糊图片上训练分类器，在清晰图片上推理"——分类器会学到更本质的特征。

---

## 问题二：为什么 128 anchors 在一些指标上比 64 anchors 差？

### 1. anchor 间距变小 → 分类任务变难

从训练曲线中可以观察到：

- `anchor_best_distance`（128）< `anchor_best_distance`（64） — 128 个 anchor 覆盖更密，最近 anchor 到 GT 更近 ✅
- `anchor_margin`（128）<< `anchor_margin`（64） — **但最近和次近 anchor 之间的距离也变得很小** ❌

当两个 anchor 在轨迹空间中几乎重叠时：
- 正样本标签变得不稳定（微小噪声就可能改变 `pos_k`）
- score head 被要求在**几乎相同的输入**上输出**不同的分数**，这是一个 ill-defined 的任务

```
K=64:  anchor 间距适中 → score head 容易区分 → top1_acc 较高
K=128: anchor 间距太小 → 很多 anchor 对 GT 距离差不多 → score head 学不好
```

### 2. B\*K 扩展导致训练效率下降

| | K=64 | K=128 |
|---|---|---|
| Decoder 实际 batch | B×64 | B×128 |
| 显存占用 | 高 | **极高** |
| 可能的 B | 较大 | 较小 |
| 场景多样性 | 较好 | **较差** |

当 K 翻倍时，要么 B 减半（场景多样性减半），要么显存不够。**去噪主任务的训练质量直接受损。**

### 3. Focal loss 的正负样本比恶化

```
K=64:   正:负 = 1:63
K=128:  正:负 = 1:127
```

虽然 focal loss 设计了 alpha 和 gamma 来缓解不平衡，但 1:127 的比例仍然比 1:63 更困难。score head 会倾向于预测所有分支都是负样本。

从曲线中 `negative_logit_mean` 和 `positive_logit_mean` 的对比可以验证这一点——128 anchors 时正负 logit 的分离度可能更差。

### 4. 推理时选错的代价更大

即使 score head 的 top-1 accuracy 相同（比如都是 60%），128 anchors 选错时的后果更严重：
- K=64 时，随机选错的期望距离 ∝ anchor 间距
- K=128 时，虽然间距更小，但可能选到完全不同方向的 anchor

---

## 建议

### 短期：不要追求训练-推理完全对齐

当前原版（`ego_anchor_128`，noisy 轨迹打分）的方案已经是可行的。"不对齐"实际上是一种有益的正则化。

### 中期：优化 anchor 数量

> [!TIP]
> 更多的 anchor 不一定更好。考虑从以下角度优化：

1. **动态选择 top-M anchors**：128 anchors 不需要全部参与训练。在 loss.py 中先算 `anchor_dist`，只选距离最近的 M=8~16 个 anchor 构造分支，其余的 skip。这样 B\*M 远小于 B\*128，训练效率大幅提升。

2. **层次化 anchor**：先用粗粒度 anchor（比如 8 个方向类别）做第一级筛选，再在选中类别内用细粒度 anchor 做第二级。

3. **适当的 K**：从曲线看，K=64 在大多数指标上已经优于 K=128。可以尝试 K=32 或 K=48，找到 anchor 覆盖度和训练效率的甜点。

### 长期：改进 score head 架构

如果确实想让 score head 在 clean 轨迹上训练：
- 不要用第二次 DiT forward，而是在第一次 forward 的 feature 基础上加一个轻量级的 MLP head
- 用 stop-gradient 只阻断 score loss 到 backbone 的梯度，但不要重跑 feature extraction
- 参考 Diffusion Drive 的做法：score head 直接在 backbone feature 上工作，不需要重构 pred_x_start

