# 仿真问题诊断分析（64 Anchor 版本）

## 实际训练配置

| 参数 | 值 |
|------|-----|
| `num_ego_anchors` | **64** |
| `batch_size` | 768 |
| `train_epochs` | 150 |
| `ego_anchor_t_min` | 0.001 |
| `ego_anchor_t_max` | 0.2 |
| `anchor_sampling_t_start` | 0.2 |
| `anchor_sampling_steps` | 10 |
| `anchor_score_loss` weight | 0.1 |
| `diffusion_model_type` | x_start |

## Anchor 覆盖度（OK，不是问题）

64 个 anchor 覆盖了多种场景：
- 直行（多种速度）: 39 个
- 转弯（lateral > 5m）: 25 个
- 大转弯（heading > 45°）: 16 个
- 近乎静止（< 1km/h）: **1 个** (Anchor 4, 位移仅 0.21m)
- 蠕行（1-5km/h）: 3 个 (Anchor 1/31/57)

> [!NOTE]
> Anchor 覆盖度本身没有问题，转弯场景有足够多的 anchor 可选。

---

## 问题现象
1. **转弯场景原地不动**：规划出的轨迹不走
2. **频繁跳变**：在接近 GT 的轨迹和原地不动之间反复切换

---

## 根因分析

### 🔴 根因 1: Score Head 训练-推理分布严重不匹配

这是最核心的问题。

**训练时** score head 看到的输入（[loss.py L200-208](file:///home/wangchenggang/wcg/Diffusion-Planner_1/diffusion_planner/loss.py#L200-L208)）：
```
x_t = 0.811 * anchor_normalized + 0.584 * noise   (t ∈ [0.001, 0.2])
```
Score head 的输入 `x[:, 0]` 来自 DiT 处理加噪后的 anchor 轨迹。**训练时 ego 通道的均值被替换为 anchor 的均值**（L203: `mean[:, 0] = ego_anchor_mean`），所以 score head 学到的是从「anchor 附近的噪声轨迹特征」判断哪个 anchor 最接近 GT。

**推理时** score head 看到的输入（[decoder.py L163-172](file:///home/wangchenggang/wcg/Diffusion-Planner_1/diffusion_planner/model/module/decoder.py#L163-L172)）：
```python
score_t = torch.full((B * K,), 1e-3, device=device)   # 假装 t=0.001
_, branch_logits = self.dit.forward_with_logits(
    x0,        # ← 完全去噪后的轨迹！不是加噪的 anchor！
    score_t,   # ← t=0.001
    ...
)
```
推理时 score head 看到的是 **DPM-Solver 去噪 10 步后的 x0**，这个 x0：
- 每个 anchor 分支去噪后可能收敛到相似的轨迹（都接近场景的合理轨迹）
- 与训练时的输入分布完全不同
- score head 从未在训练中见过这种输入

> [!CAUTION]
> **结果**：score head 在推理时输出不可靠，可能随机选中静止 anchor (Anchor 4) 的分支，导致输出原地不动的轨迹。

### 🔴 根因 2: 去噪能力不足

| 参数 | 值 | 含义 |
|------|-----|------|
| `t_start` | 0.2 | 只从 20% 噪声水平开始 |
| `steps` | 10 | 仅 10 步 DPM-Solver |

在 t=0.2 时，alpha=0.811，**anchor 信号保留了 81%**。这意味着：
- 去噪的起点非常接近 anchor 本身
- 模型只能做很小幅度的修正（从 anchor 微调到合理轨迹）
- 对于转弯场景，如果 score head 错误选中了直行或静止 anchor，**10 步去噪无法将其矫正为转弯轨迹**

### 🟡 根因 3: Score Head 欠拟合

从训练曲线可以看到：
- **`anchor_top1_acc` 剧烈波动**：在 0.2-0.8 之间，说明 score head 学得很不稳定
- **`anchor_score_loss` 上升趋势**：score head 的学习在退化

原因：
1. **正负样本极度不均衡**：64 个 anchor 中 1 正 63 负（1:63）
2. **score head 太简单**：只有 `LayerNorm(192) + Linear(192→1)` = 193 参数
3. **loss 权重太小**：`anchor_score_loss=0.1`，梯度信号被 diffusion loss 淹没
4. **focal loss 在 64 类上效果不佳**：alpha=0.25 对正样本的权重相对于 63 个负样本来说太低

---

## 「原地不动」的具体路径

```
推理时转弯场景:

1. 所有 64 个 anchor 分支都从各自 anchor 出发做 10 步去噪
   → anchor 4 (静止) 从 "不动" 出发，去噪后仍然接近 "不动"
   → anchor 30/36/... (转弯) 从转弯 anchor 出发，去噪后得到合理转弯轨迹

2. Score head 对所有 64 个去噪结果打分
   → 由于训练-推理分布不匹配，打分不可靠
   → 可能给 anchor 4 (静止) 分支最高分

3. best_k = branch_logits.argmax(dim=1)
   → 选中静止分支
   → 输出原地不动的轨迹
```

## 「频繁跳变」的具体路径

```
帧 t:   score head 恰好选中了转弯 anchor → 输出合理轨迹 ✓
帧 t+1: 场景微变，score head 随机波动选中静止 anchor → 输出不动 ✗
帧 t+2: 再次选中转弯 anchor → 又输出合理轨迹 ✓
...

每帧独立推理，没有时序一致性约束 → 频繁跳变
```

---

## 修复建议（按优先级排序）

### 🥇 方案 1: 推理时绕过 score head，用轨迹质量选分支

不依赖不可靠的 score head，改为直接评估每个分支去噪结果的质量：

```python
# decoder.py _anchored_inference() 中，替换 score head 选择逻辑
# 方案A: 选择与 route_lanes 最对齐的分支
# 方案B: 选择位移最合理的分支（排除近乎静止的异常分支）
# 方案C: 用多个分支的加权平均

# 简单示例: 排除静止分支，选位移最大且与 route 一致的
ego_predictions = x0[:, :, 0, :, :2]  # [B, K, T, 2]
displacements = torch.norm(ego_predictions[:, :, -1, :], dim=-1)  # [B, K]

# 排除异常静止分支
valid_mask = displacements > 1.0  # 至少移动1米
if valid_mask.any(dim=1).all():
    branch_logits[~valid_mask] = -float('inf')
    
best_k = branch_logits.argmax(dim=1)
```

### 🥈 方案 2: 增大去噪范围

增大 `t_start` 和去噪步数，让模型有更强的修正能力：

```bash
--anchor_sampling_t_start 0.5   # 从 50% 噪声开始（当前 0.2）
--anchor_sampling_steps 20      # 20 步（当前 10）
```

在 t=0.5 时：
- alpha ≈ 0.44，sigma ≈ 0.90
- anchor 信号只保留 44%，模型有更大的自由度修正轨迹
- 即使选错 anchor，也有可能通过去噪修正到合理轨迹

> [!WARNING]
> 需要同步修改训练时的 `ego_anchor_t_max` 为对应值，否则训练-推理的 t 范围不匹配。

### 🥉 方案 3: 加强 Score Head 训练

如果要保留 score head 方案，需要大幅改进：

1. **增大 score head 容量**:
```python
# 当前: LayerNorm + Linear(192, 1)
# 建议: 
self.score_head = nn.Sequential(
    nn.LayerNorm(hidden_dim),
    nn.Linear(hidden_dim, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, 1),
)
```

2. **增大 score loss 权重**: `--anchor_score_loss 0.5`（当前 0.1）

3. **修复训练-推理不匹配**: 训练时也在去噪后的 x0 上评估 score head：
```python
# 训练时额外做一次无噪声的 forward 来训练 score head
with torch.no_grad():
    # 用 GT 本身作为 x0 来训练 score head
    clean_score_t = torch.full((B*K,), 1e-3, device=device)
    _, clean_logits = model.dit.forward_with_logits(
        all_gt_clean, clean_score_t, cross_c, route_lanes, neighbor_mask
    )
# 用 clean_logits 计算额外的 score loss
```

### 方案 4: 推理时加时序平滑

在 `_anchored_inference` 中加入时序一致性：
```python
# 对 branch_logits 做指数移动平均
momentum = 0.7
if hasattr(self, '_prev_logits') and self._prev_logits is not None:
    branch_logits = momentum * branch_logits + (1 - momentum) * self._prev_logits
self._prev_logits = branch_logits.detach().clone()
```

---

## 建议实施顺序

1. **立即**：实施方案 1 (绕过 score head) → 快速验证去噪本身是否正常
2. **短期**：实施方案 2 (增大 t_start) → 需要重新训练
3. **中期**：实施方案 3 (改进 score head) → 需要重新训练
4. **可选**：方案 4 作为补充
