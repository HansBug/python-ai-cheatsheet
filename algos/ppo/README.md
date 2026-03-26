# PPO 面试攻略

## 专题顺序

- 第一篇：[强化学习基础与发展沿革](../reinforcement_learning_basics/README.md)
- 第二篇：[DQN](../dqn/README.md)
- 第三篇：[PPO](README.md)

## 这是什么？

PPO 的全称是 Proximal Policy Optimization。

如果你想把它放在强化学习的发展脉络里，最自然的说法是：

> DQN 代表 value-based 的深度强化学习主线，PPO 代表 actor-critic / policy optimization 这条主线里最常见、最好用、也最常被面试问到的工程化算法之一。

它的重要性有两层：

- 在经典强化学习里，PPO 是连续控制和 on-policy 训练的常用基线
- 在大模型对齐里，PPO 也是 RLHF 时代非常高频的关键词

## 核心机制

### 1. 为什么 policy gradient 需要 PPO 这种“限步器”？

最原始的策略梯度目标很直接：

$$ \nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A_t] $$

问题是，直接按这个方向猛走，很容易出现：

- 新旧策略差太大
- 一次更新把已经不错的策略直接推坏
- 训练震荡

TRPO 的思路是显式约束新旧策略距离，但实现比较重。PPO 的思路更务实：

> 不强行解复杂约束优化，而是用 clip 把策略更新限制在一个“别走太远”的区间里。

### 2. PPO 的核心目标函数是什么？

它最核心的一版写法是：

$$ L^{\mathrm{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) A_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right] $$

其中：

$$ r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)} $$

这里最值得你现场解释的是：

- `ratio` 表示新策略相对旧策略，把当前动作概率放大了多少
- 如果 `ratio` 偏离 1 太多，就会被 clip 回来
- `min(...)` 的作用是防止目标被“虚高”的更新方向利用

### 3. advantage 为什么在 PPO 里这么重要？

PPO 不是直接拿 return 去推策略，而是更偏向使用 advantage：

$$ A_t = Q(s_t, a_t) - V(s_t) $$

直觉上它在问：

> 当前动作到底比“这个状态下的平均水平”好多少？

这能明显降低策略梯度的方差。

实际实现里，通常不会显式先学出完整的 `Q`，而是用 GAE 近似 advantage。

### 4. GAE 在代码里通常长什么样？

在 [minimal.py](minimal.py) 里，我把 GAE 直接写进 rollout 收集过程里：

```python
for t in reversed(range(rollout_steps)):
    mask = 1.0 - dones[t]
    delta = rewards[t] + gamma * next_values[t] * mask - values[t]
    gae = delta + gamma * gae_lambda * mask * gae
    advantages[t] = gae
```

这几行分别表示：

- `delta`：一步 TD 残差
- `gae`：把未来残差按 `gamma * lambda` 递推累计
- `mask`：如果 episode 结束，就停止继续 bootstrap

这也是 PPO 里 actor 和 critic 结合得最紧的地方：

- critic 提供 `value`
- actor 更新时用 `advantage`

### 5. actor-critic 在 PPO 里是怎么落成网络结构的？

最小版网络一般是一个共享骨干，再分 policy head 和 value head：

```python
class PolicyValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
```

这段代码要这样解释：

- `backbone`：先抽共享状态特征
- `policy_head`：输出动作 logits，代表策略
- `value_head`：输出标量状态价值，作为 critic

所以 PPO 并不是“只有策略网络”，而是一个标准 actor-critic 结构。

### 6. clip 目标在更新代码里怎么写？

PPO 最关键的几行是：

```python
logits, values = model(states)
dist = Categorical(logits=logits)
log_probs = dist.log_prob(actions)
ratios = torch.exp(log_probs - old_log_probs)

unclipped = ratios * advantages
clipped = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
policy_loss = -torch.minimum(unclipped, clipped).mean()
```

这里每一行分别在做：

- `dist.log_prob(actions)`：算新策略下这些旧动作的对数概率
- `ratios`：恢复成新旧策略概率比
- `unclipped`：原始策略梯度目标
- `clipped`：把更新幅度裁到可接受范围
- `minimum(...)`：取更保守的一边

如果面试官只问“PPO 为什么稳”，你就围绕这几行讲。

## 面试高频问题

### 1. PPO 为什么要用 old policy？

因为 `ratio` 必须拿当前策略和采样数据时的旧策略做比较，才能知道这次更新走了多远。

### 2. PPO 为什么通常是 on-policy？

因为 rollout 数据是按当前旧策略采的，更新完策略后，这批数据很快就过时了，不能像 DQN 那样长期反复复用。

### 3. PPO 为什么比纯 policy gradient 更稳？

- 有 advantage 降方差
- 有 critic 做基线
- 有 clip 限制更新步长

### 4. PPO 和 TRPO 的关系是什么？

可以理解成：

- TRPO 更“理论严格”，显式做信赖域约束
- PPO 更“工程实用”，用 clip 或 penalty 近似这种约束

### 5. PPO 能处理连续动作吗？

能。把 `Categorical` 换成高斯策略分布就行。

### 6. 为什么常对 advantage 做标准化？

为了让优化尺度更稳定，减少不同 batch 间梯度幅度的波动。

## 最小实现

完整代码见：[minimal.py](minimal.py)。

这份实现保留了 PPO 最关键的五块：

- `LineWorld`
- `PolicyValueNet`
- `collect_rollout`
- GAE 计算
- `ppo_update`

rollout 收集和更新是两段最值得手写的代码。

先看 rollout：

```python
with torch.no_grad():
    logits, value = model(obs_tensor)
    dist = Categorical(logits=logits)
    action_tensor = dist.sample()
    log_prob = dist.log_prob(action_tensor)
```

这里：

- `logits` 来自 actor
- `value` 来自 critic
- `sample()` 用当前策略采动作
- `log_prob` 会被存下来，后面算新旧策略比值

再看更新：

```python
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这三项分别对应：

- `policy_loss`：策略优化目标
- `value_loss`：critic 回归误差
- `entropy`：鼓励探索，避免策略过早塌缩

如果你能把 rollout、GAE、clip loss 这三块讲明白，PPO 主线就已经够面试用了。

## 工程关注点

### 1. rollout 长度

- 太短：估计方差大
- 太长：策略更新滞后

### 2. clip 系数

- 太小：更新过于保守
- 太大：约束不明显

### 3. value loss 权重

critic 太弱，advantage 会很吵；critic 太强，也可能压过 policy update。

### 4. entropy bonus

早期探索期通常更重要，后期可适当减弱。

## 常见坑点

### 1. 没有保存 old log prob

那就没法算 ratio。

### 2. 更新前后混用了不同策略的数据

这会把 PPO 的 on-policy 前提搞乱。

### 3. 只讲 clip，不讲 actor-critic

PPO 不是只有一个 clip loss，它通常是完整 actor-critic 训练框架。

### 4. 把 advantage 直接写成 return

这样能跑，但通常方差更大，也不够像标准 PPO。

## 面试时怎么讲

可以压缩成下面这条线：

1. 纯 policy gradient 直接更新策略，但方差大、步子容易迈太大。
2. PPO 在 actor-critic 框架里，用 critic 估 value，用 advantage 降方差。
3. 它再通过新旧策略概率比 `ratio` 和 clip 目标，把每次更新限制在相对可信的范围内。
4. 这让 PPO 既保留了策略优化方法处理连续动作的优势，又比裸 policy gradient 更稳。
5. 在经典 RL 和 RLHF 里，PPO 都是高频基线。

## 延伸阅读

- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*
- Schulman et al., *Trust Region Policy Optimization*
- Schulman et al., *Proximal Policy Optimization Algorithms*
