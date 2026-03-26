# DQN 面试攻略

## 专题顺序

- 第一篇：[强化学习基础与发展沿革](../reinforcement_learning_basics/README.md)
- 第二篇：[DQN](README.md)
- 第三篇：[PPO](../ppo/README.md)

## 这是什么？

DQN 的全称是 Deep Q-Network。

它解决的是一个很现实的问题：

> 表格型 Q-learning 在小状态空间很好用，但一旦状态变成高维特征、图像或复杂连续观测，Q 表就根本存不下，必须换成函数逼近。

DQN 做的事很直接：

- 用神经网络代替 Q 表，近似 `Q(s, a)`
- 保留 Q-learning 的 TD 目标
- 用 replay buffer 和 target network 稳定训练

如果你要讲它在发展线里的位置，可以直接接在上一篇后面：

> Q-learning 解决了“怎么从交互中逼近最优动作价值”，DQN 解决了“状态空间太大时怎么继续学 Q 函数”。

## 核心机制

### 1. DQN 到底在学什么？

它仍然在学动作价值函数，只是把表格变成了网络：

$$ Q_\theta(s, a) $$

训练目标仍然沿用 Bellman 最优方程：

$$ y = r + \gamma \max_{a'} Q_{\theta^-}(s', a') $$

然后最小化：

$$ L(\theta) = (Q_\theta(s, a) - y)^2 $$

这里最值得注意的是：

- 当前网络 `Q_\theta` 负责拟合
- 目标网络 `Q_{\theta^-}` 负责提供相对稳定的 target

### 2. 为什么不能直接把 Q-learning 里的表项改成神经网络输出？

因为这样会同时遇到三件麻烦事：

- 输入样本强相关，连续采样的数据分布很窄
- target 也在跟着当前网络一起变
- bootstrap + off-policy + function approximation 叠在一起，本来就容易不稳

DQN 的核心贡献，不是“用了神经网络”这么简单，而是给这套训练过程加了两个稳定器。

### 3. replay buffer 在稳什么？

replay buffer 的作用是把连续轨迹打散，近似恢复 i.i.d.。

在 [minimal.py](minimal.py) 里，buffer 核心逻辑就是：

```python
class ReplayBuffer:
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
```

这里两段分别在做：

- `push(...)`：把交互得到的 transition 存起来
- `sample(...)`：从历史数据里随机抽小批量训练

这意味着模型更新时不会只盯着“刚才那几步”的局部相关样本。

### 4. target network 在稳什么？

如果 target 也由正在更新的同一套参数直接给出，等于你一边改答案、一边拿新答案监督自己，很容易震荡。

所以 DQN 会维护两套网络：

- `q_net`：当前网络，负责学习
- `target_net`：目标网络，隔一段时间才同步一次

代码里最关键的训练步骤是：

```python
q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
with torch.no_grad():
    next_q_values = target_net(next_states).max(dim=1).values
    targets = rewards + gamma * (1.0 - dones) * next_q_values
```

这里每一行都很关键：

- `q_net(states)`：当前网络给所有动作打分
- `gather(...)`：只取 batch 里真正执行过的动作对应的 Q 值
- `target_net(next_states).max(...)`：目标网络估计下一状态最优动作价值
- `(1.0 - dones)`：终止状态不再 bootstrap

你可以把 DQN 记成一句话：

> 用 replay buffer 打散数据，用 target network 稳住 target，用神经网络逼近 Q 函数。

### 5. 为什么 DQN 通常只适合离散动作？

因为它每一步都要算：

$$ \max_a Q(s, a) $$

如果动作空间是离散的，这个 `max` 很直接。

如果动作是连续向量，就没法像枚举离散动作那样简单做 `argmax`。这时通常转向 DDPG、TD3、SAC 之类的 actor-critic 路线。

### 6. 一个最小 DQN 网络长什么样？

在这个仓库里我故意没上 Atari，而是保留一个最小 `LineWorld` 环境，状态用 one-hot 表示，方便把重点放回算法本身。

Q 网络骨架就三层：

```python
class QNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
```

这段代码要这样讲：

- 第一层把状态特征投到隐藏空间
- `ReLU()` 提供非线性
- 最后一层直接输出每个动作的 Q 值

注意这里最后没有 `softmax`，因为 Q 值不是概率分布。

## 面试高频问题

### 1. DQN 为什么是 off-policy？

因为它学的是最优 Q 函数，可以复用历史策略采到的数据，不要求当前更新必须只用当前策略的数据。

### 2. replay buffer 为什么有效？

它主要解决样本相关性强、覆盖窄的问题，也提升了样本复用率。

### 3. target network 为什么有效？

因为 bootstrap target 不再每一步都和当前网络同步漂移，训练目标更稳。

### 4. DQN 能不能直接处理连续动作？

通常不行。核心障碍是连续动作下 `max_a Q(s, a)` 不好做。

### 5. DQN 的经典缺陷是什么？

一个很典型的问题是 Q 值高估。后来 Double DQN 就是在修这个问题。

### 6. DQN 和 policy gradient 最大差异是什么？

- DQN 是 value-based，直接学动作价值
- policy gradient 直接学策略分布

它们的优化对象不同。

## 最小实现

完整代码见：[minimal.py](minimal.py)。

这份最小实现保留了 DQN 最关键的五块：

- `LineWorld`：提供状态转移和奖励
- `ReplayBuffer`
- `QNet`
- `train_step`
- target network 同步逻辑

主训练循环如下：

```python
while not done:
    action = select_action(q_net, obs, epsilon=epsilon)
    next_obs, reward, done = env.step(action)
    replay_buffer.push(obs, action, reward, next_obs, float(done))

    if len(replay_buffer) >= batch_size:
        batch = replay_buffer.sample(batch_size)
        train_step(q_net, target_net, optimizer, batch, gamma=gamma)
```

对着这段代码讲时，重点要落在：

- `select_action(...)`：训练期用 epsilon-greedy 保证探索
- `push(...)`：把 transition 存进 buffer
- `sample(...)`：随机采样小批量
- `train_step(...)`：用 TD target 更新当前网络

然后再接一句：

```python
if global_step % sync_every == 0:
    target_net.load_state_dict(q_net.state_dict())
```

这就是 target network 周期同步。

如果现场时间有限，DQN 最值得手写的其实就是：

1. `QNet.forward`
2. `ReplayBuffer.sample`
3. `train_step`

这三块写对了，算法主线就已经完整了。

## 工程关注点

### 1. epsilon 调度

探索太少学不到，探索太多又收敛慢。

### 2. replay buffer 大小

- 太小：样本覆盖差
- 太大：旧数据分布和当前策略差太远

### 3. target 同步频率

- 同步太勤：稳定作用不明显
- 同步太慢：target 过旧

### 4. 损失和奖励尺度

工程里常见做法有：

- Huber loss
- reward clipping
- gradient clipping

## 常见坑点

### 1. 输出层后面接 `softmax`

这会把 Q 值错误地当成概率。

### 2. 忘了对 terminal state 关掉 bootstrap

done 状态后面不该再接未来价值。

### 3. 目标网络没有 `detach`

如果 target 还能回传梯度，逻辑就变了。

### 4. 只说“DQN = Q-learning + MLP”

这太浅了。真正该强调的是 replay buffer 和 target network。

## 面试时怎么讲

可以这样压缩成一段：

1. Q-learning 在小状态空间里用表格学 `Q(s, a)` 很有效。
2. 状态空间一大，表格装不下，就用神经网络逼近 Q 函数，这就是 DQN。
3. 直接这么做会不稳，因为样本相关、target 漂移、bootstrap 本身也容易震荡。
4. DQN 用 replay buffer 打散样本，用 target network 稳住 target。
5. 它适合离散动作场景，连续动作通常要走 actor-critic 路线。

## 延伸阅读

- Mnih et al., *Playing Atari with Deep Reinforcement Learning*
- Mnih et al., *Human-level Control through Deep Reinforcement Learning*
- Van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning*
