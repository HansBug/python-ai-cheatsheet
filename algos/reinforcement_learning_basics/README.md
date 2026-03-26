# 强化学习基础与发展沿革 面试攻略

## 专题顺序

- 第一篇：[强化学习基础与发展沿革](README.md)
- 第二篇：[DQN](../dqn/README.md)
- 第三篇：[PPO](../ppo/README.md)

## 这是什么？

这是强化学习专题的第一篇，先把最容易被追问的地基搭起来：

- 强化学习到底在解什么问题
- 状态、动作、奖励、回报这些概念怎么区分
- value / Q / advantage 分别是什么
- on-policy / off-policy、value-based / policy-based 怎么串起来
- 从 bandit、动态规划、TD、Q-learning，一路发展到 DQN、PPO 的主线是什么

如果只用一句话来讲：

> 强化学习是在“不知道最优决策规则”的情况下，让智能体通过和环境交互，用回报信号反推什么行为值得做。

这一篇先把语言体系和发展脉络讲清楚，后两篇再落到 [DQN](../dqn/README.md) 和 [PPO](../ppo/README.md)。

## 核心机制

### 1. 强化学习的基本闭环是什么？

最小交互闭环是：

```text
state s_t -> agent 选 action a_t -> environment 返回 reward r_t 和 next state s_{t+1}
```

它和监督学习最大的不同，不是“有模型没模型”，而是：

- 监督学习通常直接给正确标签
- 强化学习通常只给延迟回报
- 一个动作好不好，往往要看后面一串状态转移

所以强化学习的核心不是拟合静态映射，而是优化序列决策。

### 2. MDP 是什么？为什么几乎所有 RL 面试都会先问它？

标准强化学习建模通常写成马尔可夫决策过程（MDP）：

$$ (\mathcal{S}, \mathcal{A}, P, R, \gamma) $$

这里最重要的是“马尔可夫性”：

> 给定当前状态 $s_t$ 后，未来和过去历史无关。

直觉上它是在说：

- 当前状态必须已经足够概括决策所需信息
- 如果状态定义不完整，很多 RL 算法的假设就会被破坏

回报通常写成：

$$ G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} $$

其中：

- `reward` 是单步反馈
- `return` 是从当前时刻往后的累计目标
- `gamma` 控制长期收益的重要性

### 3. value / Q / advantage 分别在回答什么问题？

这几个量经常被混在一起背，但其实各自回答的是不同问题。

状态价值函数：

$$ V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s] $$

动作价值函数：

$$ Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a] $$

优势函数：

$$ A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s) $$

一句话记忆：

- `V` 问的是“站在这个状态，整体值多少钱”
- `Q` 问的是“在这个状态先做这个动作，值多少钱”
- `A` 问的是“这个动作比当前平均水平好多少”

PPO 里经常直接优化 advantage；DQN 里更直接盯 `Q`。

### 4. Bellman 方程为什么是 RL 的骨架？

因为它把“未来回报”拆成了“当前奖励 + 下一步价值”。

最常见的 Bellman 最优方程写法是：

$$ Q^\*(s, a) = \mathbb{E}[r + \gamma \max_{a'} Q^\*(s', a') \mid s, a] $$

如果你把这个式子翻成代码，最小版更新其实就这几行。下面这段来自 [minimal.py](minimal.py)：

```python
best_next_q = 0.0 if done else float(np.max(self.q_table[next_state]))
td_target = reward + self.gamma * best_next_q
td_error = td_target - self.q_table[state, action]
self.q_table[state, action] += self.lr * td_error
```

这四行分别在做什么：

- `best_next_q`：如果还没结束，就拿下一状态最好的动作价值
- `td_target`：把“即时奖励 + 折扣后的未来价值”拼成 TD target
- `td_error`：当前估计和 target 的差
- 最后一行：按学习率把表项往正确方向推一步

所以很多 RL 算法虽然外观看起来差很多，但骨架仍然绕不开 Bellman 递推。

### 5. RL 方法通常怎么分？

面试里最常见的划分有两组。

第一组：按学什么分。

- value-based：直接学价值函数，典型是 Q-learning、DQN
- policy-based：直接学策略分布，典型是 REINFORCE
- actor-critic：同时学策略和价值，典型是 A2C、PPO

第二组：按数据从哪来分。

- on-policy：更新时依赖当前策略采到的数据，典型是 SARSA、PPO
- off-policy：可以复用旧策略甚至别的策略采到的数据，典型是 Q-learning、DQN、SAC

这两组不要混着背。`value-based / policy-based` 讲的是优化对象；`on-policy / off-policy` 讲的是数据复用方式。

### 6. 发展沿革应该怎么讲，才不像背年表？

建议按“解决了上一代什么问题”来讲。

#### 1. Multi-Armed Bandit

最早只讨论“选哪个臂收益高”，没有状态转移。

它解决的是 exploration vs exploitation 的最简单版本。

#### 2. Dynamic Programming

在已知环境转移和奖励时，可以做 policy iteration / value iteration。

问题是：

- 需要环境模型
- 状态空间稍大就不现实

#### 3. Monte Carlo 和 Temporal Difference

这一步开始摆脱必须知道模型的限制。

- Monte Carlo：等整条轨迹结束后再回看
- TD：走一步就能用 bootstrap 更新

TD 的意义在于更在线、更高效，它直接通向 Q-learning 一支。

#### 4. Q-learning / SARSA

表格型 RL 开始成熟。

- SARSA 是 on-policy
- Q-learning 是 off-policy，并直接逼近最优动作价值

如果状态空间不大，表格法已经能解决不少问题。

#### 5. DQN

一旦状态变成图像、连续特征，Q 表就装不下了。

于是开始用神经网络逼近 `Q(s, a)`，再配上 replay buffer 和 target network 稳住训练。这就是 [DQN](../dqn/README.md)。

#### 6. Policy Gradient / Actor-Critic / PPO

value-based 方法在连续动作上不方便做 `argmax_a Q(s, a)`，于是直接学策略分布变得更自然。

- REINFORCE：直接策略梯度，但方差大
- Actor-Critic：用 critic 降低方差
- TRPO / PPO：让策略更新别跨太大步

这条线最后就落到了 [PPO](../ppo/README.md)。

#### 7. 后续工程化和大模型时代

再往后常见延伸有：

- DDPG / TD3 / SAC：处理连续动作
- model-based RL：提升样本效率
- RLHF：把 PPO、reward model 等方法接到大模型对齐里

所以面试时不要把 PPO 只当“控制领域算法”，它在大模型训练里也非常重要。

## 面试高频问题

### 1. reward 和 return 的区别是什么？

- `reward` 是单步反馈
- `return` 是未来累计回报

很多优化目标看起来在最大化 reward，本质上最大化的是 expected return。

### 2. 为什么要引入折扣因子 `gamma`？

- 防止无限时域回报发散
- 控制智能体更看重短期还是长期收益
- 在很多任务里也更符合“越远的不确定性越大”的直觉

### 3. on-policy 和 off-policy 最关键的差异是什么？

是否能高效复用旧数据。

- on-policy 更稳、更直接，但样本利用率差
- off-policy 更省样本，但分布漂移和训练稳定性问题更重

### 4. 为什么强化学习往往比监督学习更难训？

- 标签不是直接给的
- 回报延迟
- 数据分布会随着策略变化不断变化
- exploration 不足时，模型甚至采不到高价值样本

### 5. advantage 为什么常常比直接用 return 更好？

因为它在问“这个动作比基线好多少”，通常方差更低，更适合做策略梯度更新。

## 最小实现

这一篇不急着上神经网络，先把最小表格型 Q-learning 写明白。完整代码见：[minimal.py](minimal.py)。

环境是一个最简单的 `LineWorld`：

- 状态是一条线上的位置
- 动作只有 `left / right`
- 走到最左端奖励 `-1`
- 走到最右端奖励 `+1`

初始化环境的代码很短，但已经包含了 RL 的全部基本元素：

```python
class LineWorld:
    def reset(self) -> int:
        self.state = self.start_state
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        move = -1 if action == 0 else 1
        next_state = min(max(self.state + move, 0), self.length - 1)
        self.state = next_state
```

这里：

- `reset` 定义 episode 起点
- `step` 根据动作推进环境
- 返回值里同时给了 next state、reward、done

主训练循环也很值得面试时手写：

```python
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
```

这一小段就是最朴素的 RL 闭环：

- 先和环境交互
- 再用交互数据更新策略或价值
- 下一轮继续采样

如果这段你能现场写对，后面再换成 DQN、PPO，本质上都是在替换 `agent.update(...)` 的内部实现。

## 工程关注点

### 1. 奖励设计

奖励不是越复杂越好，关键是：

- 和真实目标一致
- 不要被模型钻空子
- 数值尺度别乱飞

### 2. 状态定义

如果状态里缺关键信息，环境对 agent 来说就不再近似马尔可夫，很多算法会直接变难训。

### 3. exploration 策略

采不到好轨迹，再强的更新公式也没用。

最常见的工程手段有：

- epsilon-greedy
- entropy bonus
- action noise

### 4. 样本效率

真实环境交互很贵时，样本效率往往比单步算力更重要。

这也是为什么 off-policy 方法和 model-based 方法一直有人做。

## 常见坑点

### 1. 把监督学习里的“标签”思维直接搬到 RL

RL 很多时候没有直接正确答案，只有延迟反馈。

### 2. 把 `V`、`Q`、`A` 混着说

这会直接暴露你没有真的理解优化对象。

### 3. 把 on-policy / off-policy 和 actor / critic 混为一谈

这两组维度完全不同。

### 4. 只会背 PPO、DQN 名字，不会说它们分别解决了哪类问题

面试官通常更看这个。

## 面试时怎么讲

可以按这条线讲：

1. 强化学习是在序列决策里最大化长期回报，不是拟合静态标签。
2. 形式化通常用 MDP，核心变量是 state、action、reward、return。
3. `V`、`Q`、`A` 分别回答状态值、动作值、相对优势。
4. Bellman 方程把长期目标拆成一步奖励和下一步价值，是很多算法的骨架。
5. 发展上先有 bandit、DP、MC、TD、Q-learning，再到深度强化学习里的 DQN 和策略优化里的 PPO。

这样讲，基本概念和发展脉络会一起落地。

## 延伸阅读

- Sutton and Barto, *Reinforcement Learning: An Introduction*
- Richard S. Sutton, *Learning to Predict by the Methods of Temporal Differences*
- Watkins and Dayan, *Q-learning*
