# LayerNorm / RMSNorm 面试攻略

## 这是什么？

这是 Transformer 里负责“稳定训练、控制特征尺度”的归一化模块。

如果只用一句话来讲：

> LayerNorm 是对每个 token 自己的隐藏维做标准化；RMSNorm 则进一步简化，只按均方根缩放，不再减去均值。

在早期 Transformer 里，最经典的是 LayerNorm。到了现代 LLM，RMSNorm 变得越来越常见。

## 核心机制

### 1. 为什么 Transformer 需要归一化？

Transformer 很深，而且每一层都有：

- attention
- MLP
- residual connection

如果每层输出的数值范围不停漂移，训练会变得不稳定。

归一化模块的目标就是：

- 控制不同层之间的数值尺度
- 改善梯度传播
- 提高训练稳定性

### 2. LayerNorm 到底在归一化什么？

这是高频题。

LayerNorm 不是按 batch 归一化，也不是按整个序列归一化。

它是：

> 对单个 token 的隐藏维向量，在最后一个维度上做归一化。

如果一个 token 的隐藏状态是：

$$ x \in \mathbb{R}^{d} $$

那么它先计算均值和方差：

$$ \mu = \frac{1}{d}\sum_{i=1}^{d} x_i $$

$$ \sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2 $$

然后做标准化：

$$ \hat{x}_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}} $$

最后再加可学习参数：

$$ y_i = \gamma_i \hat{x}_i + \beta_i $$

这里：

- $\gamma$ 是缩放参数
- $\beta$ 是平移参数

### 3. 直觉上怎么理解 LayerNorm？

你可以把它理解成：

> 先把一个 token 自己内部的各维特征拉回到一个更稳定的尺度，再交给后面的层去处理。

注意它管的是“这个 token 内部各维度的分布”，不是样本和样本之间的分布。

### 4. RMSNorm 又是什么？

RMSNorm 可以看成 LayerNorm 的简化版。

它不再显式减去均值，只保留按均方根做缩放：

$$ \mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\varepsilon} $$

$$ \hat{x}_i=\frac{x_i}{\mathrm{RMS}(x)} $$

$$ y_i=\gamma_i \hat{x}_i $$

很多实现里 RMSNorm 没有独立的 $\beta$ 偏置项。

### 5. 为什么 RMSNorm 不减均值也能工作？

这是现代 LLM 高频题。

核心点是：

> 对很多大模型来说，更关键的是控制向量整体尺度，而不一定非要把均值挪到 0。

RMSNorm 仍然能稳定数值范围，但计算更简单，开销更低，也更贴合现代 LLM 的工程实现。

### 6. LayerNorm 和 RMSNorm 的核心差异是什么？

最简洁的对比是：

- LayerNorm：减均值，再除标准差
- RMSNorm：不减均值，只除均方根

所以 LayerNorm 更像完整标准化；RMSNorm 更像只做长度归一化。

### 7. shape 怎么记？

如果输入是：

```text
X: [B, T, D]
```

无论是 LayerNorm 还是 RMSNorm，通常都是沿最后一个维度 $D$ 做归一化：

```text
mean / var / rms: [B, T, 1]
output          : [B, T, D]
gamma / beta    : [D]
```

记忆重点是：

> 对每个 token，沿隐藏维做归一化，不改变整体 shape。

## 面试高频问题

### 1. LayerNorm 和 BatchNorm 有什么区别？

BatchNorm 依赖 batch 统计量；LayerNorm 对单个样本内部做归一化。

Transformer 里更适合用 LayerNorm，因为：

- 序列长度不固定
- batch 统计量不稳定
- 自回归推理时 batch 很小甚至是 1

### 2. LayerNorm 是按 token 归一化还是按序列归一化？

更准确地说，是对每个 token 的隐藏维向量归一化。

如果输入是 $[B, T, D]$，通常是在 $D$ 这个维度上算均值和方差。

### 3. 为什么现代 LLM 很多用 RMSNorm？

常见回答：

- 更简单
- 计算更省
- 工程实现更直接
- 实践里效果通常足够好

### 4. RMSNorm 为什么没有减均值？

因为它的目标重点不是“把分布中心移到 0”，而是“控制整体尺度”。

### 5. LayerNorm 一定比 RMSNorm 强吗？

不一定。

LayerNorm 更完整，但 RMSNorm 更轻，很多现代 LLM 实践里已经证明它足够有效。

### 6. Pre-LN 和 Post-LN 是什么？

这是 Transformer 高频追问。

- Pre-LN：先做归一化，再进 attention / MLP
- Post-LN：先经过子层，再做归一化

现代大模型更常见 Pre-LN，因为训练通常更稳。

### 7. 为什么说 Pre-LN 更稳？

一个常见回答是：

> 因为残差分支上的梯度传播更顺，深层训练时更不容易不稳定。

## 最小实现

### 1. LayerNorm 最小版

```python
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias
```

记忆主线只有三步：

```text
减均值 -> 除标准差 -> 乘 gamma 加 beta
```

### 2. RMSNorm 最小版

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm
```

记忆主线更短：

```text
不减均值 -> 只按 rms 缩放 -> 乘 gamma
```

### 3. 它们的差异怎么直接看？

如果输入是：

```python
x = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 2.0, 2.0, 2.0],
], dtype=torch.float32)
```

那么：

- LayerNorm 输出会是“零均值、单位方差附近”的结果
- RMSNorm 输出则只是把向量长度缩放到更稳定的范围，不保证零均值

## 工程关注点

### 1. 为什么归一化模块对大模型训练这么重要？

因为深层网络里，数值尺度和梯度稳定性是长期问题。归一化位置选得不对，训练就可能更难收敛。

### 2. LayerNorm 和 RMSNorm 的工程差异主要在哪？

- LayerNorm 要算均值和方差
- RMSNorm 只算均方根

所以 RMSNorm 在实现和计算上更轻一些。

### 3. 为什么 LLM 文献里经常顺带聊 Pre-LN？

因为归一化不只是“用哪一种 norm”，还包括“norm 放在子层前还是后”。这会直接影响训练稳定性。

## 常见坑点

### 1. 以为 LayerNorm 是按 batch 归一化

不是。那是 BatchNorm 的思路。

### 2. 记不清 LayerNorm 归一化的维度

面试时一定要说清楚：通常是最后一个隐藏维。

### 3. 以为 RMSNorm 只是把名字换了

不是。它真的去掉了减均值这一步。

### 4. 把 RMSNorm 说成“效果一定更好”

不严谨。更准确的说法是：它更简单，在很多现代 LLM 里实践效果足够好。

### 5. 只记住公式，不会讲为什么 Pre-LN 更稳

这在大模型面试里很容易被继续追问。

## 面试时怎么讲

如果面试官让你介绍 LayerNorm / RMSNorm，可以按这个顺序讲：

1. 归一化的目标是稳定训练和控制尺度
2. LayerNorm 是对每个 token 的隐藏维做标准化
3. RMSNorm 是简化版，只按均方根缩放，不减均值
4. 现代 LLM 常用 RMSNorm，因为更简单、实现更轻、实践有效
5. 继续延伸时，再聊 Pre-LN / Post-LN 和训练稳定性

一个简洁版本可以直接背：

> LayerNorm 会对单个 token 的隐藏维向量先减均值、再除标准差，然后乘可学习参数；RMSNorm 则去掉减均值，只按均方根缩放。现代 LLM 常用 RMSNorm，因为它更简单，计算更轻，而且通常已经足够稳定。进一步追问时，还要能接到 Pre-LN 比 Post-LN 更稳这个点。

## 延伸阅读

- 下一步可以继续看：KV Cache、MoE、为什么 Pre-LN 更稳
- 对照代码看：[minimal.py](minimal.py)
