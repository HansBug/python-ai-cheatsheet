# Self-Attention / Multi-Head Attention 面试攻略

## 这是什么？

这是 Transformer 里最核心的模块之一。

如果只用一句话来讲：

> Self-Attention 的作用，是让序列里的每个 token 都能“按需查看”其他 token，并把最相关的信息加权汇总回来。

比如一句话里有代词“它”，模型要判断“它”指代谁，只看当前位置本身往往不够，必须结合前后文。Self-Attention 干的就是这件事。

Multi-Head Attention 则是在这个基础上做并行化：

> 不只用一种“相关性视角”看上下文，而是让多个 head 同时学习不同的关注模式。

## 核心机制

### 1. 为什么 Transformer 需要 Attention？

在 RNN 里，信息是一步一步传的，序列很长时，远处信息不容易传过来。

Transformer 的思路是：

- 不再强制按时间步递推
- 直接让每个 token 和所有 token 计算相关性
- 用相关性分数决定应该吸收谁的信息

所以它更擅长建模长距离依赖，也更适合并行计算。

### 2. 什么是 Query / Key / Value？

这是初学者最容易卡住的地方。可以先不要把它想得太数学。

一个更容易记的说法是：

- `Query`：我现在想找什么信息
- `Key`：我这里有什么信息，可不可以被你匹配到
- `Value`：如果你决定关注我，你最终拿走什么内容

面试里可以这样举例：

一句话是“猫趴在垫子上，因为它累了”。

当模型处理“它”时：

- `Query` 表示“它”当前想找指代对象
- 其他 token 的 `Key` 用来和这个需求做匹配
- 匹配到“猫”之后，再把“猫”的 `Value` 信息聚合过来

### 3. Self-Attention 的计算流程

输入记作：

```text
X: [seq_len, d_model]
```

先线性映射得到：

```text
Q = X W_Q
K = X W_K
V = X W_V
```

然后做四步：

1. 计算相似度分数：`QK^T`
2. 缩放：除以 `sqrt(d_k)`
3. 归一化：做 `softmax`
4. 加权求和：用注意力权重乘 `V`

公式是：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

### 4. 为什么要除以 `sqrt(d_k)`？

这是经典高频题。

如果 `d_k` 很大，`QK^T` 的数值会变大，softmax 容易变得特别尖锐：

- 最大项接近 1
- 其他项接近 0

这会导致梯度不稳定，训练变差。

所以除以 `sqrt(d_k)` 的核心目的就是：

> 控制分数尺度，避免 softmax 过早饱和。

### 5. 什么是 Multi-Head Attention？

单头注意力只学一种相关性。

但真实语言里，相关性不止一种：

- 有的 head 关注语法依赖
- 有的 head 关注指代关系
- 有的 head 关注局部邻近词
- 有的 head 关注长距离信息

所以 Multi-Head Attention 的做法是：

- 先把 `Q / K / V` 映射到多个子空间
- 每个子空间单独做 attention
- 最后把多个 head 的结果拼接起来，再过一个线性层

公式是：

```text
head_i = Attention(Q_i, K_i, V_i)
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
```

### 6. 形状怎么记？

这是面试手写时最容易写乱的地方。

假设：

- `batch = B`
- `seq_len = T`
- `d_model = D`
- `num_heads = H`
- `head_dim = D / H`

那么常见形状是：

```text
X                : [B, T, D]
Q, K, V          : [B, T, D]
reshape 后       : [B, T, H, head_dim]
transpose 后     : [B, H, T, head_dim]
attention score  : [B, H, T, T]
attention output : [B, H, T, head_dim]
拼接回去         : [B, T, D]
```

你可以把它记成一句话：

> 先切 head，再在每个 head 里做一个 `T x T` 的相关性矩阵。

## 面试高频问题

### 1. Self-Attention 和 Self-Attention 的“self”是什么意思？

意思是 `Q / K / V` 都来自同一个输入序列。

如果 `Q` 来自解码器当前状态，而 `K / V` 来自编码器输出，那就是 Cross-Attention，不是 Self-Attention。

### 2. Self-Attention 为什么比 RNN 更容易并行？

因为它不需要按时间步递推。整个序列的相关性矩阵可以一次算出来。

### 3. Self-Attention 的时间复杂度是多少？

核心瓶颈是注意力分数矩阵：

```text
QK^T -> [T, T]
```

所以时间复杂度和显存复杂度通常都与 `T^2` 相关。

这是它长序列昂贵的根本原因。

### 4. Multi-Head 一定比 Single-Head 好吗？

通常更强，但不是“head 越多越好”。

head 太多时，每个 head 的维度太小，表达能力可能下降；同时实现和显存访问也更复杂。

### 5. 为什么说不同 head 会学到不同模式？

因为每个 head 都有自己独立的投影矩阵 `W_Q, W_K, W_V`，所以它们会在不同子空间中学习不同的匹配方式。

### 6. causal mask 是干什么的？

在自回归语言模型里，当前位置不能看未来 token。

所以要给未来位置加一个 mask，让它们在 softmax 前变成极小值，这样权重接近 0。

### 7. 注意力权重能不能直接解释成“模型理解了什么”？

不能过度解释。

attention weight 有一定可解释性，但它不是完整解释。真实模型行为还受到残差、MLP、LayerNorm、多层堆叠等影响。

## 最小实现

下面先看一个最小的单头注意力版本。

### 1. 单头 Self-Attention 最小版

```python
import torch
import torch.nn as nn


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        scores = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=-1)
        out = weights @ v
        return out, weights
```

你要记住的主线只有四步：

```text
X -> Q/K/V -> 分数 -> softmax -> 加权求和
```

### 2. 一个能手算理解的 toy 例子

```python
import torch

x = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
], dtype=torch.float32)

attn = SingleHeadSelfAttention(d_model=2)
with torch.no_grad():
    attn.w_q.weight.copy_(torch.eye(2))
    attn.w_k.weight.copy_(torch.eye(2))
    attn.w_v.weight.copy_(torch.eye(2))
```

这时：

- 第 1 个 token 比较像 `[1, 0]`
- 第 2 个 token 比较像 `[0, 1]`
- 第 3 个 token 同时包含两者信息

因为 `Q = K = V = X`，所以谁和谁更像，完全由输入向量本身决定。

这样你能直观看到：

- 第 1 个 token 会更关注自己和第 3 个 token
- 第 2 个 token 会更关注自己和第 3 个 token
- 第 3 个 token 会比较平均地看前两个 token

### 3. Multi-Head 最小版

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        seq_len, d_model = x.shape
        return x.reshape(seq_len, self.num_heads, self.head_dim).permute(1, 0, 2)

    def combine_heads(self, x):
        num_heads, seq_len, head_dim = x.shape
        return x.permute(1, 0, 2).reshape(seq_len, num_heads * head_dim)

    def forward(self, x, mask=None):
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))

        head_outputs = []
        for q_i, k_i, v_i in zip(q, k, v):
            scores = q_i @ k_i.transpose(-1, -2) / (q_i.shape[-1] ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(~mask, -1e9)

            weights = torch.softmax(scores, dim=-1)
            head_outputs.append(weights @ v_i)

        concat = self.combine_heads(torch.stack(head_outputs, dim=0))
        out = self.w_o(concat)
        return out
```

记忆重点不是代码细节，而是：

- 先线性映射
- 再拆成多个 head
- 每个 head 单独做 attention
- 拼接回来
- 再做一次输出投影

## 工程关注点

### 1. 为什么 Attention 在长序列下贵？

因为 `T x T` 的分数矩阵太大。

序列长度翻倍，相关性矩阵面积接近变成 4 倍。

所以长上下文场景里，attention 往往是显存和算力瓶颈。

### 2. 训练和推理的瓶颈一样吗？

不完全一样。

- 训练时更在意完整序列上的 `T x T` attention 开销
- 自回归推理时更在意 KV cache、memory bandwidth、batching 和 latency

### 3. 为什么后面会有 FlashAttention？

不是因为公式变了，而是因为原始 attention 的中间矩阵太占显存、访存开销太大。

FlashAttention 的核心是更高效地做分块计算和 IO 优化。

## 常见坑点

### 1. 把 Q / K / V 当成三份完全不同的输入

在 Self-Attention 里，它们通常来自同一个输入 `X`，只是乘了不同权重矩阵。

### 2. 忘了缩放

手写时最容易把 `/ sqrt(d_k)` 漏掉。

### 3. 把 softmax 维度写错

softmax 应该沿着“我要关注哪些 token”这个维度做，也就是最后那个 token 维度。

### 4. Multi-Head 只会背“多个头更强”

面试里要能接着说：

- 为什么更强
- 代价是什么
- head 太多会怎样

### 5. 只会写 attention，不会解释 mask

在语言模型面试里，causal mask 是高频追问，不能跳过。

## 面试时怎么讲

如果面试官让你介绍 Self-Attention，可以按这个顺序讲：

1. 它解决什么问题：让每个 token 动态聚合全局上下文
2. 它怎么算：`QK^T -> 缩放 -> softmax -> 加权 V`
3. 为什么有效：相关性是动态计算的，不是固定窗口
4. Multi-Head 为什么需要：不同 head 学不同关系
5. 它的代价：`T^2` 复杂度，长序列昂贵
6. 工程延伸：mask、KV cache、FlashAttention、GQA/MQA

一个简洁版本可以直接背：

> Self-Attention 的本质，是让每个 token 用自己的 query 去和所有 token 的 key 做匹配，得到注意力权重后，再对所有 value 做加权汇总。Multi-Head Attention 则是在多个子空间里并行做这件事，让模型能同时捕捉不同类型的依赖关系。它的优点是全局建模和并行性强，缺点是注意力矩阵带来 `O(T^2)` 的时间和显存开销。

## 延伸阅读

- 原始论文：Transformer 的 attention 模块
- 后续可以继续看：RoPE、KV Cache、FlashAttention、MQA / GQA
- 对照代码看：[minimal.py](minimal.py)
