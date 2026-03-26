# Positional Encoding / RoPE 面试攻略

## 这是什么？

这是 Transformer 里“位置信息怎么注入”的问题。

如果只用一句话来讲：

> Attention 本身只会看 token 之间像不像，不知道谁在前、谁在后，所以必须额外注入位置信息。

这就是 Positional Encoding 要解决的核心问题。

而 RoPE 则是现代 LLM 里最常见的位置编码方案之一。它不再像早期方法那样“把位置向量直接加到输入上”，而是把位置信息编码进 `Q / K` 的相对几何关系里。

## 核心机制

### 1. 为什么 Transformer 需要位置信息？

Self-Attention 会计算：

$$ \mathrm{Attention}(Q, K, V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V $$

这里本质上只关心向量之间的匹配程度。

如果你把一句话里的 token 顺序打乱，只保留 token embedding，不额外告诉模型位置，attention 本身并不能天然知道：

- 哪个词在前
- 哪个词在后
- 谁离谁更近
- 当前 token 是第几个位置

所以 Transformer 必须补位置信息。

### 2. 最早怎么做位置编码？

最经典的是原始 Transformer 里的 sinusoidal positional encoding，也就是正弦余弦位置编码。

它的思路是：

- 给每个位置 `pos` 构造一个固定向量
- 这个向量的不同维度对应不同频率的正弦/余弦
- 再把这个位置向量直接加到 token embedding 上

公式是：

$$ \mathrm{PE}(pos, 2i)=\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) $$

$$ \mathrm{PE}(pos, 2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) $$

这里几个符号的意思要分清：

- $pos$：序列里的位置编号，也就是当前 token 在第几个位置
- $i$：位置编码向量中的“频率编号”，不是 token 编号；它决定当前这一对维度使用哪一个频率
- $d_{\text{model}}$：模型隐藏维度，也就是 token embedding 的总维度

可以这样理解：

- 第 $0/1$ 维用一组频率
- 第 $2/3$ 维用另一组频率
- 第 $4/5$ 维再用另一组频率

其中 $i$ 越大，频率越低，变化越慢；$i$ 越小，频率越高，变化越快。

你可以把它理解成：

> 用一组不同频率的波形，把“当前位置是第几个”编码成一个连续向量。

### 3. 为什么正弦余弦位置编码有用？

它有两个直观优点：

- 不需要训练额外参数
- 不同位置之间能形成规律性的几何关系

早期面试里，一个常见回答是：

> 因为不同频率的 sin/cos 组合后，不同位置会对应不同向量，而且这种编码在更长序列上也有一定外推能力。

### 4. 它有什么局限？

在现代 LLM 里，大家越来越关注长上下文和相对位置关系。

而传统“直接把位置向量加到输入 embedding 上”的方法有几个问题：

- 更偏绝对位置，不够强调相对距离
- 在超长上下文外推时，效果未必稳定
- 对 attention 的相对位置结构利用得不够直接

所以后面出现了很多更适合 attention 的相对位置方案，比如 RoPE。

### 5. RoPE 到底在做什么？

RoPE 全称 Rotary Position Embedding。

它的关键点不是把位置向量加到 `X` 上，而是：

> 在 attention 前，对 `Q` 和 `K` 的每一对相邻维度做一个与位置相关的旋转。

如果看单个位置 $pos$、单个频率对 $j$，RoPE 可以写成：

$$ \theta_j = pos \cdot 10000^{-2j/d} $$

$$ \begin{bmatrix} x'_{2j} \\ x'_{2j+1} \end{bmatrix} = \begin{bmatrix} \cos \theta_j & -\sin \theta_j \\ \sin \theta_j & \cos \theta_j \end{bmatrix} \begin{bmatrix} x_{2j} \\ x_{2j+1} \end{bmatrix} $$

也就是把第 $2j$ 维和第 $2j+1$ 维看成一个二维向量，然后按位置相关的角度做旋转。

展开后就是：

$$ x'_{2j}=x_{2j}\cos\theta_j-x_{2j+1}\sin\theta_j $$

$$ x'_{2j+1}=x_{2j}\sin\theta_j+x_{2j+1}\cos\theta_j $$

这也是为什么代码实现里，RoPE 往往都是“偶数维和奇数维成对处理”。

这会带来一个很重要的性质：

> $Q$ 和 $K$ 做内积时，结果天然带上相对位置信息。

这也是为什么 RoPE 在 LLM 里很受欢迎，因为 attention 最终看的就是 $QK^\top$。

### 6. 为什么 RoPE 常常只作用在 Q 和 K 上？

这是高频题。

因为 attention score 来自 $QK^\top$，而位置信息最关键就是体现在“谁和谁该更相关”上。

所以 RoPE 一般旋转：

- `Q`
- `K`

而不直接旋转 `V`。

一个简洁回答是：

> 位置主要影响注意力权重的计算，所以把位置信息编码到 $Q$ 和 $K$ 里就够了；$V$ 主要承载被聚合的内容本身。

### 7. RoPE 的 shape 怎么理解？

先看单个 head。

如果：

- `seq_len = T`
- `head_dim = D`

那么：

```text
q, k: [T, D]
cos, sin: [T, D]
rope 后的 q, k: [T, D]
```

如果带 batch 和多头，常见是：

```text
q, k: [B, H, T, D]
cos, sin: [1, 1, T, D] 或 [T, D]
```

记忆重点是：

> RoPE 不改变 shape，它只是在每个位置上，对偶数维和奇数维做旋转变换。

## 面试高频问题

### 1. 为什么 Transformer 不能不用位置编码？

因为 attention 本身对顺序不敏感。没有位置信息，模型只知道 token 集合，不知道 token 顺序。

### 2. 为什么原始 Transformer 用 sin/cos？

因为它不需要学习参数，形式简单，而且不同位置之间有连续、可泛化的规律。

### 3. RoPE 相比绝对位置编码，优势是什么？

面试里常见说法：

- 更自然地把相对位置信息注入 attention
- 对长上下文通常更友好
- 在现代 LLM 里实践效果更好、应用更广

### 4. RoPE 为什么适合 LLM？

因为 LLM 的核心计算就是大规模自注意力，而 RoPE 直接作用在 `Q / K` 上，和 attention score 的计算方式非常贴合。

### 5. RoPE 为什么只旋转成对维度？

因为它本质上是在二维平面上做旋转。

所以常见写法都是把向量拆成：

- 第 0 维和第 1 维一对
- 第 2 维和第 3 维一对
- 第 4 维和第 5 维一对

### 6. RoPE 能解决无限长上下文吗？

不能。

RoPE 对长上下文更友好，但不是“天然无限长度”。上下文继续拉长时，仍然会遇到训练分布、频率缩放和外推能力的问题，这也是后面各种 RoPE scaling 方案出现的原因。

### 7. RoPE 和 KV Cache 有关系吗？

有。

自回归推理时，历史 token 的 `K / V` 会被缓存。由于 RoPE 是按位置作用在 `Q / K` 上的，所以生成新 token 时，位置索引必须和缓存里的历史位置严格对齐。

## 最小实现

### 1. Sinusoidal Positional Encoding 最小版

```python
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.shape[-2]
        return x + self.pe[:seq_len]
```

这个类的核心就一句话：

> 先预计算每个位置的 sin/cos 向量，前向时直接加到输入上。

### 2. RoPE 最小版

```python
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000):
        super().__init__()
        assert head_dim % 2 == 0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        position = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(position, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()
```

这个模块先产生每个位置对应的 `cos / sin`，然后配合下面的旋转函数使用：

```python
def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(start_dim=-2)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin
```

记忆方法：

- `rotate_half(x)`：把每对相邻维度做 $90^\circ$ 旋转
- `x * cos + rotate_half(x) * sin`：再按当前位置角度混合

### 3. 在 attention 里怎么接？

最小接法是：

```python
q = self.w_q(x)
k = self.w_k(x)

cos, sin = rope(seq_len=x.shape[-2], device=x.device)
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
```

然后再去算：

$$ \text{scores}=qk^\top $$

## 工程关注点

### 1. 位置编码加在哪里？

- 经典 sinusoidal PE：通常直接加到输入 embedding 上
- RoPE：通常加在 attention 内部的 $Q / K$ 上

### 2. 为什么 LLM 更常聊 RoPE，而不是早期绝对位置编码？

因为现代 LLM 更关心：

- 长上下文
- 相对位置关系
- 推理时和 attention / KV cache 的耦合方式

RoPE 在这些点上更贴合实践需求。

### 3. 为什么后面会有 RoPE scaling？

因为训练时的上下文长度有限，推理时想扩更长上下文，就会碰到 RoPE 原始频率设计的外推问题。

## 常见坑点

### 1. 以为 attention 自带顺序信息

不是。顺序信息必须显式注入。

### 2. 把 RoPE 理解成“再加一个位置向量”

不对。RoPE 的核心不是加法，而是旋转。

### 3. 忘了 RoPE 通常作用在 Q / K，而不是 V

这是非常常见的手写和口头表达错误。

### 4. 写 RoPE 时把偶数维和奇数维配对写乱

一旦偶数维/奇数维没有正确配对，旋转就错了。

### 5. 只会背结论，不会解释为什么 LLM 更偏爱 RoPE

面试里至少要能说出“因为它更直接作用在 attention 的相对位置关系上”。

## 面试时怎么讲

如果面试官让你介绍 Positional Encoding / RoPE，可以按这个顺序讲：

1. attention 本身不感知顺序，所以必须加位置信息
2. 早期 Transformer 用 sin/cos 绝对位置编码，直接加到输入 embedding 上
3. 现代 LLM 更常用 RoPE，因为它把位置信息编码进 $Q / K$ 的旋转关系里
4. 这样 attention score 会天然带上相对位置信息
5. 工程上它和长上下文、KV cache、RoPE scaling 强相关

一个简洁版本可以直接背：

> Positional Encoding 解决的是 Transformer 不感知顺序的问题。早期做法是用 sin/cos 构造绝对位置向量并加到输入上；现代 LLM 更常用 RoPE，它通过对 $Q$ 和 $K$ 做位置相关的旋转，把相对位置信息直接编码进 attention score 里，所以更适合大语言模型的注意力计算。

## 延伸阅读

- 下一步可以继续看：LayerNorm / RMSNorm、KV Cache、RoPE scaling
- 对照代码看：[minimal.py](minimal.py)
