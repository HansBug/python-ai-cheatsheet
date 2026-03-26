# Vision Transformer / ViT 面试攻略

## 这是什么？

这是 Transformer 被系统性用到 CV 任务里的代表性模型。

如果只用一句话来讲：

> ViT 的核心思路，是把图像切成一串 patch token，再把图像分类问题改写成一个标准的序列建模问题，交给 Transformer encoder 去做。

它的重要性不只是“图像分类里也能用 attention”，而是：

- Transformer 不再只属于 NLP
- 图像也可以被改写成 token 序列
- 后面的 DETR、SegFormer、Swin 这类 CV Transformer，基本都沿着这个方向继续展开

所以如果面试官问“Transformer 是怎么被用到 CV 上的”，ViT 往往就是最自然的切入口。

## 核心机制

### 1. 为什么图像也能喂给 Transformer？

Transformer 最早处理的是 token 序列。

图像看起来不是序列，而是：

$$
X \in \mathbb{R}^{B \times C \times H \times W}
$$

ViT 的关键改写是：

> 先把一张图切成很多个固定大小的 patch，再把每个 patch 当成一个 token。

如果 patch size 是 $P \times P$，那么 patch 数量就是：

$$
N = \frac{H}{P} \times \frac{W}{P}
$$

于是图像就被改写成长度为 $N$ 的 patch 序列。

这一步是 ViT 成立的前提。没有 patchify，Transformer 就没法直接把图像当成序列来处理。

### 2. ViT 的主数据流是什么？

最简主线可以记成：

```text
image -> patch embedding -> patch tokens
patch tokens + cls token + position embedding
-> Transformer encoder stack
-> cls token
-> linear classifier
```

也就是：

1. 把图像切成 patch
2. 每个 patch 映射到 `d_model` 维
3. 加上位置编码
4. 送进多层 Transformer encoder
5. 取 `cls token` 做分类

这也是为什么 ViT 通常被视为一个 **encoder-only** 结构。

### 3. Patch Embedding 到底在干什么？

这是 ViT 最关键的输入层。

它做了两件事：

- 空间切块：把图像划成一组不重叠 patch
- 通道映射：把每个 patch 投影成 Transformer 能处理的向量

在最小实现里，我们可以直接用一个卷积完成这件事：

```python
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, d_model):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
```

这里每一步的含义是：

- `kernel_size=patch_size, stride=patch_size`：每次正好取一个 patch，而且 patch 之间不重叠
- `self.proj(x)`：把每个 patch 从像素空间映射到 `d_model` 维特征空间
- `flatten(2).transpose(1, 2)`：把二维 patch 网格改写成一维 token 序列

所以 patch embedding 本质上就是：

> 把图像从 `[B, C, H, W]` 变成 `[B, N, D]`。

### 4. 为什么图像里也要位置编码？

因为 patch 序列一旦被摊平，Transformer 本身并不知道：

- 左上角 patch 和右下角 patch 的空间位置
- 哪两个 patch 原来是相邻的
- 哪些 patch 在同一行或同一列

所以 ViT 一样需要位置编码。

最常见做法就是给 patch tokens 加一个可学习的位置向量：

```python
x = x + self.pos_embed[:, : x.shape[1]]
```

图像虽然原本是二维结构，但进入 Transformer 后已经被改写成一维 token 序列，所以仍然需要显式补位置信息。

### 5. `cls token` 在 ViT 里起什么作用？

ViT 的分类版本通常会在 patch tokens 前面额外拼一个可学习的 `cls token`：

```python
cls_tokens = self.cls_token.expand(batch_size, -1, -1)
x = torch.cat([cls_tokens, x], dim=1)
```

它的直觉是：

> 让模型专门保留一个“全局汇总位”，最后用它来代表整张图做分类。

经过多层 self-attention 后，`cls token` 会和所有 patch 交互，逐渐聚合整张图的信息。

最后分类时通常直接取：

```python
cls_state = x[:, 0]
logits = self.head(cls_state)
```

### 6. Transformer 在 CV 里到底是怎么被用起来的？

如果面试官问的是更泛化的问题，不要只答 ViT 分类这一种。

更完整的回答应该是：

#### 1. 图像分类：把图像当 patch 序列，用 encoder-only 做全局建模

典型代表：

- ViT
- DeiT

这类模型的核心是：

- patchify
- encoder stack
- `cls token` 或 pooling 分类头

#### 2. 目标检测：让 Transformer 负责全局建模和目标查询

典型代表：

- DETR

这类模型的关键不是简单分类，而是：

- backbone 先提视觉特征
- Transformer encoder 建模全局关系
- decoder 用 object queries 去预测目标框和类别

也就是说，Transformer 在检测里不仅是 backbone，还可以直接参与集合预测。

#### 3. 语义分割：让 Transformer 产生更强的全局特征，再接分割头

典型代表：

- SETR
- SegFormer

分割任务需要 dense prediction，所以更关注：

- patch / token 级别特征
- 多尺度表示
- 上采样或轻量 decoder

#### 4. 更进一步：做分层视觉 backbone

典型代表：

- Swin Transformer

它不是简单把所有 patch 全局全连，而是通过窗口注意力和层级结构，把 Transformer 做得更像一个适合视觉任务的 backbone。

所以一句话总结：

> Transformer 在 CV 里，既可以拿来做图像分类的 encoder-only backbone，也可以进入检测、分割等更复杂任务，承担全局建模、目标查询和视觉表征学习的角色。

### 7. ViT 相比 CNN 到底新在哪？

最核心的变化有三个：

- 感受野来源不同：CNN 靠卷积层逐步扩大，ViT 直接靠 self-attention 做全局交互
- 归纳偏置不同：CNN 天然更有局部性和平移等结构偏置，ViT 更弱，需要更多数据或更强训练策略
- 架构统一性更强：ViT 更容易和 NLP / 多模态 Transformer 统一到一套 token + attention 范式里

所以 ViT 的价值不只是“分类精度能做上去”，而是：

> 它把视觉任务也拉进了统一的 Transformer 建模框架。

### 8. 为什么早期 ViT 往往更吃数据？

因为它相比 CNN，少了很多视觉任务里很有帮助的先验：

- 局部连接
- 平移等价性
- 多尺度层级结构

这些先验在 CNN 里几乎是“白送”的，但在纯 ViT 里需要靠数据和训练去学出来。

所以早期 ViT 往往更依赖：

- 大规模预训练
- 更强的数据增强
- 蒸馏或更仔细的训练 recipe

这也是为什么后来会有 DeiT、Swin 这类更贴近视觉特点的改进路线。

## 面试高频问题

### 1. 为什么 Transformer 能做图像任务？

因为图像可以先被切成 patch 序列，再把每个 patch 映射成 token embedding，之后就能交给 Transformer encoder 做序列建模。

### 2. ViT 里的 patch embedding 和卷积是什么关系？

patch embedding 可以直接用一个 `kernel_size = stride = patch_size` 的卷积实现。它本质上是在做分块采样加线性投影。

### 3. ViT 为什么需要位置编码？

因为 patch 被摊平成序列后，Transformer 本身不知道二维空间位置关系，必须显式补位置信息。

### 4. ViT 为什么常常是 encoder-only？

因为最基础的 ViT 主要做图像分类，任务目标是“理解整张图”，不是自回归生成，所以用 encoder-only 很自然。

### 5. `cls token` 的作用是什么？

它是一个专门用于聚合全局信息的可学习 token，最后通常拿它做分类。

### 6. ViT 和 CNN 的核心差异是什么？

CNN 更依赖局部卷积和层级结构，ViT 更依赖全局 self-attention 和大规模数据驱动学习。

### 7. Transformer 在 CV 里除了分类还能怎么用？

可以用于目标检测、语义分割、视觉 backbone、多模态视觉语言模型等场景。

### 8. 为什么 Swin 这类模型会出现？

因为纯全局 ViT 的计算量和视觉归纳偏置都存在问题，所以需要更适合视觉任务的窗口化、层级化改造。

## 最小实现

这个专题的最小实现聚焦在最基础的 ViT 图像分类主线，只保留下面几块：

- `PatchEmbedding`
- `MLP`
- `TransformerEncoderBlock`
- `VisionTransformer`

### 1. `PatchEmbedding`：把图像改写成 token 序列

```python
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, d_model):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
```

这一步输出的是：

$$
[B, C, H, W] \rightarrow [B, N, D]
$$

也就是从图像张量，变成 Transformer 能吃的 token 序列。

### 2. `TransformerEncoderBlock`：ViT 的核心编码单元

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, hidden_dim=d_model * mlp_ratio)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
```

它的作用和 NLP 里的 encoder block 本质一致：

- `LayerNorm`
- self-attention
- residual
- MLP
- residual

区别只在于现在输入的不是词 token，而是图像 patch token。

### 3. `VisionTransformer.forward`：完整分类主线

```python
x = self.patch_embed(x)

cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
x = torch.cat([cls_tokens, x], dim=1)
x = x + self.pos_embed[:, : x.shape[1]]

for block in self.blocks:
    x = block(x)

x = self.norm(x)
cls_state = x[:, 0]
logits = self.head(cls_state)
```

你可以直接把这段背成：

> 先 patchify，再加 `cls token` 和位置编码，再过多层 encoder，最后取 `cls token` 做分类。

### 4. 这份最小实现最值得手写哪几段？

如果面试现场不能全写完，至少要能写并讲清：

- `PatchEmbedding.forward`
- `TransformerEncoderBlock.forward`
- `VisionTransformer.forward`

因为这三段正好对应：

- 图像怎么变 token
- Transformer 怎么处理 patch 序列
- 最后怎么做分类

完整代码见：[minimal.py](minimal.py)

## 工程关注点

### 1. patch size 会直接影响 token 数和计算量

如果 patch 更小，token 数就更多，attention 复杂度会明显上涨。

如果输入大小固定为 $H \times W$，patch size 为 $P$，那么 token 数是：

$$
N = \frac{H}{P} \times \frac{W}{P}
$$

self-attention 的主要复杂度会随 $N^2$ 增长。

### 2. 纯 ViT 对数据和训练 recipe 更敏感

相比 CNN，ViT 更少视觉归纳偏置，所以通常更吃：

- 预训练数据规模
- 数据增强
- 正则化和蒸馏策略

### 3. 高分辨率场景里全局 attention 成本会很高

这也是为什么很多视觉 Transformer 会做：

- 分层结构
- 窗口注意力
- token 下采样

### 4. 检测和分割通常不会只靠一个分类版 ViT 头

分类版 ViT 主要输出全局类别；到了检测和分割，还要搭配：

- query 机制
- 多尺度特征
- 上采样或任务 decoder

## 常见坑点

### 1. 以为 ViT 就是“把 CNN 最后一层换成 attention”

不对。ViT 是从输入表示开始就把图像改写成 patch token 序列。

### 2. 把 patch embedding 理解成普通 flatten

不够准确。它不是简单拉平整张图，而是按 patch 切块后再做投影。

### 3. 忘了位置编码

没有位置编码，Transformer 很难保留图像的空间布局信息。

### 4. 把 ViT 和 DETR / Swin 混成一个东西

它们都属于 CV Transformer，但任务目标和结构重点并不一样。

### 5. 只会说“attention 能看到全局”，不会说具体怎么进入 CV

更好的回答应该明确讲出：

- patchify
- tokenization
- encoder-only 分类
- 检测 / 分割里的进一步扩展

## 面试时怎么讲

如果面试官让你介绍 ViT，可以按这个顺序讲：

1. Transformer 原本处理序列，ViT 先把图像切成 patch 序列
2. 每个 patch 通过 patch embedding 映射成 token，再加 `cls token` 和位置编码
3. 然后把整串 token 送进 Transformer encoder 做全局建模
4. 分类时通常取最后的 `cls token` 过线性头得到 logits
5. 这说明 Transformer 不只可以做 NLP，也可以通过 patch tokenization 用在 CV 上
6. 更进一步，Transformer 在 CV 里还能扩展到检测、分割和视觉 backbone

一个简洁版本可以直接讲：

> Vision Transformer 的核心思想，是把图像切成很多 patch，把每个 patch 当成 token，然后像处理文本序列一样交给 Transformer encoder。输入侧先做 patch embedding，再拼上 `cls token` 和位置编码，多层 encoder 之后取 `cls token` 做分类。它说明了 Transformer 可以通过 patch tokenization 进入 CV 任务，后面也进一步扩展到了检测、分割和更通用的视觉 backbone。

## 延伸阅读

- Attention 基础：[Self-Attention / Multi-Head Attention](../self_attention/README.md)
- 位置编码：[Positional Encoding / RoPE](../positional_encoding/README.md)
- 归一化：[LayerNorm / RMSNorm](../normalization/README.md)
- 编码器结构：[Encoder / Decoder 结构与区别](../encoder_decoder/README.md)
- 配套代码：[minimal.py](minimal.py)
