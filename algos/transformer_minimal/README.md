# 最小完整 Transformer 实现 面试攻略

## 这是什么？

这是一个“最简单但完整”的原始 Transformer 实现。

这里的“完整”指的是：它不是只写一个 attention 模块，而是把 encoder-decoder Transformer 的关键部件真正串起来，包括：

- source / target embedding
- positional encoding
- encoder stack
- decoder stack
- decoder masked self-attention
- decoder cross-attention
- feed-forward network
- residual connection
- layer normalization
- output projection
- source padding mask
- target causal mask

如果只用一句话来讲：

> 这是一个最小版的 encoder-decoder Transformer，保留了原始论文的主干数据流，但把实现压到适合面试讲解和手动阅读的程度。

如果你还没完全分清 encoder 和 decoder 各自做什么，建议先看上一篇：[Encoder / Decoder 结构与区别](../encoder_decoder/README.md)。

## 核心机制

### 1. 完整 Transformer 的主数据流

原始 Transformer 不是 decoder-only，而是标准的 encoder-decoder。

最简主线可以记成：

```text
src tokens -> embedding -> positional encoding -> encoder -> memory
tgt tokens -> embedding -> positional encoding -> decoder(memory) -> hidden
hidden -> linear projection -> logits
```

这条数据流里最重要的点只有两个：

- source 先进入 encoder，被编码成 memory
- target 再进入 decoder，在因果约束下结合这份 memory 产生输出

### 2. 为什么它叫“完整 Transformer”？

因为这里保留的是原始母体结构，而不是只实现其中一小块。

![Transformer encoder-decoder architecture](https://jalammar.github.io/images/xlnet/transformer-encoder-decoder.png)

这张图里真正要记住的是：

- 左边是 encoder stack
- 右边是 decoder stack
- decoder 中间会通过 cross-attention 读取 encoder 输出

图片来源：

- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

### 3. 这个最小实现里保留了哪些关键模块？

这次实现我刻意保留了下面这些“少了就不完整”的要素：

- source / target 双 embedding
- sinusoidal positional encoding
- encoder layer 堆叠
- decoder layer 堆叠
- masked self-attention
- cross-attention
- feed-forward
- residual + layernorm
- output head
- source padding mask
- target causal mask

如果缺少其中几项，通常就会退化成：

- 只是在写 attention
- 只是在写 encoder block
- 只是在写 decoder-only block

而不是一个完整的 encoder-decoder Transformer。

### 4. 这里刻意没放什么？

为了保证“可手写、可讲清”，这篇没有追求训练脚手架和生产工程细节，暂时不展开：

- dropout
- label smoothing
- teacher forcing 细节
- beam search
- KV cache
- mixed precision

这不是因为它们不重要，而是因为这个专题的目标是：

> 先把完整结构和前向主线讲清楚。

## 面试高频问题

### 1. 原始 Transformer 和现代 LLM 的结构一样吗？

不一样。

原始 Transformer 通常指 encoder-decoder；很多现代 LLM 是 decoder-only。

### 2. 为什么这里还要实现 encoder-decoder，而不是直接写 GPT 风格？

因为从“Transformer 基础”角度看，encoder-decoder 是更完整的母体结构。先把这个结构讲清楚，再去看 decoder-only 会更顺。

### 3. 一个完整 Transformer 最核心的几个模块是什么？

如果只能列最关键的几个：

- embedding
- positional encoding
- multi-head attention
- feed-forward
- residual
- layernorm
- output projection

### 4. 为什么完整实现里一定要有 residual 和 norm？

因为没有这两样，深层训练的稳定性会明显变差。真正能工作的 Transformer，不只是“把 attention 堆起来”。

### 5. 这个实现的复杂度瓶颈在哪？

attention 仍然是主要瓶颈，尤其是序列长度相关的二次复杂度。

### 6. 这一篇和上一专题怎么分工？

- 上一篇重点讲 encoder / decoder 的角色差异和结构选型
- 这一篇重点讲完整 Transformer 如何把各个模块串成一条前向数据流

## 最小实现

### 1. 最小模块划分

这个实现拆成以下类：

- `PositionalEncoding`
- `FeedForward`
- `EncoderLayer`
- `DecoderLayer`
- `Encoder`
- `Decoder`
- `Transformer`

这样既保留了完整结构，又不会像生产代码那样分得很碎。

### 2. `Transformer` 初始化骨架

```python
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        max_len=128,
    ):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
```

一眼看过去就该知道：

- source 和 target 是两路输入
- 中间有 encoder 和 decoder 两个栈
- 最后还要接一个词表投影头

### 3. 整体前向传播主线

```python
tgt_mask = build_causal_mask(tgt_tokens.shape[1], device=tgt_tokens.device)

src_x = self.src_embedding(src_tokens)
src_x = self.pos_encoding(src_x)
memory = self.encoder(src_x, src_key_padding_mask=src_key_padding_mask)

tgt_x = self.tgt_embedding(tgt_tokens)
tgt_x = self.pos_encoding(tgt_x)
hidden = self.decoder(
    tgt_x,
    memory,
    tgt_mask=tgt_mask,
    tgt_key_padding_mask=tgt_key_padding_mask,
    memory_key_padding_mask=src_key_padding_mask,
)

logits = self.output_proj(hidden)
```

可以直接概括成：

> source 先编码成 memory，target 再在 causal mask 约束下结合这份 memory 产生 hidden，最后映射到词表得到 logits。

### 4. 这篇最值得你手写的部分是什么？

如果面试现场不能把整份代码都写完，至少要能手写并讲清：

- `EncoderLayer`
- `DecoderLayer`
- `Transformer.forward`

因为这三块能把“模块结构”和“数据流”同时交代出来。

完整代码见：[minimal.py](minimal.py)

## 工程关注点

### 1. 为什么这个实现选 `nn.MultiheadAttention`？

因为这次目标是“最小完整结构”，不是手写 attention kernel。attention 本身已经在前面专题展开了，这里更关注模块如何拼起来。

### 2. 为什么还保留 sinusoidal positional encoding？

因为这是“Transformer 基础”专题，不是现代 LLM 工程专题。这里优先展示原始结构里的位置编码注入方式。

### 3. 这里为什么没有训练循环？

因为这篇的重点是模型结构，而不是训练技巧。训练循环、loss、teacher forcing、label smoothing 可以单独开专题。

## 常见坑点

### 1. 把完整 Transformer 和 decoder-only LLM 混为一谈

面试里一定要分清。

### 2. 忘了 decoder 有两层 attention

这是最常见的结构性错误。

### 3. 忘了 target causal mask

没有这个，decoder self-attention 就不再满足因果性。

### 4. source mask 和 target mask 用混

一个是 padding mask，一个是 causal mask，职责完全不同。

### 5. 只会背结构图，不会说数据流

面试里更重要的是你能按顺序讲清楚张量怎么流动。

## 面试时怎么讲

如果面试官让你介绍一个完整 Transformer，可以按这个顺序讲：

1. 输入分成 source 和 target，两边先做 embedding 和位置编码
2. source 进 encoder，得到上下文化的 memory
3. target 进 decoder，先做 masked self-attention，再通过 cross-attention 读取 memory
4. 每个子层周围都有 residual 和 layernorm，之后接 FFN
5. 最后把 decoder hidden 投影到词表，得到每个位置的 logits

一个简洁版本可以直接背：

> 一个完整 Transformer 是 encoder-decoder 结构。source 先经过 embedding、位置编码和 encoder stack，被编码成 memory；target 再经过 embedding、位置编码和 decoder stack，先做 masked self-attention，再通过 cross-attention 读取 encoder memory，最后把 hidden 投影到词表得到 logits。每个子层周围都带 residual 和 layernorm。

## 延伸阅读

- 先看结构角色：[Encoder / Decoder 结构与区别](../encoder_decoder/README.md)
- Attention 基础：[Self-Attention / Multi-Head Attention](../self_attention/README.md)
- 位置编码：[Positional Encoding / RoPE](../positional_encoding/README.md)
- 归一化：[LayerNorm / RMSNorm](../normalization/README.md)
- 对照代码看：[minimal.py](minimal.py)
