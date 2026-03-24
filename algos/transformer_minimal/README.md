# 最小完整 Transformer 实现 面试攻略

## 这是什么？

这是一个“最简单但完整”的 Transformer 实现。

这里的“完整”指的是：它不是只写一个 attention 模块，而是把原始 Transformer 的关键部件都串起来，包括：

- token embedding
- positional encoding
- encoder self-attention
- decoder masked self-attention
- decoder cross-attention
- feed-forward network
- residual connection
- layer normalization
- output projection

如果只用一句话来讲：

> 这是一个最小版的 Encoder-Decoder Transformer，保留了原始论文的主干结构，但把实现压到适合面试讲解和手动阅读的程度。

## 结构图

先看整体，再看局部，会更容易理解。

### 1. 完整 Transformer 总览

![Transformer encoder-decoder architecture](https://jalammar.github.io/images/xlnet/transformer-encoder-decoder.png)

这张图最重要的是让你先建立一个总印象：

- 左边是 encoder stack
- 右边是 decoder stack
- encoder 输出会传给 decoder 做 cross-attention

### 2. Encoder Block 结构图

![Transformer encoder block](https://jalammar.github.io/images/xlnet/transformer-encoder-block-2.png)

encoder block 比较干净，只有两层主干：

- self-attention
- feed forward network

### 3. Decoder Block 结构图

![Transformer decoder block](https://jalammar.github.io/images/t/Transformer_decoder.png)

decoder 比 encoder 多出来的关键，就是中间那层 encoder-decoder attention，也就是 cross-attention。

### 4. Decoder-Only 结构图

![Transformer decoder-only block](https://jalammar.github.io/images/xlnet/transformer-decoder-intro.png)

这张图适合拿来和 GPT 一类模型对应起来看：

- 只有 decoder stack
- 没有 encoder
- 只有 masked self-attention 和 FFN

图片来源：

- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Jay Alammar, [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

这些图在理解结构上很好用，但面试时你不能只会认图，还得能讲清楚 encoder 和 decoder 各自负责什么。

## 核心机制

### 1. 完整 Transformer 的数据流是什么？

原始 Transformer 不是 decoder-only，而是 encoder-decoder 结构。

最简主线可以记成：

```text
src tokens -> embedding -> positional encoding -> encoder
tgt tokens -> embedding -> positional encoding -> decoder
decoder output -> linear projection -> logits
```

其中 decoder 里还会额外看到 encoder 的输出，也就是 cross-attention。

### 2. Encoder 层里有什么？

一个标准 encoder layer 通常有两大块：

1. multi-head self-attention
2. feed-forward network

中间和外面配套的是：

- residual connection
- layer normalization

可以写成：

$$
H' = \mathrm{LN}(H + \mathrm{MHA}(H, H, H))
$$

$$
H'' = \mathrm{LN}(H' + \mathrm{FFN}(H'))
$$

这里 encoder 的 attention 是标准 self-attention，因为 $Q, K, V$ 都来自 encoder 当前输入。

### 3. Decoder 层里有什么？

decoder 比 encoder 多一层 cross-attention，所以通常有三块：

1. masked self-attention
2. cross-attention
3. feed-forward network

可以写成：

$$
S' = \mathrm{LN}(S + \mathrm{MaskedMHA}(S, S, S))
$$

$$
S'' = \mathrm{LN}(S' + \mathrm{MHA}(S', E, E))
$$

$$
S''' = \mathrm{LN}(S'' + \mathrm{FFN}(S''))
$$

其中：

- $S$ 是 decoder 隐状态
- $E$ 是 encoder 输出

### 4. 为什么 decoder 要做 mask？

因为训练和推理语言生成时，当前位置不能偷看未来 token。

所以 decoder 的 self-attention 必须带 causal mask。

这也是为什么 decoder 的第一层 attention 叫 masked self-attention。

### 5. 为什么 decoder 还要有 cross-attention？

因为在 seq2seq 场景里，decoder 生成目标序列时，需要参考源序列的编码结果。

所以：

- decoder 自己内部先看自己已经生成的前文
- 然后再通过 cross-attention 去看 encoder 输出

面试里可以直接说：

> decoder 的 self-attention 负责建模目标端上下文，cross-attention 负责对齐源端信息。

### 6. 最小完整实现里“该有的要素”具体指什么？

这次实现我保留了下面这些关键元素：

- source / target embedding
- sinusoidal positional encoding
- encoder stack
- decoder stack
- masked self-attention
- cross-attention
- feed-forward
- residual + layernorm
- output head
- source padding mask
- target causal mask

但我刻意没有加太多工程细节，比如：

- dropout
- label smoothing
- beam search
- KV cache
- mixed precision

因为这个专题的目标是“讲清结构”，不是“模拟生产训练代码”。

### 7. Encoder 和 Decoder 分别起什么作用？

可以先用一句话记：

- encoder：把输入序列编码成上下文化表示
- decoder：在生成输出时，一边看已经生成的前缀，一边读取 encoder 提供的输入信息

更具体一点：

#### encoder 在做什么？

encoder 的任务是“理解输入”。

它读完整个 source sequence，然后把每个位置编码成带上下文的信息表示。输出通常叫：

$$
E \in \mathbb{R}^{B \times T_{\text{src}} \times D}
$$

这组表示的特点是：

- 每个 token 都能双向看完整输入
- 更适合理解、匹配、分类、检索

所以 encoder 更像一个“读懂输入的表征器”。

#### decoder 在做什么？

decoder 的任务是“按顺序生成输出”。

它在第 $t$ 步只能看：

- 已经生成的前 $t-1$ 个 token
- encoder 输出的 source 表示

所以 decoder 更像一个“条件生成器”。

如果是翻译场景，可以把它理解成：

- encoder 先把法语句子读懂
- decoder 再一个词一个词生成英语句子

### 8. 为什么有时候 encoder 和 decoder 两个都要？

因为有些任务天然就是：

> 先读一个输入，再生成另一个输出

典型就是 seq2seq：

- 机器翻译
- 文本摘要
- 语音识别转文本
- OCR / 图像描述这类“输入一种序列，输出另一种序列”

这种任务里，source 和 target 的角色不同：

- source 需要被完整理解
- target 需要被自回归生成

所以 encoder-decoder 很自然。

### 9. 为什么有时候只需要 encoder？

因为有些任务不需要“生成”，只需要“理解”。

比如：

- 文本分类
- 句子匹配
- 命名实体识别
- 检索表征
- 许多判别式任务

这类任务的核心是把输入编码成一个好的表示，然后接分类头或任务头即可。

所以只保留 encoder 就够了。

一个典型例子是：

- BERT：encoder-only

它更擅长做理解型任务，而不是直接做开放式逐 token 生成。

### 10. 为什么有时候只需要 decoder-only？

因为有些任务的核心目标就是：

> 给定前文，继续往后生成

比如：

- 语言建模
- 对话生成
- 代码补全
- 通用大语言模型

这类任务本质上都可以写成 next-token prediction。

所以只要保留 decoder，并给 self-attention 加 causal mask，就能做自回归生成。

典型例子是：

- GPT 系列
- LLaMA
- Qwen
- DeepSeek-V2/V3 这类主流 LLM

### 11. 三种结构该怎么一眼区分？

最简单的判断方式是看任务目标。

- encoder-only：理解输入，不负责开放式生成
- encoder-decoder：读一个序列，再生成另一个序列
- decoder-only：根据前缀持续生成后续 token

如果面试官追问“为什么现代 LLM 大多是 decoder-only”，一个常见回答是：

> 因为通用语言模型训练目标通常就是 next-token prediction，decoder-only 结构最直接，扩展到大规模预训练也最自然。

## 面试高频问题

### 1. 原始 Transformer 和现代 LLM 的结构一样吗？

不一样。

原始 Transformer 通常指 encoder-decoder 架构；很多现代 LLM 是 decoder-only。

### 2. 为什么这里还要实现 encoder-decoder，而不是直接写 GPT 风格？

因为从“Transformer 基础”角度看，encoder-decoder 是更完整的母体结构。先把这个结构讲清楚，再看 decoder-only LLM 会更顺。

### 3. encoder 的 self-attention 和 decoder 的 cross-attention 有什么区别？

- encoder self-attention：$Q, K, V$ 都来自 encoder 当前输入
- decoder cross-attention：$Q$ 来自 decoder，$K, V$ 来自 encoder 输出

### 4. 为什么 decoder self-attention 要 mask，encoder self-attention 不需要？

因为 encoder 一般处理完整输入序列，不需要阻止看未来；decoder 生成目标序列时必须保持因果性。

### 5. 一个完整 Transformer 里最核心的几个模块是什么？

如果只能列最关键的几个：

- embedding
- positional encoding
- multi-head attention
- feed-forward
- residual
- layernorm

### 6. 为什么完整实现里一定要有 residual 和 norm？

因为没有这两样，深层训练的稳定性会明显变差。真正能工作的 Transformer，不只是“把 attention 堆起来”。

### 7. 这个实现的复杂度瓶颈在哪？

attention 仍然是主要瓶颈，尤其是序列长度上的 $O(T^2)$ 开销。

### 8. encoder-only、encoder-decoder、decoder-only 该怎么选？

按任务目标选：

- 理解任务优先：encoder-only
- 条件生成任务优先：encoder-decoder
- 开放式自回归生成优先：decoder-only

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

这样既保留了完整结构，又不会像生产代码那样分太细。

### 2. EncoderLayer 最小骨架

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        attn_out, _ = self.self_attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
```

你可以把它记成：

```text
self-attn -> residual+norm -> ffn -> residual+norm
```

### 3. DecoderLayer 最小骨架

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
```

比 encoder 多的关键就是：

- `cross_attn`
- 一个额外的 `norm`

### 4. 整体前向传播主线

```python
src_x = self.src_embedding(src_tokens)
src_x = self.pos_encoding(src_x)
memory = self.encoder(src_x, src_key_padding_mask)

tgt_x = self.tgt_embedding(tgt_tokens)
tgt_x = self.pos_encoding(tgt_x)
hidden = self.decoder(
    tgt_x,
    memory,
    tgt_mask=tgt_mask,
    src_key_padding_mask=src_key_padding_mask,
)

logits = self.output_proj(hidden)
```

一眼看过去就是：

> source 进 encoder，target 进 decoder，decoder 再结合 encoder 输出，最后投影成词表 logits。

## 工程关注点

### 1. 为什么这个实现选 `nn.MultiheadAttention`？

因为这次目标是“最小完整结构”，不是手写 attention kernel。前面专题已经把 attention 本身讲开了，这里更关注模块如何拼起来。

### 2. 为什么还保留 sinusoidal positional encoding？

因为这是“Transformer 基础”专题，不是现代 LLM 工程专题。这里优先展示原始结构上的位置编码注入方式。

### 3. 为什么不直接实现训练循环？

因为这个专题的重点是模型结构，而不是训练技巧。训练循环、loss、teacher forcing、label smoothing 可以单独再开专题。

## 常见坑点

### 1. 把完整 Transformer 和 decoder-only LLM 混为一谈

面试里一定要分清。

### 2. 忘了 decoder 有两层 attention

这是最常见的结构性错误。

### 3. 忘了 causal mask

没有这个，decoder self-attention 就不再是因果的。

### 4. 只写 attention，不写 residual 和 layernorm

那就不是一个真正能工作的 Transformer block。

### 5. 只会背结构图，不会说数据流

面试里更重要的是你能按顺序讲清楚张量怎么流动。

## 面试时怎么讲

如果面试官让你介绍一个完整 Transformer，可以按这个顺序讲：

1. 输入先做 embedding 和位置编码
2. source 进 encoder，encoder 层由 self-attention 和 FFN 构成
3. target 进 decoder，decoder 先做 masked self-attention，再做 cross-attention，再做 FFN
4. 每个子层周围都有 residual 和 layernorm
5. 最后接输出投影得到 logits

一个简洁版本可以直接背：

> 一个完整 Transformer 是 encoder-decoder 结构。encoder 由 self-attention 和 FFN 堆叠而成，decoder 则由 masked self-attention、cross-attention 和 FFN 堆叠而成，每个子层都带 residual 和 layernorm。输入先做 embedding 和位置编码，encoder 处理 source，decoder 在看 target 前缀的同时通过 cross-attention 读取 encoder 输出，最后投影到词表得到 logits。

## 延伸阅读

- 和基础模块对照看：[Self-Attention / Multi-Head Attention](../self_attention/README.md)
- 和位置编码对照看：[Positional Encoding / RoPE](../positional_encoding/README.md)
- 和归一化对照看：[LayerNorm / RMSNorm](../normalization/README.md)
- 对照代码看：[minimal.py](minimal.py)
