# Encoder / Decoder 结构与区别 面试攻略

## 这是什么？

这是专门讲 Transformer 里 `encoder` 和 `decoder` 两部分的专题。

很多人会背结构图，但一到面试里就容易说混：

- encoder 到底在“理解”什么
- decoder 到底在“生成”什么
- 为什么 decoder 比 encoder 多一层 attention
- 为什么有时只要 encoder，有时只要 decoder-only

如果只用一句话来讲：

> encoder 负责把输入序列编码成上下文化表示，decoder 负责在因果约束下按顺序生成输出，并在需要时通过 cross-attention 读取 encoder 提供的条件信息。

这个专题放在“最小完整 Transformer”之前看，会更顺。因为先把两边各自做什么搞清楚，再看完整数据流，就不容易把结构记成死图。

## 结构图

### 1. Encoder-Decoder 总览

![Transformer encoder-decoder architecture](https://jalammar.github.io/images/xlnet/transformer-encoder-decoder.png)

这张图最值得先建立的印象是：

- 左边是 encoder stack
- 右边是 decoder stack
- encoder 输出会送到 decoder 做 cross-attention

### 2. Encoder Block 结构

![Transformer encoder block](https://jalammar.github.io/images/xlnet/transformer-encoder-block-2.png)

只看 block 级别，encoder 很干净，主干只有：

- self-attention
- feed-forward network

### 3. Decoder Block 结构

![Transformer decoder block](https://jalammar.github.io/images/t/Transformer_decoder.png)

decoder 相比 encoder 多出来的关键，就是中间那层 encoder-decoder attention，也就是 cross-attention。

图片来源：

- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

## 核心机制

### 1. 先从任务视角理解两者分工

原始 Transformer 不是单塔结构，而是典型的 encoder-decoder。

最简主线是：

```text
source tokens -> encoder -> memory
target tokens -> decoder(memory) -> logits
```

这里最关键的是：

- encoder 读的是 source
- decoder 生成的是 target
- decoder 生成时还会条件化地读取 encoder 输出

所以它天然适合这类任务：

> 先读一个序列，再生成另一个序列。

典型就是：

- 机器翻译
- 文本摘要
- 语音识别转文本
- OCR / 图像描述这类条件生成任务

### 2. Encoder 在做什么？

encoder 的核心任务是“理解输入”。

它会把整段 source sequence 编码成带上下文的信息表示，输出通常记作：

$$
E \in \mathbb{R}^{B \times T_{\mathrm{src}} \times D}
$$

这组表示有几个关键特征：

- 每个位置都能看完整个输入序列
- 更偏向表征学习，而不是逐 token 生成
- 后续任务头可以拿这些表示去做分类、匹配、检索，或者提供给 decoder 做条件生成

一个标准 encoder layer 的主干只有两块：

1. multi-head self-attention
2. feed-forward network

写成最常见的形式就是：

$$
H' = \mathrm{LN}(H + \mathrm{MHA}(H, H, H))
$$

$$
H'' = \mathrm{LN}(H' + \mathrm{FFN}(H'))
$$

这里之所以叫 self-attention，是因为：

- $Q, K, V$ 都来自同一份 encoder 输入

如果对着最小实现看，`EncoderLayer` 就是这件事的直接代码化：

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
            x,
            x,
            x,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x
```

这里每一部分都对应一个明确作用：

- `self.self_attn`：做 multi-head self-attention。因为传进去的是 `x, x, x`，所以 `Q / K / V` 都来自 encoder 当前输入。
- `self.ffn`：做逐位置的非线性变换。attention 负责“和别的位置交换信息”，FFN 负责“每个位置自己做特征变换”。
- `self.norm1` / `self.norm2`：分别放在 attention 子层和 FFN 子层后面，配合 residual 稳定训练、控制特征尺度。
- `src_key_padding_mask`：告诉 attention 哪些 source 位置是 padding，避免模型把补齐位也当成有效上下文。

再按 `forward` 逐行看：

- `attn_out, _ = self.self_attn(x, x, x, ...)`：每个 source token 都去看整段 source，拿回一份融合上下文后的表示。
- `x = self.norm1(x + attn_out)`：残差分支保留原始输入，attention 分支补充上下文信息，再做归一化。
- `x = self.norm2(x + self.ffn(x))`：在已经完成上下文交互的表示上，再做一层逐位置 MLP 变换。
- `return x`：输出仍然是和输入同形状的 hidden states，但每个位置都已经带上更强的上下文信息。

所以如果你只看代码，也可以把 encoder layer 记成：

```text
self-attn -> residual + norm -> ffn -> residual + norm
```

### 3. Decoder 在做什么？

decoder 的核心任务是“按顺序生成输出”。

它在第 $t$ 步不能偷看未来，只能看：

- 已经生成的前 $t - 1$ 个 token
- encoder 提供的 source 表示

所以 decoder block 比 encoder 多一层 cross-attention，通常有三块：

1. masked self-attention
2. cross-attention
3. feed-forward network

公式可以写成：

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

- $S$ 是 decoder 当前隐状态
- $E$ 是 encoder 输出

这三步分别对应：

- 先建模目标端已经生成的前缀
- 再去读取源端条件信息
- 最后再做一层逐位置非线性变换

对着最小实现看，`DecoderLayer` 必须把这三步完整写出来：

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        self_attn_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self_attn_out)

        cross_attn_out, _ = self.cross_attn(
            x,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = self.norm2(x + cross_attn_out)
        x = self.norm3(x + self.ffn(x))
        return x
```

这里每个成员和每一段前向逻辑都不能讲混：

- `self.self_attn`：target 侧的 masked self-attention。它建模的是“已经生成的前缀之间如何相互作用”。
- `self.cross_attn`：decoder 读取 encoder memory 的入口。这里不是自注意力，而是条件注意力。
- `self.ffn`：和 encoder 一样，负责逐位置的非线性特征变换。
- `self.norm1 / norm2 / norm3`：decoder 比 encoder 多一个子层，所以也多一个对应的 residual + norm。
- `memory`：encoder 输出的上下文化表示，也就是 source 侧已经“读懂后”的结果。
- `tgt_mask`：因果 mask，专门阻止 decoder 看未来 token。
- `tgt_key_padding_mask`：target 里的 padding mask。
- `memory_key_padding_mask`：source 里的 padding mask，在 cross-attention 时屏蔽掉无效 source 位置。

再对着 `forward` 一步一步解释：

- `self.self_attn(x, x, x, attn_mask=tgt_mask, ...)`：先在 target 侧做自注意力，但因为有 `tgt_mask`，当前位置只能看自己和左边，不能看未来。
- `x = self.norm1(x + self_attn_out)`：先把 target 前缀内部的信息混合起来，再通过 residual 保留原始 target 表示。
- `self.cross_attn(x, memory, memory, ...)`：这一步最关键。`Q` 来自 decoder 当前状态 `x`，`K / V` 来自 encoder 输出 `memory`，也就是“拿当前生成需求去 source memory 里检索相关信息”。
- `x = self.norm2(x + cross_attn_out)`：把读到的 source 条件信息并回 decoder 当前状态。
- `x = self.norm3(x + self.ffn(x))`：最后再做一层逐位置非线性变换，增强表示能力。

这也是为什么 decoder 不能简单理解成“带 mask 的 encoder”。它比 encoder 多出来的，不只是一个 mask，而是一整段：

```text
current target state -> query
encoder memory -> key/value
cross-attention -> condition on source
```

### 4. 为什么 decoder 的第一层要 mask？

因为 decoder 是自回归生成。

如果当前位置能直接看到未来 token，训练时就等于偷答案，推理时也和真实生成过程不一致。

所以 decoder 的 self-attention 必须加 causal mask。对应到上面的代码，就是：

```python
self.self_attn(
    x,
    x,
    x,
    attn_mask=tgt_mask,
    ...
)
```

而 encoder 通常处理的是完整输入序列，不需要维持这种因果约束，所以一般不加 causal mask。

### 5. Cross-Attention 到底在干什么？

这是面试里很容易答虚的一题。

可以直接记住：

- decoder self-attention：建模 target 端上下文
- decoder cross-attention：对齐并读取 source 端信息

形式上看，cross-attention 和 self-attention 的最大区别是来源不同：

- self-attention：$Q, K, V$ 都来自同一侧
- cross-attention：$Q$ 来自 decoder，$K, V$ 来自 encoder

也就是：

$$
\mathrm{CrossAttention}(S, E)=\mathrm{Attention}(Q(S), K(E), V(E))
$$

对应到代码，就是这一句：

```python
cross_attn_out, _ = self.cross_attn(
    x,
    memory,
    memory,
    key_padding_mask=memory_key_padding_mask,
    need_weights=False,
)
```

这里要直接讲出来源关系：

- `x` 是 query，代表“decoder 当前想找什么信息”
- `memory` 是 key/value，代表“encoder 里有哪些 source 信息可供检索”

直觉上可以理解成：

> decoder 先明确“我现在要生成什么”，再去 encoder memory 里找和当前生成需求最相关的 source 信息。

### 6. 三种结构怎么区分？

这是一个非常高频的结构题。

- encoder-only：只负责理解输入，不负责开放式逐 token 生成
- encoder-decoder：先读 source，再生成 target
- decoder-only：只根据前缀持续生成后续 token

按任务目标看最容易：

- 理解型任务优先 encoder-only
- 条件生成任务优先 encoder-decoder
- 通用语言建模优先 decoder-only

典型例子可以这样记：

- BERT：encoder-only
- 原始 Transformer / T5：encoder-decoder
- GPT / LLaMA / Qwen：decoder-only

### 7. 为什么现代 LLM 大多是 decoder-only？

因为现代通用大语言模型的核心训练目标通常就是 next-token prediction。

这种目标天然对应：

> 给定前缀，预测下一个 token。

这时只要保留 decoder，并让 self-attention 满足因果性，就能直接做大规模自回归预训练。

所以现代 LLM 更多采用 decoder-only，而不是原始论文里的完整 encoder-decoder。

## 面试高频问题

### 1. encoder 和 decoder 的一句话区别是什么？

- encoder：读懂输入，产出上下文化表示
- decoder：按顺序生成输出，并在需要时读取 encoder memory

### 2. 为什么 decoder 比 encoder 多一层 attention？

因为 decoder 既要看 target 前缀，也要看 source 信息。前者用 masked self-attention，后者用 cross-attention。

### 3. 为什么 encoder 不需要 causal mask？

因为 encoder 处理完整输入，目标不是“逐步预测下一个 token”，所以不需要阻止当前位置看后文。

### 4. 为什么 decoder self-attention 必须 mask？

为了保持因果性，避免当前位置在训练时偷看未来 token。

### 5. 什么时候只用 encoder 就够了？

当任务重点是理解输入，而不是开放式生成时，比如分类、匹配、检索、序列标注。

### 6. 什么时候 encoder-decoder 更自然？

当 source 和 target 角色明显不同，需要“先理解输入，再条件生成输出”时，比如翻译、摘要、ASR。

### 7. 为什么现代 LLM 不常用完整 encoder-decoder？

因为通用预训练目标通常是 next-token prediction，decoder-only 更直接，也更符合统一生成接口。

## 最小实现

这个专题的最小实现不去搭一个完整 Transformer，而是只保留最能体现差异的几层：

- `EncoderLayer`
- `DecoderLayer`
- `Encoder`
- `Decoder`
- `build_causal_mask`

### 1. `Encoder` 和 `Decoder` 怎么堆叠 layer

```python
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x
```

```python
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(
        self,
        x,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return x
```

这两段代码表达的就是：

- encoder stack：不断重复“self-attn + FFN”的表征提炼
- decoder stack：不断重复“masked self-attn + cross-attn + FFN”的条件生成

### 2. 一个最小的数据流主线

因果 mask 的构造也很值得一起看：

```python
def build_causal_mask(seq_len, device):
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
```

它做出来的是一个上三角 mask，含义是：

- 对角线以上的位置不能看
- 也就是当前位置不能看未来位置

整个 encoder-decoder 主线则是：

```python
memory = encoder(src_x, src_key_padding_mask=src_key_padding_mask)
hidden = decoder(
    tgt_x,
    memory,
    tgt_mask=tgt_mask,
    tgt_key_padding_mask=tgt_key_padding_mask,
    memory_key_padding_mask=src_key_padding_mask,
)
```

可以直接概括成：

> encoder 先把 source 编成 memory，decoder 再在因果约束下结合这份 memory 生成 target 侧表示。

### 3. 阅读这份代码时应该怎么抓重点？

建议按这个顺序看：

1. 先看 `EncoderLayer.forward`，理解 encoder 只做 source 内部的信息交互
2. 再看 `DecoderLayer.forward`，重点盯住 `self.self_attn(...)` 和 `self.cross_attn(...)` 两段
3. 最后看 `Encoder` / `Decoder` 的 stack，把“单层机制”映射到“多层堆叠”

完整代码见：[minimal.py](minimal.py)

## 工程关注点

### 1. source padding mask 和 target causal mask 不一样

这是最容易在实现里混掉的地方。

- source padding mask：告诉模型哪些 source 位置是补齐的
- target causal mask：保证 decoder 不能看未来

两者职责完全不同。

### 2. cross-attention 的复杂度和 source / target 长度都相关

decoder self-attention 主要看 $T_{\mathrm{tgt}}^2$，而 cross-attention 主要看：

$$
T_{\mathrm{tgt}} \times T_{\mathrm{src}}
$$

所以 source 很长时，decoder 读取 encoder memory 的成本也会上来。

### 3. 训练和推理时 decoder 的工作方式不完全一样

- 训练时通常拿完整 target 序列并加 causal mask
- 推理时是一步一步自回归生成

如果你把这两种过程讲混，面试里很容易被追问卡住。

## 常见坑点

### 1. 以为 decoder 只是“带 mask 的 encoder”

不对。decoder 不只是多了 mask，还多了 cross-attention。

### 2. 把 self-attention 和 cross-attention 的 `Q / K / V` 来源讲反

这是最典型的概念错误。

### 3. 只会背结构图，不会按 source / target 数据流讲

面试里更重要的是你能讲清楚“谁先进去，谁给谁提供条件”。

### 4. 结构选型只背名字，不按任务目标判断

真正的判断标准不是模型名字，而是任务是在做理解、条件生成，还是开放式自回归生成。

## 面试时怎么讲

如果面试官问你 encoder 和 decoder 的区别，可以按这个顺序讲：

1. 原始 Transformer 是 encoder-decoder，不是现代常见的 decoder-only
2. encoder 负责把 source 编成上下文化表示，通常能双向看完整输入
3. decoder 负责按顺序生成 target，所以第一层 attention 必须带 causal mask
4. decoder 还要通过 cross-attention 读取 encoder 输出，所以它比 encoder 多一层 attention
5. 按任务目标选结构：理解任务偏 encoder-only，条件生成偏 encoder-decoder，通用语言建模偏 decoder-only

一个简洁版本可以直接讲：

> encoder 负责读懂输入，decoder 负责按顺序生成输出。encoder block 只有 self-attention 和 FFN；decoder block 除了 masked self-attention 和 FFN，还多一层 cross-attention，用来读取 encoder 产出的 memory。理解任务通常只要 encoder，条件生成更适合 encoder-decoder，而现代通用 LLM 多数是 decoder-only。

## 延伸阅读

- Attention 基础：[Self-Attention / Multi-Head Attention](../self_attention/README.md)
- 位置编码：[Positional Encoding / RoPE](../positional_encoding/README.md)
- 归一化：[LayerNorm / RMSNorm](../normalization/README.md)
- 完整结构串联：[最小完整 Transformer 实现](../transformer_minimal/README.md)
- 配套代码：[minimal.py](minimal.py)
