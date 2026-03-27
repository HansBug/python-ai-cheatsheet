# LLM 结构与推理流程 面试攻略

## 专题顺序

- 第一篇：[LLM 结构与推理流程](README.md)
- 第二篇：KV Cache（待补）
- 第三篇：MoE（待补）

## 这是什么？

这是 LLM 机制与工程部分的第一篇，先把最容易混的主线讲清楚：

- LLM 和 Transformer 到底是什么关系
- LLM 真的只是“把 Transformer 做大一点”吗
- 一个 decoder-only LLM 的结构到底长什么样
- 推理时每一步模型输出的是什么
- 为什么每一轮只取最后一个位置的输出
- 文本是怎么一步一步生成出来的

如果只用一句话概括：

> LLM 的主干确实来自 Transformer，但它不是“把参数放大”就结束了，而是围绕下一 token 预测、decoder-only 结构、训练配方和推理系统一起演化出来的一整套体系。

如果后面要看 `KV Cache`，这一篇应该先吃透，因为 KV Cache 讲的就是这里这条推理流程怎么被加速。

## 核心机制

### 1. LLM 和 Transformer 到底是什么关系？

先给最短结论：

- 现代主流 LLM 的结构骨架，通常就是 Transformer
- 但更常见的是 **decoder-only Transformer**
- 同时还叠加了 tokenizer、海量预训练、长上下文、推理优化、指令微调等一整套工程体系

所以如果面试官问：

> LLM 是不是只是把 Transformer 搞大一点？

比较稳的回答不是简单说“是”或“不是”，而是：

> 从骨架上看，LLM 确实大量继承了 Transformer，尤其是 decoder-only 结构；但从训练目标、数据规模、位置编码、归一化方式、推理流程和部署优化来看，它已经不是“只把模型放大”这么简单。

### 2. 为什么现代 LLM 大多是 decoder-only？

因为它最适合做下一 token 预测。

LLM 预训练最核心的目标通常可以写成：

$$ p(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, x_2, \ldots, x_{t-1}) $$

这里：

- $x_t$：第 `t` 个 token
- $x_1, x_2, \ldots, x_{t-1}$：当前位置之前的所有 token

这句话翻成人话就是：

> 给你前面的上下文，预测下一个 token 最可能是什么。

而 decoder-only 结构天然带 causal mask，正好满足“只能看左边、不能看右边”的要求。

### 3. 一个最小 decoder-only LLM 长什么样？

你可以先看这份最小代码骨架，完整文件在 [minimal_decoder_only.py](minimal_decoder_only.py)。

```python
class TinyDecoderOnlyLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        max_len=64,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
```

这段代码里，每一块都对应一个 LLM 的关键部件：

- `token_embedding`：把 token id 变成向量
- `position_embedding`：告诉模型位置顺序
- `blocks`：多层 decoder block 堆叠
- `ln_f`：最终归一化
- `lm_head`：把最后的隐藏状态投影回词表空间

你要注意，这里没有 encoder，也没有 cross-attention。

这就是 decoder-only LLM 和“完整 encoder-decoder Transformer”的最大结构差异之一。

### 4. LLM 的 block 和普通 Transformer block 有什么关系？

核心仍然是两块：

- causal self-attention
- feed-forward network

在最小实现里，一个 block 是这样写的：

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, x):
        seq_len = x.shape[1]
        causal_mask = build_causal_mask(seq_len=seq_len, device=x.device)

        h = self.ln_1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out

        h = self.ln_2(x)
        x = x + self.ffn(h)
        return x
```

你可以直接这么讲：

- `ln_1` + `attn`：让每个 token 在因果约束下看左侧上下文
- `ln_2` + `ffn`：对每个位置再做一层非线性变换

这里每一步对应的含义很明确：

- `build_causal_mask(...)`：保证当前位置不能偷看未来 token
- `self.attn(h, h, h, ...)`：decoder-only 的自注意力
- `x = x + attn_out`：第一条残差支路
- `x = x + self.ffn(h)`：第二条残差支路

所以从结构上说，LLM 确实是在用 Transformer block，但它用的是“适合自回归生成”的那一半。

### 5. LLM 前向传播时，模型到底输出什么？

这是最容易被误解的地方。

LLM 每次前向传播，输出的不是“文字答案”，而是：

> 对词表里每个 token 的分数，也就是 logits。

在最小实现里最后一步是：

```python
def forward(self, input_ids):
    batch_size, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    positions = positions.expand(batch_size, seq_len)

    x = self.token_embedding(input_ids) + self.position_embedding(positions)
    for block in self.blocks:
        x = block(x)

    x = self.ln_f(x)
    logits = self.lm_head(x)
    return logits
```

这里：

- `x` 的形状是 `[B, T, D]`
- `lm_head(x)` 之后，`logits` 的形状会变成 `[B, T, V]`

其中：

- `B`：batch size
- `T`：当前序列长度
- `V`：词表大小

所以：

> 序列里每个位置，模型都会输出一个长度为 `V` 的分数向量。

这个分数向量表示：

- 词表里每个 token 作为“下一个 token”时有多像正确答案
- 分数越大，模型越倾向于选它

### 6. 为什么说“每个位置都在预测下一个 token”？

假设输入 token 是：

```text
<bos> I like apples
```

那模型会对每个位置都给出一组 logits：

- 位置 0 的输出：预测 `<bos>` 后面最可能是什么
- 位置 1 的输出：预测 `I` 后面最可能是什么
- 位置 2 的输出：预测 `like` 后面最可能是什么
- 位置 3 的输出：预测 `apples` 后面最可能是什么

换句话说：

> 第 `t` 个位置输出的 logits，表示“看完前 `t` 个位置后，下一 token 的分布”。

这也是为什么训练时，通常会把输入序列和目标序列错开一位。

举个很简单的例子：

```text
输入:  <bos> I like apples
目标:  I like apples <eos>
```

这时：

- 看到 `<bos>`，学着预测 `I`
- 看到 `<bos> I`，学着预测 `like`
- 看到 `<bos> I like`，学着预测 `apples`
- 看到 `<bos> I like apples`，学着预测 `<eos>`

### 7. 推理时为什么只取最后一个位置？

这是理解 `KV Cache` 前最关键的问题。

如果当前输入是：

```text
<bos> I like
```

模型前向后会输出：

```text
logits shape = [1, 3, V]
```

也就是：

- 第 0 个位置一组 logits
- 第 1 个位置一组 logits
- 第 2 个位置一组 logits

但推理时真正有用的是最后一个位置：

```python
next_token_logits = logits[:, -1, :]
```

为什么？

因为你现在真正要问的是：

> 已经看完完整前缀 `<bos> I like` 之后，下一个 token 应该是什么？

这个问题正对应最后一个位置的输出。

前面那些位置的 logits 并不是“错的”，只是它们对应的是更短前缀下的预测：

- 第 0 个位置只对应 `<bos>`
- 第 1 个位置只对应 `<bos> I`

而你生成下一词时，需要的是“最长上下文条件下”的那一组分数。

### 8. logits、probability、token 各自是什么关系？

这是另一个很容易混的点。

假设最后一个位置的 logits 长这样：

```text
apples  : 4.2
bananas : 3.6
today   : 0.5
<eos>   : -1.0
```

这里：

- 这些还只是分数，不是概率
- 经过 softmax 后，才会变成概率分布

比如 softmax 后可能变成：

```text
apples  : 0.53
bananas : 0.29
today   : 0.05
<eos>   : 0.01
```

然后再根据生成策略选 token：

- greedy：直接选概率最大的 `apples`
- sampling：按分布随机采样，可能采到 `bananas`

所以流程是：

```text
hidden state -> logits -> softmax -> 概率分布 -> 选一个 token id
```

模型输出的是 logits，不是直接输出字符串。

### 9. 一次完整的 infer 到底怎么走？

推理主线其实很固定：

```text
文本 prompt -> tokenizer -> token ids -> 模型前向 -> 取最后位置 logits -> 选 next token -> 追加回输入 -> 下一轮继续
```

在最小实现里，生成循环是：

```python
def generate_greedy(model, input_ids, tokenizer, max_new_tokens):
    generated = input_ids.clone()

    for step in range(max_new_tokens):
        logits = model(generated)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token_id], dim=1)
```

这一小段里，每一行都要能讲：

- `model(generated)`：对当前完整前缀做一次前向
- `logits[:, -1, :]`：只取“当前完整前缀下的下一 token 分数”
- `argmax(...)`：用 greedy 策略选下一个 token
- `torch.cat(...)`：把新 token 拼回输入，准备下一轮

这就是自回归生成的本质：

> 不是一次把整句话“想出来”，而是每次只决定下一个 token，然后把它接回上下文继续算。

### 10. 一个完整的流程例子

假设 prompt 是：

```text
I like
```

tokenizer 之后，可能得到：

```text
[<bos>, I, like]
```

第一轮前向：

- 输入长度是 3
- 输出 logits 形状是 `[1, 3, V]`
- 取最后一个位置 logits
- 如果选中 `apples`

那序列会变成：

```text
[<bos>, I, like, apples]
```

第二轮前向：

- 输入长度变成 4
- 输出 logits 形状是 `[1, 4, V]`
- 再取最后一个位置 logits
- 如果选中 `.`

那序列继续变成：

```text
[<bos>, I, like, apples, .]
```

第三轮前向：

- 如果模型这次选出 `<eos>`
- 那生成结束

你会发现，整件事的核心其实非常机械：

```text
不断“前向 -> 取最后位置 -> 选 token -> 拼回去”
```

后面 `KV Cache` 优化的，就是这里每一轮都全量重算太浪费的问题。

### 11. 所以 LLM 真不是“只把 Transformer 做大一点”？

从“结构骨架”上看，确实大体可以说是：

- token embedding
- positional information
- 多层 decoder block
- 最后接词表投影头

但如果你只说到这里，还是太浅了。

真正让 LLM 和“课堂上的最小 Transformer”拉开差距的，至少还有这些：

#### 1. 目标不一样

LLM 重点是大规模 next-token prediction，而不是小数据集上的特定任务拟合。

#### 2. 结构细节不一样

现代 LLM 常见还有：

- decoder-only
- RoPE 或其他更适合长上下文的位置编码
- RMSNorm / Pre-Norm
- SwiGLU 等变体 FFN

#### 3. 训练配方不一样

包括：

- 超大规模语料
- 大 batch 训练
- 学习率和 warmup 策略
- 混合精度
- 分布式训练

#### 4. 推理系统不一样

包括：

- KV Cache
- continuous batching
- quantization
- speculative decoding

所以一个更准确的说法是：

> LLM 的结构骨架确实来自 Transformer，但现代 LLM 是“Transformer 主干 + 大规模训练 + 自回归推理系统 + 指令对齐”的组合体，不是只把层数和参数量调大就自动成立。

## 面试高频问题

### 1. LLM 和 Transformer 的关系是什么？

主流 LLM 大多建立在 Transformer 上，尤其是 decoder-only Transformer。

### 2. 为什么 LLM 推理时只取最后一个位置的 logits？

因为我们要的是“当前完整前缀下，下一个 token 的分布”，这正对应最后一个位置。

### 3. 模型输出的是文字吗？

不是，模型直接输出的是 logits，也就是词表上每个 token 的分数。

### 4. logits 和概率是什么关系？

logits 经过 softmax 才会变成概率分布。

### 5. 生成为什么是一轮一轮来的？

因为 decoder-only LLM 是自回归生成，每次只决定下一个 token。

### 6. 为什么 LLM 推理比分类模型慢？

因为它不是一次前向就结束，而是每生成一个 token 都要再来一轮。

## 最小实现

完整代码见：[minimal_decoder_only.py](minimal_decoder_only.py)。

这份最小实现刻意保留了最重要的几块：

- `SimpleTokenizer`
- `DecoderBlock`
- `TinyDecoderOnlyLM`
- `generate_greedy`

### 1. tokenizer 在这里是干什么的？

最小 tokenizer 很简单：

```python
class SimpleTokenizer:
    def encode(self, text, add_bos=True):
        tokens = text.strip().split()
        ids = [self.token_to_id[token] for token in tokens]
        if add_bos:
            ids = [self.bos_token_id] + ids
        return ids
```

这里虽然只是空格切词的 toy 版，但它足够说明：

- 模型输入不是字符串
- 模型输入是 token ids
- `BOS` 这种特殊 token 也会进入模型

### 2. 模型前向为什么会输出 `[B, T, V]`？

看 `TinyDecoderOnlyLM.forward`：

```python
def forward(self, input_ids):
    batch_size, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    positions = positions.expand(batch_size, seq_len)

    x = self.token_embedding(input_ids) + self.position_embedding(positions)
    for block in self.blocks:
        x = block(x)

    x = self.ln_f(x)
    logits = self.lm_head(x)
    return logits
```

这段代码要这样理解：

- `token_embedding(input_ids)`：把 `[B, T]` 的 token ids 变成 `[B, T, D]`
- `+ position_embedding(...)`：加上位置信息
- `for block in self.blocks`：经过多层 decoder block
- `lm_head(x)`：把每个位置的隐藏状态投到词表空间

所以最后自然会得到：

- 每个 batch 一个输出
- 每个 token 位置一个输出
- 每个位置都对应整张词表的一组分数

### 3. 生成循环为什么是 LLM 推理的核心？

因为真正的 infer 并不神秘，核心就这几行：

```python
logits = model(generated)
next_token_logits = logits[:, -1, :]
next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
generated = torch.cat([generated, next_token_id], dim=1)
```

这段代码分别在做：

- `model(generated)`：先把当前前缀整段送进模型
- `logits[:, -1, :]`：只拿最后一个位置的分数
- `argmax(...)`：从词表中选出下一个 token
- `cat(...)`：把这个 token 拼回序列

只要这 4 行你能讲清楚，LLM 推理流程你就已经讲对了大半。

### 4. 这份脚本跑出来会看到什么？

脚本会打印：

- prompt 的 token ids
- 前向输出的 `logits` 形状
- 每个位置 top-k token 概率
- greedy 生成时每一轮序列怎么变长

这里需要特别强调：

> 这是一份未训练的 toy 模型，输出内容本身没有语言意义，重点是看数据流和形状变化。

## 工程关注点

### 1. 真正线上推理不会每轮都全量傻算

否则每生成一个 token 都重算整段前缀，代价太高。

这正是后面 `KV Cache` 要解决的问题。

### 2. tokenizer 设计会影响整个系统

词表大小、切词粒度、特殊 token 设计都会影响训练和推理。

### 3. 生成策略不只一种

实际推理里常见还有：

- temperature
- top-k
- top-p
- repetition penalty

### 4. 最大上下文长度不是白送的

长上下文不仅和位置编码有关，也和显存、attention 复杂度、cache 管理有关。

## 常见坑点

### 1. 以为模型直接输出文本

模型直接输出的是 logits，不是字符串。

### 2. 不知道为什么只取最后一个位置

这是没真正弄清 next-token prediction。

### 3. 把 LLM 和 encoder-decoder Transformer 混为一谈

现代通用 LLM 更常见的是 decoder-only。

### 4. 觉得 LLM 只是“参数更多”

如果只这么说，说明对训练目标和推理系统理解还太浅。

## 面试时怎么讲

可以按这条线讲：

1. 现代 LLM 的结构骨架通常是 decoder-only Transformer。
2. 输入文本先经过 tokenizer 变成 token ids，再经过 embedding 和多层 decoder block。
3. 模型前向输出的不是文本，而是每个位置对整个词表的 logits，形状一般是 `[B, T, V]`。
4. 推理时只取最后一个位置的 logits，因为我们关心的是“当前完整前缀下，下一个 token 的分布”。
5. 选出 next token 后把它接回输入，继续下一轮，所以生成本质是一个自回归循环。
6. LLM 当然继承了 Transformer，但它不是“只做大”，还包括自回归目标、大规模训练和一整套推理优化。

## 延伸阅读

- Attention Is All You Need
- GPT-2 / GPT-3 technical report
- LLaMA technical report
- Transformer 推理优化与 KV Cache 相关资料
