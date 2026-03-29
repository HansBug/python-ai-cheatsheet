# VLM 总览：LLM 是怎么获得视觉能力的 面试攻略

## 专题顺序

- 第一篇：[VLM 总览：LLM 是怎么获得视觉能力的](README.md)
- 第二篇：[Vision Encoder + Projector + LLM 的基础拼接范式](vision_encoder_projector_llm.md)
- Paper Reading：[VLM 论文精读索引](paper_reading/README.md)

## 这是什么？

这是 VLM 机制部分的第一篇，先把主线讲清楚：

- LLM 原本为什么不会看图
- VLM 到底是在 LLM 前面接了什么
- image encoder 和 projector 分别负责什么
- 为什么图像最后能变成 LLM 可以消费的 token 序列

如果只用一句话概括：

> VLM 不是让 LLM 直接学像素，而是先用视觉编码器把图像压成视觉特征，再用 projector 把这些特征桥接到 LLM 的隐藏空间里，让 LLM 像读前缀 token 一样读视觉信息。

## 核心机制

### 1. 为什么原始 LLM 没有视觉能力？

原始 LLM 最熟悉的数据流是：

```text
text -> tokenizer -> token ids -> token embeddings -> Transformer -> next-token logits
```

这里没有任何一步会直接处理：

- 像素网格
- 图像 patch
- 空间位置关系

所以 LLM 默认学到的是语言分布，不是视觉分布。

### 2. VLM 最核心的数据流是什么？

最小主线可以记成：

```text
image -> vision encoder -> visual features -> projector -> visual tokens
text -> tokenizer -> text tokens
visual tokens + text tokens -> LLM -> answer tokens
```

写成抽象形式就是：

$$ v = f_{\mathrm{img}}(x), \quad z = g(v), \quad h_0 = [z; e(t)] $$

这里：

- $x$：输入图像
- $f_{\mathrm{img}}$：视觉编码器
- $v$：视觉特征
- $g$：projector
- $e(t)$：文本 token embedding
- $[z; e(t)]$：视觉 token 和文本 token 拼接后的联合序列

最关键的一句话是：

> LLM 不是“突然能看图”，而是有人先把图像翻译成了它能读的向量序列。

### 3. VLM 里的 image encoder 常见有哪些？

这里先澄清一个很容易混的点：

> VLM 里常说“用 CLIP encoder”或“用 SigLIP encoder”，更准确地说，是拿 CLIP / SigLIP 这类图文预训练方法训出来的视觉 tower 当 image encoder 用，而不是把整个 CLIP / SigLIP 原封不动塞进来。

常见 image encoder 大致有这几类：

#### 1. CLIP 风格的 ViT visual tower

这是最经典的一类。

特点是：

- 已经有很强的图文对齐先验
- 语义级表征成熟
- 很多早期开源 VLM 都直接用它当视觉前端

面试里如果你说：

> 很多 LLaVA-style 模型早期常直接接 CLIP ViT 视觉塔

这个方向是对的。

#### 2. SigLIP 风格的 ViT visual tower

这类 encoder 本质上还是 ViT，但预训练目标从 CLIP 的 softmax 对比学习，换成了 sigmoid matching。

特点通常是：

- 视觉表征常更强
- 对训练 batch 组织的依赖相对更弱
- 近几年越来越常被拿来当 VLM 的视觉前端

#### 3. EVA / EVA-CLIP / InternViT 这类更强的大视觉 backbone

这类本质上还是大规模 ViT 路线，但参数量、分辨率和细节感知能力更强。

常见用途是：

- 希望视觉底座更强
- 希望 OCR、文档、细粒度感知能力更好
- 希望高分辨率输入时保留更多视觉细节

#### 4. CNN / Swin / ConvNeXt 一类视觉 backbone

也能用，但在通用 VLM 里没有 ViT 路线那么主流。

主要原因是：

- 现代 VLM 喜欢把视觉表示改写成 token 序列
- ViT 天然更容易和 LLM 的 token 流对接

#### 5. 视频或多图场景下的视觉 encoder

当输入变成多图或视频时，常见做法是：

- 逐帧用图像 encoder 编码
- 再加 temporal module
- 或者直接用视频视觉 backbone

所以从大方向上看：

> 通用 VLM 里最主流的 image encoder 仍然是 ViT 家族，只是预训练方法和规模不同。

### 4. projector 一般是什么结构？

projector 不是“顺手接一层线性层”那么简单，它至少要解决三件事：

- 维度对齐：`vision_width -> d_model`
- 分布对齐：视觉特征分布和 LLM embedding 分布通常不同
- 语义桥接：让视觉 token 更像 LLM 能消费的上下文

常见 projector 结构大致可以分成四类：

#### 1. Linear projector

最简单：

```text
[B, N, vision_width] -> Linear -> [B, N, d_model]
```

优点：

- 参数少
- 稳定
- 容易训

适合：

- 最小可跑链路
- 先验证视觉桥接是否成立

#### 2. MLP projector

比线性层多一点非线性变换，例如两层 MLP：

```text
Linear -> GELU -> Linear
```

优点是表达力更强，很多开源 VLM 都喜欢从这里起步。

#### 3. Q-Former / Cross-Attention bridge

这类 projector 不只是改维度，而是主动“从视觉特征里查询有用信息”。

优点：

- 更强
- 更适合从大量视觉 token 里抽关键信息

代价：

- 更重
- 训练更复杂

#### 4. Resampler / Token Compression 模块

这类 projector 会顺手做 token 压缩。

适合：

- 高分辨率图像
- 多图输入
- 视频输入

因为这时真正痛的，不只是维度不匹配，而是：

> 视觉 token 太多，直接把上下文窗口和显存打爆了。

### 5. 结合代码看，VLM 的主干到底在做什么？

完整代码见 [minimal_vlm_bridge.py](minimal_vlm_bridge.py)。

下面这段代码，已经把 `image encoder -> projector -> LLM` 的主链路连起来了：

```python
class TinyVLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        patch_size=16,
        vision_width=64,
        d_model=128,
        num_heads=4,
        num_layers=2,
        max_len=512,
        projector_type="mlp",
    ):
        super().__init__()
        # 真实系统里这通常对应 CLIP / SigLIP / EVA / InternViT 之类的视觉塔。
        self.vision_encoder = PatchEncoder(
            in_channels=3,
            patch_size=patch_size,
            width=vision_width,
        )

        # projector 负责把视觉宽度桥接到 LLM 的隐藏维度。
        self.projector = build_projector(
            projector_type=projector_type,
            in_dim=vision_width,
            out_dim=d_model,
        )

        # 文本 token 走正常的语言模型嵌入。
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, images, input_ids):
        # 1) 先把像素图像压成视觉 token。
        visual_tokens = self.vision_encoder(images)

        # 2) 再把视觉 token 映射到语言模型隐藏空间。
        visual_tokens = self.projector(visual_tokens)

        # 3) 文本 token 走普通 embedding。
        text_tokens = self.token_embedding(input_ids)

        # 4) 把视觉 token 当成文本前缀，和文本拼到同一条序列里。
        x = torch.cat([visual_tokens, text_tokens], dim=1)

        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        # 5) 文本能看见全部视觉 token，但文本之间仍保持自回归 mask。
        mask = build_prefix_causal_mask(
            num_visual_tokens=visual_tokens.shape[1],
            num_text_tokens=text_tokens.shape[1],
            device=x.device,
        )
        x = self.decoder(x, mask=mask)

        # 6) 最后只取文本位置输出词表 logits。
        text_hidden = x[:, visual_tokens.shape[1] :]
        text_hidden = self.ln_f(text_hidden)
        return self.lm_head(text_hidden)
```

你要把这段代码一口气讲顺，最好按下面这个顺序：

- `self.vision_encoder`：把图像改写成视觉 token 序列
- `self.projector`：把视觉宽度对齐到 LLM 隐藏维
- `self.token_embedding`：把文本 id 变成文本 token
- `torch.cat([visual_tokens, text_tokens], dim=1)`：把视觉 token 当作前缀条件
- `build_prefix_causal_mask(...)`：保证文本能读取视觉，但文本生成仍是自回归
- `x[:, visual_tokens.shape[1] :]`：最后只对文本位置做语言建模输出

这也是 VLM 面试里最应该讲清楚的一件事：

> `__init__` 决定你接了哪些桥，`forward` 决定视觉 token 如何一步一步变成语言模型的上下文。

### 6. VLM 是怎么训出来的？

通常至少有两层意思：

#### 1. 图文对齐

让视觉内容和语言内容进入相容的语义空间。

#### 2. 视觉条件下的指令跟随

让模型学会：

- 先读视觉 token
- 再读用户问题
- 最后按 next-token prediction 生成回答

所以从训练视角看，VLM 不是一个单点技巧，而是：

> 视觉表征、跨模态桥接和语言生成三件事叠在一起。

### 7. 多图和视频为什么还能沿着这条路继续做？

因为从 LLM 的视角看，它最终读到的仍然是一段前缀 token。

多图时可以：

- 每张图各自编码
- 插入分隔符 token
- 再统一拼接

视频时可以：

- 先做帧采样
- 每帧走 image encoder
- 再引入时间位置编码、temporal module 或 token 压缩

所以很多视频 VLM 的本质仍然没变，只是：

> 前缀 token 从单张图像扩展成了带时间结构的视觉 token 序列。

## 面试高频问题

### 1. VLM 里常说的 CLIP encoder / SigLIP encoder，准确是什么意思？

通常是指拿 CLIP / SigLIP 训练出来的视觉 tower 当 image encoder，用的往往还是 ViT 主干。

### 2. 为什么现在很多通用 VLM 喜欢 ViT 家族视觉 encoder？

因为它天然输出 token 序列，更容易和 LLM 的 token 流对接。

### 3. projector 真的只是改维度吗？

不是。弱 projector 看起来像改维度，强 projector 还承担分布对齐、语义桥接和 token 压缩。

### 4. 为什么很多 VLM 喜欢先冻结 image encoder 或 LLM？

因为两边都很大，先冻结大 backbone 只训桥接层，更稳定也更省算力。

### 5. 为什么高分辨率图像会让 VLM 很痛苦？

因为视觉 token 数暴涨，直接挤占上下文窗口和显存预算。

## 工程关注点

- image encoder 选 CLIP、SigLIP，还是更强的 ViT 家族
- projector 是线性层、MLP，还是更重的 Q-Former / Resampler
- 视觉 token 数量和上下文长度怎么平衡
- 多图、视频、高分辨率输入如何压缩 token
- OCR、表格、图表、grounding 类数据是否足够

## 常见坑点

- 把“CLIP / SigLIP”误说成具体 backbone 名，而不是图文预训练路线
- 只会说“接个 projector”，却说不清 projector 解决了什么问题
- 把 projector 讲成纯维度映射，忽略 token 压缩和语义桥接
- 只会说结构，不会按代码顺序把 `__init__ -> forward` 讲通

## 面试时怎么讲

比较稳的一种讲法是：

> VLM 本质上是在 LLM 前面接了一套视觉前端。常见 image encoder 大多是 ViT 家族，比如 CLIP 或 SigLIP 训练出来的视觉 tower；再往后通过 linear、MLP、Q-Former 或 resampler 这类 projector，把视觉特征桥接到 LLM 隐藏空间里。之后把视觉 token 当成前缀条件拼到文本前面，让语言模型按 next-token prediction 的方式生成回答。

## 延伸阅读

- 第二篇：[Vision Encoder + Projector + LLM 的基础拼接范式](vision_encoder_projector_llm.md)
- 论文精读索引：[paper_reading/README.md](paper_reading/README.md)
