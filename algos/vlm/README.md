# VLM 总览：LLM 是怎么获得视觉能力的 面试攻略

## 专题顺序

- 第一篇：[VLM 总览：LLM 是怎么获得视觉能力的](README.md)
- 第二篇：[Vision Encoder + Projector + LLM 的基础拼接范式](vision_encoder_projector_llm.md)
- Paper Reading：[VLM 论文精读索引](paper_reading/README.md)

## 这是什么？

这是 VLM 机制部分的第一篇，先把最大的问题讲清楚：

- LLM 原本只会处理 token，为什么后来能看图？
- VLM 到底是在 LLM 前面接了什么东西？
- 图像是怎么被改写成 LLM 能消费的“视觉 token”的？
- 训练时到底在教模型学什么？

如果只用一句话概括：

> VLM 的本质，不是“让 LLM 直接看像素”，而是先用视觉编码器把图像变成视觉特征，再用 projector 把视觉特征对齐到 LLM 的表示空间里，让 LLM 像读一段前缀 token 一样去消费这些视觉信息。

## 核心机制

### 1. 为什么原始 LLM 没有视觉能力？

因为原始 LLM 的输入通常只有离散 token id。

它最熟悉的数据流是：

```text
text -> tokenizer -> token ids -> token embeddings -> Transformer blocks -> next-token logits
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

如果写成更抽象的形式：

$$ v = f_{\mathrm{img}}(x), \quad z = g(v), \quad h_0 = [z; e(t)] $$

这里：

- $x$：输入图像
- $f_{\mathrm{img}}$：视觉编码器，比如 ViT
- $v$：视觉特征
- $g$：projector，把视觉特征映射到 LLM 的隐藏空间
- $e(t)$：文本 token embedding
- $[z; e(t)]$：把视觉 token 和文本 token 拼成一个联合序列

这就是 VLM 最值得先讲明白的一点：

> LLM 不是突然“长出眼睛”了，而是前面先有人把图像翻译成了它能理解的向量序列。

### 3. 为什么需要 projector？

因为视觉编码器和 LLM 的表示空间通常不一样。

例如：

- 视觉编码器输出维度可能是 `1024`
- LLM 隐藏维度可能是 `4096`
- 两边的统计分布和语义组织方式也不一样

所以中间要有一层桥接：

- 最简单是 `Linear`
- 常见一点是 `MLP`
- 更复杂会用 `Resampler / Q-Former / Cross-Attention`

projector 做的事可以直接讲成：

> 它负责把“视觉 backbone 的特征”改写成“LLM 愿意接收的输入 token”。

### 4. VLM 是怎么被训练出来的？

通常至少包含两层意思：

#### 1. 先学图文对齐

目标是让视觉内容和语言描述进入相容的语义空间。

这一步里常见的 supervision 包括：

- 图文对
- 图像描述
- 区域和短文本对应
- 问答对

#### 2. 再学视觉条件下的指令跟随

这一步更像 instruction tuning，只不过上下文里多了图像。

模型会学：

- 先读视觉 token
- 再结合用户问题
- 最后按语言模型方式生成回答

所以从训练视角看，VLM 不是一个单点技巧，而是：

> 视觉表征、跨模态对齐和语言生成三件事叠在一起。

### 5. 推理时，VLM 到底在做什么？

它做的仍然是 next-token prediction。

只是当前上下文不再只有文字，而是：

```text
[visual tokens] + [prompt tokens]
```

所以推理时的本质没有变：

- 前面先塞进图像对应的视觉 token
- 再塞进问题文本
- 模型按自回归方式一步一步生成答案

这也是为什么很多 VLM 的输出头，仍然就是普通 `lm_head`。

### 6. 为什么不把像素直接喂给 LLM？

因为代价太高，效果通常也不好。

主要问题有三个：

- 图像分辨率很高，直接展平成 token 长度太大
- 像素空间过于底层，LLM 不擅长直接从原始像素里学视觉归纳偏置
- 训练成本会非常高

所以更合理的路径是：

- 先用视觉模型提取局部和全局结构
- 再把压缩后的语义特征交给 LLM

### 7. 多图和视频，为什么还能沿着这条路继续做？

因为从 LLM 的视角看，它最终只是在读一段更长的前缀。

多图时可以：

- 每张图分别编码，再拼接视觉 token
- 插入图像分隔符 token

视频时可以：

- 先做帧采样
- 每帧做视觉编码
- 再加时间位置编码或 temporal module

所以很多视频 VLM 的本质仍然没有变，只是：

> 前缀 token 从单张图像扩展成了“带时间结构的视觉 token 序列”。

## 面试高频问题

### 1. VLM 和纯 LLM 的根本差异是什么？

VLM 在 LLM 前面多了视觉编码和跨模态对齐步骤，让图像能被改写成 LLM 可以消费的 token 序列。

### 2. 为什么 VLM 一般不是直接让 LLM 读像素？

因为像素序列太长、过于底层、训练代价高，而且 LLM 本身没有视觉归纳偏置。

### 3. projector 的作用是什么？

把视觉特征映射到 LLM 的隐藏空间里，让视觉 token 和文本 token 能进入同一条计算图。

### 4. VLM 推理时是不是还是 next-token prediction？

是。区别只是上下文里除了文字，还多了视觉前缀 token。

### 5. 为什么很多 VLM 喜欢冻结一部分视觉 encoder 或 LLM？

因为两边都很大，先冻结大 backbone 再只训 projector 或少量适配层，更稳定也更省算力。

### 6. 为什么高分辨率图像会让 VLM 很痛苦？

因为视觉 token 数会迅速膨胀，直接挤占 LLM 的上下文窗口和显存预算。

## 最小实现

完整代码见 [minimal_vlm_bridge.py](minimal_vlm_bridge.py)。

下面这段骨架展示了最常见的 prefix-style VLM：

```python
class TinyVLM(nn.Module):
    def __init__(self, vocab_size, vision_width=64, d_model=128):
        super().__init__()
        self.vision_encoder = PatchEncoder(width=vision_width)
        self.projector = MLPProjector(in_dim=vision_width, out_dim=d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder = nn.TransformerEncoder(...)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def encode_image(self, images):
        visual_tokens = self.vision_encoder(images)
        return self.projector(visual_tokens)

    def forward(self, images, input_ids):
        visual_tokens = self.encode_image(images)
        text_tokens = self.token_embedding(input_ids)
        x = torch.cat([visual_tokens, text_tokens], dim=1)
        mask = build_prefix_causal_mask(
            num_visual_tokens=visual_tokens.shape[1],
            num_text_tokens=text_tokens.shape[1],
            device=x.device,
        )
        x = self.decoder(x, mask=mask)
        return self.lm_head(x[:, visual_tokens.shape[1] :])
```

这段代码里最值得讲清楚的是：

- `self.vision_encoder`：先把图像切成 patch 并提成视觉特征
- `self.projector`：把视觉特征映射到 `d_model`
- `torch.cat([visual_tokens, text_tokens], dim=1)`：把视觉 token 当成文本前缀
- `build_prefix_causal_mask(...)`：让文本能看见全部视觉 token，同时保持文本自回归约束
- `x[:, visual_tokens.shape[1] :]`：最终只对文本位置做语言建模输出

## 工程关注点

- 视觉 token 数量和上下文长度怎么平衡
- projector 是训线性层、MLP，还是更重的 Q-Former
- 视觉 encoder 和 LLM 哪部分冻结，哪部分微调
- 高分辨率、多图、长视频时怎么控显存
- 训练数据里 OCR、表格、图表、定位类样本是否足够

## 常见坑点

- 只会背“图像接到 LLM 前面”，但说不清中间为什么需要 projector
- 说 VLM 学会视觉，是因为“LLM 参数够大”，忽略了视觉 encoder 的作用
- 混淆“图文对齐”和“视觉问答指令微调”这两个训练阶段
- 忽视视觉 token 数膨胀导致的上下文和显存问题

## 面试时怎么讲

比较稳的一种讲法是：

> VLM 本质上是在 LLM 前面接了一套视觉前端。图像先被 vision encoder 编成视觉特征，再通过 projector 映射到 LLM 的隐藏空间，最后把这些视觉 token 当成前缀上下文，让 LLM 按 next-token prediction 的方式生成回答。所以它不是让 LLM 直接学像素，而是让 LLM 学会如何消费已经编码好的视觉语义。

## 延伸阅读

- 第二篇：[Vision Encoder + Projector + LLM 的基础拼接范式](vision_encoder_projector_llm.md)
- 论文精读索引：[paper_reading/README.md](paper_reading/README.md)
