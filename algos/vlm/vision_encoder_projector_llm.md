# Vision Encoder + Projector + LLM 的基础拼接范式

## 这是什么？

这是 VLM 机制部分的第二篇，专门讲最常见的一种结构：

```text
Vision Encoder -> Projector -> LLM
```

这篇不再泛泛讲“VLM 是什么”，而是直接拆它的主干：

- 视觉编码器在做什么
- projector 为什么是桥，而不是可有可无的一层线性层
- 视觉 token 和文本 token 是怎么拼在一起的
- 为什么很多开源 VLM 都围绕这条范式展开

## 核心机制

### 1. 这条范式为什么最常见？

因为它最符合工程现实：

- 视觉领域已经有成熟 backbone
- 语言领域已经有成熟 LLM
- 中间只需要一个可训练的桥接层，就能把两边接起来

所以这条路线天然适合：

- 复用预训练模型
- 分阶段训练
- 按预算控制冻结和微调范围

### 2. Vision Encoder 在做什么？

Vision encoder 的任务不是“直接回答问题”，而是把图像编码成一串视觉 token。

常见选择包括：

- ViT
- EVA
- ConvNeXt + tokenization
- 视频场景下带时序模块的视觉 backbone

它通常负责：

- patch 级局部模式提取
- 空间位置信息建模
- 形成较高层语义特征

所以你可以把它理解成：

> 负责把像素空间，压缩成语义空间。

### 3. Projector 为什么关键？

很多人会把 projector 说得太轻，但它其实是跨模态桥接的核心。

因为它至少要解决三件事：

- 维度对齐：`vision_width -> d_model`
- 分布对齐：视觉 backbone 的统计分布和 LLM 的 token embedding 分布通常不同
- 语义对齐：让视觉 token 更像“可被语言模型消费的上下文”

常见 projector 设计大致可以分成三类：

#### 1. Linear / MLP projector

优点是：

- 简单
- 便宜
- 容易训练

适合先搭一条最小链路。

#### 2. Resampler / Perceiver-style 模块

优点是：

- 能把大量视觉 token 压缩到更少 token
- 更适合高分辨率输入

#### 3. Q-Former / Cross-Attention bridge

优点是：

- 能主动从视觉特征里抽取对语言任务更有用的子集
- 比单纯线性映射更强

代价是结构更重、训练更复杂。

### 4. 视觉 token 和文本 token 是怎么拼的？

最直观的方式就是 prefix 拼接：

```text
[v_1, v_2, ..., v_m, t_1, t_2, ..., t_n]
```

这里：

- $v_i$：视觉 token
- $t_i$：文本 token

一个常见注意力约束是：

- 视觉 token 之间可以彼此看见
- 文本 token 可以看见全部视觉 token
- 文本 token 之间保持 causal mask

这样做的直觉是：

> 视觉内容是已知条件，文本回答要在这些条件基础上自回归生成。

### 5. 最小代码里每一步在干什么？

完整代码见 [minimal_vlm_bridge.py](minimal_vlm_bridge.py)。

```python
class TinyVLM(nn.Module):
    def __init__(self, vocab_size, vision_width=64, d_model=128):
        super().__init__()
        self.vision_encoder = PatchEncoder(width=vision_width)
        self.projector = MLPProjector(in_dim=vision_width, out_dim=d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)
        self.decoder = nn.TransformerEncoder(...)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
```

这里：

- `vision_encoder`：把图像编码成视觉特征序列
- `projector`：把视觉特征拉到 LLM 隐藏维
- `token_embedding`：把文本 id 变成文本 token
- `pos_embedding`：给联合序列补位置信息
- `decoder`：对视觉和文本做统一建模
- `lm_head`：只对文本位置输出词表 logits

继续看 `forward`：

```python
def forward(self, images, input_ids):
    visual_tokens = self.encode_image(images)
    text_tokens = self.token_embedding(input_ids)
    x = torch.cat([visual_tokens, text_tokens], dim=1)

    positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    x = x + self.pos_embedding(positions)

    mask = build_prefix_causal_mask(
        num_visual_tokens=visual_tokens.shape[1],
        num_text_tokens=text_tokens.shape[1],
        device=x.device,
    )
    x = self.decoder(x, mask=mask)
    text_hidden = x[:, visual_tokens.shape[1] :]
    return self.lm_head(text_hidden)
```

最关键的几行是：

- `torch.cat(...)`：说明视觉 token 和文本 token 已经进入同一条主干
- `build_prefix_causal_mask(...)`：说明视觉是条件，文本是生成目标
- `x[:, visual_tokens.shape[1] :]`：说明最终只保留文本位置做语言建模

### 6. 这条范式有哪些典型 trade-off？

#### 1. 简单 projector vs 强 projector

- 简单 projector：快、便宜、好训，但能力上限可能更早撞墙
- 强 projector：表达力更强，但训练和部署都更重

#### 2. 冻结 backbone vs 全量微调

- 冻结更多：稳定、省显存
- 微调更多：潜力更大，但更容易过拟合和训练不稳

#### 3. 多视觉 token vs 压缩 token

- token 多：信息更充分
- token 少：上下文更省、推理更快

这几个 trade-off 基本就是大多数 VLM 工程决策的主轴。

## 面试高频问题

### 1. 为什么 projector 不是“可有可无”的一层线性映射？

因为它本质上承担了跨模态对齐的职责，不只是改个维度。

### 2. prefix 拼接和 cross-attention 拼接有什么区别？

prefix 拼接更直接，把视觉 token 当作统一序列前缀；cross-attention 会把视觉作为独立 memory，由语言侧主动读取。

### 3. 为什么很多开源 VLM 先冻结 vision encoder 和 LLM，只训 projector？

因为这样成本更低，也更容易先把视觉语言桥接关系训稳。

### 4. 为什么高分辨率常常要配 resampler？

因为视觉 patch token 太多，会严重挤占上下文窗口和显存。

## 最小实现

完整实现见 [minimal_vlm_bridge.py](minimal_vlm_bridge.py)。

这个实现不是为了追求效果，而是为了把 VLM 的主干讲明白：

- `PatchEncoder`：最小视觉编码器
- `MLPProjector`：最小桥接层
- `build_prefix_causal_mask`：最小跨模态注意力约束
- `TinyVLM`：最小拼接范式主模型

## 工程关注点

- 不同分辨率下视觉 token 数怎么控
- 是否需要多图分隔符、区域 token、时间 token
- projector 的容量是否足够
- 视觉 encoder 和 LLM 的微调比例
- 训练样本里 OCR、grounding、doc、chart、video 的覆盖是否均衡

## 常见坑点

- 把 projector 说成“只是一层线性层”
- 只会说拼接，不会说注意力 mask 约束
- 不了解视觉 token 太多会挤占文本上下文
- 混淆 prefix-style VLM 和 cross-attention-style VLM

## 面试时怎么讲

比较稳的讲法是：

> 很多 VLM 的主干都能抽象成 Vision Encoder + Projector + LLM。前两者负责把图像压成能被语言模型消费的视觉 token，后者负责把这些视觉条件和文本问题统一到一条自回归序列里做 next-token prediction。工程上最核心的 trade-off 就是视觉 token 数、projector 复杂度以及 backbone 冻结比例。
