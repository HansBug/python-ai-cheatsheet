# Vision Encoder + Projector + LLM 的基础拼接范式

## 这是什么？

这是 VLM 机制部分的第二篇，专门把最常见的一条主干拆细：

```text
Vision Encoder -> Projector -> LLM
```

这篇的重点不是再泛泛说“VLM 是什么”，而是直接回答：

- image encoder 具体常见哪些路线
- projector 到底一般做成什么结构
- 为什么 projector 不只是“顺手接一层线性层”
- 一套最小代码里，`__init__` 和 `forward` 分别在把哪条链路接起来

## 核心机制

### 1. 为什么这条范式这么常见？

因为它最符合工程现实：

- 视觉领域已经有成熟 backbone
- 语言领域已经有成熟 LLM
- 中间只需要一层桥接，就能把两边接起来

所以它特别适合：

- 复用预训练模型
- 分阶段训练
- 按预算冻结不同模块

### 2. image encoder 常见有哪些？

这里最容易混的是命名。

很多人口头会说：

- CLIP encoder
- SigLIP encoder
- EVA encoder
- InternViT encoder

更准确的理解是：

> 真正作为 image encoder 工作的，通常是一个视觉 backbone，多数是 ViT；而 CLIP、SigLIP、EVA-CLIP 这些名字，很多时候说的是“这个视觉 backbone 是按什么预训练路线训出来的”。

可以把常见选择记成下面几类：

#### 1. CLIP 风格视觉塔

最经典。

直觉上它的优势是：

- 图文对齐先验成熟
- 语义级表征稳定
- 做图文检索、zero-shot 分类和早期 VLM 桥接都很自然

#### 2. SigLIP 风格视觉塔

主干通常还是 ViT，只是预训练 loss 换成了 sigmoid matching。

直觉上：

- 视觉表示经常更强
- 新一点的 VLM 很喜欢拿它当前端

#### 3. EVA / EVA-CLIP / InternViT 一类大视觉 backbone

这类的重点在于：

- backbone 更大
- 分辨率和细节保持更强
- 更适合 OCR、文档、细粒度感知更重的场景

#### 4. CNN / Swin / ConvNeXt 这类视觉 backbone

也能用，但在通用 VLM 里没有 ViT 路线主流。

原因是：

- VLM 喜欢把图像改写成 token 序列
- ViT 的输出天然更容易和 LLM token 流拼接

所以一句话总结就是：

> 现在通用 VLM 里，最常见的 image encoder 仍然是 ViT 家族，只是预训练路线不同，常见名字包括 CLIP、SigLIP、EVA-CLIP、InternViT 等。

### 3. projector 一般做成什么结构？

projector 常见结构可以按“弱桥接”到“强桥接”来记。

#### 1. Linear projector

最简单：

```text
[B, N, vision_width] -> Linear -> [B, N, d_model]
```

它主要解决：

- 维度不匹配

优点：

- 简单
- 稳
- 便宜

缺点：

- 表达力有限

#### 2. MLP projector

最典型的是：

```text
Linear -> GELU -> Linear
```

它比线性层多了一点非线性，常见于很多 LLaVA-style 最小桥接路线。

它解决的已经不只是改维度，还包括：

- 一点分布对齐
- 一点语义重排

#### 3. Q-Former / Cross-Attention bridge

这类 projector 的关键思想不是“把所有视觉 token 原样送过去”，而是：

> 用少量 query 去视觉特征里主动抽有用信息。

优点：

- 更强
- 更适合从大量视觉 token 里提炼关键信息

代价：

- 更重
- 更难训

#### 4. Resampler / Token Compression 模块

这类结构常用于：

- 高分辨率图像
- 多图输入
- 视频输入

它解决的不只是 `vision_width -> d_model`，还解决：

- token 数太多
- 上下文窗口太挤
- 显存开销太大

所以 projector 的更完整理解应该是：

> 弱版本主要做维度桥接，强版本同时做信息筛选、token 压缩和跨模态对齐。

### 4. 结合代码看，这条主干到底是怎么接起来的？

完整代码见 [minimal_vlm_bridge.py](minimal_vlm_bridge.py)。

先看最小模块：

```python
class PatchEncoder(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, width=64):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            width,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, images):
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class LinearProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, visual_tokens):
        return self.proj(visual_tokens)


class MLPProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, visual_tokens):
        return self.net(visual_tokens)
```

这三段代码对应的直觉非常清楚：

- `PatchEncoder`：把图像切 patch，再变成视觉 token
- `LinearProjector`：最小维度桥接
- `MLPProjector`：稍强一点的非线性桥接

再看主模型：

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

        # 真实系统里这通常对应 CLIP / SigLIP / EVA / InternViT 视觉塔。
        self.vision_encoder = PatchEncoder(
            in_channels=3,
            patch_size=patch_size,
            width=vision_width,
        )

        # 视觉宽度要先桥接到语言模型隐藏维。
        self.projector = build_projector(
            projector_type=projector_type,
            in_dim=vision_width,
            out_dim=d_model,
        )

        # 文本侧还是正常 token embedding。
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
        # 1) 图像先走视觉编码器，变成视觉 token。
        visual_tokens = self.vision_encoder(images)

        # 2) projector 把视觉 token 对齐到语言空间。
        visual_tokens = self.projector(visual_tokens)

        # 3) 文本 token 走正常 embedding。
        text_tokens = self.token_embedding(input_ids)

        # 4) 把视觉 token 当前缀条件拼到文本前面。
        x = torch.cat([visual_tokens, text_tokens], dim=1)

        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        # 5) 文本能读取全部视觉 token，但文本之间仍保持 causal mask。
        mask = build_prefix_causal_mask(
            num_visual_tokens=visual_tokens.shape[1],
            num_text_tokens=text_tokens.shape[1],
            device=x.device,
        )
        x = self.decoder(x, mask=mask)

        # 6) 只保留文本位置输出词表 logits。
        text_hidden = x[:, visual_tokens.shape[1] :]
        text_hidden = self.ln_f(text_hidden)
        return self.lm_head(text_hidden)
```

这一段最值得注意的是：

- `__init__` 里先把桥全部搭好
- `forward` 里再按“视觉编码 -> 桥接 -> 拼接 -> 自回归生成”这个顺序跑完

如果面试官让你对着代码讲，你最稳的节奏就是：

#### 1. 先讲 `self.vision_encoder`

它把 `[B, C, H, W]` 的像素图像，变成 `[B, N_v, vision_width]` 的视觉 token 序列。

#### 2. 再讲 `self.projector`

它把 `[B, N_v, vision_width]` 变成 `[B, N_v, d_model]`，让视觉 token 能和文本 token 进入同一条主干。

#### 3. 再讲 `self.token_embedding`

文本 token id 还是按标准语言模型方式变 embedding。

#### 4. 再讲 `torch.cat([visual_tokens, text_tokens], dim=1)`

这一步就是 prefix-style VLM 的核心：

> 把视觉 token 当成文本前缀条件。

#### 5. 最后讲 `build_prefix_causal_mask(...)`

这一步决定注意力规则：

- 视觉 token 互相可见
- 文本 token 可以看全部视觉 token
- 文本 token 之间仍然只能看左边

这就是为什么它最后还能保持 next-token prediction。

### 5. 这条范式的典型 trade-off 是什么？

#### 1. 弱 projector vs 强 projector

- 弱 projector：便宜、稳、快
- 强 projector：能力上限更高，但更重更复杂

#### 2. 少 token vs 多 token

- token 多：信息更多
- token 少：上下文更省、推理更快

#### 3. 冻结 backbone vs 深度微调

- 冻结更多：训练稳、省显存
- 微调更多：潜力更大，但训练更难

所以大多数 VLM 工程问题，最后都会收敛到这三组 trade-off。

## 面试高频问题

### 1. 为什么现在通用 VLM 里大多还是 ViT 家族 image encoder？

因为它天然输出 token 序列，最容易和 LLM 的 token 流拼接。

### 2. projector 为什么不能简单理解成“改个维度”？

因为强一点的 projector 还会承担跨模态对齐、信息筛选和 token 压缩的职责。

### 3. 什么情况下 linear projector 可能不够？

当视觉信息更复杂、高分辨率更高、任务需要更强跨模态桥接时，纯线性层的表达力可能不够。

### 4. 为什么高分辨率常常逼着你上 resampler？

因为真正先撞墙的，通常不是维度，而是 token 数和显存预算。

## 工程关注点

- image encoder 的预训练路线是否够强
- projector 容量是否够、是否过重
- 视觉 token 数是否挤占文本上下文
- 是否需要多图分隔符、区域 token、时间 token
- OCR、grounding、doc、chart、video 这些任务的样本覆盖是否均衡

## 常见坑点

- 把 CLIP / SigLIP 当成具体 backbone 名字，而不是视觉预训练路线
- 把 projector 讲成纯维度映射
- 只会讲 `forward` 的拼接，不会讲 `__init__` 到底接了哪些桥
- 不了解高分辨率场景下 token 压缩的重要性

## 面试时怎么讲

比较稳的讲法是：

> 现在最常见的 VLM 主干就是 Vision Encoder + Projector + LLM。前面的 image encoder 大多是 ViT 家族，只是预训练路线可能是 CLIP、SigLIP、EVA-CLIP 或 InternViT。中间的 projector 弱一点是 linear/MLP，强一点会做 cross-attention 或 token resampling。最后把视觉 token 当成前缀条件拼到文本前面，让 LLM 继续按自回归方式生成回答。
