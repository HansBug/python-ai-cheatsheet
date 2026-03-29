# CLIP 面试攻略

## 这是什么？

CLIP 是视觉表征学习里非常高频的一篇，因为它把图像分类、检索和开放词表识别统一到了“图文对齐”这条线上。

如果只用一句话概括：

> CLIP 的核心做法，是把图像编码器和文本编码器同时训练，让正确图文对在同一个语义空间里更接近，错误配对更远离。

它的重要性在于：

- 把监督信号从固定类别标签扩展成自然语言
- 打通了图文检索、zero-shot 分类、开放词表识别
- 成了很多 VLM 和多模态系统的视觉表征底座

## 核心机制

### 1. CLIP 为什么和传统分类不一样？

传统分类通常是：

```text
image -> image encoder -> classifier head -> fixed label set
```

CLIP 变成了：

```text
image -> image encoder -> image embedding
text -> text encoder -> text embedding
similarity(image, text) -> match score
```

所以它不再把类别写死在一个线性分类头里，而是：

> 让“类别描述文本”本身成为可比较对象。

### 2. CLIP 的双塔结构在做什么？

最常见的结构是：

- 一个 image encoder
- 一个 text encoder
- 一个共享语义空间

训练时会得到：

$$ z_i = \frac{f_{\mathrm{img}}(x_i)}{\|f_{\mathrm{img}}(x_i)\|}, \quad z_t = \frac{f_{\mathrm{text}}(t_i)}{\|f_{\mathrm{text}}(t_i)\|} $$

然后计算相似度矩阵：

$$ s_{ij} = \tau \cdot z_i^\top z_{t_j} $$

这里：

- $z_i$：图像 embedding
- $z_t$：文本 embedding
- $\tau$：可学习温度系数

### 3. CLIP 的 loss 到底怎么写？

CLIP 常见写法是双向对比学习：

$$ L = \frac{1}{2} \left[\mathrm{CE}(S, y) + \mathrm{CE}(S^\top, y)\right] $$

其中：

- `S` 是图像到文本的相似度矩阵
- `S^T` 是文本到图像的相似度矩阵
- `y` 是对角线上的正确匹配标签

直觉上：

- 每张图要在一批文本里找对自己的描述
- 每段文本也要在一批图像里找对自己的图片

这也是为什么 CLIP 通常很依赖大 batch 或大量负样本。

### 4. CLIP 为什么能做 zero-shot 分类？

因为类别名可以写成 prompt。

例如猫狗分类时，不一定要训练一个 `2-way classifier`，而是可以直接写成：

```text
"a photo of a cat"
"a photo of a dog"
```

然后：

- 把待分类图像编码成图像 embedding
- 把这些类别 prompt 编码成文本 embedding
- 比谁更相似

所以 CLIP 的 zero-shot 能力来自：

> 它学的不是固定分类头，而是图像和自然语言之间的语义对齐。

### 5. 最小代码里每一段在做什么？

完整代码见 [minimal.py](minimal.py)。

```python
class TinyCLIP(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.image_encoder = SimpleImageEncoder(embed_dim=embed_dim)
        self.text_encoder = SimpleTextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_image(self, images):
        image_features = self.image_encoder(images)
        return F.normalize(image_features, dim=-1)

    def encode_text(self, token_ids):
        text_features = self.text_encoder(token_ids)
        return F.normalize(text_features, dim=-1)

    def forward(self, images, token_ids):
        image_features = self.encode_image(images)
        text_features = self.encode_text(token_ids)
        return self.logit_scale.exp() * image_features @ text_features.T
```

这里每一步都很典型：

- `image_encoder`：提图像特征
- `text_encoder`：提文本特征
- `F.normalize(...)`：做单位向量归一化，方便直接比较余弦相似度
- `self.logit_scale.exp()`：可学习温度
- `image_features @ text_features.T`：得到整批图文相似度矩阵

## 面试高频问题

### 1. CLIP 和普通图像分类最大的差异是什么？

CLIP 不再依赖固定类别头，而是把图像和文本映射到同一语义空间里做相似度匹配。

### 2. 为什么 CLIP 能做 zero-shot 分类？

因为类别可以写成自然语言 prompt，分类本质变成图像和候选文本描述之间的相似度比较。

### 3. 为什么 CLIP 训练时需要大量负样本？

因为它的对比学习目标依赖正确匹配和错误匹配之间的拉开，大 batch 能自然提供更多难负样本。

### 4. 为什么要做 embedding 归一化？

这样相似度更接近余弦相似度，训练更稳定，也更方便统一尺度。

## 最小实现

完整代码见 [minimal.py](minimal.py)。

这个实现重点保留了：

- 双塔编码器
- embedding 归一化
- 相似度矩阵
- 图像到文本、文本到图像的双向交叉熵

## 工程关注点

- batch size 和负样本规模
- 文本 prompt 模板设计
- 不同语言和风格描述的覆盖
- 图像和文本 encoder 的容量平衡
- 温度参数是否稳定

## 常见坑点

- 把 CLIP 理解成“带文本标签的分类器”
- 忘了 CLIP 是双向 loss，不是单向 image-to-text
- 只会说 zero-shot，不会说为什么能 zero-shot
- 忽视 prompt 模板对效果的影响

## 面试时怎么讲

一个比较稳的讲法是：

> CLIP 的核心是图文对比学习。它用图像编码器和文本编码器分别提特征，把正确图文对拉近、错误图文对拉远。因为类别名称本身可以写成自然语言 prompt，所以分类任务可以改写成图像和候选文本描述的匹配问题，这就是它 zero-shot 的来源。
