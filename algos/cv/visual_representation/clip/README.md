# CLIP 面试攻略

## 这是什么？

CLIP 是视觉表征学习里最重要的一篇之一，因为它把图像分类、图文检索和开放词表识别统一到了“图文共享语义空间”这条线上。

如果只用一句话概括：

> CLIP 的核心做法，是同时训练图像编码器和文本编码器，让正确图文对在共享 embedding 空间里更接近，错误图文对更远离。

它的重要性在于：

- 把监督信号从固定类别标签扩展成自然语言
- 打通了 zero-shot 分类、图文检索、开放词表识别
- 成了很多 VLM 的视觉表征底座

## 核心机制

### 1. CLIP 和传统图像分类到底差在哪？

传统分类通常是：

```text
image -> image encoder -> classifier head -> fixed labels
```

CLIP 则改成：

```text
image -> image encoder -> image embedding
text -> text encoder -> text embedding
similarity(image, text) -> match score
```

也就是说，CLIP 不再把知识写死在一个分类头里，而是让：

> 图像和自然语言描述一起进入一个可比较的语义空间。

### 2. CLIP 的双塔结构在做什么？

最典型的 CLIP 有两个塔：

- image encoder
- text encoder

原始 CLIP 里，图像侧可以是 ResNet，也可以是 ViT；文本侧通常是 Transformer。

训练时我们得到：

$$ z_i = \frac{f_{\mathrm{img}}(x_i)}{\|f_{\mathrm{img}}(x_i)\|}, \quad z_t = \frac{f_{\mathrm{text}}(t_i)}{\|f_{\mathrm{text}}(t_i)\|} $$

然后计算图文相似度矩阵：

$$ S = \tau \cdot Z_{\mathrm{img}} Z_{\mathrm{text}}^\top $$

这里：

- `S[i, j]`：第 `i` 张图和第 `j` 段文本的匹配分数
- $\tau$：可学习温度系数

### 3. CLIP 的 loss 到底怎么写？

这一部分是 CLIP 最该讲清楚的地方。

假设一个 batch 里有 `N` 对图文配对，且第 `i` 张图对应第 `i` 段文本。

那么：

- `S` 的第 `i` 行，表示“图 `i` 对所有文本的打分”
- `S` 的第 `i` 列，表示“文本 `i` 对所有图像的打分”

CLIP 的 image-to-text loss 是：

$$ L_{\mathrm{i2t}} = - \frac{1}{N} \sum_i \log \frac{\exp(S_{ii})}{\sum_j \exp(S_{ij})} $$

这句话翻成人话就是：

> 每张图都要在当前 batch 里，把自己的正确文本排到第一。

text-to-image loss 则是：

$$ L_{\mathrm{t2i}} = - \frac{1}{N} \sum_i \log \frac{\exp(S_{ii})}{\sum_j \exp(S_{ji})} $$

翻成人话就是：

> 每段文本也都要在当前 batch 里，把自己的正确图像排到第一。

最终 loss 是两者平均：

$$ L = \frac{1}{2} (L_{\mathrm{i2t}} + L_{\mathrm{t2i}}) $$

所以 CLIP 的 loss 本质上不是“简单余弦相似度”，而是：

> 在一个 batch 里，同时做图到文、文到图的双向 `N` 分类。

### 4. 为什么 CLIP 的 loss 很依赖 batch size？

因为一个 batch 里的其他样本，天然就构成负样本。

batch 越大，意味着：

- 候选错误文本更多
- 候选错误图像更多
- 排序任务更难，也更有监督价值

所以 CLIP 很自然就更吃：

- 大 batch
- 大规模分布式训练
- 跨设备 gather 负样本

### 5. 结合代码看，CLIP 的 `__init__`、`forward` 和 loss 是怎么连起来的？

完整代码见 [minimal.py](minimal.py)。

先看主模型：

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
        logits_per_image = self.logit_scale.exp() * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        return logits_per_image, logits_per_text
```

你要顺着代码讲：

- `self.image_encoder`：先把图像变成图像 embedding
- `self.text_encoder`：再把文本变成文本 embedding
- `F.normalize(...)`：做单位向量归一化，让点积更像余弦相似度
- `self.logit_scale.exp()`：控制相似度分布的锐度
- `image_features @ text_features.T`：一次性得到整个 batch 的图文相似度矩阵
- `logits_per_text = logits_per_image.T`：把“图找文”和“文找图”两条方向都显式拿出来

再看 loss：

```python
class CLIPLoss(nn.Module):
    def forward(self, logits_per_image, logits_per_text=None):
        if logits_per_text is None:
            logits_per_text = logits_per_image.T

        labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
        image_to_text = F.cross_entropy(logits_per_image, labels)
        text_to_image = F.cross_entropy(logits_per_text, labels)
        return 0.5 * (image_to_text + text_to_image)
```

这段代码里最重要的是三行：

- `labels = torch.arange(...)`：说明第 `i` 张图的正确文本就是第 `i` 个
- `F.cross_entropy(logits_per_image, labels)`：每张图都去 batch 里挑自己的正确文本
- `F.cross_entropy(logits_per_text, labels)`：每段文本也去 batch 里挑自己的正确图像

所以你可以直接把这段代码翻译成一句面试表达：

> CLIP 的 loss 就是对相似度矩阵做双向交叉熵，让图找对文、文也找对图。

### 6. 为什么 CLIP 能做 zero-shot 分类？

因为类别名称本身可以写成自然语言 prompt。

例如猫狗分类时，不一定要训练 `2-way classifier`，而是可以直接写：

```text
"a photo of a cat"
"a photo of a dog"
```

然后：

- 把图像编码成 image embedding
- 把类别 prompt 编码成 text embedding
- 比谁更相似

所以 CLIP 的 zero-shot 能力来自：

> 它学到的不是固定分类头，而是图像和自然语言之间的语义对齐。

### 7. 为什么说 CLIP 更自然适合 zero-shot single-label classification？

这一点和它的训练目标直接相关。

CLIP 训练时，本质上一直在学：

> 给定一张图，在一组候选文本里把唯一正确项排到最前面。

所以到了 zero-shot 分类时，如果我们把类别写成一组 prompt：

```text
"a photo of a cat"
"a photo of a dog"
"a photo of a car"
```

然后算图像和这些 prompt 的相似度，最自然的推理方式就是：

- 对这组候选 prompt 的分数做 softmax
- 取概率最大的那个类别

这和普通单标签分类的接口几乎是同构的：

- 仍然是一组互斥候选类
- 仍然默认“最终只选一个最可能类别”

所以 CLIP 会让人觉得特别适合 zero-shot classification，尤其是：

- 单标签分类
- 封闭类别集合上的开放词表替换
- top-1 / top-k 预测

但这里要注意，CLIP 不是完全不能做 multi-label，而是：

> 它的训练目标天然更像“候选类之间彼此竞争”，所以做 zero-shot multi-label 时没有那么顺手。

### 8. CLIP 适合做什么，不太适合做什么？

#### 适合

- zero-shot single-label 分类
- 图文检索
- 开放词表识别
- 作为 VLM 的视觉底座
- 召回、聚类、近邻检索

#### 不太适合

只靠 CLIP 这种整图级 embedding，一般不够解决：

- OCR-heavy 任务
- 精确 grounding
- 目标检测 / 分割
- 细粒度计数
- 强空间关系推理

因为它学到的重点是：

> 全局语义对齐，而不是区域级或像素级监督。

## 面试高频问题

### 1. CLIP 的 loss 为什么要做双向，而不是只做 image-to-text？

因为只约束一个方向，会让另一个塔受到的监督不够完整；双向约束更对称，也更稳定。

### 2. 为什么 CLIP 很吃 batch size？

因为 batch 内其他样本就是负样本，batch 越大，对比学习监督通常越强。

### 3. 为什么 CLIP 能做 zero-shot 分类？

因为类别可以写成自然语言 prompt，分类可以改写成图像和候选文本描述的匹配。

### 4. 为什么说 CLIP 更自然适合 zero-shot single-label classification？

因为它的训练目标本质上是在一组候选文本里选唯一正确项，这和单标签分类的互斥假设非常一致。

### 5. CLIP 能不能做 multi-label？

能做，但通常不如 SigLIP 那样自然。因为 CLIP 的分数更像“候选类之间的相对竞争结果”，不是天然独立的标签存在概率。

### 6. 为什么 `logit_scale` 很重要？

它控制 softmax 前相似度的温度，直接影响分布有多尖锐、训练是否稳定。

## 最小实现

完整代码见 [minimal.py](minimal.py)。

这个实现重点保留了：

- 双塔编码器
- embedding 归一化
- 双向相似度矩阵
- `nn.Module` 版 `CLIPLoss`

## 工程关注点

- batch size 和负样本规模
- 跨设备 gather 负样本的开销
- prompt 模板设计
- 图像和文本 encoder 的容量平衡
- 温度参数是否稳定

## 常见坑点

- 把 CLIP 理解成“带文本标签的分类器”
- 只会背公式，不会解释行和列分别代表什么
- 忘了 CLIP 是双向 loss，不是单向 image-to-text
- 只会说 zero-shot，不会说为什么类别 prompt 能替代分类头

## 面试时怎么讲

一个比较稳的讲法是：

> CLIP 的核心是图文双塔对比学习。它把图像和文本都映射到同一个 embedding 空间里，形成一个 `N x N` 的相似度矩阵。然后对矩阵做双向交叉熵，让每张图在 batch 里找到自己的正确文本，每段文本也找到自己的正确图像。因为类别名称本身可以写成自然语言 prompt，所以分类任务就被改写成了图像和文本描述的匹配问题。
