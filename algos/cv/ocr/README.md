# 文本检测与文本识别基本链路 面试攻略

## 专题顺序

- 第一篇：[文本检测与文本识别基本链路](README.md)
- 第二篇：CTC 与 Attention-based OCR（待补）
- 第三篇：版面分析 / 表格理解 / 图表理解（待补）

## 这是什么？

这是 OCR 部分的第一篇，先把最基本的主链路讲清楚：

- OCR 到底在做哪几件事
- 文本检测和文本识别为什么通常分开
- 从图像到文字的标准数据流长什么样
- 面试里该怎么讲 detection、crop、recognition、decode 之间的关系

如果只用一句话概括：

> OCR 最经典的路线，是先找到文本在哪，再把每段文本区域裁出来做识别，最后再做解码和后处理。

## 核心机制

### 1. OCR 为什么通常不是“一步到位”？

因为文档图像、街景图像、票据图像里，文本通常同时面临两个问题：

- 文本在哪
- 文本是什么

所以 OCR 常被拆成：

```text
image -> text detection -> text boxes
text boxes -> crop / rectify -> text recognition -> character sequence
```

这样拆的好处是：

- 检测模型专心找位置
- 识别模型专心读内容
- 对弯曲文本、旋转文本、多行文本更容易分别处理

### 2. 文本检测在做什么？

文本检测的目标是输出文本区域。

常见输出形式有：

- 水平框
- 旋转框
- 四边形
- 多边形

直觉上可以把它理解成：

> 在整张图上找出“哪些区域大概率是文字”。

如果场景是自然图像 OCR，检测器往往更重要，因为：

- 背景杂乱
- 文字方向不固定
- 文本尺度差异大

### 3. 为什么检测后还要 crop 或 rectification？

因为识别模型通常希望输入更规整。

例如：

- 高度固定
- 文本尽量横向展开
- 干扰背景尽量少

所以在检测和识别之间常会有一步：

- 裁剪文本框
- 透视变换或旋正
- resize 到固定高度

这一步的意义是：

> 把“复杂场景下的文字区域”变成“识别器容易读的标准条带”。

### 4. 文本识别在做什么？

文本识别的目标，是把文本图像变成字符序列。

一个很常见的思路是：

- 先用 CNN 提局部视觉特征
- 再把高度方向压缩掉
- 把宽度方向当成时间轴
- 对每个时间步输出字符分布

也就是把文本识别改写成：

> 一维序列建模问题。

所以 OCR 识别器里，经常能看到：

- CNN
- RNN / Transformer
- CTC 或 attention decoder

### 5. OCR 的后处理在做什么？

后处理主要包括：

- 置信度过滤
- 重叠框合并
- 解码去重
- 语言词典修正
- 按阅读顺序排序

如果是文档 OCR，还要关心：

- 段落结构
- 表格单元格
- 图文混排

所以 OCR 不只是“模型把字读出来”，还包括：

> 把局部识别结果整理成最终可用文本。

### 6. 最小代码里每一步在干什么？

完整代码见 [minimal.py](minimal.py)。

```python
class TinyTextDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, images):
        return torch.sigmoid(self.backbone(images))
```

这里：

- `Conv2d(..., 1, kernel_size=1)`：输出一张文本概率图
- `sigmoid(...)`：把每个位置变成“像不像文本”的分数

再看识别器：

```python
class TinyTextRecognizer(nn.Module):
    def __init__(self, num_chars):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, num_chars)

    def forward(self, text_crops):
        x = self.features(text_crops)
        x = x.mean(dim=2)
        x = x.transpose(1, 2)
        return self.classifier(x)
```

这几行分别对应：

- `self.features`：先从文本条带里提局部视觉特征
- `x.mean(dim=2)`：把高度压掉，保留宽度方向序列
- `x.transpose(1, 2)`：把特征改成按时间步排列
- `self.classifier(x)`：每个时间步预测一个字符分布

## 面试高频问题

### 1. OCR 为什么常拆成 detection 和 recognition 两阶段？

因为先找位置、再读内容更稳定，也更容易适配复杂背景、旋转文本和多尺度文本。

### 2. 文本识别为什么常常能转成序列建模问题？

因为文本通常沿宽度方向展开，可以把宽度维看成时间轴，对每个位置输出字符分布。

### 3. detection 输出为什么不一定是水平框？

因为真实场景里文本可能旋转、倾斜、弯曲，水平框会引入大量背景。

### 4. OCR 后处理为什么重要？

因为最终可用结果不只是单个字符预测，还包括去重、排序、纠错和版面组织。

## 最小实现

完整代码见 [minimal.py](minimal.py)。

这个实现重点保留了 OCR 最小主线：

- `TinyTextDetector`：文本概率图
- `score_map_to_boxes`：最简阈值转框
- `crop_boxes`：从原图裁出文本区域
- `TinyTextRecognizer`：把文本条带改成字符序列
- `ctc_greedy_decode`：最简字符解码

## 工程关注点

- 旋转、弯曲、小字和密集排版怎么处理
- 检测框抖动会不会拖垮识别器
- 不同语言和字符集怎么建模
- 文档 OCR 里阅读顺序怎么恢复
- OCR 结果如何和版面、表格、图表模块协同

## 常见坑点

- 把 OCR 理解成“就是一个分类器”
- 只会说识别，不会说检测与裁剪
- 忽视后处理和阅读顺序恢复
- 混淆文本检测框和检测通用目标框的差异

## 面试时怎么讲

一个比较稳的讲法是：

> OCR 最经典的做法是两阶段。先用文本检测器找出文字区域，再把这些区域裁剪和规整后交给识别器，把宽度方向当成序列逐步解码成字符。最终结果还要经过后处理，比如去重、排序和纠错，才能变成真正可用的文本输出。
