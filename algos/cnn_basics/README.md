# CNN 基础模块 面试攻略

## 这是什么？

这是 CV 面试里最基础、也最容易被问到的一组内容。

它不只是在讲 `Conv2d` 这个 API，而是在讲：

- 卷积层到底在做什么
- padding / stride / pooling 各自控制什么
- 感受野怎么形成
- CNN 为什么能长期统治视觉任务
- 在 ResNet 之前，经典 CNN 是怎么一路演进过来的

如果只用一句话来讲：

> CNN 的核心思想，是利用局部连接、权重共享和层级特征提取，把图像从低层边缘纹理逐步编码成高层语义表示。

这篇里会把 `LeNet -> AlexNet -> ZFNet -> VGG -> GoogLeNet` 串成一条历史线；`ResNet` 单独放下一篇讲。

## 核心机制

### 1. CNN 到底解决了什么问题？

图像输入通常是：

$$ X \in \mathbb{R}^{B \times C \times H \times W} $$

它和文本序列不一样，图像最重要的结构是：

- 空间局部相关性很强
- 相邻像素通常比远距离像素更相关
- 有明显的二维网格结构

CNN 的做法就是把这些先验直接写进模型结构里：

- 局部连接：每次只看一个小窗口
- 权重共享：同一个卷积核在整张图上滑动
- 层级提取：浅层学边缘纹理，深层学部件和语义

这就是为什么 CNN 在视觉任务里天然比全连接网络更合适。

### 2. 卷积层在做什么？

最基础的二维卷积可以写成：

$$ Y[b, o, i, j] = \sum_{c, u, v} X_{\mathrm{pad}}[b, c, i \cdot s + u, j \cdot s + v] \cdot W[o, c, u, v] + b[o] $$

这里最重要的几个量是：

- `W[o, c, u, v]`：第 `o` 个卷积核的权重
- `s`：stride，控制卷积核滑动步长
- `X_pad`：padding 后的输入

直觉上看：

> 一个卷积核就是一个“局部模式探测器”，它会在整张图上反复扫描，寻找某种边缘、纹理或组合结构。

### 3. padding / stride / pooling 分别在控制什么？

这是 CNN 面试里的基础高频题。

#### 1. padding

作用主要有两个：

- 控制输出空间尺寸
- 保护边缘信息，避免卷积后越变越小太快

#### 2. stride

作用是控制下采样速度。

- stride 大，输出特征图更小
- 计算量下降，但信息损失更快

#### 3. pooling

最常见的是 max pooling。

它的作用通常是：

- 下采样
- 增强局部平移鲁棒性
- 降低后续计算量

所以在经典 CNN 里，常见主线就是：

```text
conv -> activation -> pooling
```

### 4. 感受野是怎么变大的？

单层卷积只能看一个局部窗口。

但多层堆叠以后，高层特征能“间接看到”越来越大的输入区域，这就是感受野不断扩大。

一个很重要的直觉是：

- 更深，不只是参数更多
- 更深意味着高层特征能依赖更大范围的上下文

这也是为什么后来 CNN 会越来越深。

### 5. 手写一个最小 `conv2d`

面试里如果让你手写卷积，不一定要求你写得很快，但至少要能把逻辑说对。

下面这个最小实现就把卷积的本质完整展开了：

```python
def naive_conv2d(x, weight, bias=None, stride=1, padding=0):
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h))

    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    out_h = (in_h - kernel_h) // stride_h + 1
    out_w = (in_w - kernel_w) // stride_w + 1

    out = torch.zeros(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)

    for b in range(batch_size):
        for oc in range(out_channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride_h
                    w_start = j * stride_w
                    window = x[b, :, h_start : h_start + kernel_h, w_start : w_start + kernel_w]
                    out[b, oc, i, j] = (window * weight[oc]).sum()
                    if bias is not None:
                        out[b, oc, i, j] += bias[oc]

    return out
```

这里每个部分都很关键：

- `F.pad(...)`：先把 padding 显式补出来
- `window = ...`：从输入里切出当前卷积窗口
- `(window * weight[oc]).sum()`：做逐元素乘法再求和
- `i * stride_h` / `j * stride_w`：控制卷积核滑动位置

所以从机制上讲，卷积本质就是：

> 取局部窗口，和卷积核做点积，然后把这个操作在整张图上重复。

### 6. 经典 CNN 是怎么一路演进到 ResNet 之前的？

这条历史线建议你按“每一代到底改了什么”来讲，不要只背模型名。

#### 1. LeNet

它奠定了最经典的 CNN 原型：

- 卷积
- 激活
- 池化
- 全连接分类头

一句话概括：

> LeNet 建立了“局部卷积 + 下采样 + 分类头”的基础范式。

如果对着代码看，一个最小的 LeNet 风格网络通常就是：

```python
class LeNetStyleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

这段代码最值得结合结构一起讲：

- `self.features`：前半部分是卷积特征提取器，体现的是 `conv -> activation -> pool` 的经典范式
- `nn.Conv2d(1, 6, ...)` / `nn.Conv2d(6, 16, ...)`：浅层先提简单局部模式，再逐步提升通道数
- `AvgPool2d`：做下采样，减小特征图尺寸
- `self.classifier`：最后把卷积特征拉平后接分类头
- `forward`：先做局部特征提取，再做分类，这就是最早 CNN 的标准数据流

所以 LeNet 最重要的不是某个单独算子，而是它把“卷积特征提取 + 分类头”这条主线固定下来了。

#### 2. AlexNet

它的重要性不只是赢了 ImageNet，而是把深度 CNN 真正带到了大规模视觉任务里。

它相对前代最关键的变化是：

- 网络更深更宽
- 大量使用 ReLU
- 使用 dropout
- 使用数据增强
- 借助 GPU 训练大模型

一句话概括：

> AlexNet 证明了深度卷积网络在大规模视觉分类上能显著胜出。

如果只摘最有代表性的 early stem，看起来会像这样：

```python
class AlexNetStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        return self.net(x)
```

对着代码解释时，重点要落在这几件事上：

- `kernel_size=11, stride=4`：第一层卷积核很大、步长也大，说明它面向的是更大分辨率、更大规模的图像分类场景
- `ReLU()`：AlexNet 把 ReLU 大规模用起来，这是它训练更深网络的重要工程点之一
- `MaxPool2d(...)`：进一步快速下采样，压缩后续计算量
- `forward` 非常短：说明 AlexNet 的核心不是复杂控制流，而是“更大、更深、更能训”的 plain CNN

所以 AlexNet 相比 LeNet，最关键的不是范式变化，而是把 CNN 拉到了真正的大规模视觉任务里。

#### 3. ZFNet

它可以看成是对 AlexNet 的结构修正和可视化分析加强版。

改动重点主要是：

- 第一层卷积核更小
- stride 更保守
- 更重视中间层特征可视化和结构调参

一句话概括：

> ZFNet 让大家意识到，早期大卷积核和大步长可能过于粗暴，结构细节会明显影响表示能力。

如果把这种改动写成最短代码，对比 AlexNet 会更直观：

```python
class ZFNetStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        return self.net(x)
```

这里最值得你对着代码点出来的是：

- `kernel_size` 从更大的值收缩到 `7`
- `stride` 也更保守，不再一开始就过度下采样
- `forward` 形式和 AlexNet 很像，但结构细节已经明显更克制

所以 ZFNet 想表达的核心不是“提出了全新范式”，而是：

> 同样是 plain CNN，早期层的卷积核大小和步长设定会显著影响信息保留和表示能力。

#### 4. VGG

VGG 的思想非常干净：

> 用很多个小 `3x3` 卷积去替代更大的卷积核，并把网络堆深。

它的关键变化是：

- 统一使用小卷积核
- 结构非常规整
- 网络显著加深

为什么这很重要？

- 多个 `3x3` 叠起来，感受野可以接近更大卷积
- 中间多了更多非线性层
- 结构更容易统一设计

一句话概括：

> VGG 把“更深、更规整、小卷积核堆叠”这条路走得很彻底。

把它翻成最简代码，就是一个重复堆叠的小卷积块：

```python
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        for index in range(num_convs):
            layers.append(
                nn.Conv2d(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

这段代码本身就把 VGG 的设计语言暴露得很明显：

- `kernel_size=3, padding=1`：统一使用 `3x3` 小卷积核，保持空间尺寸
- `for index in range(num_convs)`：同一类卷积重复堆叠，这是 VGG 最典型的结构特征
- `MaxPool2d(...)`：每个 stage 末尾再统一做一次下采样
- `forward` 只有一句 `return self.net(x)`：说明它的核心卖点就是规整、统一、容易堆深

所以 VGG 这代最重要的是：

> 它把 CNN 结构设计从“局部技巧组合”推进成了“规则化堆叠”的工程范式。

#### 5. GoogLeNet / Inception

GoogLeNet 的重点不再只是“继续堆深”，而是开始认真思考计算效率。

它最关键的改动是：

- 多分支并行卷积
- 同时看不同尺度的感受野
- 用 `1x1` 卷积做降维和瓶颈

一句话概括：

> Inception 系列在保证表达能力的同时，更强调多尺度特征提取和计算效率。

如果只摘最核心的 Inception 模块，代码会像这样：

```python
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 4
        self.branch1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=5, padding=2),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
        )

    def forward(self, x):
        outputs = [
            self.branch1(x),
            self.branch3(x),
            self.branch5(x),
            self.branch_pool(x),
        ]
        return torch.cat(outputs, dim=1)
```

这段代码要对着分支来讲：

- `branch1`：`1x1` 分支，成本低，负责轻量线性投影
- `branch3`：先 `1x1` 再 `3x3`，体现 bottleneck + 中尺度卷积
- `branch5`：看更大感受野
- `branch_pool`：池化分支保留另一种局部统计视角
- `torch.cat(outputs, dim=1)`：多分支不是相加，而是在 channel 维拼起来

所以 GoogLeNet / Inception 相比 VGG 的关键变化是：

> 不再只靠“把同一种卷积重复堆深”，而是开始并行地看多种尺度，并认真优化计算成本。

#### 6. 到 ResNet 前，问题出在哪？

到了 VGG / GoogLeNet 这代，大家已经非常清楚一件事：

- 更深通常更强

但 plain CNN 继续往更深堆时，训练会越来越困难，出现明显的优化和退化问题。

这就是 ResNet 要解决的直接背景。

## 面试高频问题

### 1. CNN 为什么适合图像？

因为图像有明显的局部结构和空间相关性，CNN 通过局部连接和权重共享把这些先验直接编码进网络。

### 2. 卷积和全连接的本质区别是什么？

卷积保留空间结构并共享参数；全连接不利用局部性，参数量也通常更大。

### 3. 为什么很多经典 CNN 都要池化？

为了下采样、扩大有效感受野、降低计算量，并增强局部平移鲁棒性。

### 4. 为什么 VGG 喜欢堆很多 `3x3`？

因为它既能形成更大感受野，又能增加非线性层数，而且结构更统一。

### 5. GoogLeNet 的核心创新是什么？

多分支 Inception 模块和 `1x1` bottleneck，让模型能更高效地做多尺度特征提取。

### 6. AlexNet 为什么是里程碑？

因为它把深度 CNN 在 ImageNet 这类大规模视觉任务上的优势真正打出来了。

### 7. CNN 在 ResNet 之前的主线矛盾是什么？

一边想继续加深网络，一边又发现 plain CNN 越深越难训。

## 最小实现

这个专题的最小实现分两类：

- 一类是手写 `conv2d`
- 一类是用 `torch.nn` 写出几种经典 CNN 风格模块

### 1. 手写 `conv2d`

上面已经给了 `naive_conv2d` 的完整代码，这一段最适合用来讲卷积本质。

在配套代码里，我还顺手把它和 `torch.nn.functional.conv2d` 做了数值对照。

### 2. 经典 CNN 代码锚点

上面的历史讲解里，已经把这些简短代码都嵌回去了：

- `LeNetStyleCNN`
- `AlexNetStem`
- `ZFNetStem`
- `VGGBlock`
- `InceptionBlock`

建议阅读顺序就是：

1. 先看历史段落里每一代的短代码
2. 再对着解释理解“这一代到底改了哪一处结构”
3. 最后回到 [minimal.py](minimal.py) 看这些类怎么放在同一个脚本里跑通

完整代码见：[minimal.py](minimal.py)

## 工程关注点

### 1. 经典 CNN 很依赖输入分辨率和 stage 设计

第一层卷积核大小、stride、pooling 放在哪里，都会明显影响后面特征图尺寸和算力开销。

### 2. 小卷积核堆叠不只是“参数省”

它还意味着：

- 更深的非线性
- 更规整的网络设计
- 更容易统一构建 backbone

### 3. 纯 CNN 的归纳偏置很强

这通常是优点，但也意味着它的结构自由度没 Transformer 那么大。

## 常见坑点

### 1. 把卷积理解成“局部全连接”就停了

这还不够。真正关键的是滑动窗口和权重共享。

### 2. 只会背模型名，不会说每一代到底改了什么

面试里更好的答法是顺着历史线讲“上一代的痛点”和“下一代的改动”。

### 3. 只会用 `nn.Conv2d`，不会解释卷积本质

至少要能讲清楚窗口提取、逐元素乘法求和、stride 和 padding 的作用。

### 4. 把 VGG 和 GoogLeNet 的设计目标混为一谈

VGG 更强调规整加深；GoogLeNet 更强调多尺度和效率。

## 面试时怎么讲

如果面试官让你介绍 CNN 基础，可以按这个顺序讲：

1. 图像有局部相关性和二维空间结构，所以 CNN 用局部连接和权重共享来建模
2. 卷积核本质是在整张图上滑动的局部模式探测器
3. padding 控制边界和尺寸，stride / pooling 控制下采样和计算量
4. 多层卷积堆叠后，感受野不断扩大，特征从边缘纹理走向高层语义
5. 历史上 LeNet 建立范式，AlexNet 把深度 CNN 在 ImageNet 上打出来，VGG 把小卷积核堆深走到极致，GoogLeNet 开始更重视多尺度和效率
6. 再往后，继续加深 plain CNN 会越来越难训，这就引出了 ResNet

一个简洁版本可以直接讲：

> CNN 的核心是利用图像的局部结构，通过卷积核在空间上滑动提特征，并靠权重共享降低参数量。padding、stride 和 pooling 分别控制边界、下采样和计算量，多层堆叠后感受野会逐渐扩大。历史上 LeNet 建立了卷积范式，AlexNet 把深度 CNN 带到大规模视觉任务里，VGG 用很多 `3x3` 小卷积把网络堆深，GoogLeNet 则进一步强调多尺度和效率，最后才引出 ResNet 去解决更深网络的训练问题。

## 延伸阅读

- 残差网络：[ResNet](../resnet/README.md)
- Vision Transformer：[Vision Transformer](../vision_transformer/README.md)
- 配套代码：[minimal.py](minimal.py)
