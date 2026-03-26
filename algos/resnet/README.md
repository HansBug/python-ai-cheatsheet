# ResNet 面试攻略

## 这是什么？

这是经典 CNN 演进到深层网络阶段后，最重要的一次结构升级。

如果只用一句话来讲：

> ResNet 的核心思想，是让网络直接学习残差函数 `F(x)`，然后通过跳连把输出写成 `F(x) + x`，从而显著缓解深层 plain CNN 的优化困难。

它的重要性在于：

- 它把“更深的 CNN”真正做成可训练
- 它让残差块成了后续大量视觉 backbone 的默认基础件
- 后面的 DenseNet、ConvNeXt，甚至很多 Transformer 结构设计，都还能看到残差思想的影子

## 核心机制

### 1. ResNet 想解决什么问题？

在 ResNet 之前，大家已经知道一件事：

- 更深的网络通常更有潜力

但 plain CNN 一味加深，会出现很明显的训练困难。

这里最容易讲错的一点是：

> ResNet 解决的不只是梯度消失问题，更关键的是深层 plain network 的退化和优化困难。

也就是：

- 理论上更深不该比更浅更差
- 但实际训练时，更深的 plain network 往往反而效果更差

这说明问题不只是表达能力，而是优化路径本身出了问题。

### 2. 残差块到底在做什么？

ResNet 把原来“直接学目标映射”改成了“学残差映射”。

如果普通块想学的是：

$$ H(x) $$

那么残差块改成学：

$$ F(x) = H(x) - x $$

最后输出写成：

$$ y = F(x) + x $$

这里的 `+ x` 就是 identity shortcut，也就是跳连。

直觉上可以理解成：

> 如果这一层学不到什么有用的新东西，最差也可以先保留输入不变，而不是被迫把所有信息都重写一遍。

### 3. 为什么 `F(x) + x` 会更容易训练？

这是面试里最核心的一问。

可以从两个角度讲：

#### 1. 前向角度

如果当前 block 最优解接近恒等映射，那么：

- plain block 要硬学出一个接近 identity 的复杂映射
- residual block 只需要让 `F(x)` 接近 0

后者通常更容易。

#### 2. 反向角度

因为有 shortcut，梯度除了走卷积分支，还能沿 identity 分支直接往前传。

所以更深网络里，优化路径会更通畅。

这也是为什么残差连接会成为几乎所有深层网络里的默认设计。

### 4. BasicBlock 长什么样？

最经典的 ResNet block 可以直接看代码：

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out
```

这里每一部分的作用都要能讲清：

- `conv1`：第一层卷积，负责提取更强特征，必要时顺便做下采样
- `bn1`：稳定训练，控制特征尺度
- `relu`：提供非线性
- `conv2 + bn2`：再做一层变换，形成残差分支输出 `F(x)`
- `identity = x`：保留原始输入作为 shortcut
- `downsample`：当通道数或空间尺寸对不上时，把 identity 分支也映射到可相加的形状
- `out = out + identity`：做残差相加

最关键的一句就是：

> ResNet block 不是把 shortcut 当额外装饰，而是把它作为主结构的一部分来设计。

### 5. 为什么有时候 shortcut 不能直接等于 `x`？

只有在下面两件事都不变时，identity 才能直接原样相加：

- 通道数不变
- 空间尺寸不变

一旦 block 里发生了：

- stride = 2 下采样
- 通道从 `C` 变到 `2C`

那 shortcut 分支就必须做投影，也就是：

```python
identity = self.downsample(x)
```

最常见做法就是 `1x1 conv + BN`。

### 6. BasicBlock 和 Bottleneck 有什么区别？

这是 ResNet 高频追问。

#### BasicBlock

通常用于：

- ResNet-18
- ResNet-34

特点是：

- 两个 `3x3` 卷积
- 结构简单

#### Bottleneck

通常用于：

- ResNet-50
- ResNet-101
- ResNet-152

特点是：

- `1x1 -> 3x3 -> 1x1`
- 先降维 / 调整通道，再做主计算，再升维
- 更适合更深的网络

一句话区分：

> BasicBlock 更简单，Bottleneck 更省算力、也更适合超深层网络。

### 7. ResNet 相比 VGG 到底改了什么？

VGG 的主线是：

- 规整堆深
- 小卷积核重复叠加

ResNet 相比它最关键的升级不是某个卷积参数，而是：

- 引入 identity shortcut
- 把“深层可训练性”从经验问题，变成结构设计问题

所以如果面试官问“ResNet 真正解决了什么”，更好的答案是：

> ResNet 不是简单加了 skip connection，而是用残差学习显著改善了深层网络的优化路径，让更深 CNN 变得可训。

## 面试高频问题

### 1. ResNet 为什么有效？

因为残差形式 `F(x) + x` 让网络更容易逼近恒等映射，也让梯度能沿 shortcut 更直接地传播。

### 2. ResNet 解决的是梯度消失吗？

不只如此。更关键的是深层 plain network 的退化和优化困难。

### 3. 为什么 shortcut 有时要做 downsample？

因为主分支和 shortcut 分支相加前，shape 必须一致；下采样或通道变化时需要先投影对齐。

### 4. BasicBlock 和 Bottleneck 怎么区分？

BasicBlock 是两个 `3x3`；Bottleneck 是 `1x1 -> 3x3 -> 1x1`，多用于更深版本。

### 5. ResNet 和 VGG 的核心差异是什么？

VGG 主要靠规整堆深，ResNet 主要靠残差连接解决深层优化问题。

### 6. shortcut 一定没有参数吗？

不一定。shape 一致时可以直接 identity；shape 不一致时通常要用 `1x1 conv` 做投影。

## 最小实现

这个专题的最小实现聚焦三块：

- `conv3x3 / conv1x1`
- `BasicBlock`
- `TinyResNet`

### 1. `conv3x3` 和 `conv1x1`

```python
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )
```

这里最值得记的是：

- `3x3` 负责主特征提取
- `1x1` 常用于 shortcut 投影或 Bottleneck 调整通道

### 2. `BasicBlock.forward`

```python
def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out = out + identity
    out = self.relu(out)
    return out
```

你可以把它直接记成：

```text
conv-bn-relu -> conv-bn -> add shortcut -> relu
```

### 3. `TinyResNet` 的整体主线

```python
x = self.stem(x)
x = self.layer1(x)
x = self.layer2(x)
x = self.layer3(x)
x = self.avgpool(x)
x = torch.flatten(x, 1)
logits = self.fc(x)
```

这条主线体现的是：

- stem 先快速提浅层特征
- 每个 stage 堆多个残差块
- stage 之间逐步降采样、升通道
- 最后做全局池化和分类

完整代码见：[minimal.py](minimal.py)

## 工程关注点

### 1. ImageNet 风格和 CIFAR 风格的 stem 不一样

ImageNet 版常见：

- `7x7 conv + maxpool`

CIFAR 版常见：

- 更轻的 `3x3 conv` stem

不要把所有 ResNet 的 stem 都背成同一个模板。

### 2. BatchNorm 在训练和推理时行为不同

这是 CNN 工程里非常基础但常被忽略的问题。

### 3. stage 切换时最容易出 shape bug

尤其是：

- stride 改了
- 通道改了
- shortcut 忘记 downsample

## 常见坑点

### 1. 把 ResNet 理解成“单纯更深的 VGG”

不对。它最核心的是残差连接，不只是层数更多。

### 2. 认为 shortcut 一定是原样直连

shape 不一致时必须先投影。

### 3. 把梯度消失和退化问题完全等同

两者相关，但不完全是一回事。面试里最好把“优化困难”和“更深反而更差”明确说出来。

### 4. 忘了相加之后通常还要再接一次 ReLU

很多人写 block 时容易漏掉最后这个非线性。

## 面试时怎么讲

如果面试官让你介绍 ResNet，可以按这个顺序讲：

1. 在 VGG 之后，大家发现更深的 plain CNN 越来越难训，甚至更深反而更差
2. ResNet 的做法是让 block 学残差 `F(x)`，最后输出写成 `F(x) + x`
3. 这样如果最优映射接近恒等映射，网络只需要把残差学成接近 0，会更容易优化
4. shortcut 还给梯度提供了更直接的传播路径
5. shape 一致时 shortcut 可直接 identity，不一致时用 `1x1 conv` 做 downsample / projection
6. 浅层版本常用 BasicBlock，更深版本常用 Bottleneck

一个简洁版本可以直接讲：

> ResNet 解决的是深层 plain CNN 的优化和退化问题。它让每个 block 不再直接学 `H(x)`，而是学残差 `F(x)`，最后输出写成 `F(x) + x`。这样网络更容易逼近恒等映射，梯度也能沿 shortcut 更直接传播，所以更深的 CNN 才真正变得可训练。shape 不一致时，shortcut 会用 `1x1 conv` 做投影对齐。

## 延伸阅读

- 上一篇基础模块：[CNN 基础模块](../cnn_basics/README.md)
- 视觉 Transformer：[Vision Transformer](../vision_transformer/README.md)
- 配套代码：[minimal.py](minimal.py)
