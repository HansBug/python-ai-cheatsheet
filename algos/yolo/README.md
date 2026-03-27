# YOLO 面试攻略

## 这是什么？

这是目标检测里最经典、也最容易被问到的一条路线。

如果只用一句话来讲：

> YOLO 的核心思想，是把目标检测看成一次前向就同时完成“目标在哪里、是什么类别、置信度多高”的密集预测问题，然后再通过后处理筛掉重复框。

它的重要性在于：

- 它代表了 one-stage detector 的主流思路
- 它把检测从“候选框 + 分类”的两阶段范式，推进到了端到端密集预测
- 自动驾驶、工业视觉、边缘部署里，YOLO 家族长期是高频工程基线

如果面试官问“YOLO 到底是什么”，你不要只回答“一个检测模型”，更好的说法是：

> YOLO 不是单一版本，而是一条检测路线：单阶段、多尺度、密集预测、框回归 + 分类 + objectness，再配合 NMS 做去重。

## 核心机制

### 1. YOLO 到底想解决什么问题？

目标检测本质上要同时回答三件事：

- 图里有没有目标
- 目标在哪
- 目标是什么类别

早期两阶段方法，比如 R-CNN / Fast R-CNN / Faster R-CNN，通常是：

```text
先找候选区域 -> 再分类和回归
```

YOLO 的思路更直接：

```text
整张图一次前向 -> 在多个位置直接预测框、置信度和类别
```

所以它的优势很明显：

- 结构更统一
- 推理更快
- 更适合实时检测

但代价也很典型：

- 对小目标、密集目标、极端遮挡场景更敏感
- 正负样本分配和后处理设计会变得更关键

### 2. YOLOv1 的原始思想是什么？

YOLOv1 的经典说法是：

> 把输入图像切成网格，每个网格负责预测落在自己内部的目标。

如果图像被划成 `S x S` 网格，那么每个网格会输出：

- 若干个边框
- 每个边框的置信度
- 类别分布

这一步的历史意义很大，因为它第一次非常明确地把检测写成了：

> 一个 dense prediction 问题，而不是先提 proposal 再分类。

不过你面试时最好也顺手说一句：

> 现代 YOLO 家族已经不完全等同于 YOLOv1 那种“一个网格负责一个目标中心”的最原始形式了，后面引入了多尺度、anchor、anchor-free 等很多改进。

### 3. 现代 YOLO 一般长什么样？

现在更常见的 YOLO 主线通常可以概括成：

```text
Backbone -> Neck -> Detection Head -> Decode -> Score Threshold -> NMS
```

这条线一定要会讲。

#### 1. Backbone

作用是提图像特征。

常见 backbone 会把输入图像逐步下采样，得到不同尺度的 feature map。

比如：

- 浅层：分辨率高，适合小目标
- 深层：语义强，适合大目标

#### 2. Neck

作用是做多尺度特征融合。

常见结构有：

- FPN
- PAN

直觉上它在做的是：

> 让高语义的深层信息和高分辨率的浅层信息结合起来。

这对检测尤其重要，因为目标尺度差异很大。

#### 3. Detection Head

作用是在每个位置上，直接预测：

- 边框回归量
- objectness
- 分类分数

这一步就是 YOLO 的“dense prediction”真正落地的地方。

### 4. YOLO 的 head 到底输出什么？

这通常是面试里的关键追问。

最简化地讲，一个位置上的预测可以记成：

```text
[box, objectness, class_scores]
```

更细一点就是：

- `box`：边框参数，比如中心点、宽高，或者左右上下偏移
- `objectness`：这里到底像不像存在一个目标
- `class_scores`：如果有目标，更像哪一类

在这个仓库的最小实现 [minimal.py](minimal.py) 里，我用一个 toy head 写成了：

```python
class TinyYOLOHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.box_conv = nn.Conv2d(hidden_channels, 4, kernel_size=1)
        self.obj_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.cls_conv = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, feat):
        feat = F.silu(self.stem(feat))
        box_reg = self.box_conv(feat)
        objectness = self.obj_conv(feat)
        class_logits = self.cls_conv(feat)

        pred = torch.cat([box_reg, objectness, class_logits], dim=1)
        return pred.permute(0, 2, 3, 1).contiguous()
```

这段代码每一块都要能讲清：

- `stem`：先把共享特征再加工一下
- `box_conv`：专门负责回归框
- `obj_conv`：专门负责预测 objectness
- `cls_conv`：专门负责分类

最后返回的形状是：

```text
[B, H, W, 5 + C]
```

其中：

- `4` 个框参数
- `1` 个 objectness
- `C` 个类别分数

你可以把它记成一句话：

> YOLO head 本质上就是在 feature map 的每个位置，吐出一小组检测结果。

### 5. objectness 到底在表达什么？

这是很多人说不清的地方。

`objectness` 不是类别概率，它回答的是：

> 这个位置预测出来的框，到底像不像真的覆盖了一个目标。

比如一个位置可能预测：

- `objectness = 0.95`
- `car prob = 0.80`
- `person prob = 0.10`

这时常见最终分数会写成：

$$ \mathrm{score} = \mathrm{objectness} \times \mathrm{class\_prob} $$

如果取 `car` 这一类，就是：

```text
0.95 * 0.80 = 0.76
```

这条分数在工程里很重要，因为它把：

- “这里有没有目标”
- “如果有，它更像什么类”

结合到了一起。

### 6. 边框是怎么从 head 输出变成真实框的？

模型 head 直接吐出来的通常不是最终像素坐标，而是某种相对量。

在最小实现里，我用的是一版很常见的简化写法：

```python
centers = (torch.sigmoid(box_offsets) + grid) * stride
wh = torch.exp(box_scales) * stride
boxes = cxcywh_to_xyxy(torch.cat([centers, wh], dim=-1))
```

这三行分别在做：

- `grid`：告诉当前预测来自哪个网格位置
- `sigmoid(box_offsets)`：把中心偏移压到一个稳定范围
- `exp(box_scales)`：把宽高映射成正数
- `* stride`：把 feature map 尺度映射回输入图像尺度

举个直观例子：

- feature map 上某个 cell 的坐标是 `(1, 2)`
- stride 是 `32`
- 中心偏移 sigmoid 后得到 `(0.6, 0.4)`

那这个框中心大致就是：

```text
((2 + 0.6) * 32, (1 + 0.4) * 32) = (83.2, 44.8)
```

这就是从“网格上的预测”变成“图像里的框”的过程。

### 7. 为什么 YOLO 会预测出很多重复框？

因为它是 dense prediction。

整张图上的很多位置、很多尺度，都会一起出预测。结果就是：

- 同一个车，左边那个 cell 可能预测一次
- 右边那个 cell 也可能预测一次
- 不同尺度层可能还各预测一次

所以你会得到一堆：

- 类别一样
- 分数都不低
- 位置高度重叠

的框。

这不是 bug，而是 one-stage detector 的自然产物。

这也正是 NMS 必须存在的原因。

### 8. NMS 是什么？为什么一定要有？

NMS 全称是 Non-Maximum Suppression，非极大值抑制。

它解决的问题非常简单：

> 同一个目标被预测出了多个高度重叠的框时，只保留最可信的那个，其余的压掉。

最经典的 NMS 流程是：

1. 先按分数从高到低排序
2. 取当前分数最高的框，放进最终结果
3. 计算它和剩余框的 IoU
4. 把 IoU 过高的框删除
5. 重复直到没有框剩下

这里最关键的量是 IoU：

$$ \mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} $$

如果两个框高度重叠，IoU 就高；如果只是靠得近但没怎么重叠，IoU 就低。

### 9. NMS 过程最好怎么讲？

用一个车的例子最直观。

假设模型对同一辆车给了 3 个框：

```text
box1: score = 0.92
box2: score = 0.88
box3: score = 0.41
```

并且：

- `box1` 和 `box2` 的 IoU 是 `0.76`
- `box1` 和 `box3` 的 IoU 是 `0.18`

如果 NMS 阈值是 `0.5`，那流程就是：

- 先保留 `box1`
- 因为 `box2` 和 `box1` 的 IoU 太高，所以删掉 `box2`
- `box3` 和 `box1` 重叠不高，保留

最后结果是：

- 保留 `box1`
- 保留 `box3`

这就是 NMS 的核心逻辑：高分框优先，同目标的重复框被抑制。

### 10. NMS 的代码到底长什么样？

在 [minimal.py](minimal.py) 里，最小版 NMS 是：

```python
def nms(boxes, scores, iou_threshold):
    order = torch.argsort(scores, descending=True)
    keep = []

    while order.numel() > 0:
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break

        rest = order[1:]
        ious = box_iou(boxes[current : current + 1], boxes[rest]).squeeze(0)
        order = rest[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)
```

这段代码最值得你会讲的点是：

- `argsort(...)`：先让高分框排前面
- `current = order[0]`：每一轮先拿当前最强框
- `box_iou(...)`：算它和剩余框的重叠程度
- `order = rest[ious <= threshold]`：把和它太像的重复框删掉

这就是最经典的 greedy NMS。

### 11. 为什么 NMS 常常是按类别分别做？

因为两类不同目标可以位置重叠，但不一定应该互相压掉。

比如：

- 一个人骑车
- 人框和自行车框本来就会大幅重叠

如果不分类别统一做 NMS，很可能把其中一个误杀。

所以常见实践是：

> 先按类别分组，再对每一类内部做 NMS。

在最小实现里，对应的是：

```python
def batched_nms(boxes, scores, labels, iou_threshold):
    for class_id in labels.unique(sorted=True):
        class_indices = torch.nonzero(labels == class_id, as_tuple=False).squeeze(1)
        class_keep = nms(boxes[class_indices], scores[class_indices], iou_threshold)
```

这段就很清楚：

- 同类框互相竞争
- 异类框彼此不直接抑制

### 12. YOLO 家族后面主要改进了什么？

面试里不一定要求你背版本号，但最好知道主线。

#### 1. 从单尺度到多尺度

后来的 YOLO 大多不只在一个 feature map 上预测，而是多个尺度一起做。

#### 2. 从简单 head 到更强的 detection head

包括：

- 更好的特征融合
- decoupled head
- 更合理的 box 表达和损失

#### 3. 从 anchor-based 到 anchor-free

这条线很重要。

早期很多 YOLO 版本会配 anchor：

- 每个位置不是直接预测任意框
- 而是围绕几组先验框微调

后来的很多实现开始转向 anchor-free，减少先验设计负担。

#### 4. 更强的训练和标签分配策略

包括：

- IoU 系列损失
- 更合理的正负样本匹配
- 数据增强，如 Mosaic

所以如果面试官问“现代 YOLO 和 YOLOv1 一样吗”，你要敢说：

> 不一样。核心精神一样，都是 one-stage dense prediction，但现代 YOLO 家族在多尺度、head、label assignment、box loss 和后处理上已经演化很多了。

## 面试高频问题

### 1. YOLO 为什么快？

因为它把检测写成一次前向的 dense prediction，不需要像两阶段方法那样先 proposal 再逐个分类。

### 2. YOLO 是 one-stage 还是 two-stage？

标准回答是 one-stage detector。

### 3. objectness 和 class score 的区别是什么？

- `objectness`：像不像有目标
- `class score`：如果有目标，更像哪一类

### 4. 为什么需要 NMS？

因为同一个目标通常会被多个位置、多个尺度重复预测出来。

### 5. NMS 的阈值太大或太小会怎样？

- 太小：容易把本来不同的近邻目标误压掉
- 太大：重复框保留太多

### 6. YOLO 的弱点通常是什么？

- 小目标检测
- 密集遮挡场景
- 极端长尾类别或难分类边界

## 最小实现

完整代码见：[minimal.py](minimal.py)。

这份最小实现故意保留两段最值得面试讲的代码：

- `TinyYOLOHead`
- `decode_predictions + batched_nms`

### 1. 一个最小 YOLO head 长什么样？

看这个最小 head：

```python
class TinyYOLOHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.box_conv = nn.Conv2d(hidden_channels, 4, kernel_size=1)
        self.obj_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.cls_conv = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, feat):
        feat = F.silu(self.stem(feat))
        box_reg = self.box_conv(feat)
        objectness = self.obj_conv(feat)
        class_logits = self.cls_conv(feat)

        pred = torch.cat([box_reg, objectness, class_logits], dim=1)
        return pred.permute(0, 2, 3, 1).contiguous()
```

这段代码要这样讲：

- 输入是特征图 `feat`
- 输出是在每个空间位置上，吐出 `4 + 1 + C` 个值
- 这就是 one-stage detector 最核心的数据形式

### 2. decode 在做什么？

看 `decode_predictions(...)`：

```python
centers = (torch.sigmoid(box_offsets) + grid) * stride
wh = torch.exp(box_scales) * stride
boxes = cxcywh_to_xyxy(torch.cat([centers, wh], dim=-1))

best_class_probs, labels = class_probs.max(dim=-1)
scores = objectness * best_class_probs
```

这几行分别在做：

- 把相对偏移解码成真实中心点
- 把宽高解码成正值
- 把中心点表示转成左上右下框
- 取最终类别分数
- 把 `objectness` 和分类分数合成最终 `score`

这几行其实就是检测后处理的主线。

### 3. 最小 NMS 流程怎么落到代码里？

再看：

```python
boxes, scores, labels, objectness = decode_predictions(...)
keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
```

这表示：

- 先解码得到候选框
- 再按类别做 NMS
- 最终只保留一小部分真正输出的框

脚本里我专门手工造了一个例子：

- 两个 `car` 框高度重叠，代表同一辆车的重复预测
- 一个 `person` 框在别处
- 另一个 `car` 框离得较远

所以跑完后你应该能看到：

- 重叠的 `car` 框只留一个
- `person` 框保留
- 远处那个 `car` 框也保留

这正是 NMS 的直观效果。

## 工程关注点

### 1. 阈值联动很重要

实际部署里通常要一起调：

- score threshold
- NMS IoU threshold
- 最大保留框数

### 2. 解码细节要和训练定义严格一致

如果训练时框参数定义和推理时 decode 不一致，结果会直接崩。

### 3. 多尺度预测会放大后处理复杂度

尺度越多，候选框越多，NMS 开销和误检控制都会更重要。

### 4. 真正线上瓶颈不只在 backbone

后处理、数据搬运、batch 策略，也可能是延迟来源。

## 常见坑点

### 1. 把 objectness 直接当类别概率

这是两个不同概念。

### 2. 只会背 YOLO 很快，不会说为什么快

必须落到 one-stage dense prediction 这个结构层面。

### 3. 不知道为什么会有重复框

这会导致你后面 NMS 完全讲不下去。

### 4. 以为 NMS 是训练阶段主算法

NMS 主要是推理后处理，不是训练主干。

### 5. 不区分类内 NMS 和跨类冲突

这在复杂场景里会直接影响结果。

## 面试时怎么讲

可以按这条线讲：

1. YOLO 是 one-stage detector，把检测写成一次前向上的密集预测。
2. 现代 YOLO 通常由 backbone、neck、detection head 组成。
3. head 在每个位置预测框、objectness 和类别分数。
4. 候选框解码后，常把 `objectness * class_prob` 作为最终分数。
5. 因为同一个目标会被重复预测，所以推理阶段必须做 NMS 去重。
6. NMS 的逻辑就是“高分优先，IoU 过高的重复框压掉”，通常按类别分别执行。

## 延伸阅读

- YOLOv1: You Only Look Once
- YOLOv3 / YOLOv5 / YOLOX / YOLOv8 相关技术报告与源码
- Soft-NMS
- IoU / GIoU / DIoU / CIoU 系列框损失
