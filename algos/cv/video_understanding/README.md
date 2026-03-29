# 视频表示与帧采样 面试攻略

## 专题顺序

- 第一篇：[视频表示与帧采样](README.md)
- 第二篇：2D CNN + Temporal Head / 3D CNN（待补）
- 第三篇：Video Transformer / TimeSformer（待补）

## 这是什么？

这是视频理解部分的第一篇，先把最基础的问题讲清楚：

- 视频和图像的根本差别是什么
- 为什么视频任务通常离不开帧采样
- clip-level 和 video-level 到底有什么区别
- 面试里该怎么讲时间维度带来的额外难点

如果只用一句话概括：

> 视频不是“很多张图简单堆起来”这么简单，它多了一条时间维度，所以模型不仅要看每一帧长什么样，还要处理帧与帧之间的变化和顺序。

## 核心机制

### 1. 视频输入和图像输入有什么本质差别？

图像通常是：

$$ X \in \mathbb{R}^{B \times C \times H \times W} $$

视频通常是：

$$ X \in \mathbb{R}^{B \times T \times C \times H \times W} $$

这里多出来的 `T` 就是时间维。

这意味着模型要额外处理：

- 动作的先后顺序
- 目标随时间的变化
- 长视频里的冗余帧和关键帧

### 2. 为什么视频任务常常先做帧采样？

因为直接把所有帧都送进模型，成本很高。

现实里常见的问题是：

- 视频很长
- 相邻帧高度相似
- 真正有信息的片段可能只占一小段

所以采样的意义就是：

> 用尽量少的帧，保留尽量多的时序信息。

### 3. 常见帧采样策略有哪些？

#### 1. Uniform sampling

在整个视频里均匀取若干帧。

优点：

- 简单
- 覆盖全局

缺点：

- 容易错过短暂动作峰值

#### 2. Stride sampling

按固定步长取帧。

优点：

- 实现简单
- 适合固定 fps 的输入

缺点：

- 对关键片段不敏感

#### 3. Keyframe / saliency sampling

根据运动、语义或打分挑关键帧。

优点：

- 更聚焦有效信息

缺点：

- 需要额外打分逻辑

### 4. clip-level 和 video-level 有什么区别？

这是视频理解里很高频的一个区分。

#### 1. clip-level

把一段短片段当成输入单位。

适合：

- 动作识别
- 局部事件识别

#### 2. video-level

对整段视频做统一判断。

适合：

- 长视频摘要
- 视频问答
- 全局标签分类

所以很多系统实际做法是：

- 先把视频切成多个 clip
- 每个 clip 编码
- 再做聚合

### 5. 最小代码里每一步在做什么？

完整代码见 [minimal.py](minimal.py)。

```python
def uniform_sample(video, num_frames):
    total_frames = video.shape[0]
    indices = torch.linspace(0, total_frames - 1, steps=num_frames).long()
    return video[indices]
```

这里直接体现了视频理解里最常见的第一步：

- 先别急着喂全视频
- 先决定取哪些帧

再看最小模型：

```python
class TinyVideoClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.frame_encoder = FrameEncoder()
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(64, num_classes)

    def forward(self, video):
        frame_features = self.frame_encoder(video)
        frame_features = frame_features.transpose(1, 2)
        video_feature = self.temporal_pool(frame_features).squeeze(-1)
        return self.head(video_feature)
```

这里：

- `frame_encoder`：先逐帧提视觉特征
- `transpose(1, 2)`：把张量改成便于时间聚合的形状
- `temporal_pool`：把多个帧特征聚成一个视频特征
- `head`：输出视频级预测

这个实现很简单，但已经能讲出视频模型的主线：

> 先采样，再逐帧编码，再做时间聚合。

## 面试高频问题

### 1. 为什么视频理解通常离不开帧采样？

因为视频太长、冗余帧太多，直接全量处理成本很高。

### 2. 视频和图像任务相比，多出来的核心难点是什么？

多了时间维，需要处理顺序、动作变化、长短时依赖和冗余帧。

### 3. clip-level 和 video-level 为什么要区分？

因为短片段任务关注局部时序，整视频任务关注全局聚合，建模方式和评测粒度都不同。

### 4. 均匀采样一定够吗？

不一定。它覆盖全局，但可能错过非常短的关键动作。

## 最小实现

完整代码见 [minimal.py](minimal.py)。

这个实现重点保留了三件事：

- `uniform_sample` / `stride_sample`：最基本的视频采样
- `FrameEncoder`：最小逐帧视觉特征提取
- `TinyVideoClassifier`：最小时间聚合分类器

## 工程关注点

- 帧采样预算和效果怎么平衡
- 不同 fps 视频如何统一处理
- 长视频是否需要分段和层级聚合
- ASR、OCR、音频信息是否要一起接入
- clip 级预测如何聚成 video 级结论

## 常见坑点

- 把视频任务理解成“逐帧图像分类”
- 只会说 3D CNN 或 Transformer，不会先讲帧采样
- 混淆 clip-level 标注和 video-level 标注
- 忽视长视频里的冗余和关键帧稀疏问题

## 面试时怎么讲

一个比较稳的讲法是：

> 视频理解比图像理解多了一条时间维，所以第一步往往不是直接上复杂模型，而是先决定怎么采样。采样后可以逐帧提视觉特征，再通过 temporal pooling、RNN、3D CNN 或 Video Transformer 去建模时间关系。clip-level 和 video-level 任务的核心区别，在于最后是看局部时序，还是做整视频聚合。
